import os
import numpy as np
import torch
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datasets import load_from_disk
from tqdm import tqdm

from model import Transformer, TransformerConfig
from data import TranslationCollator
from tokenizer import FrenchTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

encoder_depth = 6
decoder_depth = 6
mlp_ratio = 4
attention_dropout_p = 0.1
hidden_dropout_p = 0.1
embedding_dimension = 512
num_attention_heads = 8
max_src_len = 512
max_tgt_len = 512
learn_pos_embed = False

path_to_data = "english2french/tokenized_french2english_corpus/"
batch_size = 128
gradient_accumulation_steps = 2
num_workers = 16

learning_rate = 1e-4
training_steps = 15000
warmup_steps = 2000
scheduler_type = "cosine"
evaluation_steps = 2500
bias_norm_weight_decay = False
weight_decay = 0.001
betas = (0.9, 0.98)
adam_eps = 1e-6

working_directory = "work_dir"
experiment_name = "Transformer"
logging_interval = 1
resume_from_checkpoint = None

path_to_experiment = os.path.join(working_directory, experiment_name)
accelerator = Accelerator(project_dir=path_to_experiment, log_with="wandb")
accelerator.init_trackers(experiment_name)

tgt_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json")
src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config = TransformerConfig(
    embedding_dimension=embedding_dimension,
    num_attention_heads=num_attention_heads,
    attention_dropout_p=attention_dropout_p,
    mlp_ratio=mlp_ratio,
    decoder_depth=decoder_depth,
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    max_src_len=max_src_len,
    max_tgt_len=max_tgt_len,
    learn_pos_embed=learn_pos_embed
)

dataset = load_from_disk(path_to_data)
accelerator.print(dataset)

collate_fn = TranslationCollator(src_tokenizer, tgt_tokenizer)
minbatch_size = batch_size // gradient_accumulation_steps

train_loader = DataLoader(
    dataset["train"],
    batch_size=minbatch_size,
    num_workers=num_workers,
    collate_fn=collate_fn,
    shuffle=True
)

test_loader = DataLoader(
    dataset["test"],
    batch_size=minbatch_size,
    collate_fn=collate_fn,
    shuffle=False
)

model = Transformer(config)
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
accelerator.print(f"Number of parameters: {params}")

if not bias_norm_weight_decay:
    accelerator.print("Disabling Weight Decay on some parameters")
    weight_decay_params = []
    no_weight_decay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or "layernorm" in name:
                no_weight_decay_params.append(param)
            else:
                weight_decay_params.append(param)
    
    optimizer_group = [
        {"params": weight_decay_params, "weight_decay": weight_decay},
        {"params": no_weight_decay_params, "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_group, lr=learning_rate, betas=betas, eps=adam_eps)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=adam_eps)

scheduler = get_scheduler(
    name=scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=training_steps
)

loss_fn = torch.nn.CrossEntropyLoss()

src_ids_probe = torch.tensor(src_tokenizer("I am learning about Large Language Model")["input_ids"]).unsqueeze(0)

model, optimizer, train_loader, test_loader = accelerator.prepare(
    model, optimizer, train_loader, test_loader
)
accelerator.register_for_checkpointing(scheduler)

if resume_from_checkpoint is not None:
    path_to_checkpoint = os.path.join(path_to_experiment, resume_from_checkpoint)
    with accelerator.main_process_first():
        accelerator.load_state(path_to_checkpoint)
    completed_steps = int(resume_from_checkpoint.split("_")[-1])
    accelerator.print(f"Resuming from Iteration: {completed_steps}")
else:
    completed_steps = 0

train = True
progress_bar = tqdm(range(completed_steps, training_steps), disable=not accelerator.is_local_main_process)
while train:
    accumulation_steps = 0
    accumulate_loss = 0.0
    accuracy = 0.0
    
    for batch in train_loader:
        src_input_ids = batch["src_input_ids"]
        src_pad_mask = batch["src_pad_mask"]
        tgt_input_ids = batch["tgt_input_ids"]
        tgt_pad_mask = batch["tgt_pad_mask"]
        tgt_outputs = batch["tgt_outputs"]

        output = model(src_input_ids, tgt_input_ids, src_pad_mask, tgt_pad_mask)
        output = output.flatten(0, 1)
        tgt_outputs = tgt_outputs.flatten()

        loss = loss_fn(output, tgt_outputs)
        loss = loss / gradient_accumulation_steps
        
        accumulate_loss += loss.detach()

        accelerator.backward(loss)

        with torch.no_grad():
            output_preds = output.argmax(axis=-1)
            mask = (tgt_outputs != -100)
            output_preds = output_preds[mask]
            target_masked = tgt_outputs[mask]
            if len(target_masked) > 0:
                acc = (output_preds == target_masked).sum() / len(target_masked)
                accuracy += acc / gradient_accumulation_steps

        accumulation_steps += 1

        if accumulation_steps % gradient_accumulation_steps == 0:
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if completed_steps % logging_interval == 0:
                accuracy = torch.tensor(accuracy, device=accelerator.device) if isinstance(accuracy, float) else accuracy.detach()

                if accelerator.num_processes > 1:
                    accumulate_loss = torch.mean(accelerator.gather_for_metrics(accumulate_loss))
                    accuracy = torch.mean(accelerator.gather_for_metrics(accuracy))
                
                log = {
                    "train_loss": accumulate_loss.item(),
                    "training_acc": accuracy.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                }
                accelerator.log(log, step=completed_steps)
                
                logging_string = f"[{completed_steps}/{training_steps}] Training Loss: {accumulate_loss.item():.4f} | Training Acc: {accuracy.item():.4f}"
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
            # Evaluation Validation Block
            if completed_steps % evaluation_steps == 0 and completed_steps > 0:
                model.eval()
                accelerator.print("Evaluating Pipeline Conditions...")
                test_losses = []
                test_accs = []
                
                for eval_batch in tqdm(test_loader, disable=not accelerator.is_main_process):
                    eval_src_input_ids = eval_batch["src_input_ids"]
                    eval_src_pad_mask = eval_batch["src_pad_mask"]
                    eval_tgt_input_ids = eval_batch["tgt_input_ids"]
                    eval_tgt_pad_mask = eval_batch["tgt_pad_mask"]
                    eval_tgt_outputs = eval_batch["tgt_outputs"]

                    with torch.inference_mode():
                        eval_output = model(eval_src_input_ids, eval_tgt_input_ids, eval_src_pad_mask, eval_tgt_pad_mask)
                    
                    eval_output = eval_output.flatten(0, 1)
                    eval_tgt_outputs = eval_tgt_outputs.flatten()

                    eval_loss = loss_fn(eval_output, eval_tgt_outputs)
                    eval_preds = eval_output.argmax(axis=-1)
                    eval_mask = (eval_tgt_outputs != -100)
                    eval_preds = eval_preds[eval_mask]
                    eval_tgt_masked = eval_tgt_outputs[eval_mask]
                    
                    eval_acc = (eval_preds == eval_tgt_masked).sum() / len(eval_tgt_masked) if len(eval_tgt_masked) > 0 else 0

                    eval_loss = eval_loss.detach()
                    eval_acc = torch.tensor(eval_acc, device=accelerator.device) if isinstance(eval_acc, float) else eval_acc.detach()

                    if accelerator.num_processes > 1:
                        eval_loss = torch.mean(accelerator.gather_for_metrics(eval_loss))
                        eval_acc = torch.mean(accelerator.gather_for_metrics(eval_acc))
                    
                    test_losses.append(eval_loss.item())
                    test_accs.append(eval_acc.item())
                
                test_loss = np.mean(test_losses)
                test_acc = np.mean(test_accs)

                log = {"test_loss": test_loss, "test_acc": test_acc}
                accelerator.log(log, step=completed_steps)
                
                logging_string = f"--- Evaluation Metrics | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} ---"
                if accelerator.is_main_process:
                    progress_bar.write(logging_string)
                
                accelerator.save_state(os.path.join(path_to_experiment, f"checkpoint_{completed_steps}"))

                if accelerator.is_main_process:
                    src_ids_probe = src_ids_probe.to(accelerator.device)
                    unwrapped = accelerator.unwrap_model(model)
                    translated = unwrapped.inference(
                        src_ids_probe,
                        tgt_start_id=tgt_tokenizer.special_tokens_dict["[BOS]"],
                        tgt_end_id=tgt_tokenizer.special_tokens_dict["[EOS]"]
                    )
                    translated_str = tgt_tokenizer.decode(translated, skip_special_tokens=False)
                    progress_bar.write(f"Sample Translation output: {translated_str}")
                
                model.train()

            if completed_steps >= training_steps:
                train = False
                accelerator.save_state(os.path.join(path_to_experiment, "final_checkpoint"))
                break
                
            completed_steps += 1
            progress_bar.update(1)
            
            accumulate_loss = 0.0
            accuracy = 0.0