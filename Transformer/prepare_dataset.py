import os
import argparse
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm

from tokenizer import FrenchTokenizer
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"

def build_english2French_dataset_in_memory(path_to_data_root, test_prop=0.005, cache_dir=None):
    hf_dataset = []
    
    all_dirs = [d for d in os.listdir(path_to_data_root) 
                if os.path.isdir(os.path.join(path_to_data_root, d)) and not d.startswith('.')]
    
    print("\n--- Phase 1: Loading Raw Text Files (In-Memory) ---")
    for dir in tqdm(all_dirs, desc="Aggregating text datasets"):
        path_to_dir = os.path.join(path_to_data_root, dir)
        
        french_text = english_text = None
        for txt in os.listdir(path_to_dir):
            if txt.endswith(".fr"):
                french_text = os.path.join(path_to_dir, txt)
            elif txt.endswith(".en"):
                english_text = os.path.join(path_to_dir, txt)

        if french_text is not None and english_text is not None:
            # Load text files directly into streaming Hugging Face tables
            french_dataset = load_dataset("text", data_files=french_text, cache_dir=cache_dir)["train"]
            english_dataset = load_dataset("text", data_files=english_text, cache_dir=cache_dir)["train"]
        
            english_dataset = english_dataset.rename_column("text", "english_src")
            english_dataset = english_dataset.add_column("french_tgt", french_dataset["text"])

            hf_dataset.append(english_dataset)
    
    if not hf_dataset:
        raise ValueError(f"No valid .en and .fr pairs found in {path_to_data_root}. Check file paths/unzip status!")

    print("\nConcatenating individual datasets together in RAM...")
    combined_dataset = concatenate_datasets(hf_dataset)
    
    print("Creating Train/Test Splits...")
    split_dataset = combined_dataset.train_test_split(test_size=test_prop)
    return split_dataset
    
def tokenize_englsih2french_dataset_directly(raw_dataset, path_to_save, num_workers=24, truncate=False, max_length=512, min_length=5):
    print("\n--- Phase 2: Tokenization Pipeline ---")
    french_tokenizer = FrenchTokenizer("trained_tokenizer/french_wp.json", truncate=truncate, max_length=max_length)
    english_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

    def _tokenize_text(examples):
        english_text = examples["english_src"]
        french_text = examples["french_tgt"]
        
        src_ids = english_tokenizer(english_text, truncation=True, max_length=max_length)["input_ids"]
        tgt_ids = french_tokenizer.encode(french_text)
        return {
            "src_ids": src_ids,
            "tgt_ids": tgt_ids
        }
    
    print(f"Mapping tokenizers across dataset using {num_workers} processes...")
    tokenized_dataset = raw_dataset.map(
        _tokenize_text, 
        batched=True, 
        num_proc=num_workers,
        desc="Tokenizing Src & Tgt languages"
    )
    
    print("Removing old raw string columns...")
    tokenized_dataset = tokenized_dataset.remove_columns(["english_src", "french_tgt"])

    print(f"Filtering out sequences shorter than {min_length} tokens...")
    filter_func = lambda batch: [len(e) > min_length for e in batch["tgt_ids"]]
    tokenized_dataset = tokenized_dataset.filter(
        filter_func, 
        batched=True, 
        num_proc=num_workers,
        desc="Filtering short sequences"
    )
    
    print(f"Saving final, tokenized dataset directly to: {path_to_save}...")
    tokenized_dataset.save_to_disk(path_to_save) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translation Dataset Preparation")
    parser.add_argument("--test_split_pct", default=0.005, type=float)
    parser.add_argument("--max_length", default=512, type=int)
    parser.add_argument("--min_length", default=5, type=int)
    parser.add_argument("--path_to_data_root", required=True, type=str)
    parser.add_argument("--num_workers", default=24, type=int)
    parser.add_argument("--huggingface_cache_dir", default=None, type=str)

    args = parser.parse_args()
    path_to_data_root = args.path_to_data_root
    
    path_to_data_tokenized = os.path.join(path_to_data_root, "tokenized_french2english_corpus")
    cache_dir = args.huggingface_cache_dir

    raw_memory_dataset = build_english2French_dataset_in_memory(
        path_to_data_root=path_to_data_root,
        test_prop=args.test_split_pct,
        cache_dir=cache_dir
    )
    
    tokenize_englsih2french_dataset_directly(
        raw_dataset=raw_memory_dataset,
        path_to_save=path_to_data_tokenized,
        truncate=True,
        max_length=args.max_length,
        min_length=args.min_length,
        num_workers=args.num_workers
    )
    print("\n🎉 Success! The finalized tokenized dataset is ready for training.")