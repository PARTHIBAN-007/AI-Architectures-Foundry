# Transformer Translation Architecture

Transformer-based English-to-French translation implementation. The model is built from scratch using PyTorch and includes modular embedding, positional encoding, encoder/decoder attention blocks, and training utilities for tokenization, dataset preparation, and accelerated training.

### Architecture Configuration
- **Model Type:** Encoder-Decoder Transformer
- **Embedding Dimension:** 512
- **Encoder Depth:** 6 layers
- **Decoder Depth:** 6 layers
- **Number of Attention Heads:** 8
- **Attention Dropout:** 0.1
- **Feed-Forward MLP Ratio:** 4x
- **Hidden Dropout:** 0.1
- **Source Vocabulary Size:** 30,522 (BERT tokenizer)
- **Target Vocabulary Size:** 32,000 (custom French WordPiece tokenizer)
- **Maximum Input Length:** 512 tokens
- **Maximum Target Length:** 512 tokens
- **Positional Encoding:** Sinusoidal
- **Learnable Positional Embeddings:** Disabled by default

### Model Overview
The implementation in `model.py` includes:
- `TransformerConfig`: configuration dataclass for embedding size, attention heads, dropout, vocabulary sizes, and sequence lengths.
- `PositionalEncoding`: sinusoidal positional embeddings added to token embeddings.
- `Embeddings`: separate source and target embedding layers with positional encodings.
- `Attention`: multi-head scaled dot-product attention supporting encoder self-attention, decoder self-attention, and decoder cross-attention.
- `FeedForward`: standard transformer MLP block with GELU activation and dropout.
- `TransformerEncodeLayer` and `TransformerDecoderlayer`: encoder and decoder building blocks with residual connections and layer normalization.
- `Transformer`: full encoder-decoder stack plus linear output head and greedy inference.

### Folder Contents
- `model.py`: core Transformer model definition and inference pipeline.
- `tokenizer.py`: trains and loads a custom French WordPiece tokenizer, plus `FrenchTokenizer` wrapper for encode/decode.
- `prepare_dataset.py`: builds and tokenizes an English-French parallel dataset, then saves it to disk.
- `data.py`: collates tokenized examples into padded source and target batches for training.
- `train.py`: training loop using `accelerate`, `datasets`, and Hugging Face tokenizers, with evaluation, checkpointing, and WandB logging.
- `training.ipynb`: notebook for exploratory experiments or model inspection.

### Training Pipeline
The training workflow is:
1. Prepare paired English/French text data using `prepare_dataset.py`.
2. Train a custom French tokenizer with `tokenizer.py`.
3. Load tokenized dataset and build batch collator in `data.py`.
4. Train the `Transformer` model in `train.py` with `Accelerator` for distributed/mixed-precision training.

Key training settings in `train.py`:
- `batch_size = 128`
- `gradient_accumulation_steps = 2`
- `learning_rate = 1e-4`
- `training_steps = 15000`
- `warmup_steps = 2000`
- `scheduler_type = "cosine"`
- `evaluation_steps = 2500`
- `weight_decay = 0.001`
- `adam_eps = 1e-6`

### Inference
The model offers a greedy inference method in `Transformer.inference(...)`:
- starts generation from a beginning-of-sequence token
- appends predicted tokens until the end-of-sequence token is produced or `max_len` is reached
- returns the generated token sequence

### Usage
1. Train the tokenizer:
```bash
python tokenizer.py --path_to_data_root <data_root>
```
2. Prepare the parallel dataset:
```bash
python prepare_dataset.py --path_to_data_root <data_root>
```
3. Train the model:
```bash
python train.py
```

### Notes
- The source tokenizer uses `google-bert/bert-base-uncased` for English input encoding.
- The target tokenizer is a custom French WordPiece tokenizer saved under `trained_tokenizer/`.
- `prepare_dataset.py` expects `.en` and `.fr` files paired inside dataset directories.

### Reference
- `model.py` for architecture details
- `tokenizer.py` for target vocabulary preparation
- `train.py` for training and evaluation flow
