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



### Training Pipeline
The training workflow is:
1. Prepare paired English/French text data using `prepare_dataset.py`.
2. Train a custom French tokenizer with `tokenizer.py`.
3. Load tokenized dataset and build batch collator in `data.py`.
4. Train the `Transformer` model in `train.py` with `Accelerator` for distributed/mixed-precision training.


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
### References:
- Youtube: [Priyam Mazumdar](https://youtube.com/playlist?list=PL16vydMdqFg9g9xevVEl-MVVaK9jcqyC5&si=KbI4d--B5hd8NoVT)
- Github: [PyTorch-Adventures](https://github.com/priyammaz/PyTorch-Adventures/tree/main/PyTorch%20for%20NLP/Seq2Seq%20for%20Neural%20Machine%20Translation)
