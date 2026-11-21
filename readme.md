# AI Architecture Foundry

This repository contains implementations of various large language model architectures, each in its own subdirectory. Each subdirectory includes a README describing the architecture and training details.

## Gemma
- Model Scale: 270M
- Vocabulary Size: 50,257
- Context Length: 32,768 tokens
- Embedding Dimension: 640
- Number of Heads: 4
- Number of Layers: 18
- Hidden Dimension: 2,048
- Head Dimension: 256
- QK‑Normalization: Enabled
- Interleaved Attention: 5 local sliding‑window (4,096 tokens) + 1 global
- Rotary Positional Embeddings: Local base 10k, Global base 1M
- Normalization: RMSNorm
- Training Dataset: TinyStories (Hugging Face)

## ViT
- Input Resolution: 28×28 (MNIST)
- Patch Size: 7×7
- Number of Patches: 16 + 1 [CLS] token
- Embedding Dimension: 16
- Number of Heads: 1
- Encoder Layers: 1
- MLP Dimension: 16
- Training Dataset: MNIST

## GPT
- Vocabulary Size: 50,257
- Block Size: 128 tokens
- Number of Layers: 6
- Number of Heads: 6
- Embedding Dimension: 384
- Dropout Rate: 0.1
- Bias: Enabled
- Activation: GELU
- Training Dataset: TinyStories

## GPT_OSS
- Vocabulary Size: 50,257 (GPT‑2 TikTokenizer)
- Context Length: 512 tokens
- Embedding Dimension: 128
- Number of Heads: 4
- Number of Layers: 6
- Hidden Dimension: 128
- Intermediate (MLP) Dimension: 512
- Head Dimension: 32
- Key/Value Heads: 2
- Sliding Window Attention: 64 tokens (alternates per layer)
- Mixture‑of‑Experts: 4 experts, 1 per token
- Activation: SwiGLU
- Normalization: RMSNorm
- Rotary Positional Embeddings: θ=150,000, scale=32
- Training Dataset: TinyStories

## Diffusion_LLM
- Dataset: TinyStories
- Vocabulary Size: 8,000
- Context Length: 256 tokens
- Embedding Dimension: 384
- Number of Heads: 6
- Number of Layers: 6
- Feed‑Forward Dimension: 1,536
- Activation: GELU
- Normalization: Pre‑LayerNorm
- Dropout: 0.1
- Diffusion Steps: 64
- Weight Tying: Input Embedding ↔ Output Projection

## Transformer
- Embedding Dimension: 512
- Encoder Depth: 6 layers
- Decoder Depth: 6 layers
- Attention Heads: 8
- Attention Dropout: 0.1
- Feed‑Forward MLP Ratio: 4×
- Hidden Dropout: 0.1
- Source Vocabulary Size: 30,522 (BERT)
- Target Vocabulary Size: 32,000 (French WordPiece)
- Max Input Length: 512 tokens
- Max Target Length: 512 tokens
- Positional Encoding: Sinusoidal
- Learnable Positional Embeddings: Disabled by default