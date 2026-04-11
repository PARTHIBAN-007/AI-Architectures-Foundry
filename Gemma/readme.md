# Gemma 3 Architecture

Gemma 3 is an advanced open-source model family from Google, optimized for high performance and multimodal tasks. This implementation focuses on the 270M parameter scale for training from scratch.

### Architecture Configuration
- **Model Scale:** 270M
- **Vocabulary Size:** 50,257
- **Context Length:** 32,768 tokens
- **Embedding Dimension:** 640
- **Number of Heads:** 4
- **Number of Layers:** 18
- **Hidden Dimension:** 2,048
- **Head Dimension:** 256
- **QK-Normalization:** Enabled
- **Interleaved Attention:** 5 Local (Sliding Window: 4,096 tokens) followed by 1 Global attention layer
- **Rotary Positional Embeddings (RoPE):** Local (10k base) and Global (1M base) bases
- **Normalization:** RMSNorm

### Architecture Explanation
Gemma 3 utilizes a decoder-only transformer architecture with a hybrid interleaved attention mechanism. It alternates between "Sliding Window Attention" (local context of 4,096 tokens) and "Full Attention" (global context) to maintain a balance between compute efficiency and long-range dependency modeling. The use of QK-Normalization replaces traditional soft-capping to enhance training stability at scale.

### Training Dataset
The model is trained on the **TinyStories** dataset from Hugging Face (`roneneldan/TinyStories`), which provides synthetic stories for training high-quality small-scale narrative models.
## Model Weights and Training Data: [HuggingFace Repo](https://huggingface.co/Parthiban007/Gemma_3_270M/tree/main)

### Reference
- [Vizuara](https://youtu.be/bLDlwcl6hbA?si=IJfos5UoD-mF2dvI)

### Architecture Diagram
![Gemma Architecture](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/gemma-3-270m.webp)
