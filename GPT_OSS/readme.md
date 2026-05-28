# GPT OSS Architecture

GPT OSS is a minimal, open-source implementation of Official OpenAI GPT OSS-style architecture. This repository demonstrates a modern transformer architecture with Mixture-of-Experts (MoE) MLP blocks, rotary positional embeddings(YaRN), and efficient training utilities.

### Architecture Configuration
- **Vocabulary Size:** 50,257 (GPT-2 TikTokenizer)
- **Context Length:** 512 tokens
- **Embedding Dimension:** 128
- **Number of Heads:** 4
- **Number of Layers:** 6
- **Hidden Dimension:** 128
- **Intermediate (MLP) Dimension:** 512
- **Head Dimension:** 32
- **Key/Value Heads:** 2
- **Sliding Window Attention:** 64 tokens (alternates per layer)
- **Mixture-of-Experts (MoE):** 4 experts, 1 expert per token
- **Activation:** SwiGLU
- **Normalization:** RMSNorm
- **Rotary Positional Embeddings (RoPE):** θ=150,000, scaling factor=32

### Architecture Explanation
GPT OSS uses a decoder-only transformer with alternating sliding window and full attention layers. The model integrates Mixture-of-Experts (MoE) in the MLP blocks for increased capacity and efficiency. Rotary positional embeddings (RoPE) are used for position encoding, and RMSNorm is applied for normalization. The implementation is modular and easy to extend for research or educational purposes.

### Architecture Diagram
![GPT OSS Architecture](https://sebastianraschka.com/llm-architecture-gallery/images/architectures/gpt-oss-20b.webp)


### References
- [Vizuara YT](https://youtu.be/hBUsySdcA3I?si=edoobf2bU8zQm-5i)
- [OpenAI Github](https://github.com/openai/gpt-oss)
- [Sebastian Raschka Article](https://open.substack.com/pub/sebastianraschka/p/from-gpt-2-to-gpt-oss-analyzing-the)

Open to collaborations, feedback, and contributions — feel free to raise issues or share ideas to improve the project.