# Diffusion LLM Architecture

Diffusion LLM is a minimal, open-source implementation of a discrete diffusion-based language model built using a Transformer Encoder architecture. This repository demonstrates the core concepts of diffusion language modeling with timestep embeddings, token masking, and iterative denoising on the TinyStories dataset.

### Architecture Configuration

* **Dataset:** TinyStories
* **Vocabulary Size:** 8,000
* **Context Length:** 256 tokens
* **Embedding Dimension:** 384
* **Number of Heads:** 6
* **Number of Layers:** 6
* **Feed Forward Dimension:** 1,536
* **Activation:** GELU
* **Normalization:** Pre-LayerNorm
* **Dropout:** 0.1
* **Diffusion Steps:** 64
* **Weight Tying:** Input Embedding ↔ Output Projection

### Architecture Explanation

Diffusion LLM uses a Transformer Encoder to reconstruct masked tokens from progressively corrupted text sequences. The model combines token, positional, and timestep embeddings before processing them through multiple encoder layers. During training, random tokens are replaced with a special mask token according to a diffusion schedule, and the model learns to recover the original tokens using a denoising objective.

### References
- [Vizuara](https://youtube.com/playlist?list=PLPTV0NXA_ZShhDDPgy1ygii42nwOngUaf&si=im1OKmoO126dd90T)


Open to collaborations, feedback, and contributions — feel free to raise issues or share ideas to improve the project.
