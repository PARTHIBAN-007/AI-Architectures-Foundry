# GPT-2 Architecture

An implementation of the GPT-2 architecture, a decoder-only Transformer designed for autoregressive language modeling. It demonstrates the fundamental principles of the modern Generative Pre-trained Transformer models.

### Architecture Configuration
- **Vocabulary Size:** 50,257
- **Block Size:** 128 tokens
- **Number of Layers:** 6
- **Number of Heads:** 6
- **Embedding Dimension:** 384
- **Dropout Rate:** 0.1
- **Bias:** Enabled
- **Activation:** GELU

### Architecture Explanation
The model is built with a stack of Transformer blocks. Each block features Multi-Head Causal Self-Attention, allowing the model to attend to preceding tokens while maintaining the causality of the sequence. It uses absolute learned positional embeddings and a position-wise Feed-Forward Network (MLP). Layer normalization is applied before each sub-layer (Pre-Norm architecture) to improve training stability.

### Training Dataset
The implementation is pre-trained using the **TinyStories** dataset, focusing on generating coherent and grammatical short narratives at a small scale.

### Reference
- [Vizuara](https://youtube.com/playlist?list=PLPTV0NXA_ZShuk6u31pgjHjFO2eS9p5EV&si=mkcCv-8Rk-LZtZr2)

### Architecture Diagram
![GPT Architecture](https://jalammar.github.io/images/gpt2/gpt2-transformerblock-2.png)
