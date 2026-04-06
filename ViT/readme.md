# Vision Transformer (ViT) Architecture

The Vision Transformer (ViT) architecture applies the Transformer model, originally designed for NLP tasks, directly to sequences of image patches for image classification.

### Architecture Configuration
- **Input Resolution:** 28x28 (MNIST)
- **Patch Size:** 7x7
- **Number of Patches:** 16 + 1 ([CLS] token)
- **Embedding Dimension:** 16
- **Number of Heads:** 1
- **Transformer Encoder Layers:** 1
- **MLP Dimension:** 16

### Architecture Explanation
The input image is partitioned into non-overlapping patches, which are then linearly projected into an embedding space. A learnable classification token ([CLS]) and positional embeddings are added to the patch sequence to retain spatial information. This combined input is processed through a Transformer Encoder composed of Multi-Head Self-Attention and MLP blocks. Final classification is performed by a head attached to the [CLS] token output.

### Training Dataset
The model is trained and evaluated on the **MNIST** dataset of handwritten digits, containing 60,000 training images and 10,000 testing images.


### Reference
- [Vizuara](https://youtu.be/ZRo74xnN2SI?si=K6O2jUXxZQ_ar31T)

### Architecture Diagram
![ViT Architecture](https://raw.githubusercontent.com/google-research/vision_transformer/main/vit_architecture.png)
