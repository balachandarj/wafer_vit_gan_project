# Wafer Defect Detection with ViT + VAE-GAN (Complete Implementation)

This repository contains a **complete, minimal, and runnable** pipeline for wafer defect detection combining a lightweight
Vision Transformer (ViT) classifier and a VAE-GAN generative model for anomaly scoring and synthetic data augmentation.

## Features
- Synthetic wafer dataset generator with four classes: `normal`, `scratch`, `particle`, `edge_ring`.
- Lightweight **Vision Transformer** from scratch (PyTorch) for supervised defect classification.
- Simple **VAE-GAN** (Conv encoder/decoder + discriminator) for reconstruction-based anomaly detection.
- **Fusion evaluation** combining ViT posterior and VAE-GAN reconstruction error for improved robustness.
- Attention map visualization (ViT attention rollout) and confusion matrix plotting.
- CPU-friendly defaults (small models, small image size, few epochs).

## Quick Start

### 0) Requirements
```
pip install -r requirements.txt
```

### 1) Generate Synthetic Dataset
```
python tools/generate_synthetic_wafer.py --out data/synth --n_per_class 500 --img_size 128
```

### 2) Train ViT Classifier
```
python src/train_vit.py --data data/synth --epochs 5 --batch_size 64 --img_size 128 --save ckpt/vit.pth
```

### 3) Train VAE-GAN
```
python src/train_vae_gan.py --data data/synth --epochs 5 --batch_size 64 --img_size 128 --save ckpt/vaegan.pth
```

### 4) Evaluate & Fuse Scores
```
python src/fusion_eval.py --data data/synth --vit ckpt/vit.pth --vaegan ckpt/vaegan.pth --img_size 128
```

### 5) Visualize Attention
```
python src/visualize_attention.py --data data/synth --vit ckpt/vit.pth --img_size 128 --out outputs/attention
```

> Note: Increase `--epochs` for better results; defaults are kept small for a fast demo on CPU.

## Folder Structure
```
wafer_vit_gan_project/
  ├── data/                 # your dataset goes here (generated under data/synth)
  ├── ckpt/                 # saved checkpoints
  ├── outputs/              # generated figures/plots
  ├── src/
  │   ├── models/
  │   │   ├── vit.py
  │   │   └── vae_gan.py
  │   ├── data/
  │   │   └── wafer_dataset.py
  │   ├── train_vit.py
  │   ├── train_vae_gan.py
  │   ├── fusion_eval.py
  │   ├── visualize_attention.py
  │   └── utils.py
  ├── tools/
  │   └── generate_synthetic_wafer.py
  └── requirements.txt
```

## License
MIT

