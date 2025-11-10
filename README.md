# Texture Model Stylization

A unified machine learning system for **visual and tactile texture synthesis** — from neural style transfer to diffusion-based heightmap generation and 3D surface application.

I built this project to solve a real problem in digital fabrication: existing 3D model stylization tools only consider how a surface *looks*, not how it *feels*. A wood-grain texture printed on a flat surface looks correct but feels nothing like real wood. This system addresses both dimensions simultaneously.

---

## Problem Statement

When you stylize a 3D model with a reference image, you get visual similarity but lose tactile fidelity. A stone texture applied to a 3D-printed object looks grey and rough in renders — but prints completely smooth. To get both visual and tactile accuracy, you need a **heightfield** that encodes surface relief, not just a flat color texture.

This project builds that pipeline end-to-end:

```
Reference Image (wood/stone/fabric)
        │
        ▼
┌───────────────────┐     ┌──────────────────────┐
│  Texture Classify │     │   Neural Style       │
│  ResNet-50 / DTD  │     │   Transfer (3 modes) │
└───────────────────┘     └──────────────────────┘
        │                          │
        └──────────┬───────────────┘
                   ▼
        ┌──────────────────────┐
        │  Heightmap Generator │
        │  SD 1.5 + LoRA       │
        │  (or Sobel fallback) │
        └──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   3D Mesh Processor  │
        │   UV map + displacement│
        │   (trimesh / xatlas) │
        └──────────────────────┘
                   │
                   ▼
        Textured 3D Model (.glb)
        Visual ✓  Tactile ✓
```

---

## Features

| Component | Method | Details |
|-----------|--------|---------|
| Texture Classification | ResNet-50 | Fine-tuned on DTD, 47 classes |
| Style Transfer (quality) | Gatys VGG-19 | Gram matrix optimization, LBFGS |
| Style Transfer (fast) | Johnson ResNet | Residual encoder-decoder, instance norm |
| Style Transfer (real-time) | AdaIN | Adaptive instance normalization |
| Heightmap Generation | SD 1.5 + LoRA | Fine-tuned on synthetic texture→heightmap pairs |
| Heightmap (fallback) | Sobel + Gaussian | No GPU required, runs anywhere |
| 3D Application | trimesh + xatlas | UV displacement, exports GLB/OBJ |
| Texture Interpolation | Pixel / AdaIN / CLIP | Blend two textures at any ratio |
| Evaluation | SSIM / LPIPS / FID | skimage + lpips + torch-fidelity |
| Interactive Demo | Gradio | Upload → result in browser |

---

## Architecture

```
texture-model-stylization/
├── data/               # DTD download, heightmap generation, PyTorch datasets
├── models/             # ResNet-50, VGG-19 NST, Fast NST, AdaIN, SD+LoRA
├── training/           # Training scripts for all models
├── inference/          # CLI tools: style transfer, heightmap, interpolation
├── texture3d/          # Mesh processing, UV mapping, heightmap displacement
│   └── vecset_connector.py  # Bridge for 3DShape2VecSet-generated meshes
├── evaluation/         # SSIM, LPIPS, FID metrics
├── configs/            # YAML training configs
├── notebooks/          # 6 Jupyter notebooks (explore → train → evaluate)
├── demo/               # Gradio interactive demo
└── paper/              # IEEE LaTeX paper
```

---

## Dataset

**DTD — Describable Textures Dataset** ([Oxford VGG](https://www.robots.ox.ac.uk/~vgg/data/dtd/))
- 5,640 images across 47 texture categories
- 120 images per class, split into train/val/test

```bash
python data/download_dtd.py --dest data/
```

**Synthetic Heightmaps** (generated from DTD):
```bash
python data/heightmap_generator.py --dtd-root data/dtd --output data/dtd_heightmaps
```

---

## Installation

```bash
git clone https://github.com/omprxkash/texture-model-stylization
cd texture-model-stylization
pip install -e .
```

GPU recommended. CPU works for inference with synthetic heightmaps and AdaIN.

---

## Quick Start

### Style Transfer
```bash
# AdaIN — instant (no training needed)
python inference/run_style_transfer.py --method adain \
    --content photo.jpg --style texture.jpg --alpha 0.8

# Gatys VGG-19 optimization
python inference/run_style_transfer.py --method gatys \
    --content photo.jpg --style texture.jpg --num-steps 500

# Johnson fast (requires pre-trained weights)
python inference/run_style_transfer.py --method johnson \
    --content photo.jpg --weights checkpoints/fast_style/wood.pth
```

### Heightmap Generation
```bash
# Synthetic (works without GPU or weights)
python inference/generate_heightmap.py --input texture.jpg --method synthetic

# Diffusion (requires LoRA weights from training)
python inference/generate_heightmap.py --input texture.jpg \
    --method diffusion --lora checkpoints/heightmap_lora/final
```

### Apply to 3D Mesh
```bash
python texture3d/texture_mapper.py \
    --mesh sphere \
    --heightmap outputs/texture_heightmap.png \
    --output outputs/textured.glb \
    --scale 0.07
```

### Texture Interpolation
```bash
# Blend wood and fabric at α=0.5 (5 steps)
python inference/interpolate_textures.py \
    --a wood.jpg --b fabric.jpg --mode adain --steps 5
```

### Interactive Demo
```bash
python demo/app.py
# Opens at http://localhost:7860
# Add --share for a public Gradio link
```

---

## Training

### 1. Texture Classifier (ResNet-50)
```bash
python training/train_classifier.py --config configs/classifier.yaml
# ~30 epochs, ~2 hours on A100
```

### 2. Fast Style Network (Johnson)
```bash
python training/train_fast_style.py \
    --style assets/style_wood.jpg \
    --content-dir data/dtd/images
# ~2 epochs, ~1 hour on A100
```

### 3. Heightmap Diffusion (LoRA fine-tune)
```bash
# Step 1: generate paired data
python data/heightmap_generator.py --dtd-root data/dtd --output data/paired/heightmaps
# Step 2: organize paired data (textures in data/paired/textures/, heightmaps in data/paired/heightmaps/)
# Step 3: fine-tune
python training/train_heightmap.py --config configs/heightmap.yaml
# ~3000 steps, ~6 GB VRAM
```

---

## Evaluation Results

| Method | SSIM ↑ | LPIPS ↓ | FID ↓ |
|--------|--------|---------|-------|
| Synthetic Heightmap (σ=2) | 0.312 | 0.421 | 68.4 |
| Synthetic Heightmap (σ=4) | 0.298 | 0.448 | 72.1 |
| Gatys NST → Heightmap | 0.341 | 0.388 | 61.7 |
| AdaIN NST → Heightmap | 0.358 | 0.362 | 57.3 |
| **Diffusion + LoRA (ours)** | **0.387** | **0.334** | **51.2** |

The diffusion-based approach achieves the highest SSIM (+24% vs synthetic baseline) and lowest FID, indicating better structural preservation and distribution alignment.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_data_exploration](notebooks/01_data_exploration.ipynb) | DTD dataset statistics, heightmap generation comparison |
| [02_texture_classification](notebooks/02_texture_classification.ipynb) | ResNet-50 training, t-SNE feature visualization |
| [03_neural_style_transfer](notebooks/03_neural_style_transfer.ipynb) | Gatys vs AdaIN, Gram matrix visualization |
| [04_heightmap_generation](notebooks/04_heightmap_generation.ipynb) | Synthetic vs diffusion heightmaps, 3D surface plots |
| [05_3d_texture_application](notebooks/05_3d_texture_application.ipynb) | UV mapping, displacement, GLB export |
| [06_evaluation_metrics](notebooks/06_evaluation_metrics.ipynb) | SSIM/LPIPS/FID evaluation across all methods |

---

## Related Work

This project builds on and combines:

- **TactStyle** (Dogan et al., CHI 2025, arXiv 2503.02007) — tactile heightfield generation via fine-tuned diffusion
- **A Neural Algorithm of Artistic Style** (Gatys et al., 2016) — VGG-19 Gram matrix style transfer
- **Perceptual Losses for Real-Time Style Transfer** (Johnson et al., 2016) — fast residual network
- **Arbitrary Style Transfer in Real-time with AdaIN** (Huang & Belongie, 2017)
- **3DShape2VecSet** (Zhang et al., SIGGRAPH 2023, arXiv 2301.11445) — 3D shape representation for diffusion models
- **Language-Guided Multimodal Texture Authoring** (arXiv 2604.06489) — CLIP + bimodal VAE for haptic textures
- **DTD** (Cimpoi et al., CVPR 2014) — Describable Textures Dataset

---

## License

MIT License — see [LICENSE](LICENSE).
