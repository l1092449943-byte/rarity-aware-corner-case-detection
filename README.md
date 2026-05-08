# Rarity-Aware Corner-Case Detection Framework for Autonomous Driving Visual Perception



This repository contains the official implementation of CISA-YOLO (Corner-case Instance Sensitive Attention YOLO)

> Rarity-Aware Corner-Case Detection Framework for Autonomous Driving Visual Perception


---

Overview

Autonomous driving visual perception faces a severe long-tailed distribution challenge: detectors optimized for common categories (vehicles, pedestrians) frequently fail to detect safety-critical but statistically rare objects such as strollers, traffic cones, hand carts, and sentry boxes.

CISA-YOLO addresses this by integrating three complementary components built upon the YOLOv11n backbone:

| Component | Purpose |
|-----------|---------|
| **CSAM** — Corner-case Sensitive Attention Module | Enhances feature representations for rare and small objects via serial EMA + rarity-aware CISA gating |
| **P2 Detection Head** | Adds 160×160 high-resolution detection at stride 4 for ultra-small targets |
| **AWL** — Asymmetric Wasserstein Loss | Combines CIoU and NWD (6:4 ratio) to improve bounding box regression for tiny instances |



## Requirements

- Python 3.10+
- PyTorch >= 2.0
- Ultralytics >= 8.0
- NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Laptop)
- CUDA 11.8+

---

## Installation

```bash
# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/CISA-YOLO.git
cd CISA-YOLO

# 2. Install dependencies
pip install ultralytics torch torchvision

# 3. Verify installation
python -c "import ultralytics; ultralytics.checks()"
```

---

## Datasets

### CODA (Primary Benchmark)
The CODA (Corner Cases for Object Detection in Autonomous Driving) benchmark is specifically designed for long-tailed corner-case evaluation.
- Download: [https://coda-dataset.github.io/](https://coda-dataset.github.io/)
- 4,884 training images / 4,884 validation images
- 43 object categories (29 in validation split)

### BDD100K (Cross-dataset Generalization)
- Download: [https://bdd-data.berkeley.edu/](https://bdd-data.berkeley.edu/)
- 62,993 training images / 10,000 validation images
- 10 object categories



---

## Usage

### Training

```bash
# Train CISA-YOLO on CODA (300 epochs recommended for P2 head convergence)
python train.py \
    --model yolo11_ema_cisa_p2.yaml \
    --data coda.yaml \
    --epochs 300 \
    --batch 32 \
    --imgsz 640 \
    --device 0

# Train on BDD100K
python train.py \
    --model yolo11_ema_cisa_p2.yaml \
    --data bdd100k.yaml \
    --epochs 300 \
    --batch 32 \
    --imgsz 640 \
    --device 0
```



---



## Model Architecture

CISA-YOLO is built upon **YOLOv11n** with three targeted modifications:

### 1. Corner-case Sensitive Attention Module (CSAM)
Inserted at the P3 feature level (stride 8, 80×80 resolution), CSAM serially combines:
- **EMA** (Efficient Multi-scale Attention): stabilizes feature representations via cross-dimensional channel-spatial recalibration
- **CISA** (Corner-case Instance Sensitive Attention): applies rarity gating by detecting local-context discrepancy, characteristic of rare objects that appear visually distinct from their surroundings

### 2. P2 Detection Head
An auxiliary detection head at stride 4 (160×160 resolution), providing 4× the spatial density of the standard P3 head. This is the critical component enabling simultaneous overall mAP improvement and long-tail gains.

### 3. Asymmetric Wasserstein Loss (AWL)
A hybrid localization loss combining:
- **CIoU** (weight α = 0.6): preserves localization accuracy for common objects
- **NWD** (weight 1−α = 0.4): provides stable gradients for tiny bounding boxes via Wasserstein distance on Gaussian-modeled boxes

```
M_fused = α · CIoU + (1 - α) · NWD
L_AWL   = 1 - M_fused
```

### Model Complexity

| Model | Parameters | GFLOPs | Inference (ms) |
|-------|-----------|--------|----------------|
| YOLOv11n (baseline) | 2.59M | 6.3 | ~2.0 |
| CISA-YOLO | 2.68M | 10.5 | ~2.0 |

---



---



## Acknowledgements

This work builds upon the following open-source projects:
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)
- [EMA Module](https://github.com/YOLOv8-YOLOv11-Segmentation-Studio) — Efficient Multi-scale Attention
- [CODA Dataset](https://coda-dataset.github.io/)
- [BDD100K Dataset](https://bdd-data.berkeley.edu/)
