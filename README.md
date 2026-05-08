# rarity-aware-corner-case-detection
Rarity-Aware Corner-Case Detection Framework for Autonomous Driving Visual Perception
Overview
Autonomous driving visual perception faces a severe long-tailed distribution challenge: detectors optimized for common categories (vehicles, pedestrians) frequently fail to detect safety-critical but statistically rare objects such as strollers, traffic cones, hand carts, and sentry boxes.

Requirements

Python 3.10+
PyTorch >= 2.0
Ultralytics >= 8.0
NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060 Laptop)
CUDA 11.8+

# 1. Clone this repository
git clone https://github.com/YOUR_USERNAME/CISA-YOLO.git
cd CISA-YOLO

# 2. Install dependencies
pip install ultralytics torch torchvision

# 3. Verify installation
python -c "import ultralytics; ultralytics.checks()"

Datasets
CODA (Primary Benchmark)
The CODA (Corner Cases for Object Detection in Autonomous Driving) benchmark is specifically designed for long-tailed corner-case evaluation.

Download: https://coda-dataset.github.io/
4,884 training images / 4,884 validation images
43 object categories (29 in validation split)

BDD100K (Cross-dataset Generalization)

Download: https://bdd-data.berkeley.edu/
62,993 training images / 10,000 validation images
10 object categories

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
    --epochs 200 \
    --batch 32 \
    --imgsz 640 \
    --device 0

Acknowledgements
This work builds upon the following open-source projects:

Ultralytics YOLOv11
EMA Module — Efficient Multi-scale Attention
CODA Dataset
BDD100K Dataset
