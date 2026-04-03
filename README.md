# SafePath: AI-Powered Hazard Detection System

<p align="center">
  <strong>Real-time hazard detection for low-light navigation using semantic segmentation</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#license">License</a>
</p>

---

## Overview

**SafePath** is an AI-powered computer vision application designed to assist users in navigating safely in low-light or complex environments. Using DeepLabV3+ semantic segmentation with MobileNetV3 backbone, SafePath detects potential hazards—such as potholes, poles, and uneven terrain—and provides real-time visual and auditory alerts.

**Target Platform:** Samsung Galaxy Z Fold 6

## Features

- 🎯 **Real-time Hazard Detection** - Identifies potholes, obstacles, and uneven terrain
- 🌙 **Low-Light Optimization** - Specialized preprocessing for dark environments
- 📱 **Mobile Deployment** - Optimized for on-device inference using NPU acceleration
- 🎨 **Visual Overlays** - Color-coded hazard highlighting on live camera feed
- 📢 **Audio Alerts** - Configurable auditory warnings for detected hazards
- 📄 **PDF Reports** - Automated incident documentation with snapshots

## Architecture

```
Camera Feed → Low-Light Enhancement → DeepLabV3+ Inference → 
Post-Processing → Hazard Classification → Overlay + Alerts → Report Generation
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA 11.8+ (for GPU training)
- Samsung Galaxy Z Fold 6 (for deployment)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/safepath.git
cd safepath

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

For the fastest project workflow, use the bundled helper script:

```bash
# Show available commands
./quick_start.sh help

# Verify datasets, checkpoints, and export artifacts
./quick_start.sh check

# Validate the mini dataset
./quick_start.sh validate-mini

# Validate the full processed BDD100K dataset
./quick_start.sh validate-full
```

The script also supports mini training, full-dataset smoke/full training, evaluation, ONNX export, and TensorBoard shortcuts.

### Dataset Setup

Download required datasets:

1. **BDD100K** - [bdd-data.berkeley.edu](https://bdd-data.berkeley.edu/)
2. **Cityscapes** - [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)
3. **NightCity** - [github.com/AdaBar/NightCity](https://github.com/AdaBar/NightCity)
4. **Cracks and Potholes** - [datasetninja.com](https://datasetninja.com/cracks-and-potholes-in-road)

```bash
# Organize datasets
mkdir -p data/{bdd100k,cityscapes,nightcity,potholes}
# Place downloaded datasets in respective directories
```

## Usage

### Training

```bash
# Train baseline model on the mini dataset
./quick_start.sh train-mini-base

# Train proposed DeepLabV3+ model on the mini dataset
./quick_start.sh train-mini-prop

# Run a 1-epoch smoke test on full BDD100K before long training
./quick_start.sh train-full-smoke

# Train baseline model on full BDD100K
./quick_start.sh train-full-base

# Train proposed model on full BDD100K
./quick_start.sh train-full-prop
```

For direct script execution:

```bash
python scripts/train_baseline.py \
  --config configs/train_mini.yaml \
  --data-dir data/processed/bdd100k_mini \
  --output-dir models/baseline_mobilenetv3_small

python scripts/train_proposed.py \
  --config configs/train_mini.yaml \
  --data-dir data/processed/bdd100k_mini \
  --output-dir models/proposed_deeplabv3plus

python scripts/train_baseline.py \
  --data-dir data/processed/bdd100k \
  --output-dir models/baseline_full_smoke \
  --epochs 1 \
  --batch-size 4 \
  --lr 0.001 \
  --num-workers 0
```

### Validation and Evaluation

```bash
# Validate processed dataset structure
./quick_start.sh validate-mini

# Validate the full processed BDD100K validation split
./quick_start.sh validate-full

# Evaluate baseline vs proposed checkpoints
./quick_start.sh eval-mini

# Evaluate full BDD100K checkpoints on the validation split
./quick_start.sh eval-full-val
```

For the full BDD100K workflow, use the helper script first. It already includes the corrected `--split val` and `--num-workers 0` settings for the current processed dataset.

### Export for Mobile

```bash
# Export both trained models to ONNX
./quick_start.sh export
```

Equivalent direct commands:

```bash
python scripts/export_onnx.py --model baseline \
  --checkpoint models/baseline_mobilenetv3_small/checkpoints/best.pt \
  --output exports/baseline.onnx \
  --input-size 128 256

python scripts/export_onnx.py --model proposed \
  --checkpoint models/proposed_deeplabv3plus/checkpoints/best.pt \
  --output exports/proposed.onnx \
  --input-size 128 256
```

### TensorBoard

```bash
./quick_start.sh tensorboard
```

## Project Structure

```
safepath/
├── configs/                 # Configuration files
│   └── config.yaml
├── data/                    # Dataset storage (not in repo)
│   ├── bdd100k/
│   ├── cityscapes/
│   ├── nightcity/
│   └── potholes/
├── docs/                    # Documentation
│   ├── adr/                 # Architecture Decision Records
│   │   └── ADR-0001-deeplabv3-mobile-architecture.md
│   ├── prd/                 # Product Requirements
│   │   └── safepath-prd.md
│   └── research/            # Research notes
├── models/                  # Saved model weights (not in repo)
├── notebooks/               # Jupyter notebooks
├── scripts/                 # Training and inference scripts
├── src/                     # Source code
│   ├── data/                # Dataset handling
│   │   └── dataset.py
│   ├── models/              # Model definitions
│   │   ├── __init__.py
│   │   └── deeplabv3plus.py
│   ├── pipelines/           # Processing pipelines
│   │   └── inference.py
│   └── utils/               # Utility functions
├── tests/                   # Unit tests
├── assets/                  # Sample media
│   ├── samples/
│   └── reports/
├── requirements.txt         # Python dependencies
└── README.md
```

## Model Performance

| Metric | Target | Status |
|--------|--------|--------|
| mIoU | > 85% | 🔄 Training |
| Hazard F1 | > 80% | 🔄 Training |
| Inference FPS | > 5 | 🔄 Testing |
| Model Size | < 10 MB | ✅ ~6 MB (quantized) |

## Hazard Classes

| Class | Color | Description |
|-------|-------|-------------|
| Safe Path | 🟢 Green | Clear, walkable area |
| Road | ⬜ Gray | Paved road surface |
| Sidewalk | ⬜ Light Gray | Pedestrian walkway |
| Pothole | 🔴 Red | Hole in surface |
| Pole | 🟠 Orange | Vertical obstacles |
| Obstacle | 🟡 Yellow | General obstacles |
| Water/Puddle | 🔵 Blue | Standing water |
| Uneven Terrain | 🟡 Yellow | Rough/irregular surface |

## Documentation

- [Product Requirements Document (PRD)](docs/prd/safepath-prd.md)
- [Architecture Decision Record (ADR)](docs/adr/ADR-0001-deeplabv3-mobile-architecture.md)

## Timeline

| Phase | Description | Due Date | Status |
|-------|-------------|----------|--------|
| Phase 1 | Planning & Research | Feb 27, 2026 | ✅ Complete |
| Phase 2 | Model Development | Mar 15, 2026 | 🔄 In Progress |
| Phase 3 | Proof of Concept | Apr 2, 2026 | ⬜ Pending |
| Phase 4 | Finalization | May 7, 2026 | ⬜ Pending |

## References

### Key Papers
- Chen et al. (2018) - "Encoder-Decoder with Atrous Separable Convolution" (DeepLabV3+)
- Howard et al. (2019) - "Searching for MobileNetV3"
- Liu et al. (2025) - "MFA-DeepLabv3+: A Lightweight Semantic Segmentation Network"

### Reference Projects
- [nmhaddad/semantic-segmentation](https://github.com/nmhaddad/semantic-segmentation) - Off-road DeepLabV3+
- [meiqisheng/DeepLabv3plus](https://github.com/meiqisheng/DeepLabv3plus) - MobileNet-optimized

## License

This project is developed for academic purposes as part of AI688 - Image and Vision Computing at Long Island University.

## Author

**Dwayne Crichlow**  
AI688 - Image and Vision Computing (Section 1)  
February - May 2026

---

<p align="center">
  <sub>Built with ❤️ for safer navigation</sub>
</p>
