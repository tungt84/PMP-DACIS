<div align="center">

# PMP-DACIS: Disease-Aware Pruning for Few-Shot Plant Pathology

[![arXiv](https://img.shields.io/badge/arXiv-2601.02353-b31b1b.svg)](https://arxiv.org/abs/2601.02353)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)](https://pytorch.org/)

</div>

> **78% parameter reduction** | **3.6x speedup** | **<2% accuracy drop** | **Edge-deployable**

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PMP-DACIS Framework                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Input           DACIS Scoring          3-Stage PMP           Edge         │
│   ──────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   [Leaf Image] ──► [Channel Analysis] ──► [Prune→Meta→Prune] ──► [Deploy]   │
│                         │                         │                         │
│                    ┌────┴─────┐              ┌────┴────┐                    │
│                    │ G + V + D│              │ 11.2M → │                    │
│                    │ Scoring  │              │  2.5M   │                    │
│                    └──────────┘              └─────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture

### DACIS Scoring Pipeline

```
                    ┌──────────────────────────────────────────┐
                    │         DACIS = λ₁·G + λ₂·V + λ₃·D       │
                    └──────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
            │  Gradient (G) │   │  Variance (V) │   │   Fisher (D)  │
            │   λ₁ = 0.3    │   │   λ₂ = 0.2    │   │   λ₃ = 0.5    │
            │               │   │               │   │               │
            │  ∂L/∂W        │   │  Var(GAP(a))  │   │  Sᵦ / Sᵥ      │
            └───────────────┘   └───────────────┘   └───────────────┘
```

### Three-Stage PMP Framework

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    STAGE 1      │     │    STAGE 2      │     │    STAGE 3      │
│  Initial Prune  │────►│  Meta-Learning  │────►│ Refined Prune   │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│                 │     │                 │     │                 │
│  11.2M → 6.7M   │     │  N-way K-shot   │     │  6.7M → 2.5M    │
│   (40% prune)   │     │  60K episodes   │     │  (38% prune)    │
│                 │     │                 │     │                 │
│  DACIS scores   │     │  MAML inner/    │     │  DACIS × |Gₘₑₜₐ| │
│  on base data   │     │  outer loop     │     │  refined scores │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## Results

| Method | Params | 1-shot | 5-shot | 10-shot | DES |
|--------|--------|--------|--------|---------|-----|
| ProtoNet (Full) | 100% | 71.2 | 84.6 | 89.3 | 0.42 |
| Chan. Prune | 30% | 63.7 | 77.2 | 83.0 | 1.45 |
| **PMP-DACIS** | **30%** | **68.9** | **83.2** | **88.1** | **1.98** |
| **PMP-DACIS** | **22%** | 66.4 | 81.0 | 86.3 | **2.31** |

---

## Dataset

Download the PlantVillage dataset:

```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/plantvillage
```

| Dataset | Classes | Images | Link |
|---------|---------|--------|------|
| PlantVillage | 38 | 54,306 | [GitHub](https://github.com/spMohanty/PlantVillage-Dataset) |

---

## Quick Start

```bash
# Clone
git clone https://github.com/Mudassiruddin7/PMP-DACIS.git
cd PMP-DACIS

# Install
pip install -r requirements.txt

# Download dataset
git clone https://github.com/spMohanty/PlantVillage-Dataset.git data/plantvillage

# Train
python train.py --dataset plantvillage --shots 5 --compression 0.78

# Evaluate
python evaluate.py --model checkpoints/pmp_dacis.pth --device edge
```

---

## Project Structure

```
pmp-dacis/
├── configs/
│   └── default.yaml
├── data/
│   ├── plantvillage/
│   └── plantdoc/
├── models/
│   ├── backbone.py       # ResNet-18, MobileNetV2
│   ├── dacis.py          # DACIS scoring
│   └── pmp.py            # 3-stage framework
├── utils/
│   ├── metrics.py        # DES, FSI, CSG
│   └── pruning.py        # Channel pruning
├── train.py
├── evaluate.py
└── requirements.txt
```

---

## Key Metrics

```
┌────────────────────────────────────────────────────────────────┐
│  DES = (Accuracy × FPS) / (Params × Energy)     ──► 4.7× ↑     │
│  FSI = 1 - σ_acc / μ_acc                        ──► 0.91       │
│  CSG = Acc_late / Acc_early                     ──► 0.83       │
└────────────────────────────────────────────────────────────────┘
```

---

## Hardware

| Device | FPS | Energy (mJ) | Supported |
|--------|-----|-------------|-----------|
| Raspberry Pi 4 | 12 | 85 | Yes |
| Jetson Nano | 67 | 42 | Yes |
| Edge TPU | 120 | 18 | Yes |

---

## Citation

```bibtex
@article{uddin2025pmp,
  title={Meta-Learning Guided Pruning for Few-Shot Plant Pathology on Edge Devices},
  author={Uddin, Mohammed Mudassir and Alam, Shahnawaz and Pasha, Mohammed Kaif},
  journal={arXiv preprint arXiv:2601.02353},
  year={2025}
}
```


---

## Authors

<div align="center">

**Mohammed Mudassir Uddin** · **Shahnawaz Alam** · **Mohammed Kaif Pasha**

[mohd.mudassiruddin7@gmail.com](mailto:mohd.mudassiruddin7@gmail.com) · [shahnawaz.alam1024@gmail.com](mailto:shahnawaz.alam1024@gmail.com) · [mdkaifpasha2k@gmail.com](mailto:mdkaifpasha2k@gmail.com)

*Department of CSE, MJCET, Hyderabad, India*

</div>

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**[Paper](https://arxiv.org/abs/2601.02353)** · **[Issues](https://github.com/Mudassiruddin7/PMP-DACIS/issues)** · **[Star this repo](https://github.com/Mudassiruddin7/PMP-DACIS)**

</div>
