# DLCV Project: Towards Inter-Class Separation in Transformer-Based Weakly Supervised Semantic Segmentation

## Overview
This repository contains the full implementation of my DS265 (Deep Learning for Computer Vision) course project. It implements a Weakly Supervised Semantic Segmentation (WSSS) framework utilizing MCTformer+ with feature-level token orthogonality. The repository includes end-to-end training, pseudo-label generation, and evaluation pipelines for both the BraTS-2020 (medical) and PASCAL VOC 2012 (natural) datasets.

## 📥 Data and Weights Preparation
Due to file size limits, the datasets and pre-trained classification weights are hosted externally. 

**Download the required files here:** `<this_link>`

Download the folders and weights from the drive and place them in the root directory. Before running any scripts, your project folder must look exactly like this:

```text
DLCV-Project/
├── TrainingData_2d_images/      # BraTS-2020 dataset images
├── VOCdevkit/                # PASCAL VOC dataset
├── res38_cls.pth             # Pre-trained ResNet38 classification weights
├── train.csv / val.csv / test.csv
├── run_brats.sh              # BraTS Execution Script
├── run_mct_plus.sh           # VOC Phase 1 Script
├── run_psa.sh                # VOC Phase 2 Script
├── run_seg.sh                # VOC Phase 3 Script
└── ... (Python core files)
```

## 🛠️ Environment Setup
Create and activate the required Conda environment:

```bash
conda create --name dlcv_env python==3.9 -y
conda activate dlcv_env
```

Install the base dependencies:
```bash
pip install -r requirements.txt
```
*(Note: The PyTorch version in requirements.txt is specifically configured for compatibility with RTX PRO 4500 Blackwell GPUs).*

## 🚀 Execution Guide

**Important Note:** It is highly advised to run the **BraTS** implementation first, as the PASCAL VOC pipeline requires some specific dependency downgrades for the Pixel Semantic Affinity (PSA) module to function correctly.

### 1. BraTS Pipeline (Medical)
The base requirements are sufficient for BraTS. Run the complete training, map generation, and evaluation pipeline:
```bash
bash run_brats.sh
```

### 2. PASCAL VOC Pipeline (Natural)
**Dependency Fix for VOC Phase:** Before running the VOC pipeline, you must install the CRF library and downgrade specific packages to avoid compatibility errors in the PSA code:
```bash
conda install -c conda-forge pydensecrf -y
pip uninstall numpy scipy scikit-learn -y
pip install numpy==1.26.4 scipy==1.10.1 scikit-learn==1.2.2
```

Once the environment is adjusted, run the VOC pipeline in three sequential phases:
```bash
bash run_mct_plus.sh   # Train classifier and generate initial CAMs
bash run_psa.sh        # Refine pseudo-labels using Pixel Semantic Affinity
bash run_seg.sh        # Train final segmentation model (ResNet38)
```

## 🙏 Acknowledgements
A massive credit to the original [MCTformer repository](https://github.com/xulianuwa/mctformer) from which the core codebase of this project was adapted. 
