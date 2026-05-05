# Multi-Region Brain Tumor Segmentation in 3D MRI

This project implements an end-to-end medical imaging pipeline for the automated segmentation of nested tumor sub-regions from multi-modal 3D MRI scans using the **BraTS 2020** dataset.

## Project Overview
Delineating tumor boundaries in brain MRI scans is a critical yet time-consuming task for treatment planning. This solution provides a deep learning framework to simultaneously segment three nested tumor sub-regions:
1.  **Whole Tumor (WT):** Includes all tumor tissues (Labels 1, 2, and 4).
2.  **Tumor Core (TC):** Includes the necrotic core and enhancing tumor (Labels 1 and 4).
3.  **Enhancing Tumor (ET):** Focuses specifically on the contrast-enhancing region (Label 4).

The system processes four co-registered MRI modalities: **T1, T1ce, T2, and FLAIR**.

## Architectures
We compare two state-of-the-art volumetric (3D) architectures to analyze the impact of spatial attention mechanisms on segmentation accuracy:

### 1. 3D U-Net
A volumetric extension of the classic U-Net. It utilizes an encoder-decoder structure with skip connections to recover spatial information lost during downsampling, enabling dense voxel-wise predictions.

### 2. 3D Attention U-Net
An advanced variant that integrates **Attention Gates (AGs)** within the skip connections. These gates automatically learn to focus on target structures of varying shapes and sizes, suppressing irrelevant background activations and highlighting tumor-specific features before concatenation.

## Key Technical Features
- **Volumetric Processing:** Handling 3D NIfTI (.nii.gz) data using `NiBabel` and `MONAI`.
- **Dice Loss Optimization:** Implementation of Dice Loss to address the extreme class imbalance (foreground vs. background) inherent in medical imaging, where standard Cross-Entropy often fails.
- **Advanced Metrics:** Evaluation using **Dice Similarity Coefficient (DSC)** for overlap and **95th percentile Hausdorff Distance (HD95)** for boundary precision.
- **Data Pipeline:** Includes Z-score normalization, random spatial cropping/patching, and multi-modal channel stacking.

## Workflow
1.  **Preprocessing:** Multi-modal stacking, intensity normalization, and label remapping into nested regions.
2.  **Training:** Comparison of 3D U-Net vs. 3D Attention U-Net using a hybrid Dice + BCE loss.
3.  **Inference:** Sliding window inference for full-volume reconstruction.
4.  **Evaluation:** Statistical analysis of performance across different tumor sub-regions.

## STRUCTURE
```text
/project_root
├── data/
│   ├── BraTS2020_TrainingData/
│   └── BraTS2020_ValidationData/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── losses.py
│   └── train.py
├── results/
│   ├── models/
│   ├── plots/
│   └── qualitative/
├── requirements.txt
└── README.md
```

## Dataset
This work utilizes the **BraTS 2020 (Brain Tumor Segmentation Challenge)** dataset, comprising 369 multi-institutional MRI scans with expert-annotated labels. (https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
