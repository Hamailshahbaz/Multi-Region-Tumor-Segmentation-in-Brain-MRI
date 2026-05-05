## 📄 Component Details

### 1. `data/` (The Dataset Storage)
This directory is treated as read-only to preserve the integrity of the raw NIfTI files.
*   **`BraTS2020_TrainingData/`**: Contains 369 folders (e.g., `BraTS20_Training_001`). Each folder includes four modality MRI files (T1, T1ce, T2, FLAIR) and the ground-truth segmentation (`_seg.nii.gz`).
*   **`BraTS2020_ValidationData/`**: Contains the scans used for final model testing. These typically do not include labels and are used to evaluate the model's generalization.

### 2. `src/data_loader.py` (The Input Pipeline)
Handles the conversion of raw medical data into deep-learning-ready tensors.
*   **Multi-modal Stacking**: Consolidates the four MRI modalities into a single 4-channel tensor of shape $(4, 155, 240, 240)$.
*   **Label Mapping**: A custom function to transform original labels $(0, 1, 2, 4)$ into three nested binary regions:
    *   **Whole Tumor (WT)**: Labels 1, 2, and 4.
    *   **Tumor Core (TC)**: Labels 1 and 4.
    *   **Enhancing Tumor (ET)**: Label 4.
*   **Transforms**: Utilizes `MONAI` for Z-score normalization, random spatial cropping (to fit 3D volumes into GPU memory), and data augmentation (rotations, flips).

### 3. `src/model.py` (Architecture Definitions)
Contains the PyTorch class definitions for the two comparative models.
*   **`class UNet3D`**: The baseline volumetric U-Net with standard skip connections.
*   **`class AttentionUNet3D`**: The advanced variant incorporating **Attention Gates**. These gates filter the encoder features using decoder context to suppress background noise and highlight tumor structures.

### 4. `src/losses.py` (Optimization Logic)
Fulfills the project requirement to address class imbalance in medical imaging.
*   **`DiceLoss` Class**: Implements the Dice Similarity Coefficient (DSC) formula. Unlike Cross-Entropy, which is biased toward the majority background class, Dice Loss focuses on the spatial overlap of the foreground.
*   **Hybrid Loss**: Combines `Dice + BCE` (Binary Cross-Entropy) to leverage the spatial focus of Dice and the gradient stability of BCE.

### 5. `src/train.py` (The Execution Hub)
The main entry point for training and model validation.
*   **Hyperparameters**: Standardizes the learning rate ($1e-4$), batch size, and epoch count for fair comparison between models.
*   **Validation Loop**: Implements **Sliding Window Inference**. This allows the model to process large 3D volumes in smaller patches (e.g., $128^3$) and reconstruct the full-volume mask.
*   **Logging**: Tracks `mIoU` and `Dice` scores in real-time via **TensorBoard**.

### 6. `results/` (The Output Vault)
*   **`models/`**: Stores serialized `.pth` weights for the best-performing checkpoints.
*   **`plots/`**: Automatically generates convergence and metric comparison graphs for the LaTeX report.
*   **`qualitative/`**: Saves side-by-side 2D slices comparing Ground Truth masks with Model Predictions to identify failure cases.
```
