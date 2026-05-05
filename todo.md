# Project TODO List: Multi-Region Brain Tumor Segmentation

## Phase 1: Environment & Data Setup 🏗️
- [ ] Set up Python virtual environment and install `requirements.txt`.
- [ ] Download and extract BraTS 2020 Training/Validation data from Kaggle.
- [ ] Verify data integrity: Ensure each patient has 4 modalities (T1, T1ce, T2, FLAIR) and a `_seg.nii.gz` file.
- [ ] Create a data splitting script (Train/Val split) if not using the default BraTS validation set.

## Phase 2: Data Preprocessing Pipeline 🧪
- [ ] **Modality Stacking:** Implement a script to stack 4 MRI channels into a single 4D tensor $(4, D, H, W)$.
- [ ] **Intensity Normalization:** Implement Z-score normalization (per-channel, ignoring zero-intensity background).
- [ ] **Label Mapping:** Write logic to convert labels $(0, 1, 2, 4)$ into three nested binary masks:
    - WT (Whole Tumor: 1+2+4)
    - TC (Tumor Core: 1+4)
    - ET (Enhancing Tumor: 4)
- [ ] **Spatial Transforms:** Implement random spatial cropping (e.g., $128 \times 128 \times 128$) or padding using `MONAI`.

## Phase 3: Model Architecture Implementation 🧠
- [ ] **Baseline Model:** Implement or configure the **Standard 3D U-Net**.
- [ ] **Experimental Model:** Implement the **3D Attention U-Net** with attention gates on skip connections.
- [ ] **Loss Function:** Implement a `DiceLoss` class and a hybrid `Dice + BCE` loss.
- [ ] **Complexity Check:** Verify model parameters and GPU memory consumption (use `torchsummary` or similar).

## Phase 4: Training & Optimization ⚡
- [ ] Set up the training loop with Adam optimizer and Learning Rate Scheduler.
- [ ] Integrate TensorBoard to track training/validation Dice scores and loss curves.
- [ ] Implement **Sliding Window Inference** for validation (since validation scans must be evaluated at full resolution).
- [ ] Run baseline 3D U-Net training.
- [ ] Run 3D Attention U-Net training.

## Phase 5: Evaluation & Analysis 📊
- [ ] **Quantitative Metrics:** Calculate per-region Dice scores for WT, TC, and ET.
- [ ] **Boundary Metrics:** Implement 95th percentile Hausdorff Distance (HD95).
- [ ] **Comparative Analysis:** Create tables comparing U-Net vs. Attention U-Net performance.
- [ ] **Failure Case Analysis:** Save 2D slices of cases where the model failed (e.g., small ET regions or noisy boundaries).

## Phase 6: Final Documentation 📄
- [ ] Finalize GitHub repository structure and clean up code comments.
- [ ] Generate LaTeX plots for loss convergence and Dice score improvements.
- [ ] Draft the technical report explaining why Dice loss was superior to Cross-Entropy for this dataset.
