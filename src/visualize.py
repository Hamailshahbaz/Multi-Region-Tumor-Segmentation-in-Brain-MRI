#src/visualize.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from data_loader import BraTSDataset3D

def visualize_predictions(images, masks, outputs, epoch, loss_name, model_name, out_dir="results/plots"):
    os.makedirs(out_dir, exist_ok=True)

    img = images[0].cpu().numpy()      # (4, H, W, D)
    mask = masks[0].cpu().numpy()      # (3, H, W, D)
    pred = outputs[0].cpu().detach().numpy()  # (3, H, W, D)
    pred_bin = (pred > 0.5).astype(np.float32)

    # Extract middle axial slice for plotting
    mid = img.shape[-1] // 2

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    class_names = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]

    for i in range(3):
        axes[i, 0].imshow(img[1, :, :, mid], cmap='gray')  # FLAIR middle slice
        axes[i, 0].set_title(f"Input (FLAIR) - {class_names[i]}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask[i, :, :, mid], cmap='jet', vmin=0, vmax=1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred_bin[i, :, :, mid], cmap='jet', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Prediction (Dice-based)")
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{loss_name}_{model_name}_epoch{epoch:03d}.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  [Vis] Saved prediction plot: {save_path}")


def plot_and_save(t1ce_slice, gt_mask, pred_mask, save_path, patient_id, slice_idx):
    """
    Generates a clean grid comparing ground truths against model predictions.
    Added from dummy.py for unified tracking of evaluation steps.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"Patient: {patient_id} (Axial Slice: {slice_idx})", fontsize=16, fontweight='bold')
    
    channels = ["Whole Tumor (WT)", "Tumor Core (TC)", "Enhancing Tumor (ET)"]
    colors = ['Reds', 'Oranges', 'Purples']
    
    axes[0, 0].imshow(t1ce_slice, cmap='gray')
    axes[0, 0].set_title("Base T1ce MRI")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(t1ce_slice, cmap='gray')
    axes[1, 0].set_title("Base T1ce MRI")
    axes[1, 0].axis('off')
    
    for i in range(3):
        axes[0, i+1].imshow(t1ce_slice, cmap='gray')
        axes[0, i+1].imshow(gt_mask[i], cmap=colors[i], alpha=0.5 if np.sum(gt_mask[i]) > 0 else 0)
        axes[0, i+1].set_title(f"GT: {channels[i]}")
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(t1ce_slice, cmap='gray')
        axes[1, i+1].imshow(pred_mask[i], cmap=colors[i], alpha=0.5 if np.sum(pred_mask[i]) > 0 else 0)
        axes[1, i+1].set_title(f"Pred: {channels[i]}")
        axes[1, i+1].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


if __name__ == "__main__":
    # Change this path to your actual data location
    DATA_PATH = "/data/MICCAI_BraTS2020_TrainingData/" 
    TEST_PATIENT = "BraTS20_Training_001"
    
    # Note: If using show_slices here, ensure it's defined or imported.
    # show_slices(TEST_PATIENT, DATA_PATH, slice_idx=80)