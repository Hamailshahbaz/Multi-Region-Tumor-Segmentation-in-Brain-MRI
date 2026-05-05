import os
import torch
import matplotlib.pyplot as plt
# Import the class you just wrote
from data_loader import BraTSDataset 

def show_slices(patient_id, data_dir, slice_idx=75):
    # Initialize dataset for just one patient
    dataset = BraTSDataset(patient_list=[patient_id], root_dir=data_dir)
    
    # Get the data (This calls  __getitem__ function)
    image, mask = dataset[0] 
    
    # Convert tensors back to numpy for plotting
    img = image.numpy()
    msk = mask.numpy()
    
    # Create a grid: 2 rows (Input vs Output), 4 columns
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    
    # 1. Plot Inputs (Modalities)
    modalities = ['FLAIR', 'T1ce', 'T1', 'T2']
    for i in range(4):
        axes[0, i].imshow(img[i, slice_idx, :, :], cmap='gray')
        axes[0, i].set_title(f"Input: {modalities[i]}")
        axes[0, i].axis('off')

    # 2. Plot Outputs (Nested Masks)
    # We use 'jet' or 'Reds' to make the mask stand out
    targets = ['Whole Tumor (WT)', 'Tumor Core (TC)', 'Enhancing Tumor (ET)']
    for i in range(3):
        axes[1, i].imshow(msk[i, slice_idx, :, :], cmap='Reds')
        axes[1, i].set_title(f"Target Mask: {targets[i]}")
        axes[1, i].axis('off')

    # 3. Final Comparison (Overlay)
    axes[1, 3].imshow(img[1, slice_idx, :, :], cmap='gray') # T1ce background
    axes[1, 3].imshow(msk[2, slice_idx, :, :], cmap='Reds', alpha=0.5) # ET Overlay
    axes[1, 3].set_title("Check: ET inside T1ce")
    axes[1, 3].axis('off')

    plt.suptitle(f"Visualization for {patient_id} at Slice {slice_idx}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Change this path to your actual data location
    DATA_PATH = "/home/hamail/Multi-Region-Tumor-Segmentation-in-Brain-MRI/data/MICCAI_BraTS2020_TrainingData/" 
    TEST_PATIENT = "BraTS20_Training_001"
    
    show_slices(TEST_PATIENT, DATA_PATH, slice_idx=80)