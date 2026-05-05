import os
import torch
import numpy as np
import nibabel as nib
# nibabel is a specialized library for reading NIfTI files (.nii), 
# as standard image loaders cannot process these 3D medical data formats.
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BraTSDataset(Dataset):
    def __init__(self, patient_list, root_dir, transform=None):
        """
        Args:
            patient_list (list): List of patient folder names (e.g., ['BraTS20_Training_001', ...])
            root_dir (str): Path to the TrainingData folder
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.patient_list = patient_list
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.patient_list)

    def _load_nifti(self, patient_id, modality): 
        # Modality refers to the specific MRI sequence type 
        # (FLAIR, T1, T1ce, or T2) that highlights different tissue properties.
        path = os.path.join(self.root_dir, patient_id, f"{patient_id}_{modality}.nii")
        return nib.load(path).get_fdata()

    def __getitem__(self, idx):
        #finds the corresponding patient ID from your list (e.g., BraTS20_Training_001)
        p_id = self.patient_list[idx] 

        # 1. Load Modalities
        flair = self._load_nifti(p_id, "flair")
        t1ce  = self._load_nifti(p_id, "t1ce")
        t1    = self._load_nifti(p_id, "t1")
        t2    = self._load_nifti(p_id, "t2")

        # 2. Stack to (4, D, H, W) 
        #PyTorch prefer (Channels, Depth, Height, Width) format
        # It loads 4 separate MRI files and stacks them into one 4-channel array.
        # why? 
        #A single MRI isn't enough. For example, FLAIR shows the edema, but T1ce shows the active tumor core.
        # By stacking them, the model sees all 4 "views" at once, 
        # similar to how a color photo has Red, Green, and Blue channels.
        image = np.stack([flair, t1ce, t1, t2], axis=0)

        # 3. Load Mask & Map to Nested Regions
        mask = self._load_nifti(p_id, "seg")
        
        # Whole Tumor (WT): Labels 1, 2, 4[cite: 1]
        wt = (mask > 0).astype(np.float32)
        # Tumor Core (TC): Labels 1, 4[cite: 1]
        tc = np.logical_or(mask == 1, mask == 4).astype(np.float32)
        # Enhancing Tumor (ET): Label 4[cite: 1]
        et = (mask == 4).astype(np.float32)

        combined_mask = np.stack([wt, tc, et], axis=0)

        # 4. Z-Score Normalization (Per Channel)[cite: 1]
        image = self.normalize(image)

        # Convert to Tensors
        image = torch.from_numpy(image).float()
        combined_mask = torch.from_numpy(combined_mask).float()

        return image, combined_mask

    def normalize(self, image):
        """Standard Z-score normalization per modality[cite: 1]."""
        for i in range(image.shape[0]):
            mean = np.mean(image[i])
            std = np.std(image[i])
            if std > 0:
                image[i] = (image[i] - mean) / std
        return image

def get_dataloader(data_dir, batch_size=1, train_val_test_split=(0.8, 0.1, 0.1)):
    """
    Modular function to split data and return DataLoaders.
    """
    all_patients = sorted([f for f in os.listdir(data_dir) if f.startswith('BraTS20')])
    
    # Split: Train (80%) and Temp (20%)[cite: 1]
    train_pts, temp_pts = train_test_split(all_patients, test_size=0.2, random_state=42)
    # Split Temp into Val (10%) and Test (10%)[cite: 1]
    val_pts, test_pts = train_test_split(temp_pts, test_size=0.5, random_state=42)

    train_ds = BraTSDataset(train_pts, data_dir)
    val_ds   = BraTSDataset(val_pts, data_dir)
    test_ds  = BraTSDataset(test_pts, data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader