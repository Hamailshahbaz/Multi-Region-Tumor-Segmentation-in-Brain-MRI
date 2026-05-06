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
    
    def pad_image(self, image, target_shape=(160, 240, 240)):
        c, d, h, w = image.shape
        
        # Use max(0, ...) to ensure we never have negative padding
        pad_d = max(0, target_shape[0] - d)
        pad_h = max(0, target_shape[1] - h)
        pad_w = max(0, target_shape[2] - w)

        padded_image = np.pad(image, 
                            ((0,0), (0, pad_d), (0, pad_h), (0, pad_w)), 
                            mode='constant')
        
        # If the image was larger than target_shape, crop it to match exactly
        return padded_image[:, :target_shape[0], :target_shape[1], :target_shape[2]]
    
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

        image = self.pad_image(image)
        combined_mask = self.pad_image(combined_mask)

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
    
    def pad_image(self, image, target_shape=(160, 240, 240)):
        c, d, h, w = image.shape
        
        # Use max(0, ...) to ensure we never have negative padding
        pad_d = max(0, target_shape[0] - d)
        pad_h = max(0, target_shape[1] - h)
        pad_w = max(0, target_shape[2] - w)

        padded_image = np.pad(image, 
                            ((0,0), (0, pad_d), (0, pad_h), (0, pad_w)), 
                            mode='constant')
        
        # If the image was larger than target_shape, crop it to match exactly
        return padded_image[:, :target_shape[0], :target_shape[1], :target_shape[2]]
    
    def __getitem__(self, idx):
        try:
            # Finds the corresponding patient ID from your list (e.g., BraTS20_Training_001)
            p_id = self.patient_list[idx] 
    
            # 1. Load Modalities
            # These highlight different tissue properties (edema, active core, etc.)
            flair = self._load_nifti(p_id, "flair")
            t1ce  = self._load_nifti(p_id, "t1ce")
            t1    = self._load_nifti(p_id, "t1")
            t2    = self._load_nifti(p_id, "t2")
    
            # 2. Stack to (4, D, H, W) 
            # Stacking allows the model to see all 4 "views" at once, similar to RGB channels
            image = np.stack([flair, t1ce, t1, t2], axis=0)
    
            # 3. Load Mask & Map to Nested Regions
            mask = self._load_nifti(p_id, "seg")
            
            # Whole Tumor (WT): Includes all tumor labels (1, 2, 4)
            wt = (mask > 0).astype(np.float32)
            # Tumor Core (TC): Includes necrotic and enhancing tumor (1, 4)
            tc = np.logical_or(mask == 1, mask == 4).astype(np.float32)
            # Enhancing Tumor (ET): Just the active enhancing core (Label 4)
            et = (mask == 4).astype(np.float32)
    
            combined_mask = np.stack([wt, tc, et], axis=0)
    
            # Apply Padding to ensure consistent dimensions (e.g., 160, 240, 240)
            image = self.pad_image(image)
            combined_mask = self.pad_image(combined_mask)
    
            # 4. Z-Score Normalization (Per Channel)
            image = self.normalize(image)
    
            # Convert to Tensors for PyTorch processing
            image = torch.from_numpy(image).float()
            combined_mask = torch.from_numpy(combined_mask).float()
    
            return image, combined_mask
    
        except FileNotFoundError:
            # Log the error and skip this patient index
            print(f"⚠️ Warning: Skipping patient {self.patient_list[idx]} due to missing or inconsistent naming.")
            
            # Safely increment index to try the next sample
            next_idx = (idx + 1) % len(self.patient_list)
            return self.__getitem__(next_idx)
        
    def normalize(self, image):
        """Standard Z-score normalization per modality[cite: 1]."""
        for i in range(image.shape[0]):
            mean = np.mean(image[i])
            std = np.std(image[i])
            if std > 0:
                image[i] = (image[i] - mean) / std
        return image

def get_dataloader(train_dir, val_dir, batch_size=1):
    """
    Directly creates DataLoaders from separate train and validation directories.
    """
    train_pts = sorted([f for f in os.listdir(train_dir) if f.startswith('BraTS20')])
    val_pts   = sorted([f for f in os.listdir(val_dir) if f.startswith('BraTS20')])

    # Initialize the Datasets with their specific root folders
    train_ds = BraTSDataset(train_pts, train_dir)
    val_ds   = BraTSDataset(val_pts, val_dir)

    # Create the Loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader