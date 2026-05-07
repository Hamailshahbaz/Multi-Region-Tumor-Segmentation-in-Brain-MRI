import os
import torch
import numpy as np
import nibabel as nib
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BraTSDataset25D(Dataset):
    def __init__(self, patient_list, root_dir):
        self.patient_list = patient_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.patient_list)

    def _load_nifti(self, patient_id, modality):
        patient_path = os.path.join(self.root_dir, patient_id)
        
        # Standard naming for BraTS2020
        if modality == "seg":
            path = os.path.join(patient_path, f"{patient_id}_seg.nii")
        else:
            path = os.path.join(patient_path, f"{patient_id}_{modality}.nii")
            
        # Check for .gz extension if .nii is missing
        if not os.path.exists(path):
            if os.path.exists(path + ".gz"):
                path += ".gz"
            else:
                # Fallback search
                files = glob.glob(os.path.join(patient_path, f"*{modality}*.nii*"))
                if files: path = files[0]
                else: raise FileNotFoundError(f"Missing {modality} in {patient_path}")
        
        return nib.load(path).get_fdata()

    def __getitem__(self, idx):
        try:
            p_id = self.patient_list[idx]

            # 1. Load 3D Volumes (H, W, D)
            flair = self._load_nifti(p_id, "flair") 
            t1ce  = self._load_nifti(p_id, "t1ce")
            t1    = self._load_nifti(p_id, "t1")
            t2    = self._load_nifti(p_id, "t2")
            mask_vol = self._load_nifti(p_id, "seg")

            # 2. 2.5D Slicing: Pick center slice 75 and neighbors 74, 76
            z = 75 

            # Stack 3 slices for each of the 4 modalities (Total 12 channels)
            image_slices = []
            for mod in [flair, t1ce, t1, t2]:
                image_slices.append(mod[:, :, z-1])
                image_slices.append(mod[:, :, z])
                image_slices.append(mod[:, :, z+1])

            # Shape: (12, H, W)
            image = np.stack(image_slices, axis=0) 

            # 3. Target Mask (Center slice only)
            target_mask = mask_vol[:, :, z]
            wt = (target_mask > 0).astype(np.float32)
            tc = np.logical_or(target_mask == 1, target_mask == 4).astype(np.float32)
            et = (target_mask == 4).astype(np.float32)
            combined_mask = np.stack([wt, tc, et], axis=0) # (3, H, W)

            # 4. Normalization
            for i in range(image.shape[0]):
                m, s = image[i].mean(), image[i].std()
                if s > 0: image[i] = (image[i] - m) / s

            return torch.from_numpy(image).float(), torch.from_numpy(combined_mask).float()

        except Exception as e:
            print(f"⚠️ Skipping {self.patient_list[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self.patient_list))

def get_dataloader(data_dir, batch_size=16):
    # Filter only patient folders
    all_patients = sorted([f for f in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, f)) and f.startswith('BraTS20')])

    train_pts, val_pts = train_test_split(all_patients, test_size=0.2, random_state=42)

    # Use the 2.5D class
    train_ds = BraTSDataset25D(train_pts, data_dir)
    val_ds   = BraTSDataset25D(val_pts, data_dir)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))
