#src/data_loader.py
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class BraTSDataset3D(Dataset):
    def __init__(self, patient_list, root_dir, crop_size=(128, 128, 128)):
        self.patient_list = patient_list
        self.root_dir = root_dir
        self.crop_size = crop_size

    def __len__(self):
        return len(self.patient_list)

    def _load_nifti(self, patient_id, modality):
        patient_path = os.path.join(self.root_dir, patient_id)
        path = os.path.join(patient_path, f"{patient_id}_{modality}.nii")

        if not os.path.exists(path):
            if os.path.exists(path + ".gz"): path += ".gz"
            else: raise FileNotFoundError(f"Missing {modality} in {patient_path}")

        return nib.load(path).get_fdata()

    def _center_crop(self, data, target_shape):
        """Crops the center of a 3D volume."""
        h, w, d = data.shape
        th, tw, td = target_shape

        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        start_d = (d - td) // 2

        return data[start_h:start_h+th, start_w:start_w+tw, start_d:start_d+td]

    def __getitem__(self, idx):
        try:
            p_id = self.patient_list[idx]

            # 1. Load 3D Volumes
            modalities = ["flair", "t1ce", "t1", "t2"]
            vols = []
            for mod in modalities:
                data = self._load_nifti(p_id, mod)
                data = self._center_crop(data, self.crop_size)
                # Z-Score Normalization per modality volume
                m, s = data.mean(), data.std()
                if s > 0: data = (data - m) / s
                vols.append(data)

            # Shape: (4, 128, 128, 128) -> (Channels, H, W, D)
            image = np.stack(vols, axis=0)

            # 2. Load and process Mask
            mask_vol = self._load_nifti(p_id, "seg")
            mask_vol = self._center_crop(mask_vol, self.crop_size)

            # BraTS Channels: WT, TC, ET
            wt = (mask_vol > 0).astype(np.float32)
            tc = np.logical_or(mask_vol == 1, mask_vol == 4).astype(np.float32)
            et = (mask_vol == 4).astype(np.float32)

            # Shape: (3, 128, 128, 128)
            combined_mask = np.stack([wt, tc, et], axis=0)

            return torch.from_numpy(image).float(), torch.from_numpy(combined_mask).float()

        except Exception as e:
            print(f"Skipping {self.patient_list[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self.patient_list))

def get_dataloader_3d(data_dir, batch_size=2):
    all_patients = sorted([f for f in os.listdir(data_dir)
                          if os.path.isdir(os.path.join(data_dir, f)) and f.startswith('BraTS20')])

    train_pts, val_pts = train_test_split(all_patients, test_size=0.2, random_state=42)

    # Note: Reduced batch size for 3D
    train_ds = BraTSDataset3D(train_pts, data_dir)
    val_ds   = BraTSDataset3D(val_pts, data_dir)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))