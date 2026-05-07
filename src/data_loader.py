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
        patient_path = os.path.join(self.root_dir, patient_id)
        
        if modality == "seg":
            # Look for any file ending in 'seg.nii' or 'Segm.nii' (case insensitive)
            mask_files = glob.glob(os.path.join(patient_path, "*[sS]eg*.nii"))
            if not mask_files:
                raise FileNotFoundError(f"No mask found in {patient_path}")
            path = mask_files[0]
        else:
            # For FLAIR, T1, etc., the standard naming usually holds
            path = os.path.join(patient_path, f"{patient_id}_{modality}.nii")
            
            # Fallback if the standard modality name is also weird
            if not os.path.exists(path):
                alt_files = glob.glob(os.path.join(patient_path, f"*{modality}*.nii"))
                if alt_files:
                    path = alt_files[0]
                else:
                    raise FileNotFoundError(f"Could not find {modality} in {patient_path}")
    
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
    
    class BraTSDataset25D(Dataset):
      def __init__(self, patient_list, root_dir, transform=None):
          self.patient_list = patient_list
          self.root_dir = root_dir
          self.transform = transform

      def __len__(self):
          return len(self.patient_list)

      def _load_nifti(self, patient_id, modality):
          path = os.path.join(self.root_dir, patient_id, f"{patient_id}_{modality}.nii")
          return nib.load(path).get_fdata()

      def __getitem__(self, idx):
          try:
              p_id = self.patient_list[idx]

              # 1. Load 3D Volumes
              flair = self._load_nifti(p_id, "flair") # (D, H, W)
              t1ce  = self._load_nifti(p_id, "t1ce")
              t1    = self._load_nifti(p_id, "t1")
              t2    = self._load_nifti(p_id, "t2")
              mask_vol = self._load_nifti(p_id, "seg")

              # 2. 2.5D Slicing Strategy
              # BraTS volumes are 155 slices deep. We pick a slice from the middle (20-130)
              # to avoid empty slices at the very top/bottom of the skull.
              z = np.random.randint(20, 130) 
              
              # Stack 3 slices (z-1, z, z+1) for each of the 4 modalities
              # Total input channels = 4 modalities * 3 slices = 12 channels
              image_slices = []
              for mod in [flair, t1ce, t1, t2]:
                  image_slices.append(mod[z-1, :, :])
                  image_slices.append(mod[z, :, :])
                  image_slices.append(mod[z+1, :, :])
              
              image = np.stack(image_slices, axis=0) # (12, H, W)
              
              # 3. Process Mask (Only for the center slice 'z')
              target_mask = mask_vol[z, :, :]
              wt = (target_mask > 0).astype(np.float32)
              tc = np.logical_or(target_mask == 1, target_mask == 4).astype(np.float32)
              et = (target_mask == 4).astype(np.float32)
              combined_mask = np.stack([wt, tc, et], axis=0) # (3, H, W)

              # 4. Normalization (Per 2D slice)
              for i in range(image.shape[0]):
                  mean, std = image[i].mean(), image[i].std()
                  if std > 0:
                      image[i] = (image[i] - mean) / std

              return torch.from_numpy(image).float(), torch.from_numpy(combined_mask).float()

          except Exception as e:
              print(f"Skipping {self.patient_list[idx]}: {e}")
              return self.__getitem__((idx + 1) % len(self.patient_list))

      def normalize(self, image):
          """Standard Z-score normalization per modality[cite: 1]."""
          for i in range(image.shape[0]):
              mean = np.mean(image[i])
              std = np.std(image[i])
              if std > 0:
                  image[i] = (image[i] - mean) / std
          return image


def get_dataloader(data_dir, batch_size=1):
    """
    Splits the labeled TrainingData into training and validation sets.
    """
    # Get all patient folders that actually contain labels
    all_patients = sorted([f for f in os.listdir(data_dir) if f.startswith('BraTS20')])
    
    # Split: 80% for training, 20% for your internal validation
    train_pts, val_pts = train_test_split(all_patients, test_size=0.2, random_state=42)

    # Both datasets point to the same data_dir, but use different patient lists
    train_ds = BraTSDataset(train_pts, data_dir)
    val_ds   = BraTSDataset(val_pts, data_dir)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
