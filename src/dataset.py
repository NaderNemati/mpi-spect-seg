# src/dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import scipy.ndimage as ndi

def preprocess_volume(vol, target_shape=(64,64,56)):
    zoom_factors = [t/s for t, s in zip(target_shape, vol.shape)]
    vol_resized  = ndi.zoom(vol, zoom_factors, order=1)
    vol_norm     = (vol_resized - vol_resized.min()) / (vol_resized.max() - vol_resized.min() + 1e-8)
    return vol_norm

class MPISPECTDataset(Dataset):
    def __init__(self, df, target_shape=(64,64,56)):
        self.df           = df.reset_index(drop=True)
        self.target_shape = target_shape

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample     = self.df.loc[idx]
        nifti_path = sample['nifti_path']
        mask_path  = sample.get('mask_path', None)

        if not os.path.isfile(nifti_path):
            raise FileNotFoundError(f"Volume file not found: {nifti_path}")

        vol = nib.load(nifti_path).get_fdata()
        vol = preprocess_volume(vol, target_shape=self.target_shape)
        vol_tensor = torch.from_numpy(vol).float().unsqueeze(0)  # shape (1, H, W, D)

        if mask_path is not None and os.path.isfile(mask_path):
            mask   = nib.load(mask_path).get_fdata()
            mask   = preprocess_volume(mask, target_shape=self.target_shape)
            mask_tensor = torch.from_numpy((mask > 0).astype(np.int64))
        else:
            # fallback: zero mask
            mask_tensor = torch.zeros_like(vol_tensor, dtype=torch.int64)

        return vol_tensor, mask_tensor
