# src/train.py

import os
import argparse
from glob import glob
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import MPISPECTDataset
from model import UNet3D


def strip_ext(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return os.path.splitext(name)[0]


def find_volume_path(nifti_dir: str, base: str):
    cand1 = os.path.join(nifti_dir, f"{base}.nii.gz")
    cand2 = os.path.join(nifti_dir, f"{base}.nii")
    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    return None


def find_mask_path(nifti_dir: str, base: str):
    cand1 = os.path.join(nifti_dir, f"{base}_mask.nii.gz")
    cand2 = os.path.join(nifti_dir, f"{base}_mask.nii")
    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    return None


def build_train_df_from_union(nifti_dir: str, allow_mask_as_volume: bool = True) -> pd.DataFrame:
    all_paths = sorted(glob(os.path.join(nifti_dir, "*.nii.gz")) + glob(os.path.join(nifti_dir, "*.nii")))
    print(f"Found NIfTI files: {len(all_paths)}")
    if all_paths:
        print("Some example filenames:", all_paths[:5])

    base_ids = set()
    for fp in all_paths:
        base = strip_ext(os.path.basename(fp))
        if base.endswith("_mask"):
            base_ids.add(base[:-5])
        else:
            base_ids.add(base)

    print(f"Unique base IDs discovered (count): {len(base_ids)}")

    rows = []
    missing_vols = []
    used_mask_as_vol = []

    for base in sorted(base_ids):
        vol = find_volume_path(nifti_dir, base)
        msk = find_mask_path(nifti_dir, base)

        if vol is None:
            if allow_mask_as_volume and msk is not None:
                # fallback: use mask as proxy volume to avoid empty dataset
                vol = msk
                used_mask_as_vol.append(base)
            else:
                missing_vols.append(base)
                continue

        rows.append({"file_base": base, "nifti_path": vol, "mask_path": msk})

    if missing_vols:
        print(f"⚠️ {len(missing_vols)} base IDs have NO usable volume (and no fallback used). Examples:")
        print(missing_vols[:10])

    if used_mask_as_vol:
        print(f"⚠️ Using mask as proxy volume for {len(used_mask_as_vol)} samples (temporary fallback). Examples:")
        print(used_mask_as_vol[:10])

    df = pd.DataFrame(rows)
    return df


def train_model(train_df: pd.DataFrame,
                target_shape=(64, 64, 56),
                epochs=30,
                batch_size=2,
                lr=1e-4,
                device=None,
                output_weights="models/unet3d_weights.pth"):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print("Using device:", device)

    model = UNet3D().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = MPISPECTDataset(train_df, target_shape=target_shape)
    ds_len = len(train_ds)
    print("Train dataset size (after filtering):", ds_len)
    if ds_len == 0:
        raise RuntimeError("Train dataset is empty! Verify volume/mask files and naming.")

    # اگر با num_workers مشکل دارید مقدار را 0 کنید
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for vols, masks in train_loader:
            vols = vols.to(device)
            masks = masks.to(device).float()

            optimizer.zero_grad()
            logits = model(vols)       # (B,1,D,H,W)
            logits = logits.squeeze(1) # (B,D,H,W)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * vols.size(0)

        epoch_loss = running_loss / ds_len
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f}")

    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    torch.save(model.state_dict(), output_weights)
    print("Saved model weights to:", output_weights)


def parse_args():
    ap = argparse.ArgumentParser(description="Train UNet3D on MPI/SPECT dataset (NIfTI + optional masks)")
    ap.add_argument("--data_root", type=str, required=True, help="Root of MPI_SPECT (contains NIfTI/)")
    ap.add_argument("--output_weights", type=str, default="models/unet3d_weights.pth", help="Path to save weights")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no_mask_fallback", action="store_true",
                    help="Disable using mask as proxy volume when volume is missing.")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_root = args.data_root
    nifti_dir = os.path.join(data_root, "NIfTI")

    print("Data root:", data_root)
    print("Looking for NIfTI files in:", nifti_dir)

    train_df = build_train_df_from_union(
        nifti_dir,
        allow_mask_as_volume=(not args.no_mask_fallback)
    )

    if train_df.empty:
        print("❌ Constructed train_df is empty.")
        print("Hints:")
        print(" - Ensure your volume files are named like <base>.nii.gz (without _mask).")
        print(" - Masks should be <base>_mask.nii(.gz) in the same NIfTI folder.")
        raise SystemExit(1)

    print("Sample first 10 entries of constructed train_df:")
    print(train_df.head(10))
    has_mask = train_df["mask_path"].notna().sum()
    print(f"Total usable volumes: {len(train_df)} | with masks: {has_mask} | without masks: {len(train_df)-has_mask}")

    train_model(
        train_df,
        target_shape=(64, 64, 56),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=None,
        output_weights=args.output_weights,
    )
