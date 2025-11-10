# src/test.py
import os
import argparse
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from utils import load_config, ensure_dir
from discovery import build_train_df
from postprocess import conservative_postprocess
from dataset import MPISPECTDataset
from model import UNet3D

def visualize_and_save(vol_np, mask_true_np, mask_pred_np, mask_pp_np,
                       slice_idx, out_png):
    plt.figure(figsize=(16, 4))
    titles = ["Volume", "Ground truth", "Predicted", "Post-processed"]
    imgs   = [vol_np, mask_true_np, mask_pred_np, mask_pp_np]
    for i, (img, title) in enumerate(zip(imgs, titles), start=1):
        plt.subplot(1, 4, i)
        plt.title(f"{title} (slice {slice_idx})")
        plt.imshow(img[:, :, slice_idx], cmap="gray")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight", pad_inches=0)
    plt.close()

def save_mask_nifti(mask_np, ref_nifti_path, out_path):
    ref = nib.load(ref_nifti_path)
    img = nib.Nifti1Image(mask_np.astype(np.uint8), affine=ref.affine, header=ref.header)
    nib.save(img, out_path)

def parse_args():
    p = argparse.ArgumentParser(description="Test UNet3D on MPI/SPECT volumes")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--only_masked", action="store_true")
    p.add_argument("--recursive",  action="store_true")

    p.add_argument("--vols_dir",  type=str, default=None)
    p.add_argument("--masks_dir", type=str, default=None)
    p.add_argument("--vol_suffix",  type=str, default="")
    p.add_argument("--mask_suffix", type=str, default="_mask")
    p.add_argument("--allow_mask_as_volume", action="store_true", help="Use *_mask as volume if real volume is missing (debug)")

    p.add_argument("--dry_list", action="store_true")

    p.add_argument("--threshold",  type=float, default=0.5)
    p.add_argument("--n_visualize", type=int, default=5)
    p.add_argument("--out_masks", type=str, default="outputs/masks_test")
    p.add_argument("--out_viz",   type=str, default="outputs/viz_test")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    _ = load_config(args.config)

    data_root = args.data_root
    default_nifti = os.path.join(data_root, "NIfTI")

    vols_dir  = args.vols_dir  if args.vols_dir  else default_nifti
    masks_dir = args.masks_dir if args.masks_dir else default_nifti

    print("Data root      :", data_root)
    print("Volumes dir    :", vols_dir)
    print("Masks dir      :", masks_dir)
    print("vol_suffix     :", args.vol_suffix or "<none>")
    print("mask_suffix    :", args.mask_suffix or "<none>")
    print("allow_mask_as_volume:", args.allow_mask_as_volume)
    print("Outputs        :", args.out_masks, "|", args.out_viz)

    ensure_dir(args.out_masks)
    ensure_dir(args.out_viz)

    df = build_train_df(
        nifti_dir=None,
        only_masked=args.only_masked,
        recursive=args.recursive,
        vols_dir=vols_dir,
        masks_dir=masks_dir,
        vol_suffix=args.vol_suffix,
        mask_suffix=args.mask_suffix,
        allow_mask_as_volume=args.allow_mask_as_volume,
        dry_list=args.dry_list
    )

    if args.dry_list:
        print("Dry list mode finished.")
        raise SystemExit(0)

    if df.empty:
        print("❌ No data found to test on.")
        print("Hints:")
        print(" - If your volumes are missing, either provide them or use --allow_mask_as_volume (debug)")
        print(" - If your volumes end with a suffix (e.g., _img), add --vol_suffix _img")
        print(" - If your masks use a different suffix, set --mask_suffix accordingly")
        raise SystemExit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = UNet3D().to(device)
    if not os.path.isfile(args.weights):
        print(f"❌ Weights file not found: {args.weights}")
        raise SystemExit(1)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()
    print("Loaded weights:", args.weights)

    test_ds = MPISPECTDataset(df, target_shape=(64, 64, 56))

    n_total = len(test_ds)
    n_vis   = min(args.n_visualize, n_total)

    with torch.no_grad():
        for i in range(n_total):
            vol_tensor, mask_tensor = test_ds[i]
            vol_gpu = vol_tensor.unsqueeze(0).to(device)
            out     = model(vol_gpu)
            pred    = torch.sigmoid(out).cpu().numpy()[0, 0, ...]
            mask_pred = (pred > args.threshold).astype(np.uint8)

            mask_pp = conservative_postprocess(
                mask_pred,
                min_component_voxels=30,
                keep_largest_only=True,
                hole_fill_max_size=200,
                closing_radius=1,
                median_filter_size=3,
                smoothing_sigma=0.3,
                threshold=args.threshold,
                ring_compactness_min=0.6
            )

            ref_path  = df.loc[i, "nifti_path"]
            base_name = os.path.basename(ref_path)
            out_nii   = os.path.join(
                args.out_masks,
                base_name.replace(".nii.gz", "_pred_pp.nii.gz").replace(".nii", "_pred_pp.nii")
            )
            save_mask_nifti(mask_pp, ref_path, out_nii)

            if i < n_vis:
                vol_np    = vol_tensor.numpy().squeeze()
                gt_np     = mask_tensor.numpy().squeeze()
                slice_idx = vol_np.shape[2] // 2 if vol_np.ndim == 3 else 0
                out_png   = os.path.join(args.out_viz, base_name + f"_slice{slice_idx}.png")
                visualize_and_save(vol_np, gt_np, mask_pred, mask_pp, slice_idx, out_png)

            print(f"[{i+1}/{n_total}] Saved: {out_nii}")

    print("✅ Testing done.")
