# src/inspect_data.py
import os
import argparse
from discovery import build_train_df

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--vols_dir", default=None)
    p.add_argument("--masks_dir", default=None)
    p.add_argument("--vol_suffix", default="")
    p.add_argument("--mask_suffix", default="_mask")
    p.add_argument("--recursive", action="store_true")
    args = p.parse_args()

    default_nifti = os.path.join(args.data_root, "NIfTI")
    vols_dir  = args.vols_dir  if args.vols_dir  else default_nifti
    masks_dir = args.masks_dir if args.masks_dir else default_nifti

    df = build_train_df(
        nifti_dir=None,
        only_masked=False,
        recursive=args.recursive,
        vols_dir=vols_dir,
        masks_dir=masks_dir,
        vol_suffix=args.vol_suffix,
        mask_suffix=args.mask_suffix,
        dry_list=True
    )
    print("If you disable dry_list, rows would be:", len(df))

