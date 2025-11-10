# src/discovery.py
import os
from glob import glob
import pandas as pd
from typing import List, Dict

def _strip_ext(fname: str) -> str:
    if fname.endswith(".nii.gz"):
        return fname[:-7]
    if fname.endswith(".nii"):
        return fname[:-4]
    return os.path.splitext(fname)[0]

def _scan_dir(root: str, recursive: bool) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    pattern = "**/*.nii*" if recursive else "*.nii*"
    return sorted(glob(os.path.join(root, pattern), recursive=recursive))

def _index_by_base(files: List[str], suffix: str, skip_suffix: str | None = None) -> Dict[str, str]:
    idx = {}
    for f in files:
        base = _strip_ext(os.path.basename(f))
        if suffix:
            if not base.endswith(suffix):
                continue
            true_base = base[: -len(suffix)]
        else:
            # For volumes, optionally skip files that look like masks (if requested)
            if skip_suffix and base.endswith(skip_suffix):
                continue
            true_base = base
        if true_base not in idx or f.endswith(".nii.gz"):
            idx[true_base] = f
    return idx

def build_train_df(
    nifti_dir: str | None = None,
    only_masked: bool = False,
    recursive: bool = False,
    vols_dir: str | None = None,
    masks_dir: str | None = None,
    vol_suffix: str = "",
    mask_suffix: str = "_mask",
    allow_mask_as_volume: bool = False,
    dry_list: bool = False
) -> pd.DataFrame:

    vols_root  = vols_dir  if vols_dir  else nifti_dir
    masks_root = masks_dir if masks_dir else nifti_dir

    vol_files  = _scan_dir(vols_root,  recursive) if vols_root else []
    mask_files = _scan_dir(masks_root, recursive) if masks_root else []

    vols_by_base  = _index_by_base(vol_files,  vol_suffix, skip_suffix=mask_suffix or None)
    masks_by_base = _index_by_base(mask_files, mask_suffix, skip_suffix=None)

    if dry_list:
        print("---- DISCOVERY (dry) ----")
        print("vols_root :", vols_root)
        print("masks_root:", masks_root)
        print("vol_suffix:", vol_suffix or "<none>")
        print("mask_sfx  :", mask_suffix or "<none>")
        print(f"Found volumes: {len(vol_files)} | usable: {len(vols_by_base)}")
        print(f"Found masks  : {len(mask_files)} | usable: {len(masks_by_base)}")
        print("Sample volume bases:", list(vols_by_base.keys())[:5])
        print("Sample mask   bases:", list(masks_by_base.keys())[:5])
        print("-------------------------")

    bases = sorted(set(vols_by_base.keys()) | set(masks_by_base.keys()))
    rows = []
    missing_vol = []
    missing_mask = []

    for b in bases:
        v = vols_by_base.get(b)
        m = masks_by_base.get(b)

        if v is None and allow_mask_as_volume:
            # Fallback: if no volume exists, use the mask itself as the volume (as done in the notebook)
            v = masks_by_base.get(b)

        if only_masked and (m is None):
            continue
        if v is None:
            missing_vol.append(b)
            continue

        rows.append({"file_base": b, "nifti_path": v, "mask_path": m})
        if m is None:
            missing_mask.append(b)

    if missing_vol and not allow_mask_as_volume:
        print(f" {len(missing_vol)} bases have MASK but NO VOLUME (showing up to 10):")
        print(missing_vol[:10])
    if missing_mask:
        print(f" {len(missing_mask)} bases have VOLUME but NO MASK (still OK for inference).")
        print(missing_mask[:10])

    if not rows:
        print(" No usable cases found. Returned empty DataFrame.")
        return pd.DataFrame(columns=["file_base", "nifti_path", "mask_path"])

    return pd.DataFrame(rows).sort_values("file_base").reset_index(drop=True)
