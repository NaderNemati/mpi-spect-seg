# src/inference.py
import os
import glob
import torch
import nibabel as nib
import numpy as np
from model import UNet3D
from postprocess import advanced_postprocess
from dataset import preprocess_volume

def load_model(weights_path, device):
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def find_new_files(input_dir, output_dir, ext=".nii.gz"):
    input_files  = glob.glob(os.path.join(input_dir, f"*{ext}"))
    base_inputs  = { os.path.basename(f) for f in input_files }
    output_files = glob.glob(os.path.join(output_dir, f"*{ext}"))
    base_outputs = { os.path.basename(f) for f in output_files }
    to_process   = [f for f in input_files if os.path.basename(f) not in base_outputs]
    return to_process

def run_inference(input_dir, output_dir, weights_path, device,
                  target_shape=(64,64,56), threshold=0.5, **pp_kwargs):
    os.makedirs(output_dir, exist_ok=True)
    model    = load_model(weights_path, device)
    new_files = find_new_files(input_dir, output_dir)
    print(f"Found {len(new_files)} new files to process.")

    for idx, nifti_path in enumerate(new_files):
        print(f"[{idx+1}/{len(new_files)}] Processing: {nifti_path}")
        vol = nib.load(nifti_path).get_fdata()
        vol = preprocess_volume(vol, target_shape=target_shape)
        vol_tensor = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(vol_tensor)
        out_np = out.cpu().numpy().squeeze()
        mask_pred = (out_np > threshold).astype(np.uint8)
        mask_pp   = advanced_postprocess(mask_pred, **pp_kwargs)

        ref  = nib.load(nifti_path)
        out_name = os.path.basename(nifti_path).replace(".nii.gz","_pred_pp.nii.gz")
        out_path = os.path.join(output_dir, out_name)
        nib.save(nib.Nifti1Image(mask_pp.astype(np.uint8), affine=ref.affine, header=ref.header), out_path)
        print(f"Saved output: {out_path}")

    print("Done.")

if __name__ == "__main__":
    data_root     = "/home/nader/Desktop/Diognastic/MPI_SPECT"
    input_dir     = os.path.join(data_root, "NIfTI")
    output_dir    = os.path.join(data_root, "output_masks")
    weights_path  = os.path.join(data_root, "models", "unet3d_weights.pth")
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_inference(input_dir, output_dir, weights_path, device,
                  threshold=0.5,
                  min_comp_voxels=30,
                  hole_size_threshold=200,
                  keep_largest_only=True,
                  closing_radius=1,
                  median_filter_size=3,
                  smoothing_sigma=0.3)

