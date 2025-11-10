
<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/mpi-spect-seg/blob/main/images/header.png" alt="Project Structure" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>



# 🧠 MPI-SPECT Segmentation Pipeline

This repository implements a complete deep learning pipeline for **3D segmentation of myocardial perfusion SPECT (MPI-SPECT)** scans.  
The framework is built using **PyTorch** and designed for modularity, enabling reproducible training, testing, and post-processing of volumetric medical images.

---

## 📘 Table of Contents
1. [Overview](#overview)
2. [Dataset Description](#dataset-description)
3. [Project Objective](#project-objective)
4. [Pipeline Overview](#pipeline-overview)
   - [1. Preprocessing](#1-preprocessing)
   - [2. Model Architecture](#2-model-architecture)
   - [3. Training Procedure](#3-training-procedure)
   - [4. Postprocessing](#4-postprocessing)
5. [Outputs](#outputs)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Configuration](#configuration)
9. [Repository Structure](#repository-structure)
10. [Future Work](#future-work)
11. [Citation](#citation)

---

## 🔍 Overview

This project focuses on developing a **3D U-Net–based segmentation framework** to automatically delineate cardiac structures from **Myocardial Perfusion Imaging (MPI)** SPECT scans.  
It includes all essential stages of the medical imaging pipeline — from data preparation to postprocessed mask generation — implemented in a **fully reproducible, modular, and configurable** way.

---

## 🧩 Dataset Description

The dataset used in this project is the **MPI-SPECT** dataset, which contains reconstructed nuclear medicine images of the heart acquired using SPECT scanners.  

Each subject includes:
- Volumetric **3D NIfTI** images (`.nii` or `.nii.gz`) representing the perfusion intensity distribution.
- Corresponding **segmentation masks** (`*_mask.nii.gz`) indicating the myocardial wall and other relevant cardiac structures.
- Optional metadata (e.g., demographics, acquisition parameters) stored in `demographics.csv`.

**Folder structure example:**

```python
MPI_SPECT/
├── NIfTI/
│ ├── 1.2.840....nii.gz
│ ├── 1.2.840...._mask.nii.gz
│ ├── ...
├── demographics.csv

```




> **Note:**  
> The naming convention is crucial — masks must follow the `<base>_mask.nii.gz` format.  
> If you only have mask files (for debugging), the pipeline supports `--allow_mask_as_volume` mode.

---

## 🎯 Project Objective

The main objective of this work is to create an **end-to-end automated segmentation pipeline** capable of:

1. Efficiently handling 3D nuclear imaging data (NIfTI format).  
2. Preprocessing, normalizing, and resizing scans to a uniform spatial dimension.  
3. Training a **3D U-Net** model for voxel-level cardiac structure segmentation.  
4. Applying robust **postprocessing techniques** to clean up segmentation artifacts.  
5. Producing high-quality volumetric masks and visual comparisons for validation.

This project is part of a broader initiative toward **fully automated, privacy-preserving cardiac imaging pipelines** for clinical and research use.

---

## ⚙️ Pipeline Overview

### 1. Preprocessing

All raw `.nii` or `.nii.gz` files are:
- Resampled and resized to a target shape of **(64 × 64 × 56)** voxels.  
- Intensity-normalized to a `[0, 1]` range for stable model convergence.  
- Matched by base ID between volumes and masks to create a training dataframe (`file_base`, `nifti_path`, `mask_path`).

Optional parameters:
- Recursive folder scanning  
- Automatic filtering of missing or invalid cases  
- Configurable normalization and shape settings (see `config/default.yaml`)

---

### 2. Model Architecture

The segmentation backbone is a **3D U-Net** implemented in PyTorch (`src/model.py`).

**Main features:**
- Two encoding and decoding paths with symmetric skip connections  
- 3D convolutional layers followed by batch normalization and ReLU activations  
- Bottleneck layer with doubled feature depth  
- Transposed convolutions for upsampling with `output_padding=1` to prevent shape mismatches  
- Final 1×1×1 convolution producing a binary mask output  

**Loss Function:** `BCEWithLogitsLoss`  
**Optimizer:** `Adam`  
**Learning Rate:** `1e-4`  

---

### 3. Training Procedure

Training is performed using `src/train.py`.  
The script automatically:
- Loads all valid pairs of volumes and masks  
- Creates PyTorch datasets and data loaders  
- Trains the model over multiple epochs  
- Computes running loss and saves weights to `models/unet3d_weights.pth`

Example command:
```bash
python src/train.py \
  --data_root /home/nader/Desktop/Diognastic/MPI_SPECT \
  --epochs 30 --batch_size 2 --lr 1e-4
```

### 4. Postprocessing

After inference, the predicted masks are cleaned and refined through a multi-step advanced postprocessing module (src/postprocess.py).

Steps include:

- Small object removal using skimage.morphology.remove_small_objects.

- Component filtering to keep the largest connected region.

- Selective hole filling, guided by component geometry and location.

- Morphological closing for contour smoothing.

- Median and Gaussian filtering for edge denoising.

- Optional Denoising Autoencoder (DAE)-based refinement (if weights available).

The result is a topologically consistent, smooth binary mask that preserves anatomical detail while removing spurious regions.


## 📊 Outputs

After testing and postprocessing, the pipeline generates:



| Output Type     | Location                    | Description                                                                              |
| --------------- | --------------------------- | ---------------------------------------------------------------------------------------- |
| Model weights   | `models/unet3d_weights.pth` | Trained 3D U-Net weights                                                                 |
| Predicted masks | `outputs/masks_test/`       | Postprocessed binary masks (NIfTI format)                                                |
| Visualization   | `outputs/viz_test/`         | PNG figures with 4-panel visualization (volume, ground truth, prediction, postprocessed) |
| Logs            | `outputs/logs/` (optional)  | Training metrics and configuration info                                                  |



## 🧭 Installation
### Clone the repository
```bash
git clone https://github.com/NaderNemati/MPI-SPECT-Segmentation.git
cd MPI-SPECT-Segmentation
```

### Create and activate a virtual environment 
```bash
python3 -m venv .venv
source .venv/bin/activate
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Usage
### Training
```bash
python src/train.py --data_root /path/to/MPI_SPECT
```

### Testing

```bash
python src/test.py \
  --config config/default.yaml \
  --data_root /path/to/MPI_SPECT \
  --weights models/unet3d_weights.pth \
  --only_masked --recursive
```

## ⚙️ Configuration

All parameters are stored in the YAML config file:

```bash
config/default.yaml

```

You can override any value from the command line:

```bash
python src/test.py --data_root ... --mask_suffix _mask --threshold 0.4
```

## 📁 Repository Structure

```python
.
├── config/
│   └── default.yaml
├── src/
│   ├── train.py
│   ├── test.py
│   ├── dataset.py
│   ├── model.py
│   ├── postprocess.py
│   ├── discovery.py
│   ├── inspect_data.py
│   └── utils.py
├── MPI_SPECT/
│   ├── NIfTI/
│   ├── demographics.csv
├── outputs/
│   ├── masks_test/
│   └── viz_test/
└── README.md
```

## 🔬 Future Work

- Integration of DAE refinement for mask denoising and topology correction.

- Experimentation with attention-based U-Net or transformer-based 3D architectures.

- Quantitative evaluation (Dice, IoU, Hausdorff metrics).

- Extension to multimodal fusion (SPECT + CT) and federated learning for privacy-preserving applications.

## 📚 Citation

If you use this repository or parts of it in your research, please cite it as:

```latex
@misc{nemati2025mpispectseg,
  author = {Nader Nemati},
  title  = {MPI-SPECT Segmentation Pipeline: A 3D U-Net Framework for Cardiac Image Segmentation},
  year   = {2025},
  howpublished = {\url{https://github.com/NaderNemati/MPI-SPECT-Segmentation}}
}
```

## 📦 Version: v1.0.0
## 🧠 Framework: PyTorch
## 📅 Last updated: November 2025


## Results

This section summarizes training progress, validation metrics, and qualitative visualizations produced by the notebook.

### Training Summary

- Setup (from notebook prints):

- Data root: /home/nader/Desktop/Diognastic/MPI_SPECT

- NIfTI folder exists: Yes

- Target shape: (64, 64, 56)

- Model: UNet3D (in_channels=1, out_channels=1, init_features=32)

- Optimizer: Adam (lr=1e-4)

- Loss: BCEWithLogitsLoss

-Epochs: 30 

| Epoch |    Loss    |
| ----: | :--------: |
|     1 |   0.5315   |
|     5 |   0.3699   |
|    10 |   0.3086   |
|    15 |   0.2593   |
|    20 |   0.2182   |
|    25 |   0.1834   |
|    30 | **0.1545** |



- Final Loss (Epoch 30): 0.1545
- Best Loss: 0.1545 (at epoch 30)
- Number of epochs logged: 30

**Note:** The training curve shows a consistent downward trend across 30 epochs, indicating stable learning with the chosen configuration.

## Evaluation

**Metric:** Dice coefficient (thresholded predictions)

**Mean Dice on test set:** 0.732

This score was computed by running inference on the test split and averaging the Dice coefficient over test volumes (post-binarization). It reflects the overall overlap quality between predicted masks and ground truth.

## Qualitative Results

Below are representative 4-panel visualizations exported by the notebook for several cases. Each panel shows:

1) Volume slice, 2) Ground truth mask, 3) Raw prediction, 4) Post-processed mask.

These figures were generated at mid-slices to give a consistent view across volumes. Post-processing included: removing small components, keeping the largest connected component, light morphological closing, median filtering, and light Gaussian smoothing before thresholding. (DAE step is optional and disabled unless weights are provided.)


## Notes on Post-processing

- remove_small_objects (min component size ~30 voxels)

- Largest CC selection (optional, enabled by default)

- Binary closing (3D spherical structuring element, radius ~1)

- Median filter (size ~3)

- Light Gaussian smoothing (σ ~0.3–0.5) prior to final thresholding

- Final threshold: 0.5

## Visualization of output samples

<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/mpi-spect-seg/blob/main/images/111.png" alt="Project Structure" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/mpi-spect-seg/blob/main/images/222.png" alt="Project Structure" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>

<div align="center">
  <table border=0 style="border: 0px solid #c6c6c6 !important; border-spacing: 0px; width: auto !important;">
    <tr>
      <td valign=top style="border: 0px solid #c6c6c6 !important; padding: 0px !important;">
        <div align=center valign=top>
          <img src="https://github.com/NaderNemati/mpi-spect-seg/blob/main/images/333.png" alt="Project Structure" style="margin: 0px !important; height: 400px !important;">
        </div>
      </td>
    </tr>
  </table>
</div>



# LICENSE

#### Copyright (c) 2025 Nader Nemati
#### Licensed under the MIT License. See the LICENSE file in the project root.



