# src/postprocess.py
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology

def _remove_small(mask_bool, min_size=30):
    return morphology.remove_small_objects(mask_bool, min_size=min_size)

def _keep_largest(mask_bool):
    lbl, num = ndi.label(mask_bool)
    if num == 0:
        return mask_bool
    sizes = ndi.sum(mask_bool, lbl, index=range(1, num + 1))
    largest = int(np.argmax(sizes)) + 1
    return (lbl == largest)

def _component_perimeter(comp_bool):
    dil = morphology.binary_dilation(comp_bool)
    return (dil ^ comp_bool).sum()

def _hole_stats(inv_mask, labels, h_idx):
    region = (labels == h_idx)
    area = int(region.sum())
    per = _component_perimeter(region)
    compact = (4.0 * np.pi * area) / (per ** 2 + 1e-8) if per > 0 else 0.0
    return area, compact, region

def conservative_postprocess(
    mask_pred: np.ndarray,
    min_component_voxels: int = 30,
    keep_largest_only: bool = True,
    hole_fill_max_size: int = 200,
    closing_radius: int = 1,
    median_filter_size: int = 3,
    smoothing_sigma: float = 0.3,
    threshold: float = 0.5,
    ring_compactness_min: float = 0.6
) -> np.ndarray:
    """
    Conservative behavior: do NOT fill the central ring-like hole; only fill small/round holes.
    """
    m = (mask_pred > 0).astype(bool)

    if min_component_voxels and min_component_voxels > 0:
        m = _remove_small(m, min_size=min_component_voxels)

    if keep_largest_only:
        m = _keep_largest(m)

    if not m.any():
        return m.astype(np.uint8)

    coords = np.array(np.where(m))
    zmin, ymin, xmin = coords[0].min(), coords[1].min(), coords[2].min()
    zmax, ymax, xmax = coords[0].max(), coords[1].max(), coords[2].max()

    m_roi = m[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1]
    inv_roi = ~m_roi
    inv_roi = inv_roi.copy()
    inv_roi[0, :, :]  = False
    inv_roi[-1, :, :] = False
    inv_roi[:, 0, :]  = False
    inv_roi[:, -1, :] = False
    inv_roi[:, :, 0]  = False
    inv_roi[:, :, -1] = False

    lbl_holes, num_holes = ndi.label(inv_roi)
    filled_roi = m_roi.copy()

    for h_idx in range(1, num_holes + 1):
        area, compact, region = _hole_stats(inv_roi, lbl_holes, h_idx)
        fill_it = False
        if area < hole_fill_max_size:
            fill_it = True
        elif compact >= ring_compactness_min:
            fill_it = True
        if fill_it:
            filled_roi[region] = True

    filled = m.copy()
    filled[zmin:zmax+1, ymin:ymax+1, xmin:xmax+1] = filled_roi

    selem = morphology.ball(closing_radius) if filled.ndim == 3 else morphology.disk(closing_radius)
    closed = ndi.binary_closing(filled, structure=selem, iterations=1)

    if median_filter_size and median_filter_size > 1:
        filtered = ndi.median_filter(closed.astype(np.uint8), size=median_filter_size).astype(bool)
    else:
        filtered = closed

    if smoothing_sigma and smoothing_sigma > 0:
        smooth = ndi.gaussian_filter(filtered.astype(np.float32), sigma=smoothing_sigma)
        out = (smooth > threshold)
    else:
        out = filtered

    return out.astype(np.uint8)
