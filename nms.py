"""
Non-Maximum Suppression (NMS) for gradient magnitude thinning.
"""

import numpy as np
import cv2


def nms_gradient_magnitude(gx, gy, mag):
    """
    Non-Maximum Suppression (NMS) on gradient magnitude (Canny-style thinning).
    
    Args:
        gx, gy: Gradient components (float32)
        mag: Gradient magnitude (float32)
    
    Returns:
        mag_nms: Thinned magnitude (same shape, 0 where not local max)
    """
    h, w = mag.shape
    mag_nms = np.zeros_like(mag)
    
    # Compute angle in degrees [0, 180)
    angle = np.arctan2(gy, gx) * 180.0 / np.pi
    angle = np.mod(angle, 180.0)
    
    # Quantize to 4 directions: 0, 45, 90, 135
    # Use intervals: [0, 22.5) and [157.5, 180) -> 0°, [22.5, 67.5) -> 45°, [67.5, 112.5) -> 90°, [112.5, 157.5) -> 135°
    dir_0 = ((angle >= 0) & (angle < 22.5)) | ((angle >= 157.5) & (angle < 180))
    dir_45 = (angle >= 22.5) & (angle < 67.5)
    dir_90 = (angle >= 67.5) & (angle < 112.5)
    dir_135 = (angle >= 112.5) & (angle < 157.5)
    
    # For each direction, compare with neighbors along gradient direction
    # Direction 0°: compare with left/right neighbors
    mask_0 = dir_0[1:-1, 1:-1] & (mag[1:-1, 1:-1] >= mag[1:-1, :-2]) & (mag[1:-1, 1:-1] >= mag[1:-1, 2:])
    mag_nms[1:-1, 1:-1] = np.where(mask_0, mag[1:-1, 1:-1], mag_nms[1:-1, 1:-1])
    
    # Direction 45°: compare with diagonal neighbors (top-left / bottom-right)
    mask_45 = dir_45[1:-1, 1:-1] & (mag[1:-1, 1:-1] >= mag[:-2, :-2]) & (mag[1:-1, 1:-1] >= mag[2:, 2:])
    mag_nms[1:-1, 1:-1] = np.where(mask_45, mag[1:-1, 1:-1], mag_nms[1:-1, 1:-1])
    
    # Direction 90°: compare with top/bottom neighbors
    mask_90 = dir_90[1:-1, 1:-1] & (mag[1:-1, 1:-1] >= mag[:-2, 1:-1]) & (mag[1:-1, 1:-1] >= mag[2:, 1:-1])
    mag_nms[1:-1, 1:-1] = np.where(mask_90, mag[1:-1, 1:-1], mag_nms[1:-1, 1:-1])
    
    # Direction 135°: compare with diagonal neighbors (top-right / bottom-left)
    mask_135 = dir_135[1:-1, 1:-1] & (mag[1:-1, 1:-1] >= mag[:-2, 2:]) & (mag[1:-1, 1:-1] >= mag[2:, :-2])
    mag_nms[1:-1, 1:-1] = np.where(mask_135, mag[1:-1, 1:-1], mag_nms[1:-1, 1:-1])
    
    return mag_nms


def hysteresis_from_masks(weak_mask: np.ndarray, strong_mask: np.ndarray) -> np.ndarray:
    """
    Hysteresis thresholding: keep weak edges that are 8-connected to strong edges.
    
    Args:
        weak_mask: Boolean array (H, W) of weak edge candidates
        strong_mask: Boolean array (H, W) of strong edges
    
    Returns:
        edge_mask: Boolean array (H, W) including all weak components 8-connected to strong edges
    """
    # Use cv2.connectedComponents on weak_mask with connectivity=8
    weak_uint8 = weak_mask.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(weak_uint8, connectivity=8)
    
    # Find labels that occur in strong_mask
    strong_labels = np.unique(labels[strong_mask])
    # Exclude label 0 (background)
    strong_labels = strong_labels[strong_labels > 0]
    
    # Build edge_mask: include all weak components that are connected to strong edges
    edge_mask = np.isin(labels, strong_labels)
    
    return edge_mask


def hysteresis_on_mag(mag_nms, band_mask, align_mask, high_percentile, low_frac):
    """
    Apply hysteresis thresholding on NMS magnitude within a region.
    
    Args:
        mag_nms: NMS-thinned gradient magnitude (float32)
        band_mask: Boolean mask for radius band (H, W)
        align_mask: Boolean mask for alignment filter (H, W)
        high_percentile: Percentile for strong threshold (e.g., 85.0)
        low_frac: Fraction of high threshold for weak threshold (e.g., 0.50)
    
    Returns:
        edge_mask: Final edge mask after hysteresis (bool array)
        strong_mask: Strong edge mask (bool array)
        weak_mask: Weak edge mask (bool array)
        thresholds: Tuple (t_high, t_low) as floats
    """
    # Region of interest: intersection of band and alignment
    region = band_mask & align_mask
    
    # Compute thresholds from NMS magnitude in region
    mag_nms_in_region = mag_nms[region & (mag_nms > 0)]
    
    if len(mag_nms_in_region) == 0:
        # No valid points in region
        h, w = mag_nms.shape
        empty_mask = np.zeros((h, w), dtype=bool)
        return empty_mask, empty_mask, empty_mask, (0.0, 0.0)
    
    t_high = np.percentile(mag_nms_in_region, high_percentile)
    t_low = low_frac * t_high
    
    # Build strong and weak masks
    strong_mask = region & (mag_nms >= t_high)
    weak_mask = region & (mag_nms >= t_low) & (mag_nms < t_high)
    
    # Apply hysteresis
    edge_mask = hysteresis_from_masks(weak_mask, strong_mask)
    # Include strong edges in final mask (important: strong edges are always kept)
    edge_mask = edge_mask | strong_mask
    
    # Debug: ensure we have some edges
    n_strong = np.sum(strong_mask)
    n_weak = np.sum(weak_mask)
    n_edge = np.sum(edge_mask)
    
    return edge_mask, strong_mask, weak_mask, (float(t_high), float(t_low))

