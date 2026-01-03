"""
Non-Maximum Suppression (NMS) for gradient magnitude thinning.
"""

import numpy as np


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

