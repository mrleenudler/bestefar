"""
Computer Vision modul for å detektere poengsoner på skyteskive.
Dette er en prototype i Python/OpenCV som senere skal implementeres i C++.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from datetime import datetime
import os


def get_config():
    """
    Returnerer en dictionary med alle konstanter og input-variabler.
    Endre verdiene her for å justere oppførselen til algoritmene.
    
    Returns:
        dict: Dictionary med alle konfigurasjonsparametere
    """
    return {
        # Filer og kataloger
        'visualization_dir': Path("Visualiseringer"),
        'log_file_kronologisk': "ytelse.txt",
        'debug_file': "debug_detect_outer_circle.txt",
        
        # Visualisering - Farger (BGR format)
        'color_green': (0, 255, 0),       # Grønn for contours/sirkler
        'color_red': (0, 0, 255),         # Rød for sentrum
        'color_blue': (255, 255, 0),        # Cyan? for tekst
        'color_magenta': (255, 0, 255),   # Magenta for hoved-sentrum
        
        # Visualisering - Tegneparametere
        'circle_thickness': 2,            # Tykkelse på sirkler
        'circle_center_size': 5,           # Størrelse på sentrum-markør
        
        # Visualisering - Tekst
        'font': cv2.FONT_HERSHEY_SIMPLEX,
        'font_scale_medium': 0.7,         # Font-størrelse for radius-tekst
        'font_thickness': 2,              # Font-tykkelse
        
        # Matplotlib
        'figure_size': (10, 10),          # Størrelse på matplotlib-figurer
        
        # Test-bilde (i main)
        # 'test_image_path': Path("Elektronisk skive.jpg"),
        'test_image_path': Path("Real 1.jpg"),
        
        # Outer circle detection - Gradient-normal center voting
        'outer_circle_max_side': 1200,              # Maks side i px for downscaling
        'outer_circle_blur_sigma': 2.0,             # Gaussian blur sigma
        'outer_circle_mag_floor': 1e-3,             # Minimum gradient magnitude
        'outer_circle_mag_percentile': 60.0,        # Percentile for edge threshold
        'outer_circle_filter_angle_deg': 2.0,       # Filter gradients within ±N degrees of horizontal/vertical
        'outer_circle_max_edge_points': 12000,      # Maks antall edge points
        'outer_circle_center_win': 5,               # Halvbredde for subpixel center refinement (win=5 => 11×11 vindu)
        'center_refine_enable': True,                # Enable center refinement with hillclimbing
        'center_refine_r_smooth_sigma': 0.8,          # Smoothing sigma for center refine histogram (0.6-1.0)
        'center_refine_eps': 1e-9,                    # Epsilon for sqrt transform
        'center_refine_max_steps': 80,                # Maximum hillclimbing steps
        'center_refine_neighborhood': 8,              # Neighborhood size (8 neighbors)
        'center_refine_step_px': 1,                   # Step size in pixels (downscaled)
        'center_refine_max_radius_px': 12,            # Stop if moved too far from start
        'center_refine_use_m4': False,                 # Use m4 moment in score (optional)
        'center_refine_m4_lambda': 0.05,              # Weight for m4 moment if enabled
        'outermost_ring_refine_enable': False,         # Enable radial variance refinement on outermost ring
        'outermost_ring_mag_percentile': 75.0,         # Percentile for magnitude threshold in outermost ring
        'outermost_ring_refine_max_steps': 60,         # Max steps for radial variance refinement
        'outermost_ring_refine_step_px': 1,            # Step size for refinement
        'outermost_ring_refine_max_radius_px': 12,     # Max radius from start for refinement
        'manual_center_enable': True,                 # Enable manual center for comparison
        'manual_center_xy': None,                    # Manual center (cx, cy) in downscaled coords (None to disable)
        'outer_circle_center_pairs': 40000,          # Antall random par for intersection-of-normals voting
        'outer_circle_parallel_eps': 0.15,           # Reject pairs if abs(cross) < this (for numerical stability)
        'outer_circle_max_center_distance_frac': 0.8,  # Max distance from image center (None to disable)
        'outer_circle_border_margin_frac': 0.05,     # Drop edge points near image border
        'outer_circle_cross_min_pass1': 0.30,        # Minimum cross product for pass 1 (quality filter)
        'outer_circle_cross_min_pass2': 0.40,        # Minimum cross product for pass 2 (quality filter)
        'outer_circle_line_dist_frac_pass2': 0.05,   # Max distance from c0 to normal line for pass 2 point filter (disabled by default)
        'outer_circle_use_line_filter_pass2': False,  # Enable/disable line distance filter in pass 2
        'outer_circle_cov_bins': 120,                 # Number of theta-bins for coverage (90-120 is fine)
        'outer_circle_cov_min_frac': 0.65,            # Min coverage fraction to accept ring (0.6-0.8)
        'outer_circle_fwhm_min_bins': 2,              # Minimum number of radius-bins in FWHM band
        'outer_circle_peaks_cluster_px': 3.0,         # Merge peaks closer than this (double edge)
        # Note: outer_circle_pass2_max_rings removed - now uses ALL accepted rings
        'outer_circle_pass2_outer_cut_eps': 2.0,      # Cut points with r > r_outer + eps
        'outer_circle_ground_truth_center': (1093, 1940),  # Ground truth center (cx, cy) in original image coords (None to disable) - DEPRECATED, use manual_center_xy
        # 'outer_circle_ground_truth_center': (251, 244),  # Old ground truth center (previous image)
        'outer_circle_align_min': 0.7,              # Minimum radial alignment
        'outer_circle_r_bin_px': 1.0,               # Bin width for radius histogram
        'outer_circle_r_smooth_sigma': 2.0,         # Smoothing sigma for radius histogram
        'outer_circle_peaks_keep': 10,              # Antall topper å beholde
        'outer_circle_peak_score_min_frac': 0.2,    # Minimum score fraction for peak
        'outer_circle_r_inlier_eps': 3.0,           # Inlier epsilon for final fit
    }


# Global variabel for å holde styr på visualiseringsindeks
visualization_index = 0

# Global buffer for å samle logg-oppføringer (for optimalisering)
_log_buffer = []

# Global variabel for å holde styr på total tid brukt i loggde operasjoner (i ms)
_logged_time_ms = 0.0

# Last inn konfigurasjon
config = get_config()


def save_visualization(image, title, index=None):
    """
    Lagrer en visualisering som bilde i Visualiseringer katalogen.
    
    Args:
        image: Bildet å lagre (BGR eller RGB)
        title: Beskrivende tittel
        index: Indeks for rekkefølge (hvis None, brukes global index)
    """
    global visualization_index, config
    
    # Opprett katalog hvis den ikke eksisterer
    config['visualization_dir'].mkdir(exist_ok=True)
    
    # Bruk global index hvis ikke spesifisert
    if index is None:
        index = visualization_index
        visualization_index += 1
    
    # Formater tittel for filnavn (fjern spesialtegn)
    safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
    safe_title = safe_title.replace(' ', '_')
    
    # Lag filnavn med indeks og tittel
    filename = f"{index:03d}_{safe_title}.png"
    filepath = config['visualization_dir'] / filename
    
    # Konverter til RGB hvis nødvendig (matplotlib forventer RGB)
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Sjekk om det er BGR (OpenCV) eller RGB
        # Vi antar BGR hvis det kommer fra OpenCV
        if isinstance(image, np.ndarray) and image.dtype == np.uint8:
            # Konverter BGR til RGB for lagring
            start_time = time.time()
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            log_operation_time("visualization", "save_visualization", "cv2.cvtColor", time.time() - start_time)
        else:
            image_rgb = image
    else:
        image_rgb = image
    
    # Lagre bildet
    start_time = time.time()
    if len(image_rgb.shape) == 2:  # Gråskala
        plt.imsave(filepath, image_rgb, cmap='gray')
    else:  # Farge
        plt.imsave(filepath, image_rgb)
    log_operation_time("visualization", "save_visualization", "plt.imsave", time.time() - start_time)
    
    print(f"Lagret visualisering: {filepath}")
    return filepath


def log_performance(filename, method_name, algorithm, execution_time):
    """
    Logger ytelsesdata til to filer: kronologisk og alfabetisk.
    Bruker buffer for å unngå å skrive til fil for hver operasjon.
    
    Args:
        filename: Navn på bildet som ble prosessert
        method_name: Navn på metoden/funksjonen
        algorithm: Navn på algoritmen
        execution_time: Kjøretid i sekunder
    """
    global _log_buffer
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Konverter til millisekunder og bruk komma som desimaltegn
    time_ms = execution_time * 1000
    time_str = f"{time_ms:.6f}".replace('.', ',')
    entry = f'"{timestamp}" | {filename} | {method_name} | {algorithm} | {time_str}\n'
    
    # Legg til i buffer i stedet for å skrive direkte
    _log_buffer.append(entry)


def log_operation_time(filename, method_name, operation_name, execution_time):
    """
    Logger kjøretid for en spesifikk operasjon (f.eks. cv2-funksjon).
    Bruker buffer for å unngå å skrive til fil for hver operasjon.
    Logger kun operasjoner som tar minst 1 ms.
    
    Args:
        filename: Navn på bildet som ble prosessert
        method_name: Navn på metoden/funksjonen som kaller operasjonen
        operation_name: Navn på operasjonen (f.eks. "cv2.imread", "cv2.cvtColor")
        execution_time: Kjøretid i sekunder
    
    Returns:
        True hvis logget, False hvis ikke (pga. for kort tid)
    """
    global _log_buffer
    # Kun logg hvis >= 1 ms
    if execution_time < 0.001:
        return False
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Konverter til millisekunder og bruk komma som desimaltegn
    time_ms = execution_time * 1000
    time_str = f"{time_ms:.6f}".replace('.', ',')
    entry = f'"{timestamp}" | {filename} | {method_name} | {operation_name} | {time_str}\n'
    
    # Legg til i buffer i stedet for å skrive direkte
    _log_buffer.append(entry)
    
    # Legg til i total logget tid
    global _logged_time_ms
    _logged_time_ms += time_ms
    
    return True


def flush_log_buffer():
    """
    Skriver alle buffrede logg-oppføringer til filen.
    Appender til eksisterende fil.
    Dette bør kalles når en funksjon/metode er ferdig.
    """
    global _log_buffer, config
    if not _log_buffer:
        return
    
    # Skriv alle til kronologisk fil (append)
    with open(config['log_file_kronologisk'], "a", encoding="utf-8") as f:
        f.writelines(_log_buffer)
    
    # Tøm buffer
    _log_buffer = []


def log_unaccounted_time(filename, total_time_ms, logged_time_ms):
    """
    Logger uforklart tid (tid som ikke er logget i individuelle operasjoner).
    
    Args:
        filename: Navn på bildet som ble prosessert
        total_time_ms: Total kjøretid i millisekunder
        logged_time_ms: Sum av all logget tid i millisekunder
    """
    unaccounted_ms = total_time_ms - logged_time_ms
    if unaccounted_ms >= 1.0:  # Kun logg hvis >= 1 ms
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        time_str = f"{unaccounted_ms:.6f}".replace('.', ',')
        entry = f'"{timestamp}" | {filename} | program | uforklart_tid | {time_str}\n'
        
        global _log_buffer
        _log_buffer.append(entry)




def build_radius_histogram(x, y, w, ux, uy, cx, cy, cfg, new_w, new_h, smooth_sigma=None):
    """
    Build radius histogram from point set with given center.
    
    Args:
        smooth_sigma: Optional smoothing sigma (if None, uses outer_circle_r_smooth_sigma)
    
    Returns:
        hist_smooth: Smoothed histogram
        bin_edges: Bin edges
    """
    eps = 1e-6
    align_min = cfg['outer_circle_align_min']
    rmax_search = 0.6 * min(new_h, new_w)
    bin_width = cfg['outer_circle_r_bin_px']
    bins = np.arange(0, rmax_search + bin_width, bin_width)
    if smooth_sigma is None:
        smooth_sigma = cfg['outer_circle_r_smooth_sigma']
    
    # Compute radii and alignment
    dx = x - cx
    dy = y - cy
    ri = np.sqrt(dx**2 + dy**2)
    
    # Radial alignment
    ri_safe = ri + eps
    vi = np.column_stack((dx / ri_safe, dy / ri_safe))
    u = np.column_stack((ux, uy))
    ai = np.abs(vi[:, 0] * u[:, 0] + vi[:, 1] * u[:, 1])
    
    # Filter by alignment
    keep_mask = ai >= align_min
    ri_aligned = ri[keep_mask]
    w_aligned = w[keep_mask]
    ai_aligned = ai[keep_mask]
    
    # Build histogram
    weights_hist = w_aligned * ai_aligned
    hist, bin_edges = np.histogram(ri_aligned, bins=bins, weights=weights_hist)
    
    # Smooth histogram
    try:
        from scipy import ndimage
        hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float32), smooth_sigma)
    except ImportError:
        # Fallback: simple box filter if scipy not available
        kernel_size = int(smooth_sigma * 2) * 2 + 1
        kernel = np.ones(kernel_size) / kernel_size
        hist_smooth = np.convolve(hist.astype(np.float32), kernel, mode='same')
    
    return hist_smooth, bin_edges


def nms_gradient_magnitude(Gx, Gy, mag):
    """
    Non-Maximum Suppression (NMS) on gradient magnitude (Canny-style thinning).
    
    Args:
        Gx, Gy: Gradient components (float32)
        mag: Gradient magnitude (float32)
    
    Returns:
        mag_nms: Thinned magnitude (same shape, 0 where not local max)
    """
    h, w = mag.shape
    mag_nms = np.zeros_like(mag)
    
    # Compute angle in degrees [0, 180)
    angle = np.arctan2(Gy, Gx) * 180.0 / np.pi
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


def build_thinned_pointset_outermost_ring(mag_nms, mag_raw, Gx, Gy, c_start, peak_info, cfg, new_w, new_h):
    """
    Build thinned point set for outermost ring using NMS.
    
    Args:
        mag_nms: NMS-thinned gradient magnitude
        mag_raw: Raw gradient magnitude (for pre-thinning visualization)
        Gx, Gy: Gradient components
        c_start: Start center (cx, cy) for radius calculation
        peak_info: List of peak info dicts from Pass 1
        cfg: Config dict
        new_w, new_h: Image dimensions
    
    Returns:
        x, y: Point coordinates (numpy arrays)
        x_pre, y_pre: Point coordinates before thinning (for visualization)
    """
    if len(peak_info) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Select peak with largest radius (not highest amplitude)
    outermost_peak = max(peak_info, key=lambda p: p['r_peak'])
    r_peak = outermost_peak['r_peak']
    r_lo = outermost_peak['r_lo']
    r_hi = outermost_peak['r_hi']
    
    # Expand band with padding
    pad_px = max(2.0, np.ceil(0.2 * (r_hi - r_lo)))
    r_lo2 = max(0, r_lo - pad_px)
    r_hi2 = r_hi + pad_px
    
    # Compute radii from start center
    y_coords, x_coords = np.mgrid[0:new_h, 0:new_w].astype(np.float32)
    dx = x_coords - c_start[0]
    dy = y_coords - c_start[1]
    ri = np.sqrt(dx**2 + dy**2)
    
    # Band mask
    band_mask = (ri >= r_lo2) & (ri <= r_hi2)
    
    # Alignment filter
    eps = 1e-6
    mag_safe = np.sqrt(Gx**2 + Gy**2) + eps
    ux = Gx / mag_safe
    uy = Gy / mag_safe
    
    ri_safe = ri + eps
    vx = dx / ri_safe
    vy = dy / ri_safe
    
    align = np.abs(vx * ux + vy * uy) >= cfg['outer_circle_align_min']
    
    # Pre-thinning: magnitude threshold for raw magnitude
    mag_raw_in_band = mag_raw[band_mask & (mag_raw > 0)]
    if len(mag_raw_in_band) > 0:
        t_mag_raw_percentile = cfg.get('outermost_ring_mag_percentile', 75.0)
        t_mag_raw = np.percentile(mag_raw_in_band, t_mag_raw_percentile)
        mag_raw_mask = mag_raw >= t_mag_raw
        keep_pre = band_mask & mag_raw_mask & align
        y_pre, x_pre = np.where(keep_pre)
        x_pre = x_pre.astype(np.float32)
        y_pre = y_pre.astype(np.float32)
    else:
        x_pre, y_pre = np.array([]), np.array([])
    
    # Post-thinning: magnitude threshold for NMS magnitude
    mag_nms_in_band = mag_nms[band_mask & (mag_nms > 0)]
    if len(mag_nms_in_band) == 0:
        return np.array([]), np.array([]), x_pre, y_pre
    
    t_mag_percentile = cfg.get('outermost_ring_mag_percentile', 75.0)
    t_mag = np.percentile(mag_nms_in_band, t_mag_percentile)
    
    # Magnitude mask
    mag_mask = mag_nms >= t_mag
    
    # Combined mask
    keep = band_mask & mag_mask & align
    
    # Extract points
    y_pts, x_pts = np.where(keep)
    
    return x_pts.astype(np.float32), y_pts.astype(np.float32), x_pre, y_pre


def refine_center_radial_variance(x, y, c_start, cfg, new_w, new_h):
    """
    Refine center using hillclimbing to minimize radial variance.
    
    Args:
        x, y: Point coordinates (numpy arrays)
        c_start: Start center (cx, cy)
        cfg: Config dict
        new_w, new_h: Image dimensions
    
    Returns:
        c_best: (cx, cy) refined center
        best_variance: best variance achieved
        steps_taken: number of steps taken
    """
    if len(x) == 0:
        return c_start, float('inf'), 0
    
    max_steps = cfg.get('outermost_ring_refine_max_steps', 60)
    step_px = cfg.get('outermost_ring_refine_step_px', 1)
    max_radius_px = cfg.get('outermost_ring_refine_max_radius_px', 12)
    tiny_eps = 1e-12
    
    # Start from rounded pixel center
    c = np.array([round(c_start[0]), round(c_start[1])], dtype=np.float32)
    c_start_rounded = c.copy()
    
    # Compute initial variance
    dx = x - c[0]
    dy = y - c[1]
    ri = np.sqrt(dx**2 + dy**2)
    r_mean = np.mean(ri)
    best_variance = np.mean((ri - r_mean)**2)
    
    # 8-neighborhood offsets
    offsets = [(step_px, 0), (-step_px, 0), (0, step_px), (0, -step_px),
               (step_px, step_px), (-step_px, -step_px), (step_px, -step_px), (-step_px, step_px)]
    
    steps_taken = 0
    for step in range(max_steps):
        best_neighbor_variance = best_variance
        best_neighbor = None
        
        for dx_offset, dy_offset in offsets:
            c2 = c + np.array([dx_offset, dy_offset])
            
            # Check bounds
            if c2[0] < 0 or c2[0] >= new_w or c2[1] < 0 or c2[1] >= new_h:
                continue
            
            # Check max radius constraint
            dist_from_start = np.sqrt((c2[0] - c_start_rounded[0])**2 + (c2[1] - c_start_rounded[1])**2)
            if dist_from_start > max_radius_px:
                continue
            
            # Compute variance
            dx2 = x - c2[0]
            dy2 = y - c2[1]
            ri2 = np.sqrt(dx2**2 + dy2**2)
            r_mean2 = np.mean(ri2)
            variance2 = np.mean((ri2 - r_mean2)**2)
            
            if variance2 < best_neighbor_variance:
                best_neighbor_variance = variance2
                best_neighbor = c2.copy()
        
        if best_neighbor is not None and best_neighbor_variance < best_variance - tiny_eps:
            c = best_neighbor
            best_variance = best_neighbor_variance
            steps_taken += 1
        else:
            break  # Local minimum reached
    
    return tuple(c), best_variance, steps_taken


def center_refine_score_moment(hist_smooth, cfg):
    """
    Compute moment-based score for center refinement.
    Belønner konsentrasjon rundt en "klump" (platå OK), straffer haler.
    
    Returns:
        score: float (higher is better, negative value)
    """
    h = np.clip(hist_smooth, 0.0, None)
    eps = cfg['center_refine_eps']
    
    # Concave transform to prevent single bin from dominating
    g = np.sqrt(h + eps)
    s = g.sum()
    
    if s <= 0:
        return -1e9  # Bad score
    
    p = g / s
    
    # Compute moments
    i = np.arange(len(p), dtype=np.float32)
    mu = np.sum(i * p)  # Mass center (bin index)
    m2 = np.sum((i - mu)**2 * p)  # Second moment (variance)
    
    score = -m2
    
    # Optionally add m4 moment
    if cfg.get('center_refine_use_m4', False):
        m4 = np.sum((i - mu)**4 * p)
        lambda_m4 = cfg.get('center_refine_m4_lambda', 0.05)
        score = -(m2 + lambda_m4 * m4)
    
    return float(score)


def refine_center_hillclimb_moment(x, y, w, ux, uy, c_start, cfg, new_w, new_h):
    """
    Refine center using hillclimbing on moment-based score.
    
    Returns:
        c_best: (cx, cy) refined center
        best_score: best score achieved
        steps: number of steps taken
    """
    max_steps = cfg['center_refine_max_steps']
    step_px = cfg['center_refine_step_px']
    max_radius_px = cfg['center_refine_max_radius_px']
    tiny_eps = 1e-12
    refine_sigma = cfg['center_refine_r_smooth_sigma']
    
    # Start from rounded pixel center for stable debug
    c = np.array([round(c_start[0]), round(c_start[1])], dtype=np.float32)
    c_start_rounded = c.copy()
    
    # Build histogram with refine-specific smoothing
    hist_smooth, _ = build_radius_histogram(x, y, w, ux, uy, c[0], c[1], cfg, new_w, new_h, smooth_sigma=refine_sigma)
    best_score = center_refine_score_moment(hist_smooth, cfg)
    
    # 8-neighborhood offsets
    offsets = [(step_px, 0), (-step_px, 0), (0, step_px), (0, -step_px),
               (step_px, step_px), (-step_px, -step_px), (step_px, -step_px), (-step_px, step_px)]
    
    steps_taken = 0
    for step in range(max_steps):
        best_neighbor_score = best_score
        best_neighbor = None
        
        for dx, dy in offsets:
            c2 = c + np.array([dx, dy])
            
            # Check bounds
            if c2[0] < 0 or c2[0] >= new_w or c2[1] < 0 or c2[1] >= new_h:
                continue
            
            # Check max radius constraint from rounded start
            dist_from_start = np.sqrt((c2[0] - c_start_rounded[0])**2 + (c2[1] - c_start_rounded[1])**2)
            if dist_from_start > max_radius_px:
                continue
            
            # Compute score with refine-specific smoothing
            hist_smooth2, _ = build_radius_histogram(x, y, w, ux, uy, c2[0], c2[1], cfg, new_w, new_h, smooth_sigma=refine_sigma)
            score2 = center_refine_score_moment(hist_smooth2, cfg)
            
            if score2 > best_neighbor_score:
                best_neighbor_score = score2
                best_neighbor = c2.copy()
        
        if best_neighbor is not None and best_neighbor_score > best_score + tiny_eps:
            c = best_neighbor
            best_score = best_neighbor_score
            steps_taken += 1
        else:
            break  # Local maximum reached
    
    return tuple(c), best_score, steps_taken


def fit_circle_pratt(points, weights):
    """
    Fit circle using Pratt's algebraic method (weighted).
    Minimizes algebraic distance with weights.
    """
    n = len(points)
    if n < 3:
        return None
    
    x = points[:, 0]
    y = points[:, 1]
    w = weights
    
    # Weighted means
    x_mean = np.sum(w * x) / np.sum(w)
    y_mean = np.sum(w * y) / np.sum(w)
    
    # Center points
    u = x - x_mean
    v = y - y_mean
    
    # Weighted moments
    suu = np.sum(w * u * u)
    svv = np.sum(w * v * v)
    suv = np.sum(w * u * v)
    suuu = np.sum(w * u * u * u)
    svvv = np.sum(w * v * v * v)
    suvv = np.sum(w * u * v * v)
    svuu = np.sum(w * v * u * u)
    
    # Solve for center offset
    A = np.array([[suu, suv], [suv, svv]])
    b = np.array([0.5 * (suuu + suvv), 0.5 * (svvv + svuu)])
    
    try:
        delta = np.linalg.solve(A, b)
        cx = x_mean + delta[0]
        cy = y_mean + delta[1]
        
        # Compute radius
        r = np.sqrt(np.sum(w * ((x - cx)**2 + (y - cy)**2)) / np.sum(w))
        return (cx, cy, r)
    except np.linalg.LinAlgError:
        return None


def detect_outer_circle(img: np.ndarray, cfg: dict, debug: bool = False, 
                       filename: str = "unknown"):
    """
    Detekterer ytterste sirkel ved hjelp av gradient-normal center voting.
    
    Args:
        img: Input bilde (BGR eller gråskala)
        cfg: Konfigurasjonsdictionary
        debug: Om debug-artifakter skal returneres
        filename: Filnavn for logging
    
    Returns:
        (cx, cy, r) i original bilde-koordinater, og eventuelt debug_dict
    """
    start_total = time.time()
    
    # 1) Downscale
    start_resize = time.time()
    h, w = img.shape[:2]
    max_dim = max(h, w)
    target_max = cfg['outer_circle_max_side']
    scale = target_max / max_dim if max_dim > target_max else 1.0
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_down = img.copy()
        new_w, new_h = w, h
    log_operation_time(filename, "detect_outer_circle", "resize", time.time() - start_resize)
    
    # 2) Grayscale + blur
    start_preprocess = time.time()
    if len(img_down.shape) == 3:
        gray = cv2.cvtColor(img_down, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_down.copy()
    
    sigma = cfg['outer_circle_blur_sigma']
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1
    gray = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    log_operation_time(filename, "detect_outer_circle", "preprocess", time.time() - start_preprocess)
    
    # 3) Gradients
    start_gradient = time.time()
    Gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    Gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag_raw = cv2.magnitude(Gx, Gy)  # Keep raw magnitude for pass 2
    
    # Create filtered magnitude for pass 1 (axis filtering)
    mag_for_pass1 = mag_raw.copy()
    angle_threshold_deg = cfg['outer_circle_filter_angle_deg']
    if angle_threshold_deg > 0:
        angle_threshold_rad = np.deg2rad(angle_threshold_deg)
        angle_threshold_tan = np.tan(angle_threshold_rad)
        eps = 1e-6
        
        # Check if gradient is nearly horizontal or vertical
        nearly_horizontal = np.abs(Gy) / (np.abs(Gx) + eps) < angle_threshold_tan
        nearly_vertical = np.abs(Gx) / (np.abs(Gy) + eps) < angle_threshold_tan
        
        # Set magnitude to 0 for nearly horizontal or vertical gradients (pass 1 only)
        mag_for_pass1[nearly_horizontal | nearly_vertical] = 0.0
    
    # Gradient directions (normalized) - use raw magnitude for normalization
    eps = 1e-6
    mag_safe = mag_raw + eps
    ux = Gx / mag_safe
    uy = Gy / mag_safe
    log_operation_time(filename, "detect_outer_circle", "gradient", time.time() - start_gradient)
    
    # Helper function for center voting using intersection-of-normals
    def center_vote_intersections(x, y, ux, uy, w, W, H, cfg, cross_min):
        """
        Vote for center using intersection-of-normals method.
        
        Returns:
            (cx, cy): refined center (float)
            accf: accumulator (float32 array)
            stats: dict with statistics
        """
        accf = np.zeros((H, W), dtype=np.float32)
        accf_flat = accf.ravel()  # Use ravel() for efficiency
        
        M = cfg['outer_circle_center_pairs']
        rng = np.random.default_rng(42)
        parallel_eps = cfg['outer_circle_parallel_eps']
        max_dist_frac = cfg.get('outer_circle_max_center_distance_frac', None)
        win = cfg['outer_circle_center_win']
        
        n_points = len(x)
        stats = {
            'sampled_pairs': M,
            'parallel_reject': 0,
            'cross_reject': 0,
            'dist_reject': 0,
            'oob_reject': 0,
            'votes_cast': 0,
            'peak_value': 0.0
        }
        
        if n_points < 2:
            return (W/2.0, H/2.0), accf, stats
        
        # Generate random pairs (i != j)
        i_indices = rng.integers(0, n_points, size=M)
        j_indices = rng.integers(0, n_points, size=M)
        same_mask = i_indices == j_indices
        j_indices[same_mask] = (j_indices[same_mask] + 1) % n_points
        
        # Extract pairs
        x1, y1 = x[i_indices], y[i_indices]
        x2, y2 = x[j_indices], y[j_indices]
        ux1, uy1 = ux[i_indices], uy[i_indices]
        ux2, uy2 = ux[j_indices], uy[j_indices]
        w1, w2 = w[i_indices], w[j_indices]
        
        # Compute cross product
        cross = ux1 * uy2 - uy1 * ux2
        
        # Filter 1: Numerical stability (parallel_eps)
        valid_parallel = np.abs(cross) >= parallel_eps
        stats['parallel_reject'] = np.sum(~valid_parallel)
        
        # Filter 2: Quality filter (cross_min)
        valid_cross = np.abs(cross) >= cross_min
        stats['cross_reject'] = np.sum(valid_parallel & ~valid_cross)
        valid = valid_parallel & valid_cross
        
        # Filter to valid pairs
        x1 = x1[valid]
        y1 = y1[valid]
        x2 = x2[valid]
        y2 = y2[valid]
        ux1 = ux1[valid]
        uy1 = uy1[valid]
        ux2 = ux2[valid]
        uy2 = uy2[valid]
        w1 = w1[valid]
        w2 = w2[valid]
        cross = cross[valid]
        
        # Solve for intersection
        dx = x2 - x1
        dy = y2 - y1
        s = (dx * uy2 - dy * ux2) / cross
        cx = x1 + s * ux1
        cy = y1 + s * uy1
        
        # Filter by distance from image center
        if max_dist_frac is not None:
            img_center_x = W / 2.0
            img_center_y = H / 2.0
            dist_from_center = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
            max_dist = max_dist_frac * np.hypot(W, H)
            valid_distance = dist_from_center <= max_dist
            stats['dist_reject'] = np.sum(~valid_distance)
            cx = cx[valid_distance]
            cy = cy[valid_distance]
            w1 = w1[valid_distance]
            w2 = w2[valid_distance]
        else:
            valid_distance = np.ones(len(cx), dtype=bool)
        
        # Round to integer coordinates
        cx_int = np.round(cx).astype(int)
        cy_int = np.round(cy).astype(int)
        
        # Filter out-of-bounds
        valid_bounds = (cx_int >= 0) & (cx_int < W) & (cy_int >= 0) & (cy_int < H)
        stats['oob_reject'] = np.sum(~valid_bounds)
        
        cx_int = cx_int[valid_bounds]
        cy_int = cy_int[valid_bounds]
        vote_weights = np.minimum(w1[valid_bounds], w2[valid_bounds])
        
        # Vote
        if len(cx_int) > 0:
            idx = cy_int * W + cx_int
            np.add.at(accf_flat, idx, vote_weights)
            stats['votes_cast'] = len(cx_int)
        
        # Find peak
        peak_idx = np.argmax(accf)
        peak_y = peak_idx // W
        peak_x = peak_idx % W
        stats['peak_value'] = float(accf[peak_y, peak_x])
        
        # Subpixel refinement
        x_min = max(0, peak_x - win)
        x_max = min(W, peak_x + win + 1)
        y_min = max(0, peak_y - win)
        y_max = min(H, peak_y + win + 1)
        
        window = accf[y_min:y_max, x_min:x_max]
        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
        
        total_weight = np.sum(window)
        if total_weight > 0:
            cx_refined = np.sum(x_grid * window) / total_weight
            cy_refined = np.sum(y_grid * window) / total_weight
        else:
            cx_refined, cy_refined = float(peak_x), float(peak_y)
        
        return (cx_refined, cy_refined), accf, stats
    
    # Helper function to extract edge points with border margin filter
    def extract_edge_points(mag_used, ux, uy, cfg, new_w, new_h):
        """Extract edge points with border margin filtering."""
        mag_floor = cfg['outer_circle_mag_floor']
        mag_nonzero = mag_used[mag_used > mag_floor]
        if len(mag_nonzero) == 0:
            return None, None, None, None
        
        percentile = cfg['outer_circle_mag_percentile']
        threshold = np.percentile(mag_nonzero, percentile)
        edge_mask = mag_used >= threshold
        
        # Extract edge points
        y_coords, x_coords = np.where(edge_mask)
        x = x_coords.astype(np.float32)
        y = y_coords.astype(np.float32)
        u = np.column_stack((ux[y_coords, x_coords], uy[y_coords, x_coords]))
        w = mag_used[y_coords, x_coords]
        
        # Border margin filter
        margin = cfg['outer_circle_border_margin_frac'] * min(new_w, new_h)
        keep = (x >= margin) & (x < new_w - margin) & (y >= margin) & (y < new_h - margin)
        x = x[keep]
        y = y[keep]
        u = u[keep]
        w = w[keep]
        
        # Random subsample if too many
        max_points = cfg['outer_circle_max_edge_points']
        if len(x) > max_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(len(x), max_points, replace=False)
            x = x[indices]
            y = y[indices]
            u = u[indices]
            w = w[indices]
        
        return x, y, u, w
    
    # 4) Pass 1: Coarse center with axis filtering
    start_threshold = time.time()
    x1, y1, u1, w1 = extract_edge_points(mag_for_pass1, ux, uy, cfg, new_w, new_h)
    if x1 is None:
        raise ValueError("Ingen edge points funnet i Pass 1")
    log_operation_time(filename, "detect_outer_circle", "threshold_extract_pass1", time.time() - start_threshold)
    
    # 5) Pass 1: Center voting
    start_voting_pass1 = time.time()
    debug_lines = []
    if debug:
        debug_lines.append(f"=== Pass 1 (coarse center) ===")
        debug_lines.append(f"Edge points: {len(x1)}")
    
    cross_min_pass1 = cfg['outer_circle_cross_min_pass1']
    (c0x, c0y), accf_pass1, stats_pass1 = center_vote_intersections(
        x1, y1, u1[:, 0], u1[:, 1], w1, new_w, new_h, cfg, cross_min_pass1
    )
    
    # Save pass 1 center explicitly
    c0x_pass1, c0y_pass1 = c0x, c0y
    
    if debug:
        debug_lines.append(f"Pairs sampled: {stats_pass1['sampled_pairs']}")
        debug_lines.append(f"Rejected (parallel): {stats_pass1['parallel_reject']}")
        debug_lines.append(f"Rejected (cross_min): {stats_pass1['cross_reject']}")
        debug_lines.append(f"Rejected (distance): {stats_pass1['dist_reject']}")
        debug_lines.append(f"Rejected (out-of-bounds): {stats_pass1['oob_reject']}")
        debug_lines.append(f"Votes cast: {stats_pass1['votes_cast']}")
        debug_lines.append(f"Peak value: {stats_pass1['peak_value']:.2f}")
        debug_lines.append(f"Coarse center: ({c0x:.2f}, {c0y:.2f})")
    
    log_operation_time(filename, "detect_outer_circle", "voting_pass1", time.time() - start_voting_pass1)
    
    # 6) Radius estimation for Pass 1 (needed for Pass 2 point selection)
    start_radius_pass1 = time.time()
    # Use pass 1 point set (x1, y1, u1, w1) and center c0_pass1 (from pass 1)
    # Compute radii and alignment
    dx1 = x1 - c0x_pass1
    dy1 = y1 - c0y_pass1
    ri1 = np.sqrt(dx1**2 + dy1**2)
    
    # Radial alignment
    eps = 1e-6
    ri1_safe = ri1 + eps
    vi1 = np.column_stack((dx1 / ri1_safe, dy1 / ri1_safe))
    ai1 = np.abs(vi1[:, 0] * u1[:, 0] + vi1[:, 1] * u1[:, 1])
    
    # Filter by alignment
    align_min = cfg['outer_circle_align_min']
    keep_mask1 = ai1 >= align_min
    x1_aligned = x1[keep_mask1]
    y1_aligned = y1[keep_mask1]
    ri1_aligned = ri1[keep_mask1]
    w1_aligned = w1[keep_mask1]
    ai1_aligned = ai1[keep_mask1]
    
    # Build histogram for pass 1
    rmax_search = 0.6 * min(new_h, new_w)
    bin_width = cfg['outer_circle_r_bin_px']
    bins = np.arange(0, rmax_search + bin_width, bin_width)
    weights_hist1 = w1_aligned * ai1_aligned
    hist1, bin_edges1 = np.histogram(ri1_aligned, bins=bins, weights=weights_hist1)
    
    # Smooth histogram (using standard sigma for pass 1)
    smooth_sigma = cfg['outer_circle_r_smooth_sigma']
    try:
        from scipy import ndimage
        hist1_pass1_smooth = ndimage.gaussian_filter1d(hist1.astype(np.float32), smooth_sigma)
    except ImportError:
        # Fallback: simple box filter if scipy not available
        kernel_size = int(smooth_sigma * 2) * 2 + 1
        kernel = np.ones(kernel_size) / kernel_size
        hist1_pass1_smooth = np.convolve(hist1.astype(np.float32), kernel, mode='same')
    
    # Use hist1_pass1_smooth for peak detection (keep old name for compatibility in this section)
    hist1_smooth = hist1_pass1_smooth
    
    log_operation_time(filename, "detect_outer_circle", "radius_hist_pass1", time.time() - start_radius_pass1)
    
    # 7) Find peaks in Pass 1 histogram and build Pass 2 point set
    start_pass2_selection = time.time()
    
    # B) Find peaks
    peak_min = cfg['outer_circle_peak_score_min_frac'] * np.max(hist1_smooth)
    peaks_keep = cfg['outer_circle_peaks_keep']
    
    try:
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist1_smooth, height=peak_min)
    except ImportError:
        # Fallback: simple local maxima detection
        peaks = []
        for i in range(1, len(hist1_smooth) - 1):
            if hist1_smooth[i] > hist1_smooth[i-1] and hist1_smooth[i] > hist1_smooth[i+1]:
                if hist1_smooth[i] >= peak_min:
                    peaks.append(i)
        peaks = np.array(peaks)
    
    if len(peaks) == 0:
        # Fallback: use maximum
        peaks = np.array([np.argmax(hist1_smooth)])
    
    # Sort by radius (largest first) and keep top N
    r_peaks = (bin_edges1[peaks] + bin_edges1[peaks + 1]) / 2
    sorted_indices = np.argsort(r_peaks)[::-1][:peaks_keep]
    sorted_peaks = peaks[sorted_indices]
    sorted_r_peaks = r_peaks[sorted_indices]
    
    # C) Cluster nearby peaks (double edges)
    cluster_px = cfg['outer_circle_peaks_cluster_px']
    clustered_peaks = []
    clustered_r = []
    used = np.zeros(len(sorted_peaks), dtype=bool)
    
    for i in range(len(sorted_peaks)):
        if used[i]:
            continue
        cluster = [i]
        cluster_r = [sorted_r_peaks[i]]
        for j in range(i + 1, len(sorted_peaks)):
            if not used[j] and abs(sorted_r_peaks[i] - sorted_r_peaks[j]) < cluster_px:
                cluster.append(j)
                cluster_r.append(sorted_r_peaks[j])
                used[j] = True
        
        # Choose peak with highest value, or weighted average
        if len(cluster) == 1:
            clustered_peaks.append(sorted_peaks[i])
            clustered_r.append(sorted_r_peaks[i])
        else:
            # Use weighted average of radius
            cluster_vals = hist1_smooth[sorted_peaks[cluster]]
            total_weight = np.sum(cluster_vals)
            if total_weight > 0:
                r_weighted = np.sum(np.array(cluster_r) * cluster_vals) / total_weight
                # Find closest bin
                best_idx = cluster[np.argmax(cluster_vals)]
                clustered_peaks.append(sorted_peaks[best_idx])
                clustered_r.append(r_weighted)
            else:
                clustered_peaks.append(sorted_peaks[i])
                clustered_r.append(sorted_r_peaks[i])
    
    # D) Compute FWHM bands and coverage for each peak
    fwhm_min_bins = cfg['outer_circle_fwhm_min_bins']
    cov_bins = cfg['outer_circle_cov_bins']
    cov_min_frac = cfg['outer_circle_cov_min_frac']
    
    accepted_peaks = []
    peak_info = []
    
    for peak_idx, r_peak in zip(clustered_peaks, clustered_r):
        peak_val = hist1_smooth[peak_idx]
        half = 0.5 * peak_val
        
        # Find FWHM: go left and right
        left = peak_idx
        while left > 0 and hist1_smooth[left] >= half:
            left -= 1
        left += 1  # First bin where hist < half
        
        right = peak_idx
        while right < len(hist1_smooth) - 1 and hist1_smooth[right] >= half:
            right += 1
        right -= 1  # Last bin where hist >= half
        
        # Check FWHM width
        if (right - left + 1) < fwhm_min_bins:
            continue
        
        # Get radius interval from bin edges
        r_lo = bin_edges1[left]
        r_hi = bin_edges1[right + 1]
        
        # Compute coverage using pass 1 aligned points
        mask = (ri1_aligned >= r_lo) & (ri1_aligned <= r_hi)
        if np.sum(mask) < 10:  # Too few points
            continue
        
        x_band = x1_aligned[mask]
        y_band = y1_aligned[mask]
        theta = np.arctan2(y_band - c0y_pass1, x_band - c0x_pass1)
        theta_bins = np.floor((theta + np.pi) / (2 * np.pi) * cov_bins).astype(int)
        theta_bins = np.clip(theta_bins, 0, cov_bins - 1)
        unique_bins = len(np.unique(theta_bins))
        coverage = unique_bins / cov_bins
        
        if coverage >= cov_min_frac:
            accepted_peaks.append(peak_idx)
            peak_info.append({
                'idx': peak_idx,
                'r_peak': r_peak,
                'r_lo': r_lo,
                'r_hi': r_hi,
                'coverage': coverage
            })
    
    if debug:
        debug_lines.append(f"\n=== Pass 1 histogram peaks ===")
        debug_lines.append(f"Initial peaks found: {len(peaks)}")
        debug_lines.append(f"After clustering: {len(clustered_peaks)}")
        debug_lines.append(f"Accepted peaks (FWHM + coverage): {len(accepted_peaks)}")
        for info in peak_info:
            debug_lines.append(f"  Peak at r={info['r_peak']:.2f}, FWHM=[{info['r_lo']:.2f}, {info['r_hi']:.2f}], coverage={info['coverage']:.3f}")
    
    # E) Select rings for Pass 2 - use ALL accepted rings (automatic)
    if len(accepted_peaks) == 0:
        # Fallback: use maximum
        peak_idx = np.argmax(hist1_smooth)
        r_peak = (bin_edges1[peak_idx] + bin_edges1[peak_idx + 1]) / 2
        r_outer = r_peak
        selected_rings = [{'r_lo': r_peak - 5, 'r_hi': r_peak + 5}]
    else:
        # Use ALL accepted peaks (not just top N)
        selected_rings = peak_info  # All accepted rings
        # Find outer radius (largest peak)
        r_outer = max(ring['r_peak'] for ring in selected_rings)
    
    if debug:
        debug_lines.append(f"\n=== Pass 2 point selection ===")
        debug_lines.append(f"Selected r_outer: {r_outer:.2f}")
        debug_lines.append(f"Number of accepted rings (all used): {len(selected_rings)}")
    
    # Extract Pass 2 edge points (before line filter)
    x2, y2, u2, w2 = extract_edge_points(mag_raw, ux, uy, cfg, new_w, new_h)
    if x2 is None:
        raise ValueError("Ingen edge points funnet i Pass 2")
    
    n_points_before = len(x2)
    
    # Compute radii from c0_pass1 (for pass 2 point selection)
    ri2 = np.sqrt((x2 - c0x_pass1)**2 + (y2 - c0y_pass1)**2)
    
    # Build keep mask: outer cutoff + FWHM bands
    outer_cut_eps = cfg['outer_circle_pass2_outer_cut_eps']
    keep_outer = ri2 <= (r_outer + outer_cut_eps)
    keep_band = np.zeros(len(x2), dtype=bool)
    
    for ring in selected_rings:
        keep_band |= (ri2 >= ring['r_lo']) & (ri2 <= ring['r_hi'])
    
    # Apply line distance filter only if enabled
    if cfg.get('outer_circle_use_line_filter_pass2', False):
        d_max = cfg['outer_circle_line_dist_frac_pass2'] * min(new_w, new_h)
        d = np.abs((c0x_pass1 - x2) * u2[:, 1] - (c0y_pass1 - y2) * u2[:, 0])
        keep_line = d < d_max
        keep2 = keep_outer & keep_band & keep_line
    else:
        keep2 = keep_outer & keep_band
    
    x2 = x2[keep2]
    y2 = y2[keep2]
    u2 = u2[keep2]
    w2 = w2[keep2]
    
    n_points_after = len(x2)
    
    if debug:
        debug_lines.append(f"Pass 2 points before band filter: {n_points_before}")
        debug_lines.append(f"Pass 2 points after band filter: {n_points_after}")
    
    log_operation_time(filename, "detect_outer_circle", "pass2_selection", time.time() - start_pass2_selection)
    
    # 8) Pass 2: Center voting
    start_voting_pass2 = time.time()
    cross_min_pass2 = cfg['outer_circle_cross_min_pass2']
    (c1x, c1y), accf_pass2, stats_pass2 = center_vote_intersections(
        x2, y2, u2[:, 0], u2[:, 1], w2, new_w, new_h, cfg, cross_min_pass2
    )
    
    # Save pass 2 voting center explicitly
    c1x_pass2, c1y_pass2 = c1x, c1y
    
    if debug:
        debug_lines.append(f"Pairs sampled: {stats_pass2['sampled_pairs']}")
        debug_lines.append(f"Rejected (parallel): {stats_pass2['parallel_reject']}")
        debug_lines.append(f"Rejected (cross_min): {stats_pass2['cross_reject']}")
        debug_lines.append(f"Rejected (distance): {stats_pass2['dist_reject']}")
        debug_lines.append(f"Rejected (out-of-bounds): {stats_pass2['oob_reject']}")
        debug_lines.append(f"Votes cast: {stats_pass2['votes_cast']}")
        debug_lines.append(f"Peak value: {stats_pass2['peak_value']:.2f}")
        debug_lines.append(f"Precise center: ({c1x:.2f}, {c1y:.2f})")
    
    log_operation_time(filename, "detect_outer_circle", "voting_pass2", time.time() - start_voting_pass2)
    
    # Build histogram before refinement (for comparison) - use standard smoothing
    hist2_before_refine = None
    if cfg.get('center_refine_enable', False):
        hist2_before_refine, _ = build_radius_histogram(x2, y2, w2, u2[:, 0], u2[:, 1], c1x, c1y, cfg, new_w, new_h)
    
    # Center refinement with hillclimbing (moment-based score)
    c2x, c2y = c1x, c1y  # Final center (may be refined)
    if cfg.get('center_refine_enable', False):
        start_refine = time.time()
        c_ref, score_ref, steps_taken = refine_center_hillclimb_moment(
            x2, y2, w2, u2[:, 0], u2[:, 1], (c1x, c1y), cfg, new_w, new_h
        )
        
        # Compute m2 before/after for debug
        refine_sigma = cfg['center_refine_r_smooth_sigma']
        hist_before, _ = build_radius_histogram(x2, y2, w2, u2[:, 0], u2[:, 1], c1x, c1y, cfg, new_w, new_h, smooth_sigma=refine_sigma)
        hist_after, _ = build_radius_histogram(x2, y2, w2, u2[:, 0], u2[:, 1], c_ref[0], c_ref[1], cfg, new_w, new_h, smooth_sigma=refine_sigma)
        
        h_before = np.clip(hist_before, 0.0, None)
        g_before = np.sqrt(h_before + cfg['center_refine_eps'])
        p_before = g_before / (g_before.sum() + 1e-12)
        i_before = np.arange(len(p_before), dtype=np.float32)
        mu_before = np.sum(i_before * p_before)
        m2_before = np.sum((i_before - mu_before)**2 * p_before)
        
        h_after = np.clip(hist_after, 0.0, None)
        g_after = np.sqrt(h_after + cfg['center_refine_eps'])
        p_after = g_after / (g_after.sum() + 1e-12)
        i_after = np.arange(len(p_after), dtype=np.float32)
        mu_after = np.sum(i_after * p_after)
        m2_after = np.sum((i_after - mu_after)**2 * p_after)
        
        if debug:
            debug_lines.append(f"\n=== Center refinement (hillclimb moment) ===")
            debug_lines.append(f"Start center (rounded): ({round(c1x):.0f}, {round(c1y):.0f})")
            debug_lines.append(f"Refined center: ({c_ref[0]:.2f}, {c_ref[1]:.2f})")
            debug_lines.append(f"Displacement: ({c_ref[0] - c1x:.2f}, {c_ref[1] - c1y:.2f})")
            debug_lines.append(f"Best score: {score_ref:.6f}")
            debug_lines.append(f"Steps taken: {steps_taken}")
            debug_lines.append(f"m2 before: {m2_before:.6f}, m2 after: {m2_after:.6f}")
        
        log_operation_time(filename, "detect_outer_circle", "center_refine_hillclimb", time.time() - start_refine)
        c2x_ref, c2y_ref = c_ref[0], c_ref[1]  # Save refined center
    else:
        c2x_ref, c2y_ref = c1x_pass2, c1y_pass2  # No refinement, use pass 2 center
    
    # Final center for output (may be further refined with radial variance)
    c_finalx, c_finaly = c2x_ref, c2y_ref
    
    # Optional: Refine with radial variance on thinned outermost ring point set
    mag_nms = None
    x_thin, y_thin = np.array([]), np.array([])
    x_pre, y_pre = np.array([]), np.array([])
    if cfg.get('outermost_ring_refine_enable', False) and len(peak_info) > 0:
        start_nms = time.time()
        # Apply NMS to gradient magnitude
        mag_nms = nms_gradient_magnitude(Gx, Gy, mag_raw)
        log_operation_time(filename, "detect_outer_circle", "nms", time.time() - start_nms)
        
        # Build thinned point set for outermost ring
        start_pointset = time.time()
        x_thin, y_thin, x_pre, y_pre = build_thinned_pointset_outermost_ring(
            mag_nms, mag_raw, Gx, Gy, (c_finalx, c_finaly), peak_info, cfg, new_w, new_h
        )
        log_operation_time(filename, "detect_outer_circle", "build_thinned_pointset", time.time() - start_pointset)
        
        if len(x_thin) > 0:
            start_var_refine = time.time()
            c_var_ref, var_ref, steps_var = refine_center_radial_variance(
                x_thin, y_thin, (c_finalx, c_finaly), cfg, new_w, new_h
            )
            
            if debug:
                debug_lines.append(f"\n=== Outermost ring refinement (radial variance) ===")
                debug_lines.append(f"Start center: ({c_finalx:.2f}, {c_finaly:.2f})")
                debug_lines.append(f"Refined center: ({c_var_ref[0]:.2f}, {c_var_ref[1]:.2f})")
                debug_lines.append(f"Displacement: ({c_var_ref[0] - c_finalx:.2f}, {c_var_ref[1] - c_finaly:.2f})")
                debug_lines.append(f"Best variance: {var_ref:.6f}")
                debug_lines.append(f"Steps taken: {steps_var}")
                debug_lines.append(f"Thinned points: {len(x_thin)}")
            
            log_operation_time(filename, "detect_outer_circle", "radial_variance_refine", time.time() - start_var_refine)
            c_finalx, c_finaly = c_var_ref[0], c_var_ref[1]
        else:
            if debug:
                debug_lines.append(f"\n=== Outermost ring refinement (radial variance) ===")
                debug_lines.append(f"No thinned points found, skipping refinement")
    
    # 9) Radius estimation for Pass 2 (using final refined center)
    start_radius_pass2 = time.time()
    # Use pass 2 point set (x2, y2, u2, w2) and final refined center c_final
    # Compute radii and alignment
    dx2 = x2 - c_finalx
    dy2 = y2 - c_finaly
    ri2 = np.sqrt(dx2**2 + dy2**2)
    
    # Radial alignment
    ri2_safe = ri2 + eps
    vi2 = np.column_stack((dx2 / ri2_safe, dy2 / ri2_safe))
    ai2 = np.abs(vi2[:, 0] * u2[:, 0] + vi2[:, 1] * u2[:, 1])
    
    # Filter by alignment
    keep_mask2 = ai2 >= align_min
    x2_aligned = x2[keep_mask2]
    y2_aligned = y2[keep_mask2]
    ri2_aligned = ri2[keep_mask2]
    w2_aligned = w2[keep_mask2]
    ai2_aligned = ai2[keep_mask2]
    
    # Build histogram for pass 2
    weights_hist2 = w2_aligned * ai2_aligned
    hist2, bin_edges2 = np.histogram(ri2_aligned, bins=bins, weights=weights_hist2)
    
    # Smooth histogram (using standard sigma for pass 2)
    smooth_sigma = cfg['outer_circle_r_smooth_sigma']
    try:
        from scipy import ndimage
        hist2_pass2_smooth = ndimage.gaussian_filter1d(hist2.astype(np.float32), smooth_sigma)
    except ImportError:
        # Fallback: simple box filter if scipy not available
        kernel_size = int(smooth_sigma * 2) * 2 + 1
        kernel = np.ones(kernel_size) / kernel_size
        hist2_pass2_smooth = np.convolve(hist2.astype(np.float32), kernel, mode='same')
    
    log_operation_time(filename, "detect_outer_circle", "radius_hist_pass2", time.time() - start_radius_pass2)
    
    # Ground truth histogram for Pass 2 (if manual center is provided)
    hist2_gt_smooth = None
    bin_edges2_gt = None
    if cfg.get('manual_center_enable', False) and cfg.get('manual_center_xy') is not None:
        # Use manual_center_xy (already in downscaled coordinates)
        gt_cx, gt_cy = cfg['manual_center_xy']
    elif cfg.get('outer_circle_ground_truth_center') is not None:
        # Fallback to old ground truth center (convert from original to downscaled)
        gt_cx_orig, gt_cy_orig = cfg['outer_circle_ground_truth_center']
        gt_cx = gt_cx_orig * scale
        gt_cy = gt_cy_orig * scale
    else:
        gt_cx = None
        gt_cy = None
    
    if gt_cx is not None and gt_cy is not None:
        
        # Use same pass 2 point set (x2, y2, u2, w2) but with ground truth center
        dx2_gt = x2 - gt_cx
        dy2_gt = y2 - gt_cy
        ri2_gt = np.sqrt(dx2_gt**2 + dy2_gt**2)
        
        # Radial alignment
        ri2_gt_safe = ri2_gt + eps
        vi2_gt = np.column_stack((dx2_gt / ri2_gt_safe, dy2_gt / ri2_gt_safe))
        ai2_gt = np.abs(vi2_gt[:, 0] * u2[:, 0] + vi2_gt[:, 1] * u2[:, 1])
        
        # Filter by alignment
        keep_mask2_gt = ai2_gt >= align_min
        ri2_gt_aligned = ri2_gt[keep_mask2_gt]
        w2_gt_aligned = w2[keep_mask2_gt]
        ai2_gt_aligned = ai2_gt[keep_mask2_gt]
        
        # Build histogram
        weights_hist2_gt = w2_gt_aligned * ai2_gt_aligned
        hist2_gt, _ = np.histogram(ri2_gt_aligned, bins=bins, weights=weights_hist2_gt)
        
        # Smooth histogram
        try:
            from scipy import ndimage
            hist2_gt_smooth = ndimage.gaussian_filter1d(hist2_gt.astype(np.float32), smooth_sigma)
        except ImportError:
            # Fallback: simple box filter if scipy not available
            kernel_size = int(smooth_sigma * 2) * 2 + 1
            kernel = np.ones(kernel_size) / kernel_size
            hist2_gt_smooth = np.convolve(hist2_gt.astype(np.float32), kernel, mode='same')
        
        if debug:
            if cfg.get('manual_center_enable', False) and cfg.get('manual_center_xy') is not None:
                debug_lines.append(f"\n=== Manual center histogram (center=({gt_cx:.2f}, {gt_cy:.2f}) downscaled) ===")
            else:
                gt_cx_orig, gt_cy_orig = cfg['outer_circle_ground_truth_center']
                debug_lines.append(f"\n=== Ground truth histogram (center=({gt_cx_orig}, {gt_cy_orig}) original) ===")
            peak_idx_gt = np.argmax(hist2_gt_smooth)
            r_gt = (bin_edges2[peak_idx_gt] + bin_edges2[peak_idx_gt + 1]) / 2
            debug_lines.append(f"Ground truth histogram max at bin: {peak_idx_gt}, radius: {r_gt:.2f}")
    
    # Use maximum peak from pass 2 histogram as final radius
    peak_idx = np.argmax(hist2_pass2_smooth)
    r_final = (bin_edges2[peak_idx] + bin_edges2[peak_idx + 1]) / 2
    
    # 10) Map back to original coordinates
    cx_orig = c_finalx / scale
    cy_orig = c_finaly / scale
    r_orig = r_final / scale
    
    total_time = time.time() - start_total
    log_performance(filename, "detect_outer_circle", "total", total_time)
    
    # 10) Write debug file
    if debug and len(debug_lines) > 0:
        debug_filepath = cfg.get('debug_file', 'debug_detect_outer_circle.txt')
        with open(debug_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Debug output for detect_outer_circle\n")
            f.write(f"Image: {filename}\n")
            f.write(f"Downscaled size: {new_w}×{new_h}\n")
            f.write(f"Scale factor: {scale:.4f}\n")
            f.write(f"\n")
            for line in debug_lines:
                f.write(f"{line}\n")
            f.write(f"\nPass 1 center (downscaled): ({c0x_pass1:.2f}, {c0y_pass1:.2f})\n")
            f.write(f"Pass 2 center (downscaled): ({c1x_pass2:.2f}, {c1y_pass2:.2f})\n")
            f.write(f"Final center (downscaled): ({c_finalx:.2f}, {c_finaly:.2f})\n")
            f.write(f"Final center (original): ({cx_orig:.2f}, {cy_orig:.2f})\n")
            f.write(f"Final radius (downscaled): {r_final:.2f}\n")
            f.write(f"Final radius (original): {r_orig:.2f}\n")
            f.write(f"\nPass 1 histogram max at bin: {np.argmax(hist1_pass1_smooth)}\n")
            f.write(f"Pass 2 histogram max at bin: {np.argmax(hist2_pass2_smooth)}\n")
    
    # 11) Debug output
    debug_dict = None
    if debug:
        debug_dict = {
            'downscaled_gray': gray,
            'mag': mag_raw,
            'mag_nms': mag_nms,
            'accumulator_pass1': accf_pass1,
            'accumulator_pass2': accf_pass2,
            'c0_pass1': (c0x_pass1, c0y_pass1),  # Pass 1 voting center
            'c1_pass2_vote': (c1x_pass2, c1y_pass2),  # Pass 2 voting center
            'c_final': (c_finalx, c_finaly),  # Final refined center
            'radius_histogram_pass1': hist1_pass1_smooth,  # Pass 1 histogram (with pass 1 center)
            'radius_histogram_pass2': hist2_pass2_smooth,  # Pass 2 histogram (with final center)
            'radius_histogram_pass2_before_refine': hist2_before_refine,  # Before refine (with pass 2 center)
            'radius_histogram_pass2_gt': hist2_gt_smooth,  # Ground truth histogram
            'bin_edges_pass1': bin_edges1,
            'bin_edges_pass2': bin_edges2,
            'scale': scale,
            'thinned_points': (x_thin, y_thin) if len(x_thin) > 0 else None,  # Thinned point set
            'pre_thinning_points': (x_pre, y_pre) if len(x_pre) > 0 else None  # Points before thinning
        }
    
    return (cx_orig, cy_orig, r_orig, debug_dict)


if __name__ == "__main__":
    # Start timing av hele programmet
    program_start_time = time.time()
    
    # Test med bilde fra config
    image_path = config['test_image_path']
    filename = Path(image_path).name
    
    # Tøm loggfil ved start (overskriv for å starte på nytt)
    with open(config['log_file_kronologisk'], "w", encoding="utf-8") as f:
        f.write("")  # Tøm filen
    
    # Nullstill global logget tid
    _logged_time_ms = 0.0
    
    print("Tester detect_outer_circle (Gradient-normal center voting)...")
    
    # Les bildet
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Kunne ikke lese bildet: {image_path}")
    
    # Set manual_center_xy if ground truth center is provided (convert to downscaled coords)
    if config.get('outer_circle_ground_truth_center') is not None and config.get('manual_center_enable', False):
        gt_cx_orig, gt_cy_orig = config['outer_circle_ground_truth_center']
        # We need to compute scale first to convert
        h, w = img.shape[:2]
        max_dim = max(h, w)
        target_max = config['outer_circle_max_side']
        scale = target_max / max_dim if max_dim > target_max else 1.0
        config['manual_center_xy'] = (gt_cx_orig * scale, gt_cy_orig * scale)
    
    save_visualization(img, "01_Originalt_bilde")
    
    # Kjør deteksjon
    cx, cy, r, debug_dict = detect_outer_circle(img, config, debug=True, filename=filename)
    
    print(f"Funnet ytterste sirkel: sentrum=({cx:.2f}, {cy:.2f}), radius={r:.2f}")
    
    # Visualiser resultatet
    result = img.copy()
    center = (int(cx), int(cy))
    radius = int(r)
    
    # Tegn sentrum (rød prikk) - ingen "outer circle" tegning for nå
    cv2.circle(result, center, config['circle_center_size'], config['color_red'], -1)
    
    save_visualization(result, "02_Resultat_detect_outer_circle", 2)
    
    # Vis debug-artifakter hvis tilgjengelig
    if debug_dict:
        save_visualization(debug_dict['downscaled_gray'], "03_Downscaled_graaskala", 3)
        
        # Visualiser gradient magnitude
        if 'mag' in debug_dict:
            mag = debug_dict['mag']
            # Normaliser gradient magnitude for visualisering
            mag_norm = (mag / (np.max(mag) + 1e-6) * 255).astype(np.uint8)
            save_visualization(mag_norm, "04_Gradient_magnitude", 4)
        
        # Visualiser accumulator pass 1 med peak markert (pass 1 center)
        acc_pass1 = debug_dict['accumulator_pass1']
        acc1_norm = (acc_pass1 / (np.max(acc_pass1) + 1e-6) * 255).astype(np.uint8)
        acc1_colored = cv2.applyColorMap(acc1_norm, cv2.COLORMAP_JET)
        c0_pass1 = debug_dict['c0_pass1']
        cv2.circle(acc1_colored, (int(c0_pass1[0]), int(c0_pass1[1])), 5, (255, 255, 255), 2)
        save_visualization(acc1_colored, "05_Accumulator_Pass1", 5)
        
        # Visualiser accumulator pass 2 med peak markert (pass 2 voting center)
        acc_pass2 = debug_dict['accumulator_pass2']
        acc2_norm = (acc_pass2 / (np.max(acc_pass2) + 1e-6) * 255).astype(np.uint8)
        acc2_colored = cv2.applyColorMap(acc2_norm, cv2.COLORMAP_JET)
        c1_pass2 = debug_dict['c1_pass2_vote']
        cv2.circle(acc2_colored, (int(c1_pass2[0]), int(c1_pass2[1])), 5, (255, 255, 255), 2)
        # Also mark final center if different
        c_final = debug_dict['c_final']
        if abs(c_final[0] - c1_pass2[0]) > 0.5 or abs(c_final[1] - c1_pass2[1]) > 0.5:
            cv2.circle(acc2_colored, (int(c_final[0]), int(c_final[1])), 5, (0, 255, 0), 2)  # Green for refined
        save_visualization(acc2_colored, "06_Accumulator_Pass2", 6)
        
        # Visualiser radius histogram pass 1
        if 'radius_histogram_pass1' in debug_dict:
            hist_img1 = np.zeros((200, 400), dtype=np.uint8)
            hist1 = debug_dict['radius_histogram_pass1']
            if len(hist1) > 0:
                hist1_norm = (hist1 / (np.max(hist1) + 1e-6) * 199).astype(int)
                for i, h in enumerate(hist1_norm[:400]):
                    cv2.line(hist_img1, (i, 199), (i, 199 - h), 255, 1)
            save_visualization(hist_img1, "07_Radius_histogram_Pass1", 7)
        
        # Visualiser radius histogram pass 2 (before refine, if available)
        if 'radius_histogram_pass2_before_refine' in debug_dict and debug_dict['radius_histogram_pass2_before_refine'] is not None:
            hist_img2_before = np.zeros((200, 400), dtype=np.uint8)
            hist2_before = debug_dict['radius_histogram_pass2_before_refine']
            if len(hist2_before) > 0:
                hist2_before_norm = (hist2_before / (np.max(hist2_before) + 1e-6) * 199).astype(int)
                for i, h in enumerate(hist2_before_norm[:400]):
                    cv2.line(hist_img2_before, (i, 199), (i, 199 - h), 255, 1)
            save_visualization(hist_img2_before, "08_Radius_histogram_Pass2_BeforeRefine", 8)
        
        # Visualiser radius histogram pass 2 (after refine)
        if 'radius_histogram_pass2' in debug_dict:
            hist_img2 = np.zeros((200, 400), dtype=np.uint8)
            hist2 = debug_dict['radius_histogram_pass2']
            if len(hist2) > 0:
                hist2_norm = (hist2 / (np.max(hist2) + 1e-6) * 199).astype(int)
                for i, h in enumerate(hist2_norm[:400]):
                    cv2.line(hist_img2, (i, 199), (i, 199 - h), 255, 1)
            save_visualization(hist_img2, "09_Radius_histogram_Pass2_Refined", 9)
        
        # Visualiser ground truth histogram pass 2
        if 'radius_histogram_pass2_gt' in debug_dict and debug_dict['radius_histogram_pass2_gt'] is not None:
            hist_img2_gt = np.zeros((200, 400), dtype=np.uint8)
            hist2_gt = debug_dict['radius_histogram_pass2_gt']
            if len(hist2_gt) > 0:
                hist2_gt_norm = (hist2_gt / (np.max(hist2_gt) + 1e-6) * 199).astype(int)
                for i, h in enumerate(hist2_gt_norm[:400]):
                    cv2.line(hist_img2_gt, (i, 199), (i, 199 - h), 255, 1)
            save_visualization(hist_img2_gt, "10_Radius_histogram_Pass2_GroundTruth", 10)
        
        # Visualiser punktsett før tynning (pre-thinning)
        if 'pre_thinning_points' in debug_dict and debug_dict['pre_thinning_points'] is not None:
            x_pre, y_pre = debug_dict['pre_thinning_points']
            if len(x_pre) > 0:
                gray_copy = debug_dict['downscaled_gray'].copy()
                if len(gray_copy.shape) == 2:
                    gray_copy = cv2.cvtColor(gray_copy, cv2.COLOR_GRAY2BGR)
                # Draw points as small circles
                for i in range(len(x_pre)):
                    cv2.circle(gray_copy, (int(x_pre[i]), int(y_pre[i])), 1, (0, 255, 0), -1)
                save_visualization(gray_copy, "11_Pointset_PreThinning", 11)
        
        # Visualiser punktsett etter tynning (NMS)
        if 'thinned_points' in debug_dict and debug_dict['thinned_points'] is not None:
            x_thin, y_thin = debug_dict['thinned_points']
            if len(x_thin) > 0:
                gray_copy = debug_dict['downscaled_gray'].copy()
                if len(gray_copy.shape) == 2:
                    gray_copy = cv2.cvtColor(gray_copy, cv2.COLOR_GRAY2BGR)
                # Draw points as small circles
                for i in range(len(x_thin)):
                    cv2.circle(gray_copy, (int(x_thin[i]), int(y_thin[i])), 1, (0, 0, 255), -1)
                save_visualization(gray_copy, "12_Pointset_PostThinning_NMS", 12)
    
    # Beregn total tid og uforklart tid
    program_total_time = time.time() - program_start_time
    program_total_time_ms = program_total_time * 1000
    
    # Logg uforklart tid
    log_unaccounted_time(filename, program_total_time_ms, _logged_time_ms)
    
    # Logg total programtid
    log_performance(filename, "program", "total_tid", program_total_time)
    
    # Skriv alle logg-oppføringer til fil
    flush_log_buffer()
    
    print(f"\nTotal programtid: {program_total_time_ms:.2f} ms")
    print(f"Logget tid: {_logged_time_ms:.2f} ms")
    print(f"Uforklart tid: {program_total_time_ms - _logged_time_ms:.2f} ms")
