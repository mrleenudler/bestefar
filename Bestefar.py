"""
Bestefar - Main module for circle detection pipeline.
"""

import cv2
import numpy as np
import time
from pathlib import Path

from config import DEFAULT_CONFIG, merge_config
import preprocess
import voting
import histogram
import points
import nms
import refine
import debug_tools


def build_thinned_pointset_outermost_ring(mag_nms, mag_raw, gx, gy, c_start, peak_info, cfg):
    """
    Build thinned point set for outermost ring using NMS.
    
    Args:
        mag_nms: NMS-thinned gradient magnitude
        mag_raw: Raw gradient magnitude (for pre-thinning visualization)
        gx, gy: Gradient components
        c_start: Start center (cx, cy) for radius calculation
        peak_info: List of peak info dicts from Pass 1
        cfg: Config dict
    
    Returns:
        points_refine: Dict with 'x', 'y' for refinement
        points_pre: Dict with 'x', 'y' for pre-thinning visualization
    """
    if len(peak_info) == 0:
        return None, None
    
    # Select peak with largest radius (not highest amplitude)
    outermost_peak = max(peak_info, key=lambda p: p['r_peak'])
    r_peak = outermost_peak['r_peak']
    r_lo = outermost_peak['r_lo']
    r_hi = outermost_peak['r_hi']
    
    # Expand band with padding
    pad_px = max(2.0, np.ceil(0.2 * (r_hi - r_lo)))
    r_lo2 = max(0, r_lo - pad_px)
    r_hi2 = r_hi + pad_px
    
    # Get image dimensions
    h, w = mag_raw.shape
    
    # Compute radii from start center
    y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = x_coords - c_start[0]
    dy = y_coords - c_start[1]
    ri = np.sqrt(dx**2 + dy**2)
    
    # Band mask
    band_mask = (ri >= r_lo2) & (ri <= r_hi2)
    
    # Alignment filter
    eps = 1e-6
    mag_safe = np.sqrt(gx**2 + gy**2) + eps
    ux = gx / mag_safe
    uy = gy / mag_safe
    
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
        points_pre = {'x': x_pre.astype(np.float32), 'y': y_pre.astype(np.float32)}
    else:
        points_pre = None
    
    # Post-thinning: magnitude threshold for NMS magnitude
    mag_nms_in_band = mag_nms[band_mask & (mag_nms > 0)]
    if len(mag_nms_in_band) == 0:
        return None, points_pre
    
    t_mag_percentile = cfg.get('outermost_ring_mag_percentile', 75.0)
    t_mag = np.percentile(mag_nms_in_band, t_mag_percentile)
    
    # Magnitude mask
    mag_mask = mag_nms >= t_mag
    
    # Combined mask
    keep = band_mask & mag_mask & align
    
    # Extract points
    y_pts, x_pts = np.where(keep)
    points_refine = {'x': x_pts.astype(np.float32), 'y': y_pts.astype(np.float32)}
    
    return points_refine, points_pre


def detect_outer_circle(img_bgr, cfg, debug=False, filename="unknown"):
    """
    Detekterer ytterste sirkel ved hjelp av gradient-normal center voting.
    
    Args:
        img_bgr: Input bilde (BGR)
        cfg: Konfigurasjonsdictionary
        debug: Om debug-artifakter skal returneres
        filename: Filnavn for logging
    
    Returns:
        (cx_orig, cy_orig, r_orig, debug_dict)
    """
    start_total = time.time()
    debug_lines = []
    
    # 1) Downscale -> gray -> blur -> gradients
    start_preprocess = time.time()
    img_down, scale = preprocess.downscale_max_side(img_bgr, cfg['outer_circle_max_side'])
    gray = preprocess.to_gray(img_down)
    blur = preprocess.gaussian_blur(gray, cfg['outer_circle_blur_sigma'])
    gx, gy, mag_raw, ux, uy = preprocess.compute_gradients(blur)
    new_h, new_w = gray.shape
    debug_tools.log_operation_time(filename, "detect_outer_circle", "preprocess", time.time() - start_preprocess)
    
    # 2) Pass 1: Coarse center with axis filtering
    start_pass1 = time.time()
    mag_pass1 = preprocess.suppress_axis_normals(ux, uy, mag_raw, cfg['outer_circle_filter_angle_deg'])
    points1 = points.extract_edge_points(mag_pass1, ux, uy, cfg, mode="pass1")
    if points1 is None:
        raise ValueError("Ingen edge points funnet i Pass 1")
    
    # Pass 1 voting
    cfg_pass1 = cfg.copy()
    cfg_pass1['cross_min'] = cfg['outer_circle_cross_min_pass1']
    c0_pass1, acc1, stats1 = voting.intersection_vote(
        points1, {'ux': points1['ux'], 'uy': points1['uy']}, cfg_pass1,
        image_shape=gray.shape, return_accumulator=debug
    )
    if debug:
        debug_lines.append(f"=== Pass 1 (coarse center) ===")
        debug_lines.append(f"Points: {len(points1['x'])}")
        debug_lines.append(f"Voting stats: {stats1}")
        debug_lines.append(f"Center: ({c0_pass1[0]:.2f}, {c0_pass1[1]:.2f})")
    
    # Pass 1 histogram
    hist1, bin_edges1 = histogram.build_radius_histogram(
        points1, c0_pass1, points1['mag'], {'ux': points1['ux'], 'uy': points1['uy']}, cfg
    )
    hist1_s = histogram.smooth_hist(hist1, cfg['outer_circle_r_smooth_sigma'])
    accepted_peaks = histogram.select_accepted_peaks(points1, c0_pass1, hist1_s, bin_edges1, cfg)
    debug_tools.log_operation_time(filename, "detect_outer_circle", "pass1", time.time() - start_pass1)
    
    if debug:
        debug_lines.append(f"\n=== Pass 1 histogram peaks ===")
        debug_lines.append(f"Accepted peaks: {len(accepted_peaks)}")
        for info in accepted_peaks:
            debug_lines.append(f"  Peak at r={info['r_peak']:.2f}, FWHM=[{info['r_lo']:.2f}, {info['r_hi']:.2f}], coverage={info['coverage']:.3f}")
    
    # 3) Pass 2: Precise center
    start_pass2 = time.time()
    points2_raw = points.extract_edge_points(mag_raw, ux, uy, cfg, mode="pass2")
    if points2_raw is None:
        raise ValueError("Ingen edge points funnet i Pass 2")
    
    # Filter by radius bands
    if len(accepted_peaks) == 0:
        # Fallback: use maximum
        peak_idx = np.argmax(hist1_s)
        r_peak = (bin_edges1[peak_idx] + bin_edges1[peak_idx + 1]) / 2
        accepted_peaks = [{'r_peak': r_peak, 'r_lo': r_peak - 5, 'r_hi': r_peak + 5}]
    
    mask_radius = points.filter_points_by_radius_bands(points2_raw, c0_pass1, accepted_peaks, cfg)
    mask_align = points.filter_points_by_alignment(
        points2_raw, c0_pass1, {'ux': points2_raw['ux'], 'uy': points2_raw['uy']},
        cfg['outer_circle_align_min']
    )
    mask_pass2 = mask_radius & mask_align
    
    points2 = {
        'x': points2_raw['x'][mask_pass2],
        'y': points2_raw['y'][mask_pass2],
        'ux': points2_raw['ux'][mask_pass2],
        'uy': points2_raw['uy'][mask_pass2],
        'mag': points2_raw['mag'][mask_pass2]
    }
    
    # Pass 2 voting
    cfg_pass2 = cfg.copy()
    cfg_pass2['cross_min'] = cfg['outer_circle_cross_min_pass2']
    c1_pass2, acc2, stats2 = voting.intersection_vote(
        points2, {'ux': points2['ux'], 'uy': points2['uy']}, cfg_pass2,
        image_shape=gray.shape, return_accumulator=debug
    )
    if debug:
        debug_lines.append(f"\n=== Pass 2 (precise center) ===")
        debug_lines.append(f"Points: {len(points2['x'])}")
        debug_lines.append(f"Voting stats: {stats2}")
        debug_lines.append(f"Center: ({c1_pass2[0]:.2f}, {c1_pass2[1]:.2f})")
    debug_tools.log_operation_time(filename, "detect_outer_circle", "pass2", time.time() - start_pass2)
    
    # 4) NMS + outermost ring pointset + refine
    c_final = c1_pass2
    points_refine = None
    points_pre = None
    mag_nms = None
    outermost_peak_info = None
    
    # Find outermost peak (highest peak in Pass 1 histogram)
    if len(accepted_peaks) > 0:
        # Select peak with largest radius (outermost ring)
        outermost_peak_info = max(accepted_peaks, key=lambda p: p['r_peak'])
        
        # Compute NMS only for the outermost ring band
        start_nms = time.time()
        mag_nms = nms.nms_gradient_magnitude(gx, gy, mag_raw)
        debug_tools.log_operation_time(filename, "detect_outer_circle", "nms", time.time() - start_nms)
    
    if cfg.get('outermost_ring_refine_enable', False) and mag_nms is not None and outermost_peak_info is not None:
        
        start_pointset = time.time()
        # Pass only the outermost peak (not all accepted peaks)
        points_refine, points_pre = build_thinned_pointset_outermost_ring(
            mag_nms, mag_raw, gx, gy, c_final, [outermost_peak_info], cfg
        )
        debug_tools.log_operation_time(filename, "detect_outer_circle", "build_thinned_pointset", time.time() - start_pointset)
        
        if points_refine is not None and len(points_refine['x']) > 0:
            start_refine = time.time()
            c_ref, var_ref, steps_var = refine.refine_center_radial_variance(c_final, points_refine, cfg)
            debug_tools.log_operation_time(filename, "detect_outer_circle", "radial_variance_refine", time.time() - start_refine)
            c_final = c_ref
            if debug:
                debug_lines.append(f"\n=== Outermost ring refinement (radial variance) ===")
                debug_lines.append(f"Start center: ({c1_pass2[0]:.2f}, {c1_pass2[1]:.2f})")
                debug_lines.append(f"Refined center: ({c_final[0]:.2f}, {c_final[1]:.2f})")
                debug_lines.append(f"Displacement: ({c_final[0] - c1_pass2[0]:.2f}, {c_final[1] - c1_pass2[1]:.2f})")
                debug_lines.append(f"Best variance: {var_ref:.6f}, Steps: {steps_var}")
        else:
            if debug:
                debug_lines.append(f"\n=== Outermost ring refinement (radial variance) ===")
                debug_lines.append(f"No thinned points found, skipping refinement")
    
    # 5) Final radius
    start_radius = time.time()
    hist2, bin_edges2 = histogram.build_radius_histogram(
        points2, c_final, points2['mag'], {'ux': points2['ux'], 'uy': points2['uy']}, cfg
    )
    hist2_s = histogram.smooth_hist(hist2, cfg['outer_circle_r_smooth_sigma'])
    peak_idx = np.argmax(hist2_s)
    r_final = (bin_edges2[peak_idx] + bin_edges2[peak_idx + 1]) / 2
    debug_tools.log_operation_time(filename, "detect_outer_circle", "radius_estimation", time.time() - start_radius)
    
    # 6) Map back to original
    cx_orig = c_final[0] / scale
    cy_orig = c_final[1] / scale
    r_orig = r_final / scale
    
    # 7) Debug artifacts
    debug_dict = None
    if debug:
        # Ground truth histograms (for comparison only)
        hist1_gt = None
        hist2_gt = None
        if cfg.get('manual_center_enable', False) and cfg.get('manual_center_xy') is not None:
            c_gt = cfg['manual_center_xy']
            hist1_gt, _ = histogram.build_radius_histogram(
                points1, c_gt, points1['mag'], {'ux': points1['ux'], 'uy': points1['uy']}, cfg
            )
            hist1_gt_s = histogram.smooth_hist(hist1_gt, cfg['outer_circle_r_smooth_sigma'])
            
            hist2_gt, _ = histogram.build_radius_histogram(
                points2, c_gt, points2['mag'], {'ux': points2['ux'], 'uy': points2['uy']}, cfg
            )
            hist2_gt_s = histogram.smooth_hist(hist2_gt, cfg['outer_circle_r_smooth_sigma'])
        
        debug_dict = {
            'downscaled_gray': gray,
            'downscaled_blur': blur,  # Smoothed grayscale for NMS overlay
            'mag': mag_raw,
            'mag_nms': mag_nms,  # NMS computed only for outermost ring
            'outermost_peak_info': outermost_peak_info,  # For NMS visualization
            'accumulator_pass1': acc1,
            'accumulator_pass2': acc2,
            'c0_pass1': c0_pass1,
            'c1_pass2_vote': c1_pass2,
            'c_final': c_final,
            'radius_histogram_pass1': hist1_s,
            'radius_histogram_pass2': hist2_s,
            'radius_histogram_pass1_gt': hist1_gt_s if hist1_gt is not None else None,
            'radius_histogram_pass2_gt': hist2_gt_s if hist2_gt is not None else None,
            'bin_edges_pass1': bin_edges1,
            'bin_edges_pass2': bin_edges2,
            'scale': scale,
            'accepted_peaks': accepted_peaks,
            'thinned_points': points_refine,
            'pre_thinning_points': points_pre
        }
        
        # Write debug file
        debug_file = cfg.get('debug_file', 'debug_detect_outer_circle.txt')
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(debug_lines))
            f.write(f"\n\nFinal center (downscaled): ({c_final[0]:.2f}, {c_final[1]:.2f})")
            f.write(f"\nFinal center (original): ({cx_orig:.2f}, {cy_orig:.2f})")
            f.write(f"\nFinal radius (downscaled): {r_final:.2f}")
            f.write(f"\nFinal radius (original): {r_orig:.2f}")
    
    total_time = time.time() - start_total
    debug_tools.log_operation_time(filename, "detect_outer_circle", "total", total_time)
    
    return (cx_orig, cy_orig, r_orig, debug_dict)


if __name__ == "__main__":
    # CLI / Debug runner
    import sys
    
    # Load config
    config = DEFAULT_CONFIG.copy()
    
    # Override test image if provided
    if len(sys.argv) > 1:
        config['test_image_path'] = Path(sys.argv[1])
    
    image_path = config['test_image_path']
    filename = image_path.stem
    
    # Clear log file
    log_file = config['log_file_kronologisk']
    Path(log_file).write_text("", encoding='utf-8')
    debug_tools.reset_visualization_index()
    
    # Set manual center if ground truth is provided
    if config.get('outer_circle_ground_truth_center') is not None and config.get('manual_center_enable', False):
        # Read image to compute scale
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            max_dim = max(h, w)
            target_max = config['outer_circle_max_side']
            scale = target_max / max_dim if max_dim > target_max else 1.0
            gt_cx_orig, gt_cy_orig = config['outer_circle_ground_truth_center']
            config['manual_center_xy'] = (gt_cx_orig * scale, gt_cy_orig * scale)
    
    program_start_time = time.time()
    
    print("Tester detect_outer_circle (Gradient-normal center voting)...")
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Kunne ikke lese bildet: {image_path}")
    
    debug_tools.save_visualization(img, "01_Originalt_bilde", 1, config['visualization_dir'])
    
    # Run detection
    cx, cy, r, debug_dict = detect_outer_circle(img, config, debug=True, filename=filename)
    
    print(f"Funnet ytterste sirkel: sentrum=({cx:.2f}, {cy:.2f}), radius={r:.2f}")
    
    # Visualize result
    result = img.copy()
    center = (int(cx), int(cy))
    cv2.circle(result, center, config['circle_center_size'], config['color_red'], -1)
    debug_tools.save_visualization(result, "02_Resultat_detect_outer_circle", 2, config['visualization_dir'])
    
    # Visualize debug artifacts
    if debug_dict:
        debug_tools.save_visualization(debug_dict['downscaled_gray'], "03_Downscaled_graaskala", 3, config['visualization_dir'])
        
        # Gradient magnitude
        if 'mag' in debug_dict:
            mag = debug_dict['mag']
            mag_norm = (mag / (np.max(mag) + 1e-6) * 255).astype(np.uint8)
            debug_tools.save_visualization(mag_norm, "04_Gradient_magnitude", 4, config['visualization_dir'])
        
        # NMS overlay (green on smoothed grayscale) - only for outermost ring FWHM band
        if 'mag_nms' in debug_dict and debug_dict['mag_nms'] is not None and 'outermost_peak_info' in debug_dict and debug_dict['outermost_peak_info'] is not None:
            mag_nms = debug_dict['mag_nms']
            peak_info = debug_dict['outermost_peak_info']
            blur_img = debug_dict.get('downscaled_blur', debug_dict['downscaled_gray'])
            base = cv2.cvtColor(blur_img, cv2.COLOR_GRAY2BGR)
            
            # Get FWHM band for outermost ring
            r_lo = peak_info['r_lo']
            r_hi = peak_info['r_hi']
            pad_px = max(2.0, np.ceil(0.2 * (r_hi - r_lo)))
            r_lo2 = max(0, r_lo - pad_px)
            r_hi2 = r_hi + pad_px
            
            # Get center (use c0_pass1 for consistency with Pass 1 histogram)
            c0 = debug_dict['c0_pass1']
            
            # Compute radii from center
            h, w = blur_img.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w].astype(np.float32)
            dx = x_coords - c0[0]
            dy = y_coords - c0[1]
            ri = np.sqrt(dx**2 + dy**2)
            
            # Band mask (FWHM band of outermost ring)
            band_mask = (ri >= r_lo2) & (ri <= r_hi2)
            
            # NMS points within the band only
            m = (mag_nms > 0) & band_mask
            
            # Set NMS pixels to green (BGR)
            base[m] = (0, 255, 0)
            debug_tools.save_visualization(base, "09_NMS_overlay_green", 9, config['visualization_dir'])
        
        # Accumulators
        acc1 = debug_dict['accumulator_pass1']
        if acc1 is not None:
            acc1_norm = (acc1 / (np.max(acc1) + 1e-6) * 255).astype(np.uint8)
            acc1_colored = cv2.applyColorMap(acc1_norm, cv2.COLORMAP_JET)
            c0 = debug_dict['c0_pass1']
            cv2.circle(acc1_colored, (int(c0[0]), int(c0[1])), 5, (255, 255, 255), 2)
            debug_tools.save_visualization(acc1_colored, "05_Accumulator_Pass1", 5, config['visualization_dir'])
        
        acc2 = debug_dict['accumulator_pass2']
        if acc2 is not None:
            acc2_norm = (acc2 / (np.max(acc2) + 1e-6) * 255).astype(np.uint8)
            acc2_colored = cv2.applyColorMap(acc2_norm, cv2.COLORMAP_JET)
            c1 = debug_dict['c1_pass2_vote']
            cv2.circle(acc2_colored, (int(c1[0]), int(c1[1])), 5, (255, 255, 255), 2)  # White for vote center
            c_final = debug_dict['c_final']
            if abs(c_final[0] - c1[0]) > 0.5 or abs(c_final[1] - c1[1]) > 0.5:
                cv2.circle(acc2_colored, (int(c_final[0]), int(c_final[1])), 5, (0, 255, 0), 2)  # Green for final center
            debug_tools.save_visualization(acc2_colored, "06_Accumulator_Pass2_vote_white_final_green", 6, config['visualization_dir'])
        
        # Histograms
        def draw_histogram(hist, max_len=400):
            hist_img = np.zeros((200, max_len), dtype=np.uint8)
            if len(hist) > 0:
                hist_norm = (hist / (np.max(hist) + 1e-6) * 199).astype(int)
                for i, h in enumerate(hist_norm[:max_len]):
                    cv2.line(hist_img, (i, 199), (i, 199 - h), 255, 1)
            return hist_img
        
        debug_tools.save_visualization(draw_histogram(debug_dict['radius_histogram_pass1']), "07_Radius_histogram_Pass1", 7, config['visualization_dir'])
        debug_tools.save_visualization(draw_histogram(debug_dict['radius_histogram_pass2']), "09_Radius_histogram_Pass2_Refined", 9, config['visualization_dir'])
        
        if debug_dict.get('radius_histogram_pass2_gt') is not None:
            debug_tools.save_visualization(draw_histogram(debug_dict['radius_histogram_pass2_gt']), "10_Radius_histogram_Pass2_GroundTruth", 10, config['visualization_dir'])
        
        # Point sets
        if debug_dict.get('pre_thinning_points') is not None:
            overlay = debug_tools.create_pointset_overlay(debug_dict['downscaled_gray'], debug_dict['pre_thinning_points'], (0, 255, 0))
            debug_tools.save_visualization(overlay, "11_Pointset_PreThinning", 11, config['visualization_dir'])
        
        if debug_dict.get('thinned_points') is not None:
            overlay = debug_tools.create_pointset_overlay(debug_dict['downscaled_gray'], debug_dict['thinned_points'], (0, 0, 255))
            debug_tools.save_visualization(overlay, "12_Pointset_PostThinning_NMS", 12, config['visualization_dir'])
    
    # Flush logs
    debug_tools.flush_log_buffer(log_file)
    
    # Log total time
    program_total_time = time.time() - program_start_time
    program_total_time_ms = program_total_time * 1000
    debug_tools.log_unaccounted_time(filename, program_total_time_ms, debug_tools._logged_time_ms)
    
    print(f"\nTotal programtid: {program_total_time_ms:.2f} ms")
    print(f"Logget tid: {debug_tools._logged_time_ms:.2f} ms")
    print(f"Uforklart tid: {program_total_time_ms - debug_tools._logged_time_ms:.2f} ms")

