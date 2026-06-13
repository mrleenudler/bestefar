"""
Configuration module for Bestefar circle detection.
"""

from pathlib import Path


DEFAULT_CONFIG = {
    # Filer og kataloger
    'visualization_dir': Path("Visualiseringer"),
    'log_file_kronologisk': "ytelse.txt",
    'debug_file': "debug_detect_outer_circle.txt",
    
    # Visualisering - Farger (BGR format)
    'color_green': (0, 255, 0),       # Grønn for contours/sirkler
    'color_red': (0, 0, 255),         # Rød for sentrum
    'color_blue': (255, 255, 0),      # Cyan for tekst
    'color_magenta': (255, 0, 255),   # Magenta for hoved-sentrum
    
    # Visualisering - Tegneparametere
    'circle_thickness': 2,            # Tykkelse på sirkler
    'circle_center_size': 5,          # Størrelse på sentrum-markør
    
    # Visualisering - Tekst
    'font': 0,  # cv2.FONT_HERSHEY_SIMPLEX
    'font_scale_medium': 0.7,         # Font-størrelse for radius-tekst
    'font_thickness': 2,              # Font-tykkelse
    
    # Test-bilde (i main)
    'test_image_path': Path("Real 1.jpg"),
    
    # Outer circle detection - Preprocessing
    'outer_circle_max_side': 1200,              # Maks side i px for downscaling
    'outer_circle_blur_sigma': 2.0,             # Gaussian blur sigma
    'outer_circle_mag_floor': 1e-3,             # Minimum gradient magnitude
    'outer_circle_mag_percentile': 60.0,        # Percentile for edge threshold
    'outer_circle_filter_angle_deg': 2.0,       # Filter gradients within ±N degrees of horizontal/vertical
    'outer_circle_max_edge_points': 12000,      # Maks antall edge points
    'outer_circle_border_margin_frac': 0.05,    # Drop edge points near image border
    
    # Voting parameters
    'outer_circle_center_pairs': 40000,          # Antall random par for intersection-of-normals voting
    'outer_circle_parallel_eps': 0.15,           # Reject pairs if abs(cross) < this (for numerical stability)
    'outer_circle_max_center_distance_frac': 0.8,  # Max distance from image center (None to disable)
    'outer_circle_center_win': 5,                # Halvbredde for subpixel center refinement (win=5 => 11×11 vindu)
    'outer_circle_cross_min_pass1': 0.30,        # Minimum cross product for pass 1 (quality filter)
    'outer_circle_cross_min_pass2': 0.40,        # Minimum cross product for pass 2 (quality filter)
    
    # Histogram and peak detection
    'outer_circle_align_min': 0.7,              # Minimum radial alignment
    'outer_circle_r_bin_px': 1.0,               # Bin width for radius histogram
    'outer_circle_r_smooth_sigma': 2.0,         # Smoothing sigma for radius histogram
    'outer_circle_peaks_keep': 10,              # Antall topper å beholde
    'outer_circle_peak_score_min_frac': 0.2,    # Minimum score fraction for peak
    'outer_circle_cov_bins': 120,                # Number of theta-bins for coverage
    'outer_circle_cov_min_frac': 0.65,           # Min coverage fraction to accept ring
    'outer_circle_fwhm_min_bins': 2,            # Minimum number of radius-bins in FWHM band
    'outer_circle_peaks_cluster_px': 3.0,       # Merge peaks closer than this (double edge)
    'outer_circle_pass2_outer_cut_eps': 2.0,     # Cut points with r > r_outer + eps
    
    # Polar transform (debug visualization only)
    'polar_debug_enable': True,                   # Enable polar transform visualization
    'polar_r_max_frac': 1.10,                     # Max radius as fraction of outer circle radius (or min(half_width, half_height))
    'polar_theta_samples': 720,                  # Columns (angle resolution)
    'polar_r_samples': 600,                       # Rows (radius resolution)
    'polar_interpolation': 'linear',              # 'nearest' or 'linear'
    
    # Polar hysteresis debug prototype (erstattet av rings.py multiring-raffinering;
    # kan slås på for debug, men er treg og mindre presis)
    'polar_hyst_debug_enable': False,           # Enable polar hysteresis debug prototype
    'polar_band_halfwidth_px': 20,              # Half-bandwidth in pixels for radius band (±20px in radius direction)
    'polar_ridge_start_row': 0,                 # Starting row for ridge tracking (0 = top)
    # Viterbi/DP ridge tracking parameters
    'polar_viterbi_smooth_weight': 1.0,         # λ_smooth: penalty for jumps between rows (higher = smoother)
    'polar_viterbi_curve_weight': 0.5,          # λ_curve: penalty for curvature/drift (higher = straighter)
    # Legacy parameters (kept for backward compatibility, not used in ridge tracking)
    'polar_band_width': 10,                       # Radius band width around peak (in bins) - deprecated
    'polar_hyst_high_percentile': 98.0,          # High threshold percentile for hysteresis - deprecated
    'polar_hyst_low_frac': 0.5,                  # Low threshold = low_frac * high threshold - deprecated
    
    # NMS and radial variance refinement
    'outermost_ring_refine_enable': False,       # Enable radial variance refinement on outermost ring
    'outermost_ring_mag_percentile': 75.0,       # Percentile for magnitude threshold in outermost ring (pre-thinning visualization only)
    'outermost_ring_hyst_high_percentile': 70.0, # Strong threshold percentile for hysteresis (within band) - lower than pre-thinning since mag_nms is already thinned
    'outermost_ring_hyst_low_frac': 0.40,       # Low threshold = low_frac * high threshold
    'outermost_ring_hyst_min_pixels': 200,      # Minimum pixels in edge_mask, else fallback
    'outermost_ring_refine_max_steps': 60,       # Max steps for radial variance refinement
    'outermost_ring_refine_step_px': 1,          # Step size for refinement
    'outermost_ring_refine_max_radius_px': 12,   # Max radius from start for refinement
    
    # Ringkalibrering og senterraffinering (rings.py)
    'ring_calib_enable': True,
    'ring_theta_samples': 720,            # Vinkelsamples i polartransform
    'ring_rho_sigma': 3.0,                # Glatting langs radius (orig-px) - smelter sammen dobbeltkanter
    'ring_r_max_frac': 1.06,              # r_max = frac * estimert ytterring-radius
    'ring_peak_min_frac': 0.12,           # Min topphøyde i radialprofil (andel av max)
    'ring_peak_min_sep_frac': 0.4,        # Min toppavstand (andel av ringavstand)
    'ring_fit_max_resid_frac': 0.06,      # Maks residual i progresjonsfit (andel av delta).
                                          # Må romme systematikk fra linsedistorsjon/strekpolaritet (~+-3-4px),
                                          # men forkaste støytopper i markørklyngen (>30px avvik)
    'ring_value_outermost': 1,            # Poengverdi for ytterste synlige ring
    'ring_refine_iters': 4,               # Iterasjoner for senterraffinering
    'ring_refine_min_coverage': 0.5,      # Min andel gyldige vinkler per ring
    'ring_harm_reject_rounds': 3,         # Outlier-rejection-runder i harmonisk fit
    'ring_harm_reject_sigma': 2.5,        # Outlier-terskel (sigma)

    # Treffdeteksjon (hits.py)
    'hit_marker_radius_frac': 0.41,       # Markørradius som andel av ringavstand (80px dia / 98px delta)
    'hit_dot_radius_frac': 0.105,         # Senterprikk-radius som andel av ringavstand
    'hit_search_r_max_frac': 1.05,        # Søk innenfor frac * R1
    'hit_hough_dp': 2.0,                  # Hough akkumulator-nedskalering
    'hit_hough_param1': 120,              # Canny høy terskel for Hough
    'hit_hough_param2': 30,               # Hough akkumulator-terskel
    'hit_min_dist_frac': 0.6,             # Min senteravstand mellom treff (andel av markørradius)
    'hit_validate_min_score': 0.15,       # Min mønster-score for å godta kandidat

    # Kvalitetsport: forkast bilder uten gyldig poengskive (rings.validate_calibration)
    'gate_min_delta_px': 25.0,            # Min ringavstand i px (ekte foto: 60-115; støy: <20)
    'gate_min_rings': 4,                  # Min antall tilpassede ringer (4 jevnt fordelte
                                          # konsentriske ringer er allerede en sterk signatur;
                                          # skråfoto kan mangle de svakeste ringene før retting)
    'gate_max_resid_frac': 0.12,          # Maks avvik fra jevn ringfordeling (andel av ringavstand)
    'gate_require_hits': True,            # Forkast bilder der ingen treff blir funnet

    # Perspektivretting (perspektiv.py)
    'persp_rectify_enable': True,         # Rett opp perspektiv før treff/poeng
    'persp_n_theta': 180,                 # Punkter per ring i homografi-tilpasningen
    'ring_recal_iters': 3,                # Kalibreringsiterasjoner etter retting

    # Poengberegning (scoring.py)
    'score_use_local_rings': True,        # Bruk lokal ringgeometri ved treffets vinkel (perspektiv)
    'score_gauge_frac_of_delta': 0.14,    # Poenggrense utenfor ringstrek-SENTERLINJEN: ringstreken tegnes
                                          # innenfor sonen, så grensen er strekens ytterkant (~halv strekbredde,
                                          # ~5px) + halv kaliber (~9px). Kalibrert mot displayverdiene i Real 1;
                                          # bør etterprøves med flere bilder
    'score_round_mode': 'truncate',       # 'truncate' (offisiell) eller 'nearest'

    # Ground truth (for debug only, does not affect algorithm)
    'manual_center_enable': True,                # Enable manual center for comparison
    'manual_center_xy': None,                    # Manual center (cx, cy) in downscaled coords (None to disable)
    'outer_circle_ground_truth_center': (1093, 1940),  # Ground truth center (cx, cy) in original image coords
}


def merge_config(overrides: dict) -> dict:
    """
    Merge override dictionary into default config.
    
    Args:
        overrides: Dictionary with config overrides
    
    Returns:
        Merged config dictionary
    """
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config

