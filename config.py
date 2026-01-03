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
    
    # NMS and radial variance refinement
    'outermost_ring_refine_enable': False,       # Enable radial variance refinement on outermost ring
    'outermost_ring_mag_percentile': 75.0,       # Percentile for magnitude threshold in outermost ring
    'outermost_ring_refine_max_steps': 60,       # Max steps for radial variance refinement
    'outermost_ring_refine_step_px': 1,          # Step size for refinement
    'outermost_ring_refine_max_radius_px': 12,   # Max radius from start for refinement
    
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

