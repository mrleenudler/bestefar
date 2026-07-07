// Konfigurasjon for CV-kjernen. Portert fra config.py — verdiene her ER
// referanseverdiene; endre dem kun synkront med Python-fasiten.
// Kun parametre som leses av den porterte (produksjons-)stien er med:
// død-kode-stier (Hough-treff, ring_comb_v2, outermost_ring_refine) er utelatt.
#pragma once

namespace bestefar {

struct Config {
    // ---- Ytre sirkel (gradient-voting, Bestefar.py) ----
    int    outer_circle_max_side              = 1200;
    double outer_circle_blur_sigma            = 2.0;
    double outer_circle_mag_floor             = 1e-3;
    double outer_circle_mag_percentile        = 60.0;
    double outer_circle_filter_angle_deg      = 2.0;
    // AVVIK FRA PYTHON-REFERANSEN (bevisst, se docs/ARCHITECTURE.md):
    // Python subsampler til 12000 punkter (ytelses-tak) og bruker 40000 stemme-par.
    // Kontrollerte forsoek (C4-crop) viste at 12k-subsamplet er et LOTTERI: senter-
    // estimatet svinger +-10px mellom trekk, og Python-referansens numpy-seed
    // tilfeldigvis vinner paa alle C-bildene (Pythons egen vote paa et annet trekk
    // bommet med 13px). Vi fjerner lotteriet: bruk ALLE ekstraherte punkter
    // (max_edge_points=0) og skaler par-antallet opp. Samme estimator, tettere
    // sampling, konvergerer mot populasjonsverdien. Verifisert mot C-settet.
    int    outer_circle_max_edge_points       = 0;      // 0 = ingen subsampling (Python: 12000)
    double outer_circle_border_margin_frac    = 0.05;

    int    outer_circle_center_pairs          = 1000000; // Python: 40000 (se merknad over)
    double outer_circle_parallel_eps          = 0.15;
    double outer_circle_max_center_distance_frac = 0.8;   // <0 = av
    int    outer_circle_center_win            = 5;
    double outer_circle_cross_min_pass1       = 0.30;
    double outer_circle_cross_min_pass2       = 0.40;

    double outer_circle_align_min             = 0.7;
    double outer_circle_r_bin_px              = 1.0;
    double outer_circle_r_smooth_sigma        = 2.0;
    double outer_circle_peak_score_min_frac   = 0.2;
    int    outer_circle_cov_bins              = 120;
    double outer_circle_cov_min_frac          = 0.65;
    int    outer_circle_fwhm_min_bins         = 2;
    double outer_circle_peaks_cluster_px      = 3.0;
    double outer_circle_pass2_outer_cut_eps   = 2.0;

    // ---- Ringkalibrering (rings.py) ----
    int    ring_theta_samples        = 720;
    double ring_rho_sigma            = 1.2;   // fjernes den, faller C3-kalibrering
    double ring_r_max_frac           = 1.06;
    double ring_peak_min_frac        = 0.12;
    double ring_peak_min_sep_frac    = 0.4;
    double ring_fit_max_resid_frac   = 0.06;
    int    ring_value_outermost      = 1;
    bool   ring_anchor_R10_eq_delta  = true;
    double ring_R10_over_delta       = 1.0;
    bool   ring_generous_rmax        = true;
    double ring_generous_mult        = 2.6;
    double ring_generous_edge_frac   = 0.95;
    double ring_pre_blur_sigma       = 1.5;
    int    ring_refine_iters         = 4;
    double ring_refine_min_coverage  = 0.5;
    int    ring_harm_reject_rounds   = 3;
    double ring_harm_reject_sigma    = 2.5;
    int    ring_recal_iters          = 3;     // etter perspektivretting

    // ---- Treffdeteksjon (hits.py, kun enhetlig detektor-sti) ----
    double hit_marker_radius_frac    = 0.41;
    double hit_dot_radius_frac       = 0.105;
    double hit_search_r_max_frac     = 1.05;
    double hit_min_dist_frac         = 0.6;
    double hit_unified_inner_frac    = 0.0;

    // ---- Stemmekart-detektor (circles.py) ----
    double circ_pre_gauss_frac       = 0.10;
    double circ_median_frac          = 0.18;
    int    circ_median_passes        = 1;
    double circ_edge_pct             = 70.0;
    int    circ_n_radii              = 5;
    double circ_radial_reject        = 0.80;  // NOEDVENDIG (bevist med sweep)
    double circ_center_exempt_frac   = 0.0;   // 1.5 lagde falsk 10-er paa ALLE bilder
    double circ_acc_blur_frac        = 0.06;  // 0.12 smeltet nabotopper
    double circ_nms_frac             = 0.7;
    double circ_peak_min_frac        = 0.45;
    double circ_min_votes_frac       = 0.30;
    double circ_dot_vote_weight      = 1.0;

    // ---- Overlapp-pass + sentrum-sveip (overlap.py) ----
    bool   hit_overlap_pass          = true;
    int    hit_overlap_trigger       = 10;
    double hit_overlap_max_anchor_r_frac = 3.0;
    double hit_overlap_nms_frac      = 0.20;
    double hit_overlap_vote_frac     = 0.25;
    double hit_overlap_max_dist_frac = 1.0;
    double hit_overlap_min_offset_frac = 0.20;
    double hit_overlap_tmpl_r_lo     = 0.20;
    double hit_overlap_tmpl_r_hi     = 1.05;
    // NCC-terskler REKALIBRERT til C++-implementasjonens maalte fordeling
    // (maskert TM_CCOEFF_NORMED-realisering avviker fra Python-referansens:
    // annen template-kilde/ramme/OpenCV-versjon). Maalt paa C-settet:
    //   overlapp: ekte delvis-skjult (C3 10.6) = 0.47, falsk topp (C10) = 0.12
    //             (Python: ekte ~0.15, terskel 0.10)
    //   sentrum:  ekte sentertreff (C10) = 0.95, falsk (C1) = 0.40
    //             (Python: ekte ~0.90, FP ~0.18, terskel 0.40)
    double hit_overlap_ncc_thresh    = 0.25;   // Python: 0.10
    bool   hit_center_scan           = true;
    double hit_center_scan_r_frac    = 1.0;
    double hit_center_ncc_thresh     = 0.60;   // Python: 0.40

    // ---- Kvalitetsport (rings.validate_calibration) ----
    double gate_min_delta_px         = 25.0;
    int    gate_min_rings            = 4;
    double gate_max_resid_frac       = 0.12;
    bool   gate_require_hits         = true;

    // ---- Skjermdeteksjon (screen.py) ----
    bool   screen_rectify_enable     = true;
    int    screen_work_size          = 480;
    double screen_stretch_pct        = 1.0;
    double screen_blur_sigma         = 3.5;
    double screen_low_frac           = 0.55;
    double screen_blob_open_frac     = 0.0;   // av
    double screen_side_outer_tol_frac = 0.0;  // av
    bool   screen_blob_convex_hull   = false; // DOED: streng nedgradering
    double screen_min_area_frac      = 0.10;
    bool   analyze_screen_fallback   = false;
    bool   screen_use_contrast_roi   = true;
    double screen_contrast_win_frac  = 0.04;
    double screen_roi_open_frac      = 0.025;
    int    screen_roi_bright_min     = 190;
    double screen_roi_dilate_frac    = 0.06;
    double screen_side_inlier_tol_frac = 0.025;
    double screen_side_min_inlier_frac = 0.15;
    double screen_side_min_span_frac = 0.60;
    bool   screen_refine_gradient_lines = true;  // kant-snapping (C10-fiksen)
    int    screen_snap_iters         = 4;
    double screen_snap_band_frac     = 0.045;
    double screen_snap_dir_cos       = 0.5;
    double screen_snap_end_skip_frac = 0.08;
    double screen_snap_strong_frac   = 0.6;

    // ---- Perspektivretting (perspektiv.py) ----
    bool   persp_rectify_enable      = true;
    int    persp_n_theta             = 180;

    // ---- Poengberegning (scoring.py) ----
    bool   score_use_local_rings     = true;
    double score_gauge_frac_of_delta = 0.14;
    bool   score_truncate            = true;   // 'truncate' (offisiell regel)
};

// Auto-capture-parametre (kravspec §4). Startverdiene under er FOERSTE
// KALIBRERINGSPASS (2026-07-07, felttest paa telefon): de opprinnelige
// tersklene gjorde kvalitetskriteriet nesten uoppnaaelig —
//   min_coverage=0.98 kolliderte med ROI-ens egen 6%-dilatasjon (boksen
//   naar nesten alltid 3%-marginen), og klipp-takene var for stramme for
//   en lys hvit skjerm. Verdiene skal fortsatt finkalibreres mot maalte
//   probe-verdier (live-overlay paa capture-skjermen viser dem).
struct AutoCaptureParams {
    int    stability_frames        = 8;     // antall paafoelgende frames
    double min_sharpness           = 40.0;  // Laplacian-varians i ROI (<40 = reelt uskarpt)
    double stability_max_move_frac = 0.01;  // maks hjoerneflytt per frame (andel av diagonal)
    double max_clip_lo_frac        = 0.30;  // maks andel piksler i moerkeste bin
    double max_clip_hi_frac        = 0.20;  // maks andel piksler i lyseste bin
    double min_coverage            = 0.80;  // andel av ROI-boks innenfor frame m/margin
    double frame_margin_frac       = 0.02;  // margin mot framekant
    int    probe_max_side          = 480;   // arbeidsopploesning for FrameProbe
};

} // namespace bestefar
