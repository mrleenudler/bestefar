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
    'ring_rho_sigma': 1.2,                # Lett histogram-glatting. RESULTAT: glattingen smelter IKKE dobbeltkantene
                                          # (de er ekte geometri), men denoiser marginale kalibreringer - uten den
                                          # faller C3 ut ("fant ingen poengringer"). Tyngre (>=1.5) er ikke-monoton/skjor.
    'ring_r_max_frac': 1.06,              # r_max = frac * estimert ytterring-radius
    'ring_peak_min_frac': 0.12,           # Min topphøyde i radialprofil (andel av max)
    'ring_peak_min_sep_frac': 0.4,        # Min toppavstand (andel av ringavstand)
    'ring_fit_max_resid_frac': 0.06,      # Maks residual i progresjonsfit (andel av delta).
                                          # Må romme systematikk fra linsedistorsjon/strekpolaritet (~+-3-4px),
                                          # men forkaste støytopper i markørklyngen (>30px avvik)
    'ring_value_outermost': 1,            # Poengverdi for ytterste synlige ring (før forankring)
    'ring_anchor_R10_eq_delta': True,     # Forankre absolutte ringverdier via R10 ~= delta (ring 10 ~1 avstand fra senter)
    'ring_R10_over_delta': 1.0,           # Forventet R10/delta-forhold for DFS-skive
    # Generøs r_max (ta med hvite ytre ringer) + pre-blur (demp moiré/dobbeltkant) - VINNERKONFIG
    'ring_generous_rmax': True,           # r_max dekker hele skiva, ikke bare grov-radius
    'ring_generous_mult': 2.6,            # r_max = mult * grov-radius ...
    'ring_generous_edge_frac': 0.95,      # ... begrenset til frac * avstand til nærmeste bildekant
    'ring_pre_blur_sigma': 1.5,           # Pre-blur av gråbilde før polar (fikset bl.a. T6-oktaven)
    # Eksperimentell prominens-comb (regredierte pga dobbeltkant-oktav; ikke i bruk)
    'ring_comb_v2': False,                # Bruk fit_equidistant_comb (prominens + ekvidistant + tak)
    'comb_max_rings': 11,                 # Tak på antall ringer (forbyr 2x-oktav)
    'comb_min_rings': 4,                  # Min antall ringer i comb
    'comb_smooth_frac': 0.02,             # Glatting av profil (andel av radius) - slår sammen dobbeltkanter/skjev-topper
    'comb_prom_frac': 0.04,               # Min topp-prominens (andel av maks) for kandidat
    'comb_min_dist_bins': 6,              # Min avstand mellom topper (bins, absolutt gulv)
    'comb_min_dist_frac': 0.35,           # Min toppavstand som andel av grov ringavstand (slår sammen dobbeltkanter)
    'comb_inner_ignore_frac': 0.04,       # Ignorer topper innenfor denne andelen av radius (senterklynge)
    'comb_strong_frac': 0.5,              # "Sterk topp" = prominens >= frac * maks-prominens
    'comb_delta_min_bins': 25.0,          # Min ringavstand (bins)
    'comb_tol_frac': 0.18,                # Toleranse tann<->topp (andel av delta)
    'comb_empty_penalty': 0.5,            # Straff per tom tann (for-liten avstand)
    'comb_skip_penalty': 1.0,             # Straff per hoppet sterk topp (oktav-for-stor)
    'ring_refine_iters': 4,               # Iterasjoner for senterraffinering
    'ring_refine_min_coverage': 0.5,      # Min andel gyldige vinkler per ring
    'ring_harm_reject_rounds': 3,         # Outlier-rejection-runder i harmonisk fit
    'ring_harm_reject_sigma': 2.5,        # Outlier-terskel (sigma)

    # Treffdeteksjon (hits.py)
    'hit_marker_radius_frac': 0.41,       # Markørradius som andel av ringavstand (80px dia / 98px delta)
    'hit_dot_radius_frac': 0.105,         # Senterprikk-radius som andel av ringavstand
    'hit_search_r_max_frac': 1.05,        # Søk innenfor frac * ytterste poengring (comb: R10+9*delta)
    'hit_hough_dp': 2.0,                  # Hough akkumulator-nedskalering
    'hit_hough_param1': 120,              # Canny høy terskel for Hough
    'hit_hough_param2': 30,               # Hough akkumulator-terskel
    'hit_min_dist_frac': 0.6,             # Min senteravstand mellom treff (andel av markørradius)
    'hit_validate_min_score': 0.15,       # Min mønster-score for å godta kandidat

    # Hvit-sone-pass (circles.py): maalrettet sirkeldetektor for svake graa markorer
    # i lyssonen som Hough ikke foreslaar. Kjorer i ringen utenfor bull-sonen.
    'hit_white_zone_pass': True,
    'hit_white_inner_frac': 3.4,          # Indre grense (x delta): innenfor = bull (Hough-pass)
    'hit_white_blur_frac': 0.12,          # Egen acc-blur for (superseded) hvit-sone-passet, slik at
                                          # det fortsatt virker naar enhetlig detektor slaas av

    # Enhetlig detektor: kjor circles.py-stemmedetektoren over HELE skiva (bull +
    # hvit sone) og dropp HoughCircles helt. Naar True ignoreres Hough- og det
    # separate hvit-sone-passet. Eksperimentell - mal med verify_cset.py.
    'hit_unified_detector': True,
    'hit_unified_inner_frac': 0.0,        # Indre eksklusjon (x delta) for doed senter; 0 = hele skiva
    'circ_pre_gauss_frac': 0.10,          # Gauss-forblur foer median (x markor_r). Dreper LED-
                                          # rutenett godt nok til at 1x median holder; uten gauss
                                          # trengtes 2x median, men det produserte 2 falske positiver
                                          # i C3-klyngen. g10 gir sterkere peaks enn g06 (valgt).
    'circ_median_frac': 0.18,             # Median-kjerne (x markor_r) mot LED-rutenett-stoy
    'circ_median_passes': 1,              # 1x median etter gauss er nok (2x uten gauss ga 2 FP i C3)
    'circ_edge_pct': 70.0,                # Persentil-terskel for kant-piksler (lav: kompletthet siler)
    'circ_n_radii': 5,                    # Antall stemme-radier rundt markor-skiven
    'circ_radial_reject': 0.80,           # Forkast kanter med |cos| > dette mot maalsenter (poengringer)
    'circ_center_exempt_frac': 0.0,       # Naer senteret (x markor_r) unntas retnings-gating.
                                          # 0 = ingen unntak: ellers konvergerer de konsentriske
                                          # poengringenes radielle stemmer i maalsenteret og lager
                                          # en garantert falsk 10-er (enhetlig detektor). Paavirker
                                          # ikke hvit-sone-passet (det ekskluderer <3.4*delta uansett).
    'circ_acc_blur_frac': 0.06,           # Glatting av stemmekartet (x markor_r). Lav: 0.12
                                          # smeltet sammen nabotopper i tette klynger (C1/C3/C8
                                          # under-detekterte); 0.06 skiller overlappende treff.
    'circ_nms_frac': 0.7,                 # NMS-radius (x markor_r)
    'circ_peak_min_frac': 0.45,           # Relativ topp-terskel (andel av maks)
    'circ_min_votes_frac': 0.30,          # Absolutt stemme-gulv (x markor_r) - kompletthet-krav
    'circ_dot_vote_weight': 1.0,          # Vekt paa prikk-stemmer (x skive-stemmer). >1 forankrer
                                          # toppen paa det enkelte skuddets senterprikk i tette klynger

    # Overlapp-pass (overlap.py): subterskel stemmekart + NCC for delvis-skjulte treff
    'hit_overlap_pass': True,
    'hit_overlap_trigger': 10,            # Kjoer kun hvis < dette antall treff funnet
    'hit_overlap_max_anchor_r_frac': 3.0, # Soek kun naer treff innenfor dette (x delta) fra senter
    'hit_overlap_nms_frac': 0.20,         # Liten NMS-radius (x marker_r); std ~0.7 undertrykket 22px-nabo
    'hit_overlap_vote_frac': 0.25,        # Subterskel-gulv (andel av amax) for overlapp-kandidater
    'hit_overlap_max_dist_frac': 1.0,     # Maks avstand fra anker (x marker_r) = overlapp-definisjon
    'hit_overlap_min_offset_frac': 0.20,  # Min avstand fra anker (x marker_r) for aa ikke duplisere
    'hit_overlap_tmpl_r_lo': 0.20,        # Indre grense paa halvt NCC-template (x marker_r).
                                          # Halvmaske inkluderer graa disc-interior + ytterkant,
                                          # men ekskluderer senterprikk (skjult ved overlapp).
    'hit_overlap_tmpl_r_hi': 1.05,        # Ytre grense paa halvt NCC-template (x marker_r)
    'hit_overlap_ncc_thresh': 0.10,       # NCC-terskel for maanesigd-maske. Sann maanesigd gir
                                          # NCC~0.15 for delvis-skjult treff (probe [G/H]). Lavt
                                          # terskel OK fordi votemap-gulv + naeerhets-filter allerede
                                          # er strenge; typen falsk positiv (firkant, stoy) scorer ~0.

    # Midtpunkt-firkant (MPI-markor): displayet tegner en liten firkant med kryss (+)
    # i tyngdepunktet til treffgruppa (invertert overlegg -> motsatt farge av bakgrunn).
    # Maalt paa C-sett: side 53px vs markordiameter 75px = 1/sqrt(2) (innskrevet),
    # strekbredde ~7px (~1 display-piksel). Matches i GRADIENT-domenet (polaritets-
    # invariant: inversjon bevarer kanter), IKKE paa intensitet.
    'mpi_square_side_frac': 0.58,       # Firkantside / delta (53/91.5; = 1.414*marker_r)
    'mpi_square_stroke_frac': 0.077,    # Strekbredde / delta (7/91.5)
    'mpi_search_r_frac': 2.0,           # Soekeradius rundt treff-tyngdepunkt (x delta)
    'mpi_median_frac': 0.06,            # Median-rensing av rutenett foer gradient (x delta)
    'mpi_radial_reject': False,         # Radiell gradient-avstoying: SKADELIG (dreper firkantens
                                        # egne akse-kanter naer ringsenter). Behold False.
    'mpi_prior_sigma_frac': 0.6,        # Sigma paa lokasjons-prior (Gauss om treff-tyngdepunkt, x delta).
                                        # Demper fjerne tall/ring-topper; taaler skjult-treff-bias (~0.3d)
    'mpi_struct_thresh': 0.30,          # Aksept-terskel paa strukturell 3-rygg-score (diagnostikk)

    # Sentrum-sveip (overlap.py): template-skan over bull-sonen for treff der
    # stemmekart er undertrykt av radial gating mot poengringer.
    'hit_center_scan': True,
    'hit_center_scan_r_frac': 1.0,     # Soekeradius = frac * R10 (innerste poengring)
    'hit_center_ncc_thresh': 0.40,      # NCC-terskel for sentralt treff. Hoyere enn
                                        # overlapp-terskel (0.10) fordi det ikke finnes
                                        # stemmekart-forhaaandsfiltrering her. Sentralt isolert
                                        # treff scorer NCC~0.90; falske positiver ~0.18.

    # Kvalitetsport: forkast bilder uten gyldig poengskive (rings.validate_calibration)
    'gate_min_delta_px': 25.0,            # Min ringavstand i px (ekte foto: 60-115; støy: <20)
    'gate_min_rings': 4,                  # Min antall tilpassede ringer (4 jevnt fordelte
                                          # konsentriske ringer er allerede en sterk signatur;
                                          # skråfoto kan mangle de svakeste ringene før retting)
    'gate_max_resid_frac': 0.12,          # Maks avvik fra jevn ringfordeling (andel av ringavstand)
    'gate_require_hits': True,            # Forkast bilder der ingen treff blir funnet

    # Skjermdeteksjon (screen.py)
    'screen_rectify_enable': True,        # Finn skjerm, rett perspektiv, beskjær før analyse
    'screen_work_size': 480,             # Arbeidsoppløsning for skjermdeteksjon (kraftig nedskalert)
    'screen_stretch_pct': 1.0,           # Kontraststrekk: klipp ytterste pct% i hver hale (midtre 98%)
    'screen_blur_sigma': 3.5,            # Kraftig uskarphet (fjerner aliasing + indre UI-detalj)
    'screen_low_frac': 0.55,             # Hysterese lav terskel = frac * Otsu (tar med grå skjermdeler)
    'screen_blob_open_frac': 0.0,        # Morfologisk åpning på blob (fjerner smale forbindelser til apparatus); 0=av
    'screen_side_outer_tol_frac': 0.0,   # Maks relativ avstand utenfor grovt-estimat-side; 0=av
    'screen_blob_convex_hull': False,    # DØD: streng nedgradering (absorberer glare/støy). C10-fiks/C9-brekk-bytte. Behold av.
    'screen_min_area_frac': 0.10,        # Min andel av bildet skjermen må dekke
    'analyze_screen_fallback': False,    # Fall tilbake til hele originalbildet hvis skjermutklippet ikke kalibrerer
    # Contrast-ROI: isoler apparatet via lokal struktur (uavhengig av bakgrunnsfarge)
    'screen_use_contrast_roi': True,     # Maskér bort alt utenfor apparatet før segmentering
    'screen_contrast_win_frac': 0.04,    # Vindu for lokalt kontrastkart (andel av maks side)
    'screen_roi_open_frac': 0.025,       # Åpning: fjern kontrast-flekker / kutt broer til rot
    'screen_roi_bright_min': 190,        # Kontrast-region må inneholde en så lys piksel (skjerm)
    'screen_roi_dilate_frac': 0.06,      # Utvid ROI litt for å få med skjermkant/ramme
    # Kantraffinering av skjermhjørner (robust linjetilpasning pr. side)
    'screen_side_inlier_tol_frac': 0.025,# Inlier-toleranse for linje (andel av sidelengde).
                                         # Må romme uskarphet-spredning (~5px), ellers teller selv
                                         # en rett kant som uteliggere
    # Godkjenning er SPENN-primær: en ekte skjermkant er en rett linje som dekker
    # mesteparten av siden. Inlier-ANDEL er upålitelig (uskarphet sprer punktene),
    # så vi krever bare et lavt gulv der + at inliers spenner mesteparten av siden.
    'screen_side_min_inlier_frac': 0.15, # Lavt gulv: linja må finnes
    'screen_side_min_span_frac': 0.60,   # Inliers må spenne >=60% av siden
    # Kant-snapping: etter grov linjetilpasning, la hver side "vandre" mot den
    # sterkeste gradient-kammen (ekte skjermkant). Løser glare-forskjøvet blob
    # (C10) uten å stole på en fast korridor rundt grov-siden.
    'screen_refine_gradient_lines': True, # Slå på iterativ kant-snapping etter grov linje
    'screen_snap_iters': 4,               # Antall snap-iterasjoner (linja re-sentreres hver gang)
    'screen_snap_band_frac': 0.045,       # Halv søkebredde vinkelrett på siden (andel av maks side)
    'screen_snap_dir_cos': 0.5,           # Gradientretning må ligge innenfor ~60° av sidenormalen
    'screen_snap_end_skip_frac': 0.08,    # Hopp over denne andelen nær hvert hjørne (unngå nabokant)
    'screen_snap_strong_frac': 0.6,       # Velg YTTERSTE gradient >= denne andelen av skann-toppen
                                          # (skjermkant er ytre+sterk; indre ringer/UI-linjer avvises)

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

