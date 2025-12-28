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
        'log_file_kronologisk': "ytelse_kronologisk.txt",
        
        # Visualisering - Farger (BGR format)
        'color_green': (0, 255, 0),       # Grønn for contours/sirkler
        'color_red': (0, 0, 255),         # Rød for sentrum
        'color_blue': (255, 0, 0),        # Blå for tekst
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
        'test_image_path': Path("Elektronisk skive.jpg"),
        
        # Outer circle detection - Gradient-normal center voting
        'outer_circle_max_side': 1200,              # Maks side i px for downscaling
        'outer_circle_blur_sigma': 1.0,             # Gaussian blur sigma
        'outer_circle_mag_floor': 1e-3,             # Minimum gradient magnitude
        'outer_circle_mag_percentile': 60.0,        # Percentile for edge threshold
        'outer_circle_max_edge_points': 12000,      # Maks antall edge points
        'outer_circle_rmin_frac': 0.25,             # Minimum radius fraction
        'outer_circle_rmax_frac': 0.55,              # Maximum radius fraction
        'outer_circle_t_steps': 9,                  # Antall t-verdier for voting
        'outer_circle_center_win': 5,               # Vindu for subpixel center refinement
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
    mag = cv2.magnitude(Gx, Gy)
    
    # Gradient directions (normalized)
    # Note: For a dark circle on light background, gradient points outward (away from center)
    # For a light circle on dark background, gradient points inward (toward center)
    # We vote in both directions, so it should work either way
    eps = 1e-6
    mag_safe = mag + eps
    ux = Gx / mag_safe
    uy = Gy / mag_safe
    log_operation_time(filename, "detect_outer_circle", "gradient", time.time() - start_gradient)
    
    # 4) Dynamic threshold
    start_threshold = time.time()
    mag_floor = cfg['outer_circle_mag_floor']
    mag_nonzero = mag[mag > mag_floor]
    if len(mag_nonzero) == 0:
        raise ValueError("Ingen gradient magnituder funnet")
    
    percentile = cfg['outer_circle_mag_percentile']
    threshold = np.percentile(mag_nonzero, percentile)
    edge_mask = mag >= threshold
    
    # Extract edge points
    y_coords, x_coords = np.where(edge_mask)
    x = x_coords.astype(np.float32)
    y = y_coords.astype(np.float32)
    u = np.column_stack((ux[y_coords, x_coords], uy[y_coords, x_coords]))
    w = mag[y_coords, x_coords]
    
    # Random subsample if too many
    max_points = cfg['outer_circle_max_edge_points']
    if len(x) > max_points:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(x), max_points, replace=False)
        x = x[indices]
        y = y[indices]
        u = u[indices]
        w = w[indices]
    log_operation_time(filename, "detect_outer_circle", "threshold_extract", time.time() - start_threshold)
    
    # 5) Center voting
    start_voting = time.time()
    acc = np.zeros((new_h, new_w), dtype=np.float32)
    
    rmin = cfg['outer_circle_rmin_frac'] * min(new_h, new_w)
    rmax = cfg['outer_circle_rmax_frac'] * min(new_h, new_w)
    t_steps = cfg['outer_circle_t_steps']
    t_values = np.linspace(rmin, rmax, t_steps)
    
    for t in t_values:
        # Vote both directions (inward and outward)
        # For a dark circle on light background: gradient points outward (away from center)
        # For a light circle on dark background: gradient points inward (toward center)
        # We vote in both directions to handle both cases
        cx1 = x - t * u[:, 0]  # Vote in direction opposite to gradient (toward center for dark circle)
        cy1 = y - t * u[:, 1]
        cx2 = x + t * u[:, 0]  # Vote in gradient direction (away from center for dark circle)
        cy2 = y + t * u[:, 1]
        
        # Round to integers
        cx1_int = np.round(cx1).astype(int)
        cy1_int = np.round(cy1).astype(int)
        cx2_int = np.round(cx2).astype(int)
        cy2_int = np.round(cy2).astype(int)
        
        # Filter out-of-bounds votes (ignore instead of clipping)
        valid1 = (cx1_int >= 0) & (cx1_int < new_w) & (cy1_int >= 0) & (cy1_int < new_h)
        valid2 = (cx2_int >= 0) & (cx2_int < new_w) & (cy2_int >= 0) & (cy2_int < new_h)
        
        # Only vote for valid positions
        if np.any(valid1):
            idx1 = cy1_int[valid1] * new_w + cx1_int[valid1]
            w1 = w[valid1]
            acc_flat = acc.flatten()
            np.add.at(acc_flat, idx1, w1)
            acc = acc_flat.reshape((new_h, new_w))
        
        if np.any(valid2):
            idx2 = cy2_int[valid2] * new_w + cx2_int[valid2]
            w2 = w[valid2]
            acc_flat = acc.flatten()
            np.add.at(acc_flat, idx2, w2)
            acc = acc_flat.reshape((new_h, new_w))
    
    # Find peak
    peak_idx = np.argmax(acc)
    peak_y = peak_idx // new_w
    peak_x = peak_idx % new_w
    c_hat = (peak_x, peak_y)
    log_operation_time(filename, "detect_outer_circle", "voting", time.time() - start_voting)
    
    # 6) Subpixel center refinement
    start_center_refine = time.time()
    win = cfg['outer_circle_center_win']
    x_min = max(0, peak_x - win)
    x_max = min(new_w, peak_x + win + 1)
    y_min = max(0, peak_y - win)
    y_max = min(new_h, peak_y + win + 1)
    
    window = acc[y_min:y_max, x_min:x_max]
    y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]
    
    total_weight = np.sum(window)
    if total_weight > 0:
        c0x = np.sum(x_grid * window) / total_weight
        c0y = np.sum(y_grid * window) / total_weight
    else:
        c0x, c0y = float(peak_x), float(peak_y)
    log_operation_time(filename, "detect_outer_circle", "center_refine", time.time() - start_center_refine)
    
    # 7) Radius estimation
    start_radius = time.time()
    # Compute radii and alignment
    dx = x - c0x
    dy = y - c0y
    ri = np.sqrt(dx**2 + dy**2)
    
    # Radial alignment
    ri_safe = ri + eps
    vi = np.column_stack((dx / ri_safe, dy / ri_safe))
    ai = np.abs(vi[:, 0] * u[:, 0] + vi[:, 1] * u[:, 1])
    
    # Filter by alignment
    align_min = cfg['outer_circle_align_min']
    keep_mask = ai >= align_min
    x_aligned = x[keep_mask]
    y_aligned = y[keep_mask]
    ri_aligned = ri[keep_mask]
    w_aligned = w[keep_mask]
    ai_aligned = ai[keep_mask]
    
    # Build histogram
    rmax_search = 0.6 * min(new_h, new_w)
    bin_width = cfg['outer_circle_r_bin_px']
    bins = np.arange(0, rmax_search + bin_width, bin_width)
    weights_hist = w_aligned * ai_aligned
    hist, bin_edges = np.histogram(ri_aligned, bins=bins, weights=weights_hist)
    
    # Smooth histogram
    smooth_sigma = cfg['outer_circle_r_smooth_sigma']
    try:
        from scipy import ndimage
        hist_smooth = ndimage.gaussian_filter1d(hist.astype(np.float32), smooth_sigma)
    except ImportError:
        # Fallback: simple box filter if scipy not available
        kernel_size = int(smooth_sigma * 2) * 2 + 1
        kernel = np.ones(kernel_size) / kernel_size
        hist_smooth = np.convolve(hist.astype(np.float32), kernel, mode='same')
    
    # Find peaks
    peaks_keep = cfg['outer_circle_peaks_keep']
    try:
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(hist_smooth, prominence=np.max(hist_smooth) * 0.1)
    except ImportError:
        # Fallback: simple local maxima detection
        peaks = []
        for i in range(1, len(hist_smooth) - 1):
            if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
                if hist_smooth[i] > np.max(hist_smooth) * 0.1:
                    peaks.append(i)
        peaks = np.array(peaks)
    
    if len(peaks) == 0:
        # Fallback: use maximum
        peak_idx = np.argmax(hist_smooth)
        r0 = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    else:
        # Sort by score
        peak_scores = hist_smooth[peaks]
        sorted_indices = np.argsort(peak_scores)[::-1][:peaks_keep]
        sorted_peaks = peaks[sorted_indices]
        sorted_scores = peak_scores[sorted_indices]
        
        # Choose outermost peak with sufficient score
        best_score = sorted_scores[0]
        min_score = cfg['outer_circle_peak_score_min_frac'] * best_score
        
        # Find outermost peak above threshold
        outer_peak_idx = None
        for i in range(len(sorted_peaks) - 1, -1, -1):  # Start from largest radius
            if sorted_scores[i] >= min_score:
                outer_peak_idx = sorted_peaks[i]
                break
        
        if outer_peak_idx is None:
            outer_peak_idx = sorted_peaks[-1]  # Use outermost if none above threshold
        
        r0 = (bin_edges[outer_peak_idx] + bin_edges[outer_peak_idx + 1]) / 2
    
    log_operation_time(filename, "detect_outer_circle", "radius_hist_peaks", time.time() - start_radius)
    
    # 8) Final circle refinement
    start_final_fit = time.time()
    r_eps = cfg['outer_circle_r_inlier_eps']
    inlier_mask = np.abs(ri_aligned - r0) < r_eps
    inlier_points = np.column_stack((x_aligned[inlier_mask], y_aligned[inlier_mask]))
    inlier_weights = (w_aligned[inlier_mask] * ai_aligned[inlier_mask])
    
    if len(inlier_points) >= 3:
        result = fit_circle_pratt(inlier_points, inlier_weights)
        if result:
            cx_final, cy_final, r_final = result
        else:
            cx_final, cy_final, r_final = c0x, c0y, r0
    else:
        cx_final, cy_final, r_final = c0x, c0y, r0
    
    log_operation_time(filename, "detect_outer_circle", "final_fit", time.time() - start_final_fit)
    
    # 9) Map back to original coordinates
    cx_orig = cx_final / scale
    cy_orig = cy_final / scale
    r_orig = r_final / scale
    
    total_time = time.time() - start_total
    log_performance(filename, "detect_outer_circle", "total", total_time)
    
    # 10) Debug output
    debug_dict = None
    if debug:
        debug_dict = {
            'downscaled_gray': gray,
            'mag': mag,
            'edge_mask': edge_mask,
            'accumulator': acc,
            'c_hat': c_hat,
            'c0': (c0x, c0y),
            'radius_histogram': hist_smooth,
            'bin_edges': bin_edges,
            'chosen_peak': r0,
            'inlier_points': inlier_points,
            'scale': scale
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
    
    save_visualization(img, "01_Originalt_bilde")
    
    # Kjør deteksjon
    cx, cy, r, debug_dict = detect_outer_circle(img, config, debug=True, filename=filename)
    
    print(f"Funnet ytterste sirkel: sentrum=({cx:.2f}, {cy:.2f}), radius={r:.2f}")
    
    # Visualiser resultatet
    result = img.copy()
    center = (int(cx), int(cy))
    radius = int(r)
    
    # Tegn sirkelen
    cv2.circle(result, center, radius, config['color_green'], config['circle_thickness'])
    cv2.circle(result, center, config['circle_center_size'], config['color_red'], -1)
    cv2.putText(result, f"Radius: {radius}", (center[0] - 50, center[1] - radius - 20),
                config['font'], config['font_scale_medium'], config['color_blue'], config['font_thickness'])
    
    save_visualization(result, "02_Resultat_detect_outer_circle", 2)
    
    # Vis debug-artifakter hvis tilgjengelig
    if debug_dict:
        save_visualization(debug_dict['downscaled_gray'], "03_Downscaled_graaskala", 3)
        save_visualization(debug_dict['edge_mask'].astype(np.uint8) * 255, "04_Edge_mask", 4)
        
        # Visualiser accumulator med peak markert
        acc = debug_dict['accumulator']
        acc_norm = (acc / (np.max(acc) + 1e-6) * 255).astype(np.uint8)
        acc_colored = cv2.applyColorMap(acc_norm, cv2.COLORMAP_JET)
        
        # Mark peak position
        c_hat = debug_dict['c_hat']
        c0 = debug_dict['c0']
        # We're visualizing the downscaled accumulator, so use c_hat and c0 directly
        cv2.circle(acc_colored, (int(c_hat[0]), int(c_hat[1])), 5, (255, 255, 255), 2)
        cv2.circle(acc_colored, (int(c0[0]), int(c0[1])), 3, (0, 0, 0), 2)
        
        save_visualization(acc_colored, "05_Accumulator", 5)
        
        # Visualiser radius histogram
        if 'radius_histogram' in debug_dict:
            hist_img = np.zeros((200, 400), dtype=np.uint8)
            hist = debug_dict['radius_histogram']
            if len(hist) > 0:
                hist_norm = (hist / (np.max(hist) + 1e-6) * 199).astype(int)
                for i, h in enumerate(hist_norm[:400]):
                    cv2.line(hist_img, (i, 199), (i, 199 - h), 255, 1)
            save_visualization(hist_img, "06_Radius_histogram", 6)
    
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
