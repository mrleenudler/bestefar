"""
Ring-kalibrerings-mellomsteg for C10: skjerm-beskjaert vs. fullt bilde.

Viser kvifor ring-kalibrering feiler pa skjerm-beskjaert bilde (feil sirkel funnet)
men fungerer pa fullt originalbilde.

Output: Visualiseringer/outputs/C10_ringcalib_montage.png
"""
import cv2
import numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import preprocess
import screen as sc
import rings
from Bestefar import detect_outer_circle

cfg = DEFAULT_CONFIG.copy()
OUT = Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)

FONT = cv2.FONT_HERSHEY_SIMPLEX
theta_samples = cfg.get('ring_theta_samples', 720)
sigma_rho     = cfg.get('ring_rho_sigma', 3.0)

# ── Last C10 ──────────────────────────────────────────────────────────────────
img_full = cv2.imread('Testsett/C10.jpg')
if img_full is None:
    raise FileNotFoundError('Testsett/C10.jpg ikke funnet')
print(f'C10 original: {img_full.shape[1]}x{img_full.shape[0]}')

# ── Rute 1: skjerm-beskjaert ─────────────────────────────────────────────────
res = sc.rectify_to_screen(img_full, cfg, dbg := [])
if res is None:
    raise RuntimeError(f'Screen-deteksjon feilet: {dbg}')
warped, M, rect, rough = res
print(f'Skjerm-beskjaert: {warped.shape[1]}x{warped.shape[0]}')

cx1, cy1, r1, _ = detect_outer_circle(warped, cfg, debug=False, filename='C10_crop')
print(f'Rute 1 ytre sirkel: ({cx1:.0f},{cy1:.0f}) r={r1:.0f}')

gray_crop = preprocess.to_gray(warped)

# r_max for crop
h_c, w_c = gray_crop.shape
r_edge_c = min(cx1, w_c - cx1, cy1, h_c - cy1)
r_max_c  = max(cfg.get('ring_r_max_frac', 1.06) * r1,
               min(cfg.get('ring_generous_mult', 2.6) * r1,
                   cfg.get('ring_generous_edge_frac', 0.95) * r_edge_c))
r_samples_c = int(round(r_max_c))
print(f'Rute 1 r_max={r_max_c:.0f}  r_edge={r_edge_c:.0f}')

P_c = rings.warp_polar_gray(gray_crop, (cx1, cy1), r_max_c, r_samples_c, theta_samples)
G_c = rings.radial_gradient_abs(P_c, sigma_rho)
H_c = G_c.mean(axis=0)

# ── Rute 2: fullt bilde ──────────────────────────────────────────────────────
cx2, cy2, r2, _ = detect_outer_circle(img_full, cfg, debug=False, filename='C10_full')
print(f'Rute 2 ytre sirkel: ({cx2:.0f},{cy2:.0f}) r={r2:.0f}')

gray_full = preprocess.to_gray(img_full)

h_f, w_f = gray_full.shape
r_edge_f = min(cx2, w_f - cx2, cy2, h_f - cy2)
r_max_f  = max(cfg.get('ring_r_max_frac', 1.06) * r2,
               min(cfg.get('ring_generous_mult', 2.6) * r2,
                   cfg.get('ring_generous_edge_frac', 0.95) * r_edge_f))
r_samples_f = int(round(r_max_f))
print(f'Rute 2 r_max={r_max_f:.0f}  r_edge={r_edge_f:.0f}')

P_f = rings.warp_polar_gray(gray_full, (cx2, cy2), r_max_f, r_samples_f, theta_samples)
G_f = rings.radial_gradient_abs(P_f, sigma_rho)
H_f = G_f.mean(axis=0)

# ── Hjelpefunksjoner for visualisering ───────────────────────────────────────
def norm8(arr):
    lo, hi = arr.min(), arr.max()
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def polar_vis(P, title, ring_r=None, r_max=None, r_outer=None, w_out=900, h_out=400):
    """Render polar image as BGR with title + optional ring marker."""
    p8 = norm8(P)
    # Resize to fixed width x height
    vis = cv2.resize(cv2.cvtColor(p8, cv2.COLOR_GRAY2BGR), (w_out, h_out))
    if ring_r is not None and r_max is not None and r_max > 0:
        col_frac = ring_r / r_max
        col_px   = int(round(col_frac * w_out))
        cv2.line(vis, (col_px, 0), (col_px, h_out), (0, 80, 220), 2)
        cv2.putText(vis, f'r_outer={ring_r:.0f}px', (col_px + 4, 22),
                    FONT, 0.48, (0, 80, 220), 1, cv2.LINE_AA)
    # Title bar
    bar = np.full((32, w_out, 3), 30, np.uint8)
    cv2.putText(bar, title, (6, 22), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def profile_vis(H, r_max, title, w_out=900, h_out=300, r_outer=None):
    """Plot radial profile H (1D) as a line chart image."""
    canvas = np.full((h_out, w_out, 3), 25, np.uint8)
    n = len(H)
    if n < 2:
        return canvas
    Hn = (H - H.min()) / max(H.max() - H.min(), 1e-6)
    pts = []
    for i, v in enumerate(Hn):
        x = int(round(i / (n - 1) * (w_out - 1)))
        y = int(round((1.0 - v) * (h_out - 1)))
        pts.append((x, y))
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i+1], (80, 200, 80), 1)

    # Mark r_outer as vertical line
    if r_outer is not None and r_max > 0:
        col_frac = r_outer / r_max
        col_px   = int(round(col_frac * (w_out - 1)))
        cv2.line(canvas, (col_px, 0), (col_px, h_out), (0, 80, 220), 2)
        cv2.putText(canvas, f'r={r_outer:.0f}', (col_px + 4, 20),
                    FONT, 0.45, (0, 80, 220), 1, cv2.LINE_AA)

    # X-axis ticks at 0, 25%, 50%, 75%, r_max
    for frac, lbl in [(0.0, '0'), (0.25, f'{r_max*0.25:.0f}'),
                       (0.5, f'{r_max*0.5:.0f}'), (0.75, f'{r_max*0.75:.0f}'),
                       (1.0, f'{r_max:.0f}px')]:
        xp = int(round(frac * (w_out - 1)))
        cv2.line(canvas, (xp, h_out - 8), (xp, h_out - 1), (100, 100, 100), 1)
        cv2.putText(canvas, lbl, (xp + 2, h_out - 10),
                    FONT, 0.38, (130, 130, 130), 1, cv2.LINE_AA)

    bar = np.full((32, w_out, 3), 30, np.uint8)
    cv2.putText(bar, title, (6, 22), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([bar, canvas])


def thumb_with_circle(img_bgr, cx, cy, r, title, w_out=900):
    """Draw detected circle on a thumbnail."""
    h_img, w_img = img_bgr.shape[:2]
    scale = w_out / w_img
    small = cv2.resize(img_bgr, (w_out, int(h_img * scale)))
    cv2.circle(small, (int(round(cx * scale)), int(round(cy * scale))),
               int(round(r * scale)), (0, 60, 220), 3)
    cv2.circle(small, (int(round(cx * scale)), int(round(cy * scale))), 6, (0, 220, 255), -1)
    bar = np.full((32, w_out, 3), 30, np.uint8)
    cv2.putText(bar, title, (6, 22), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    h_cell = 380
    vis = np.vstack([bar, small])
    return cv2.resize(vis, (w_out, h_cell + 32))


# ── Bygg kolonne 1: skjerm-beskjaert ─────────────────────────────────────────
W = 900   # bredde per kolonne
col1 = []

# Panel A: beskjaert bilde med feil sirkel
col1.append(thumb_with_circle(
    warped, cx1, cy1, r1,
    f'A SCREEN CROP  {warped.shape[1]}x{warped.shape[0]}'
    f'  FEIL sirkel: ({cx1:.0f},{cy1:.0f}) r={r1:.0f}',
    w_out=W))

# Panel B: polart bilde (screen crop)
col1.append(polar_vis(
    P_c,
    f'B POLART BILDE (screen crop)  r_max={r_max_c:.0f}px  [{r_samples_c}x{theta_samples}]',
    ring_r=r1, r_max=r_max_c, w_out=W, h_out=380))

# Panel C: gradient-magnitud polart (screen crop)
col1.append(polar_vis(
    G_c,
    f'C |dP/drho| (screen crop)  — ingen synlige ringband',
    ring_r=r1, r_max=r_max_c, w_out=W, h_out=380))

# Panel D: radialprofil (screen crop)
col1.append(profile_vis(
    H_c, r_max_c,
    f'D RADIALPROFIL H(rho) (screen crop)  — ingen tydeige topper -> 0 ringer',
    w_out=W, h_out=280, r_outer=r1))

# ── Bygg kolonne 2: fullt bilde ───────────────────────────────────────────────
col2 = []

# Panel E: fullt bilde med riktig sirkel
col2.append(thumb_with_circle(
    img_full, cx2, cy2, r2,
    f'E FULLT BILDE  RIKTIG sirkel: ({cx2:.0f},{cy2:.0f}) r={r2:.0f}',
    w_out=W))

# Panel F: polart bilde (fullt)
col2.append(polar_vis(
    P_f,
    f'F POLART BILDE (fullt)  r_max={r_max_f:.0f}px  [{r_samples_f}x{theta_samples}]',
    ring_r=r2, r_max=r_max_f, w_out=W, h_out=380))

# Panel G: gradient-magnitud polart (fullt)
col2.append(polar_vis(
    G_f,
    f'G |dP/drho| (fullt)  — tydeige vertikale band = ringer',
    ring_r=r2, r_max=r_max_f, w_out=W, h_out=380))

# Panel H: radialprofil (fullt)
col2.append(profile_vis(
    H_f, r_max_f,
    f'H RADIALPROFIL H(rho) (fullt)  — tydeige topper -> 9 ringer funnet',
    w_out=W, h_out=280, r_outer=r2))

# ── Gjor alle paneler like hoye innen kvar kolonne ───────────────────────────
def make_col(panels):
    heights = [p.shape[0] for p in panels]
    # Pad til maks hoyde i kvar rad
    col = np.vstack(panels)
    return col

left  = make_col(col1)
right = make_col(col2)

# Juster hoyder
h_left, h_right = left.shape[0], right.shape[0]
if h_left > h_right:
    pad = np.full((h_left - h_right, W, 3), 20, np.uint8)
    right = np.vstack([right, pad])
elif h_right > h_left:
    pad = np.full((h_right - h_left, W, 3), 20, np.uint8)
    left = np.vstack([left, pad])

# Header
hdr = np.full((50, W * 2, 3), 15, np.uint8)
cv2.putText(hdr,
    'C10 RINGKALIBRERING: SCREEN CROP (FEILER, feil sirkel) vs. FULLT BILDE (OK)',
    (10, 33), FONT, 0.65, (0, 200, 255), 1, cv2.LINE_AA)

montage = np.hstack([left, right])
montage = np.vstack([hdr, montage])

out_path = OUT / 'C10_ringcalib_montage.png'
cv2.imwrite(str(out_path), montage)
print(f'\nLagret -> {out_path}')
print(f'Montasje: {montage.shape[1]}x{montage.shape[0]}px')
