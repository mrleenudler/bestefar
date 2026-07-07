"""
Montasje av gradientbilde, stemmekart og detektert yttersirkel for hele C-settet.
Kjores pa fullt bilde for C1-C10, pluss C10 skjerm-beskjaert som ekstra rad.

Output: Visualiseringer/outputs/Cset_outer_circle_montage.png
"""
import cv2
import numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import preprocess
import screen as sc
from Bestefar import detect_outer_circle

cfg = DEFAULT_CONFIG.copy()
OUT = Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
ROW_H = 380          # height of each image panel in the montage
BAR_H = 28           # label bar height per panel
CELL_H = ROW_H + BAR_H

# ── helpers ──────────────────────────────────────────────────────────────────
def norm8(arr):
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def add_label(img_bgr, text, ok=True):
    h, w = img_bgr.shape[:2]
    bar = np.full((BAR_H, w, 3), 25, np.uint8)
    col = (80, 200, 80) if ok else (60, 60, 220)
    cv2.putText(bar, text, (5, BAR_H - 7), FONT, 0.42, col, 1, cv2.LINE_AA)
    return np.vstack([img_bgr, bar])


def fit_h(img, H=ROW_H):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = H / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), H))


def draw_circle_scaled(base_bgr, cx_ds, cy_ds, r_ds, color=(0, 60, 220), thick=2):
    """Draw circle on a display image that was scaled from the downscaled image."""
    disp_h, disp_w = base_bgr.shape[:2]
    ds_h, ds_w = base_bgr.shape[:2]   # will be overridden by caller via sx/sy
    out = base_bgr.copy()
    cv2.circle(out, (int(round(cx_ds)), int(round(cy_ds))), int(round(r_ds)), color, thick)
    cv2.circle(out, (int(round(cx_ds)), int(round(cy_ds))), 4, (0, 220, 255), -1)
    return out


def make_row(name, img_bgr, is_crop=False):
    """Run detect_outer_circle and return a horizontal strip of panels."""
    print(f'  {name} {"[screen crop]" if is_crop else ""}...', end=' ', flush=True)
    try:
        cx_orig, cy_orig, r_orig, dbg = detect_outer_circle(
            img_bgr, cfg, debug=True, filename=name)
    except Exception as e:
        print(f'FEIL: {e}')
        blank = np.full((CELL_H, 200, 3), 40, np.uint8)
        cv2.putText(blank, str(e)[:30], (5, 30), FONT, 0.4, (0, 60, 220), 1)
        return np.hstack([blank, blank, blank, blank])

    scale    = dbg['scale']
    cx_ds    = dbg['c_final'][0]
    cy_ds    = dbg['c_final'][1]
    r_ds     = r_orig * scale
    mag_ds   = dbg['mag']
    acc2     = dbg['accumulator_pass2']
    gray_ds  = dbg['downscaled_gray']

    ok_str   = 'OK' if True else 'FEIL'   # we got a result

    # ── Panel 1: gradient magnitude ───────────────────────────────────────────
    mag_vis = cv2.applyColorMap(norm8(mag_ds), cv2.COLORMAP_INFERNO)
    mag_vis = fit_h(mag_vis)
    # Draw detected circle on gradient too (for spatial reference)
    sx = mag_vis.shape[1] / mag_ds.shape[1]
    sy = mag_vis.shape[0] / mag_ds.shape[0]
    cv2.circle(mag_vis,
               (int(round(cx_ds * sx)), int(round(cy_ds * sy))),
               int(round(r_ds * sx)), (0, 220, 60), 2)
    cv2.circle(mag_vis,
               (int(round(cx_ds * sx)), int(round(cy_ds * sy))), 3, (0, 220, 60), 1)
    p1 = add_label(mag_vis, f'{name} GRADIENT  scale={scale:.3f}')

    # ── Panel 2: vote map (pass 2) + detected circle ──────────────────────────
    if acc2 is not None and acc2.max() > 0:
        acc_vis = cv2.applyColorMap(norm8(acc2), cv2.COLORMAP_JET)
    else:
        acc_vis = np.full((*gray_ds.shape, 3), 40, np.uint8)
        cv2.putText(acc_vis, 'acc=None', (5, 30), FONT, 0.5, (180, 180, 180), 1)
    acc_vis = fit_h(acc_vis)
    sx2 = acc_vis.shape[1] / mag_ds.shape[1]
    sy2 = acc_vis.shape[0] / mag_ds.shape[0]
    cv2.circle(acc_vis,
               (int(round(cx_ds * sx2)), int(round(cy_ds * sy2))),
               int(round(r_ds * sx2)), (255, 255, 255), 2)
    peak_val = float(acc2.max()) if acc2 is not None else 0.0
    p2 = add_label(acc_vis,
        f'{name} STEMMEKART  peak={peak_val:.1f}  c=({cx_ds:.0f},{cy_ds:.0f}) r={r_ds:.0f}ds')

    # ── Panel 3: raw image thumbnail + detected circle (original coords) ──────
    raw_disp = fit_h(img_bgr.copy())
    sx3 = raw_disp.shape[1] / img_bgr.shape[1]
    sy3 = raw_disp.shape[0] / img_bgr.shape[0]
    cv2.circle(raw_disp,
               (int(round(cx_orig * sx3)), int(round(cy_orig * sy3))),
               int(round(r_orig * sx3)), (0, 60, 220), 2)
    src_tag = 'CROP' if is_crop else 'FULL'
    p3 = add_label(raw_disp,
        f'{name} [{src_tag}]  c=({cx_orig:.0f},{cy_orig:.0f}) r={r_orig:.0f}px')

    print(f'c=({cx_orig:.0f},{cy_orig:.0f}) r={r_orig:.0f}')

    # Make all three panels the same width (use widest)
    W = max(p1.shape[1], p2.shape[1], p3.shape[1])
    def pad_w(p):
        if p.shape[1] < W:
            pad = np.full((p.shape[0], W - p.shape[1], 3), 20, np.uint8)
            return np.hstack([p, pad])
        return p

    return np.hstack([pad_w(p1), pad_w(p2), pad_w(p3)])


# ── Kjor for C1..C10 — bruk skjerm-beskjaert bilde der det finnes ────────────
rows = []
for i in range(1, 11):
    name = f'C{i}'
    path = Path(f'Testsett/{name}.jpg')
    img  = cv2.imread(str(path))
    if img is None:
        print(f'  {name}: finner ikke bilde, hopper over')
        continue

    dbg_sc = []
    res = sc.rectify_to_screen(img, cfg, dbg_sc)
    if res is not None:
        work_img  = res[0]
        is_crop   = True
        src_label = f'crop {work_img.shape[1]}x{work_img.shape[0]}'
    else:
        work_img  = img
        is_crop   = False
        src_label = 'full (ingen skjerm funnet)'
    print(f'  {name}: {src_label}', end=' -> ')

    row = make_row(name, work_img, is_crop=is_crop)
    rows.append(row)

# ── Tilpass alle rader til same bredde ────────────────────────────────────────
max_w = max(r.shape[1] for r in rows)
def pad_row(r):
    if r.shape[1] < max_w:
        pad = np.full((r.shape[0], max_w - r.shape[1], 3), 15, np.uint8)
        return np.hstack([r, pad])
    return r

rows = [pad_row(r) for r in rows]

# ── Header ────────────────────────────────────────────────────────────────────
hdr = np.full((42, max_w, 3), 15, np.uint8)
cv2.putText(hdr,
    'C-SETT: GRADIENT  |  STEMMEKART (pass2) + sirkel  |  RAW + detektert sirkel',
    (10, 28), FONT, 0.60, (0, 200, 255), 1, cv2.LINE_AA)

montage = np.vstack([hdr] + rows)

out_path = OUT / 'Cset_outer_circle_montage.png'
cv2.imwrite(str(out_path), montage)
print(f'\nLagret -> {out_path}  ({montage.shape[1]}x{montage.shape[0]}px)')
