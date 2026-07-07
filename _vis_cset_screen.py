"""
Montasje av skjermdeteksjons-trinnene for hele C-settet.
Kolonner: blob - grovt estimat - gradient - raffinert - crop

Output: Visualiseringer/outputs/Cset_screen_montage.png
"""
import cv2
import numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import screen as sc

cfg = DEFAULT_CONFIG.copy()
OUT = Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
ROW_H = 340
BAR_H = 24


def add_label(img, text, ok=True):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    bar = np.full((BAR_H, w, 3), 22, np.uint8)
    col = (80, 200, 80) if ok else (60, 60, 220)
    cv2.putText(bar, text, (4, BAR_H - 6), FONT, 0.38, col, 1, cv2.LINE_AA)
    return np.vstack([img, bar])


def fit_h(img, H=ROW_H):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = H / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), H),
                      interpolation=cv2.INTER_AREA)


def draw_quad_on(base, quad, color=(0, 0, 220), dot_col=(0, 220, 0), thick=2):
    out = base.copy() if base.ndim == 3 else cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    pts = np.round(quad).astype(np.int32)
    cv2.polylines(out, [pts], True, color, thick)
    for p in pts:
        cv2.circle(out, tuple(p), 4, dot_col, -1)
    return out


def make_row(name, img_bgr):
    h0, w0 = img_bgr.shape[:2]
    scale = min(cfg.get('screen_work_size', 480) / max(h0, w0), 1.0)
    small = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)),
                       interpolation=cv2.INTER_AREA)
    gray     = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray_n   = sc._normalize(gray, cfg)
    gray_b   = cv2.GaussianBlur(gray_n, (0, 0), cfg.get('screen_blur_sigma', 3.5))

    dbg = []
    roi = sc._apparatus_roi(gray_n, cfg, dbg) if cfg.get('screen_use_contrast_roi', True) else None

    contour, area_frac = sc._screen_blob(gray_b, cfg, dbg, roi=roi)

    gx   = cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    cells = []

    # ── Panel 1: blob (før eventuell konveks innhylling) ─────────────────────
    if contour is not None:
        mask = np.zeros(gray_b.shape, np.uint8)
        cv2.drawContours(mask, [contour.astype(np.int32)], -1, 255, -1)
        blob_vis = cv2.cvtColor(gray_b, cv2.COLOR_GRAY2BGR)
        blob_vis[mask > 0] = (blob_vis[mask > 0] * 0.5 +
                              np.array([0, 80, 0], np.float32) * 0.5).astype(np.uint8)
        cv2.drawContours(blob_vis, [contour.astype(np.int32)], -1, (0, 220, 80), 1)
        p1 = add_label(fit_h(blob_vis), f'{name} BLOB  a={area_frac:.2f}')
    else:
        err = np.full((*gray_b.shape, 3), 35, np.uint8)
        cv2.putText(err, 'FEIL', (8, 30), FONT, 0.8, (0, 60, 220), 2)
        p1 = add_label(fit_h(err), f'{name} BLOB FEILET', ok=False)

    cells.append(p1)

    # Konveks innhylling (samme logikk som i detect_screen_quad)
    if contour is not None and cfg.get('screen_blob_convex_hull', False):
        hull_pts = cv2.convexHull(contour.astype(np.int32))
        hull_mask = np.zeros(gray_b.shape, np.uint8)
        cv2.fillConvexPoly(hull_mask, hull_pts, 255)
        hull_cnts, _ = cv2.findContours(hull_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        if hull_cnts:
            contour = hull_cnts[0].reshape(-1, 2).astype(np.float32)

    # ── Panel 2: grovt estimat ────────────────────────────────────────────────
    if contour is not None:
        rough_ds = sc._rough_quad(contour)
        rough_vis = draw_quad_on(gray_b, rough_ds, color=(0, 200, 255), dot_col=(0, 200, 255))
        p2 = add_label(fit_h(rough_vis), f'{name} GROVT')
    else:
        rough_ds = None
        p2 = add_label(fit_h(np.full((*gray_b.shape, 3), 35, np.uint8)),
                       f'{name} GROVT (hoppet over)', ok=False)
    cells.append(p2)

    # ── Panel 3: gradient-magnitud ────────────────────────────────────────────
    gmag_vis = cv2.normalize(gmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    gmag_bgr = cv2.cvtColor(gmag_vis, cv2.COLOR_GRAY2BGR)
    if contour is not None:
        cv2.drawContours(gmag_bgr, [contour.astype(np.int32)], -1, (0, 200, 80), 1)
    p3 = add_label(fit_h(gmag_bgr), f'{name} GRADIENT')
    cells.append(p3)

    # ── Panel 4: raffinert ────────────────────────────────────────────────────
    if contour is not None and rough_ds is not None:
        dbg_ref = []
        refined_ds = sc._refine_from_contour(contour, rough_ds, gmag, cfg, dbg_ref,
                                             gx=gx, gy=gy, roi=roi)
        if refined_ds is not None:
            ref_vis = draw_quad_on(gray_b, rough_ds, color=(0, 200, 255), dot_col=(0, 200, 255), thick=1)
            ref_vis = draw_quad_on(ref_vis, refined_ds, color=(0, 0, 220), dot_col=(0, 220, 80), thick=2)
            p4 = add_label(fit_h(ref_vis), f'{name} RAFFINERT (rod=raffinert, gult=grovt)')
        else:
            ref_vis = draw_quad_on(gray_b, rough_ds, color=(0, 200, 255), dot_col=(0, 200, 255))
            cv2.putText(ref_vis, 'FEIL', (8, 30), FONT, 0.8, (0, 60, 220), 2)
            msg = dbg_ref[-1] if dbg_ref else '?'
            p4 = add_label(fit_h(ref_vis), f'{name} RAFFINERT FEIL: {msg[:40]}', ok=False)
            refined_ds = None
    else:
        refined_ds = None
        p4 = add_label(fit_h(np.full((*gray_b.shape, 3), 35, np.uint8)),
                       f'{name} RAFFINERT (hoppet over)', ok=False)
    cells.append(p4)

    # ── Panel 5: crop ─────────────────────────────────────────────────────────
    res = sc.rectify_to_screen(img_bgr, cfg, dbg := [])
    if res is not None:
        warped = res[0]
        p5 = add_label(fit_h(warped), f'{name} CROP  {warped.shape[1]}x{warped.shape[0]}')
    else:
        err = np.full((ROW_H, 200, 3), 35, np.uint8)
        cv2.putText(err, 'FEIL', (8, 60), FONT, 1.0, (0, 60, 220), 2)
        p5 = add_label(err, f'{name} CROP FEILET', ok=False)
    cells.append(p5)

    # Make uniform height then hstack
    H_max = max(c.shape[0] for c in cells)
    def pad_h(c):
        if c.shape[0] < H_max:
            pad = np.full((H_max - c.shape[0], c.shape[1], 3), 18, np.uint8)
            return np.vstack([c, pad])
        return c
    return np.hstack([pad_h(c) for c in cells])


# ── Run for C1..C10 ───────────────────────────────────────────────────────────
rows = []
for i in range(1, 11):
    name = f'C{i}'
    img  = cv2.imread(f'Testsett/{name}.jpg')
    if img is None:
        print(f'{name}: ikke funnet, hopper over')
        continue
    print(f'{name}...', end=' ', flush=True)
    row = make_row(name, img)
    rows.append(row)
    print('OK')

max_w = max(r.shape[1] for r in rows)
def pad_w(r):
    if r.shape[1] < max_w:
        p = np.full((r.shape[0], max_w - r.shape[1], 3), 15, np.uint8)
        return np.hstack([r, p])
    return r
rows = [pad_w(r) for r in rows]

hdr = np.full((40, max_w, 3), 12, np.uint8)
cv2.putText(hdr,
    'C-SETT SKJERMDETEKSJON:  BLOB  |  GROVT  |  GRADIENT  |  RAFFINERT  |  CROP',
    (8, 27), FONT, 0.58, (0, 200, 255), 1, cv2.LINE_AA)

montage = np.vstack([hdr] + rows)
out_path = OUT / 'Cset_screen_montage.png'
cv2.imwrite(str(out_path), montage)
print(f'\nLagret -> {out_path}  ({montage.shape[1]}x{montage.shape[0]}px)')
