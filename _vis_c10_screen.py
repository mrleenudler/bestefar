"""
Montasje av alle trinn i skjermdeteksjon for C10.
Viser hvert mellomsteg og fanger opp noyaktig HVOR det feiler.
Output: Visualiseringer/outputs/C10_screen_montage.png
"""
import cv2, numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import screen as sc

cfg = DEFAULT_CONFIG.copy()
OUT = Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)

CELL_H = 600   # hoyde per panel i montasjen
FONT   = cv2.FONT_HERSHEY_SIMPLEX

img = cv2.imread('Testsett/C10.jpg')
h0, w0 = img.shape[:2]
print(f'C10 originalstorrelse: {w0}x{h0}')

# ─── hjelpefunksjoner ──────────────────────────────────────────────────────────
def label(im, txt, ok=True):
    """Legg til statuslinje under bildet."""
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    bar_h = 36
    h, w = im.shape[:2]
    bar = np.full((bar_h, w, 3), 30, np.uint8)
    col = (80, 200, 80) if ok else (60, 60, 220)
    cv2.putText(bar, txt, (6, 24), FONT, 0.52, col, 1, cv2.LINE_AA)
    return np.vstack([im, bar])

def fit_cell(im, H=CELL_H):
    s = H / im.shape[0]
    return cv2.resize(im, (max(1, int(im.shape[1]*s)), H))

def overlay_contour(base, contour, color=(0,255,0), thick=2):
    out = base.copy()
    cv2.drawContours(out, [contour.astype(np.int32)], -1, color, thick)
    return out

def overlay_quad(base, quad, color=(0,0,255), pt_color=(0,255,0)):
    out = base.copy()
    pts = quad.astype(np.int32)
    cv2.polylines(out, [pts], True, color, 2)
    for p in pts:
        cv2.circle(out, tuple(p), 6, pt_color, -1)
    return out

# ─── Steg 0: original ─────────────────────────────────────────────────────────
cells = []
cells.append(fit_cell(label(img, '0 ORIGINAL')))

# ─── Steg 1: nedskalert + normalisert ─────────────────────────────────────────
scale = min(cfg['screen_work_size'] / max(h0, w0), 1.0)
small = cv2.resize(img, (int(w0*scale), int(h0*scale)), interpolation=cv2.INTER_AREA)
gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
gray_norm = sc._normalize(gray, cfg)
cells.append(fit_cell(label(gray_norm,
    f'1 NEDSKALERT+NORM  {small.shape[1]}x{small.shape[0]}  scale={scale:.3f}')))

# ─── Steg 2: Gaussisk uskarphet ───────────────────────────────────────────────
grayb = cv2.GaussianBlur(gray_norm, (0,0), cfg['screen_blur_sigma'])
cells.append(fit_cell(label(grayb,
    f'2 GAUSSBLUR  sigma={cfg["screen_blur_sigma"]}')))

# ─── Steg 3: kontrast-ROI (apparatus) ─────────────────────────────────────────
dbg = []
roi = sc._apparatus_roi(gray_norm, cfg, dbg)
if roi is not None:
    roi_vis = cv2.cvtColor(gray_norm, cv2.COLOR_GRAY2BGR)
    roi_vis[roi == 0] = [0, 0, 60]   # mork utenfor ROI
    cells.append(fit_cell(label(roi_vis, '3 KONTRAST-ROI  (groent = innenfor)')))
    print('Steg 3 OK')
else:
    err_img = np.full((*gray_norm.shape, 3), 40, np.uint8)
    cv2.putText(err_img, 'FEILET', (20,40), FONT, 1.2, (0,0,220), 2)
    cells.append(fit_cell(label(err_img, f'3 KONTRAST-ROI FEIL: {dbg[-1] if dbg else "?"}', ok=False)))
    print(f'Steg 3 FEIL: {dbg}')

# ─── Steg 4: skjerm-blob (hysterese-terskling) ────────────────────────────────
contour, area_frac = sc._screen_blob(grayb, cfg, dbg, roi=roi)
if contour is not None:
    blob_vis = cv2.cvtColor(grayb, cv2.COLOR_GRAY2BGR)
    blob_vis = overlay_contour(blob_vis, contour, color=(0,220,0))
    cells.append(fit_cell(label(blob_vis,
        f'4 SKJERM-BLOB  areal={area_frac:.2f}  min={cfg["screen_min_area_frac"]}')))
    print(f'Steg 4 OK  areal={area_frac:.2f}')
else:
    err_img = np.full((*grayb.shape, 3), 40, np.uint8)
    cv2.putText(err_img, 'FEILET', (20,40), FONT, 1.2, (0,0,220), 2)
    cells.append(fit_cell(label(err_img,
        f'4 SKJERM-BLOB FEIL  areal={area_frac:.2f}  {dbg[-1] if dbg else "?"}', ok=False)))
    print(f'Steg 4 FEIL: areal={area_frac:.2f}  {dbg}')

# ─── Steg 5: grovt firkant-estimat ────────────────────────────────────────────
if contour is not None:
    rough = sc._rough_quad(contour)
    rough_vis = overlay_contour(cv2.cvtColor(grayb, cv2.COLOR_GRAY2BGR), contour, (80,80,80))
    rough_vis = overlay_quad(rough_vis, rough, color=(0,200,255), pt_color=(0,200,255))
    cells.append(fit_cell(label(rough_vis, '5 GROVT ESTIMAT  (gult)')))
    print('Steg 5 OK')
else:
    rough = None
    cells.append(fit_cell(label(np.full((*grayb.shape,3),40,np.uint8),
                                '5 GROVT ESTIMAT  (hoppet over)', ok=False)))

# ─── Steg 6: gradient-vektet linjetilpasning + raffinerte hjorner ─────────────
if contour is not None and rough is not None:
    gx = cv2.Sobel(grayb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grayb, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    # Vis gradient-magnitudkart
    gmag_vis = cv2.normalize(gmag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cells.append(fit_cell(label(gmag_vis, '6a GRADIENT-MAGNITUD')))

    dbg_ref = []
    refined = sc._refine_from_contour(contour, rough, gmag, cfg, dbg_ref)
    if refined is not None:
        ref_vis = overlay_contour(cv2.cvtColor(grayb, cv2.COLOR_GRAY2BGR), contour, (60,60,60))
        ref_vis = overlay_quad(ref_vis, rough,   color=(0,200,255), pt_color=(0,200,255))
        ref_vis = overlay_quad(ref_vis, refined, color=(0,0,220),   pt_color=(0,255,0))
        cells.append(fit_cell(label(ref_vis, '6b RAFFINERT (rod) vs. GROVT (gult)')))
        print('Steg 6 OK')
    else:
        err_img = overlay_contour(cv2.cvtColor(grayb, cv2.COLOR_GRAY2BGR), contour, (60,60,60))
        err_img = overlay_quad(err_img, rough, color=(0,200,255), pt_color=(0,200,255))
        msg = dbg_ref[-1] if dbg_ref else '?'
        cv2.putText(err_img, 'LINJE-FIT FEIL', (8,30), FONT, 0.7, (0,0,220), 2)
        cells.append(fit_cell(label(err_img, f'6b RAFFINERING FEIL: {msg}', ok=False)))
        print(f'Steg 6 FEIL: {dbg_ref}')
        refined = None
else:
    refined = None
    cells.append(fit_cell(label(np.full((*grayb.shape,3),40,np.uint8),
                                '6 LINJETILPASNING  (hoppet over)', ok=False)))
    cells.append(cells[-1])   # placeholder for 6b

# ─── Steg 7: perspektivkorrigert skjerm-beskjaering ──────────────────────────
full_dbg = []
res = sc.rectify_to_screen(img, cfg, full_dbg)
if res is not None:
    warped, M, rect, rough_full = res
    cells.append(fit_cell(label(warped, f'7 BESKJAERT SKJERM  {warped.shape[1]}x{warped.shape[0]}')))
    print(f'Steg 7 OK: {warped.shape[1]}x{warped.shape[0]}')
else:
    err_img = np.full((400, 600, 3), 40, np.uint8)
    cv2.putText(err_img, 'SCREEN CROP FEILET', (20,60), FONT, 1.0, (0,0,220), 2)
    for i, ln in enumerate(full_dbg[-6:]):
        cv2.putText(err_img, ln, (10, 110+i*28), FONT, 0.5, (180,180,180), 1)
    cells.append(fit_cell(label(err_img,
        f'7 RECTIFY FEIL: {full_dbg[-1] if full_dbg else "?"}', ok=False)))
    print(f'Steg 7 FEIL. Debug-linjer:')
    for ln in full_dbg:
        print(f'  {ln}')

# ─── Monter alle celler sideom sideom ─────────────────────────────────────────
# Gjor dem alle like hoye (CELL_H + labelbar)
row = np.hstack(cells)
cv2.imwrite(str(OUT / 'C10_screen_montage.png'), row)
print(f'\nMontasje lagret -> Visualiseringer/outputs/C10_screen_montage.png')
print(f'Totalt {len(cells)} paneler, {row.shape[1]}x{row.shape[0]}px')
