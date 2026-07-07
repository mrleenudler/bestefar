"""
Visualisering for C10: sentrum-sveip (template-skan over bull-sone).
Output: Visualiseringer/outputs/C10_center_{scan,final}.png
"""
import cv2, numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import inspect_hits as ih
import circles as circle_det
from overlap import _odd, _extract_patch, _annular_mask, find_center_hits

cfg = DEFAULT_CONFIG.copy()
OUT = Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)

# ── last og preparer ──────────────────────────────────────────────────────────
img = cv2.imread('Testsett/C10.jpg')
got = ih.prep(img, 'C10')
if got is None:
    print('C10: KALIB FEIL – avslutter'); raise SystemExit(1)
img_d, gray_d, calib = got
cx0, cy0 = calib['center']
delta    = calib['delta_px']
R10      = calib['R10_px']
search_r = cfg['hit_search_r_max_frac'] * (R10 + 9.0 * delta)
marker_r = cfg['hit_marker_radius_frac'] * delta
dot_r    = cfg['hit_dot_radius_frac'] * delta
scan_r   = cfg['hit_center_scan_r_frac'] * R10

print(f'C10: center=({cx0:.0f},{cy0:.0f}) delta={delta:.1f} R10={R10:.1f} marker_r={marker_r:.1f}')
print(f'Sentrum-soekeradius: {scan_r:.1f}px')

# ── hoveddetektor: eksisterende treff ────────────────────────────────────────
cands = circle_det.detect_circles(
    gray_d, (cx0, cy0), search_r, marker_r, dot_r, cfg, inner_r=0.0)
hits8 = [{'x': X, 'y': Y, 'score': sc, 'type': 'filled', 'hough_r': marker_r}
         for X, Y, sc in cands]
print(f'Hoveddetektor: {len(hits8)} treff')
for h in hits8:
    d = np.hypot(h['x']-cx0, h['y']-cy0)
    print(f"  ({h['x']:.0f},{h['y']:.0f}) d={d:.1f}px={d/delta:.2f}*delta score={h['score']:.3f}")

# ── sentrum-sveip: NCC-kart over bull-sonen ──────────────────────────────────
# Bygg template
def _iso(h):
    others = [o for o in hits8 if o is not h]
    if not others: return 0.0
    return min(np.hypot(h['x']-o['x'], h['y']-o['y']) for o in others) * h['score']

src      = max(hits8, key=_iso)
r_lo     = cfg['hit_overlap_tmpl_r_lo'] * marker_r
r_hi     = cfg['hit_overlap_tmpl_r_hi'] * marker_r
tmpl_r_px = int(r_hi) + 2
tmpl_size = 2 * tmpl_r_px + 1
tmask     = _annular_mask(tmpl_r_px, r_lo, r_hi)
tmask_f   = (tmask > 0).astype(np.float32)
tmpl_f    = _extract_patch(gray_d, src['x'], src['y'], tmpl_r_px).astype(np.float32) / 255.0
print(f'Template-kilde: ({src["x"]:.0f},{src["y"]:.0f}) d={np.hypot(src["x"]-cx0,src["y"]-cy0)/delta:.2f}*delta')

# ROI
x0 = max(0, int(cx0 - scan_r) - tmpl_r_px)
y0 = max(0, int(cy0 - scan_r) - tmpl_r_px)
x1 = min(gray_d.shape[1], int(cx0 + scan_r) + tmpl_r_px + 1)
y1 = min(gray_d.shape[0], int(cy0 + scan_r) + tmpl_r_px + 1)
roi   = gray_d[y0:y1, x0:x1]
roi_f = roi.astype(np.float32) / 255.0
corr  = cv2.matchTemplate(roi_f, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=tmask_f)
print(f'NCC-kart: shape={corr.shape} min={corr.min():.3f} max={corr.max():.3f}')

# NMS og terskel
min_dist = cfg['hit_min_dist_frac'] * marker_r
ncc_thresh = cfg['hit_center_ncc_thresh']
nms_k  = _odd(min_dist)
kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_k, nms_k))
dil    = cv2.dilate(corr, kern)
ry_arr, rx_arr = np.where((corr == dil) & (corr >= ncc_thresh))
candidates = []
for ry, rx in zip(ry_arr, rx_arr):
    X = float(x0 + rx + tmpl_r_px)
    Y = float(y0 + ry + tmpl_r_px)
    if np.hypot(X - cx0, Y - cy0) > scan_r:
        continue
    if any(np.hypot(X-h['x'], Y-h['y']) < min_dist for h in hits8):
        continue
    candidates.append({'x': X, 'y': Y, 'ncc': float(corr[ry, rx])})

# Dedup mot hverandre
candidates.sort(key=lambda c: -c['ncc'])
new_hits = []
for c in candidates:
    if all(np.hypot(c['x']-k['x'], c['y']-k['y']) >= min_dist for k in new_hits):
        new_hits.append(c)
print(f'Sentrum-sveip: {len(new_hits)} nye kandidater (ncc_thresh={ncc_thresh})')
for c in new_hits:
    d = np.hypot(c['x']-cx0, c['y']-cy0)
    print(f"  ({c['x']:.0f},{c['y']:.0f}) d={d:.1f}px={d/delta:.2f}*delta NCC={c['ncc']:.3f}")

# ── 1. NCC-KART over bull-sonen ──────────────────────────────────────────────
# Legg NCC-kartet tilbake i bildets koordinatsystem
ncc_canvas = np.full(gray_d.shape, -1.0, np.float32)
for ry in range(corr.shape[0]):
    for rx in range(corr.shape[1]):
        X = x0 + rx + tmpl_r_px; Y = y0 + ry + tmpl_r_px
        if np.hypot(X - cx0, Y - cy0) <= scan_r:
            ncc_canvas[Y, X] = corr[ry, rx]

valid = ncc_canvas >= -0.5
vmin  = float(ncc_canvas[valid].min()) if valid.any() else -0.5
vmax  = float(ncc_canvas[valid].max()) if valid.any() else 1.0
norm  = np.zeros_like(ncc_canvas)
norm[valid] = np.clip((ncc_canvas[valid] - vmin) / max(vmax - vmin, 1e-6), 0, 1)
heat  = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
# Piksler utenfor bull-sonen = mork graa
heat[~valid] = [25, 25, 25]

# Tegn eksisterende treff (hvite) og nye kandidater (cyan)
for h in hits8:
    pt = (int(round(h['x'])), int(round(h['y'])))
    cv2.circle(heat, pt, int(round(marker_r)), (255, 255, 255), 2)
for c in new_hits:
    pt = (int(round(c['x'])), int(round(c['y'])))
    cv2.circle(heat, pt, int(round(marker_r)), (0, 255, 255), 3)
    cv2.putText(heat, f"NCC={c['ncc']:.2f}", (pt[0]+6, pt[1]-6),
                0, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
cv2.circle(heat, (int(round(cx0)), int(round(cy0))), int(round(scan_r)), (0, 200, 255), 1)
cv2.putText(heat, f"NCC-sveipkart bull-sone (R10={R10:.0f}px)",
            (20, 35), 0, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
cv2.imwrite(str(OUT / 'C10_center_scan.png'), heat)
print('Lagret C10_center_scan.png')

# ── 2. FINALE TREFF ───────────────────────────────────────────────────────────
final_img = img_d.copy()
for h in hits8:
    pt = (int(round(h['x'])), int(round(h['y'])))
    cv2.circle(final_img, pt, int(round(marker_r)), (0, 200, 0), 2)
for c in new_hits:
    pt = (int(round(c['x'])), int(round(c['y'])))
    cv2.circle(final_img, pt, int(round(marker_r)), (0, 255, 255), 3)
    cv2.arrowedLine(final_img, (pt[0]+90, pt[1]-70), (pt[0]+10, pt[1]-10),
                    (0, 255, 255), 2, tipLength=0.3)
    cv2.putText(final_img, f"NCC={c['ncc']:.2f}", (pt[0]+95, pt[1]-75),
                0, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
cv2.circle(final_img, (int(round(cx0)), int(round(cy0))), int(round(scan_r)),
           (0, 200, 255), 1)
total = len(hits8) + len(new_hits)
cv2.putText(final_img, f"Finale treff: {total}  (+{len(new_hits)} sentrum-sveip)",
            (20, 35), 0, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
cv2.imwrite(str(OUT / 'C10_center_final.png'), final_img)
print('Lagret C10_center_final.png')
