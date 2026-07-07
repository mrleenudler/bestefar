"""Diagnostikk: test halvt annulaer NCC-maske mot (1413,941) overlapp-kandidat."""
import cv2, numpy as np
from config import DEFAULT_CONFIG
import inspect_hits as ih, circles as circle_det
from overlap import _odd, _extract_patch, _annular_mask, _half_annular_mask, _crescent_mask

cfg = DEFAULT_CONFIG.copy()
img = cv2.imread('Testsett/C3.jpg')
got = ih.prep(img, 'C3')
img_d, gray_d, calib = got
cx0, cy0 = calib['center']
delta = calib['delta_px']; R10 = calib['R10_px']
search_r = cfg['hit_search_r_max_frac'] * (R10 + 9.0 * delta)
marker_r = cfg['hit_marker_radius_frac'] * delta
dot_r    = cfg['hit_dot_radius_frac'] * delta

cands = circle_det.detect_circles(gray_d, (cx0,cy0), search_r, marker_r, dot_r, cfg, inner_r=0.0)
hits  = [{'x':X,'y':Y,'score':sc,'type':'filled','hough_r':marker_r} for X,Y,sc in cands]
print(f'{len(hits)} treff, marker_r={marker_r:.1f}')

# Template-kilde: mest isolert treff
def isolation(h):
    others = [o for o in hits if o is not h]
    return min(np.hypot(h['x']-o['x'],h['y']-o['y']) for o in others)*h['score'] if others else 0

src    = max(hits, key=isolation)
r_lo   = cfg['hit_overlap_tmpl_r_lo'] * marker_r
r_hi   = cfg['hit_overlap_tmpl_r_hi'] * marker_r
tmpl_r = int(r_hi) + 2
tmpl_patch = _extract_patch(gray_d, src['x'], src['y'], tmpl_r)
tmpl_f     = tmpl_patch.astype(np.float32) / 255.0
print(f'Template-kilde: ({src["x"]:.0f},{src["y"]:.0f}), tmpl_r={tmpl_r}, r_lo={r_lo:.1f} r_hi={r_hi:.1f}')

# Maalkandidat
target_X, target_Y = 1413.0, 941.0
ax, ay = 1393.0, 949.0   # Naermeste anker (dominerende 10.6-treff)
dx, dy = target_X - ax, target_Y - ay
norm   = np.hypot(dx, dy)
dx_n, dy_n = dx / norm, dy / norm
print(f'\nKandidat ({target_X:.0f},{target_Y:.0f})  anker ({ax:.0f},{ay:.0f})')
print(f'Retning bort fra anker: ({dx_n:.3f},{dy_n:.3f})  norm={norm:.1f}px')

cand_patch = _extract_patch(gray_d, target_X, target_Y, tmpl_r)
cand_f     = cand_patch.astype(np.float32) / 255.0

print(f'r_lo={r_lo:.1f}px  r_hi={r_hi:.1f}px  tmpl_r={tmpl_r}')
print()

# --- A: full annulaer, r_lo=0.20 ---
full_mask   = _annular_mask(tmpl_r, r_lo, r_hi)
full_mask_f = (full_mask > 0).astype(np.float32)
res = cv2.matchTemplate(cand_f, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=full_mask_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'[A] Full annulaer r_lo=0.20   piksler={int(full_mask_f.sum()):4d}  NCC={ncc:+.3f}')

# --- B: halvt annulaer, r_lo=0.20 (naa)  ---
half_mask   = _half_annular_mask(tmpl_r, r_lo, r_hi, dx_n, dy_n)
half_mask_f = (half_mask > 0).astype(np.float32)
res = cv2.matchTemplate(cand_f, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=half_mask_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'[B] Halvt annulaer r_lo=0.20  piksler={int(half_mask_f.sum()):4d}  NCC={ncc:+.3f}')

# --- C: halvt annulaer, r_lo=0.50 (gammel) ---
r_lo_50 = 0.50 * marker_r
half_mask50   = _half_annular_mask(tmpl_r, r_lo_50, r_hi, dx_n, dy_n)
half_mask50_f = (half_mask50 > 0).astype(np.float32)
tmpl_50 = _extract_patch(gray_d, src['x'], src['y'], tmpl_r).astype(np.float32)/255.0
res = cv2.matchTemplate(cand_f, tmpl_50, cv2.TM_CCOEFF_NORMED, mask=half_mask50_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'[C] Halvt annulaer r_lo=0.50  piksler={int(half_mask50_f.sum()):4d}  NCC={ncc:+.3f}')

# --- D: halvt annulaer, adaptiv r_lo = offset * 0.90 ---
r_lo_adapt = max(0.20*marker_r, norm * 0.85)
half_adap   = _half_annular_mask(tmpl_r, r_lo_adapt, r_hi, dx_n, dy_n)
half_adap_f = (half_adap > 0).astype(np.float32)
res = cv2.matchTemplate(cand_f, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=half_adap_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'[D] Halvt adaptiv r_lo={r_lo_adapt:.1f}px={r_lo_adapt/marker_r:.2f}  piksler={int(half_adap_f.sum()):4d}  NCC={ncc:+.3f}')

# --- E: tynn ytterring halvt, r_lo=0.85 ---
r_lo_thin = 0.85 * marker_r
half_thin   = _half_annular_mask(tmpl_r, r_lo_thin, r_hi, dx_n, dy_n)
half_thin_f = (half_thin > 0).astype(np.float32)
tmpl_thin   = _extract_patch(gray_d, src['x'], src['y'], tmpl_r).astype(np.float32)/255.0
res = cv2.matchTemplate(cand_f, tmpl_thin, cv2.TM_CCOEFF_NORMED, mask=half_thin_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'[E] Halvt tynn ytterkant r_lo=0.85  piksler={int(half_thin_f.sum()):4d}  NCC={ncc:+.3f}')

print(f'\nNCC-terskel = {cfg["hit_overlap_ncc_thresh"]:.2f}')

# --- F: enda tynnere ytterkant r_lo=0.90 ---
for rlo_frac in [0.90, 0.93, 0.96]:
    r_lo_f = rlo_frac * marker_r
    m   = _half_annular_mask(tmpl_r, r_lo_f, r_hi, dx_n, dy_n)
    m_f = (m > 0).astype(np.float32)
    tp  = _extract_patch(gray_d, src['x'], src['y'], tmpl_r).astype(np.float32)/255.0
    res = cv2.matchTemplate(cand_f, tp, cv2.TM_CCOEFF_NORMED, mask=m_f)
    ncc = float(res.max()) if res.size > 0 else 0.0
    print(f'[F] Halvt r_lo={rlo_frac:.2f}  piksler={int(m_f.sum()):4d}  NCC={ncc:+.3f}')

# --- G: geometrisk korrekt synlig maanesigd-maske ---
# Piksler INNI kandidat-skiven OG UTENFOR dominant-skiven
print()
size = 2*tmpl_r + 1
yy, xx = np.mgrid[:size, :size]
cx = cy = tmpl_r
# Dominant disc center relativt til kandidat-senter (i piksler)
dom_dx = ax - target_X   # = 1393 - 1413 = -20
dom_dy = ay - target_Y   # = 949  - 941  =  +8
inside_cand  = ((xx - cx)**2 + (yy - cy)**2) <= r_hi**2
outside_dom  = ((xx - cx - dom_dx)**2 + (yy - cy - dom_dy)**2) >= marker_r**2
crescent_mask = (inside_cand & outside_dom).astype(np.uint8) * 255
# Legg til r_lo-ring gulv
inside_rlo   = ((xx - cx)**2 + (yy - cy)**2) <= r_lo**2
crescent_mask[inside_rlo] = 0
cres_f = (crescent_mask > 0).astype(np.float32)
tp = _extract_patch(gray_d, src['x'], src['y'], tmpl_r).astype(np.float32)/255.0
res = cv2.matchTemplate(cand_f, tp, cv2.TM_CCOEFF_NORMED, mask=cres_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'[G] Sann maanesigd (inni cand, utenfor dom, r>{r_lo:.0f}px)  piksler={int(cres_f.sum()):4d}  NCC={ncc:+.3f}')

# --- H: ekte maanesigd + kun ytre halvdel (r_lo=0.50) ---
inside_rlo50 = ((xx - cx)**2 + (yy - cy)**2) <= (0.50*marker_r)**2
cres50 = crescent_mask.copy(); cres50[inside_rlo50] = 0
cres50_f = (cres50 > 0).astype(np.float32)
res = cv2.matchTemplate(cand_f, tp, cv2.TM_CCOEFF_NORMED, mask=cres50_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'[H] Sann maanesigd r_lo=0.50  piksler={int(cres50_f.sum()):4d}  NCC={ncc:+.3f}')

# Lagre patch-visualiseringer for inspeksjon
import pathlib
OUT = pathlib.Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(OUT / 'probe3_crescent_mask.png'), crescent_mask)
cv2.imwrite(str(OUT / 'probe3_crescent50_mask.png'), cres50)

# Lagre patch med true crescent mask overlay
def _show_patch(patch, mask, name):
    vis = patch.copy()
    overlay = np.zeros_like(vis)
    overlay[mask > 0] = 180
    combined = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    cv2.imwrite(str(OUT / name), combined)

_show_patch(tmpl_patch, crescent_mask, 'probe3_template_crescent.png')
_show_patch(cand_patch, crescent_mask, 'probe3_cand_crescent.png')

# --- I: bruk _crescent_mask() funksjonen direkte ---
dom_dx = ax - target_X   # -20
dom_dy = ay - target_Y   # +8
cres_new = _crescent_mask(tmpl_r, 0.20*marker_r, r_hi, dom_dx, dom_dy, marker_r)
cres_new_f = (cres_new > 0).astype(np.float32)
tp = _extract_patch(gray_d, src['x'], src['y'], tmpl_r).astype(np.float32)/255.0
res = cv2.matchTemplate(cand_f, tp, cv2.TM_CCOEFF_NORMED, mask=cres_new_f)
ncc = float(res.max()) if res.size > 0 else 0.0
print(f'\n[I] _crescent_mask(dom_dx={dom_dx:.0f},dom_dy={dom_dy:.0f})  piksler={int(cres_new_f.sum()):4d}  NCC={ncc:+.3f}  => {"PASS" if ncc >= 0.10 else "FAIL"} (terskel=0.10)')

print(f'Lagret patch-bilder i Visualiseringer/outputs/')
