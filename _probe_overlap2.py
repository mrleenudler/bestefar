"""Diagnostikk: kjoer overlapp-pass manuelt, vis hvorfor (1413,941) godtas/avvises."""
import cv2, numpy as np
from config import DEFAULT_CONFIG
import inspect_hits as ih, circles as circle_det

cfg = DEFAULT_CONFIG.copy()
img = cv2.imread('Testsett/C3.jpg')
got = ih.prep(img, 'C3')
img_d, gray_d, calib = got
cx0, cy0 = calib['center']
delta = calib['delta_px']; R10 = calib['R10_px']
search_r = cfg['hit_search_r_max_frac'] * (R10 + 9.0 * delta)
marker_r = cfg['hit_marker_radius_frac'] * delta
dot_r    = cfg['hit_dot_radius_frac'] * delta

# Simuler hits fra circle_det (8 treff)
cands = circle_det.detect_circles(gray_d, (cx0,cy0), search_r, marker_r, dot_r, cfg, inner_r=0.0)
hits  = [{'x':X,'y':Y,'score':sc,'type':'filled','hough_r':marker_r} for X,Y,sc in cands]
print(f'{len(hits)} treff, marker_r={marker_r:.1f}')

# Stemmekart
acc, dbg = circle_det.circle_vote_map(gray_d,(cx0,cy0),search_r,marker_r,dot_r,cfg,inner_r=0.0)
ox, oy = dbg['offset']; amax = float(acc.max())
vote_floor = cfg['hit_overlap_vote_frac'] * amax
print(f'amax={amax:.1f} vote_floor={vote_floor:.1f} ({cfg["hit_overlap_vote_frac"]}*amax)')

# Liten NMS
from overlap import _odd, _extract_patch, _annular_mask
nms_r  = _odd(cfg['hit_overlap_nms_frac'] * marker_r)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_r, nms_r))
dil    = cv2.dilate(acc, kernel)
py_all, px_all = np.where((acc == dil) & (acc >= vote_floor))
print(f'Lokale maxima med NMS={nms_r}px over {vote_floor:.1f}: {len(px_all)} totalt')

# NCC-template
def isolation(h):
    others = [o for o in hits if o is not h]
    return min(np.hypot(h['x']-o['x'],h['y']-o['y']) for o in others)*h['score'] if others else 0
src = max(hits, key=isolation)
r_lo = cfg['hit_overlap_tmpl_r_lo'] * marker_r
r_hi = cfg['hit_overlap_tmpl_r_hi'] * marker_r
tmpl_r = int(r_hi) + 2
tmask  = _annular_mask(tmpl_r, r_lo, r_hi)
tmask_f = (tmask > 0).astype(np.float32)
tmpl_f  = _extract_patch(gray_d, src['x'], src['y'], tmpl_r).astype(np.float32)/255.0
print(f'Template-kilde: ({src["x"]:.0f},{src["y"]:.0f}), tmpl_r={tmpl_r}, r_lo={r_lo:.1f} r_hi={r_hi:.1f}')

# Sjekk spesifikt peak (1413,941)
target_X, target_Y = 1413.0, 941.0
max_dist   = cfg['hit_overlap_max_dist_frac'] * marker_r
min_offset = cfg['hit_overlap_min_offset_frac'] * marker_r
min_dist_other = cfg['hit_min_dist_frac'] * marker_r
max_anchor_r   = cfg['hit_overlap_max_anchor_r_frac'] * delta
anchors = [h for h in hits if np.hypot(h['x']-cx0,h['y']-cy0) <= max_anchor_r]
print(f'\n{len(anchors)} ankre i bull-sonen')

for anchor in anchors:
    ax, ay = anchor['x'], anchor['y']
    d_anker = np.hypot(target_X-ax, target_Y-ay)
    print(f'\nAnker ({ax:.0f},{ay:.0f}) d/delta={np.hypot(ax-cx0,ay-cy0)/delta:.2f}:')
    print(f'  d_anker={d_anker:.1f}  min_offset={min_offset:.1f}  max_dist={max_dist:.1f}')
    print(f'  offset-sjekk: {"PASS" if min_offset <= d_anker <= max_dist else "FAIL"}')

    # too_close mot andre?
    for k in hits:
        if k is anchor: continue
        dk = np.hypot(target_X-k['x'], target_Y-k['y'])
        if dk < min_dist_other:
            print(f'  TOO_CLOSE mot ({k["x"]:.0f},{k["y"]:.0f}) dk={dk:.1f} < {min_dist_other:.1f}')

    # NCC
    if min_offset <= d_anker <= max_dist:
        patch = _extract_patch(gray_d, target_X, target_Y, tmpl_r)
        cand_f = patch.astype(np.float32)/255.0
        print(f'  patch.shape={patch.shape}  tmpl_f.shape={tmpl_f.shape}')
        res = cv2.matchTemplate(cand_f, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=tmask_f)
        ncc = float(res.max()) if res.size>0 else 0.0
        print(f'  NCC={ncc:.3f}  terskel={cfg["hit_overlap_ncc_thresh"]}  => {"PASS" if ncc>=cfg["hit_overlap_ncc_thresh"] else "FAIL"}')

# Er (1413,941) i peak-lista?
TARGET_px = int(target_X) - ox
TARGET_py = int(target_Y) - oy
in_peaks = any(px==TARGET_px and py==TARGET_py for py,px in zip(py_all,px_all))
print(f'\nEr ({target_X:.0f}-ox={TARGET_px},{target_Y:.0f}-oy={TARGET_py}) i peak-lista? {in_peaks}')
# Naermeste peak
best_d = min((abs(px-TARGET_px)+abs(py-TARGET_py), px, py) for py,px in zip(py_all,px_all))
print(f'Naermeste peak: acc_coord=({best_d[1]},{best_d[2]}) Manhattan-dist={best_d[0]}')
