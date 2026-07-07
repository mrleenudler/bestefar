"""Debug: kjoer find_overlap_hits direkte paa C3, med verbose output."""
import cv2, numpy as np
from config import DEFAULT_CONFIG
import inspect_hits as ih, circles as circle_det
from overlap import (find_overlap_hits, _odd, _extract_patch,
                     _annular_mask, _crescent_mask)

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
print(f'{len(hits)} treff:')
for h in hits:
    d = np.hypot(h['x']-cx0, h['y']-cy0)
    print(f"  ({h['x']:.0f},{h['y']:.0f}) score={h['score']:.3f} d/delta={d/delta:.2f}")

# Kjor overlapp-pass og sjekk steg for steg
print(f'\nhit_overlap_pass = {cfg["hit_overlap_pass"]}')
print(f'trigger = {cfg["hit_overlap_trigger"]}  =>  kjorer: {len(hits) < cfg["hit_overlap_trigger"]}')

acc, dbg = circle_det.circle_vote_map(gray_d,(cx0,cy0),search_r,marker_r,dot_r,cfg,inner_r=0.0)
ox, oy = dbg['offset']; amax = float(acc.max())
vote_floor = cfg['hit_overlap_vote_frac'] * amax
nms_r = _odd(cfg['hit_overlap_nms_frac'] * marker_r)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_r, nms_r))
dil    = cv2.dilate(acc, kernel)
py_all, px_all = np.where((acc == dil) & (acc >= vote_floor))
print(f'\namax={amax:.1f}  vote_floor={vote_floor:.1f}  nms_r={nms_r}px')
print(f'Totale peaks: {len(px_all)}')

# Template
def isolation(h):
    others = [o for o in hits if o is not h]
    return min(np.hypot(h['x']-o['x'],h['y']-o['y']) for o in others)*h['score'] if others else 0

src    = max(hits, key=isolation)
r_lo   = cfg['hit_overlap_tmpl_r_lo'] * marker_r
r_hi   = cfg['hit_overlap_tmpl_r_hi'] * marker_r
tmpl_r = int(r_hi) + 2
tmpl_f = _extract_patch(gray_d, src['x'], src['y'], tmpl_r).astype(np.float32)/255.0
print(f'Template: ({src["x"]:.0f},{src["y"]:.0f})  r_lo={r_lo:.1f}  r_hi={r_hi:.1f}')

max_dist    = cfg['hit_overlap_max_dist_frac'] * marker_r
min_offset  = cfg['hit_overlap_min_offset_frac'] * marker_r
ncc_thresh  = cfg['hit_overlap_ncc_thresh']
min_dist_other = cfg['hit_min_dist_frac'] * marker_r
max_anchor_r   = cfg['hit_overlap_max_anchor_r_frac'] * delta

anchors = [h for h in hits if np.hypot(h['x']-cx0,h['y']-cy0) <= max_anchor_r]
print(f'\nAnkere i bull-sonen (max_r={max_anchor_r:.0f}px={max_anchor_r/delta:.1f}*delta): {len(anchors)}')

already = list(hits)
candidates_checked = 0
for anchor in anchors:
    ax, ay = anchor['x'], anchor['y']
    for py, px in zip(py_all, px_all):
        X = float(px + ox); Y = float(py + oy)
        d_anker = np.hypot(X-ax, Y-ay)
        if d_anker < min_offset or d_anker > max_dist:
            continue
        too_close = any(np.hypot(X-k['x'],Y-k['y']) < min_dist_other
                        for k in already if k is not anchor)
        if too_close:
            continue
        if np.hypot(X-cx0,Y-cy0) > search_r:
            continue

        candidates_checked += 1
        dom_dx = ax - X; dom_dy = ay - Y
        cmask   = _crescent_mask(tmpl_r, r_lo, r_hi, dom_dx, dom_dy, marker_r)
        cmask_f = (cmask > 0).astype(np.float32)
        active  = int(cmask_f.sum())
        if active < 10:
            print(f'  [{ax:.0f},{ay:.0f}]->[{X:.0f},{Y:.0f}] SKIP aktive_piksler={active} < 10')
            continue
        cand_patch = _extract_patch(gray_d, X, Y, tmpl_r)
        cand_f     = cand_patch.astype(np.float32)/255.0
        res = cv2.matchTemplate(cand_f, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=cmask_f)
        ncc = float(res.max()) if res.size > 0 else 0.0
        ok  = ncc >= ncc_thresh
        v   = float(acc[py, px])
        print(f'  [{ax:.0f},{ay:.0f}]->[{X:.0f},{Y:.0f}] v={v:.1f}({v/amax:.2f}) '
              f'd={d_anker:.1f} aktive={active} NCC={ncc:+.3f} => {"GODKJENT" if ok else "AVVIST"}')

print(f'\nTotalt kandidater sjekket: {candidates_checked}')
print(f'ncc_thresh={ncc_thresh}')
