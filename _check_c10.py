"""Sammenlign C10-hits med og uten sentrum-sveip."""
import cv2, numpy as np
from config import DEFAULT_CONFIG
import inspect_hits as ih
import circles as circle_det
from overlap import find_center_hits

cfg = DEFAULT_CONFIG.copy()
img = cv2.imread('Testsett/C10.jpg')
got = ih.prep(img, 'C10')
if got is None:
    print('C10: KALIB FEIL'); raise SystemExit(1)
img_d, gray_d, calib = got
cx0, cy0 = calib['center']
delta = calib['delta_px']; R10 = calib['R10_px']
search_r = cfg['hit_search_r_max_frac'] * (R10 + 9.0 * delta)
marker_r = cfg['hit_marker_radius_frac'] * delta
dot_r    = cfg['hit_dot_radius_frac'] * delta

cands = circle_det.detect_circles(gray_d, (cx0,cy0), search_r, marker_r, dot_r, cfg, inner_r=0.0)
hits  = [{'x':X,'y':Y,'score':sc,'type':'filled','hough_r':marker_r} for X,Y,sc in cands]
print(f'Hoveddetektor: {len(hits)} treff')

extra = find_center_hits(gray_d, hits, calib, cfg)
print(f'Sentrum-sveip: {len(extra)} nye treff')
for e in extra:
    d = np.hypot(e['x']-cx0, e['y']-cy0)
    print(f"  ({e['x']:.0f},{e['y']:.0f}) d={d:.1f}px={d/delta:.2f}*delta NCC={e['score']:.3f}")
print(f'Totalt: {len(hits)+len(extra)}')
