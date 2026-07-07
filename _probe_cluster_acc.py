"""
Vis raa stemmekart-verdier naer klynge-treffet (1393,949) uten NMS-filtrering,
for aa finne det undertrykte overlapp-treffet.
"""
import cv2, numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import inspect_hits as ih, circles

cfg = DEFAULT_CONFIG.copy()
OUT = Path('Visualiseringer/outputs')

img = cv2.imread('Testsett/C3.jpg')
got = ih.prep(img, 'C3')
img_d, gray_d, calib = got
cx, cy = calib['center']
delta = calib['delta_px']
R10  = calib['R10_px']
search_r = cfg['hit_search_r_max_frac'] * (R10 + 9.0 * delta)
marker_r = cfg['hit_marker_radius_frac'] * delta
dot_r    = cfg['hit_dot_radius_frac'] * delta

acc, dbg = circles.circle_vote_map(
    gray_d, (cx, cy), search_r, marker_r, dot_r, cfg, inner_r=0.0)
ox, oy = dbg['offset']
amax = float(acc.max())
print(f'amax={amax:.1f}  center=({cx:.0f},{cy:.0f})  marker_r={marker_r:.1f}')

# Anker = detektert klynge-treff
ax, ay = 1393.0, 949.0

# Finn ALLE lokale maxima med LITEN NMS (radius = 8px) innen 1.5*marker_r
nms_r = 8
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*nms_r+1, 2*nms_r+1))
dil    = cv2.dilate(acc, kernel)
py_all, px_all = np.where((acc == dil) & (acc >= 0.05 * amax))

print(f'\nAlle lokale maxima (NMS={nms_r}px) innen 1.5*marker_r={1.5*marker_r:.0f}px av ankeret:')
nearby = []
for py, px in zip(py_all, px_all):
    X = px + ox; Y = py + oy
    d_anker  = np.hypot(X - ax, Y - ay)
    d_center = np.hypot(X - cx, Y - cy)
    if d_anker > 1.5 * marker_r:
        continue
    v = acc[py, px]
    nearby.append((v, X, Y, d_anker, d_center))

nearby.sort(reverse=True)
for v, X, Y, da, dc in nearby:
    score_est = 10 + (R10 + 0.14*delta - dc) / delta
    known = da < 5
    print(f'  ({X:.0f},{Y:.0f})  v={v:.1f} ({v/amax:.2f}*amax)'
          f'  d_anker={da:.0f}px  d/delta={dc/delta:.2f}'
          f'  poeng~{score_est:.1f}{"  <- ANKER" if known else ""}')

# Lagre heat-bilde av klynge-omraadet med LITEN NMS-topper merket
zoom = int(2.0 * marker_r)
axi, ayi = int(ax) - ox, int(ay) - oy
accv = np.clip(acc / amax, 0, 1.0)
heat = cv2.applyColorMap((accv * 255).astype(np.uint8), cv2.COLORMAP_JET)
for v, X, Y, da, dc in nearby:
    xi, yi = int(X) - ox, int(Y) - oy
    col = (0,255,0) if v >= dbg['thr'] else (0,165,255) if v >= 0.15*amax else (180,180,255)
    cv2.circle(heat, (xi, yi), 6, col, 2)
    cv2.putText(heat, f'{v/amax:.2f}', (xi+4, yi-4), 0, 0.5, col, 1)
crop = heat[max(0,ayi-zoom):ayi+zoom, max(0,axi-zoom):axi+zoom]
cv2.imwrite(str(OUT / 'C3_cluster_acc_smallNMS.png'), crop)
print(f'\nLagret C3_cluster_acc_smallNMS.png')
