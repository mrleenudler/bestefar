"""
Inspeksjon av den maalrettede sirkeldetektoren (circles.py) vs HoughCircles.
Viser mellomprodukter: ROI -> median -> CLAHE -> gradient -> stemmekart -> kandidater.
Sammenligner antall kandidater og om det ensomme ytre skuddet (hvit sone) fanges.

Output: Visualiseringer/outputs/circ_*.png
"""
import cv2
import numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import inspect_hits as ih      # gjenbruk prep() (de-skew + calib)

import os
cfg = DEFAULT_CONFIG.copy()
# Valgfrie overstyringer for avstoyings-eksperiment (PRE_GAUSS/MED_PASSES) + TAG
# slik at flere varianter kan ligge side om side i outputs/.
if os.environ.get('PRE_GAUSS'):
    cfg['circ_pre_gauss_frac'] = float(os.environ['PRE_GAUSS'])
if os.environ.get('MED_PASSES'):
    cfg['circ_median_passes'] = int(os.environ['MED_PASSES'])
TAG = os.environ.get('TAG', '')
import circles
OUT = Path("Visualiseringer/outputs")
OUT.mkdir(parents=True, exist_ok=True)
for f in OUT.glob(f"circ_*{TAG}.png"):
    f.unlink()
CELL = 920


def norm8(a):
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    return np.zeros(a.shape, np.uint8) if mx - mn < 1e-6 else ((a - mn) / (mx - mn) * 255).astype(np.uint8)


def cell(img, label, T=CELL, heat=False):
    if img.ndim == 2:
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET) if heat else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = min(T / img.shape[0], T / img.shape[1])
    v = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    c = np.full((T, T, 3), 30, np.uint8)
    c[:v.shape[0], :v.shape[1]] = v
    cv2.putText(c, label, (5, 18), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return c


def process(name):
    img = cv2.imread(str(Path("Testsett") / f"{name}.jpg")) if name != "Real 1" else cv2.imread("Real 1.jpg")
    got = ih.prep(img, name)
    if got is None:
        print(f"{name}: KALIB FEIL"); return
    img_d, gray_d, calib = got
    cx, cy = calib['center']; delta = calib['delta_px']
    search_r = cfg.get('hit_search_r_max_frac', 1.05) * (calib['R10_px'] + 9.0 * delta)
    marker_r = cfg.get('hit_marker_radius_frac', 0.41) * delta
    dot_r = cfg.get('hit_dot_radius_frac', 0.105) * delta

    inner_r = cfg.get('hit_unified_inner_frac', 0.0) * delta   # enhetlig detektor: HELE skiva
    cands, dbg = circles.detect_circles(gray_d, (cx, cy), search_r, marker_r, dot_r, cfg,
                                        inner_r=inner_r, return_dbg=True)

    # kandidat-overlay paa de-skew
    ov = img_d.copy()
    ci = (int(round(cx)), int(round(cy)))
    cv2.circle(ov, ci, int(round(search_r)), (255, 255, 0), 1)   # cyan = ytre soke-grense
    cv2.circle(ov, ci, int(round(inner_r)), (255, 120, 0), 1)    # blaa  = indre bull-grense
    cv2.drawMarker(ov, ci, (255, 255, 0), cv2.MARKER_CROSS, 24, 1)
    for (X, Y, sc) in cands:
        col = (0, 255, 0) if sc >= 0.6 else (0, 165, 255)
        cv2.circle(ov, (int(round(X)), int(round(Y))), int(round(marker_r)), col, 2)
        cv2.putText(ov, f"{sc:.2f}", (int(X) + 6, int(Y) - 6), 0, 0.5, col, 1, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"circ_{name}_cands{TAG}.png"), ov)

    # stemmekart skalert mot TERSKELEN: under terskel = moerkt, ekte topp = lyst
    accv = np.clip(dbg['acc'] / (dbg['thr'] + 1e-6), 0, 1.5) / 1.5
    accv = (accv * 255).astype(np.uint8)

    # mellomprodukt-montasje
    row = np.hstack([
        cell(dbg['roi'], "1 ROI (de-skew)"),
        cell(dbg['med'], f"2 gauss{cfg.get('circ_pre_gauss_frac',0.0):.2f}+med x{cfg.get('circ_median_passes',2)}"),
        cell(norm8(dbg['mag']), f"3 gradient (terskel p{cfg.get('circ_edge_pct',70):.0f})"),
        cell(accv, "4 stemmekart vs terskel", heat=True),
        cell(ov, f"5 kandidater: {len(cands)}"),
    ])
    cv2.imwrite(str(OUT / f"circ_{name}_stages{TAG}.png"), row)
    print(f"{name}: {len(cands)} kand (marker_r={marker_r:.1f} amax={dbg['amax']:.0f} terskel={dbg['thr']:.0f})")
    return row


if __name__ == '__main__':
    rows = []
    for i in range(1, 11):
        r = process(f"C{i}")
        if r is not None:
            rows.append(r)
    if rows:
        cv2.imwrite(str(OUT / f"circ_Cset_stages{TAG}.png"), np.vstack(rows))
    print(f"ferdig -> {OUT}")
