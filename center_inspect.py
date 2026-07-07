"""
Ring-inspeksjon (v4): viser tre ting per bilde (på det DE-SKEWEDE bildet):
  1) HELE det ekstrapolerte, NUMMERERTE ringsettet (1-10) fra R10/delta
     (gul, ring 1 = magenta) - viser R10-forankringen: korrekt nummerering +
     komplett ringsett selv om noen ringer ikke ble detektert.
  2) Radial-histogram FØR glatting (rå gradient) og ETTER glatting (pipeline),
     med detekterte topper (grønn) og valgt comb (rød tann) på den glattede.
  3) Endelig korrigert (de-skew) polarbilde.

Fulloppløst: <navn>_rings.png (nummerert overlay), <navn>_polar_deskew.png
"""
import shutil
import cv2
import numpy as np
from pathlib import Path
from config import DEFAULT_CONFIG
import screen
import rings
import preprocess
import perspektiv
from Bestefar import detect_outer_circle

cfg = DEFAULT_CONFIG.copy()          # live config (comb_v2=False, pre_blur=1.5, generøs r_max, R10-anker)
OUT = Path("Visualiseringer/Center_out")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
THETA = 720
CELL = 260


def norm8(a):
    a = a.astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    return np.zeros(a.shape, np.uint8) if mx - mn < 1e-6 else ((a - mn) / (mx - mn) * 255).astype(np.uint8)


def fit_cell(img, label, T=CELL, color=(0, 255, 255)):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = min(T / img.shape[0], T / img.shape[1])
    v = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    c = np.full((T, T, 3), 30, np.uint8)
    c[:v.shape[0], :v.shape[1]] = v
    cv2.putText(c, label, (4, 16), 0, 0.42, color, 1, cv2.LINE_AA)
    return c


def hist_cell(H, label, rho_scale, peaks_px=None, teeth_px=None, T=CELL):
    n = len(H)
    img = np.full((T, T, 3), 18, np.uint8)
    Hn = H / (H.max() + 1e-9) * (T - 26)
    xs = (np.arange(n) / n * (T - 1)).astype(int)
    for i in range(n):
        cv2.line(img, (xs[i], T - 1), (xs[i], T - 1 - int(Hn[i])), (170, 170, 170), 1)
    if teeth_px:
        for r in teeth_px:
            x = int((r / rho_scale) / n * (T - 1))
            cv2.line(img, (x, 0), (x, T - 1), (0, 0, 255), 1)        # comb-tann rød
    if peaks_px:
        for r in peaks_px:
            x = int((r / rho_scale) / n * (T - 1))
            cv2.line(img, (x, T - 30), (x, T - 1), (0, 255, 0), 1)   # topp grønn
    cv2.putText(img, label, (4, 14), 0, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "radius ->", (T - 70, T - 6), 0, 0.34, (180, 180, 180), 1)
    return img


def profile(gray, center, r_max, pre_blur, sigma_rho):
    rs = int(round(r_max)); rho = r_max / rs
    g = cv2.GaussianBlur(gray, (0, 0), pre_blur) if pre_blur > 0 else gray
    P = rings.warp_polar_gray(g, center, r_max, rs, THETA)
    H = rings.radial_gradient_abs(P, sigma_rho).mean(axis=0)
    return H, P, rho


def analyze_one(img, name):
    cands = []
    res = screen.rectify_to_screen(img, cfg, [])
    if res is not None:
        cands.append(res[0])
    cands.append(img)
    last = None
    for work in cands:
        try:
            cx, cy, r0, _ = detect_outer_circle(work, cfg, debug=False, filename=name)
        except Exception:
            continue
        g = preprocess.to_gray(work)
        calib = rings.calibrate_and_refine(g, (cx, cy), r0, cfg)
        last = (work, g, calib)
        if calib is not None:
            return last
    return last


def process(name, path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    got = analyze_one(img, name)
    if got is None or got[2] is None:
        return fit_cell(img, f"{name} KALIB FEIL", color=(0, 0, 255))
    work, gray, calib = got

    # De-skew + rekalibrer i rettet ramme
    fr = perspektiv.fit_rectification(calib, cfg)
    if fr is not None:
        Hmg = fr[0]
        img_d = perspektiv.warp_image(work, Hmg)
        gray_d = preprocess.to_gray(img_d)
        c_d = perspektiv.transform_points([calib['center']], Hmg)[0]
        cu = rings.calibrate_and_refine(gray_d, (c_d[0], c_d[1]), max(calib['ring_radii_px']), cfg) or calib
    else:
        img_d, gray_d, cu = work, gray, calib

    cc = (float(cu['center'][0]), float(cu['center'][1]))
    R10, dlt = cu['R10_px'], cu['delta_px']
    r_max = 1.12 * (R10 + 9 * dlt)

    # 1) Nummerert, ekstrapolert ringsett 1-10
    ov = img_d.copy()
    ci = (int(round(cc[0])), int(round(cc[1])))
    detected = set(int(round(v)) for v in cu.get('ring_values', []))
    for v in range(1, 11):
        r = R10 + (10 - v) * dlt
        if r <= 0:
            continue
        if v == 1:
            col = (255, 0, 255)        # magenta = ytterste ring
        elif v in detected:
            col = (0, 255, 0)          # gronn = detektert ring
        else:
            col = (255, 255, 0)        # cyan = ekstrapolert ring
        cv2.circle(ov, ci, int(round(r)), col, 3)
        ty = int(round(cc[1] - r))
        cv2.putText(ov, str(v), (ci[0] + 3, ty + 14), 0, 0.5, col, 1, cv2.LINE_AA)
    cv2.drawMarker(ov, ci, (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
    cv2.imwrite(str(OUT / f"{name}_rings.png"), ov)

    # 2) Histogrammer før/etter glatting (i de-skew ramme)
    H_raw, _, rho = profile(gray_d, cc, r_max, pre_blur=0.0, sigma_rho=0.7)
    H_sm, P_d, _ = profile(gray_d, cc, r_max, pre_blur=cfg['ring_pre_blur_sigma'],
                           sigma_rho=cfg['ring_rho_sigma'])
    d0 = rings.estimate_spacing_autocorr(H_sm, int(0.02 * len(H_sm)), int(0.30 * len(H_sm)))
    pk = rings.find_profile_peaks(H_sm, cfg['ring_peak_min_frac'], min_sep=max(3, int(0.3 * (d0 or 40))))
    peaks_px = [p[0] * rho for p in pk if p[0] > 0.4 * (d0 or 40)]
    teeth_px = list(cu['ring_radii_px'])
    cv2.imwrite(str(OUT / f"{name}_polar_deskew.png"), norm8(P_d))

    status = f"{name} d={dlt:.0f} R10={R10:.0f} n={len(cu['ring_radii_px'])}"
    row = np.hstack([
        fit_cell(ov, status),
        hist_cell(H_raw, "FOR glatting (raa)", rho),
        hist_cell(H_sm, "ETTER glatting +topper+comb", rho, peaks_px=peaks_px, teeth_px=teeth_px),
        fit_cell(norm8(P_d), "polar @ de-skew (korrigert)"),
    ])
    return row


names = [("Real 1", Path("Real 1.jpg"))]
for grp, cnt in [("C", 10), ("T", 9)]:
    for i in range(1, cnt + 1):
        p = Path("Testsett") / f"{grp}{i}.jpg"
        if p.exists():
            names.append((f"{grp}{i}", p))

rows = []
for name, path in names:
    print(f"prosesserer {name} ...")
    r = process(name, path)
    if r is not None:
        rows.append(r)
W = max(r.shape[1] for r in rows)
rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 30, np.uint8)]) if r.shape[1] < W else r for r in rows]
cv2.imwrite(str(OUT / "_RINGS_montage.png"), np.vstack(rows))
print(f"ferdig -> {OUT}")
