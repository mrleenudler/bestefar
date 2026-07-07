"""
Treff-inspeksjon: visualiserer hvert trinn i treffdeteksjonen (hits.detect_hits)
paa det DE-SKEWEDE bildet - samme ramme som scoringen bruker.

Per kandidat vises:
  * raa Hough-sirkel (tynn gul)            -> hvor Hough fant en sirkel
  * raffinert senter + markor-radius:
      gronn  = akseptert treff (score >= terskel, innen ROI)
      roed   = forkastet (score < terskel eller utenfor ROI)
  * score (beste av filled/outline) ved hver kandidat
  * sok-ROI (cyan sirkel) og senterkryss

Slik ser vi om et bortfalt skudd (f.eks. C5 sin 5.6) er en HOUGH-MISS
(ingen kandidat der) eller en SCORE-FORKASTELSE (kandidat finnes, men scorer lavt).

Fulloppl: <navn>_hits.png   Montasje: _HITS_montage.png
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
import hits as hits_mod
from Bestefar import detect_outer_circle

cfg = DEFAULT_CONFIG.copy()
OUT = Path("Visualiseringer/Hits_out")
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True, exist_ok=True)
CELL = 420


def prep(img, name):
    """Reproduser scoringsbanen frem til (de-skew BGR, de-skew gray, calib)."""
    cands = []
    res = screen.rectify_to_screen(img, cfg, [])
    if res is not None:
        cands.append(res[0])
    cands.append(img)
    for work in cands:
        try:
            cx, cy, r0, _ = detect_outer_circle(work, cfg, debug=False, filename=name)
        except Exception:
            continue
        g = preprocess.to_gray(work)
        calib = rings.calibrate_and_refine(g, (cx, cy), r0, cfg)
        if calib is None or not rings.validate_calibration(calib, cfg)[0]:
            continue
        fit = perspektiv.fit_rectification(calib, cfg)
        if fit is not None:
            H = fit[0]
            gray_d = perspektiv.warp_image(g, H)
            img_d = perspektiv.warp_image(work, H)
            c_rect = perspektiv.transform_points([calib['center']], H)[0]
            cfgr = cfg.copy(); cfgr['ring_refine_iters'] = cfg.get('ring_recal_iters', 3)
            calib2 = rings.calibrate_and_refine(gray_d, (c_rect[0], c_rect[1]),
                                                max(calib['ring_radii_px']), cfgr)
            if calib2 is not None and rings.validate_calibration(calib2, cfg)[0]:
                return img_d, gray_d, calib2
        return work.copy(), g, calib
    return None


def detect_instrumented(gray, calib, cfg):
    """Som hits.detect_hits, men returnerer alle kandidater med score + status."""
    cx0, cy0 = calib['center']
    delta = calib['delta_px']
    r1 = max(calib['ring_radii_px']) if calib['ring_radii_px'] else 10 * delta
    search_r = cfg.get('hit_search_r_max_frac', 1.05) * r1
    marker_r = cfg.get('hit_marker_radius_frac', 0.41) * delta
    dot_r = cfg.get('hit_dot_radius_frac', 0.105) * delta

    h, w = gray.shape
    x0 = max(0, int(cx0 - search_r)); x1 = min(w, int(cx0 + search_r) + 1)
    y0 = max(0, int(cy0 - search_r)); y1 = min(h, int(cy0 + search_r) + 1)
    roi = gray[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(roi, (0, 0), 2.0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=cfg.get('hit_hough_dp', 2.0),
        minDist=cfg.get('hit_min_dist_frac', 0.6) * marker_r,
        param1=cfg.get('hit_hough_param1', 120),
        param2=cfg.get('hit_hough_param2', 30),
        minRadius=int(0.78 * marker_r),
        maxRadius=int(1.22 * marker_r),
    )
    cands = []
    if circles is not None:
        for (x, y, r) in circles[0]:
            cands.append((float(x) + x0, float(y) + y0, float(r)))

    min_score = cfg.get('hit_validate_min_score', 0.35)
    recs = []
    for (x, y, r) in cands:
        d = np.hypot(x - cx0, y - cy0)
        in_roi = d <= search_r
        best = None
        for mtype in ('filled', 'outline'):
            rx, ry = hits_mod._refine_on_dot(gray, x, y, dot_r, mtype)
            if np.hypot(rx - x, ry - y) > dot_r * 1.5:
                rx, ry = x, y
            sf, so = hits_mod._marker_pattern_scores(gray, rx, ry, marker_r, dot_r)
            s = sf if mtype == 'filled' else so
            if best is None or s > best['score']:
                best = {'rx': rx, 'ry': ry, 'type': mtype, 'score': float(s),
                        'sf': float(sf), 'so': float(so)}
        rec = {'x': x, 'y': y, 'r': r, 'in_roi': in_roi}
        rec.update(best)
        rec['accepted'] = bool(in_roi and best['score'] >= min_score)
        recs.append(rec)

    # dedup blant aksepterte (samme regel som detect_hits)
    min_dist = cfg.get('hit_min_dist_frac', 0.6) * marker_r
    acc = sorted([r for r in recs if r['accepted']], key=lambda t: -t['score'])
    kept = []
    for t in acc:
        if all(np.hypot(t['rx'] - k['rx'], t['ry'] - k['ry']) >= min_dist for k in kept):
            kept.append(t); t['kept'] = True
        else:
            t['kept'] = False; t['accepted'] = False  # tapte dedup
    info = dict(search_r=search_r, marker_r=marker_r, dot_r=dot_r,
                center=(cx0, cy0), n_cand=len(cands), n_kept=len(kept), min_score=min_score)
    return recs, info


def draw(img_d, calib, recs, info):
    vis = img_d.copy() if img_d.ndim == 3 else cv2.cvtColor(img_d, cv2.COLOR_GRAY2BGR)
    cx, cy = info['center']
    ci = (int(round(cx)), int(round(cy)))
    mr = info['marker_r']
    cv2.circle(vis, ci, int(round(info['search_r'])), (255, 255, 0), 1)   # ROI cyan
    cv2.drawMarker(vis, ci, (255, 255, 0), cv2.MARKER_CROSS, 24, 1)
    near = 0.05    # "nesten-treff": plausibel markor som sklei under terskel
    for rc in recs:
        ref = (int(round(rc['rx'])), int(round(rc['ry'])))
        if rc.get('kept'):                                   # akseptert treff
            cv2.circle(vis, ref, int(round(mr)), (0, 255, 0), 2)
            cv2.putText(vis, f"{rc['type'][0]}{rc['score']:.2f}", (ref[0] + 6, ref[1] - 6),
                        0, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        elif rc['score'] >= near:                            # nesten-treff (oransje)
            cv2.circle(vis, ref, int(round(mr)), (0, 165, 255), 1)
            cv2.putText(vis, f"{rc['score']:.2f}", (ref[0] + 6, ref[1] - 6),
                        0, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
        else:                                                # tydelig ikke-markor (svak prikk)
            cv2.circle(vis, ref, 2, (0, 0, 200), -1)
    return vis


def fit_cell(img, label, T=CELL):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    s = min(T / img.shape[0], T / img.shape[1])
    v = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
    c = np.full((T, T, 3), 30, np.uint8)
    c[:v.shape[0], :v.shape[1]] = v
    cv2.putText(c, label, (4, 16), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    return c


def main():
  names = [("Real 1", Path("Real 1.jpg"))]
  for i in range(1, 11):
    p = Path("Testsett") / f"C{i}.jpg"
    if p.exists():
        names.append((f"C{i}", p))

  cells = []
  for name, path in names:
    print(f"prosesserer {name} ...")
    img = cv2.imread(str(path))
    if img is None:
        continue
    got = prep(img, name)
    if got is None:
        cells.append(fit_cell(img, f"{name} KALIB FEIL"))
        continue
    img_d, gray_d, calib = got
    recs, info = detect_instrumented(gray_d, calib, cfg)
    vis = draw(img_d, calib, recs, info)
    cv2.imwrite(str(OUT / f"{name}_hits.png"), vis)
    lbl = f"{name}  kand={info['n_cand']} treff={info['n_kept']}"
    cells.append(fit_cell(vis, lbl))
    print(f"  {lbl}  (terskel score={info['min_score']})")

  per_row = 4
  rows = []
  for i in range(0, len(cells), per_row):
    row = cells[i:i + per_row]
    while len(row) < per_row:
        row.append(np.full((CELL, CELL, 3), 30, np.uint8))
    rows.append(np.hstack(row))
  cv2.imwrite(str(OUT / "_HITS_montage.png"), np.vstack(rows))
  print(f"ferdig -> {OUT}")


if __name__ == '__main__':
    main()
