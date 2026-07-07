"""Print MPI-square oriented score + robusthet per C-bilde."""
import cv2, numpy as np
from config import DEFAULT_CONFIG
import _vis_mpi_square as V

for radial in (False,):
    print(f'\n===== mpi_radial_reject = {radial} =====')
    for i in range(1, 11):
        name = f'C{i}'
        img = cv2.imread(f'Testsett/{name}.jpg')
        if img is None:
            print(f'{name}: mangler'); continue
        cfg = DEFAULT_CONFIG.copy()
        cfg['analyze_screen_fallback'] = False
        cfg['mpi_radial_reject'] = radial
        try:
            gray, calib, hits = V.analysis_frame(img, cfg)
        except Exception as e:
            print(f'{name}: FEIL {e}'); continue
        if len(hits) < 2:
            print(f'{name}: <2 treff'); continue
        delta = calib['delta_px']
        pts = np.array([[h['x'], h['y']] for h in hits], float)
        seed = pts.mean(0)
        Rv, Rh = V.oriented_maps(gray, calib, cfg)
        det = V.detect_mpi_square(Rv, Rh, calib, seed, cfg)
        if not det:
            print(f'{name}: ingen deteksjon'); continue
        locs = []
        for k in range(len(pts)):
            d = V.detect_mpi_square(Rv, Rh, calib, np.delete(pts, k, 0).mean(0), cfg)
            if d:
                locs.append((d['X'], d['Y']))
        locs = np.array(locs)
        spread = np.hypot(*(locs - [det['X'], det['Y']]).T).max() if len(locs) else 0.0
        d_seed = np.hypot(det['X'] - seed[0], det['Y'] - seed[1])
        print(f"{name}: comb={det['comb']:.3f} struct@loc={det['struct']:.3f}  "
              f"d_seed={d_seed:5.0f}px  LOO={spread:4.0f}px  n={len(hits)}")
