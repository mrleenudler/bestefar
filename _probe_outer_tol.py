import cv2, sys
sys.path.insert(0, '.')
from config import DEFAULT_CONFIG
import screen as sc
from Bestefar import detect_outer_circle

for tol in [0.0]:
    cfg = DEFAULT_CONFIG.copy()
    cfg['screen_side_outer_tol_frac'] = tol
    print(f'--- tol={tol} ---')
    for i in range(1, 11):
        img = cv2.imread(f'Testsett/C{i}.jpg')
        res = sc.rectify_to_screen(img, cfg, [])
        if res is None:
            print(f'  C{i}: CROP FEIL')
            continue
        warped = res[0]
        try:
            cx, cy, r, _ = detect_outer_circle(warped, cfg, debug=False, filename=f'C{i}')
            print(f'  C{i}: {warped.shape[1]}x{warped.shape[0]}  c=({cx:.0f},{cy:.0f}) r={r:.0f}')
        except Exception as e:
            print(f'  C{i}: {e}')
