"""
A/B: kant-snapping AV vs PAA. For hvert C-bilde: crop-dims + ytre sirkel.
Kjoer:  .venv\\Scripts\\python.exe _probe_snap_ab.py
"""
import cv2
from config import DEFAULT_CONFIG
import screen as sc
from Bestefar import detect_outer_circle


def run(cfg, name):
    img = cv2.imread(f'Testsett/{name}.jpg')
    if img is None:
        return f'{name:>4}: bilde ikke funnet'
    res = sc.rectify_to_screen(img, cfg, [])
    if res is None:
        return f'{name:>4}: CROP FEIL'
    w = res[0]
    dims = f'{w.shape[1]}x{w.shape[0]}'
    try:
        cx, cy, r, _ = detect_outer_circle(w, cfg, debug=False, filename=name)
        return (f'{name:>4}: crop {dims:>11}  senter=({cx:5.0f},{cy:5.0f})  '
                f'r={r:5.0f}  cx/W={cx/w.shape[1]:.2f} cy/H={cy/w.shape[0]:.2f}')
    except Exception as e:
        return f'{name:>4}: crop {dims:>11}  RING FEIL: {str(e)[:40]}'


for snap in (False, True):
    cfg = DEFAULT_CONFIG.copy()
    cfg['screen_refine_gradient_lines'] = snap
    print(f'\n===== screen_refine_gradient_lines = {snap} =====')
    for i in range(1, 11):
        print(run(cfg, f'C{i}'))
