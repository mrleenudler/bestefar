"""
Full-pipeline A/B: kant-snapping AV vs PAA. Rapporter treff-antall / forkastet.
Kjoer:  .venv\\Scripts\\python.exe _probe_snap_full.py
"""
import cv2
from config import DEFAULT_CONFIG
from Bestefar import analyze_target


def run(cfg, name):
    img = cv2.imread(f'Testsett/{name}.jpg')
    if img is None:
        return f'{name:>4}: mangler'
    try:
        a = analyze_target(img, cfg, filename=name)
        return f'{name:>4}: OK   treff={len(a["results"])}  sum={a["sum_integer"]}'
    except ValueError as e:
        return f'{name:>4}: FORKASTET  {str(e)[:45]}'
    except Exception as e:
        return f'{name:>4}: FEIL {type(e).__name__}: {str(e)[:40]}'


for snap in (False, True):
    cfg = DEFAULT_CONFIG.copy()
    cfg['analyze_screen_fallback'] = False
    cfg['screen_refine_gradient_lines'] = snap
    print(f'\n===== screen_refine_gradient_lines = {snap} =====')
    for i in range(1, 11):
        print(run(cfg, f'C{i}'))
