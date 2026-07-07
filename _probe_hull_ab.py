"""
A/B-eksperiment: konveks innhylling AV vs PAA for hele C-settet.
For hvert bilde: kjoer skjerm-crop, deretter detect_outer_circle paa cropen,
og rapporter cropstoerrelse + detektert ytre sirkel (senter, radius).
Dette viser om hull-en faktisk gir en 'alternate output' for C10, og hva den
koster paa C9/resten.

Kjoer med:  .venv\\Scripts\\python.exe _probe_hull_ab.py
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
    warped = res[0]
    dims = f'{warped.shape[1]}x{warped.shape[0]}'
    try:
        cx, cy, r, _ = detect_outer_circle(warped, cfg, debug=False, filename=name)
        return f'{name:>4}: crop {dims:>11}  senter=({cx:5.0f},{cy:5.0f})  r={r:5.0f}'
    except Exception as e:
        return f'{name:>4}: crop {dims:>11}  RING FEIL: {str(e)[:45]}'


for hull in (False, True):
    cfg = DEFAULT_CONFIG.copy()
    cfg['screen_blob_convex_hull'] = hull
    print(f'\n===== screen_blob_convex_hull = {hull} =====')
    for i in range(1, 11):
        print(run(cfg, f'C{i}'))
