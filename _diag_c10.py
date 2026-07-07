"""Diagnose where C10 fails in analyze_target."""
import cv2
from config import DEFAULT_CONFIG
import screen as sc, rings, preprocess, perspektiv
from Bestefar import detect_outer_circle

cfg = DEFAULT_CONFIG.copy()
img = cv2.imread('Testsett/C10.jpg')

print('=== RUTE 1: screen-rektifisert bilde ===')
screen_res = sc.rectify_to_screen(img, cfg, dbg := [])
if screen_res is None:
    print(f'  Screen-deteksjon FEIL: {dbg}')
else:
    warped = screen_res[0]
    print(f'  Screen OK: {warped.shape[1]}x{warped.shape[0]}')
    try:
        cx, cy, r0, _ = detect_outer_circle(warped, cfg, debug=False, filename='C10')
        print(f'  Ytre sirkel: ({cx:.0f},{cy:.0f}) r={r0:.0f}')
        gray = preprocess.to_gray(warped)
        dbg2 = []
        calib = rings.calibrate_and_refine(gray, (cx, cy), r0, cfg, debug_lines=dbg2)
        if calib is None:
            print(f'  Ring-kalib: FEIL (returnerte None)')
        else:
            ok, reason = rings.validate_calibration(calib, cfg)
            print(f'  Ring-kalib: {"OK" if ok else "FEIL"} — {reason if not ok else "gyldig"}')
            print(f'    delta={calib["delta_px"]:.1f}  rings={len(calib.get("ring_radii_px",[]))}'
                  f'  resid={calib.get("fit_resid_frac",float("nan")):.3f}')
        for ln in dbg2[-8:]:
            print(f'    {ln}')
    except Exception as e:
        print(f'  UNNTAK: {e}')

print()
print('=== RUTE 2: fullt originalbilde ===')
try:
    cx, cy, r0, _ = detect_outer_circle(img, cfg, debug=False, filename='C10')
    print(f'  Ytre sirkel: ({cx:.0f},{cy:.0f}) r={r0:.0f}')
    gray = preprocess.to_gray(img)
    dbg3 = []
    calib = rings.calibrate_and_refine(gray, (cx, cy), r0, cfg, debug_lines=dbg3)
    if calib is None:
        print('  Ring-kalib: FEIL (returnerte None)')
    else:
        ok, reason = rings.validate_calibration(calib, cfg)
        print(f'  Ring-kalib: {"OK" if ok else "FEIL"} — {reason if not ok else "gyldig"}')
        print(f'    delta={calib["delta_px"]:.1f}  rings={len(calib.get("ring_radii_px",[]))}'
              f'  resid={calib.get("fit_resid_frac",float("nan")):.3f}')
    for ln in dbg3[-8:]:
        print(f'    {ln}')
except Exception as e:
    print(f'  UNNTAK: {e}')
