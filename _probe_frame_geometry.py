"""
Maal skjerm-geometri for scan-rammen (musings 2026-07: ny capture-ramme).

For hvert C-bilde: rektifiser skjermen -> analysér -> rapporter som ANDELER av
skjermbredde/-hoeyde:
  - ringsenter (cx, cy)
  - ytre hvite skive-radius (radial luminansprofil fra senter)
  - sort bull-radius (radial luminansprofil)
  - kalibrert ytterste ring (max ring_radii_px)
Lagrer ogsaa selve croppen for C1/C5/C9 til eyeballing av scoreboard-omraadet.

Kjoer:  .venv\Scripts\python.exe _probe_frame_geometry.py
Output: Visualiseringer/outputs/_framegeo_C*.png + tabell paa stdout
"""
import cv2
import numpy as np
from pathlib import Path

from config import DEFAULT_CONFIG
import screen as sc
import preprocess
import perspektiv
from Bestefar import analyze_target

OUT = Path('Visualiseringer/outputs')
OUT.mkdir(parents=True, exist_ok=True)


def radial_profile(gray, cx, cy, r_max, n_r=400, n_th=180):
    """Middel-luminans per radius (sirkulaert gjennomsnitt)."""
    rs = np.linspace(1, r_max, n_r)
    th = np.linspace(0, 2 * np.pi, n_th, endpoint=False)
    ct, st = np.cos(th), np.sin(th)
    H, W = gray.shape
    prof = np.zeros(n_r)
    for i, r in enumerate(rs):
        xs = np.clip((cx + r * ct).astype(int), 0, W - 1)
        ys = np.clip((cy + r * st).astype(int), 0, H - 1)
        prof[i] = gray[ys, xs].mean()
    return rs, prof


def main():
    cfg = DEFAULT_CONFIG.copy()
    cfg['analyze_screen_fallback'] = False
    rows = []
    for i in range(1, 11):
        name = f'C{i}'
        img = cv2.imread(f'Testsett/{name}.jpg')
        if img is None:
            continue
        res = sc.rectify_to_screen(img, cfg, [])
        if res is None:
            print(f'{name}: ingen skjermcrop'); continue
        crop = res[0]
        a = analyze_target(img, cfg, filename=name)
        if a.get('calib') is None:
            print(f'{name}: ingen kalibrering'); continue
        gray = preprocess.to_gray(crop)
        if a.get('H') is not None:
            gray = perspektiv.warp_image(gray, a['H'])
        calib = a['calib']
        cx, cy = calib['center']
        delta = calib['delta_px']
        r_ring = max(calib['ring_radii_px'])
        Hh, Ww = gray.shape

        rs, prof = radial_profile(gray, cx, cy, min(Ww, Hh) * 0.6)
        # Bull-kant: foerste overgang moerk->lys (profil krysser midtverdien)
        mid = 0.5 * (prof.min() + prof.max())
        dark = prof < mid
        bull_r = rs[np.argmax(~dark)] if dark[0] else 0.0
        # Hvit skive-kant: siste radius der profilen fortsatt er "lys"
        bright_thresh = prof.max() - 0.35 * (prof.max() - prof.min())
        bright_idx = np.where(prof > bright_thresh)[0]
        white_r = rs[bright_idx[-1]] if len(bright_idx) else 0.0

        rows.append((name, Ww, Hh, cx / Ww, cy / Hh, bull_r / Ww, white_r / Ww,
                     r_ring / Ww, delta / Ww))
        if name in ('C1', 'C5', 'C9'):
            vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.circle(vis, (int(cx), int(cy)), int(bull_r), (0, 0, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), int(white_r), (0, 255, 0), 2)
            cv2.circle(vis, (int(cx), int(cy)), int(r_ring), (255, 0, 0), 1)
            p = OUT / f'_framegeo_{name}.png'
            cv2.imwrite(str(p), vis)
            print(f'Lagret -> {p}')

    print()
    print('navn    W     H     cx/W   cy/H   bull_r/W  white_r/W  ring_r/W  delta/W')
    for r in rows:
        print('%-6s %5d %5d  %.3f  %.3f   %.3f     %.3f      %.3f     %.4f' % r)
    if rows:
        arr = np.array([r[3:] for r in rows])
        med = np.median(arr, axis=0)
        print('MEDIAN            %.3f  %.3f   %.3f     %.3f      %.3f     %.4f' % tuple(med))
        ar = np.median([r[1] / r[2] for r in rows])
        print('median W/H (aspekt): %.3f' % ar)


if __name__ == '__main__':
    main()
