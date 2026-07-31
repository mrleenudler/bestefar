"""
Mock av NY scan-ramme (musings 2026-07) tegnet over rektifiserte C-bilder.
Verifiserer at ramme-geometrien (sirkler + poeng/oppsummering-rektangler)
matcher en virkelig skjerm. Samme fraksjoner som ScanFrameView.kt bruker.

Kjoer:  .venv\Scripts\python.exe _vis_scan_frame_mock.py
Output: Visualiseringer/outputs/scan_frame_mock.png
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

# Geometri som ANDELER av rammebredde/-hoeyde (maalt paa C-settet,
# _probe_frame_geometry*.py). Holdes i synk med ScanFrameView.kt.
CX, CY = 0.415, 0.420        # ringsenter
R_WHITE = 0.304              # hvit skive (av W)
R_BULL = 0.121               # sort bull (av W)
TAB_X0, TAB_X1 = 0.752, 0.990
LIST_Y0, LIST_Y1 = 0.016, 0.516
SUM_Y1 = 0.824


def draw_frame(img):
    # NB: i appen er rammen MATT HVIT (0x66FFFFFF); her tegnes den ORANSJE
    # og kraftigere slik at geometrien er synlig mot den hvite skiva ved
    # verifisering. Posisjoner/stoerrelser er identiske med appen.
    Hh, Ww = img.shape[:2]
    ov = img.copy()
    col = (0, 140, 255)
    thin = max(2, Ww // 400)          # indre elementer
    thick = 3 * thin                  # ytre ramme: 3x
    # ytre 4:3-ramme
    cv2.rectangle(ov, (thick // 2, thick // 2),
                  (Ww - thick // 2, Hh - thick // 2), col, thick)
    # sirkler om hvitt og sort omraade
    cv2.circle(ov, (int(CX * Ww), int(CY * Hh)), int(R_WHITE * Ww), col, thin)
    cv2.circle(ov, (int(CX * Ww), int(CY * Hh)), int(R_BULL * Ww), col, thin)
    # poengliste + oppsummering
    cv2.rectangle(ov, (int(TAB_X0 * Ww), int(LIST_Y0 * Hh)),
                  (int(TAB_X1 * Ww), int(LIST_Y1 * Hh)), col, thin)
    cv2.rectangle(ov, (int(TAB_X0 * Ww), int(LIST_Y1 * Hh)),
                  (int(TAB_X1 * Ww), int(SUM_Y1 * Hh)), col, thin)
    # halvtransparent, men godt synlig for verifisering
    return cv2.addWeighted(ov, 0.8, img, 0.2, 0)


def main():
    cfg = DEFAULT_CONFIG.copy()
    cfg['analyze_screen_fallback'] = False
    panels = []
    for name in ('C1', 'C5', 'C9'):
        img = cv2.imread(f'Testsett/{name}.jpg')
        res = sc.rectify_to_screen(img, cfg, [])
        a = analyze_target(img, cfg, filename=name)
        crop = res[0]
        if a.get('H') is not None:
            crop = perspektiv.warp_image(crop, a['H'])
        vis = draw_frame(crop)
        vis = cv2.resize(vis, (1200, int(1200 * vis.shape[0] / vis.shape[1])))
        cv2.putText(vis, name, (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 220, 255), 2, cv2.LINE_AA)
        panels.append(vis)
        print(f'{name} OK')
    Hm = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.full((Hm - p.shape[0], p.shape[1], 3), 15, np.uint8)])
              if p.shape[0] < Hm else p for p in panels]
    out = OUT / 'scan_frame_mock.png'
    cv2.imwrite(str(out), np.hstack(panels))
    print(f'Lagret -> {out}')


if __name__ == '__main__':
    main()
