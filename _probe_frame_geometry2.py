"""
Steg 2: bull-radius + scoreboard-tabellens grenser, maalt paa rektifisert crop.

  - bull: stoerste moerke komponent naer ringsenteret (bbox-halvbredde)
  - skillelinje (vertikal) mellom skive og tabell: |Gx|-kolonneprofil
  - horisontale tabellinjer: |Gy|-radprofil i tabellomraadet
  - graa "snitt"-rad: dip i rad-middelintensitet (skiller liste fra oppsummering)

Kjoer:  .venv\Scripts\python.exe _probe_frame_geometry2.py
"""
import cv2
import numpy as np

from config import DEFAULT_CONFIG
import screen as sc
import preprocess
import perspektiv
from Bestefar import analyze_target


def measure(name):
    cfg = DEFAULT_CONFIG.copy()
    cfg['analyze_screen_fallback'] = False
    img = cv2.imread(f'Testsett/{name}.jpg')
    res = sc.rectify_to_screen(img, cfg, [])
    a = analyze_target(img, cfg, filename=name)
    gray = preprocess.to_gray(res[0])
    if a.get('H') is not None:
        gray = perspektiv.warp_image(gray, a['H'])
    calib = a['calib']
    cx, cy = calib['center']
    delta = calib['delta_px']
    Hh, Ww = gray.shape

    # ---- bull-radius: stoerste moerke komponent rundt senteret ----
    r_lim = int(6 * delta)
    x0 = max(0, int(cx - r_lim)); x1 = min(Ww, int(cx + r_lim))
    y0 = max(0, int(cy - r_lim)); y1 = min(Hh, int(cy + r_lim))
    reg = gray[y0:y1, x0:x1]
    _, dark = cv2.threshold(reg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(dark, 8)
    best, bw = 0, 0
    for l in range(1, n):
        if stats[l, cv2.CC_STAT_AREA] > best:
            best = stats[l, cv2.CC_STAT_AREA]
            bw = max(stats[l, cv2.CC_STAT_WIDTH], stats[l, cv2.CC_STAT_HEIGHT])
    bull_r = bw / 2.0

    # ---- vertikal skillelinje: |Gx|-kolonneprofil i x=[0.6,0.9]W ----
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    band = gx[int(0.05 * Hh):int(0.85 * Hh), :]
    colp = band.sum(axis=0)
    xa, xb = int(0.60 * Ww), int(0.90 * Ww)
    div_x = xa + int(np.argmax(colp[xa:xb]))

    # ---- horisontale linjer i tabellomraadet ----
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
    tab = gy[:, div_x + 5:int(0.98 * Ww)]
    rowp = tab.sum(axis=1)
    rowp_s = cv2.GaussianBlur(rowp.reshape(-1, 1), (1, 9), 0).ravel()
    thr = 0.35 * rowp_s.max()
    peaks = []
    for y in range(1, Hh - 1):
        if rowp_s[y] > thr and rowp_s[y] >= rowp_s[y - 1] and rowp_s[y] >= rowp_s[y + 1]:
            if not peaks or y - peaks[-1] > 0.01 * Hh:
                peaks.append(y)

    # ---- graa snitt-rad: dip i rad-middel i tabellomraadet ----
    inten = gray[:, div_x + 8:int(0.97 * Ww)].mean(axis=1)
    lo = int(0.30 * Hh); hi = int(0.75 * Hh)
    seg = inten[lo:hi]
    gray_row = lo + int(np.argmin(seg))

    print(f'{name}: W={Ww} H={Hh} delta/W={delta/Ww:.4f}')
    print(f'  bull_r/W = {bull_r/Ww:.4f}  (bull_r/delta = {bull_r/delta:.2f})')
    print(f'  divider_x/W = {div_x/Ww:.4f}')
    print(f'  graa snitt-rad y/H = {gray_row/Hh:.4f}')
    print('  H-linjer y/H:', ' '.join(f'{p/Hh:.3f}' for p in peaks))
    print()


for nm in ('C1', 'C5', 'C9'):
    measure(nm)
