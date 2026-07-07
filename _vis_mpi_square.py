"""
MPI-firkant (midtpunkt-markor): deteksjon + overlegg paa C-settet. STEG 2.

Firkanten er displayets tyngdepunkt-markor for treffgruppa (invertert overlegg).
Matches POLARITETS-INVARIANT i gradient-domenet, med ORIENTERT dekomponering:
  - Rv = |Gx| (svarer paa VERTIKALE linjer: venstre/midt/hoyre bar)
  - Rh = |Gy| (svarer paa HORISONTALE linjer: topp/midt/bunn bar)
  - Tv = 3 vertikale barer, Th = 3 horisontale barer (side = mpi_square_side_frac*delta)
  - score = min(NCC(Rv,Tv), NCC(Rh,Th))  -> krever BEGGE orienteringer (kan ikke
    vinne paa vertikaler alene; avviser ring+tall-falske topper)
  - valgfri radiell gradient-avstoying om ringsenteret (som circles.py)
Presis senter: raffiner mot linje-ryggene (kolonne/rad-profil av Rv/Rh).

Robusthet (manglende treff): seed = tyngdepunkt av DETEKTERTE treff; leave-one-out
+ jitter viser at deteksjonen gjenfinnes selv med bias i seed.

Kjoer:  .venv\\Scripts\\python.exe _vis_mpi_square.py
Output: Visualiseringer/outputs/Cset_mpi_square.png
        Visualiseringer/outputs/Cset_mpi_maps.png   (mellomsteg for eyeballing)
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
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _odd(k):
    k = int(round(k))
    return max(3, k + 1 if k % 2 == 0 else k)


def oriented_templates(delta, cfg):
    """Tv (3 vertikale barer), Th (3 horisontale barer), samme boks-stoerrelse."""
    side = cfg.get('mpi_square_side_frac', 0.58) * delta
    stroke = max(1, int(round(cfg.get('mpi_square_stroke_frac', 0.077) * delta)))
    S = max(6, int(round(side)))
    pad = stroke + 2
    size = S + 2 * pad + 1
    Tv = np.zeros((size, size), np.float32)
    Th = np.zeros((size, size), np.float32)
    x0, x1, xc = pad, pad + S, pad + S // 2
    y0, y1, yc = pad, pad + S, pad + S // 2
    hw = stroke // 2 + 1
    for xx in (x0, xc, x1):
        Tv[y0:y1 + 1, xx - hw:xx + hw + 1] = 1.0
    for yy in (y0, yc, y1):
        Th[yy - hw:yy + hw + 1, x0:x1 + 1] = 1.0
    return Tv, Th, S, stroke, pad


def oriented_maps(gray, calib, cfg):
    """Rv=|Gx|, Rh=|Gy| paa median-renset bilde, valgfri radiell avstoying."""
    delta = calib['delta_px']
    cx, cy = calib['center']
    kmed = _odd(cfg.get('mpi_median_frac', 0.06) * delta)
    med = cv2.medianBlur(gray, kmed)
    gx = cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)
    Rv = np.abs(gx).astype(np.float32)
    Rh = np.abs(gy).astype(np.float32)
    if cfg.get('mpi_radial_reject', False):
        H, W = gray.shape
        yy, xx = np.mgrid[0:H, 0:W]
        dcx = cx - xx; dcy = cy - yy
        dn = np.hypot(dcx, dcy) + 1e-6
        mag = np.hypot(gx, gy) + 1e-6
        align = np.abs((gx * dcx + gy * dcy) / (mag * dn))   # |cos| mot radiell
        reject = cfg.get('mpi_radial_reject_cos', 0.8)
        exempt = dn < cfg.get('mpi_center_exempt_frac', 0.0) * delta
        w = np.where((align >= reject) & (~exempt), 0.2, 1.0).astype(np.float32)
        Rv = Rv * w; Rh = Rh * w
    return Rv, Rh


def _refine_center(Rv, Rh, X, Y, S, stroke):
    """Presis senter fra linje-ryggene: tyngdepunkt av kolonne/rad-profil i boksen."""
    half = S // 2 + 2 * stroke
    H, W = Rv.shape
    x0 = max(0, int(round(X - half))); x1 = min(W, int(round(X + half)) + 1)
    y0 = max(0, int(round(Y - half))); y1 = min(H, int(round(Y + half)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return X, Y
    colp = Rv[y0:y1, x0:x1].sum(axis=0)
    rowp = Rh[y0:y1, x0:x1].sum(axis=1)
    rx = x0 + (np.arange(len(colp)) * colp).sum() / (colp.sum() + 1e-9)
    ry = y0 + (np.arange(len(rowp)) * rowp).sum() / (rowp.sum() + 1e-9)
    return float(rx), float(ry)


def _ridge_template_1d(S, stroke):
    """1D-template: tre barer (venstre/midt/hoyre) med avstand S/2, bredde stroke."""
    Lt = S + 2 * stroke + 3
    t = np.zeros(Lt, np.float32)
    c = Lt // 2
    hw = stroke // 2 + 1
    for off in (-S // 2, 0, S // 2):
        a = max(0, c + off - hw); b = min(Lt, c + off + hw + 1)
        t[a:b] = 1.0
    return t, c


def _ridge_score(profile, S, stroke):
    """Skyv 3-bar-template over 1D-profil. Returnerer (ncc, senter-indeks i profil)."""
    t, tc = _ridge_template_1d(S, stroke)
    if len(profile) < len(t) + 1:
        return -1.0, len(profile) // 2
    p2 = profile.reshape(1, -1).astype(np.float32)
    t2 = t.reshape(1, -1).astype(np.float32)
    r = cv2.matchTemplate(p2, t2, cv2.TM_CCOEFF_NORMED)[0]
    j = int(np.argmax(r))
    return float(r[j]), j + tc


def _structural_validate(Rv, Rh, X, Y, S, stroke):
    """Verifiser 3 jevnt spredte rygger i BEGGE orienteringer rundt (X,Y).
    Returnerer (struct_score, X_ref, Y_ref) med presis senter fra ryggene."""
    half = S  # gir slingringsrom til 1D-templaten (~2S bred profil)
    Hh, Ww = Rv.shape
    x0 = max(0, int(round(X - half))); x1 = min(Ww, int(round(X + half)) + 1)
    y0 = max(0, int(round(Y - half))); y1 = min(Hh, int(round(Y + half)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return -1.0, X, Y
    colp = Rv[y0:y1, x0:x1].sum(axis=0)   # vertikale barer -> topper i |Gx|-kolonneprofil
    rowp = Rh[y0:y1, x0:x1].sum(axis=1)    # horisontale barer -> topper i |Gy|-radprofil
    nv, cx_idx = _ridge_score(colp, S, stroke)
    nh, cy_idx = _ridge_score(rowp, S, stroke)
    return min(nv, nh), float(x0 + cx_idx), float(y0 + cy_idx)


def detect_mpi_square(Rv, Rh, calib, seed, cfg):
    """Vid ROI -> comb=min(V,H)-argmax lokaliserer firkanten (paalitelig <=17px).
    Struktur-score (3 rygger begge orienteringer) beregnes ved lokasjonen som en
    RAPPORTERT diagnostikk - IKKE som re-rangering (rings/tall scorer ogsaa hoyt).
    Returnerer dict(X,Y,comb,struct,nccV,nccH,S) eller None."""
    delta = calib['delta_px']
    Tv, Th, S, stroke, pad = oriented_templates(delta, cfg)
    tsz = Tv.shape[0]
    H, W = Rv.shape
    search_r = cfg.get('mpi_search_r_frac', 2.0) * delta
    ext = int(search_r) + tsz
    sx, sy = int(round(seed[0])), int(round(seed[1]))
    x0 = max(0, sx - ext); x1 = min(W, sx + ext + 1)
    y0 = max(0, sy - ext); y1 = min(H, sy + ext + 1)
    if x1 - x0 < tsz or y1 - y0 < tsz:
        return None
    resV = cv2.matchTemplate(Rv[y0:y1, x0:x1], Tv, cv2.TM_CCOEFF_NORMED)
    resH = cv2.matchTemplate(Rh[y0:y1, x0:x1], Th, cv2.TM_CCOEFF_NORMED)
    comb = np.minimum(resV, resH)

    # MAP: likelihood (comb) x lokasjons-prior (Gauss om seed = forventet MPI).
    # Demper fjerne tall/ring-topper uten aa straffe en skjult-treff-forskjovet
    # firkant (~0.3 delta), som ligger godt innenfor prior-sigma.
    ch, cw = comb.shape
    gy_i, gx_i = np.mgrid[0:ch, 0:cw]
    cX = x0 + gx_i + tsz // 2
    cY = y0 + gy_i + tsz // 2
    sigma = cfg.get('mpi_prior_sigma_frac', 0.6) * delta
    prior = np.exp(-0.5 * (((cX - seed[0]) ** 2 + (cY - seed[1]) ** 2) / (sigma ** 2)))
    scored = comb * prior.astype(np.float32)
    _, _, _, loc = cv2.minMaxLoc(scored)
    mx = float(comb[loc[1], loc[0]])
    Xc = x0 + loc[0] + tsz // 2
    Yc = y0 + loc[1] + tsz // 2
    X, Y = _refine_center(Rv, Rh, Xc, Yc, S, stroke)
    struct, _, _ = _structural_validate(Rv, Rh, X, Y, S, stroke)
    return dict(X=float(X), Y=float(Y), comb=float(mx), score=float(mx),
                struct=float(struct), nccV=float(resV[loc[1], loc[0]]),
                nccH=float(resH[loc[1], loc[0]]), S=S)


def analysis_frame(img, cfg):
    a = analyze_target(img, cfg, filename='mpi')
    screen_res = sc.rectify_to_screen(img, cfg, [])
    gray = preprocess.to_gray(img if screen_res is None else screen_res[0])
    if a.get('H') is not None:
        gray = perspektiv.warp_image(gray, a['H'])
    return gray, a['calib'], a['hits']


def draw_square(vis, X, Y, S, col, thick=2):
    h = S / 2.0
    cv2.rectangle(vis, (int(round(X - h)), int(round(Y - h))),
                  (int(round(X + h)), int(round(Y + h))), col, thick)
    cv2.line(vis, (int(X - h), int(Y)), (int(X + h), int(Y)), col, thick)
    cv2.line(vis, (int(X), int(Y - h)), (int(X), int(Y + h)), col, thick)


def make_panel(name, img, cfg):
    gray, calib, hits = analysis_frame(img, cfg)
    if len(hits) < 2:
        panel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel, f'{name}: <2 treff', (10, 30), FONT, 0.8, (0, 60, 220), 2)
        return panel

    delta = calib['delta_px']
    pts = np.array([[h['x'], h['y']] for h in hits], float)
    seed = pts.mean(axis=0)
    Rv, Rh = oriented_maps(gray, calib, cfg)
    det = detect_mpi_square(Rv, Rh, calib, seed, cfg)

    robo = []
    for i in range(len(pts)):
        sub = np.delete(pts, i, axis=0).mean(axis=0)
        d = detect_mpi_square(Rv, Rh, calib, sub, cfg)
        if d:
            robo.append((d['X'], d['Y'], 'loo'))
    rng = np.random.default_rng(0)
    for _ in range(6):
        d = detect_mpi_square(Rv, Rh, calib, seed + rng.uniform(-1.5, 1.5, 2) * delta, cfg)
        if d:
            robo.append((d['X'], d['Y'], 'jit'))

    cx, cy = seed
    win = int(4.0 * delta)
    H, W = gray.shape
    x0 = max(0, int(cx - win)); x1 = min(W, int(cx + win))
    y0 = max(0, int(cy - win)); y1 = min(H, int(cy + win))
    up = 3
    crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    crop = cv2.resize(crop, (crop.shape[1] * up, crop.shape[0] * up),
                      interpolation=cv2.INTER_NEAREST)

    def to_c(px, py):
        return int((px - x0) * up), int((py - y0) * up)

    for p in pts:
        cv2.circle(crop, to_c(*p), int(0.41 * delta * up), (255, 160, 0), 1)
    cv2.drawMarker(crop, to_c(*seed), (0, 220, 220), cv2.MARKER_CROSS, 18, 2)
    for (rx, ry, kind) in robo:
        col = (220, 220, 0) if kind == 'loo' else (220, 0, 220)
        cv2.circle(crop, to_c(rx, ry), 3, col, -1)

    if det:
        ok = det['score'] >= cfg.get('mpi_struct_thresh', 0.30)
        col = (80, 220, 80) if ok else (60, 60, 220)
        Xc, Yc = to_c(det['X'], det['Y'])
        draw_square(crop, Xc, Yc, det['S'] * up, col, 2)
        d_seed = np.hypot(det['X'] - seed[0], det['Y'] - seed[1])
        txt = (f"{name}  struct={det['score']:.2f} comb={det['comb']:.2f}"
               f"  d_seed={d_seed:.0f}px  n={len(hits)}")
    else:
        col = (60, 60, 220)
        txt = f'{name}  INGEN DETEKSJON'

    bar = np.full((26, crop.shape[1], 3), 22, np.uint8)
    cv2.putText(bar, txt, (6, 19), FONT, 0.5, col, 1, cv2.LINE_AA)
    return np.vstack([crop, bar])


def make_maps_panel(name, img, cfg):
    """Mellomsteg: Rv | Rh | comb-heatmap rundt gruppa (for eyeballing)."""
    gray, calib, hits = analysis_frame(img, cfg)
    if len(hits) < 2:
        return None
    delta = calib['delta_px']
    seed = np.array([[h['x'], h['y']] for h in hits], float).mean(0)
    Rv, Rh = oriented_maps(gray, calib, cfg)
    Tv, Th, S, stroke, pad = oriented_templates(delta, cfg)
    win = int(4.0 * delta)
    H, W = gray.shape
    x0 = max(0, int(seed[0] - win)); x1 = min(W, int(seed[0] + win))
    y0 = max(0, int(seed[1] - win)); y1 = min(H, int(seed[1] + win))
    rv = Rv[y0:y1, x0:x1]; rh = Rh[y0:y1, x0:x1]
    resV = cv2.matchTemplate(rv, Tv, cv2.TM_CCOEFF_NORMED)
    resH = cv2.matchTemplate(rh, Th, cv2.TM_CCOEFF_NORMED)
    comb = np.minimum(resV, resH)

    def norm(a):
        a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return cv2.applyColorMap(a, cv2.COLORMAP_JET)
    up = 2
    cells = [cv2.cvtColor(cv2.normalize(rv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR),
             cv2.cvtColor(cv2.normalize(rh, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR),
             norm(comb)]
    labs = [f'{name} Rv=|Gx|', 'Rh=|Gy|', 'min(V,H)']
    out = []
    Ht = max(c.shape[0] for c in cells)
    for c, l in zip(cells, labs):
        c = cv2.resize(c, (c.shape[1] * up, c.shape[0] * up), interpolation=cv2.INTER_NEAREST)
        bar = np.full((22, c.shape[1], 3), 22, np.uint8)
        cv2.putText(bar, l, (4, 16), FONT, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
        out.append(np.vstack([c, bar]))
    Hm = max(o.shape[0] for o in out)
    out = [np.vstack([o, np.full((Hm - o.shape[0], o.shape[1], 3), 15, np.uint8)]) if o.shape[0] < Hm else o for o in out]
    return np.hstack(out)


def main():
    for pat in ('Cset_mpi_square*.png', 'Cset_mpi_maps*.png', '_mpi_*.png'):
        for f in OUT.glob(pat):
            f.unlink()

    cfg = DEFAULT_CONFIG.copy()
    cfg['analyze_screen_fallback'] = False
    panels, maps = [], []
    for i in range(1, 11):
        name = f'C{i}'
        img = cv2.imread(f'Testsett/{name}.jpg')
        if img is None:
            print(f'{name}: mangler'); continue
        print(f'{name}...', end=' ', flush=True)
        try:
            panels.append(make_panel(name, img, cfg))
            m = make_maps_panel(name, img, cfg)
            if m is not None:
                maps.append(m)
            print('OK')
        except Exception as e:
            print(f'FEIL: {e}')

    def stack2(items, sep=6):
        Wt = max(p.shape[1] for p in items)
        items = [np.hstack([p, np.full((p.shape[0], Wt - p.shape[1], 3), 15, np.uint8)])
                 if p.shape[1] < Wt else p for p in items]
        rows = []
        for i in range(0, len(items), 2):
            pair = items[i:i + 2]
            h = max(p.shape[0] for p in pair)
            pair = [np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 15, np.uint8)])
                    if p.shape[0] < h else p for p in pair]
            if len(pair) == 1:
                pair.append(np.full((h, pair[0].shape[1], 3), 15, np.uint8))
            rows.append(np.hstack([pair[0], np.full((h, sep, 3), 40, np.uint8), pair[1]]))
        Wr = max(r.shape[1] for r in rows)
        rows = [np.hstack([r, np.full((r.shape[0], Wr - r.shape[1], 3), 15, np.uint8)])
                if r.shape[1] < Wr else r for r in rows]
        return np.vstack(rows)

    if panels:
        out = OUT / 'Cset_mpi_square.png'
        cv2.imwrite(str(out), stack2(panels))
        print(f'Lagret -> {out}')
    if maps:
        Wt = max(m.shape[1] for m in maps)
        maps = [np.hstack([m, np.full((m.shape[0], Wt - m.shape[1], 3), 15, np.uint8)])
                if m.shape[1] < Wt else m for m in maps]
        out = OUT / 'Cset_mpi_maps.png'
        cv2.imwrite(str(out), np.vstack(maps))
        print(f'Lagret -> {out}')


if __name__ == '__main__':
    main()
