"""
Maalrettet sirkeldetektor for treffmarkorer - erstatter cv2.HoughCircles sin
kandidatgenerering.

Ideen (fra musings): markorens kant er bare litt svakere enn ringlinjene, men
markoren er en KOMPLETT sirkel mens ringlinjer/sifre bare gir korte buer.
  1) median-filter fjerner LED-rutenettets salt-og-pepper-stoy (kantbevarende)
  2) CLAHE lokal kontrast loefter de svake markorkantene
  3) gradient-stemmer: hver kantpiksel stemmer paa et senter i +-gradientretning
     i avstand r (begge polariteter -> uavhengig av lys/moerk markor). En hel
     sirkel faar alle stemmene til Aa konvergere i ett punkt (hoey topp); en rett
     ringlinje sprer stemmene langs to parallelle linjer (lav, diffus topp).
     => topphoyden maaler sirkel-KOMPLETTHET, ikke gradientstyrke.
"""
import cv2
import numpy as np


def _odd(k):
    k = int(round(k))
    return k + 1 if k % 2 == 0 else max(1, k)


def circle_vote_map(gray, center, search_r, marker_r, dot_r, cfg, inner_r=0.0):
    """Returner (acc, dbg) der acc er stemmekartet i ROI-koordinater.
    Stemmer paa BAaDE markor-skiven (radius marker_r) og den indre prikken
    (radius dot_r) - de deler senter, saa begge forsterker samme topp.
    inner_r>0 utelukker kanter innenfor denne radien (bull-sonen)."""
    cx, cy = center
    h, w = gray.shape
    x0 = max(0, int(cx - search_r)); x1 = min(w, int(cx + search_r) + 1)
    y0 = max(0, int(cy - search_r)); y1 = min(h, int(cy + search_r) + 1)
    roi = gray[y0:y1, x0:x1]

    # 1) avstoying mot LED-rutenettet. Valgfri Gauss-forblur foerst, deretter
    #    median (kjoeres N ganger). CLAHE droppet (gjeninnforte rutenett-stoyen).
    kmed = _odd(cfg.get('circ_median_frac', 0.18) * marker_r)
    pg = cfg.get('circ_pre_gauss_frac', 0.0) * marker_r
    med = cv2.GaussianBlur(roi, (0, 0), pg) if pg > 0 else roi
    for _ in range(cfg.get('circ_median_passes', 2)):
        med = cv2.medianBlur(med, max(3, kmed))

    # 2) gradient (ingen kontrastforsterkning)
    gx = cv2.Sobel(med, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(med, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    enh = med   # for dbg-visning

    # lav terskel: behold relativt svake kanter, la kompletthet sile
    thr = np.percentile(mag, cfg.get('circ_edge_pct', 70.0))
    ys, xs = np.where(mag > thr)
    if len(xs) == 0:
        return np.zeros(roi.shape, np.float32), dict(roi=roi, med=med, enh=enh, mag=mag, thr=thr)
    ux = gx[ys, xs] / mag[ys, xs]
    uy = gy[ys, xs] / mag[ys, xs]

    # Avvis ringlinje-kanter: de printede poengringene er konsentriske med MAaL-
    # senteret, saa kanten deres peker radielt mot maalsenteret. Markorkanter peker
    # mot sin LOKALE senter. Slipp stemmen kun hvis gradienten IKKE er radiell mot
    # maalsenteret (unntatt naer senteret, der bull-markorer selv ligger).
    dcx = cx - (xs + x0); dcy = cy - (ys + y0)
    dn = np.hypot(dcx, dcy) + 1e-6
    # Retnings-gating er NOEDVENDIG (empirisk): stoerrelse-gating alene stopper ikke
    # de konsentriske poengringene, som avsetter en TETT offset-sirkel av stemmer ved
    # markor-radius. AaA forkaste radielle (mot maalsenter) kanter fjerner dem ved kilden.
    reject = cfg.get('circ_radial_reject', 0.80)
    keep = np.ones(len(xs), bool)
    if reject < 1.0:
        align = np.abs(ux * (dcx / dn) + uy * (dcy / dn))   # |cos| mot radiell retning
        near_c = dn < cfg.get('circ_center_exempt_frac', 1.5) * marker_r
        keep &= (near_c | (align < reject))
    if inner_r > 0:
        keep &= (dn >= inner_r)                              # utelukk bull-sonen
    xs, ys, ux, uy = xs[keep], ys[keep], ux[keep], uy[keep]

    acc = np.zeros(roi.shape, np.float32)
    Hh, Ww = roi.shape
    nr = cfg.get('circ_n_radii', 5)
    # (radius, vekt): skive-stemmer vekt 1, prikk-stemmer hoeyere. Prikk-kantene
    # konvergerer paa det ENKELTE skuddets senter; skive-kantene fra overlappende
    # markorer smelter til en blob hvis ytre kant trekker toppen mot blob-sentroiden.
    # AaA vekte prikken opp forankrer toppen paa det ekte senteret i tette klynger.
    dot_w = cfg.get('circ_dot_vote_weight', 1.0)
    rw = [(r, 1.0) for r in np.linspace(0.82, 1.18, nr) * marker_r]   # markor-skive
    if dot_r and dot_r > 2.0:                                         # indre prikk (samme senter)
        rw += [(r, dot_w) for r in np.linspace(0.80, 1.20, max(2, nr // 2)) * dot_r]
    for r, wgt in rw:
        for sgn in (+1.0, -1.0):
            vx = np.round(xs + sgn * r * ux).astype(np.int32)
            vy = np.round(ys + sgn * r * uy).astype(np.int32)
            ok = (vx >= 0) & (vx < Ww) & (vy >= 0) & (vy < Hh)
            np.add.at(acc, (vy[ok], vx[ok]), wgt)   # binaer stemme (kompletthet > styrke)

    # glatt stemmekartet til markorprikkens skala
    acc = cv2.GaussianBlur(acc, (0, 0), max(1.0, cfg.get('circ_acc_blur_frac', 0.12) * marker_r))
    return acc, dict(roi=roi, med=med, enh=enh, mag=mag, thr=thr, offset=(x0, y0))


def detect_circles(gray, center, search_r, marker_r, dot_r, cfg, inner_r=0.0, return_dbg=False):
    """Finn markorkandidater via stemmekartet + NMS. Returner liste (x,y,score)
    i fullbilde-koordinater (score = relativ topphoyde i [0,1])."""
    acc, dbg = circle_vote_map(gray, center, search_r, marker_r, dot_r, cfg, inner_r=inner_r)
    x0, y0 = dbg.get('offset', (0, 0))

    # NMS: lokale maks innen markor-avstand
    nms_r = _odd(cfg.get('circ_nms_frac', 0.7) * marker_r)
    dil = cv2.dilate(acc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_r, nms_r)))
    peakmask = (acc == dil) & (acc > 0)

    amax = float(acc.max()) if acc.size else 0.0
    # Absolutt gulv: en ekte sirkel maa samle et minimum konvergerende stemmer
    # (~ andel av omkretsen), uavhengig av hva annet finnes i bildet. Hindrer at
    # et svakt/diffust stemmekart (lav amax) slipper inn stoy-topper.
    floor = cfg.get('circ_min_votes_frac', 0.30) * marker_r
    keep_frac = cfg.get('circ_peak_min_frac', 0.45)
    thr_keep = max(keep_frac * amax, floor)
    cands = []
    if amax > 0:
        pys, pxs = np.where(peakmask & (acc >= thr_keep))
        cx0, cy0 = center
        for px, py in zip(pxs, pys):
            X, Y = px + x0, py + y0
            d = np.hypot(X - cx0, Y - cy0)
            if d > search_r or d < inner_r:
                continue
            cands.append((float(X), float(Y), float(acc[py, px] / amax)))
    cands.sort(key=lambda t: -t[2])
    if return_dbg:
        dbg['acc'] = acc; dbg['amax'] = amax; dbg['floor'] = floor; dbg['thr'] = thr_keep
        return cands, dbg
    return cands
