"""
Skjermdeteksjon: finn den lyse, rektangulære Kongsberg-skjermen i fotoet,
korriger perspektiv og beskjær til skjermen.

Strategi (robust mot bakgrunn og lys):
  1) Normaliser lys (CLAHE) på et kraftig nedskalert bilde.
  2) Contrast-ROI: skjermen har den skarpeste svart/hvitt-strukturen i bildet
     (ringer/tekst) – uansett om bakgrunnen er gress, betong e.l. Et lokalt
     kontrastkart isolerer apparatet/skjermen. Alt utenfor ROI maskeres bort,
     så segmenteringen ikke kan "lekke" ut i bakgrunnen (gjenskinn/lyse flekker).
  3) Hele den opplyste skjermen segmenteres med hysterese-terskling INNE i ROI
     (hvitt felt vokser ut til den mørke rammen, tar med grå topplinje/panel).
  4) De fire sidene tilpasses robuste, gradient-vektede linjer; hjørnene er
     skjæringspunktene. Lokale forstyrrelser blir uteliggere; mangler en side
     tydelig rett kant, forkastes deteksjonen.
"""

import cv2
import numpy as np


def _order_corners(pts):
    """Sorter til (TL, TR, BR, BL), robust for moderat skråstilling."""
    pts = np.asarray(pts, dtype=np.float32)
    idx = np.argsort(pts[:, 1])
    top = pts[idx[:2]]; bot = pts[idx[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bot[np.argsort(bot[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _normalize(gray, cfg):
    """
    Global kontraststrekk: klipp ytterste pct% i hver hale (robust mot gjenskinn
    og mørke flekker), skaler midtre 98% til 0-255. Bevarer skjerm/bakgrunn-
    kontrasten (i motsetning til CLAHE som flatet den ut).
    """
    pct = cfg.get('screen_stretch_pct', 1.0)
    lo, hi = np.percentile(gray, [pct, 100.0 - pct])
    if hi - lo < 1.0:
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def multiotsu3(gray):
    """
    Vektorisert 3-klasse multi-Otsu. Returnerer (t1, t2) som deler histogrammet
    i mørk/grå/lys ved å maksimere mellomklasse-variansen over alle kuttpar.
    t1 isolerer den mørkeste toppen (rammen/bezel). Sub-millisekund (256x256-rutenett).
    """
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    tot = hist.sum()
    if tot < 1:
        return 85, 170
    p = hist / tot
    P = np.cumsum(p)                      # kumulativ vekt
    S = np.cumsum(np.arange(256) * p)     # kumulativ vektet intensitet
    muT = S[-1]
    idx = np.arange(256)
    A, B = np.meshgrid(idx, idx, indexing='ij')   # A=t1, B=t2
    eps = 1e-12
    w0 = P[A]; s0 = S[A]
    w1 = P[B] - P[A]; s1 = S[B] - S[A]
    w2 = 1.0 - P[B]; s2 = muT - S[B]
    valid = (A < B) & (w0 > eps) & (w1 > eps) & (w2 > eps)
    sb = np.where(valid,
                  s0 * s0 / np.maximum(w0, eps)
                  + s1 * s1 / np.maximum(w1, eps)
                  + s2 * s2 / np.maximum(w2, eps),
                  -1.0)
    t1, t2 = np.unravel_index(int(np.argmax(sb)), sb.shape)
    return int(t1), int(t2)


def _apparatus_roi(gray, cfg, debug_lines):
    """
    Contrast-basert ROI rundt apparatet/skjermen. Returnerer binær maske
    (1 inne) eller None. Bruker lokal standardavvik som strukturmål.
    """
    H, W = gray.shape
    win = max(5, int(cfg.get('screen_contrast_win_frac', 0.04) * max(H, W)))
    win += (win % 2 == 0)
    g = gray.astype(np.float32)
    mean = cv2.boxFilter(g, -1, (win, win))
    msq = cv2.boxFilter(g * g, -1, (win, win))
    std = np.sqrt(np.maximum(msq - mean * mean, 0.0))
    stdn = cv2.normalize(std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, hi = cv2.threshold(stdn, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mx = max(H, W)
    kc = max(3, int(cfg.get('screen_contrast_win_frac', 0.04) * mx))
    hi = cv2.morphologyEx(hi, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    # Åpning: fjern små kontrast-flekker (bakgrunnstekst, løv) og kutt tynne
    # broer til rot, slik at største komponent blir ren skjerm.
    ko = max(3, int(cfg.get('screen_roi_open_frac', 0.025) * mx))
    hi = cv2.morphologyEx(hi, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(hi, connectivity=8)
    if num <= 1:
        debug_lines.append("skjerm: ingen kontrast-region")
        return None
    # Største strukturerte region som inneholder en lys (skjerm-) piksel
    order = np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1] + 1
    seed = None
    for lab in order:
        if gray[labels == lab].max() >= cfg.get('screen_roi_bright_min', 190):
            seed = lab; break
    if seed is None:
        debug_lines.append("skjerm: kontrast-region uten lys skjerm")
        return None
    mask = (labels == seed).astype(np.uint8)
    # Fyll regionen ved å tegne dens EGEN ytre kontur fylt (følger skjermens
    # form og fyller indre hull) - i motsetning til konveks innhylling som
    # blåser seg ut til fjerne flekker og "vandrer".
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi = np.zeros_like(mask)
    cv2.drawContours(roi, cnts, -1, 1, thickness=-1)
    d = max(3, int(cfg.get('screen_roi_dilate_frac', 0.06) * mx))
    roi = cv2.dilate(roi, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d)))
    return roi


def _screen_blob(gray_blur, cfg, debug_lines, roi=None):
    """
    Segmenter den opplyste skjermen med hysterese-terskling (innenfor ROI).
    Returnerer (kontur[N,2] float, area_frac) eller (None, frac).
    """
    if roi is not None:
        vals = gray_blur[roi > 0]
        if len(vals) < 50:
            debug_lines.append("skjerm: for få ROI-piksler")
            return None, 0.0
        otsu, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        otsu, _ = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    t_low = cfg.get('screen_low_frac', 0.55) * otsu
    ml = (gray_blur >= t_low).astype(np.uint8)
    mh = gray_blur >= otsu
    if roi is not None:
        ml[roi == 0] = 0
        mh = mh & (roi > 0)

    num, labels = cv2.connectedComponents(ml, connectivity=8)
    seed_labels = np.unique(labels[mh])
    seed_labels = seed_labels[seed_labels > 0]
    if len(seed_labels) == 0:
        debug_lines.append("skjerm: ingen lyse frø")
        return None, 0.0
    sizes = np.bincount(labels.ravel())
    best = max(seed_labels, key=lambda l: sizes[l])
    mask = (labels == best).astype(np.uint8) * 255

    k = max(3, int(0.04 * max(gray_blur.shape)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Morfologisk åpning: bryter smale forbindelser mellom skjermen og
    # apparatkassen/gulv uten å krympe hoveddelen av skjerm-blobben.
    open_frac = cfg.get('screen_blob_open_frac', 0.0)
    if open_frac > 0:
        k_o = max(3, int(open_frac * max(gray_blur.shape)))
        k_o += (k_o % 2 == 0)
        kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_o, k_o))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_o)
        # Åpningen kan splitte blobben – velg største gjenværende komponent
        n_lab, lab_img, lab_stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        if n_lab > 2:
            best_lab = 1 + int(np.argmax(lab_stats[1:, cv2.CC_STAT_AREA]))
            mask = ((lab_img == best_lab).astype(np.uint8) * 255)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        debug_lines.append("skjerm: ingen konturer")
        return None, 0.0
    cand = max(contours, key=cv2.contourArea)
    area_frac = cv2.contourArea(cand) / (gray_blur.shape[0] * gray_blur.shape[1])
    if area_frac < cfg.get('screen_min_area_frac', 0.10):
        debug_lines.append(f"skjerm: skjerm-blob for liten ({area_frac:.2f})")
        return None, area_frac
    return cand.reshape(-1, 2).astype(np.float32), area_frac


def _rough_quad(contour):
    peri = cv2.arcLength(contour.astype(np.float32), True)
    for eps_frac in np.linspace(0.01, 0.08, 8):
        approx = cv2.approxPolyDP(contour.astype(np.float32), eps_frac * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return _order_corners(approx.reshape(4, 2).astype(np.float32))
    s = contour.sum(axis=1); d = contour[:, 0] - contour[:, 1]
    quad = np.array([contour[np.argmin(s)], contour[np.argmax(d)],
                     contour[np.argmax(s)], contour[np.argmin(d)]], dtype=np.float32)
    return _order_corners(quad)


def _seg_dist(P, A, B):
    AB = B - A
    L2 = float(AB @ AB)
    if L2 < 1e-9:
        return np.linalg.norm(P - A, axis=1)
    t = np.clip(((P - A) @ AB) / L2, 0.0, 1.0)
    proj = A[None, :] + t[:, None] * AB[None, :]
    return np.linalg.norm(P - proj, axis=1)


def _fit_line_robust(P, w0, n_iter=6):
    """
    Vektet total-minste-kvadrat med IRLS. Linje {p : n·p = c}, |n|=1.
    Starter med Huber (konveks, stabil init), bytter så til Tukey biweight
    (redescending) som NULLER ut uteliggere helt – så linja låser seg på
    majoriteten av punktene og ignorerer lokale utstikk (gjenskinn o.l.).
    """
    P = np.asarray(P, float); w0 = np.asarray(w0, float)
    if len(P) < 8:
        return None
    w = w0.copy()
    n = c = resid = None
    for it in range(n_iter):
        wsum = w.sum()
        if wsum < 1e-9:
            return None
        mean = (w[:, None] * P).sum(0) / wsum
        Q = P - mean
        C = (Q.T * w) @ Q / wsum
        _, evecs = np.linalg.eigh(C)
        n = evecs[:, 0]
        c = float(n @ mean)
        resid = np.abs(P @ n - c)
        sigma = max(1.4826 * np.median(resid), 0.5)
        if it < 2:
            rw = np.minimum(1.0, (1.345 * sigma) / np.maximum(resid, 1e-6))  # Huber
        else:
            u = resid / (4.685 * sigma)                                       # Tukey
            rw = np.where(u < 1.0, (1.0 - u * u) ** 2, 0.0)
        w = w0 * rw
    return n, c, resid


def _intersect(l1, l2):
    (n1, c1), (n2, c2) = l1, l2
    Amat = np.array([n1, n2]); b = np.array([c1, c2])
    if abs(np.linalg.det(Amat)) < 1e-6:
        return None
    return np.linalg.solve(Amat, b)


def _snap_line_to_edge(line, P0, P1, center, gx, gy, gmag, roi, cfg, debug_lines):
    """
    La en side-linje 'vandre' mot den ekte skjermkanten. Skanner vinkelrett på
    linja innenfor et bånd, og plukker per skann-linje det YTTERSTE punktet (lengst
    fra skjermsenteret) som fortsatt er en STERK gradient (>= frac av skann-toppen)
    OG peker på tvers av kanten (retningsport). Re-tilpasser robust og re-sentreres
    hver iterasjon -> kan migrere selv om den grove sida startet et stykke unna.

    Hvorfor ytterste-sterke og ikke globalt sterkeste: bezelen er allerede utenfor
    ROI, så innenfor ROI er skjermkanten den ytterste sterke gradienten. Indre
    distraktorer (poengringer, UI-delelinje, MENY) er sterke MEN lenger inne, og
    lurte det tidligere argmax-valget til å skjevstille C2/C4. Glare ligger utenfor
    men er diffus/svak og faller under styrke-gulvet.
    Returnerer (n, c) eller None.
    """
    n, c = np.asarray(line[0], float), float(line[1])
    center = np.asarray(center, float)
    H, W = gmag.shape
    mx = max(H, W)
    band = max(4, int(cfg.get('screen_snap_band_frac', 0.045) * mx))
    n_iter = int(cfg.get('screen_snap_iters', 4))
    dir_cos = float(cfg.get('screen_snap_dir_cos', 0.5))
    skip = float(cfg.get('screen_snap_end_skip_frac', 0.08))
    strong_frac = float(cfg.get('screen_snap_strong_frac', 0.6))
    A = np.asarray(P0, float); B = np.asarray(P1, float)
    d = np.arange(-band, band + 1, 1.0)

    for _ in range(n_iter):
        # Projiser endepunktene ned på gjeldende linje, så skann-basen følger den
        A_p = A - (n @ A - c) * n
        B_p = B - (n @ B - c) * n
        seg = B_p - A_p
        Lp = float(np.hypot(*seg))
        if Lp < 8:
            return None
        t = seg / Lp
        nrm = np.array([-t[1], t[0]])
        m0, m1 = int(skip * Lp), int((1.0 - skip) * Lp)
        if m1 - m0 < 8:
            m0, m1 = 0, int(Lp)
        s = np.arange(m0, m1, 1.0)
        base = A_p[None, :] + s[:, None] * t[None, :]                 # (M,2)
        # "Utover" = retningen langs nrm som øker avstand fra skjermsenteret
        out_sign = 1.0 if float(nrm @ (base.mean(0) - center)) >= 0 else -1.0
        signed_d = d * out_sign                                        # (D,)
        Q = base[:, None, :] + d[None, :, None] * nrm[None, None, :]  # (M,D,2)
        xi = np.clip(np.round(Q[..., 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(Q[..., 1]).astype(int), 0, H - 1)
        g = gmag[yi, xi]
        gxv = gx[yi, xi]; gyv = gy[yi, xi]
        align = np.abs((gxv * nrm[0] + gyv * nrm[1]) / (np.hypot(gxv, gyv) + 1e-6))
        valid = align >= dir_cos
        if roi is not None:
            valid &= roi[yi, xi] > 0
        gmasked = np.where(valid, g, -1.0)                            # (M,D)
        row_max = gmasked.max(axis=1, keepdims=True)                  # (M,1)
        strong = (gmasked > 0) & (gmasked >= strong_frac * np.maximum(row_max, 1e-6))
        # Blant sterke kandidater: velg den ytterste (størst signed_d)
        score = np.where(strong, signed_d[None, :], -np.inf)
        best = np.argmax(score, axis=1)
        row_ok = strong.any(axis=1)
        idx = np.arange(len(s))
        gbest = gmasked[idx, best]
        keep = row_ok & (gbest > 0)
        if keep.sum() < 10:
            return None
        edge_pts = base[keep] + d[best[keep]][:, None] * nrm[None, :]
        fit = _fit_line_robust(edge_pts, gbest[keep])
        if fit is None:
            return None
        n, c = np.asarray(fit[0], float), float(fit[1])
    return (n, c)


def _refine_from_contour(contour, rough, gmag, cfg, debug_lines, gx=None, gy=None, roi=None):
    tl, tr, br, bl = rough
    seg = [(tl, tr, 'topp'), (tr, br, 'høyre'), (br, bl, 'bunn'), (bl, tl, 'venstre')]
    dists = np.stack([_seg_dist(contour, A, B) for (A, B, _) in seg], axis=0)
    assign = np.argmin(dists, axis=0)

    H, W = gmag.shape
    xi = np.clip(np.round(contour[:, 0]).astype(int), 0, W - 1)
    yi = np.clip(np.round(contour[:, 1]).astype(int), 0, H - 1)
    wts = gmag[yi, xi] + 1e-3

    # Grovt senterpunkt brukes til å bestemme "innover"-retning per side
    center_rough = rough.mean(axis=0)
    outer_tol_frac = cfg.get('screen_side_outer_tol_frac', 0.0)

    lines = []
    for i, (A, B, name) in enumerate(seg):
        sel = assign == i
        P = contour[sel]
        w = wts[sel].copy()
        L = float(np.hypot(*(B - A)))
        if len(P) < 10 or L < 8:
            debug_lines.append(f"skjerm: side {name} for få punkter")
            return None

        if outer_tol_frac > 0:
            # Klipp bort konturpunkter som stikker for langt UTENFOR den grove siden.
            # Signert avstand: positiv = mot skjermsenteret (innsiden), negativ = utside.
            t_dir = (B - A) / (L + 1e-9)
            n_dir = np.array([-t_dir[1], t_dir[0]], dtype=np.float64)
            if float((center_rough - A) @ n_dir) < 0:
                n_dir = -n_dir
            perp = (P - A[None, :]) @ n_dir          # positiv = innover
            keep = perp > -(outer_tol_frac * L)
            if keep.sum() >= 10:
                P = P[keep]
                w = w[keep]

        fit = _fit_line_robust(P, w)
        if fit is None:
            debug_lines.append(f"skjerm: side {name} linjetilpasning feilet")
            return None
        n, c, resid = fit
        tol = max(2.0, cfg.get('screen_side_inlier_tol_frac', 0.012) * L)
        inl = resid <= tol
        frac = inl.mean()
        t = (B - A) / L
        span = 0.0
        if inl.sum() >= 2:
            tp = P[inl] @ t
            span = (tp.max() - tp.min()) / L
        if frac < cfg.get('screen_side_min_inlier_frac', 0.35) or \
           span < cfg.get('screen_side_min_span_frac', 0.50):
            debug_lines.append(f"skjerm: side {name} svak kant "
                               f"(inliers {frac:.2f}, spenn {span:.2f})")
            return None
        lines.append((n, c))

    top, right, bottom, left = lines
    TL = _intersect(left, top); TR = _intersect(top, right)
    BR = _intersect(right, bottom); BL = _intersect(bottom, left)
    if any(p is None for p in (TL, TR, BR, BL)):
        debug_lines.append("skjerm: hjørne-skjæring feilet (parallelle sider)")
        return None
    orig_quad = _order_corners(np.array([TL, TR, BR, BL], dtype=np.float32))

    # Kant-snapping: la hver side vandre mot den sterkeste gradient-kammen.
    # Grov linje = kun startgjetning; endelig kant følger gradienten strengt.
    if cfg.get('screen_refine_gradient_lines', False) and gx is not None:
        sides = [(top, TL, TR), (right, TR, BR), (bottom, BR, BL), (left, BL, TL)]
        q_center = orig_quad.mean(axis=0)
        snapped = []
        for ln, Pa, Pb in sides:
            sl = _snap_line_to_edge(ln, Pa, Pb, q_center, gx, gy, gmag, roi, cfg, debug_lines)
            snapped.append(sl if sl is not None else ln)
        s_top, s_right, s_bottom, s_left = snapped
        sTL = _intersect(s_left, s_top); sTR = _intersect(s_top, s_right)
        sBR = _intersect(s_right, s_bottom); sBL = _intersect(s_bottom, s_left)
        if all(p is not None for p in (sTL, sTR, sBR, sBL)):
            snap_quad = _order_corners(np.array([sTL, sTR, sBR, sBL], np.float32))
            diag = float(np.hypot(*(orig_quad[2] - orig_quad[0]))) + 1e-6
            shift = float(np.linalg.norm(snap_quad - orig_quad, axis=1).max())
            if shift < 0.25 * diag:
                debug_lines.append(f"skjerm: kant-snap ok (flytt {shift:.0f}px)")
                return snap_quad
            debug_lines.append(f"skjerm: kant-snap forkastet (flytt {shift:.0f}px)")

    return orig_quad


def detect_screen_quad(img_bgr, cfg, debug_lines=None):
    """Finn skjermhjørner i originalkoordinater. (quad, area_frac, rough_quad) eller None."""
    if debug_lines is None:
        debug_lines = []
    h0, w0 = img_bgr.shape[:2]
    scale = min(cfg.get('screen_work_size', 480) / max(h0, w0), 1.0)
    small = cv2.resize(img_bgr, (int(w0 * scale), int(h0 * scale)),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = _normalize(gray, cfg)
    grayb = cv2.GaussianBlur(gray, (0, 0), cfg.get('screen_blur_sigma', 3.5))

    roi = None
    if cfg.get('screen_use_contrast_roi', True):
        roi = _apparatus_roi(gray, cfg, debug_lines)
        if roi is None:
            return None

    contour, area_frac = _screen_blob(grayb, cfg, debug_lines, roi=roi)
    if contour is None:
        return None

    # Konveks innhylling: fjerner konkave utbuktninger (smal forbindelse til apparatus)
    # uten å endre konvekse deler av skjerm-blobben.
    if cfg.get('screen_blob_convex_hull', False):
        hull_pts = cv2.convexHull(contour.astype(np.int32))
        h_tmp, w_tmp = gray.shape
        hull_mask = np.zeros((h_tmp, w_tmp), np.uint8)
        cv2.fillConvexPoly(hull_mask, hull_pts, 255)
        hull_cnts, _ = cv2.findContours(hull_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_NONE)
        if hull_cnts:
            contour = hull_cnts[0].reshape(-1, 2).astype(np.float32)

    rough = _rough_quad(contour)

    gx = cv2.Sobel(grayb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grayb, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)

    refined = _refine_from_contour(contour, rough, gmag, cfg, debug_lines,
                                   gx=gx, gy=gy, roi=roi)
    if refined is None:
        return None
    debug_lines.append(f"skjerm: raffinert (areal {area_frac:.2f})")
    return refined / scale, area_frac, rough / scale


def rectify_to_screen(img_bgr, cfg, debug_lines=None):
    """Perspektivkorriger og beskjær. Returns (warped, M, quad, rough_quad) eller None."""
    res = detect_screen_quad(img_bgr, cfg, debug_lines)
    if res is None:
        return None
    rect, area_frac, rough = res
    (tl, tr, br, bl) = rect
    wA = np.hypot(*(br - bl)); wB = np.hypot(*(tr - tl))
    hA = np.hypot(*(tr - br)); hB = np.hypot(*(tl - bl))
    W = int(round(max(wA, wB))); H = int(round(max(hA, hB)))
    if W < 50 or H < 50:
        if debug_lines is not None:
            debug_lines.append("skjerm: for liten etter retting")
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_LINEAR)
    return warped, M, rect, rough


def draw_quad(img_bgr, quad, rough=None, scale_to=900):
    """Tegn detektert firkant (rød) og evt. grovt estimat (gult) på nedskalert kopi."""
    h0, w0 = img_bgr.shape[:2]
    s = min(scale_to / max(h0, w0), 1.0)
    vis = cv2.resize(img_bgr, (int(w0 * s), int(h0 * s)))
    if rough is not None:
        cv2.polylines(vis, [(rough * s).astype(np.int32)], True, (0, 200, 255), 1)
    if quad is not None:
        pts = (quad * s).astype(np.int32)
        cv2.polylines(vis, [pts], True, (0, 0, 255), 2)
        for p in pts:
            cv2.circle(vis, tuple(p), 5, (0, 255, 0), -1)
    return vis
