"""
Treffdeteksjon: finner kulehull-markører på elektronisk skive.

Markørtyper (Kongsberg-display):
  - 'filled': hvit fylt sirkel (~0.82 * delta i diameter) med liten mørk
    senterprikk. Brukes for tidligere skudd.
  - 'outline': tynn lys sirkelkontur med liten lys senterprikk (siste skudd).

Deteksjon: HoughCircles på ROI rundt senteret (markørradius er kjent fra
ringkalibreringen, skala-invariant), deretter mønstervalidering og
senterprikk-raffinering.
"""

import cv2
import numpy as np

import circles as circle_det
import overlap as overlap_mod


def _annulus_percentile(gray, cx, cy, r_in, r_out, q):
    """Intensitets-persentil q i annulus [r_in, r_out] rundt (cx, cy)."""
    h, w = gray.shape
    x0 = max(0, int(cx - r_out)); x1 = min(w, int(cx + r_out) + 1)
    y0 = max(0, int(cy - r_out)); y1 = min(h, int(cy + r_out) + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr2 = (xx - cx) ** 2 + (yy - cy) ** 2
    m = (rr2 >= r_in ** 2) & (rr2 <= r_out ** 2)
    if not np.any(m):
        return None
    return float(np.percentile(gray[y0:y1, x0:x1][m], q))


def _marker_pattern_scores(gray, cx, cy, marker_r, dot_r, strict=True):
    """
    Mønster-score for de to markørtypene, normalisert til ~[0, 1].
    Persentiler i stedet for middel gjør scorene robuste mot at
    nabomarkører overlapper sonene.

    filled:  lys disk (0.35-0.75 r) mot mørk senterprikk og mørkere omgivelse
    outline: lys rand (0.85-1.1 r) og lys prikk, mørkt imellom

    strict=False hopper over vinkeldekning-kravene; brukes på rå
    Hough-posisjoner som kan ligge flere px unna markørsenteret.
    Full validering (strict=True) gjøres etter senterraffinering.
    """
    dot = _annulus_percentile(gray, cx, cy, 0.0, dot_r * 0.7, 50)
    disc = _annulus_percentile(gray, cx, cy, marker_r * 0.35, marker_r * 0.75, 50)
    # Bakgrunn: nedre kvartil, slik at lyse nabomarkører ikke teller med
    bg = _annulus_percentile(gray, cx, cy, marker_r * 1.25, marker_r * 1.6, 25)
    if None in (dot, disc, bg):
        return 0.0, 0.0

    # Normaliser med en typisk kontrast (hvit-svart på displayet)
    span = 255.0

    # Fylt markør: disk lysere enn prikk OG lysere enn bakgrunn
    s_filled = min(disc - dot, disc - bg) / span

    # Strukturkrav: markørens egen hvite disk skal være lys rundt (nesten)
    # hele omkretsen. Mørke lommer mellom ekte markører består ikke testen.
    if strict and s_filled > 0:
        disc_thr = 0.5 * (disc + max(dot, bg))
        cov = _disc_angular_coverage(gray, cx, cy, marker_r, disc_thr)
        if cov < 0.75:
            s_filled = 0.0

    # Konturmarkør: rand og prikk lysere enn sonen mellom dem.
    # Randstreken er tynn, så øvre kvartil fanger den i annulusen.
    rim = _annulus_percentile(gray, cx, cy, marker_r * 0.85, marker_r * 1.1, 75)
    dot_hi = _annulus_percentile(gray, cx, cy, 0.0, dot_r * 0.7, 75)
    between = _annulus_percentile(gray, cx, cy, marker_r * 0.4, marker_r * 0.7, 50)
    if None in (rim, dot_hi, between):
        return max(0.0, float(s_filled)), 0.0
    s_outline = min(rim - between, dot_hi - between) / span

    # Konturmarkøren er en KONTINUERLIG sirkel: krev at randen er lys
    # rundt (nesten) hele omkretsen, ellers er det bare lyse nabofragmenter.
    if strict and s_outline > 0:
        cov = _rim_angular_coverage(gray, cx, cy, marker_r,
                                    thresh=between + 0.15 * span)
        if cov < 0.75:
            s_outline = 0.0

    return max(0.0, float(s_filled)), max(0.0, float(s_outline))


def _disc_angular_coverage(gray, cx, cy, marker_r, thresh, n_angles=48):
    """
    Andel av vinkler der disksonen (0.35-0.7 r) i snitt er lys.
    Snitt per vinkel tåler display-pikselering og tynne ringstreker.
    """
    h, w = gray.shape
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)
    radii = np.linspace(0.35 * marker_r, 0.7 * marker_r, 6)
    xs = np.clip((cx + np.outer(radii, ca)).round().astype(int), 0, w - 1)
    ys = np.clip((cy + np.outer(radii, sa)).round().astype(int), 0, h - 1)
    vals = gray[ys, xs].astype(np.float32)   # (n_radii, n_angles)
    return float(np.mean(np.mean(vals, axis=0) > thresh))


def _rim_angular_coverage(gray, cx, cy, marker_r, thresh, n_angles=48):
    """
    Andel av vinkler der randsonen (0.85-1.1 r) har en piksel over thresh
    OG sonen innenfor (0.45-0.7 r) er mørk. En ekte konturmarkør oppfyller
    begge rundt hele omkretsen; tilfeldige lyse nabofragmenter gjør ikke det.
    """
    h, w = gray.shape
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    ca, sa = np.cos(angles), np.sin(angles)

    def _sample(r_lo, r_hi):
        radii = np.linspace(r_lo, r_hi, 6)
        xs = np.clip((cx + np.outer(radii, ca)).round().astype(int), 0, w - 1)
        ys = np.clip((cy + np.outer(radii, sa)).round().astype(int), 0, h - 1)
        return gray[ys, xs]          # (n_radii, n_angles)

    rim_bright = np.max(_sample(0.85 * marker_r, 1.1 * marker_r), axis=0) > thresh
    inner_dark = np.max(_sample(0.45 * marker_r, 0.7 * marker_r), axis=0) <= thresh
    return float(np.mean(rim_bright & inner_dark))


def _refine_on_dot(gray, cx, cy, dot_r, marker_type):
    """
    Raffiner treffsenter mot senterprikken.
    Fylt markør: prikken er mørk. Konturmarkør: prikken er lys.

    Displayet har kraftig pikselrutenett (mørke spalter overalt), så en
    global vektet sentroid fortynnes. I stedet: glatt bort rutenettet,
    finn ekstremum, og ta sentroid kun i prikkens nabolag.
    """
    h, w = gray.shape
    win = int(round(dot_r * 2.0))
    x0 = max(0, int(round(cx)) - win); x1 = min(w, int(round(cx)) + win + 1)
    y0 = max(0, int(round(cy)) - win); y1 = min(h, int(round(cy)) + win + 1)
    patch = gray[y0:y1, x0:x1].astype(np.float32)
    if patch.size == 0:
        return cx, cy
    sm = cv2.GaussianBlur(patch, (0, 0), max(1.0, dot_r * 0.5))
    if marker_type == 'filled':
        dev = np.max(sm) - sm   # mørk prikk -> høyt avvik
    else:
        dev = sm - np.min(sm)   # lys prikk
    py, px = np.unravel_index(int(np.argmax(dev)), dev.shape)

    # Sentroid begrenset til prikkens nabolag rundt ekstremet
    yy, xx = np.mgrid[0:dev.shape[0], 0:dev.shape[1]]
    near = (xx - px) ** 2 + (yy - py) ** 2 <= dot_r ** 2
    wgt = np.where(near, dev, 0.0) ** 2
    s = np.sum(wgt)
    if s < 1e-6:
        return cx, cy
    return float(x0 + np.sum(xx * wgt) / s), float(y0 + np.sum(yy * wgt) / s)


def _refine_white(gray, x, y, marker_r, dot_r):
    """Raffiner hvit-sone-kandidat: velg polaritet etter prikken (filled = moerk
    senterprikk, outline = lys senterprikk) og sentrer paa den."""
    disc = _annulus_percentile(gray, x, y, marker_r * 0.35, marker_r * 0.75, 50)
    dot = _annulus_percentile(gray, x, y, 0.0, dot_r * 0.7, 50)
    mtype = 'filled' if (disc is not None and dot is not None and dot <= disc) else 'outline'
    rx, ry = _refine_on_dot(gray, x, y, dot_r, mtype)
    if np.hypot(rx - x, ry - y) > dot_r * 1.5:
        rx, ry = x, y
    return rx, ry, mtype


def detect_hits(gray, calib, cfg, debug_lines=None):
    """
    Finn alle treffmarkører.

    Args:
        gray: gråskalabilde i originaloppløsning (uint8)
        calib: ringkalibrering (rings.calibrate_and_refine)
        cfg: config
        debug_lines: valgfri liste for debug-tekst

    Returns:
        Liste av dicts: {'x', 'y', 'type', 'score', 'hough_r'}
    """
    def log(s):
        if debug_lines is not None:
            debug_lines.append(s)

    cx0, cy0 = calib['center']
    delta = calib['delta_px']
    R10 = calib['R10_px']
    # Soeke-ROI helt UT til utenfor ytterste poengring (verdi 1). Bruk comb-en
    # (ekstrapolert: R10 + 9*delta), ikke bare ytterste DETEKTERTE ring - den ligger
    # ofte lenger inne, slik at ROI-grensen ellers havner inni 1-poengsbaandet.
    r_ring1 = R10 + 9.0 * delta
    search_r = cfg.get('hit_search_r_max_frac', 1.05) * r_ring1

    marker_r = cfg.get('hit_marker_radius_frac', 0.41) * delta
    dot_r = cfg.get('hit_dot_radius_frac', 0.105) * delta

    # Enhetlig stemme-detektor over HELE skiva (erstatter HoughCircles + det
    # separate hvit-sone-passet). Retnings-gatingen fjerner de konsentriske
    # poengringene (ogsaa de svarte i bull), saa markorene staar igjen overalt.
    if cfg.get('hit_unified_detector', False):
        inner_r = cfg.get('hit_unified_inner_frac', 0.0) * delta
        circ = circle_det.detect_circles(gray, (cx0, cy0), search_r, marker_r, dot_r,
                                          cfg, inner_r=inner_r)
        hits = []
        for (X, Y, vscore) in circ:
            rx, ry, mtype = _refine_white(gray, X, Y, marker_r, dot_r)
            hits.append({'x': rx, 'y': ry, 'type': mtype,
                         'score': float(vscore), 'hough_r': marker_r})
        log(f"Enhetlig stemme-detektor: {len(circ)} kandidater (markor_r={marker_r:.1f}px)")
        min_dist = cfg.get('hit_min_dist_frac', 0.6) * marker_r
        hits.sort(key=lambda t: -t['score'])
        kept = []
        for t in hits:
            if all(np.hypot(t['x'] - k['x'], t['y'] - k['y']) >= min_dist for k in kept):
                kept.append(t)
        log(f"Treff etter dedup: {len(kept)}")
        for t in kept:
            log(f"  ({t['x']:.1f}, {t['y']:.1f}) type={t['type']} score={t['score']:.2f}")
        # Overlapp-pass maa kjore ogsaa etter enhetlig detektor
        if cfg.get('hit_overlap_pass', True):
            extra = overlap_mod.find_overlap_hits(gray, kept, calib, cfg)
            if extra:
                log(f"Overlapp-pass (enhetlig): {len(extra)} nye kandidater funnet")
                for e in extra:
                    log(f"  ({e['x']:.1f}, {e['y']:.1f}) ncc={e['score']:.2f}")
                kept.extend(extra)
        # Sentrum-sveip: template-scan over bull-sonen for radialt undertrykte treff
        if cfg.get('hit_center_scan', True):
            extra = overlap_mod.find_center_hits(gray, kept, calib, cfg)
            if extra:
                log(f"Sentrum-sveip: {len(extra)} nye kandidater funnet")
                for e in extra:
                    log(f"  ({e['x']:.1f}, {e['y']:.1f}) ncc={e['score']:.2f}")
                kept.extend(extra)
        return kept

    h, w = gray.shape
    x0 = max(0, int(cx0 - search_r)); x1 = min(w, int(cx0 + search_r) + 1)
    y0 = max(0, int(cy0 - search_r)); y1 = min(h, int(cy0 + search_r) + 1)
    roi = gray[y0:y1, x0:x1]

    blur = cv2.GaussianBlur(roi, (0, 0), 2.0)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=cfg.get('hit_hough_dp', 2.0),
        minDist=cfg.get('hit_min_dist_frac', 0.6) * marker_r,
        param1=cfg.get('hit_hough_param1', 120),
        param2=cfg.get('hit_hough_param2', 30),
        minRadius=int(0.78 * marker_r),
        maxRadius=int(1.22 * marker_r),
    )

    candidates = []
    if circles is not None:
        for (x, y, r) in circles[0]:
            candidates.append((float(x) + x0, float(y) + y0, float(r)))
    log(f"Hough: {len(candidates)} kandidater (markor_r={marker_r:.1f}px)")

    min_score = cfg.get('hit_validate_min_score', 0.35)
    hits = []
    for (x, y, r) in candidates:
        d_center = np.hypot(x - cx0, y - cy0)
        if d_center > search_r:
            continue
        # Hough-posisjonen kan ligge flere px unna markørsenteret. Raffiner
        # alltid mot senterprikken (begge polariteter), og valider strengt
        # KUN i raffinert posisjon.
        best = None
        for mtype in ('filled', 'outline'):
            rx, ry = _refine_on_dot(gray, x, y, dot_r, mtype)
            # Ikke la raffineringen dra senteret langt vekk
            if np.hypot(rx - x, ry - y) > dot_r * 1.5:
                rx, ry = x, y
            s_filled, s_outline = _marker_pattern_scores(gray, rx, ry, marker_r, dot_r)
            s = s_filled if mtype == 'filled' else s_outline
            if s >= min_score and (best is None or s > best['score']):
                best = {'x': rx, 'y': ry, 'type': mtype, 'score': s, 'hough_r': r}
        if best is None:
            log(f"  forkastet ({x:.0f},{y:.0f})")
            continue
        hits.append(best)

    # Hvit-sone-pass: maalrettet sirkeldetektor (circles.py) for de svake graa
    # markorene i lyssonen som Hough ikke foreslaar. Stemme-gating (kompletthet +
    # stoerrelse + retning) er validering nok; vi raffinerer kun senter + polaritet.
    if cfg.get('hit_white_zone_pass', True):
        inner_r = cfg.get('hit_white_inner_frac', 3.4) * delta
        # Den globale circ_acc_blur_frac er senket til 0.06 for at den enhetlige
        # detektoren skal skille tette bull-klynger; det over-detekterer i hvit-
        # sonen. Denne (eldre, naa superseded) stien bruker sin egen hoeyere blur.
        cfg_white = dict(cfg, circ_acc_blur_frac=cfg.get('hit_white_blur_frac', 0.12))
        circ = circle_det.detect_circles(gray, (cx0, cy0), search_r, marker_r, dot_r,
                                          cfg_white, inner_r=inner_r)
        for (X, Y, vscore) in circ:
            rx, ry, mtype = _refine_white(gray, X, Y, marker_r, dot_r)
            hits.append({'x': rx, 'y': ry, 'type': mtype,
                         'score': float(vscore), 'hough_r': marker_r})
        log(f"Hvit-sone-pass: {len(circ)} kandidater (innenfor {inner_r:.0f}px ekskludert)")

    # Dedupliser: behold høyest score innen min avstand
    min_dist = cfg.get('hit_min_dist_frac', 0.6) * marker_r
    hits.sort(key=lambda t: -t['score'])
    kept = []
    for t in hits:
        if all(np.hypot(t['x'] - k['x'], t['y'] - k['y']) >= min_dist for k in kept):
            kept.append(t)

    # Overlapp-pass: template-matching for delvis-skjulte treff naer eksisterende.
    # Trigges kun hvis antall treff < hit_overlap_trigger (default 10).
    if cfg.get('hit_overlap_pass', True):
        extra = overlap_mod.find_overlap_hits(gray, kept, calib, cfg)
        if extra:
            log(f"Overlapp-pass: {len(extra)} nye kandidater funnet")
            for e in extra:
                log(f"  ({e['x']:.1f}, {e['y']:.1f}) ncc={e['score']:.2f}")
            kept.extend(extra)
    # Sentrum-sveip: template-scan over bull-sonen for radialt undertrykte treff
    if cfg.get('hit_center_scan', True):
        extra = overlap_mod.find_center_hits(gray, kept, calib, cfg)
        if extra:
            log(f"Sentrum-sveip: {len(extra)} nye kandidater funnet")
            for e in extra:
                log(f"  ({e['x']:.1f}, {e['y']:.1f}) ncc={e['score']:.2f}")
            kept.extend(extra)

    log(f"Treff etter validering/dedup: {len(kept)}")
    for t in kept:
        log(f"  ({t['x']:.1f}, {t['y']:.1f}) type={t['type']} score={t['score']:.2f}")
    return kept
