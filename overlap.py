"""
Template-matching for delvis-skjulte treffmarkorer naer eksisterende deteksjoner.
Trigges naar antall treff < hit_overlap_trigger (default 10).

Strategi (musings 2026-07-01):
  1. Hent stemmekart med circle_vote_map (samme som hoveddetektor).
  2. Finn lokale maxima med LITEN NMS (standard NMS ~28px undertrykket den
     naerestliggende overlapp-toppen paa bare 22px avstand).
  3. For hvert detektert treff i bull-sonen: godta subterskel-topper som er
     naeere enn hit_overlap_max_dist_frac * marker_r (overlapp-definisjon) og
     over hit_overlap_vote_frac * amax.
  4. Valider med NCC mot template fra rent isolert treff.
     Template er annulaer (ytre halvdel av skiva) - tolererer skjult senterprikk
     og serie-senter-firkanten. Kravet er lavt (0.40) fordi kun halvt synlig disc.

Funn: peak ved (1413,941) 22px fra detektert 10.6-treff, 0.34*amax, estimert 10.5.
Den ble undertrykket av standard NMS med radius 28px.
"""
import cv2
import numpy as np
import circles as circle_det


def _odd(k):
    k = int(round(k))
    return max(1, k + 1 if k % 2 == 0 else k)


def _extract_patch(gray, cx, cy, radius):
    """Hent patch rundt (cx, cy). Padder med median-fill ved kanter."""
    h, w = gray.shape
    r = int(radius)
    size = 2 * r + 1
    x0 = int(round(cx)) - r
    y0 = int(round(cy)) - r
    pad_l = max(0, -x0);  pad_t = max(0, -y0)
    pad_r = max(0, x0 + size - w)
    pad_b = max(0, y0 + size - h)
    x0c = max(0, x0);  y0c = max(0, y0)
    crop = gray[y0c:min(h, y0+size), x0c:min(w, x0+size)]
    if pad_l or pad_t or pad_r or pad_b:
        fill = int(np.median(gray))
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r,
                                  cv2.BORDER_CONSTANT, value=fill)
    return crop


def _annular_mask(radius_px, r_lo, r_hi):
    """Annulaer maske: 1 i [r_lo, r_hi], 0 ellers."""
    size = 2 * radius_px + 1
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (radius_px, radius_px), int(r_hi), 255, -1)
    cv2.circle(mask, (radius_px, radius_px), max(0, int(r_lo) - 1), 0, -1)
    return mask


def _half_annular_mask(radius_px, r_lo, r_hi, dx, dy):
    """Halvt annulaert maske: kun halvkulen som peker i retning (dx, dy)."""
    size = 2 * radius_px + 1
    full = _annular_mask(radius_px, r_lo, r_hi)
    yy, xx = np.mgrid[:size, :size]
    cx = cy = radius_px
    dot = (xx - cx) * dx + (yy - cy) * dy
    half = (dot >= 0).astype(np.uint8) * 255
    return cv2.bitwise_and(full, half)


def _crescent_mask(radius_px, r_lo, r_hi, dom_dx, dom_dy, dom_r):
    """Geometrisk korrekt synlig maanesigd-maske for et delvis-skjult treff.

    Aktiverer piksler som er:
    - inni kandidat-skiven    (r_lo < r <= r_hi)
    - utenfor dominant-skiven (avstand fra dom-senter >= dom_r)

    dom_dx, dom_dy = dominant disc center - candidate center (i piksler i patch-ramme)
    dom_r          = radius til dominant-skiven (= marker_r)
    """
    size = 2 * radius_px + 1
    yy, xx = np.mgrid[:size, :size]
    cx = cy = radius_px
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    dom_r2 = (xx - cx - dom_dx) ** 2 + (yy - cy - dom_dy) ** 2
    inside_cand  = r2 <= r_hi ** 2
    outside_dom  = dom_r2 >= dom_r ** 2
    outside_rlo  = r2 > r_lo ** 2
    mask = (inside_cand & outside_dom & outside_rlo).astype(np.uint8) * 255
    return mask


def find_overlap_hits(gray, hits, calib, cfg):
    """
    Sekundaer deteksjon av delvis-skjulte treff naer eksisterende deteksjoner.
    Returnerer liste med nye hits (samme dict-format som hits.py).
    Kaller kun hvis len(hits) < cfg['hit_overlap_trigger'].
    """
    trigger = cfg.get('hit_overlap_trigger', 10)
    if len(hits) >= trigger:
        return []

    cx0, cy0 = calib['center']
    delta    = calib['delta_px']
    R10      = calib['R10_px']
    search_r = cfg.get('hit_search_r_max_frac', 1.05) * (R10 + 9.0 * delta)
    marker_r = cfg.get('hit_marker_radius_frac', 0.41) * delta
    dot_r    = cfg.get('hit_dot_radius_frac', 0.105) * delta

    # ---- 1. Stemmekart (gjenbruk av hoveddetektor-infrastruktur) ----
    acc, dbg = circle_det.circle_vote_map(
        gray, (cx0, cy0), search_r, marker_r, dot_r, cfg, inner_r=0.0)
    ox, oy = dbg['offset']
    amax   = float(acc.max()) if acc.size else 0.0
    if amax < 1e-6:
        return []

    # ---- 2. Lokale maxima med LITEN NMS ----
    # Standard NMS bruker ~0.7*marker_r radius (~28px). Det undertrykte toppen
    # for det delvis-skjulte treffet paa bare 22px fra dominerende treff.
    small_nms_r = _odd(cfg.get('hit_overlap_nms_frac', 0.20) * marker_r)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (small_nms_r, small_nms_r))
    dil    = cv2.dilate(acc, kernel)
    vote_floor = cfg.get('hit_overlap_vote_frac', 0.25) * amax
    py_all, px_all = np.where((acc == dil) & (acc >= vote_floor))

    # ---- 3. Bygg NCC-template fra mest isolert eksisterende treff ----
    def _isolation(h):
        others = [o for o in hits if o is not h]
        if not others:
            return 0.0
        return min(np.hypot(h['x']-o['x'], h['y']-o['y']) for o in others) * h['score']

    src    = max(hits, key=_isolation)
    # r_lo lavere (0.20) enn full-ring (0.50): inkluderer graa disc, ikke bare ytterkant.
    # Halvmaske bruker bare halvkulen som vender bort fra ankeret = synlig maanesigd.
    r_lo   = cfg.get('hit_overlap_tmpl_r_lo', 0.20) * marker_r
    r_hi   = cfg.get('hit_overlap_tmpl_r_hi', 1.05) * marker_r
    tmpl_r = int(r_hi) + 2
    tmpl_patch = _extract_patch(gray, src['x'], src['y'], tmpl_r)
    tmpl_f = tmpl_patch.astype(np.float32) / 255.0

    # ---- 4. Kjoer gjennom peaks og valider ----
    max_dist    = cfg.get('hit_overlap_max_dist_frac', 1.0) * marker_r
    min_offset  = cfg.get('hit_overlap_min_offset_frac', 0.20) * marker_r
    ncc_thresh  = cfg.get('hit_overlap_ncc_thresh', 0.40)
    min_dist_other = cfg.get('hit_min_dist_frac', 0.6) * marker_r
    max_anchor_r   = cfg.get('hit_overlap_max_anchor_r_frac', 3.0) * delta

    # Treff i bull-sonen som er ankerpunkter
    anchors = [h for h in hits
               if np.hypot(h['x']-cx0, h['y']-cy0) <= max_anchor_r]

    new_hits = []
    already  = list(hits)

    for anchor in anchors:
        ax, ay = anchor['x'], anchor['y']
        for py, px in zip(py_all, px_all):
            X = float(px + ox)
            Y = float(py + oy)
            d_anker = np.hypot(X - ax, Y - ay)
            if d_anker < min_offset or d_anker > max_dist:
                continue
            # Ikke for naert ANDRE eksisterende treff (ikke ankeret selv)
            too_close = any(
                np.hypot(X - k['x'], Y - k['y']) < min_dist_other
                for k in already if k is not anchor
            )
            if too_close:
                continue
            # Innenfor skivens soekeomraade
            if np.hypot(X - cx0, Y - cy0) > search_r:
                continue

            # NCC-validering: geometrisk korrekt synlig maanesigd-maske.
            # Aktiverer kun piksler inni kandidat-skiven OG utenfor dominant-skiven
            # (= den faktisk synlige halvmaanen). Mye bedre enn halvkule-maske
            # fordi den unngaar regionen som dekkes av dominant-treffet.
            dom_dx = ax - X;  dom_dy = ay - Y   # dominant senter relativt til kandidat
            cmask   = _crescent_mask(tmpl_r, r_lo, r_hi, dom_dx, dom_dy, marker_r)
            cmask_f = (cmask > 0).astype(np.float32)
            if cmask_f.sum() < 10:
                continue

            cand_patch = _extract_patch(gray, X, Y, tmpl_r)
            if cand_patch.shape != tmpl_f.shape:
                continue
            cand_f = cand_patch.astype(np.float32) / 255.0
            res = cv2.matchTemplate(cand_f, tmpl_f,
                                    cv2.TM_CCOEFF_NORMED, mask=cmask_f)
            ncc_val = float(res.max()) if res.size > 0 else 0.0
            if ncc_val < ncc_thresh:
                continue

            v = float(acc[py, px])
            new_hit = {
                'x': X, 'y': Y,
                'type': anchor.get('type', 'filled'),
                'score': ncc_val,
                'hough_r': marker_r,
            }
            new_hits.append(new_hit)
            already.append(new_hit)

    # Dedupliser nye treff mot hverandre
    min_dist_frac = cfg.get('hit_min_dist_frac', 0.6) * marker_r
    deduped = []
    for h in sorted(new_hits, key=lambda t: -t['score']):
        if all(np.hypot(h['x']-k['x'], h['y']-k['y']) >= min_dist_frac
               for k in deduped):
            deduped.append(h)

    return deduped


def find_center_hits(gray, hits, calib, cfg):
    """
    Template-sveip over bull-sonen for treff der stemmekart er undertrykt.

    Nær senteret er stemmekart-topper undertrykt fordi:
    - circ_radial_reject fjerner kanter som peker radialt mot maalsenter
    - skivekanter for et sentralt treff peker nettopp radialt
    => stemmekart gir null/lite signal for treff i innerste sone.

    Strategi: bruk cv2.matchTemplate for aa gli full annulaer disc-template
    over hele bull-sonen (innenfor R10) og finn posisjoner der NCC er hoy.
    Ingen stemmekart-veiledning - rent template-drevet soek.
    """
    trigger = cfg.get('hit_overlap_trigger', 10)
    if len(hits) >= trigger:
        return []
    if not cfg.get('hit_center_scan', True):
        return []

    cx0, cy0 = calib['center']
    delta    = calib['delta_px']
    R10      = calib['R10_px']
    marker_r = cfg.get('hit_marker_radius_frac', 0.41) * delta

    scan_r    = cfg.get('hit_center_scan_r_frac', 1.0) * R10
    ncc_thresh = cfg.get('hit_center_ncc_thresh', 0.10)
    min_dist   = cfg.get('hit_min_dist_frac', 0.6) * marker_r

    # ---- Template fra mest isolert eksisterende treff ----
    def _iso(h):
        others = [o for o in hits if o is not h]
        if not others:
            return 0.0
        return min(np.hypot(h['x']-o['x'], h['y']-o['y']) for o in others) * h['score']

    src      = max(hits, key=_iso)
    r_lo     = cfg.get('hit_overlap_tmpl_r_lo', 0.20) * marker_r
    r_hi     = cfg.get('hit_overlap_tmpl_r_hi', 1.05) * marker_r
    tmpl_r_px = int(r_hi) + 2
    tmpl_size = 2 * tmpl_r_px + 1

    # Full annulaer maske: sentralt treff er ikke okkludert av nabo-treff,
    # saa vi kan bruke hele disc-omrisset som referanse.
    tmask   = _annular_mask(tmpl_r_px, r_lo, r_hi)
    tmask_f = (tmask > 0).astype(np.float32)
    tmpl_f  = _extract_patch(gray, src['x'], src['y'], tmpl_r_px).astype(np.float32) / 255.0

    # ---- ROI rundt scan-omraadet (med padding = tmpl_r_px) ----
    h_img, w_img = gray.shape
    x0 = max(0, int(cx0 - scan_r) - tmpl_r_px)
    y0 = max(0, int(cy0 - scan_r) - tmpl_r_px)
    x1 = min(w_img, int(cx0 + scan_r) + tmpl_r_px + 1)
    y1 = min(h_img, int(cy0 + scan_r) + tmpl_r_px + 1)
    roi = gray[y0:y1, x0:x1]
    if roi.shape[0] < tmpl_size or roi.shape[1] < tmpl_size:
        return []
    roi_f = roi.astype(np.float32) / 255.0

    # ---- Glidende krysskorrellasjon over ROI ----
    corr = cv2.matchTemplate(roi_f, tmpl_f, cv2.TM_CCOEFF_NORMED, mask=tmask_f)
    # corr[ry, rx] svarer til template-senter ved bilde-coord
    # (X = x0+rx+tmpl_r_px, Y = y0+ry+tmpl_r_px)

    # NMS: behold lokale maxima paa skala min_dist
    nms_k  = _odd(min_dist)
    kern   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nms_k, nms_k))
    dil    = cv2.dilate(corr, kern)
    ry_arr, rx_arr = np.where((corr == dil) & (corr >= ncc_thresh))

    candidates = []
    for ry, rx in zip(ry_arr, rx_arr):
        X = float(x0 + rx + tmpl_r_px)
        Y = float(y0 + ry + tmpl_r_px)
        if np.hypot(X - cx0, Y - cy0) > scan_r:
            continue
        if any(np.hypot(X - h['x'], Y - h['y']) < min_dist for h in hits):
            continue
        candidates.append({
            'x': X, 'y': Y,
            'score': float(corr[ry, rx]),
            'type': 'filled',
            'hough_r': marker_r,
        })

    # Dedupliser mot hverandre (behold hoyeste NCC)
    candidates.sort(key=lambda c: -c['score'])
    kept = []
    for c in candidates:
        if all(np.hypot(c['x']-k['x'], c['y']-k['y']) >= min_dist for k in kept):
            kept.append(c)

    return kept
