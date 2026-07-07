"""
Ring-kalibrering og senterraffinering i polart domene.

Arbeider på originaloppløst gråskalabilde:
  1) warpPolar rundt antatt senter (radius langs kolonner, vinkel langs rader)
  2) Radiell gradient |dP/drho| beregnes direkte på det lille polarbildet
  3) Radiell profil -> topper -> robust tilpasning av aritmetisk progresjon
     (ringene er ekvidistante) gir delta_px (px per poeng) og ringradiene
  4) Senterraffinering: per-ring harmonisk tilpasning r(theta); k=1-leddet
     gir senterfeil (ex, ey). Midles over flere ringer og itereres.

Alle returnerte størrelser er i originale bildekoordinater.
"""

import cv2
import numpy as np


def warp_polar_gray(gray, center, r_max, r_samples, theta_samples):
    """
    Polar transform: P[theta_row, rho_col], rho = col * (r_max / r_samples).
    Vinkel for rad i: phi = 2*pi * i / theta_samples (x mot høyre, y nedover).
    """
    P = cv2.warpPolar(
        gray.astype(np.float32),
        (int(r_samples), int(theta_samples)),
        (float(center[0]), float(center[1])),
        float(r_max),
        cv2.WARP_POLAR_LINEAR | cv2.INTER_LINEAR
    )
    return P


def radial_gradient_abs(P, sigma_rho):
    """Absolutt radiell gradient av polarbildet (Sobel langs rho-aksen).
    sigma_rho<=0 => ingen glatting langs radius (kun Sobel ksize=3)."""
    if sigma_rho and sigma_rho > 0:
        Pb = cv2.GaussianBlur(P, (0, 0), sigmaX=sigma_rho, sigmaY=1.0)
    else:
        Pb = P
    g = cv2.Sobel(Pb, cv2.CV_32F, 1, 0, ksize=3)
    return np.abs(g)


def _subpixel_parabola(v_m, v_0, v_p):
    denom = 2.0 * (v_m - 2.0 * v_0 + v_p)
    if abs(denom) < 1e-12:
        return 0.0
    delta = (v_m - v_p) / denom
    if not np.isfinite(delta) or abs(delta) > 0.5:
        return 0.0
    return float(delta)


def estimate_spacing_autocorr(H, min_lag, max_lag):
    """Estimer ringavstand (i profil-bins) via autokorrelasjon av radialprofilen."""
    Hc = H.astype(np.float64) - np.mean(H)
    ac = np.correlate(Hc, Hc, mode='full')[len(Hc) - 1:]
    max_lag = min(max_lag, len(ac) - 2)
    if max_lag <= min_lag:
        return None
    lag = int(np.argmax(ac[min_lag:max_lag])) + min_lag

    # Hvis den valgte laggen er en multippel av den sanne avstanden,
    # sjekk om lag/2 også er en sterk topp.
    while lag // 2 >= min_lag:
        half = lag // 2
        lo, hi = max(min_lag, half - 3), min(max_lag, half + 4)
        local = int(np.argmax(ac[lo:hi])) + lo
        if ac[local] > 0.5 * ac[lag] and abs(local - half) <= 3:
            lag = local
        else:
            break

    # Subpixel
    if 1 <= lag < len(ac) - 1:
        lag = lag + _subpixel_parabola(ac[lag - 1], ac[lag], ac[lag + 1])
    return float(lag)


def find_profile_peaks(H, min_frac, min_sep):
    """Lokale maksima i 1D-profil, grådig etter høyde med minimumsavstand."""
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    thresh = min_frac * np.max(H)
    idx = [i for i in range(1, n - 1)
           if H[i] >= thresh and H[i] >= H[i - 1] and H[i] > H[i + 1]]
    idx.sort(key=lambda i: -H[i])
    kept = []
    for i in idx:
        if all(abs(i - j) >= min_sep for j in kept):
            kept.append(i)
    peaks = []
    for i in kept:
        peaks.append((i + _subpixel_parabola(H[i - 1], H[i], H[i + 1]), float(H[i])))
    peaks.sort(key=lambda p: p[0])
    return peaks  # liste av (subpixel-posisjon, høyde)


def fit_ring_progression(peak_rho, delta0, cfg):
    """
    Robust tilpasning av ekvidistant progresjon til toppene.

    Args:
        peak_rho: array av topp-posisjoner (rho-px)
        delta0: startestimat for avstanden (rho-px)
        cfg: config

    Returns:
        dict med 'delta', 'rho_of_k' (callable er unødig - bruk a + delta*k),
        'a' (rho ved k=0), 'k_inlier', 'rho_inlier', eller None.
    """
    rho = np.asarray(sorted(peak_rho, reverse=True), dtype=np.float64)
    if len(rho) < 3 or delta0 is None or delta0 <= 0:
        return None

    # Relative indekser: k=0 for ytterste topp, k øker innover (radius avtar)
    k = np.round((rho[0] - rho) / delta0).astype(int)

    # Initial inliers: alle. Iterativ LSQ med fjerning av verste residual.
    sel = np.ones(len(rho), dtype=bool)
    a, d = rho[0], delta0
    max_resid_px = max(cfg.get('ring_fit_max_resid_frac', 0.04) * delta0, 2.0)
    for _ in range(len(rho)):
        kk, rr = k[sel], rho[sel]
        if len(rr) < 3:
            break
        A = np.column_stack([np.ones_like(kk, dtype=np.float64), -kk.astype(np.float64)])
        coef, *_ = np.linalg.lstsq(A, rr, rcond=None)
        a, d = float(coef[0]), float(coef[1])
        resid = np.abs(rho - (a - d * k))
        worst = np.argmax(np.where(sel, resid, -np.inf))
        if resid[worst] <= max_resid_px:
            break
        sel[worst] = False

    if d <= 0 or np.sum(sel) < 3:
        return None

    return {
        'a': a,                # rho for k=0 (ytterste inlier-ring)
        'delta': d,            # ringavstand i rho-px
        'k': k, 'inlier': sel,
        'rho': rho,
    }


def fit_equidistant_comb(H, cfg):
    """
    Robust ekvidistant ring-comb (v2).

    Idé (etter brukerens 'fyll opp nedenfra'): bruk TOPOGRAFISK PROMINENS for å
    velge robuste topper (persistens - hvor høyt 'vannet' stiger før en øy
    smelter inn i en høyere). Tilpass så en ekvidistant comb med TAK på antall
    ringer (<= comb_max_rings), forankret på de sterke ytre toppene. Score =
    sum av forklart topp-prominens, minus straff for hoppede sterke topper
    (oktav-for-stor) og tomme tenner (for-liten avstand), vektet med spenn.
    Returnerer dict kompatibel med fit_ring_progression eller None.
    """
    from scipy.signal import find_peaks
    from scipy.ndimage import gaussian_filter1d
    H = np.asarray(H, dtype=np.float64)
    n = len(H)
    if n < 10 or H.max() <= 0:
        return None

    # Glatt profilen slik at hver ring blir ÉN topp: dobbeltkanter (~strekbredde)
    # og skjev-duppeltopper (~0.45*avstand) smelter sammen. Bredden er en andel
    # av radien (skala-invariant, uavhengig av oktav-utsatt autokorrelasjon).
    sm = cfg.get('comb_smooth_frac', 0.02) * n
    if sm >= 0.5:
        H = gaussian_filter1d(H, sm)

    prom_min = cfg.get('comb_prom_frac', 0.04) * float(H.max())
    d_rough = estimate_spacing_autocorr(H, max(4, int(0.02 * n)), int(0.30 * n)) or 40.0
    min_dist = max(int(cfg.get('comb_min_dist_bins', 6)),
                   int(cfg.get('comb_min_dist_frac', 0.35) * d_rough))
    pk, props = find_peaks(H, prominence=prom_min, distance=min_dist)
    if len(pk) < cfg.get('comb_min_rings', 4):
        return None
    pk = pk.astype(np.float64)
    prom = props['prominences'].astype(np.float64)

    # ignorer senter-klynge (markører/kors)
    keep = pk > cfg.get('comb_inner_ignore_frac', 0.04) * n
    pk, prom = pk[keep], prom[keep]
    if len(pk) < cfg.get('comb_min_rings', 4):
        return None

    strong_thr = cfg.get('comb_strong_frac', 0.5) * float(prom.max())
    strong = pk[prom >= strong_thr]
    r_out = float(strong.max()) if len(strong) else float(pk.max())
    r_in = float(pk.min())

    max_teeth = int(cfg.get('comb_max_rings', 11))
    min_rings = int(cfg.get('comb_min_rings', 4))
    delta_min = max(cfg.get('comb_delta_min_bins', 25.0), r_out / max_teeth)
    delta_max = r_out / min_rings
    if delta_max <= delta_min:
        return None

    med_prom = float(np.median(prom))
    w_empty = cfg.get('comb_empty_penalty', 0.5)
    w_skip = cfg.get('comb_skip_penalty', 1.0)
    tol_frac = cfg.get('comb_tol_frac', 0.18)
    best = None
    for delta in np.arange(delta_min, delta_max + 1e-9, 1.0):
        tol = tol_frac * delta
        phases = np.unique(np.round(pk % delta))
        for phase in phases:
            kmax = int((r_out - phase) / delta)
            if kmax + 1 > max_teeth or kmax + 1 < min_rings:
                continue
            teeth = phase + np.arange(0, kmax + 1) * delta
            teeth = teeth[teeth >= 0.4 * delta]
            if len(teeth) < min_rings:
                continue
            explained = 0.0
            n_empty = 0
            used = np.zeros(len(pk), dtype=bool)
            for t in teeth:
                d = np.abs(pk - t)
                j = int(np.argmin(d))
                if d[j] <= tol:
                    explained += prom[j]
                    used[j] = True
                else:
                    n_empty += 1
            skipped = float(prom[(~used) & (prom >= strong_thr)].sum())
            span = (teeth[-1] - teeth[0]) / (r_out - r_in + 1e-9)
            score = (explained - w_empty * n_empty * med_prom - w_skip * skipped)
            score *= (0.5 + 0.5 * min(span, 1.0))
            if best is None or score > best[0]:
                best = (score, float(delta), float(phase), teeth.copy())

    if best is None:
        return None
    _, delta, phase, teeth = best
    rho = teeth[::-1].astype(np.float64)               # ytterst først
    k = np.round((rho[0] - rho) / delta).astype(int)
    return {'a': float(rho[0]), 'delta': float(delta),
            'k': k, 'inlier': np.ones(len(rho), dtype=bool), 'rho': rho}


def _fit_harmonics(theta, x, w, n_reject_rounds, reject_sigma):
    """
    Robust LSQ av x(theta) = C + B1 sin + D1 cos + B2 sin2 + D2 cos2.
    Returns (beta, rmse, mask) eller None.
    """
    mask = np.isfinite(x) & (w > 0)
    if np.sum(mask) < 16:
        return None
    X = np.column_stack([np.ones_like(theta), np.sin(theta), np.cos(theta),
                         np.sin(2 * theta), np.cos(2 * theta)])
    beta = None
    for _ in range(n_reject_rounds + 1):
        Xm, xm, wm = X[mask], x[mask], w[mask]
        sw = np.sqrt(wm)
        try:
            beta, *_ = np.linalg.lstsq(Xm * sw[:, None], xm * sw, rcond=None)
        except np.linalg.LinAlgError:
            return None
        resid = x - X @ beta
        s = np.sqrt(np.mean(resid[mask] ** 2))
        new_mask = mask & (np.abs(resid) <= max(reject_sigma * s, 0.5))
        if np.array_equal(new_mask, mask) or np.sum(new_mask) < 16:
            break
        mask = new_mask
    resid = (x - X @ beta)[mask]
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return beta, rmse, mask


def _ring_track(G, rho_c, band_px, min_mag_frac=0.2):
    """
    Per-vinkel radiell posisjon for én ring: vektet sentroid av |g|
    i båndet rundt rho_c. Returnerer (x(theta) i rho-px, vekt per rad).
    """
    theta_rows, rho_cols = G.shape
    lo = max(0, int(np.floor(rho_c - band_px)))
    hi = min(rho_cols, int(np.ceil(rho_c + band_px)) + 1)
    if hi - lo < 3:
        return None, None
    B = G[:, lo:hi]
    cols = np.arange(lo, hi, dtype=np.float64)
    s = B.sum(axis=1)
    smax = np.max(s)
    ok = s > min_mag_frac * smax
    x = np.full(theta_rows, np.nan)
    x[ok] = (B[ok] @ cols) / s[ok]
    w = np.where(ok, s, 0.0)
    return x, w


def calibrate_and_refine(gray, center0, r_outer_est, cfg, debug_lines=None):
    """
    Hovedinngang: kalibrer ringgeometri og raffiner senteret.

    Args:
        gray: gråskalabilde i originaloppløsning (uint8)
        center0: (cx, cy) startestimat i originalkoordinater
        r_outer_est: estimert radius for ytterste poengring (originalpx)
        cfg: config
        debug_lines: valgfri liste for debug-tekst

    Returns:
        calib dict:
            'center': (cx, cy)
            'delta_px': px per poengsone
            'R10_px': radius for 10-ringen
            'ring_values': array av ringverdier (1 = ytterst)
            'ring_radii_px': gjennomsnittsradius per ring
            'ring_harmonics': liste av (value, beta[5]) i originalpx
            'r_max', 'rho_scale': polar-geometri
        eller None ved feil.
    """
    def log(s):
        if debug_lines is not None:
            debug_lines.append(s)

    theta_samples = cfg.get('ring_theta_samples', 720)
    sigma_rho = cfg.get('ring_rho_sigma', 3.0)
    n_iters = cfg.get('ring_refine_iters', 4)

    pre_sigma = cfg.get('ring_pre_blur_sigma', 0.0)
    if pre_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), pre_sigma)  # demp moiré/aliasing

    center = (float(center0[0]), float(center0[1]))
    base_r_max = cfg.get('ring_r_max_frac', 1.06) * float(r_outer_est)
    if cfg.get('ring_generous_rmax', True):
        # Generøs r_max: ta med de hvite ytre ringene selv om grov-radius bommer
        # innover. Begrens til nær bildekant. Aldri mindre enn standard.
        h_, w_ = gray.shape
        r_edge = min(center[0], w_ - center[0], center[1], h_ - center[1])
        r_max = max(base_r_max,
                    min(cfg.get('ring_generous_mult', 2.6) * float(r_outer_est),
                        cfg.get('ring_generous_edge_frac', 0.95) * r_edge))
    else:
        r_max = base_r_max
    r_samples = int(round(r_max))  # ~1 px per rho-bin
    rho_scale = r_max / r_samples

    calib = None
    for it in range(n_iters):
        P = warp_polar_gray(gray, center, r_max, r_samples, theta_samples)
        G = radial_gradient_abs(P, sigma_rho)
        H = G.mean(axis=0)

        # Ring-comb: ny prominens/ekvidistant-modell (v2) eller gammel progresjon
        if cfg.get('ring_comb_v2', False):
            fit = fit_equidistant_comb(H, cfg)
            if fit is None:
                log(f"iter {it}: comb-v2 fant ingen ekvidistant ring-comb")
                return None
        else:
            delta0 = estimate_spacing_autocorr(
                H, min_lag=int(0.03 * r_samples), max_lag=int(0.35 * r_samples))
            if delta0 is None:
                log(f"iter {it}: autokorrelasjon feilet")
                return None
            peaks = find_profile_peaks(
                H, cfg.get('ring_peak_min_frac', 0.12),
                min_sep=max(3, int(cfg.get('ring_peak_min_sep_frac', 0.4) * delta0)))
            # Ignorer topper helt inntil senteret (senterkors/markørklynge)
            peaks = [p for p in peaks if p[0] > 0.4 * delta0]
            fit = fit_ring_progression([p[0] for p in peaks], delta0, cfg)
            if fit is None:
                log(f"iter {it}: progresjonstilpasning feilet ({len(peaks)} topper)")
                return None

        # Per-ring harmonisk tilpasning -> senterkorreksjon
        band_frac = 0.25 if it == 0 else 0.15
        band_px = band_frac * fit['delta']
        theta = 2.0 * np.pi * np.arange(theta_samples) / theta_samples

        corrections = []
        weights = []
        harmonics = []   # (k_rel, beta)
        for k_rel, rho_p, is_in in zip(fit['k'], fit['rho'], fit['inlier']):
            if not is_in:
                continue
            rho_fit = fit['a'] - fit['delta'] * k_rel
            x, w = _ring_track(G, rho_fit, band_px)
            if x is None:
                continue
            res = _fit_harmonics(theta, x, w,
                                 cfg.get('ring_harm_reject_rounds', 3),
                                 cfg.get('ring_harm_reject_sigma', 2.5))
            if res is None:
                continue
            beta, rmse, mask = res
            coverage = np.mean(mask)
            if coverage < cfg.get('ring_refine_min_coverage', 0.5):
                continue
            C, B1, D1 = beta[0], beta[1], beta[2]
            # r(phi) = R + ex*cos(phi) + ey*sin(phi)  =>  ex = D1, ey = B1 (rho-px)
            corrections.append((D1 * rho_scale, B1 * rho_scale))
            # Indre ringer vektes opp: treffene ligger der, og indre ringer
            # er mindre utsatt for perspektivskjevhet.
            w_inner = 1.0 / max(float(C), 1.0)
            weights.append(w_inner * coverage / (rmse + 0.3))
            harmonics.append((k_rel, beta, rmse, coverage))

        if len(corrections) < 2:
            log(f"iter {it}: for få ringer til senterkorreksjon ({len(corrections)})")
            return None

        corr = np.average(np.array(corrections), axis=0, weights=np.array(weights))
        center = (center[0] + corr[0], center[1] + corr[1])
        log(f"iter {it}: delta={fit['delta'] * rho_scale:.2f}px, "
            f"{len(corrections)} ringer, korreksjon=({corr[0]:.2f}, {corr[1]:.2f})")

        calib = (fit, harmonics)
        if abs(corr[0]) < 0.3 and abs(corr[1]) < 0.3:
            break

    if calib is None:
        return None

    fit, harmonics = calib
    delta_px = fit['delta'] * rho_scale

    # Ringverdier: ytterste inlier-ring antas å være verdien cfg['ring_value_outermost']
    v_out = cfg.get('ring_value_outermost', 1)
    k_out = int(np.min([k for (k, b, r, c) in harmonics]))

    ring_values = []
    ring_radii = []
    ring_harm = []
    for (k_rel, beta, rmse, cov) in sorted(harmonics, key=lambda h: h[0]):
        value = v_out + (k_rel - k_out)
        radius_px = float(beta[0]) * rho_scale
        ring_values.append(value)
        ring_radii.append(radius_px)
        ring_harm.append((value, beta * rho_scale, rmse, cov))

    # Globalt: R(v) = R10 + (10 - v) * delta  tilpasses alle ringer
    v_arr = np.array(ring_values, dtype=np.float64)
    r_arr = np.array(ring_radii, dtype=np.float64)
    A = np.column_stack([np.ones_like(v_arr), (10.0 - v_arr)])
    coef, *_ = np.linalg.lstsq(A, r_arr, rcond=None)
    R10_px, delta_fit = float(coef[0]), float(coef[1])

    # Absolutt verdi-forankring: "ytterste detekterte ring = ring 1" feiler når
    # ring 1 er svak/avskåret (da er ytterste egentlig ring 2). Fysisk invariant:
    # innerste poengring (10) ligger ~én ringavstand fra senteret -> R10 ~= delta.
    # Velg heltalls-forskyvning av ringverdiene som bringer R10 nærmest delta.
    if cfg.get('ring_anchor_R10_eq_delta', True) and delta_fit > 1e-6:
        ratio = cfg.get('ring_R10_over_delta', 1.0)
        off = int(round(ratio - R10_px / delta_fit))
        if off != 0:
            ring_values = [v + off for v in ring_values]
            ring_radii = list(ring_radii)
            ring_harm = [(v + off, b, rm, cv) for (v, b, rm, cv) in ring_harm]
            R10_px = R10_px + off * delta_fit
            log(f"verdi-forankring: forskjøv ringverdier med {off:+d} (R10~=delta)")

    log(f"Ringer (verdi: radius): " +
        ", ".join(f"{v}: {r:.1f}" for v, r in zip(ring_values, ring_radii)))
    log(f"R10={R10_px:.2f}px, delta={delta_fit:.2f}px, senter=({center[0]:.2f}, {center[1]:.2f})")

    return {
        'center': center,
        'delta_px': delta_fit,
        'R10_px': R10_px,
        'ring_values': ring_values,
        'ring_radii_px': ring_radii,
        'ring_harmonics': ring_harm,   # beta er skalert til originalpx
        'r_max': r_max,
        'rho_scale': rho_scale,
    }


def validate_calibration(calib, cfg):
    """
    Avgjør om kalibreringen beskriver en ekte poengskive, eller om den
    har "funnet" ringer i støy/feil struktur. Brukes til å forkaste
    bilder uten poengringer i stedet for å rapportere tull.

    Returns:
        (ok, reason): ok=True hvis kalibreringen er troverdig,
        ellers False med begrunnelse.
    """
    reasons = []

    # 1) Ringavstanden må være stor nok til at markører kan detekteres.
    #    Ekte skivefoto gir 60-115px; støy-"ringer" gir typisk < 20px.
    min_delta = cfg.get('gate_min_delta_px', 25.0)
    delta = calib['delta_px']
    if delta < min_delta:
        reasons.append(f"ringavstand {delta:.1f}px er for liten (min {min_delta:.0f}px)")

    # 2) Nok ringer må være funnet og tilpasset.
    min_rings = cfg.get('gate_min_rings', 5)
    n_rings = len(calib['ring_harmonics'])
    if n_rings < min_rings:
        reasons.append(f"bare {n_rings} ringer funnet (min {min_rings})")

    # 3) Ringene må ligge på en jevn progresjon (ekte skiver er ekvidistante).
    if delta > 0 and len(calib['ring_radii_px']) >= 2:
        vs = np.asarray(calib['ring_values'], dtype=np.float64)
        rs = np.asarray(calib['ring_radii_px'], dtype=np.float64)
        pred = calib['R10_px'] + (10.0 - vs) * delta
        max_resid_frac = float(np.max(np.abs(rs - pred))) / delta
        gate_resid = cfg.get('gate_max_resid_frac', 0.12)
        if max_resid_frac > gate_resid:
            reasons.append(f"ringene er ikke jevnt fordelt "
                           f"(maks avvik {max_resid_frac:.2f} av ringavstanden)")

    # 4) Geometrien må være fysisk mulig.
    if calib['R10_px'] <= 0:
        reasons.append(f"R10 = {calib['R10_px']:.1f}px er ikke fysisk mulig")

    return (len(reasons) == 0, "; ".join(reasons))


def local_ring_geometry(calib, theta):
    """
    Evaluer R10 og delta ved en gitt vinkel theta (radianer, bildekonvensjon)
    ved å regressere harmonisk ringradius mot ringverdi.
    Fanger opp perspektiv (ellipse) til første orden.

    Returns:
        (R10_theta, delta_theta)
    """
    vs, rs = [], []
    basis = np.array([1.0, np.sin(theta), np.cos(theta),
                      np.sin(2 * theta), np.cos(2 * theta)])
    for (value, beta, rmse, cov) in calib['ring_harmonics']:
        vs.append(value)
        rs.append(float(np.dot(beta, basis)))
    vs = np.array(vs, dtype=np.float64)
    rs = np.array(rs, dtype=np.float64)
    if len(vs) < 2:
        return calib['R10_px'], calib['delta_px']
    A = np.column_stack([np.ones_like(vs), (10.0 - vs)])
    coef, *_ = np.linalg.lstsq(A, rs, rcond=None)
    return float(coef[0]), float(coef[1])
