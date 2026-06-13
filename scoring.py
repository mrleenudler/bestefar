"""
Poengberegning for treff basert på ringkalibrering.

Desimalpoeng: s = 10 + (R10 - d) / delta, begrenset til [0, 10.9].
R10 og delta kan evalueres lokalt ved treffets vinkel ('score_use_local_rings')
for å kompensere for perspektiv.
"""

import numpy as np

import rings


def score_hit(xy, calib, cfg):
    """
    Beregn poeng for ett treff.

    Args:
        xy: (x, y) i originalkoordinater
        calib: kalibrering fra rings.calibrate_and_refine
        cfg: config

    Returns:
        dict med 'distance', 'theta', 'decimal', 'integer', 'R10', 'delta'
    """
    cx, cy = calib['center']
    dx = float(xy[0]) - cx
    dy = float(xy[1]) - cy
    d = float(np.hypot(dx, dy))
    theta = float(np.arctan2(dy, dx) % (2.0 * np.pi))

    if cfg.get('score_use_local_rings', True):
        R10, delta = rings.local_ring_geometry(calib, theta)
    else:
        R10, delta = calib['R10_px'], calib['delta_px']

    # Poenggrensen ligger litt utenfor ringstrekens senterlinje:
    # halv kaliber (gauge) + halv strekbredde. Skala-invariant som andel av delta.
    gauge = cfg.get('score_gauge_frac_of_delta', 0.10) * delta

    s = 10.0 + (R10 + gauge - d) / delta
    s = max(0.0, min(10.9, s))

    if cfg.get('score_round_mode', 'truncate') == 'truncate':
        # Offisiell desimalregel: verdien avkortes (rundes ned) til tidel
        decimal = np.floor(s * 10.0 + 1e-9) / 10.0
    else:
        decimal = round(s, 1)
    integer = int(np.floor(decimal))

    return {
        'distance': d,
        'theta': theta,
        'decimal': float(decimal),
        'integer': integer,
        'R10': R10,
        'delta': delta,
    }


def score_hits(hit_list, calib, cfg):
    """
    Beregn poeng for en liste av treff.

    Returns:
        (results, sum_decimal, sum_integer) der results er liste av dicts
        med 'x', 'y' + feltene fra score_hit, sortert synkende på poeng.
    """
    results = []
    for xy in hit_list:
        res = score_hit(xy, calib, cfg)
        res['x'], res['y'] = float(xy[0]), float(xy[1])
        results.append(res)
    results.sort(key=lambda r: -r['decimal'])
    sum_dec = round(sum(r['decimal'] for r in results), 1)
    sum_int = sum(r['integer'] for r in results)
    return results, sum_dec, sum_int
