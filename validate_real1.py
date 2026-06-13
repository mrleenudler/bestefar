"""
Valideringsskript for Real 1.jpg mot fasit fra skjermen.

Fasit (fra displayet): 9.1, 9.4, X.2, *.6, 7.5, 8.8, 9.6, 9.1, 8.9, 9.4
(X=10, *=indre tier). S-10 = 88.

Treffkoordinater (håndmålt, fra 'todo bestefar.txt') koblet til verdi
via avstand fra fasitsenter (1094, 1937).
"""

import sys
import cv2
import numpy as np

from config import DEFAULT_CONFIG
import rings
import scoring
import hits as hits_mod

GT_CENTER = (1094.0, 1937.0)

# (x, y, forventet desimalpoeng)
GT_HITS = [
    (1089, 1983, 10.6),
    (1079, 2019, 10.2),
    (984, 1837, 9.6),
    (1251, 1991, 9.4),
    (993, 2074, 9.4),
    (935, 1823, 9.1),
    (897, 1942, 9.1),
    (881, 1879, 8.9),
    (940, 1774, 8.8),
    (1443, 1962, 7.5),
]


def main():
    cfg = DEFAULT_CONFIG.copy()
    img = cv2.imread('Real 1.jpg')

    from Bestefar import analyze_target

    # Startgjetning: bruk fasitsenter forskjøvet, for å teste robusthet,
    # eller send inn '--pipeline' for å bruke hele deteksjonspipelinen.
    if '--pipeline' in sys.argv:
        analysis = analyze_target(img, cfg, filename='Real 1')
    else:
        center0 = (GT_CENTER[0] + 15, GT_CENTER[1] - 12)  # bevisst feil startpunkt
        print(f"Startsenter (forskjøvet fasit): {center0}")
        analysis = analyze_target(img, cfg, filename='Real 1',
                                  center0=center0, r_est=980.0)

    print('\n'.join(analysis['debug_lines']))
    calib = analysis['calib']
    H = analysis['H']

    cx, cy = analysis['center_orig']
    err = (cx - GT_CENTER[0], cy - GT_CENTER[1])
    print(f"\nSenter (original): ({cx:.2f}, {cy:.2f})  fasit: {GT_CENTER}  "
          f"avvik: ({err[0]:+.2f}, {err[1]:+.2f}) px")
    print(f"delta_px={calib['delta_px']:.2f}, R10_px={calib['R10_px']:.2f}, "
          f"rektifisert={'ja' if H is not None else 'nei'}")

    # --- Poeng for fasitkoordinater (uavhengig av treffdeteksjon) ---
    import perspektiv
    gt_xy = [(g[0], g[1]) for g in GT_HITS]
    gt_frame = perspektiv.transform_points(gt_xy, H) if H is not None else np.array(gt_xy, dtype=float)
    print(f"\n{'treff':>14} {'d_px':>7} {'poeng':>6} {'fasit':>6} {'avvik':>6}")
    n_ok = 0
    for (g, p) in zip(GT_HITS, gt_frame):
        res = scoring.score_hit((p[0], p[1]), calib, cfg)
        diff = res['decimal'] - g[2]
        ok = abs(diff) <= 0.1 + 1e-6
        n_ok += ok
        print(f"({g[0]:4d},{g[1]:4d}) {res['distance']:7.1f} {res['decimal']:6.1f} "
              f"{g[2]:6.1f} {diff:+6.2f} {'OK' if ok else 'FEIL'}")
    print(f"\n{n_ok}/{len(GT_HITS)} innenfor +-0.1 (fasitkoordinater)")

    # --- Detekterte treff: match mot fasit (naermeste, maks 0.25*delta) ---
    detected = analysis['results']
    tol = 0.25 * calib['delta_px']
    unmatched_gt = list(GT_HITS)
    print(f"\n{'detektert':>16} {'naermeste fasit':>16} {'avst':>6} "
          f"{'poeng':>6} {'fasit':>6}")
    n_match = 0
    n_score_ok = 0
    for res in detected:
        best, bd = None, 1e9
        for g in unmatched_gt:
            dd = np.hypot(res['x_orig'] - g[0], res['y_orig'] - g[1])
            if dd < bd:
                best, bd = g, dd
        if best is not None and bd <= tol:
            unmatched_gt.remove(best)
            n_match += 1
            sok = abs(res['decimal'] - best[2]) <= 0.1 + 1e-6
            n_score_ok += sok
            print(f"({res['x_orig']:6.1f},{res['y_orig']:6.1f}) ({best[0]:4d},{best[1]:4d}) "
                  f"{bd:6.1f} {res['decimal']:6.1f} {best[2]:6.1f} "
                  f"{'OK' if sok else 'FEIL'}")
        else:
            print(f"({res['x_orig']:6.1f},{res['y_orig']:6.1f}) {'FALSK POSITIV':>16} "
                  f"{bd:6.1f} {res['decimal']:6.1f}")
    for g in unmatched_gt:
        print(f"{'IKKE FUNNET':>16} ({g[0]:4d},{g[1]:4d})          {g[2]:6.1f}")

    print(f"\nMatchet {n_match}/{len(GT_HITS)}, poeng OK {n_score_ok}/{len(GT_HITS)}")
    print(f"Sum desimal: {analysis['sum_decimal']} (fasit 92.6)   "
          f"Sum heltall: {analysis['sum_integer']} (fasit 88)")
    ok = (n_match == len(GT_HITS) and n_score_ok == len(GT_HITS)
          and len(detected) == len(GT_HITS))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
