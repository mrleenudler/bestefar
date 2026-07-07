"""
Kjør hele analysen på alle bildene i en mappe (default: Testsett/) og
lagre annoterte bilder i Visualiseringer/<mappe>_out/. Ingen fasit -
kun for å se hvordan pipelinen oppfører seg på ekte Kongsberg-foto.
"""

import sys
import time
import traceback
from pathlib import Path

import cv2

from config import DEFAULT_CONFIG
from Bestefar import analyze_target, visualize_analysis

IN_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Testsett")
OUT_DIR = DEFAULT_CONFIG['visualization_dir'] / f"{IN_DIR.name}_out"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in IN_DIR.iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    summary = []
    n_scored = 0
    for path in paths:
        name = path.stem
        img = cv2.imread(str(path))
        if img is None:
            summary.append(f"{name}: kunne ikke lese bildet")
            continue
        cfg = DEFAULT_CONFIG.copy()
        t0 = time.time()
        try:
            a = analyze_target(img, cfg, filename=name)
        except Exception as e:
            dt = time.time() - t0
            msg = f"{name}: FORKASTET - {e}  [{dt:.1f}s]"
            summary.append(msg)
            print(msg)
            if not isinstance(e, ValueError):
                traceback.print_exc()
            # Lagre originalen med banner, så alle bildene har output
            vis = img.copy()
            cv2.putText(vis, "FORKASTET", (40, 90), cfg['font'], 2.0, (0, 0, 255), 5)
            cv2.imwrite(str(OUT_DIR / f"{name}.png"), vis)
            continue

        dt = time.time() - t0
        n_scored += 1
        cox, coy = a['center_orig']
        scores = ", ".join(f"{r['decimal']:.1f}" for r in a['results'])
        msg = (f"{name}: {len(a['results'])} treff, sum {a['sum_integer']} "
               f"({a['sum_decimal']:.1f}), delta={a['calib']['delta_px']:.1f}px, "
               f"rekt={'ja' if a['H'] is not None else 'nei'}  [{dt:.1f}s]")
        summary.append(msg)
        print(msg)
        print(f"    poeng: {scores}")
        vis = visualize_analysis(img, a, cfg)
        cv2.imwrite(str(OUT_DIR / f"{name}.png"), vis)

    print("\n========== OPPSUMMERING ==========")
    for line in summary:
        print(line)
    print(f"\nScoret {n_scored}/{len(paths)} bilder. Output i {OUT_DIR}")


if __name__ == '__main__':
    main()
