"""
Kjør hele analysen på Test2-Test14 og lagre annoterte bilder i Visualiseringer.
Ingen fasit - kun for å se hvordan pipelinen oppfører seg på varierte bilder.
"""

import traceback
import cv2

from config import DEFAULT_CONFIG
from Bestefar import analyze_target, visualize_analysis

OUT_DIR = DEFAULT_CONFIG['visualization_dir']


def main():
    summary = []
    for n in range(2, 15):
        name = f"Test{n}"
        path = f"{name}.jpg"
        img = cv2.imread(path)
        if img is None:
            summary.append(f"{name}: kunne ikke lese bildet")
            continue
        cfg = DEFAULT_CONFIG.copy()
        print(f"\n=== {name} ({img.shape[1]}x{img.shape[0]}) ===")
        try:
            a = analyze_target(img, cfg, filename=name)
        except Exception as e:
            summary.append(f"{name}: FEILET - {type(e).__name__}: {e}")
            print(f"FEILET: {e}")
            traceback.print_exc()
            # Lagre originalen med feilbanner, så alle Test-bildene har output
            vis = img.copy()
            cv2.putText(vis, f"ANALYSE FEILET: {type(e).__name__}", (40, 80),
                        cfg['font'], 1.5, (0, 0, 255), 4)
            cv2.imwrite(str(OUT_DIR / f"{name}_Treff_og_poeng.png"), vis)
            continue

        calib = a['calib']
        cox, coy = a['center_orig']
        scores = ", ".join(f"{r['decimal']:.1f}" for r in a['results'])
        line = (f"{name}: {len(a['results'])} treff, sum {a['sum_integer']} "
                f"({a['sum_decimal']:.1f}), delta={calib['delta_px']:.1f}px, "
                f"senter=({cox:.0f},{coy:.0f}), "
                f"rektifisert={'ja' if a['H'] is not None else 'nei'}")
        summary.append(line)
        print(line)
        print(f"  poeng: {scores}")

        vis = visualize_analysis(img, a, cfg)
        cv2.imwrite(str(OUT_DIR / f"{name}_Treff_og_poeng.png"), vis)

    print("\n========== OPPSUMMERING ==========")
    for line in summary:
        print(line)


if __name__ == '__main__':
    main()
