"""
Score C-settet (C1-C11) ende-til-ende med hovedpipelinen (analyze_target), og
lagre annoterte bilder slik at våre poeng kan sammenlignes mot panelet i bildet.
"""
import re
import traceback
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_CONFIG
from Bestefar import analyze_target, visualize_analysis

IN = Path("Testsett")
OUT = DEFAULT_CONFIG['visualization_dir'] / "Cset_out"


def main():
    import shutil
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted([p.stem for p in IN.glob("C*.jpg") if re.fullmatch(r"C\d+", p.stem)],
                   key=lambda s: int(s[1:]))
    summary = []
    for name in names:
        img = cv2.imread(str(IN / f"{name}.jpg"))
        if img is None:
            summary.append(f"{name}: mangler"); continue
        cfg = DEFAULT_CONFIG.copy()
        cfg['analyze_screen_fallback'] = False  # krev gyldig skjermutklipp (ingen fallback til original)
        try:
            a = analyze_target(img, cfg, filename=name)
        except ValueError as e:
            summary.append(f"{name}: FORKASTET - {e}")
            continue
        except Exception as e:
            summary.append(f"{name}: FEIL - {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        scores = ", ".join(f"{r['decimal']:.1f}" for r in a['results'])
        summary.append(f"{name}: {len(a['results'])} treff, sum {a['sum_integer']} "
                       f"({a['sum_decimal']:.1f}), delta={a['calib']['delta_px']:.1f} :: {scores}")
        vis = visualize_analysis(img, a, cfg)
        cv2.imwrite(str(OUT / f"{name}.png"), vis)

    print("\n========== C-SETT SCORING ==========")
    for line in summary:
        print(line)
    print(f"\nOutput i {OUT}")


if __name__ == '__main__':
    main()
