"""
Kjør skjermdeteksjon + perspektivretting + beskjæring (screen.py) på alle
T-bildene, og forsøk scoring på det rettede skjermbildet.

Output i Visualiseringer/Tscreen_out/:
  T#_detect.png : original med detektert skjermfirkant tegnet på
  T#_rect.png   : perspektivkorrigert, beskåret skjerm
  T#_score.png  : forsøkt scoring på rettet skjerm (eller FORKASTET-banner)
Pluss montasjer for rask inspeksjon.
"""

import re
import shutil
import traceback
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_CONFIG
from Bestefar import analyze_target, visualize_analysis
import screen

IN = Path("Testsett")
OUT = DEFAULT_CONFIG['visualization_dir'] / "Tscreen_out"


def fit_tile(img, tile, label):
    h, w = img.shape[:2]
    s = tile / max(h, w)
    t = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    canvas = np.full((tile, tile, 3), 30, np.uint8)
    y0 = (tile - t.shape[0]) // 2; x0 = (tile - t.shape[1]) // 2
    canvas[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
    cv2.putText(canvas, label, (6, 26), 0, 0.7, (0, 255, 255), 2)
    return canvas


def montage(tiles, cols, path):
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 30))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    # Tøm output-mappa helt mellom kjøringer (ingen gamle bilder å forveksle)
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # 2-fargers ringskiver: T etterfulgt av tall (ikke Ta/Tb/... mørke)
    names = sorted([p.stem for p in IN.glob("T*.jpg") if re.fullmatch(r"T\d+", p.stem)],
                   key=lambda s: int(s[1:]))
    summary = []
    rect_tiles = []
    score_tiles = []

    for name in names:
        img = cv2.imread(str(IN / f"{name}.jpg"))
        if img is None:
            summary.append(f"{name}: mangler bilde")
            continue
        cfg = DEFAULT_CONFIG.copy()
        dbg = []
        res = screen.rectify_to_screen(img, cfg, dbg)

        # detection overlay (refined red, rough yellow)
        quad = res[2] if res is not None else None
        rough = res[3] if res is not None else None
        cv2.imwrite(str(OUT / f"{name}_detect.png"), screen.draw_quad(img, quad, rough))

        if res is None:
            summary.append(f"{name}: SKJERM IKKE FUNNET ({'; '.join(dbg)})")
            rect_tiles.append(fit_tile(img, 320, f"{name} ingen skjerm"))
            score_tiles.append(fit_tile(img, 320, f"{name} -"))
            continue

        warped, M, quad, rough = res
        cv2.imwrite(str(OUT / f"{name}_rect.png"), warped)
        rect_tiles.append(fit_tile(warped, 320, name))

        try:
            a = analyze_target(warped, cfg, filename=name)
            vis = visualize_analysis(warped, a, cfg)
            scores = ", ".join(f"{r['decimal']:.1f}" for r in a['results'])
            summary.append(f"{name}: skjerm OK -> {len(a['results'])} treff, "
                           f"sum {a['sum_integer']} ({a['sum_decimal']:.1f}) :: {scores}")
            cv2.imwrite(str(OUT / f"{name}_score.png"), vis)
            score_tiles.append(fit_tile(vis, 320, f"{name} sum{a['sum_integer']}"))
        except ValueError as e:
            msg = str(e).replace("Bilde forkastet: ", "")
            summary.append(f"{name}: skjerm OK, scoring forkastet - {msg}")
            vis = warped.copy()
            cv2.putText(vis, "FORKASTET", (30, 80), 0, 2.0, (0, 0, 255), 5)
            cv2.imwrite(str(OUT / f"{name}_score.png"), vis)
            score_tiles.append(fit_tile(vis, 320, f"{name} forkastet"))
        except Exception as e:
            summary.append(f"{name}: SCORING-FEIL - {type(e).__name__}: {e}")
            traceback.print_exc()
            score_tiles.append(fit_tile(warped, 320, f"{name} FEIL"))

    montage(rect_tiles, 5, OUT / "_montage_rect.png")
    montage(score_tiles, 5, OUT / "_montage_score.png")

    print("\n========== OPPSUMMERING (skjerm-rett + scoring) ==========")
    for line in summary:
        print(line)
    print(f"\nOutput i {OUT}")


if __name__ == '__main__':
    main()
