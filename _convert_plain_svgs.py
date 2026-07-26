"""Porter plain SVG-er (viewBox + currentColor) til Android vector drawables.

Runde 4 (musingsUI): stillingsikoner og hjort-silhuetter. Path-data kopieres
verbatim; viewBox-offset foldes inn i en <group translate>, og fyll settes til
svart for runtime-tinting.
"""
import re
import os

OUT = "android/app/src/main/res/drawable/"


def convert(src, dst, dp_h):
    svg = open(src, encoding="utf-8").read()
    vb = re.search(r'viewBox="([\d.\- ]+)"', svg).group(1).split()
    minx, miny, w, h = (float(x) for x in vb)
    paths = re.findall(r'<path d="([^"]+)"', svg)
    dp_w = round(dp_h * w / h)
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f"<!-- Portert fra {os.path.basename(src)} (musingsUI runde 4) -->",
        '<vector xmlns:android="http://schemas.android.com/apk/res/android"',
        f'    android:width="{dp_w}dp" android:height="{dp_h}dp"',
        f'    android:viewportWidth="{w:.2f}" android:viewportHeight="{h:.2f}">',
        f'    <group android:translateX="{-minx:.2f}" android:translateY="{-miny:.2f}">',
    ]
    for d in paths:
        parts.append(
            f'        <path android:fillColor="#FF000000" android:pathData="{d}"/>')
    parts.append("    </group>")
    parts.append("</vector>")
    open(dst, "w", encoding="utf-8").write("\n".join(parts) + "\n")
    print(f"{os.path.basename(dst)}: {w:.0f}x{h:.0f} -> {dp_w}x{dp_h}dp, "
          f"{len(paths)} path(s)")


# Stillingsikoner (brede) -> 40dp høye
for pos in ["liggende", "sittende", "knestaaende", "staaende"]:
    convert(f"UI/skytestilling-{pos}.svg",
            OUT + f"ic_stilling_{pos}.xml", 40)

# Hjort-silhuetter (høye) -> 64dp høye, for jaktloggens posisjonsvalg
for view in ["front", "side"]:
    convert(f"UI/hjort-{view}.svg", OUT + f"ic_hjort_{view}.xml", 64)
convert("UI/hjort-skrå.svg", OUT + "ic_hjort_skraa.xml", 64)
