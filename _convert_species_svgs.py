"""Porter plain-SVG viltsilhuetter (Elg/Villsvin, side+front) til Android vector
drawables i samme format som ic_hjort_side.xml (musingsUI runde 9).

Plain SVG: <svg viewBox="minx miny w h" ...><path d="..."/>. Drawable:
viewport = (w, h), group translate = (-minx, -miny), fyllfarge #FF000000
(tintes i UI-koden). dp-hoeyde 64, bredde proporsjonal.
"""
import re

OUT = "android/app/src/main/res/drawable/"

JOBS = [
    ("UI/Elg_silhuett_side.svg", OUT + "ic_elg_side.xml"),
    ("UI/Elg_silhuett_front.svg", OUT + "ic_elg_front.xml"),
    ("UI/Villsvin_silhuett_side.svg", OUT + "ic_villsvin_side.xml"),
    ("UI/Villsvin_silhuett_front.svg", OUT + "ic_villsvin_front.xml"),
]


def convert(src, dst):
    svg = open(src, encoding="utf-8").read()
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1).split()
    minx, miny, w, h = (float(x) for x in vb)
    paths = re.findall(r'<path[^>]*\bd="([^"]+)"', svg, re.S)
    dp_h = 64
    dp_w = round(dp_h * w / h)
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f"<!-- Portert fra {src} (musingsUI runde 9) -->",
        '<vector xmlns:android="http://schemas.android.com/apk/res/android"',
        f'    android:width="{dp_w}dp" android:height="{dp_h}dp"',
        f'    android:viewportWidth="{w}" android:viewportHeight="{h}">',
        f'    <group android:translateX="{-minx}" android:translateY="{-miny}">',
    ]
    for d in paths:
        d = " ".join(d.split())
        lines.append(
            f'        <path android:fillColor="#FF000000" android:pathData="{d}"/>')
    lines.append("    </group>")
    lines.append("</vector>")
    open(dst, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{dst}: viewport {w:.0f}x{h:.0f}, {dp_w}x{dp_h}dp, {len(paths)} path(s)")


for src, dst in JOBS:
    convert(src, dst)
