"""Porter UI/*.svg (potrace-stil) til Android vector drawables.

Runde 3 (musingsUI): beregner tett bounding-boks fra path-dataene og setter
viewporten til den, slik at ikonene fyller knappene i stedet for aa tegnes
smaatt i et stort lerret.
"""
import re

TOKEN = re.compile(r"([MmCcLlVvHhZz])|(-?\d+\.?\d*)")
NEED = {"M": 2, "m": 2, "L": 2, "l": 2, "C": 6, "c": 6,
        "H": 1, "h": 1, "V": 1, "v": 1}


def path_points(d, pts):
    """Samler alle punkter (inkl. kontrollpunkter) i en path."""
    cmd = None
    cx = cy = 0.0
    args = []
    for c, num in TOKEN.findall(d):
        if c:
            cmd = None if c in "Zz" else c
            args = []
            continue
        args.append(float(num))
        if cmd is None or len(args) < NEED[cmd]:
            continue
        chunk, args = args, []
        if cmd in "MmLl":
            dx, dy = chunk
            if cmd.islower():
                cx, cy = cx + dx, cy + dy
            else:
                cx, cy = dx, dy
            pts.append((cx, cy))
            # Etter M/m er paafoelgende koordinatpar implisitte L/l
            if cmd == "M":
                cmd = "L"
            elif cmd == "m":
                cmd = "l"
        elif cmd in "Cc":
            for i in range(0, 6, 2):
                px, py = chunk[i], chunk[i + 1]
                if cmd.islower():
                    px, py = cx + px, cy + py
                pts.append((px, py))
            cx, cy = pts[-1]
        elif cmd in "Hh":
            cx = cx + chunk[0] if cmd.islower() else chunk[0]
            pts.append((cx, cy))
        elif cmd in "Vv":
            cy = cy + chunk[0] if cmd.islower() else chunk[0]
            pts.append((cx, cy))


def convert(src, dst, scale, tx, ty, dp_h=24):
    svg = open(src, encoding="utf-8").read()
    paths = re.findall(r'<path d="([^"]+)"', svg)
    pts = []
    for d in paths:
        path_points(" ".join(d.split()), pts)
    # Transformer til endelige koordinater (y flippes av potrace-transformen)
    fpts = [(scale * x + tx, -scale * y + ty) for x, y in pts]
    xs = [p[0] for p in fpts]
    ys = [p[1] for p in fpts]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    m = 0.02 * max(maxx - minx, maxy - miny)   # 2 % margin
    minx -= m; miny -= m; maxx += m; maxy += m
    w, h = maxx - minx, maxy - miny
    dp_w = round(dp_h * w / h)

    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f"<!-- Portert fra {src}; viewport = tett bboks (musingsUI runde 3) -->",
        '<vector xmlns:android="http://schemas.android.com/apk/res/android"',
        f'    android:width="{dp_w}dp" android:height="{dp_h}dp"',
        f'    android:viewportWidth="{w:.1f}" android:viewportHeight="{h:.1f}">',
        f'    <group android:scaleX="{scale:.6f}" android:scaleY="{-scale:.6f}"'
        f' android:translateX="{tx - minx:.2f}" android:translateY="{ty - miny:.2f}">',
    ]
    for d in paths:
        d = " ".join(d.split())
        parts.append(
            f'        <path android:fillColor="#FF000000" android:pathData="{d}"/>')
    parts.append("    </group>")
    parts.append("</vector>")
    open(dst, "w", encoding="utf-8").write("\n".join(parts) + "\n")
    print(f"{dst}: bbox {w:.0f}x{h:.0f}")


OUT = "android/app/src/main/res/drawable/"

# distance: translate(30.80,96.49) scale(0.331777); indre translate(0,624)
s = 0.331777
convert("UI/menu_icon_distance.svg", OUT + "ic_menu_distance.xml",
        0.1 * s, 30.80, 96.49 + s * 624)

# position: translate(30.80,71.80) scale(0.424502); indre translate(0,604)
s = 0.424502
convert("UI/menu_icon_position.svg", OUT + "ic_menu_position.xml",
        0.1 * s, 30.80, 71.80 + s * 604)

# rifle: scale(0.25); indre translate(0,4000)
convert("UI/menu_icon_rifle.svg", OUT + "ic_menu_rifle.xml",
        0.025, 0.0, 0.25 * 4000)

# moose (ny SVG): translate(186.20,50.00) scale(0.790861); indre translate(0,1138)
s = 0.790861
convert("UI/menu_icon_moose.svg", OUT + "ic_menu_moose.xml",
        0.1 * s, 186.20, 50.0 + s * 1138)
