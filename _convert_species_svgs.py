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
    # Rein (musingsUI runde 10) - foerste art med EGEN skraa-silhuett.
    ("UI/rein_silhuett_side.svg", OUT + "ic_rein_side.xml"),
    ("UI/rein_silhuett_front.svg", OUT + "ic_rein_front.xml"),
    ("UI/rein_silhuett_skrå.svg", OUT + "ic_rein_skraa.xml"),
    # Raadyr (musingsUI runde 12). Sidevisningen laa foerst inne som
    # "..._side_utilfredsstillende.svg"; eieren erstattet den under runden.
    ("UI/rådyr_silhuett_front.svg", OUT + "ic_raadyr_front.xml"),
    ("UI/rådyr_silhuett_skrå.svg", OUT + "ic_raadyr_skraa.xml"),
    ("UI/rådyr_silhuett_side.svg", OUT + "ic_raadyr_side.xml"),
]


TOKEN = re.compile(r"([MmCcLlVvHhZz])|(-?\d*\.?\d+(?:[eE][-+]?\d+)?)")
NEED = {"M": 2, "m": 2, "L": 2, "l": 2, "C": 6, "c": 6,
        "H": 1, "h": 1, "V": 1, "v": 1}


def _points(d):
    """Alle punkter (inkl. kontrollpunkter) i en path — nok til en bboks."""
    pts, cmd, cx, cy, args = [], None, 0.0, 0.0, []
    sx = sy = 0.0
    for c, num in TOKEN.findall(d):
        if c:
            if c in "Zz":
                cx, cy = sx, sy
                cmd = None
            else:
                cmd = c
            args = []
            continue
        if cmd is None:
            continue
        args.append(float(num))
        if len(args) < NEED[cmd]:
            continue
        chunk, args = args, []
        if cmd in "MmLl":
            dx, dy = chunk
            cx, cy = (cx + dx, cy + dy) if cmd.islower() else (dx, dy)
            if cmd in "Mm":
                sx, sy = cx, cy
                cmd = "L" if cmd == "M" else "l"   # implisitte linjer etter M
            pts.append((cx, cy))
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
    return pts


def _bbox(d):
    pts = _points(d)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _covers_canvas(d, minx, miny, w, h):
    """Sant for potrace sitt hvite bakgrunnsrektangel (dekker hele lerretet)."""
    x0, y0, x1, y1 = _bbox(d)
    return (x1 - x0) >= 0.98 * w and (y1 - y0) >= 0.98 * h


def convert(src, dst):
    svg = open(src, encoding="utf-8").read()
    vb = re.search(r'viewBox="([^"]+)"', svg).group(1).split()
    minx, miny, w, h = (float(x) for x in vb)
    tags = re.findall(r"<path\b[^>]*>", svg, re.S)
    paths = [re.search(r'\bd="([^"]+)"', t).group(1) for t in tags]
    whites = ['fill="white"' in t for t in tags]
    fill_type = ""

    if any(whites):
        # POTRACE-VARIANT (raadyr side, musingsUI runde 12): eksporten maler et
        # hvitt bakgrunnsrektangel og tegner HULL som hvite paths oppaa. Vi
        # tinter drawablen i UI-koden, saa «hvit» finnes ikke som farge her -
        # males alt svart, blir ikonet en solid klump.
        # Bakgrunnsrektangelet kastes, og resten slaas sammen til EN path med
        # evenOdd, som er nettopp hull-semantikken potrace koder med de hvite
        # konturene.
        keep = [d for d, white in zip(paths, whites)
                if not (white and _covers_canvas(d, minx, miny, w, h))]
        paths = [" ".join(" ".join(d.split()) for d in keep)]
        fill_type = ' android:fillType="evenOdd"'
        bx0, by0, bx1, by1 = _bbox(paths[0])
        minx, miny, w, h = bx0, by0, bx1 - bx0, by1 - by0

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
        lines.append(f'        <path android:fillColor="#FF000000"'
                     f'{fill_type} android:pathData="{d}"/>')
    lines.append("    </group>")
    lines.append("</vector>")
    open(dst, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"{dst}: viewport {w:.0f}x{h:.0f}, {dp_w}x{dp_h}dp, {len(paths)} path(s)")


for src, dst in JOBS:
    convert(src, dst)
