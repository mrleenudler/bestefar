"""Engangs: porter UI/*.svg (potrace-stil) til Android vector drawables."""
import re


def convert(src, dst, vw, vh, sx, sy, tx, ty, size_w, size_h):
    svg = open(src, encoding="utf-8").read()
    paths = re.findall(r'<path d="([^"]+)"', svg)
    parts = [
        '<?xml version="1.0" encoding="utf-8"?>',
        f"<!-- Portert fra {src} (potrace-transformene foldet inn i gruppa) -->",
        '<vector xmlns:android="http://schemas.android.com/apk/res/android"',
        f'    android:width="{size_w}dp" android:height="{size_h}dp"',
        f'    android:viewportWidth="{vw}" android:viewportHeight="{vh}">',
        f'    <group android:scaleX="{sx}" android:scaleY="{sy}"'
        f' android:translateX="{tx}" android:translateY="{ty}">',
    ]
    for d in paths:
        d = " ".join(d.split())
        parts.append(
            f'        <path android:fillColor="#FF000000" android:pathData="{d}"/>')
    parts.append("    </group>")
    parts.append("</vector>")
    open(dst, "w", encoding="utf-8").write("\n".join(parts) + "\n")
    print(dst, len(paths), "paths")


# distance: ytre translate(30.80,96.49) scale(0.331777); indre translate(0,624) scale(0.1,-0.1)
s = 0.331777
convert("UI/menu_icon_distance.svg",
        "android/app/src/main/res/drawable/ic_menu_distance.xml",
        616, 400, round(0.1 * s, 6), round(-0.1 * s, 6),
        30.80, round(96.49 + s * 624, 2), 37, 24)

# position: ytre translate(30.80,71.80) scale(0.424502); indre translate(0,604) scale(0.1,-0.1)
s = 0.424502
convert("UI/menu_icon_position.svg",
        "android/app/src/main/res/drawable/ic_menu_position.xml",
        616, 400, round(0.1 * s, 6), round(-0.1 * s, 6),
        30.80, round(71.80 + s * 604, 2), 37, 24)
