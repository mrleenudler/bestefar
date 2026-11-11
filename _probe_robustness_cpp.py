"""
Robusthetsprobe for C++-kjernen: syntetiske perturbasjoner av C-settet
(perspektiv/vinkel, rotasjon, skala, gjenskinn, eksponering), skaaret mot
samme fasit som verify_cset_cpp.py.

Ingen nye bilder finnes (2026-11): dette er den eneste maaten aa maale
vinkel-/gjenskinnsrobusthet paa. En perturbasjon endrer IKKE fasiten
(antall+sum er invariant under geometri/fotometri saa lenge skiva er lesbar).

Kjoer:  .venv\\Scripts\\python.exe _probe_robustness_cpp.py [filter]
  filter: valgfri delstreng som velger perturbasjoner (f.eks. "glare")
Skriver perturberte bilder til en temp-katalog og en PASS/FAIL-matrise til stdout.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

CLI = Path("core/build/bestefar_cli.exe")
IN = Path("Testsett")
TRUTH = Path("hits_truth.txt")


def load_truth(path):
    truth = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\S+)\s+(\d+)\s+(\d+)\s+(.*)", line)
        if not m:
            continue
        truth[m.group(1)] = dict(count=int(m.group(2)), sum=int(m.group(3)))
    return truth


# ---- perturbasjoner -------------------------------------------------------

def persp_tilt(img, ax, deg):
    """Projektiv vipping om aksen ax ('x': kamera over/under, 'y': fra siden).
    Modell: roter bildeplanet deg grader og projiser tilbake (f = bildebredde)."""
    h, w = img.shape[:2]
    f = float(max(h, w))
    t = np.deg2rad(deg)
    # 3D-hjoerner sentrert, roter, projiser med brennvidde f
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]],
                   np.float64)
    out = []
    for x, y in pts:
        if ax == 'y':
            xr = x * np.cos(t)
            z = f + x * np.sin(t)
            yr = y
        else:
            yr = y * np.cos(t)
            z = f + y * np.sin(t)
            xr = x
        out.append([xr * f / z + w / 2, yr * f / z + h / 2])
    M = cv2.getPerspectiveTransform(
        np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32),
        np.array(out, np.float32))
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def rotate(img, deg):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def scale_down(img, f):
    """Skiva mindre i rammen: nedskaler og legg paa replikert kant (samme str)."""
    h, w = img.shape[:2]
    small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    top = (h - small.shape[0]) // 2
    left = (w - small.shape[1]) // 2
    return cv2.copyMakeBorder(small, top, h - small.shape[0] - top,
                              left, w - small.shape[1] - left,
                              cv2.BORDER_REPLICATE)


def glare(img, dx_frac, dy_frac, r_frac, strength):
    """Myk speilrefleks: gaussisk lys flekk, posisjon relativ til bildesenter."""
    h, w = img.shape[:2]
    cx = w / 2 + dx_frac * w
    cy = h / 2 + dy_frac * h
    r = r_frac * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))
    out = img.astype(np.float32) + (strength * 255.0) * g[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def gradient_light(img, strength):
    """Lineaer lysgradient over bildet (vindu paa siden)."""
    h, w = img.shape[:2]
    ramp = np.linspace(-strength, strength, w, dtype=np.float32) * 255.0
    out = img.astype(np.float32) + ramp[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def gamma(img, g):
    lut = (np.power(np.arange(256) / 255.0, g) * 255.0).astype(np.uint8)
    return lut[img]


def defocus(img, sigma):
    return cv2.GaussianBlur(img, (0, 0), sigma)


PERTURBATIONS = [
    ("tilt_y+10", lambda im: persp_tilt(im, 'y', 10)),
    ("tilt_y-10", lambda im: persp_tilt(im, 'y', -10)),
    ("tilt_y+20", lambda im: persp_tilt(im, 'y', 20)),
    ("tilt_x+10", lambda im: persp_tilt(im, 'x', 10)),
    ("tilt_x+20", lambda im: persp_tilt(im, 'x', 20)),
    ("rot+5",     lambda im: rotate(im, 5)),
    ("rot-15",    lambda im: rotate(im, -15)),
    ("scale0.7",  lambda im: scale_down(im, 0.7)),
    ("scale0.5",  lambda im: scale_down(im, 0.5)),
    ("glare_hi",  lambda im: glare(im, 0.10, -0.10, 0.10, 0.9)),
    ("glare_lo",  lambda im: glare(im, -0.12, 0.08, 0.14, 0.5)),
    ("gradlight", lambda im: gradient_light(im, 0.25)),
    ("gamma1.6",  lambda im: gamma(im, 1.6)),
    ("gamma0.6",  lambda im: gamma(im, 0.6)),
    ("defocus3",  lambda im: defocus(im, 3.0)),
]


def main():
    if not CLI.exists():
        print("Mangler", CLI); return 1
    filt = sys.argv[1] if len(sys.argv) > 1 else ""
    perts = [(n, f) for n, f in PERTURBATIONS if filt in n]
    truth = load_truth(TRUTH)
    names = sorted(truth, key=lambda s: int(re.sub(r"\D", "", s)))
    tmp = Path(tempfile.mkdtemp(prefix="bf_robust_"))
    print("temp:", tmp)

    col_w = 11
    print("".ljust(6) + "".join(n[:col_w].ljust(col_w) for n, _ in perts))
    totals = {n: 0 for n, _ in perts}
    for name in names:
        img = cv2.imread(str(IN / f"{name}.jpg"))
        row = []
        for pname, fn in perts:
            p = tmp / f"{name}_{pname}.jpg"
            cv2.imwrite(str(p), fn(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
            proc = subprocess.run([str(CLI), str(p)], capture_output=True,
                                  text=True, timeout=600)
            try:
                out = json.loads(proc.stdout.strip())
            except json.JSONDecodeError:
                row.append("JSONERR"); continue
            if out["status"] != "OK":
                row.append("REJ")
                continue
            cnt = len(out["hits"]); s = out["sum_integer"]
            ok = cnt == truth[name]["count"] and s == truth[name]["sum"]
            totals[pname] += ok
            row.append("PASS" if ok else f"{cnt}/{s}")
        print(name.ljust(6) + "".join(c.ljust(col_w) for c in row))
    print("SUM".ljust(6) + "".join(f"{totals[n]}/10".ljust(col_w) for n, _ in perts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
