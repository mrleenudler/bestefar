"""
Verifiser hele C-settet mot panel-fasiten i hits_truth.txt.

Erstatter den manuelle montasje-gjennomgangen: kjoerer hovedpipelinen
(analyze_target) paa hvert C-bilde og sammenligner mot poengene avlest fra
Kongsberg-panelet. "Regresjon?" blir EN kommando i stedet for bildevisning.

Portene (PASS/FAIL) er paa HELTALLSNIVAa - robuste mot smaa kalibreringsavvik:
  * antall treff  == panelets antall
  * sum_integer   == panelets S-10
Desimal-avviket rapporteres som diagnostikk (gjennomsnittlig |avvik| og antall
innenfor 0.15) - det maaler ring-kalibreringens noeyaktighet, men feller ikke
testen alene.

VIKTIG: fasiten brukes BARE til sammenligning her. Den paavirker ikke
deteksjonen - ellers ville vi maskert daarlig deteksjon (musings 2026-06-30).

Bruk:  python verify_cset.py
"""
import re
import sys
import traceback
from pathlib import Path

import cv2

from config import DEFAULT_CONFIG
from Bestefar import analyze_target

IN = Path("Testsett")
TRUTH = Path("hits_truth.txt")
DEC_TOL = 0.15   # desimal regnes "truffet" innenfor dette


def load_truth(path):
    truth = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # navn antall sum  rest(desimaler, komma-separert)
        m = re.match(r"(\S+)\s+(\d+)\s+(\d+)\s+(.*)", line)
        if not m:
            continue
        name, cnt, ssum, decs = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        decimals = sorted(float(x) for x in decs.replace(",", " ").split())
        truth[name] = dict(count=cnt, sum=ssum, decimals=decimals)
    return truth


def pair_decimals(truth_decs, got_decs):
    """Par sorterte fasit- og maalte desimaler 1:1 i sortert rekkefoelge.
    Returner (mean_abs_err, n_within_tol) over min-lengden."""
    t = sorted(truth_decs)
    g = sorted(got_decs)
    n = min(len(t), len(g))
    if n == 0:
        return None, 0
    diffs = [abs(t[i] - g[i]) for i in range(n)]
    within = sum(1 for d in diffs if d <= DEC_TOL)
    return sum(diffs) / n, within


def main():
    if not TRUTH.exists():
        print(f"Mangler {TRUTH}"); return 1
    truth = load_truth(TRUTH)

    names = sorted(truth.keys(), key=lambda s: int(re.sub(r"\D", "", s) or 0))
    rows = []
    n_pass = 0
    for name in names:
        ref = truth[name]
        img = cv2.imread(str(IN / f"{name}.jpg"))
        if img is None:
            rows.append((name, "MANGLER", "", "", "", "")); continue
        cfg = DEFAULT_CONFIG.copy()
        cfg['analyze_screen_fallback'] = False
        try:
            a = analyze_target(img, cfg, filename=name)
        except ValueError as e:
            rows.append((name, "FORKASTET", f"({ref['count']})", str(ref['sum']), "", str(e)[:40]))
            continue
        except Exception as e:
            rows.append((name, f"FEIL {type(e).__name__}", "", "", "", str(e)[:40]))
            traceback.print_exc()
            continue

        got_cnt = len(a['results'])
        got_sum = a['sum_integer']
        got_decs = [r['decimal'] for r in a['results']]
        cnt_ok = got_cnt == ref['count']
        sum_ok = got_sum == ref['sum']
        mae, within = pair_decimals(ref['decimals'], got_decs)
        ok = cnt_ok and sum_ok
        n_pass += ok
        status = "PASS" if ok else "FAIL"
        cnt_s = f"{got_cnt}/{ref['count']}" + ("" if cnt_ok else " X")
        sum_s = f"{got_sum}/{ref['sum']}" + ("" if sum_ok else " X")
        dec_s = "-" if mae is None else f"mae {mae:.2f}, {within}/{min(got_cnt, ref['count'])}<={DEC_TOL}"
        rows.append((name, status, cnt_s, sum_s, dec_s, ""))

    # tabell
    hdr = ("BILDE", "STATUS", "TREFF", "SUM (S-10)", "DESIMALER", "NOTAT")
    w = [6, 10, 8, 12, 28, 40]
    print("\n===== VERIFISERING MOT PANEL-FASIT =====")
    print("  ".join(h.ljust(wi) for h, wi in zip(hdr, w)))
    print("  ".join("-" * wi for wi in w))
    for r in rows:
        print("  ".join(str(c).ljust(wi) for c, wi in zip(r, w)))
    scored = [r for r in rows if r[1] in ("PASS", "FAIL")]
    print("  ".join("-" * wi for wi in w))
    print(f"\n{n_pass}/{len(scored)} bestaatt (antall+sum). "
          f"{len(rows) - len(scored)} forkastet/mangler/feil.")
    # exit-kode: 0 hvis alle scorede bestod, ellers 1 (for CI/regresjon)
    return 0 if n_pass == len(scored) and scored else 1


if __name__ == '__main__':
    sys.exit(main())
