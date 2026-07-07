"""
Verifiser C++-kjernen (core/build/bestefar_cli) mot panel-fasiten, med samme
oracle som verify_cset.py. Kjoer:
  .venv\\Scripts\\python.exe verify_cset_cpp.py
Krav for bestaatt port: 10/10 PASS (som Python-referansen).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

CLI = Path("core/build/bestefar_cli.exe")
IN = Path("Testsett")
TRUTH = Path("hits_truth.txt")
DEC_TOL = 0.15


def load_truth(path):
    truth = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"(\S+)\s+(\d+)\s+(\d+)\s+(.*)", line)
        if not m:
            continue
        name, cnt, ssum, decs = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        decimals = sorted(float(x) for x in decs.replace(",", " ").split())
        truth[name] = dict(count=cnt, sum=ssum, decimals=decimals)
    return truth


def pair_decimals(truth_decs, got_decs):
    t, g = sorted(truth_decs), sorted(got_decs)
    n = min(len(t), len(g))
    if n == 0:
        return None, 0
    diffs = [abs(t[i] - g[i]) for i in range(n)]
    return sum(diffs) / n, sum(1 for d in diffs if d <= DEC_TOL)


def main():
    if not CLI.exists():
        print(f"Mangler {CLI} - bygg foerst"); return 1
    truth = load_truth(TRUTH)
    names = sorted(truth.keys(), key=lambda s: int(re.sub(r"\D", "", s) or 0))

    n_pass = 0
    rows = []
    for name in names:
        ref = truth[name]
        img = IN / f"{name}.jpg"
        if not img.exists():
            rows.append((name, "MANGLER", "", "", "", "")); continue
        proc = subprocess.run([str(CLI), str(img)], capture_output=True, text=True,
                              timeout=300)
        try:
            out = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            rows.append((name, "FEIL JSON", "", "", "", proc.stdout[:40])); continue
        if out["status"] != "OK":
            rows.append((name, "FORKASTET", f"({ref['count']})", str(ref["sum"]), "",
                         out.get("message", "")[:40]))
            continue
        got_cnt = len(out["hits"])
        got_sum = out["sum_integer"]
        got_decs = [h["decimal"] for h in out["hits"]]
        cnt_ok = got_cnt == ref["count"]
        sum_ok = got_sum == ref["sum"]
        mae, within = pair_decimals(ref["decimals"], got_decs)
        ok = cnt_ok and sum_ok
        n_pass += ok
        rows.append((name, "PASS" if ok else "FAIL",
                     f"{got_cnt}/{ref['count']}" + ("" if cnt_ok else " X"),
                     f"{got_sum}/{ref['sum']}" + ("" if sum_ok else " X"),
                     "-" if mae is None else f"mae {mae:.2f}, {within}/{min(got_cnt, ref['count'])}<={DEC_TOL}",
                     ""))

    hdr = ("BILDE", "STATUS", "TREFF", "SUM (S-10)", "DESIMALER", "NOTAT")
    w = [6, 10, 8, 12, 28, 40]
    print("\n===== C++-KJERNE MOT PANEL-FASIT =====")
    print("  ".join(h.ljust(wi) for h, wi in zip(hdr, w)))
    print("  ".join("-" * wi for wi in w))
    for r in rows:
        print("  ".join(str(c).ljust(wi) for c, wi in zip(r, w)))
    scored = [r for r in rows if r[1] in ("PASS", "FAIL")]
    print("  ".join("-" * wi for wi in w))
    print(f"\n{n_pass}/{len(scored)} bestaatt (antall+sum). "
          f"{len(rows) - len(scored)} forkastet/mangler/feil.")
    return 0 if n_pass == len(scored) and len(scored) == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
