"""
Genererer contracts/openapi.json fra FastAPI-appen.

Fila sjekkes inn fordi UI-instansen skal kunne lese kontraktflaten uten aa
kjoere backenden. Den ligger i repo-roten og ikke under backend/, fordi den er
en DELT flate - se contracts/README.md.

Kjoeres fra backend/:
    .\\.venv\\Scripts\\python.exe tools\\gen_openapi.py           # skriv
    .\\.venv\\Scripts\\python.exe tools\\gen_openapi.py --check    # bare sammenlign

--check er det CI bruker. Den skriver ingenting og returnerer 1 ved avvik: en
innsjekket kontrakt som stille kommer i utakt med koden er verre enn ingen
kontrakt, for da bygger UI mot noe som ser autoritativt ut uten aa vaere det.

Utskriften MAA vaere deterministisk, ellers feiler CI paa stoey. Derfor
sort_keys og fast innrykk; FastAPI bygger dict-ene i deklarasjonsrekkefoelge,
som endrer seg naar ruter flyttes rundt.
"""
import argparse
import json
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
UT = BACKEND.parent / "contracts" / "openapi.json"


def skjema() -> str:
    # Miljoeet skal ikke paavirke skjemaet. ENV=dev er likegyldig for
    # app.openapi(), men settings() leses ved import av app.main, og en
    # halvfylt .env skal ikke kunne endre en innsjekket kontrakt.
    os.environ.setdefault("DATABASE_URL", "sqlite://")
    os.environ["AUTO_CREATE_TABLES"] = "false"
    sys.path.insert(0, str(BACKEND))

    from app.main import app                                   # noqa: PLC0415

    data = app.openapi()
    return json.dumps(data, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true",
                   help="sammenlign uten aa skrive; 1 ved avvik")
    args = p.parse_args()

    ny = skjema()

    if not args.check:
        UT.parent.mkdir(exist_ok=True)
        UT.write_text(ny, encoding="utf-8")
        print(f"Skrev {UT.relative_to(BACKEND.parent)} ({len(ny)} tegn)")
        return 0

    if not UT.exists():
        print(f"::error title=Kontrakten mangler::{UT.name} finnes ikke. "
              "Kjoer tools/gen_openapi.py og sjekk den inn.")
        return 1

    gammel = UT.read_text(encoding="utf-8")
    if gammel == ny:
        print("contracts/openapi.json er i takt med koden.")
        return 0

    import difflib                                             # noqa: PLC0415
    diff = list(difflib.unified_diff(
        gammel.splitlines(), ny.splitlines(),
        fromfile="contracts/openapi.json (innsjekket)",
        tofile="generert fra koden", lineterm="", n=2))
    # Bare de foerste linjene: GitHub kutter lange annotasjoner fra SLUTTEN.
    utdrag = "\n".join(diff[:120])
    print(utdrag)
    melding = (utdrag.replace("%", "%25").replace("\r", "")
               .replace("\n", "%0A"))
    print(f"::error title=contracts/openapi.json er utdatert::{melding}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
