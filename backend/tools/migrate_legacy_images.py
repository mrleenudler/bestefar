"""
Flytter feilanalyse-bilder fra databasen til R2 (AAP-B11).

Logikken ligger i app/services/legacy_bilder.py, som har testene. Dette er
skallet rundt, og det gjoer én ting utover aa kalle den: det krever at du ber om
aa faa skrive.

Verktoeyet trenger BAADE databasen og R2-noeklene, og begge deler finnes bare i
produksjon. Derfor foelger det med imaget og kjoeres der (B-47):

    flyctl ssh console -a bestefar-api -C "python tools/migrate_legacy_images.py"
    flyctl ssh console -a bestefar-api -C "python tools/migrate_legacy_images.py --utfoer"

Uten --utfoer er det en toerrkjoering: den sier hva som ville blitt flyttet, og
roerer verken R2 eller basen. Kjoer den foerst, og les lista.
"""
import argparse
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
os.chdir(BACKEND)

from app.config import settings                              # noqa: E402
from app.db import db                                        # noqa: E402
from app.services import legacy_bilder                       # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--utfoer", action="store_true",
                   help="skriv faktisk; uten den er det en toerrkjoering")
    args = p.parse_args()

    cfg = settings()
    s = next(db())
    try:
        print("TOERRKJOERING - ingenting skrives\n" if not args.utfoer
              else "UTFOERER - laster opp og toemmer image_legacy\n")
        utfall = legacy_bilder.flytt(s, cfg, toerrkjoer=not args.utfoer,
                                     skriv=print)
    finally:
        s.close()

    print(f"\n{len(utfall.flyttet)} rad(er), {utfall.byte_flyttet} byte")
    for rad, grunn in utfall.hoppet_over:
        print(f"  hoppet over: rad {rad} - {grunn}")
    for rad, grunn in utfall.feilet:
        print(f"  FEILET: rad {rad} - {grunn}")

    if not args.utfoer and utfall.flyttet:
        print("\nKjoer paa nytt med --utfoer for aa gjoere det.")
    return 0 if utfall.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
