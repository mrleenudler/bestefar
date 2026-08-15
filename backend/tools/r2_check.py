"""
Verifiserer at objektlagringen (§6) faktisk virker med de secretene som staar.

Gjoer en ekte rundtur mot R2 - PUT, GET, sammenlign, DELETE - og skriver hva
hvert steg svarte. Dette er den ene maaten aa vite at signeringen i
services/objstore.py er riktig; en feil der gir 403 fra R2, ikke en feil hos
oss, og uten dette scriptet ville den vist seg foerste gang en bruker donerte
et bilde. Kjoer den etter at secretene er satt, og skriv utfallet inn der du
paastaar at lagringen er i produksjon.

Kjoeres fra repo-roten (scriptet bytter selv til backend/, saa .env leses):
    backend\\.venv\\Scripts\\python.exe backend\\tools\\r2_check.py

Leser de samme verdiene som appen (backend/.env eller miljoevariabler) og
skriver aldri ut noekler.

Testobjektet heter `feilanalyse/_r2_check/<tilfeldig>` og slettes til slutt -
men BARE hvis nedlastingen lyktes. Feiler den, blir objektet liggende med
noekkelen skrevet ut, saa det kan ses paa i Cloudflare-panelet.
"""
import os
import secrets
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
# Settings leser `.env` relativt til arbeidskatalogen. Uten dette virker
# scriptet bare naar det kalles fra backend/, og det er ikke der man staar.
os.chdir(BACKEND)

from app.config import settings                                # noqa: E402
from app.services import objstore                              # noqa: E402


def main() -> int:
    cfg = settings()
    if not objstore.er_konfigurert(cfg):
        mangler = [navn for navn, verdi in
                   (("R2_ENDPOINT", cfg.r2_endpoint),
                    ("R2_BUCKET", cfg.r2_bucket),
                    ("R2_ACCESS_KEY_ID", cfg.r2_access_key_id),
                    ("R2_SECRET_ACCESS_KEY", cfg.r2_secret_access_key))
                   if not verdi]
        print(f"R2 er ikke konfigurert - mangler: {', '.join(mangler)}")
        return 1

    print(f"endepunkt : {cfg.r2_endpoint}")
    print(f"bucket    : {cfg.r2_bucket}   region: {cfg.r2_region}")

    noekkel = f"feilanalyse/_r2_check/{secrets.token_hex(8)}.txt"
    innhold = secrets.token_bytes(64)

    try:
        objstore.legg(cfg, noekkel, innhold, "text/plain")
        print(f"PUT       : ok  ({noekkel})")
    except objstore.LagringFeilet as exc:
        print(f"PUT       : FEILET  {exc}")
        return 1

    try:
        tilbake = objstore.hent(cfg, noekkel)
    except objstore.LagringFeilet as exc:
        print(f"GET       : FEILET  {exc}")
        print(f"            Testobjektet ligger igjen: {noekkel}")
        return 1

    if tilbake != innhold:
        print(f"GET       : FEIL INNHOLD ({len(tilbake)} byte tilbake, "
              f"{len(innhold)} lagt opp)")
        print(f"            Testobjektet ligger igjen: {noekkel}")
        return 1
    print(f"GET       : ok  ({len(tilbake)} byte, identisk)")

    try:
        objstore.slett(cfg, noekkel)
        print("DELETE    : ok")
    except objstore.LagringFeilet as exc:
        print(f"DELETE    : FEILET  {exc}")
        print(f"            Rydd bort {noekkel} manuelt.")
        return 1

    print("\nObjektlagringen virker. /health skal na svare \"bilder\": \"r2\".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
