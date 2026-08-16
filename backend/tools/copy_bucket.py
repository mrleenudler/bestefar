"""
Engangsjobb: kopier feilanalyse-bildene til den jurisdiksjonsbundne bucketen.

`bestefar-scan-failures` ble opprettet med location hint EEUR og uten
jurisdiksjonsbinding - et hint er en preferanse, ikke en garanti, og «Eastern
Europe» dekker land utenfor EOES. `bestefar-scan-failures-eur` har jurisdiksjon
`eu`. Logikken ligger i app/services/bucketflytt.py, som har testene; dette er
skallet.

Noeklene er UENDRET. `object_key` i basen peker paa hele stien, og jobben roerer
ikke databasen i det hele tatt.

MAALET er bucketen i den vanlige konfigurasjonen (R2_ENDPOINT/R2_BUCKET), altsaa
den tjenesten skriver til naa. KILDEN oppgis separat:

    R2_KILDE_ENDPOINT           https://<konto>.r2.cloudflarestorage.com
    R2_KILDE_BUCKET             bestefar-scan-failures
    R2_KILDE_ACCESS_KEY_ID      valgfri - faller tilbake paa R2_ACCESS_KEY_ID
    R2_KILDE_SECRET_ACCESS_KEY  valgfri - faller tilbake paa R2_SECRET_ACCESS_KEY

Rekkefoelgen som gjoer at ingenting kan bli liggende igjen: bytt R2_ENDPOINT og
R2_BUCKET til den nye bucketen FOERST, kjoer denne etterpaa. Da skriver
tjenesten allerede til den nye, og kilden kan ikke faa nye objekter mens jobben
gaar.

Kjoeres paa Fly-maskinen, der begge sett noekler finnes (B-47):

    flyctl ssh console -a bestefar-api -C "python tools/copy_bucket.py"
    flyctl ssh console -a bestefar-api -C "python tools/copy_bucket.py --utfoer"

Uten --utfoer er det en toerrkjoering. Den kan kjoeres om igjen: objekter som
allerede ligger i maalet med identisk innhold hoppes over.

SLETTER INGENTING. Den gamle bucketen staar uroert til den nye er verifisert i
produksjon med en ekte donasjon; oppryddingen er en egen runde.
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
from app.services import bucketflytt, objstore               # noqa: E402


def kilde_fra_env(cfg) -> objstore.Bucket:
    return objstore.Bucket(
        endpoint=os.environ.get("R2_KILDE_ENDPOINT", ""),
        bucket=os.environ.get("R2_KILDE_BUCKET", ""),
        # Faller tilbake paa hovednoeklene: er tokenet kontoomfattende, gjelder
        # det begge bucketene, og da skal man slippe aa sette dem to ganger.
        access_key_id=os.environ.get("R2_KILDE_ACCESS_KEY_ID") or cfg.r2_access_key_id,
        secret_access_key=(os.environ.get("R2_KILDE_SECRET_ACCESS_KEY")
                           or cfg.r2_secret_access_key),
        region=cfg.r2_region,
        timeout=cfg.r2_timeout_seconds,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--utfoer", action="store_true",
                   help="skriv faktisk; uten den er det en toerrkjoering")
    args = p.parse_args()

    cfg = settings()
    kilde = kilde_fra_env(cfg)
    maal = objstore.fra_settings(cfg)

    if not kilde.endpoint or not kilde.bucket:
        print("Mangler R2_KILDE_ENDPOINT og/eller R2_KILDE_BUCKET.")
        return 1

    print("TOERRKJOERING - ingenting skrives\n" if not args.utfoer
          else "UTFOERER - kopierer til maalbucketen\n")

    s = next(db())
    try:
        utfall = bucketflytt.kopier(s, kilde, maal, toerrkjoer=not args.utfoer,
                                    skriv=print)
    finally:
        s.close()

    print(f"\n{len(utfall.kopiert)} kopiert ({utfall.byte_kopiert} byte), "
          f"{len(utfall.allerede_der)} laa der fra foer")
    for noekkel, grunn in utfall.feilet:
        print(f"  FEILET: {noekkel} - {grunn}")

    if not args.utfoer and utfall.kopiert:
        print("\nKjoer paa nytt med --utfoer for aa gjoere det.")
    if utfall.ok:
        print("Kilden er IKKE roert. Slett den i en egen runde, etter at den "
              "nye bucketen er verifisert med en ekte donasjon.")
    return 0 if utfall.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
