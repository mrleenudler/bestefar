"""
Kopierer feilanalyse-bilder fra én R2-bucket til en annen (§6).

Bakgrunn: `bestefar-scan-failures` ble opprettet med location hint EEUR og UTEN
jurisdiksjonsbinding. Et hint er en preferanse, ikke en garanti, og «Eastern
Europe» dekker dessuten land utenfor EOES. `bestefar-scan-failures-eur` er
opprettet med jurisdiksjon `eu`, som binder objektene til EU.

## Noekkelen er UENDRET

`failed_analyses.object_key` peker paa hele stien, og den skal ikke maatte
oppdateres. Denne modulen roerer derfor ALDRI databasen - den leser bare hvilke
noekler som finnes, og skriver objekter. Er kopieringen ferdig, er basen allerede
riktig.

## Rekkefoelgen er den samme som i legacy_bilder.py (B-48)

Per objekt: hent fra kilden, legg i maalet, LES TILBAKE fra maalet og
sammenlign byte for byte. Foerst da regnes det som kopiert. Forskjellen fra
B-48 er at ingenting slettes her - kilden staar uroert, og oppryddingen er en
egen runde etter at den nye bucketen er verifisert i produksjon med en ekte
donasjon.

## Den kan kjoeres om igjen

Ligger objektet allerede i maalet med identisk innhold, telles det som
«allerede der» og hoppes over. Ligger det der med ANNET innhold, er det en feil
og ikke noe vi overskriver i stillhet.

JURISDIKSJONSBUNDNE BUCKETS HAR EGET ENDEPUNKT (`.../eu`), saa kilde og maal
ligger paa hver sin URL. Det er grunnen til at `objstore.Bucket` finnes.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..models import FailedAnalysis
from . import objstore

log = logging.getLogger(__name__)


@dataclass
class Utfall:
    kopiert: list[str] = field(default_factory=list)
    allerede_der: list[str] = field(default_factory=list)
    feilet: list[tuple[str, str]] = field(default_factory=list)
    byte_kopiert: int = 0

    @property
    def ok(self) -> bool:
        return not self.feilet


def noekler(s: OrmSession) -> list[str]:
    """Noeklene basen kjenner. Basen er fasiten, ikke en kataloglisting."""
    return list(s.scalars(select(FailedAnalysis.object_key)
                          .where(FailedAnalysis.object_key.is_not(None))
                          .order_by(FailedAnalysis.id)))


def kopier(s: OrmSession, kilde: objstore.Bucket, maal: objstore.Bucket,
           toerrkjoer: bool = True,
           skriv: Callable[[str], None] = log.info) -> Utfall:
    """
    Kopierer alle objekter basen kjenner, fra `kilde` til `maal`.

    `toerrkjoer=True` (standard) leser fra kilden for aa faa stoerrelsen, men
    skriver ingenting. Kalleren maa be om aa faa skrive.
    """
    utfall = Utfall()
    for b, navn in ((kilde, "kilde"), (maal, "maal")):
        if not b.konfigurert:
            raise RuntimeError(f"Bucket ({navn}) er ikke konfigurert: {b}")
        feil = objstore.feilkonfigurasjon(b)
        if feil:
            # Ellers gaar jobben i gang og feiler paa hvert eneste objekt med
            # en 403 som ikke sier hva som er galt. Navnene i teksten er
            # R2_*; for kilden heter de R2_KILDE_* (B-51).
            raise RuntimeError(f"Bucket ({navn}) er feilkonfigurert: {feil}")
    if (kilde.endpoint, kilde.bucket) == (maal.endpoint, maal.bucket):
        raise RuntimeError(f"Kilde og maal er samme bucket: {kilde}")

    skriv(f"kilde: {kilde}")
    skriv(f"maal : {maal}")

    for noekkel in noekler(s):
        try:
            data = objstore.hent_fra(kilde, noekkel)
        except objstore.LagringFeilet as exc:
            utfall.feilet.append((noekkel, f"kunne ikke hente fra kilden: {exc}"))
            skriv(f"{noekkel}: FEILET - {exc}")
            continue

        try:
            fra_foer = objstore.hent_om_finnes(maal, noekkel)
        except objstore.LagringFeilet as exc:
            utfall.feilet.append((noekkel, f"kunne ikke lese maalet: {exc}"))
            skriv(f"{noekkel}: FEILET - {exc}")
            continue

        if fra_foer is not None:
            if fra_foer == data:
                utfall.allerede_der.append(noekkel)
                skriv(f"{noekkel}: allerede der ({len(data)} byte)")
            else:
                # Ikke overskriv. To ulike objekter paa samme noekkel er noe
                # annet enn en avbrutt kopiering, og skal ses paa av et menneske.
                utfall.feilet.append(
                    (noekkel, f"finnes i maalet med ANNET innhold "
                              f"({len(fra_foer)} byte mot {len(data)})"))
                skriv(f"{noekkel}: FEILET - finnes i maalet med annet innhold")
            continue

        if toerrkjoer:
            utfall.kopiert.append(noekkel)
            utfall.byte_kopiert += len(data)
            skriv(f"{noekkel}: ville kopiert {len(data)} byte")
            continue

        type_og_endelse = objstore.bildetype(data)
        content_type = type_og_endelse[0] if type_og_endelse else "application/octet-stream"
        try:
            objstore.legg_i(maal, noekkel, data, content_type)
            tilbake = objstore.hent_fra(maal, noekkel)
        except objstore.LagringFeilet as exc:
            utfall.feilet.append((noekkel, str(exc)))
            skriv(f"{noekkel}: FEILET - {exc}")
            continue

        if tilbake != data:
            utfall.feilet.append(
                (noekkel, f"lest tilbake {len(tilbake)} byte, la opp {len(data)}"))
            skriv(f"{noekkel}: FEILET - innholdet stemmer ikke etter opplasting")
            continue

        utfall.kopiert.append(noekkel)
        utfall.byte_kopiert += len(data)
        skriv(f"{noekkel}: kopiert {len(data)} byte")

    return utfall
