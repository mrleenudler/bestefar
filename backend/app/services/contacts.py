"""
Validering av invitasjonsmottakere (backend_spec §4: «validering (identifiser
e-post vs telefon)»).

Bevisst romslig e-postregel: den eneste maaten aa vite sikkert at en adresse
finnes er aa sende til den, og en streng regel avviser gyldige adresser
(nye toppdomener, plusstegn, apostrof i navn). Vi fanger aapenbare skrivefeil
og lar leveringskvitteringen ta resten.

Telefonnumre normaliseres til E.164. Norske 8-sifrede numre uten landkode faar
+47; alt annet maa oppgis med landkode, siden vi ikke kan gjette landet.
"""
import re

from ..models import InviteTarget

EPOST = re.compile(r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$")
DEFAULT_LANDKODE = "+47"


class UgyldigMottaker(ValueError):
    pass


def classify(raw: str) -> tuple[InviteTarget, str]:
    """Returnerer (type, normalisert verdi). Kaster UgyldigMottaker."""
    verdi = raw.strip()
    if not verdi:
        raise UgyldigMottaker("Oppgi en e-postadresse eller et telefonnummer.")

    if "@" in verdi:
        if not EPOST.match(verdi):
            raise UgyldigMottaker("E-postadressen ser ikke gyldig ut.")
        return InviteTarget.email, verdi.lower()

    kompakt = re.sub(r"[\s\-()./]", "", verdi)
    if not re.fullmatch(r"\+?\d+", kompakt):
        raise UgyldigMottaker(
            "Mottakeren må være en e-postadresse eller et telefonnummer.")

    if kompakt.startswith("+"):
        if not 8 <= len(kompakt) - 1 <= 15:      # E.164
            raise UgyldigMottaker("Telefonnummeret har feil lengde.")
        return InviteTarget.phone, kompakt
    if len(kompakt) == 8:
        return InviteTarget.phone, DEFAULT_LANDKODE + kompakt
    raise UgyldigMottaker(
        "Oppgi telefonnummeret med landkode, f.eks. +47 12 34 56 78.")
