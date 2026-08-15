"""
Sensur av visningsnavn (backend_spec §3).

Navnet er det ENESTE feltet som alltid deles med venner, og det er derfor det
eneste stedet en bruker kan skrive fritt til andre. Moderasjonen skjer
server-side FOER navnet eksponeres; til da staar kontoen med status `pending`
og navnet vises ikke for andre.

Tegnsettregelen speiler klientens `Ui.nameFilters()` eksakt: bokstaver og tall
i de latinske Unicode-blokkene (inkl. æ/ø/å og aksenter), mellomrom og
tegnsettingen `-_.'`. Klientfilteret er bekvemmelighet, ikke sikkerhet - en
modifisert klient kan sende hva som helst, saa regelen maa haandheves her.

Ordlisten er en kurert standardliste i `blocklist.py`, utvidet med
`DISPLAY_NAME_BLOCKLIST` per miljoe. Den var tom fram til 2026-08-12; hvorfor
den ikke er det lenger, og hvilke ord som bevisst IKKE staar der, hoerer hjemme
i `blocklist.py` - les den foer du legger til et ord. Spec §3 aapner for
«regelsett + evt. manuell koe»; den manuelle koeen krever en admin-flate som
ikke finnes ennaa, saa navn som passerer regelsettet godkjennes direkte.
"""
import unicodedata
from collections.abc import Sequence

from ..models import NameStatus

NAME_MAX = 24
PUNCTUATION = "-_.'"
# Character.UnicodeBlock BASIC_LATIN .. LATIN_EXTENDED_B i klienten.
LATIN_MAX_CODEPOINT = 0x24F


def _allowed_char(c: str) -> bool:
    if c == " " or c in PUNCTUATION:
        return True
    return c.isalnum() and ord(c) <= LATIN_MAX_CODEPOINT


def normalize(raw: str) -> str:
    """Trimmer og slaar sammen gjentatt mellomrom, saa «A    B» ikke blir et
    triks for aa skille seg ut i lister."""
    return " ".join(raw.strip().split())


def _fold(name: str) -> str:
    """Sammenlikningsform for ordlista: uten aksenter, smaa bokstaver, uten
    tegnsetting - saa «B-a-n-n-e» ikke slipper unna en treff paa «banne»."""
    stripped = unicodedata.normalize("NFKD", name.lower())
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return "".join(c for c in stripped if c.isalnum())


def review(raw: str, blocklist: Sequence[str],
           allowlist: Sequence[str] = ()) -> tuple[str, NameStatus, str]:
    """
    Returnerer (normalisert navn, status, begrunnelse).
    Begrunnelsen er ment aa vises for brukeren (§3: «brukeren varsles»).

    `allowlist` er uttrykk som skal gaa klar av ordlista. Foldingen fjerner
    mellomrommene, saa et blokkert ord kan oppstaa PAA TVERS av et ekte navn -
    «Anne Gerd» folder til «annegerd». Unntakene klippes derfor ut av den
    foldede formen foer ordlista sjekkes, slik at «Anne Gerd Hansen» ogsaa gaar
    klar, og ikke bare navnet uten etternavn.
    """
    name = normalize(raw)
    if not name:
        return name, NameStatus.rejected, "Visningsnavnet kan ikke være tomt."
    if len(name) > NAME_MAX:
        return name, NameStatus.rejected, f"Visningsnavnet kan ha høyst {NAME_MAX} tegn."
    ulovlige = sorted({c for c in name if not _allowed_char(c)})
    if ulovlige:
        return name, NameStatus.rejected, (
            "Visningsnavnet kan bare inneholde bokstaver, tall, mellomrom og "
            f"tegnene {PUNCTUATION}. Ikke tillatt: {' '.join(ulovlige)}")

    folded = _fold(name)
    for unntak in allowlist:
        fritatt = _fold(unntak)
        if fritatt:
            folded = folded.replace(fritatt, "")
    for word in blocklist:
        needle = _fold(word)
        if needle and needle in folded:
            return name, NameStatus.rejected, (
                "Visningsnavnet ble avvist av moderasjonen. Velg et annet navn.")

    return name, NameStatus.approved, ""
