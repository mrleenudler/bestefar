"""
Offentlig bruker-ID (backend_spec §3.1).

Kort, haandskrivbar streng i et forvekslingssikkert alfabet (Crockford base32
utelater I, L, O og U). Siste tegn er et sjekksiffer, saa aapenbare tastefeil
avvises uten et databaseoppslag.

Formatet foelger eksempelet i spec-en, `BF-7Q4K-9F2M`: 8 signifikante tegn der
de 7 foerste er tilfeldige (~34 milliarder ID-er) og det 8. er sjekksifferet.
Mot forventet ~10^4 reelle brukere er gjetting upraktisk.

Sjekksifferet er `sum(verdier) mod 32` i SAMME alfabet - ikke Crockfords
mod-37-variant, som ville trukket inn fire symboler utenfor alfabetet og gjort
ID-en vanskeligere aa lese opp.
"""
import secrets

ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"      # Crockford base32
_VALUE = {c: i for i, c in enumerate(ALPHABET)}
# Crockfords anbefalte normalisering ved innlesing.
_FOLD = {"I": "1", "L": "1", "O": "0", "U": "V"}

PREFIX = "BF"
RANDOM_LEN = 7


def _checksum(chars: str) -> str:
    return ALPHABET[sum(_VALUE[c] for c in chars) % 32]


def generate() -> str:
    """Ny ID paa visningsform, f.eks. `BF-7Q4K-9F2M`."""
    body = "".join(secrets.choice(ALPHABET) for _ in range(RANDOM_LEN))
    return format_id(body + _checksum(body))


def normalize(raw: str) -> str | None:
    """
    Tar imot det brukeren skrev (med eller uten prefiks, bindestreker, smaa
    bokstaver, forvekslede tegn) og returnerer de 8 signifikante tegnene.
    None hvis lengden er feil eller et tegn ikke finnes i alfabetet.
    """
    s = raw.strip().upper().replace("-", "").replace(" ", "")
    if s.startswith(PREFIX):
        s = s[len(PREFIX):]
    s = "".join(_FOLD.get(c, c) for c in s)
    if len(s) != RANDOM_LEN + 1 or any(c not in _VALUE for c in s):
        return None
    return s


def is_valid(raw: str) -> bool:
    """Sjekksifferkontroll - avviser tastefeil foer vi slaar opp i databasen."""
    s = normalize(raw)
    return s is not None and _checksum(s[:-1]) == s[-1]


def format_id(significant: str) -> str:
    """`7Q4K9F2M` -> `BF-7Q4K-9F2M`."""
    return f"{PREFIX}-{significant[:4]}-{significant[4:]}"
