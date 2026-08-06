"""
Kryptering i ro av deponert noekkelmateriale (backend_spec §2).

Deponeringen er frivillig og betyr at serveren kan aapne backup-bloben. Det
kan vi ikke gjoere noe med - det er hele poenget med valget. Det vi KAN gjoere,
er aa sikre at materialet ikke ligger lesbart i databasen: en Supabase-dump paa
avveie skal ikke i seg selv gi noen noekkelen. Hemmeligheten ligger som en
Fly-secret, altsaa et annet sted enn basen.

AES-256-GCM med tilfeldig 12-byte nonce. Noekkelen avledes med HKDF-SHA256 fra
`BACKUP_ESCROW_SECRET`, slik at hemmeligheten kan vaere en vanlig passfrase og
ikke maa vaere noeyaktig 32 byte. AAD er bruker-ID-en: en rad kan da ikke
flyttes fra én bruker til en annen i basen uten at dekrypteringen feiler.

Formatet er versjonert (1 byte) fordi noekkelmateriale er det siste man vil
maatte gjette seg til om fem aar.
"""
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ..config import Settings

VERSJON = 1
NONCE_BYTES = 12
_INFO = b"bestefar-backup-key-escrow-v1"


class EscrowNotConfigured(RuntimeError):
    """BACKUP_ESCROW_SECRET mangler. Da lagrer vi ingenting."""


class EscrowUnreadable(RuntimeError):
    """Raden finnes, men kan ikke dekrypteres - som regel rotert hemmelighet."""


def er_konfigurert(cfg: Settings) -> bool:
    return bool(cfg.backup_escrow_secret.strip())


def _noekkel(cfg: Settings) -> bytes:
    hemmelighet = cfg.backup_escrow_secret.strip()
    if not hemmelighet:
        raise EscrowNotConfigured(
            "Noekkeldeponering er ikke konfigurert paa serveren.")
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_INFO).derive(hemmelighet.encode("utf-8"))


def krypter(cfg: Settings, klartekst: bytes, user_id: str) -> bytes:
    nonce = os.urandom(NONCE_BYTES)
    ciph = AESGCM(_noekkel(cfg)).encrypt(nonce, klartekst, user_id.encode())
    return bytes([VERSJON]) + nonce + ciph


def dekrypter(cfg: Settings, lagret: bytes, user_id: str) -> bytes:
    if not lagret or lagret[0] != VERSJON:
        raise EscrowUnreadable("Ukjent format paa deponert noekkelmateriale.")
    nonce = lagret[1:1 + NONCE_BYTES]
    try:
        return AESGCM(_noekkel(cfg)).decrypt(nonce, lagret[1 + NONCE_BYTES:],
                                             user_id.encode())
    except InvalidTag as exc:
        # Hemmeligheten er rotert, eller raden er flyttet mellom brukere.
        # Begge deler er en driftsfeil, ikke en brukerfeil - men resultatet for
        # brukeren er det samme: materialet er tapt, koden maa brukes.
        raise EscrowUnreadable(
            "Deponert noekkelmateriale kan ikke leses. Bruk "
            "gjenopprettingskoden.") from exc
