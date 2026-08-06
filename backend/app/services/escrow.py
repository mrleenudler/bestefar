"""
Kryptering i ro av deponert noekkelmateriale (backend_spec §2.1).

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

TO TING SOM GJOER EN UTSKIFTNING OVERLEVBAR:

  `BACKUP_ESCROW_SECRET_OLD` proeves naar den gjeldende hemmeligheten ikke
  aapner raden, og raden krypteres om ved foerste lesing. En utskiftning blir
  da en gradvis migrering i stedet for et stup - og en hemmelighet satt ved et
  uhell kan angres saa lenge den gamle fortsatt staar.

  NOEKKEL-ID-en (`noekkel_id`) er et fingeravtrykk av den avledede noekkelen -
  HMAC over en fast streng, altsaa ingenting om selve hemmeligheten. Den lagres
  paa raden, saa /health kan si «N rader ligger paa en annen hemmelighet enn
  den som staar naa». Uten den ville en feilsatt hemmelighet foerst vist seg
  den dagen en bruker proevde aa gjenopprette - verst tenkelige tidspunkt.
"""
import hmac
import os
from hashlib import sha256

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from ..config import Settings

VERSJON = 1
NONCE_BYTES = 12
_INFO = b"bestefar-backup-key-escrow-v1"
_KCV = b"bestefar-escrow-kcv"


class EscrowNotConfigured(RuntimeError):
    """BACKUP_ESCROW_SECRET mangler. Da lagrer vi ingenting."""


class EscrowUnreadable(RuntimeError):
    """Raden finnes, men ingen av hemmelighetene aapner den."""


def er_konfigurert(cfg: Settings) -> bool:
    return bool(cfg.backup_escrow_secret.strip())


def _avled(hemmelighet: str) -> bytes:
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=_INFO).derive(hemmelighet.encode("utf-8"))


def _noekler(cfg: Settings) -> list[bytes]:
    """Gjeldende hemmelighet foerst, deretter den forrige om den er satt."""
    ut = []
    for raa in (cfg.backup_escrow_secret, cfg.backup_escrow_secret_old):
        raa = raa.strip()
        if raa:
            ut.append(_avled(raa))
    if not ut:
        raise EscrowNotConfigured(
            "Noekkeldeponering er ikke konfigurert paa serveren.")
    return ut


def _id_for(noekkel: bytes) -> str:
    # Fingeravtrykk, ikke hemmelighet: HMAC over en KONSTANT streng. Verdien er
    # lik for alle rader kryptert med samme hemmelighet, og sier ingenting om
    # hva hemmeligheten er.
    return hmac.new(noekkel, _KCV, sha256).hexdigest()[:16]


def noekkel_id(cfg: Settings) -> str:
    """ID-en til hemmeligheten som gjelder NAA. Tom hvis ingen er satt."""
    raa = cfg.backup_escrow_secret.strip()
    return _id_for(_avled(raa)) if raa else ""


def krypter(cfg: Settings, klartekst: bytes, user_id: str) -> tuple[bytes, str]:
    """Returnerer (lagringsklar blob, noekkel-ID)."""
    noekkel = _noekler(cfg)[0]
    nonce = os.urandom(NONCE_BYTES)
    ciph = AESGCM(noekkel).encrypt(nonce, klartekst, user_id.encode())
    return bytes([VERSJON]) + nonce + ciph, _id_for(noekkel)


def dekrypter(cfg: Settings, lagret: bytes,
              user_id: str) -> tuple[bytes, bool]:
    """
    Returnerer (klartekst, om den GAMLE hemmeligheten maatte til).

    Kalleren bruker det andre elementet til aa kryptere raden om, slik at en
    utskiftning migrerer seg selv etter hvert som radene leses.
    """
    if not lagret or lagret[0] != VERSJON:
        raise EscrowUnreadable("Ukjent format paa deponert noekkelmateriale.")
    nonce = lagret[1:1 + NONCE_BYTES]
    ciph = lagret[1 + NONCE_BYTES:]

    for i, noekkel in enumerate(_noekler(cfg)):
        try:
            return AESGCM(noekkel).decrypt(nonce, ciph, user_id.encode()), i > 0
        except InvalidTag:
            continue

    # Hemmeligheten er skiftet ut uten at den gamle ble beholdt, eller raden er
    # flyttet mellom brukere. Begge deler er en driftsfeil, ikke en brukerfeil -
    # men resultatet for brukeren er det samme: materialet er tapt, koden maa
    # brukes.
    raise EscrowUnreadable(
        "Deponert noekkelmateriale kan ikke leses. Bruk gjenopprettingskoden.")
