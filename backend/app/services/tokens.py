"""
Vaare EGNE tokens (backend_spec §1).

To ulike ting, med vilje:

  access-token   kortlivet JWT (HS256) som verifiseres med signatur ALENE -
                 ingen databaseoppslag per forespoersel. Prisen er at det ikke
                 kan tilbakekalles foer det utloeper; derfor er levetiden kort.

  refresh-token  32 tilfeldige byte, ugjennomsiktig for klienten. Vi lagrer
                 bare SHA-256 av det (models.AuthSession), saa en lekket
                 databasedump ikke inneholder brukbare tokens. Det KAN
                 tilbakekalles, og roteres ved hver bruk.

Hvorfor ikke JWT ogsaa til refresh: et refresh-token maa kunne trekkes tilbake
(utlogging, tyveri), og da trengs uansett en rad i basen. Da er det enklere at
tokenet ikke sier noe i seg selv.
"""
import hashlib
import secrets
from datetime import timedelta

import jwt
from fastapi import HTTPException

from ..config import Settings
from ..models import utcnow

ALGORITHM = "HS256"
ISSUER = "bestefar-api"
# RFC 7518 §3.2: HMAC-noekkelen boer vaere minst like lang som hash-utdataet.
# En kortere noekkel kan brutes, og da kan hvem som helst utstede tokens for
# hvilken som helst bruker. Lag den med `openssl rand -base64 48`.
MIN_SECRET_BYTES = 32


def krev_hemmelighet(cfg: Settings) -> str:
    """
    Uten JWT_SECRET kan vi ikke utstede noe. Vi faller BEVISST ikke tilbake paa
    en standardverdi: den ville ligget i repoet og gjort alle tokens
    forfalskbare. 503 er riktig - tjenesten er feilkonfigurert, ikke klienten.
    """
    if not cfg.jwt_secret:
        raise HTTPException(503, "Innlogging er ikke konfigurert (JWT_SECRET mangler).")
    if len(cfg.jwt_secret.encode()) < MIN_SECRET_BYTES:
        raise HTTPException(503, "Innlogging er feilkonfigurert: JWT_SECRET maa "
                                 f"vaere minst {MIN_SECRET_BYTES} tegn.")
    return cfg.jwt_secret


def lag_access_token(cfg: Settings, user_id: str) -> tuple[str, int]:
    """Returnerer (token, levetid i sekunder)."""
    hemmelighet = krev_hemmelighet(cfg)
    ttl = timedelta(minutes=cfg.access_token_ttl_minutes)
    naa = utcnow()
    payload = {"sub": user_id, "iss": ISSUER,
               "iat": int(naa.timestamp()),
               "exp": int((naa + ttl).timestamp())}
    return jwt.encode(payload, hemmelighet, algorithm=ALGORITHM), int(ttl.total_seconds())


def les_access_token(cfg: Settings, token: str) -> str:
    """Returnerer bruker-ID-en, eller kaster 401."""
    hemmelighet = krev_hemmelighet(cfg)
    try:
        # algorithms= er ikke valgfritt: uten den ville et token som oppgir
        # alg=none blitt godtatt uten signatur.
        data = jwt.decode(token, hemmelighet, algorithms=[ALGORITHM], issuer=ISSUER)
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(401, "Tokenet er utloept.") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(401, "Ugyldig token.") from exc
    sub = data.get("sub")
    if not sub:
        raise HTTPException(401, "Ugyldig token.")
    return str(sub)


def nytt_refresh_token() -> tuple[str, str]:
    """Returnerer (klartekst til klienten, hash til basen)."""
    raa = secrets.token_urlsafe(32)
    return raa, hash_token(raa)


def hash_token(raa: str) -> str:
    return hashlib.sha256(raa.encode()).hexdigest()


def ny_engangskode() -> tuple[str, str]:
    """Sekssifret kode til e-post. Returnerer (kode, hash)."""
    kode = f"{secrets.randbelow(1_000_000):06d}"
    return kode, hash_token(kode)


def koder_er_like(oppgitt: str, lagret_hash: str) -> bool:
    """Konstant tid - ellers kan svartiden roepe hvor mange siffer som stemte."""
    return secrets.compare_digest(hash_token(oppgitt), lagret_hash)
