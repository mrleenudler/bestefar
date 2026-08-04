"""
Innlogging (backend_spec §1).

Modellen er den vanlige for en mobilapp:

  1. Klienten logger inn med leverandoerens SDK (Google/Apple) og faar et
     ID-token, ELLER ber om en engangskode paa e-post.
  2. Klienten sender det til oss. Vi verifiserer og svarer med VAARE egne
     tokens - et kortlivet access-token og et langlivet refresh-token.
  3. Alle andre endepunkter bruker access-tokenet.

Grunnen til at vi ikke bare bruker leverandoerens token videre: da hadde hver
forespoersel maattet verifiseres mot Google/Apple, e-postinnlogging hadde ikke
passet inn i det hele tatt, og vi kunne ikke logget ut en enhet.

Kontosammenslaaing: samme person som logger inn med Google én gang og Apple
neste gang skal IKKE faa to kontoer. Vi kobler paa VERIFISERT e-postadresse.
Uverifisert e-post kobles aldri - da kunne en angriper laget en konto hos en
slapp leverandoer med noen andres adresse og overtatt kontoen deres.
"""
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..config import Settings, settings
from ..db import db
from ..models import (AuthIdentity, AuthSession, EmailLoginCode, NameStatus,
                      Provider, SharingPreference, User, as_utc, utcnow)
from ..services import ids, mailer, moderation, oidc, tokens

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/auth", tags=["innlogging"])


# --------------------------------------------------------------------
# Felles
# --------------------------------------------------------------------

def _navn_fra_epost(epost: str | None) -> str:
    """
    Et foerste visningsnavn saa kontoen ikke staar tom. Brukeren endrer det i
    profilen; her handler det bare om aa ha noe lesbart aa vise.
    """
    lokal = (epost or "").split("@")[0].replace(".", " ").replace("_", " ")
    navn, status, _ = moderation.review(lokal or "Skytter", [])
    return navn if status == NameStatus.approved else "Skytter"


def _ny_bruker(s: OrmSession, epost: str | None) -> User:
    user = User(public_id=ids.generate(), display_name=_navn_fra_epost(epost))
    s.add(user)
    # MAA skylles foer radene som peker paa den: det finnes ingen
    # relationship() mellom User og SharingPreference, saa SQLAlchemy har
    # ingen rekkefoelge aa sortere etter. Se CLAUDE_CONTEXT.txt.
    s.flush()
    s.add(SharingPreference(user_id=user.id))
    return user


def _finn_eller_lag_bruker(s: OrmSession, ident: oidc.Identitet) -> tuple[User, bool]:
    """Returnerer (bruker, er_ny)."""
    rad = s.scalar(select(AuthIdentity).where(
        AuthIdentity.provider == ident.provider,
        AuthIdentity.subject == ident.subject))
    if rad is not None:
        # Leverandoeren kan ha faatt ny adresse siden sist.
        if ident.email and rad.email != ident.email:
            rad.email = ident.email
        return rad.user, False

    user: User | None = None
    if ident.email and ident.email_verifisert:
        # Samme verifiserte adresse hos en annen leverandoer = samme person.
        eksisterende = s.scalar(select(AuthIdentity).where(
            AuthIdentity.email == ident.email).order_by(AuthIdentity.id))
        if eksisterende is not None:
            user = eksisterende.user

    er_ny = user is None
    if user is None:
        user = _ny_bruker(s, ident.email)
    s.add(AuthIdentity(user_id=user.id, provider=ident.provider,
                       subject=ident.subject, email=ident.email))
    return user, er_ny


def _start_oekt(s: OrmSession, cfg: Settings, user: User,
                user_agent: str) -> dict:
    access, levetid = tokens.lag_access_token(cfg, user.id)
    raa, hash_ = tokens.nytt_refresh_token()
    s.add(AuthSession(user_id=user.id, refresh_hash=hash_,
                      user_agent=(user_agent or "")[:200],
                      expires_at=utcnow() + timedelta(days=cfg.refresh_token_ttl_days)))
    user.last_seen_at = utcnow()
    s.commit()
    return {"access_token": access, "refresh_token": raa,
            "token_type": "Bearer", "expires_in": levetid,
            "user_id": user.id, "public_id": user.public_id,
            "display_name": user.display_name}


# --------------------------------------------------------------------
# Google / Apple
# --------------------------------------------------------------------

class IdTokenIn(BaseModel):
    id_token: str = Field(min_length=1, max_length=8192)


@router.post("/google")
def logg_inn_google(body: IdTokenIn,
                    user_agent: str = Header(default=""),
                    s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    ident = oidc.verifiser_google(cfg, body.id_token)
    user, er_ny = _finn_eller_lag_bruker(s, ident)
    ut = _start_oekt(s, cfg, user, user_agent)
    ut["is_new"] = er_ny
    return ut


@router.post("/apple")
def logg_inn_apple(body: IdTokenIn,
                   user_agent: str = Header(default=""),
                   s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    ident = oidc.verifiser_apple(cfg, body.id_token)
    user, er_ny = _finn_eller_lag_bruker(s, ident)
    ut = _start_oekt(s, cfg, user, user_agent)
    ut["is_new"] = er_ny
    return ut


# --------------------------------------------------------------------
# E-post med engangskode
# --------------------------------------------------------------------

class EpostIn(BaseModel):
    email: EmailStr


class KodeIn(BaseModel):
    email: EmailStr
    code: str = Field(min_length=4, max_length=10)


@router.post("/email/start", status_code=202)
def send_kode(body: EpostIn, s: OrmSession = Depends(db)) -> dict:
    """
    Svarer ALLTID 202, uansett om adressen finnes fra foer. Et svar som skilte
    mellom «kjent» og «ukjent» ville gjort endepunktet til et oppslagsverk
    over hvem som bruker appen.
    """
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    epost = body.email.lower()

    # Ratebegrensning i BASEN, ikke i minnet: Fly kjoerer to maskiner, og en
    # teller per prosess ville i praksis doblet grensen.
    grense = utcnow() - timedelta(hours=1)
    antall = len(list(s.scalars(select(EmailLoginCode).where(
        EmailLoginCode.email == epost, EmailLoginCode.created_at >= grense))))
    if antall >= cfg.email_code_rate_per_hour:
        raise HTTPException(429, "For mange koder er sendt til denne adressen. "
                                 "Proev igjen om en time.")

    kode, hash_ = tokens.ny_engangskode()
    s.add(EmailLoginCode(email=epost, code_hash=hash_,
                         expires_at=utcnow() + timedelta(minutes=cfg.email_code_ttl_minutes)))
    s.commit()

    try:
        mailer.send(cfg, "Innloggingskode til Bestefar",
                    f"Koden din er {kode}.\n\n"
                    f"Den er gyldig i {cfg.email_code_ttl_minutes} minutter.\n"
                    "Har du ikke bedt om aa logge inn, kan du se bort fra denne "
                    "meldingen.\n", to=epost)
    except Exception:                                  # noqa: BLE001
        # Vi roeper ikke at utsendingen feilet - det ville ogsaa vaert et
        # signal om adressen. Feilen logges, brukeren ber om ny kode.
        log.exception("Kunne ikke sende innloggingskode")
    return {"status": "sendt"}


@router.post("/email/verify")
def verifiser_kode(body: KodeIn, user_agent: str = Header(default=""),
                   s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    epost = body.email.lower()
    naa = utcnow()

    rad = s.scalar(select(EmailLoginCode).where(
        EmailLoginCode.email == epost,
        EmailLoginCode.consumed_at.is_(None)).order_by(EmailLoginCode.id.desc()))
    if rad is None or as_utc(rad.expires_at) < naa:
        raise HTTPException(401, "Koden er ugyldig eller utloept.")
    if rad.attempts >= cfg.email_code_max_attempts:
        raise HTTPException(429, "For mange forsoek. Be om en ny kode.")

    rad.attempts += 1
    if not tokens.koder_er_like(body.code, rad.code_hash):
        s.commit()
        raise HTTPException(401, "Koden er ugyldig eller utloept.")

    rad.consumed_at = naa
    ident = oidc.Identitet(provider=Provider.email, subject=epost,
                           email=epost, email_verifisert=True)
    user, er_ny = _finn_eller_lag_bruker(s, ident)
    ut = _start_oekt(s, cfg, user, user_agent)
    ut["is_new"] = er_ny
    return ut


# --------------------------------------------------------------------
# Fornyelse og utlogging
# --------------------------------------------------------------------

class RefreshIn(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=512)


@router.post("/refresh")
def forny(body: RefreshIn, user_agent: str = Header(default=""),
          s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    hash_ = tokens.hash_token(body.refresh_token)
    oekt = s.scalar(select(AuthSession).where(AuthSession.refresh_hash == hash_))
    naa = utcnow()

    if oekt is None:
        raise HTTPException(401, "Ugyldig refresh-token.")
    if oekt.revoked_at is not None:
        # Et allerede brukt token dukket opp igjen. Vi kan ikke skille en
        # kopi paa avveie fra et forsoek som ble kjoert to ganger, saa vi
        # velger det trygge: alle oektene til brukeren tilbakekalles.
        log.warning("Gjenbrukt refresh-token for bruker %s - tilbakekaller alt",
                    oekt.user_id)
        for annen in s.scalars(select(AuthSession).where(
                AuthSession.user_id == oekt.user_id,
                AuthSession.revoked_at.is_(None))):
            annen.revoked_at = naa
        s.commit()
        raise HTTPException(401, "Oekten er tilbakekalt. Logg inn paa nytt.")
    if as_utc(oekt.expires_at) < naa:
        raise HTTPException(401, "Oekten er utloept. Logg inn paa nytt.")

    user = s.get(User, oekt.user_id)
    if user is None or user.deleted_at is not None:
        raise HTTPException(401, "Kontoen finnes ikke lenger.")

    # Rotasjon: den gamle raden merkes brukt og erstattes av en ny.
    oekt.revoked_at = naa
    oekt.last_used_at = naa
    return _start_oekt(s, cfg, user, user_agent or oekt.user_agent)


@router.post("/logout", status_code=204)
def logg_ut(body: RefreshIn, s: OrmSession = Depends(db)) -> None:
    """
    Idempotent: et ukjent eller allerede utlogget token gir ogsaa 204.
    Utlogging skal aldri feile - da sitter klienten igjen med et token den
    tror er gyldig.
    """
    oekt = s.scalar(select(AuthSession).where(
        AuthSession.refresh_hash == tokens.hash_token(body.refresh_token)))
    if oekt is not None and oekt.revoked_at is None:
        oekt.revoked_at = utcnow()
        s.commit()
