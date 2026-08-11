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
# Svarmodeller
#
# Finnes for at contracts/openapi.json skal beskrive SVARET og ikke bare
# forespoerselen. Bare rutene klienten faktisk kaller har fått dem; resten av
# API-et returnerer `dict` og staar under AAPNE_PUNKTER ÅP-B10.
# --------------------------------------------------------------------

class TokenPar(BaseModel):
    """Svaret fra /refresh. Alle andre endepunkter tar `access_token` som
    `Authorization: Bearer <token>`."""
    access_token: str
    refresh_token: str = Field(description="Ugjennomsiktig. Roteres ved hver "
                                          "bruk; to parallelle fornyelser med "
                                          "samme token tolkes som tyveri og "
                                          "tilbakekaller alle oekter.")
    token_type: str = Field(examples=["Bearer"])
    expires_in: int = Field(description="Sekunder til access-tokenet utloeper.")
    user_id: str
    public_id: str = Field(description="Den korte ID-en venner soeker opp, "
                                       "f.eks. BF-7Q4K-9F2M.")
    display_name: str
    email: str | None = Field(
        default=None,
        description="Adressen brukeren logget inn SOM - fra identiteten oekten "
                    "ble startet med, ikke fra kontoen. Ved kontosammenslaaing "
                    "kan de vaere ulike. `null` for Apple med skjult adresse, "
                    "og for oekter startet foer feltet fantes. Baeres uendret "
                    "gjennom /refresh.")


class TokenParNyBruker(TokenPar):
    """Svaret fra de tre innloggingsveiene: tokenparet pluss om kontoen ble
    opprettet av dette kallet."""
    is_new: bool


class KodeSendt(BaseModel):
    """
    Svaret fra /email/start. **Alltid 202**, ogsaa for en ukjent adresse - et
    svar som skilte kjent fra ukjent ville gjort endepunktet til et
    oppslagsverk over hvem som bruker appen.

    Fristene ligger her nettopp for at klienten ikke skal hardkode dem.
    """
    status: str = Field(examples=["sendt"])
    resend_after_seconds: int = Field(
        description="Sperrefrist for «send ny kode». Innen fristen svarer "
                    "endepunktet 429 med Retry-After.")
    expires_in_minutes: int = Field(description="Levetid for koden.")


# --------------------------------------------------------------------
# Felles
# --------------------------------------------------------------------

def _foerste_navn(ident: oidc.Identitet) -> str:
    """
    Et foerste visningsnavn saa kontoen ikke staar tom. Brukeren endrer det i
    profilen; her handler det bare om aa ha noe lesbart aa vise.

    Rekkefoelgen er `name` fra ID-TOKENET foerst, deretter lokaldelen av
    e-postadressen. Navnet er signaturverifisert paa lik linje med `sub` og
    `email` - det er derfor vi kan bruke det, og derfor klienten IKKE sender
    sitt eget navn fra Credential Manager: det ville byttet en verifisert kilde
    mot en klient-oppgitt, i det ene feltet som alltid deles med venner (§3).

    Navnet modereres som ethvert annet visningsnavn, med den ekte ordlista.
    En leverandoer garanterer ikke at brukeren har valgt et navn vi vil vise.
    """
    kandidat = (ident.navn or "").strip()
    if not kandidat:
        kandidat = ((ident.email or "").split("@")[0]
                    .replace(".", " ").replace("_", " "))
    navn, status, _ = moderation.review(
        kandidat or "Skytter", settings().display_name_blocklist_list)
    return navn if status == NameStatus.approved else "Skytter"


def _ny_bruker(s: OrmSession, ident: oidc.Identitet) -> User:
    # Statusen MAA settes eksplisitt. `_navn_fra_epost` lagrer bare navn som
    # har passert moderasjonen - ellers reservenavnet - saa navnet ER godkjent,
    # men standardverdien paa kolonnen er `pending`. Uten dette viste
    # sharing.friend_view «Navn under vurdering» til venner og lagkamerater for
    # hver eneste nye konto, helt til brukeren tilfeldigvis lagret profilen sin
    # paa nytt; PUT /v1/profile var det eneste stedet statusen ble satt.
    user = User(public_id=ids.generate(), display_name=_foerste_navn(ident),
                display_name_status=NameStatus.approved,
                display_name_reviewed_at=utcnow())
    s.add(user)
    # MAA skylles foer radene som peker paa den: det finnes ingen
    # relationship() mellom User og SharingPreference, saa SQLAlchemy har
    # ingen rekkefoelge aa sortere etter. Se CLAUDE.md paragraf 7.5.
    s.flush()
    s.add(SharingPreference(user_id=user.id))
    return user


def _finn_eller_lag_bruker(s: OrmSession,
                           ident: oidc.Identitet) -> tuple[User, bool, AuthIdentity]:
    """
    Returnerer (bruker, er_ny, identitetsraden som ble brukt).

    Identitetsraden foelger med ut fordi oekten skal huske hvilken den ble
    startet med - se `_start_oekt`.
    """
    rad = s.scalar(select(AuthIdentity).where(
        AuthIdentity.provider == ident.provider,
        AuthIdentity.subject == ident.subject))
    if rad is not None:
        # Leverandoeren kan ha faatt ny adresse siden sist.
        if ident.email and rad.email != ident.email:
            rad.email = ident.email
        return rad.user, False, rad

    user: User | None = None
    if ident.email and ident.email_verifisert:
        # Samme verifiserte adresse hos en annen leverandoer = samme person.
        eksisterende = s.scalar(select(AuthIdentity).where(
            AuthIdentity.email == ident.email).order_by(AuthIdentity.id))
        if eksisterende is not None:
            user = eksisterende.user

    er_ny = user is None
    if user is None:
        user = _ny_bruker(s, ident)
    rad = AuthIdentity(user_id=user.id, provider=ident.provider,
                       subject=ident.subject, email=ident.email)
    s.add(rad)
    # Flush for aa faa `rad.id` - oekten under peker paa den.
    s.flush()
    return user, er_ny, rad


def _start_oekt(s: OrmSession, cfg: Settings, user: User, user_agent: str,
                identitet: AuthIdentity | None = None) -> dict:
    """
    Utsteder et tokenpar og lagrer oekten.

    `identitet` er raden innloggingen kom gjennom. Den lagres paa oekten fordi
    `email` i svaret skal si hva brukeren logget inn SOM - og ved fornyelse
    finnes ikke ID-tokenet lenger. Uten den ville `/refresh` maattet gjette.

    Hvorfor ikke bare la klienten lese `email` ut av ID-tokenet: kontoen kan
    vaere slaatt sammen paa verifisert e-post (se modulkommentaren), og da er
    adressen kontoen er knyttet til ikke noedvendigvis den man nettopp logget
    inn med. Skjermen ville loeyet i akkurat det tilfellet den finnes for.
    """
    access, levetid = tokens.lag_access_token(cfg, user.id)
    raa, hash_ = tokens.nytt_refresh_token()
    s.add(AuthSession(user_id=user.id, refresh_hash=hash_,
                      user_agent=(user_agent or "")[:200],
                      identity_id=identitet.id if identitet else None,
                      expires_at=utcnow() + timedelta(days=cfg.refresh_token_ttl_days)))
    user.last_seen_at = utcnow()
    s.commit()
    return {"access_token": access, "refresh_token": raa,
            "token_type": "Bearer", "expires_in": levetid,
            "user_id": user.id, "public_id": user.public_id,
            "display_name": user.display_name,
            # None for oekter startet foer kolonnen fantes, og for en identitet
            # uten adresse (Apple med skjult e-post). Klienten maa taale det.
            "email": identitet.email if identitet else None}


# --------------------------------------------------------------------
# Google / Apple
# --------------------------------------------------------------------

class IdTokenIn(BaseModel):
    id_token: str = Field(min_length=1, max_length=8192)


@router.post("/google", response_model=TokenParNyBruker)
def logg_inn_google(body: IdTokenIn,
                    user_agent: str = Header(default=""),
                    s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    ident = oidc.verifiser_google(cfg, body.id_token)
    user, er_ny, identitet = _finn_eller_lag_bruker(s, ident)
    ut = _start_oekt(s, cfg, user, user_agent, identitet)
    ut["is_new"] = er_ny
    return ut


@router.post("/apple", response_model=TokenParNyBruker)
def logg_inn_apple(body: IdTokenIn,
                   user_agent: str = Header(default=""),
                   s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    ident = oidc.verifiser_apple(cfg, body.id_token)
    user, er_ny, identitet = _finn_eller_lag_bruker(s, ident)
    ut = _start_oekt(s, cfg, user, user_agent, identitet)
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


@router.post("/email/start", status_code=202, response_model=KodeSendt)
def send_kode(body: EpostIn, s: OrmSession = Depends(db)) -> dict:
    """
    Svarer ALLTID 202, uansett om adressen finnes fra foer. Et svar som skilte
    mellom «kjent» og «ukjent» ville gjort endepunktet til et oppslagsverk
    over hvem som bruker appen.
    """
    cfg = settings()
    tokens.krev_hemmelighet(cfg)
    epost = body.email.lower()
    naa = utcnow()

    # Ratebegrensning i BASEN, ikke i minnet: Fly kjoerer to maskiner, og en
    # teller per prosess ville i praksis doblet grensen.
    grense = naa - timedelta(hours=1)
    rader = list(s.scalars(select(EmailLoginCode).where(
        EmailLoginCode.email == epost, EmailLoginCode.created_at >= grense)
        .order_by(EmailLoginCode.id.desc())))
    if len(rader) >= cfg.email_code_rate_per_hour:
        raise HTTPException(429, "For mange koder er sendt til denne adressen. "
                                 "Prøv igjen om en time.")

    # Sperrefristen paa «send ny kode». Klienten teller ned paa den samme
    # verdien, men den timeren er bekvemmelighet - dette er vernet. Uten den
    # kan hvem som helst sende ti e-poster i slengen til en fremmed adresse,
    # og klient-timeren er da bare noe som kan klikkes bort.
    if rader:
        gaar_igjen = cfg.email_code_resend_cooldown_seconds - int(
            (naa - as_utc(rader[0].created_at)).total_seconds())
        if gaar_igjen > 0:
            raise HTTPException(429, "Vent litt før du ber om en ny kode.",
                                headers={"Retry-After": str(gaar_igjen)})

    kode, hash_ = tokens.ny_engangskode()
    s.add(EmailLoginCode(email=epost, code_hash=hash_,
                         expires_at=naa + timedelta(minutes=cfg.email_code_ttl_minutes)))
    s.commit()

    try:
        mailer.send(cfg, "Innloggingskode til Bestefar",
                    f"Koden din er {kode}.\n\n"
                    f"Den er gyldig i {cfg.email_code_ttl_minutes} minutter.\n"
                    "Har du ikke bedt om å logge inn, kan du se bort fra denne "
                    "meldingen.\n", to=epost)
    except Exception:                                  # noqa: BLE001
        # Vi roeper ikke at utsendingen feilet - det ville ogsaa vaert et
        # signal om adressen. Feilen logges, brukeren ber om ny kode.
        log.exception("Kunne ikke sende innloggingskode")
    # Verdiene ligger i svaret saa klienten slipper aa hardkode dem: nedtellingen
    # paa «Send ny kode» og «koden er gyldig i N minutter» skal si det samme som
    # serveren faktisk haandhever, ogsaa etter at verdiene er endret her.
    return {"status": "sendt",
            "resend_after_seconds": cfg.email_code_resend_cooldown_seconds,
            "expires_in_minutes": cfg.email_code_ttl_minutes}


@router.post("/email/verify", response_model=TokenParNyBruker)
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
        raise HTTPException(401, "Koden er ugyldig eller utløpt.")
    if rad.attempts >= cfg.email_code_max_attempts:
        raise HTTPException(429, "For mange forsøk. Be om en ny kode.")

    rad.attempts += 1
    if not tokens.koder_er_like(body.code, rad.code_hash):
        s.commit()
        raise HTTPException(401, "Koden er ugyldig eller utløpt.")

    rad.consumed_at = naa
    ident = oidc.Identitet(provider=Provider.email, subject=epost,
                           email=epost, email_verifisert=True)
    user, er_ny, identitet = _finn_eller_lag_bruker(s, ident)
    ut = _start_oekt(s, cfg, user, user_agent, identitet)
    ut["is_new"] = er_ny
    return ut


# --------------------------------------------------------------------
# Fornyelse og utlogging
# --------------------------------------------------------------------

class RefreshIn(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=512)


@router.post("/refresh", response_model=TokenPar)
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
        raise HTTPException(401, "Økten er tilbakekalt. Logg inn på nytt.")
    if as_utc(oekt.expires_at) < naa:
        raise HTTPException(401, "Økten er utløpt. Logg inn på nytt.")

    user = s.get(User, oekt.user_id)
    if user is None or user.deleted_at is not None:
        raise HTTPException(401, "Kontoen finnes ikke lenger.")

    # Rotasjon: den gamle raden merkes brukt og erstattes av en ny.
    oekt.revoked_at = naa
    oekt.last_used_at = naa
    # Identiteten baeres videre: fornyelse skal ikke endre hva klienten viser
    # under «Logget inn som».
    return _start_oekt(s, cfg, user, user_agent or oekt.user_agent,
                       oekt.identity)


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
