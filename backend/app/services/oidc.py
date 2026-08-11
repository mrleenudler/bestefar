"""
Verifisering av ID-tokens fra Google og Apple (backend_spec §1).

Klienten gjoer selve innloggingen med leverandoerens SDK og sender oss
ID-tokenet. Vi verifiserer det her - vi stoler ALDRI paa `sub` eller `email`
uten aa ha sjekket signaturen mot leverandoerens noekler foerst; ellers kunne
hvem som helst logge inn som hvem som helst ved aa sende en JSON-streng.

Tre ting maa stemme, og alle tre er noedvendige:
  signaturen  mot leverandoerens JWKS (noeklene roteres, derfor hentes de over
              nett og caches av PyJWKClient)
  `iss`       at tokenet faktisk kommer fra Google/Apple
  `aud`       at det er utstedt til VAAR app. Uten denne sjekken ville et
              gyldig Google-token utstedt til en helt annen app gitt tilgang.

Ingen klient-ID konfigurert => 503. Vi gjetter ikke.
"""
import logging
from dataclasses import dataclass

import jwt
from fastapi import HTTPException

from ..config import Settings
from ..models import Provider

log = logging.getLogger(__name__)

GOOGLE_JWKS = "https://www.googleapis.com/oauth2/v3/certs"
# Google har historisk brukt begge formene i `iss`.
GOOGLE_ISSUERS = ("https://accounts.google.com", "accounts.google.com")

APPLE_JWKS = "https://appleid.apple.com/auth/keys"
APPLE_ISSUER = "https://appleid.apple.com"

_jwks_klienter: dict[str, jwt.PyJWKClient] = {}


@dataclass(frozen=True)
class Identitet:
    provider: Provider
    subject: str
    email: str | None
    email_verifisert: bool
    # Navnet fra de VERIFISERTE kravene i ID-tokenet, ikke fra klienten.
    # Klienten har navnet fra Credential Manager, men aa sende det derfra ville
    # byttet en signaturverifisert kilde mot en klient-oppgitt - og
    # visningsnavnet er det ene feltet som alltid deles med venner (§3).
    # Tomt for e-postinnlogging og som regel for Apple, som ikke legger `name`
    # i tokenet.
    navn: str | None = None


def _jwks(url: str) -> jwt.PyJWKClient:
    """Én klient per URL - den cacher noeklene, saa vi henter dem ikke per
    innlogging."""
    if url not in _jwks_klienter:
        _jwks_klienter[url] = jwt.PyJWKClient(url, cache_keys=True)
    return _jwks_klienter[url]


def _er_verifisert(krav: dict) -> bool:
    """Google sender bool, Apple sender strengen "true". Begge maa taales."""
    verdi = krav.get("email_verified")
    return verdi is True or (isinstance(verdi, str) and verdi.lower() == "true")


def _verifiser(token: str, jwks_url: str, issuers: tuple[str, ...],
               audiences: list[str], navn: str) -> dict:
    if not audiences:
        raise HTTPException(503, f"Innlogging med {navn} er ikke konfigurert.")
    try:
        noekkel = _jwks(jwks_url).get_signing_key_from_jwt(token)
        krav = jwt.decode(token, noekkel.key, algorithms=["RS256", "ES256"],
                          audience=audiences, issuer=list(issuers),
                          options={"require": ["exp", "iat", "sub"]})
    except jwt.PyJWKClientError as exc:
        # Naadde ikke leverandoeren, eller ukjent noekkel-ID. Dette er VAAR
        # feil eller leverandoerens, ikke klientens - derfor 503 og ikke 401,
        # saa klienten proever igjen i stedet for aa logge brukeren ut.
        log.warning("Kunne ikke hente signeringsnoekkel fra %s: %s", navn, exc)
        raise HTTPException(503, f"Fikk ikke kontakt med {navn}. Prøv igjen.") from exc
    except jwt.InvalidTokenError as exc:
        log.info("Avvist %s-token: %s", navn, exc)
        raise HTTPException(401, f"Ugyldig {navn}-token.") from exc
    return krav


def verifiser_google(cfg: Settings, id_token: str) -> Identitet:
    krav = _verifiser(id_token, GOOGLE_JWKS, GOOGLE_ISSUERS,
                      cfg.google_client_id_list, "Google")
    return Identitet(provider=Provider.google, subject=str(krav["sub"]),
                     email=krav.get("email"),
                     email_verifisert=_er_verifisert(krav),
                     navn=krav.get("name"))


def verifiser_apple(cfg: Settings, id_token: str) -> Identitet:
    """
    Apple gir e-postadressen KUN ved foerste innlogging, og med «Skjul
    e-postadressen min» er den en videresendingsadresse. Begge deler er greit:
    `sub` er det stabile, og den faar vi hver gang.
    """
    krav = _verifiser(id_token, APPLE_JWKS, (APPLE_ISSUER,),
                      cfg.apple_client_id_list, "Apple")
    # Apple legger normalt ikke `name` i ID-tokenet - navnet kommer i
    # autorisasjonssvaret ved aller foerste innlogging, som klienten haandterer.
    # `krav.get("name")` staar likevel her: er den der, er den verifisert.
    return Identitet(provider=Provider.apple, subject=str(krav["sub"]),
                     email=krav.get("email"),
                     email_verifisert=_er_verifisert(krav),
                     navn=krav.get("name"))
