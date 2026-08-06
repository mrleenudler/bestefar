"""
Push-varsler via Firebase Cloud Messaging, HTTP v1 (backend_spec §11, fase 8).

Forholdet til meldingskoeen er det viktigste her, og det er med vilje skjevt:

  meldingskoeen (§11)  GARANTIEN. Raden ligger i basen til klienten har
                       kvittert. Feiler push, mister brukeren ingenting -
                       meldingen kommer ved neste app-aapning.
  push                 BEKVEMMELIGHETEN. Naar brukeren mens appen er lukket.

Derfor: push feiler ALDRI oppover. Alt her er best effort, og en feil logges
og svelges. Motsatt vei ville en nede FCM-tjeneste gjort det umulig aa bytte
lagnavn.

Ikke konfigurert (tom FCM_SERVICE_ACCOUNT_JSON) => alt logges bare. Da kan
resten av fase 8 bygges og deployes foer Firebase-prosjektet er koblet paa,
paa samme maate som mailer.py.

Hvorfor ikke google-auth-biblioteket: vi trenger noeyaktig én ting derfra -
et OAuth-token fra en tjenestekonto - og har allerede PyJWT med krypto for
§1. Ett avhengighetstre mindre aa holde oppdatert.
"""
import base64
import json
import logging
import time

import httpx
import jwt

from ..config import Settings

log = logging.getLogger(__name__)

TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE = "https://www.googleapis.com/auth/firebase.messaging"
GRANT = "urn:ietf:params:oauth:grant-type:jwt-bearer"

# Tokenet varer en time. Vi fornyer litt foer utloep saa et kall ikke rekker
# aa treffe grensen mens det er underveis.
_MARGIN = 120

# (access_token, utloeper_paa). Prosess-lokal - Fly kjoerer to maskiner, og de
# henter hvert sitt token. Det er helt greit; tokenet er ikke en delt ressurs.
_token_cache: tuple[str, float] | None = None


def _konto(cfg: Settings) -> dict | None:
    """
    Tjenestekontoen, enten som raa JSON eller base64-kodet.

    Begge godtas med vilje. Raa JSON er det enkleste naar man kan lime inn,
    men noekkelfilen er flerlinjes og inneholder «private_key» med
    \\n-sekvenser - i PowerShell blir sitering av den en felle. Base64 er
    én linje uten spesialtegn og overlever enhver kommandolinje.
    """
    raa = cfg.fcm_service_account_json.strip()
    if not raa:
        return None
    if not raa.startswith("{"):
        try:
            raa = base64.b64decode(raa, validate=True).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            log.error("FCM_SERVICE_ACCOUNT_JSON er verken JSON eller gyldig "
                      "base64 - push er av.")
            return None
    try:
        return json.loads(raa)
    except ValueError:
        log.error("FCM_SERVICE_ACCOUNT_JSON er ikke gyldig JSON - push er av.")
        return None


def backend_name(cfg: Settings) -> str:
    """Til /health, paa samme form som mailer.backend_name."""
    return "fcm" if _konto(cfg) else "log"


def project_id(cfg: Settings) -> str:
    konto = _konto(cfg)
    return cfg.fcm_project_id or (konto or {}).get("project_id", "")


def nullstill_token_cache() -> None:
    """Bare for testene - konfigurasjonen kan byttes mellom to tester."""
    global _token_cache
    _token_cache = None


def _access_token(cfg: Settings, konto: dict) -> str | None:
    global _token_cache
    naa = time.time()
    if _token_cache and _token_cache[1] > naa:
        return _token_cache[0]

    paastand = {
        "iss": konto["client_email"],
        "scope": SCOPE,
        "aud": TOKEN_URL,
        "iat": int(naa),
        "exp": int(naa + 3600),
    }
    signert = jwt.encode(paastand, konto["private_key"], algorithm="RS256")
    r = httpx.post(TOKEN_URL, timeout=cfg.push_timeout_seconds,
                   data={"grant_type": GRANT, "assertion": signert})
    r.raise_for_status()
    data = r.json()
    token = data["access_token"]
    _token_cache = (token, naa + int(data.get("expires_in", 3600)) - _MARGIN)
    return token


def _melding(push_token: str, title: str, body: str, data: dict) -> dict:
    return {"message": {
        "token": push_token,
        "notification": {"title": title, "body": body},
        # Alle verdier MAA vaere strenger - FCM avviser tall og null.
        "data": {k: str(v) for k, v in data.items() if v is not None},
        "android": {"priority": "high"},
    }}


# FCM sier fra paa to maater at et token er doedt. Begge betyr «avinstallert
# eller byttet ut», og raden skal bort - ellers vokser devices-tabellen med
# adresser vi aldri naar.
_DODE_KODER = {"UNREGISTERED", "INVALID_ARGUMENT", "NOT_FOUND"}


def send(cfg: Settings, push_tokens: list[str], title: str, body: str,
         data: dict | None = None) -> tuple[int, list[str]]:
    """
    Sender ett varsel til hver av `push_tokens`.

    Returnerer (antall sendt, doede tokens). De doede skal slettes av
    kalleren - denne modulen roerer ikke databasen, saa den kan testes uten.

    Kaster aldri. Se modulkommentaren.
    """
    data = data or {}
    konto = _konto(cfg)
    if konto is None:
        log.info("Push (kun logg) til %d enhet(er) | %s: %s",
                 len(push_tokens), title, body)
        return 0, []

    prosjekt = project_id(cfg)
    if not prosjekt:
        log.error("Fant ingen FCM-prosjekt-ID - push er av.")
        return 0, []

    try:
        token = _access_token(cfg, konto)
    except Exception:                                      # noqa: BLE001
        log.exception("Fikk ikke OAuth-token til FCM")
        return 0, []

    url = f"https://fcm.googleapis.com/v1/projects/{prosjekt}/messages:send"
    hoder = {"Authorization": f"Bearer {token}"}
    frist = time.monotonic() + cfg.push_budget_seconds
    sendt, doede = 0, []

    # Én klient for hele runden: HTTP v1 tar én mottaker per kall, saa et lag
    # paa tjue medlemmer blir tjue kall. Uten gjenbrukt forbindelse ville hvert
    # av dem betalt for ny TLS-handshake.
    with httpx.Client(timeout=cfg.push_timeout_seconds, headers=hoder) as klient:
        for pt in push_tokens:
            if time.monotonic() > frist:
                # Budsjettet er brukt opp. Dette er IKKE et datatap: koeen
                # ligger allerede i basen, og resten faar meldingen ved neste
                # app-aapning.
                log.warning("Push-budsjettet brukt opp - %d enhet(er) fikk "
                            "ikke varsel (meldingen ligger i koeen).",
                            len(push_tokens) - sendt - len(doede))
                break
            try:
                r = klient.post(url, json=_melding(pt, title, body, data))
                if r.status_code == 200:
                    sendt += 1
                    continue
                feil = r.json().get("error", {})
                kode = feil.get("status", "")
                detalj = feil.get("details", [{}])
                fcm_kode = next((d.get("errorCode", "") for d in detalj
                                 if isinstance(d, dict)), "")
                if kode in _DODE_KODER or fcm_kode in _DODE_KODER:
                    doede.append(pt)
                else:
                    log.warning("FCM avviste et varsel (%s): %s",
                                r.status_code, kode or r.text[:200])
            except Exception:                              # noqa: BLE001
                log.exception("Push feilet mot én enhet")

    return sendt, doede
