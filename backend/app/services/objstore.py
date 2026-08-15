"""
Objektlagring for feilanalyse-bilder (backend_spec §6, §0.1).

§6 er tydelig: bildene ligger i S3-kompatibel objektlagring - Cloudflare R2 -
og bare metadata ligger relasjonelt. Denne modulen er den eneste veien dit.

**Hvorfor ikke boto3.** Vi trenger noeyaktig tre operasjoner paa ett objekt:
PUT, GET og DELETE. boto3 drar med seg botocore med datafiler for hele
AWS-tjenestekatalogen. Samme avveining som i push.py, der google-auth ble
droppet til fordel for PyJWT vi allerede hadde. SigV4-signeringen er de foerste
femti linjene under, og den er dekket av tester.

**Ikke konfigurert** (tom R2_BUCKET eller manglende noekler) => `er_konfigurert`
er False, og `routers/failed_analyses.py` svarer 503 uten aa lese kroppen.
Bildekolonnen i basen er fjernet (B-49), saa det finnes ikke et annet sted aa
gjoere av donasjonen. `/health` under «bilder» sier hvilken tilstand maskinen er
i - det er stedet aa spoerre.

**En feil herfra blir aldri en stille fallback til basen.** Da ville en feilsatt
noekkel sett ut som «R2 er ikke konfigurert», bildene lagt seg i basen igjen, og
ingen oppdaget det - nettopp fellen `BackupKeys.resolve` gikk i hos klienten
(rot-CLAUDE.md §7.3). Konfigurert + feilet opplasting = 503 oppover, og klienten
proever igjen; koen i `dev_uploads/` er intakt til den lykkes.
"""
import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timezone
from urllib.parse import quote, urlparse

import httpx

from ..config import Settings

log = logging.getLogger(__name__)

_ALG = "AWS4-HMAC-SHA256"
_TJENESTE = "s3"

# sha256 av tom kropp - kreves som x-amz-content-sha256 paa GET og DELETE.
TOM_SHA256 = hashlib.sha256(b"").hexdigest()


class LagringFeilet(RuntimeError):
    """Opplasting/nedlasting naadde ikke fram, eller ble avvist av R2."""


# Magiske byte -> (content-type, filendelse). Klienten sender JPEG; PNG og WebP
# staar her fordi de er de to andre en telefon kan produsere, ikke fordi noen
# ber om dem.
_BILDETYPER = (
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
)


def bildetype(data: bytes) -> tuple[str, str] | None:
    """
    (content-type, endelse) ut fra de FOERSTE BYTENE, eller None.

    Innholdet avgjoer, ikke `Content-Type` fra avsenderen (B-46). Ligger her og
    ikke i routeren fordi flyttingen av gamle rader (services/legacy_bilder.py)
    trenger den samme avgjoerelsen - der finnes det ingen multipart aa spoerre.
    """
    for magi, ctype, endelse in _BILDETYPER:
        if data.startswith(magi):
            return ctype, endelse
    # RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    return None


def _hmac(noekkel: bytes, melding: str) -> bytes:
    return hmac.new(noekkel, melding.encode("utf-8"), hashlib.sha256).digest()


def _signeringsnoekkel(hemmelighet: str, dato: str, region: str) -> bytes:
    k = _hmac(("AWS4" + hemmelighet).encode("utf-8"), dato)
    k = _hmac(k, region)
    k = _hmac(k, _TJENESTE)
    return _hmac(k, "aws4_request")


def kanonisk_forespoersel(metode: str, sti: str, hoder: dict[str, str],
                          kropp_sha256: str) -> tuple[str, str]:
    """
    (kanonisk forespoersel, signerte hodenavn) - SigV4 «Task 1».

    Skilt ut som egen funksjon fordi den er det eneste her som kan vaere feil
    paa en maate R2 svarer 403 paa uten aa si hvorfor. Testen sammenligner den
    mot en haandregnet streng.
    """
    navn = sorted(n.lower() for n in hoder)
    kanoniske = "".join(f"{n}:{hoder[n].strip()}\n" for n in navn)
    signerte = ";".join(navn)
    # Ingen spoerringsstreng: vi adresserer ett objekt direkte.
    kanonisk = "\n".join([metode, sti, "", kanoniske, signerte, kropp_sha256])
    return kanonisk, signerte


def _autorisasjon(cfg: Settings, metode: str, sti: str, hoder: dict[str, str],
                  kropp_sha256: str, amz_dato: str) -> str:
    dato = amz_dato[:8]
    kanonisk, signerte = kanonisk_forespoersel(metode, sti, hoder, kropp_sha256)
    omfang = f"{dato}/{cfg.r2_region}/{_TJENESTE}/aws4_request"
    aa_signere = "\n".join([
        _ALG, amz_dato, omfang,
        hashlib.sha256(kanonisk.encode("utf-8")).hexdigest(),
    ])
    signatur = hmac.new(_signeringsnoekkel(cfg.r2_secret_access_key, dato,
                                           cfg.r2_region),
                        aa_signere.encode("utf-8"), hashlib.sha256).hexdigest()
    return (f"{_ALG} Credential={cfg.r2_access_key_id}/{omfang}, "
            f"SignedHeaders={signerte}, Signature={signatur}")


def er_konfigurert(cfg: Settings) -> bool:
    return bool(cfg.r2_endpoint and cfg.r2_bucket
                and cfg.r2_access_key_id and cfg.r2_secret_access_key)


def backend_name(cfg: Settings) -> str:
    """Til /health, paa samme form som mailer/push.backend_name."""
    return "r2" if er_konfigurert(cfg) else "database"


def objektnoekkel(fa_id: int, tag: str, endelse: str,
                  naa: datetime | None = None) -> str:
    """
    `feilanalyse/{tag}/{aaaa/mm/dd}/{id}-{tilfeldig}{endelse}`.

    Datoen foerst i stien saa et helt doegn kan hentes eller ryddes med ett
    prefiks. Det tilfeldige leddet gjoer noekkelen ugjettbar: bucketen er
    privat i dag, men en noekkel som er `id` alene ville vaert et oppslagsverk
    den dagen noen lager en signert lenke.
    """
    d = (naa or datetime.now(timezone.utc)).strftime("%Y/%m/%d")
    return f"feilanalyse/{tag}/{d}/{fa_id}-{secrets.token_hex(8)}{endelse}"


def _sti(cfg: Settings, noekkel: str) -> str:
    # Sti-stil (endepunktet er kontospesifikt, bucketen er foerste ledd).
    # quote lar bokstaver, tall og «-._~» staa; skraastrekene i noekkelen er
    # katalogskiller og skal ikke kodes.
    return f"/{quote(cfg.r2_bucket, safe='')}/{quote(noekkel, safe='/')}"


def _kall(cfg: Settings, metode: str, noekkel: str, kropp: bytes = b"",
          content_type: str = "") -> httpx.Response:
    if not er_konfigurert(cfg):
        raise LagringFeilet("R2 er ikke konfigurert")

    sti = _sti(cfg, noekkel)
    amz_dato = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kropp_sha = hashlib.sha256(kropp).hexdigest() if kropp else TOM_SHA256
    hoder = {
        "host": urlparse(cfg.r2_endpoint).netloc,
        "x-amz-content-sha256": kropp_sha,
        "x-amz-date": amz_dato,
    }
    if content_type:
        hoder["content-type"] = content_type
    hoder["authorization"] = _autorisasjon(cfg, metode, sti, hoder, kropp_sha,
                                           amz_dato)

    url = cfg.r2_endpoint.rstrip("/") + sti
    try:
        r = httpx.request(metode, url, headers=hoder, content=kropp,
                          timeout=cfg.r2_timeout_seconds)
    except httpx.HTTPError as exc:
        raise LagringFeilet(f"{metode} mot R2 naadde ikke fram: {exc}") from exc
    if r.status_code >= 300:
        # Kroppen er XML fra R2 og sier hvilken av de tre vanlige feilene det
        # er: SignatureDoesNotMatch, NoSuchBucket eller AccessDenied.
        raise LagringFeilet(f"{metode} mot R2 svarte {r.status_code}: "
                            f"{r.text[:300]}")
    return r


def legg(cfg: Settings, noekkel: str, data: bytes, content_type: str) -> None:
    """Laster opp ett objekt. Kaster `LagringFeilet` - svelger ingenting."""
    _kall(cfg, "PUT", noekkel, data, content_type)
    log.info("Lastet opp %d byte til R2 (%s)", len(data), noekkel)


def hent(cfg: Settings, noekkel: str) -> bytes:
    """Henter ett objekt. Brukes av tools/r2_check.py."""
    return _kall(cfg, "GET", noekkel).content


def slett(cfg: Settings, noekkel: str) -> None:
    """Sletter ett objekt. Brukes av tools/r2_check.py."""
    _kall(cfg, "DELETE", noekkel)
