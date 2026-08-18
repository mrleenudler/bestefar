"""
Objektlagring for feilanalyse-bilder (backend_spec §6, §0.1).

§6 er tydelig: bildene ligger i S3-kompatibel objektlagring - Cloudflare R2 -
og bare metadata ligger relasjonelt. Denne modulen er den eneste veien dit.

**Hvorfor ikke boto3.** Vi trenger noeyaktig tre operasjoner paa ett objekt:
PUT, GET og DELETE. boto3 drar med seg botocore med datafiler for hele
AWS-tjenestekatalogen. Samme avveining som i push.py, der google-auth ble
droppet til fordel for PyJWT vi allerede hadde. SigV4-signeringen er de foerste
femti linjene under, og den er dekket av tester.

**Ikke konfigurert** (tom R2_BUCKET eller manglende noekler) => `kan_brukes`
er False, og `routers/failed_analyses.py` svarer 503 uten aa lese kroppen.
Bildekolonnen i basen er fjernet (B-49), saa det finnes ikke et annet sted aa
gjoere av donasjonen. `/health` under «bilder» sier hvilken tilstand maskinen er
i - det er stedet aa spoerre.

**Ikke konfigurert og feilkonfigurert er to tilstander** (B-51). En verdi som er
satt er ikke en verdi som virker, og `tilstand()` sier hvilken av delene det er.
Se den funksjonen for hvorfor forskjellen maatte fram.

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
from dataclasses import dataclass
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
    """
    Opplasting/nedlasting naadde ikke fram, eller ble avvist av R2.

    `status` er HTTP-statusen fra R2 naar det var et svar, ellers None (kallet
    naadde aldri fram). Den finnes fordi 404 betyr noe annet enn resten: «dette
    objektet er ikke der» er et svar, mens 403 og 500 er feil. En kaller som
    maa skille dem, skal slippe aa lete i feilteksten etter et tall.
    """

    def __init__(self, melding: str, status: int | None = None) -> None:
        super().__init__(melding)
        self.status = status


@dataclass(frozen=True)
class Bucket:
    """
    Én bucket, med endepunktet og noeklene som hoerer til den.

    Finnes fordi en bucket ikke kan adresseres av bucketnavnet alene: en
    JURISDIKSJONSBUNDET bucket har sitt EGET endepunkt (`.../eu`), saa to
    buckets i samme konto kan ligge paa hver sin URL. Uten dette kunne koden
    bare snakke med «den bucketen som staar i konfigurasjonen», og en
    kopiering fra én bucket til en annen var umulig aa skrive.
    """
    endpoint: str
    bucket: str
    access_key_id: str
    secret_access_key: str
    region: str = "auto"
    timeout: float = 10.0

    @property
    def konfigurert(self) -> bool:
        return bool(self.endpoint and self.bucket
                    and self.access_key_id and self.secret_access_key)

    def __str__(self) -> str:
        # Til logg og feilmeldinger. Noeklene skal ALDRI med.
        return f"{self.bucket} @ {urlparse(self.endpoint).netloc}"


def fra_settings(cfg: Settings) -> Bucket:
    """Bucketen tjenesten skriver til i vanlig drift."""
    return Bucket(endpoint=cfg.r2_endpoint, bucket=cfg.r2_bucket,
                  access_key_id=cfg.r2_access_key_id,
                  secret_access_key=cfg.r2_secret_access_key,
                  region=cfg.r2_region, timeout=cfg.r2_timeout_seconds)


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


def _autorisasjon(b: Bucket, metode: str, sti: str, hoder: dict[str, str],
                  kropp_sha256: str, amz_dato: str) -> str:
    dato = amz_dato[:8]
    kanonisk, signerte = kanonisk_forespoersel(metode, sti, hoder, kropp_sha256)
    omfang = f"{dato}/{b.region}/{_TJENESTE}/aws4_request"
    aa_signere = "\n".join([
        _ALG, amz_dato, omfang,
        hashlib.sha256(kanonisk.encode("utf-8")).hexdigest(),
    ])
    signatur = hmac.new(_signeringsnoekkel(b.secret_access_key, dato, b.region),
                        aa_signere.encode("utf-8"), hashlib.sha256).hexdigest()
    return (f"{_ALG} Credential={b.access_key_id}/{omfang}, "
            f"SignedHeaders={signerte}, Signature={signatur}")


# R2s Access Key ID-er er 32 tegn, og R2 sier det selv naar de ikke er det:
# «Credential access key has length 9, should be 32». Skulle lengden noen gang
# endre seg, er det denne linjen som skal rettes - ikke sjekken som skal bort.
AKSESSNOEKKEL_TEGN = 32


def _verdier(b: Bucket) -> tuple[tuple[str, str], ...]:
    """Navn -> verdi for de fire som maa settes. Navnene er de i Fly/.env."""
    return (("R2_ENDPOINT", b.endpoint), ("R2_BUCKET", b.bucket),
            ("R2_ACCESS_KEY_ID", b.access_key_id),
            ("R2_SECRET_ACCESS_KEY", b.secret_access_key))


def manglende(b: Bucket) -> list[str]:
    """Navnene paa de verdiene som ikke er satt. Tom liste = alle fire staar."""
    return [navn for navn, verdi in _verdier(b) if not verdi]


def feilkonfigurasjon(b: Bucket) -> str | None:
    """
    Hva som er GALT med et oppsett der alle fire verdiene staar - ellers None.

    Finnes fordi `/health` svarte «r2» gjennom tre ulike feilkonfigurasjoner
    mens produksjonen ikke kunne skrive et eneste bilde og hver donasjon fikk
    503 (AAP-B12, 2026-08-18). Ingen av dem trengte et nettverkskall for aa
    kunne ses; de trengte bare at noen saa etter.

    Her staar bare det som er ALLTID feil. At tokenet mangler EU-jurisdiksjon,
    at bucketen ikke finnes, at signaturen avvises - det kan bare et faktisk
    kall svare paa, og det er `tools/r2_check.py` sitt aerend. En sjekk med
    falske positiver ville stengt en fungerende bucket, og det er verre enn den
    tilstanden dette retter.

    Verdiene skrives ALDRI ut: to av de fire er hemmeligheter, og teksten
    herfra gaar bade i `/health` og i loggen.
    """
    for navn, verdi in _verdier(b):
        # Et linjeskift paa slutten av en secret er usynlig i panelet, gir
        # SignatureDoesNotMatch, og ser ut som feil noekkel.
        if verdi != verdi.strip():
            return f"{navn} har mellomrom eller linjeskift rundt verdien"

    u = urlparse(b.endpoint)
    if u.scheme not in ("http", "https") or not u.netloc:
        return ("R2_ENDPOINT er ikke en URL - formen er "
                "https://<konto-id>.r2.cloudflarestorage.com")
    if u.path.strip("/"):
        # `_sti()` legger selv paa /{bucket}/{noekkel}. Staar bucketnavnet i
        # endepunktet ogsaa, dekker signaturen en annen sti enn forespoerselen.
        return (f"R2_ENDPOINT har sti ('{u.path}') - bucketnavnet legges paa "
                f"av koden og skal ikke staa i endepunktet")
    if "/" in b.bucket:
        return "R2_BUCKET er et bucketnavn alene, ikke en sti"
    if len(b.access_key_id) != AKSESSNOEKKEL_TEGN:
        return (f"R2_ACCESS_KEY_ID er {len(b.access_key_id)} tegn, skal vaere "
                f"{AKSESSNOEKKEL_TEGN}")
    return None


def tilstand(cfg: Settings) -> str:
    """
    Tilstanden lagringen er i, paa den formen `/health` viser den under
    «bilder»: «r2», «ikke konfigurert (§6)» eller «feilkonfigurert (...)».

    TRE tilstander, ikke to. «Ingenting satt» er en funksjon som er av, og det
    er en normal tilstand i utvikling. «Tre av fire satt» eller «Access Key ID
    paa ni tegn» er noe i stykker, og skal ikke se ut som det foerste - det var
    nettopp likheten som gjorde at fire feilkonfigurasjoner passerte som normal
    drift (AAP-B12). Samme tredeling som `database` allerede har i `/health`:
    ok | feilkonfigurert (...) | utilgjengelig.

    «r2» betyr at oppsettet ikke kan vaere feil paa noen maate vi kan se uten
    aa spoerre Cloudflare. Det betyr ikke at en opplasting vil lykkes.
    """
    b = fra_settings(cfg)
    mangler = manglende(b)
    if len(mangler) == len(_verdier(b)):
        return "ikke konfigurert (§6)"
    if mangler:
        # Halvveis satt er ikke «av» - noen har vaert her og ikke blitt ferdig.
        return f"feilkonfigurert (mangler {', '.join(mangler)})"
    feil = feilkonfigurasjon(b)
    return f"feilkonfigurert ({feil})" if feil else "r2"


def kan_brukes(cfg: Settings) -> bool:
    """
    Om en donasjon kan tas imot.

    Definert som «tilstand() sier r2» med vilje: da KAN ikke `/health` og
    mottaket vaere uenige om hvorvidt lagringen virker, og det var uenigheten
    som var feilen i AAP-B12.
    """
    return tilstand(cfg) == "r2"


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


def _sti(b: Bucket, noekkel: str) -> str:
    # Sti-stil (endepunktet er kontospesifikt, bucketen er foerste ledd).
    # quote lar bokstaver, tall og «-._~» staa; skraastrekene i noekkelen er
    # katalogskiller og skal ikke kodes.
    return f"/{quote(b.bucket, safe='')}/{quote(noekkel, safe='/')}"


def _kall(b: Bucket, metode: str, noekkel: str, kropp: bytes = b"",
          content_type: str = "") -> httpx.Response:
    if not b.konfigurert:
        raise LagringFeilet("R2 er ikke konfigurert")

    sti = _sti(b, noekkel)
    amz_dato = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    kropp_sha = hashlib.sha256(kropp).hexdigest() if kropp else TOM_SHA256
    hoder = {
        "host": urlparse(b.endpoint).netloc,
        "x-amz-content-sha256": kropp_sha,
        "x-amz-date": amz_dato,
    }
    if content_type:
        hoder["content-type"] = content_type
    hoder["authorization"] = _autorisasjon(b, metode, sti, hoder, kropp_sha,
                                           amz_dato)

    url = b.endpoint.rstrip("/") + sti
    try:
        r = httpx.request(metode, url, headers=hoder, content=kropp,
                          timeout=b.timeout)
    except httpx.HTTPError as exc:
        raise LagringFeilet(f"{metode} mot {b} naadde ikke fram: {exc}") from exc
    if r.status_code >= 300:
        # Kroppen er XML fra R2 og sier hvilken av de tre vanlige feilene det
        # er: SignatureDoesNotMatch, NoSuchBucket eller AccessDenied.
        raise LagringFeilet(f"{metode} mot {b} svarte {r.status_code}: "
                            f"{r.text[:300]}", status=r.status_code)
    return r


# --- mot en navngitt bucket (kopiering mellom to) ------------------------

def legg_i(b: Bucket, noekkel: str, data: bytes, content_type: str) -> None:
    _kall(b, "PUT", noekkel, data, content_type)
    log.info("Lastet opp %d byte til %s (%s)", len(data), b, noekkel)


def hent_fra(b: Bucket, noekkel: str) -> bytes:
    return _kall(b, "GET", noekkel).content


def hent_om_finnes(b: Bucket, noekkel: str) -> bytes | None:
    """
    Som `hent_fra`, men None naar objektet ikke finnes (404).

    Alt annet kaster fortsatt. Et 403 skal ALDRI se ut som et fravaer - det er
    den forvekslingen som gjorde at en manglende deponering og et mislykket
    kall saa like ut hos klienten i tre versjoner (rot-CLAUDE.md §7.3).
    """
    try:
        return hent_fra(b, noekkel)
    except LagringFeilet as exc:
        if exc.status == 404:
            return None
        raise


# --- mot bucketen i konfigurasjonen (vanlig drift) ----------------------

def legg(cfg: Settings, noekkel: str, data: bytes, content_type: str) -> None:
    """Laster opp ett objekt. Kaster `LagringFeilet` - svelger ingenting."""
    legg_i(fra_settings(cfg), noekkel, data, content_type)


def hent(cfg: Settings, noekkel: str) -> bytes:
    """Henter ett objekt. Brukes av tools/r2_check.py."""
    return hent_fra(fra_settings(cfg), noekkel)


def slett(cfg: Settings, noekkel: str) -> None:
    """Sletter ett objekt. Brukes av tools/r2_check.py."""
    _kall(fra_settings(cfg), "DELETE", noekkel)
