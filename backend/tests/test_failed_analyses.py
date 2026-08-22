"""
§6: feilanalyse-donasjonen og objektlagringen.

Endepunktet hadde ingen tester foer bildene ble flyttet til R2 (2026-08-15).
De to viktigste her er `test_r2_feil_gir_503` og `test_r2_feil_lager_ingen_rad`:
en opplasting som ikke naadde fram skal aldri kunne se ut som en vellykket
donasjon - hverken for klienten eller i basen.
"""
import hashlib

import pytest

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64          # gyldig magi, resten fyll
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64

# `series_id` staar med vilje: ruten tar den ikke lenger imot (B-52), og en
# klient som fortsatt sender den skal komme like godt gjennom. Testen for at
# den faktisk ignoreres ligger nederst.
SKJEMA = {"status_code": "3", "confidence": "0.42", "core_version": "1.2.3",
          "tag": "ocr_mismatch", "series_id": "s-1",
          "detected_scores": "[10.4, 9.1]", "ocr_scores": "[10.4, 9.2]"}

# Et oppsett som skal passere alle sjekkene i objstore.feilkonfigurasjon.
# Access Key ID-en er 32 tegn fordi kortere ALLTID er feil hos R2 og nå
# stoppes her (B-51) - den var «AKIDEXAMPLE» til 2026-08-18.
R2_ENV = {
    "R2_ENDPOINT": "https://konto.r2.cloudflarestorage.com",
    "R2_BUCKET": "bestefar-bilder",
    "R2_ACCESS_KEY_ID": "0123456789abcdef0123456789abcdef",
    "R2_SECRET_ACCESS_KEY": "hemmelig-testnoekkel",
}


def _send(client, data=JPEG, felt=None):
    return client.post("/v1/failed-analyses", data={**SKJEMA, **(felt or {})},
                       files={"image": ("skive.jpg", data, "image/jpeg")})


# --- Uten R2: vi tar ikke imot noe ---------------------------------------

def test_uten_r2_svarer_503(client):
    """
    Kolonnen som holdt bildene i basen er borte (a3f7c1e59b24), saa det finnes
    ikke et annet sted aa gjoere av dem. 503 og ikke 201: en kvittering paa noe
    vi kastet ville vaert stille datatap. 503 er `retryable`, saa klienten
    beholder koen sin.
    """
    assert _send(client).status_code == 503


def test_uten_r2_lagres_ingen_rad(client, session):
    from sqlalchemy import func, select

    from app.models import FailedAnalysis

    _send(client)
    assert session.scalar(select(func.count()).select_from(FailedAnalysis)) == 0


def test_health_melder_at_lagringen_mangler(client):
    assert client.get("/health").json()["bilder"] == "ikke konfigurert (§6)"


# --- Med R2: bildet ligger der, og basen har bare noekkelen ---------------

@pytest.fixture()
def r2(monkeypatch, db_url):
    """Konfigurerer R2 og bytter ut selve HTTP-kallet.

    `lagret` er noekkel -> (byte, content-type), altsaa den bucketen testen
    later som den skriver til.
    """
    for navn, verdi in R2_ENV.items():
        monkeypatch.setenv(navn, verdi)

    from app.config import settings
    from app.services import objstore

    # `client` bygger appen foer denne fixturen kjoerer og har allerede varmet
    # opp lru_cache-en i settings(). Uten dette leser endepunktet fortsatt en
    # konfigurasjon uten R2, og testen ville stille ha testet basen-veien.
    settings.cache_clear()

    lagret: dict[str, tuple[bytes, str]] = {}
    feil: list[str] = []

    def _legg(cfg, noekkel, data, content_type):
        if feil:
            raise objstore.LagringFeilet(feil[0])
        lagret[noekkel] = (data, content_type)

    monkeypatch.setattr(objstore, "legg", _legg)
    return {"lagret": lagret, "feil": feil}


def test_med_r2_lagres_bildet_utenfor_basen(client, session, r2):
    from app.models import FailedAnalysis

    r = _send(client)
    assert r.status_code == 201, r.text

    fa = session.get(FailedAnalysis, r.json()["id"])
    assert fa.object_key in r2["lagret"]
    data, content_type = r2["lagret"][fa.object_key]
    assert data == JPEG
    assert content_type == "image/jpeg"
    # Noekkelen navngis med tag, dato og rad-id - se objstore.objektnoekkel.
    assert fa.object_key.startswith("feilanalyse/ocr_mismatch/")
    assert f"/{fa.id}-" in fa.object_key
    assert fa.object_key.endswith(".jpg")


def test_png_faar_riktig_endelse_og_type(client, session, r2):
    from app.models import FailedAnalysis

    r = client.post("/v1/failed-analyses", data=SKJEMA,
                    files={"image": ("skive.png", PNG, "image/jpeg")})
    assert r.status_code == 201
    fa = session.get(FailedAnalysis, r.json()["id"])
    # Innholdet bestemmer, ikke det klienten paastaar i multiparten.
    assert fa.object_key.endswith(".png")
    assert r2["lagret"][fa.object_key][1] == "image/png"


def test_r2_feil_gir_503(client, r2):
    r2["feil"].append("SignatureDoesNotMatch")
    r = _send(client)
    # 503 og ikke 4xx: klienten prooever igjen paa >= 500 og beholder koen sin.
    assert r.status_code == 503


def test_r2_feil_lager_ingen_rad(client, session, r2):
    from sqlalchemy import func, select

    from app.models import FailedAnalysis

    r2["feil"].append("nede")
    _send(client)
    antall = session.scalar(select(func.count()).select_from(FailedAnalysis))
    assert antall == 0, "en feilet opplasting skal ikke etterlate en rad"


def test_health_melder_r2(client, r2):
    assert client.get("/health").json()["bilder"] == "r2"


def test_ingen_series_id_paa_donasjonen(session, r2):
    """
    §6 som invariant: donasjonen skal ikke kunne kobles til en konto (B-52).

    `series_id` var samme ID som serien lagres under i `/v1/stats`, altsaa den
    ene veien fra et donert bilde til en person. `user_id` var en fremmednoekkel
    som aldri ble satt - ingen kobling, men en ferdig oppkoblet mulighet for aa
    lage én. Testen faller hvis noen tar en av dem inn igjen.
    """
    from app.models import FailedAnalysis

    assert "series_id" not in FailedAnalysis.__table__.columns
    assert "user_id" not in FailedAnalysis.__table__.columns
    # Ingen vei til brukertabellen i det hele tatt, uansett kolonnenavn.
    assert not FailedAnalysis.__table__.foreign_keys


def test_gammel_klient_som_sender_series_id_slipper_gjennom(client, session, r2):
    """
    Feltet ignoreres, det avvises ikke.

    Klientene i felten sender det fortsatt, og et 4xx her ville vaert
    ikke-`retryable` - altsaa stille tap av donasjonen hos en bruker vi ikke
    kan oppdatere i samme oeyeblikk.
    """
    from app.models import FailedAnalysis

    r = _send(client, felt={"series_id": "0f4c9b2e-1111-2222-3333-444455556666"})
    assert r.status_code == 201, r.text

    fa = session.get(FailedAnalysis, r.json()["id"])
    assert fa.object_key in r2["lagret"]
    # Verdien skal ikke ha naadd basen i noen form.
    assert "0f4c9b2e" not in repr(fa.__dict__)


def test_ingen_bildekolonne_igjen(session, r2):
    """
    §6 som invariant, ikke som praksis: modellen skal ikke ha et sted aa legge
    bilder i basen. Testen faller hvis noen tar kolonnen inn igjen.
    """
    from app.models import FailedAnalysis

    assert "image_legacy" not in FailedAnalysis.__table__.columns


# --- capture_trigger: hvordan bildet ble tatt (issue #11, B-53) -----------
#
# Eget felt og ikke en tag-verdi, fordi de to er ortogonale: `tag` sier hva
# donasjonen viser, `capture_trigger` hvordan bildet ble tatt, og en
# timeout-capture kan ende som hvilken som helst tag.

def test_capture_trigger_lagres(client, session, r2):
    from app.models import FailedAnalysis

    r = _send(client, felt={"capture_trigger": "timeout"})
    assert r.status_code == 201, r.text
    assert session.get(FailedAnalysis, r.json()["id"]).capture_trigger == "timeout"


def test_capture_trigger_er_ortogonal_til_tag(client, session, r2):
    """
    Hele begrunnelsen for eget felt: en timeout-capture kan ende som et
    vellykket OCR-treff. Med «timeout» som tag-verdi ville utfallet vaert borte.
    """
    from app.models import FailedAnalysis

    r = _send(client, felt={"capture_trigger": "timeout", "tag": "ocr_match"})
    fa = session.get(FailedAnalysis, r.json()["id"])
    assert (fa.capture_trigger, fa.tag) == ("timeout", "ocr_match")


def test_uten_capture_trigger_er_den_null_og_ikke_auto(client, session, r2):
    """
    NULL betyr «klienten sa det ikke». En default paa `auto` ville stemplet
    donasjonene fra v0.29-vinduet - der timeout-capture fantes, men feltet ikke
    ble sendt - som gatede, og det er nettopp maalingen AAP-K1 hviler paa.
    """
    from app.models import FailedAnalysis

    r = _send(client)
    assert session.get(FailedAnalysis, r.json()["id"]).capture_trigger is None


def test_ukjent_capture_trigger_gir_422(client, r2):
    """
    Enumet er vaart, som `tag`. Konsekvensen er dokumentert i modellen: nye
    verdier settes inn server-side FOERST, ellers kastes de donasjonene den nye
    verdien handler om (422 er ikke `retryable`).
    """
    assert _send(client, felt={"capture_trigger": "manuell"}).status_code == 422


# --- Feilkonfigurert R2: satt er ikke det samme som virker (AAP-B12) ------
#
# Alle fire verdiene staar, men oppsettet kan umulig virke. Foer B-51 svarte
# /health «r2» paa nettopp dette, mens hver donasjon fikk 503 - fire forsoek i
# produksjon 2026-08-18 gikk med paa aa finne ut hvorfor.

def _bucket(**overstyring):
    from app.services import objstore

    v = {**R2_ENV, **overstyring}
    return objstore.Bucket(endpoint=v["R2_ENDPOINT"], bucket=v["R2_BUCKET"],
                           access_key_id=v["R2_ACCESS_KEY_ID"],
                           secret_access_key=v["R2_SECRET_ACCESS_KEY"])


def test_riktig_oppsett_har_ingen_feil():
    from app.services import objstore

    assert objstore.feilkonfigurasjon(_bucket()) is None


@pytest.mark.parametrize("overstyring, i_teksten", [
    # De to som faktisk skjedde: bucketnavnet i endepunktet, og Access Key
    # ID-en satt til plassholderteksten.
    ({"R2_ENDPOINT": "https://konto.eu.r2.cloudflarestorage.com/bestefar-eur"}, "sti"),
    ({"R2_ACCESS_KEY_ID": "din-key-id"}, "tegn"),
    ({"R2_ENDPOINT": "konto.eu.r2.cloudflarestorage.com"}, "URL"),
    ({"R2_BUCKET": "konto/bestefar-bilder"}, "sti"),
    ({"R2_SECRET_ACCESS_KEY": "hemmelig-testnoekkel\n"}, "linjeskift"),
])
def test_feilkonfigurasjon_fanges_uten_nettverk(overstyring, i_teksten):
    from app.services import objstore

    (navn,) = overstyring
    grunn = objstore.feilkonfigurasjon(_bucket(**overstyring))
    assert grunn is not None, f"{navn} slapp gjennom"
    # Teksten skal si HVILKEN verdi som er feil - den er hele nytten.
    assert navn in grunn and i_teksten in grunn
    # ... men aldri verdien: to av de fire er hemmeligheter, og teksten gaar
    # bade i /health og i loggen.
    assert "hemmelig-testnoekkel" not in grunn


@pytest.fixture()
def r2_feilkonfigurert(monkeypatch, db_url):
    """
    Alle fire satt, endepunktet har bucketnavnet i stien.

    Patcher ikke `objstore.legg`: kommer en donasjon saa langt som til et
    HTTP-kall, er det testen som har feilet.
    """
    for navn, verdi in R2_ENV.items():
        monkeypatch.setenv(navn, verdi)
    monkeypatch.setenv("R2_ENDPOINT",
                       "https://konto.r2.cloudflarestorage.com/bestefar-bilder")

    from app.config import settings
    settings.cache_clear()


def test_feilkonfigurert_r2_gir_503(client, r2_feilkonfigurert):
    assert _send(client).status_code == 503


def test_feilkonfigurert_r2_lager_ingen_rad(client, session, r2_feilkonfigurert):
    from sqlalchemy import func, select

    from app.models import FailedAnalysis

    _send(client)
    assert session.scalar(select(func.count()).select_from(FailedAnalysis)) == 0


def test_health_skiller_feilkonfigurert_fra_av(client, r2_feilkonfigurert):
    bilder = client.get("/health").json()["bilder"]
    assert bilder.startswith("feilkonfigurert (")
    assert "R2_ENDPOINT" in bilder


@pytest.fixture()
def r2_halvveis(monkeypatch, db_url):
    """Bare bucketnavnet satt - noen har vaert her og ikke blitt ferdig."""
    monkeypatch.setenv("R2_BUCKET", "bestefar-bilder")

    from app.config import settings
    settings.cache_clear()


def test_halvveis_satt_er_ikke_av(client, r2_halvveis):
    """
    Tre manglende secrets er en halvferdig jobb, ikke «funksjonen er av», og
    skal ikke se ut som en maskin uten R2 i det hele tatt.
    """
    bilder = client.get("/health").json()["bilder"]
    assert bilder.startswith("feilkonfigurert (mangler ")
    assert "R2_ENDPOINT" in bilder and "R2_BUCKET" not in bilder
    assert _send(client).status_code == 503


# --- Avvisninger ---------------------------------------------------------
#
# Alle med `r2`: sjekken av lagringen staar FOERST i endepunktet, saa uten den
# ville disse testene faatt 503 og ikke naadd fram til det de tester.

def test_for_stort_bilde_gir_413(client, r2):
    from app.config import settings

    stort = JPEG + b"\x00" * settings().max_upload_bytes
    assert _send(client, stort).status_code == 413


def test_ikke_et_bilde_gir_415(client, r2):
    assert _send(client, b"dette er ikke et bilde").status_code == 415


def test_ugyldig_json_i_poengfeltet_gir_422(client, r2):
    assert _send(client, felt={"detected_scores": "{ikke json"}).status_code == 422


def test_ukjent_tag_gir_422(client):
    assert _send(client, felt={"tag": "noe_annet"}).status_code == 422


# --- Signeringen ---------------------------------------------------------

def test_kanonisk_forespoersel():
    """
    Haandregnet mot SigV4-spesifikasjonen: hodene sorteres og skrives i smaa
    bokstaver, verdiene trimmes, listen over signerte hoder gjentas, og
    kropp-hashen staar til slutt. En feil her gir 403 fra R2 uten forklaring,
    saa den maa fanges her.
    """
    from app.services.objstore import kanonisk_forespoersel

    kanonisk, signerte = kanonisk_forespoersel(
        "PUT", "/bestefar-bilder/feilanalyse/a.jpg",
        {"x-amz-date": "20260814T101112Z", "host": "konto.example",
         "content-type": " image/jpeg "},
        "abc123")
    assert signerte == "content-type;host;x-amz-date"
    assert kanonisk == (
        "PUT\n"
        "/bestefar-bilder/feilanalyse/a.jpg\n"
        "\n"
        "content-type:image/jpeg\n"
        "host:konto.example\n"
        "x-amz-date:20260814T101112Z\n"
        "\n"
        "content-type;host;x-amz-date\n"
        "abc123")


def test_tom_kropp_hash():
    """GET og DELETE signerer med sha256 av tom streng."""
    from app.services.objstore import TOM_SHA256

    assert TOM_SHA256 == hashlib.sha256(b"").hexdigest()
    assert TOM_SHA256.startswith("e3b0c44298fc1c14")


def test_signeringen_settes_paa_forespoerselen(monkeypatch):
    """
    Hele veien ut, med httpx byttet ut: riktig URL (sti-stil, bucket foerst),
    Authorization med noekkel-ID og signatur, og x-amz-content-sha256 lik
    hashen av kroppen.
    """
    import httpx

    from app.config import settings
    from app.services import objstore

    monkeypatch.setenv("R2_ENDPOINT", "https://konto.r2.cloudflarestorage.com/")
    monkeypatch.setenv("R2_BUCKET", "bestefar-bilder")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "hemmelig-testnoekkel")
    monkeypatch.setenv("R2_REGION", "auto")
    settings.cache_clear()
    cfg = settings()

    sett = {}

    def _request(metode, url, **kw):
        sett.update(metode=metode, url=url, hoder=kw["headers"],
                    kropp=kw["content"])
        return httpx.Response(200, request=httpx.Request(metode, url))

    monkeypatch.setattr(objstore.httpx, "request", _request)
    objstore.legg(cfg, "feilanalyse/rejected/2026/08/14/7-abcd.jpg", JPEG,
                  "image/jpeg")

    assert sett["metode"] == "PUT"
    assert sett["url"] == ("https://konto.r2.cloudflarestorage.com/"
                           "bestefar-bilder/feilanalyse/rejected/2026/08/14/"
                           "7-abcd.jpg")
    assert sett["kropp"] == JPEG
    assert sett["hoder"]["x-amz-content-sha256"] == hashlib.sha256(JPEG).hexdigest()
    auth = sett["hoder"]["authorization"]
    assert auth.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/")
    assert "/auto/s3/aws4_request" in auth
    assert "SignedHeaders=content-type;host;x-amz-content-sha256;x-amz-date" in auth
    assert "hemmelig-testnoekkel" not in auth


def test_r2_svarer_403_kaster(monkeypatch):
    import httpx

    from app.config import settings
    from app.services import objstore

    monkeypatch.setenv("R2_ENDPOINT", "https://konto.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_BUCKET", "b")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "a")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "s")
    settings.cache_clear()

    def _request(metode, url, **kw):
        return httpx.Response(403, text="<Error>SignatureDoesNotMatch</Error>",
                              request=httpx.Request(metode, url))

    monkeypatch.setattr(objstore.httpx, "request", _request)
    with pytest.raises(objstore.LagringFeilet, match="403"):
        objstore.legg(settings(), "x.jpg", b"a", "image/jpeg")
