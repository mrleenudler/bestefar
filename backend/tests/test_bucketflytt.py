"""
Kopiering mellom to R2-buckets (§6, jurisdiksjonsbytte).

Jobben kjoeres én gang mot produksjon. Det testene bryr seg om, i denne
rekkefoelgen:

  1. den skriver ingenting i toerrkjoering,
  2. den regner ingenting som kopiert foer innholdet er lest tilbake fra maalet,
  3. den overskriver ALDRI et objekt som ligger der med annet innhold,
  4. den kan kjoeres om igjen uten aa gjoere skade,
  5. den roerer ikke databasen - noeklene skal vaere uendret etterpaa.
"""
import pytest

JPEG = b"\xff\xd8\xff\xe0" + b"\x07" * 300
PNG = b"\x89PNG\r\n\x1a\n" + b"\x08" * 90


@pytest.fixture()
def boetter(monkeypatch, db_url):
    """To buckets i minnet, med hver sin URL slik produksjonen har det."""
    from app.services import objstore

    lager: dict[str, dict[str, bytes]] = {"kilde": {}, "maal": {}}
    # `get_maal` gjelder BARE lesing fra maalet. En feil som slaar inn paa
    # kilden ogsaa, ville stoppet jobben foer den naadde det testen sikter paa.
    feil: dict[str, str | None] = {"put": None, "get_maal": None}

    kilde = objstore.Bucket(endpoint="https://konto.r2.cloudflarestorage.com",
                            bucket="bestefar-scan-failures",
                            access_key_id="a", secret_access_key="s")
    maal = objstore.Bucket(endpoint="https://konto.eu.r2.cloudflarestorage.com",
                           bucket="bestefar-scan-failures-eur",
                           access_key_id="a", secret_access_key="s")

    def _navn(b):
        return "maal" if b.bucket.endswith("-eur") else "kilde"

    def _hent_fra(b, noekkel):
        if feil["get_maal"] and _navn(b) == "maal":
            raise objstore.LagringFeilet(feil["get_maal"], status=403)
        rom = lager[_navn(b)]
        if noekkel not in rom:
            raise objstore.LagringFeilet(f"GET svarte 404: {noekkel}", status=404)
        return rom[noekkel]

    def _legg_i(b, noekkel, data, content_type):
        if feil["put"]:
            raise objstore.LagringFeilet(feil["put"], status=403)
        lager[_navn(b)][noekkel] = data

    monkeypatch.setattr(objstore, "hent_fra", _hent_fra)
    monkeypatch.setattr(objstore, "legg_i", _legg_i)
    return {"lager": lager, "feil": feil, "kilde": kilde, "maal": maal}


def _rad(session, noekkel, data, boetter, tag=None):
    """Én rad i basen + objektet i kildebucketen."""
    from app.models import FailedAnalysis, FailedTag

    fa = FailedAnalysis(status_code=3, confidence=0.5, core_version="1.0.0",
                        tag=tag or FailedTag.rejected, object_key=noekkel)
    session.add(fa)
    session.commit()
    boetter["lager"]["kilde"][noekkel] = data
    return fa.id


def _kopier(session, boetter, **kw):
    from app.services import bucketflytt

    return bucketflytt.kopier(session, boetter["kilde"], boetter["maal"],
                              skriv=lambda s: None, **kw)


def test_toerrkjoering_skriver_ingenting(session, boetter):
    _rad(session, "feilanalyse/rejected/2026/08/03/1-aa.jpg", JPEG, boetter)
    utfall = _kopier(session, boetter)

    assert utfall.kopiert == ["feilanalyse/rejected/2026/08/03/1-aa.jpg"]
    assert utfall.byte_kopiert == len(JPEG)
    assert boetter["lager"]["maal"] == {}, "toerrkjoering skal ikke skrive"


def test_kopierer_med_uendret_noekkel(session, boetter):
    from app.models import FailedAnalysis

    noekkel = "feilanalyse/ocr_mismatch/2026/08/11/3-bb.jpg"
    rad = _rad(session, noekkel, JPEG, boetter)
    utfall = _kopier(session, boetter, toerrkjoer=False)

    assert utfall.ok and utfall.kopiert == [noekkel]
    assert boetter["lager"]["maal"][noekkel] == JPEG
    assert boetter["lager"]["kilde"][noekkel] == JPEG, "kilden skal staa uroert"
    # Basen skal ikke vaere roert - hele poenget med lik noekkel.
    session.expire_all()
    assert session.get(FailedAnalysis, rad).object_key == noekkel


def test_kan_kjoeres_om_igjen(session, boetter):
    noekkel = "feilanalyse/rejected/2026/08/03/1-cc.jpg"
    _rad(session, noekkel, JPEG, boetter)
    _kopier(session, boetter, toerrkjoer=False)

    utfall = _kopier(session, boetter, toerrkjoer=False)
    assert utfall.ok
    assert utfall.kopiert == [] and utfall.allerede_der == [noekkel]


def test_overskriver_ikke_annet_innhold(session, boetter):
    """
    Samme noekkel med ANNET innhold i maalet er ikke en avbrutt kopiering, og
    skal ses paa av et menneske - ikke overskrives i stillhet.
    """
    noekkel = "feilanalyse/rejected/2026/08/03/1-dd.jpg"
    _rad(session, noekkel, JPEG, boetter)
    boetter["lager"]["maal"][noekkel] = PNG

    utfall = _kopier(session, boetter, toerrkjoer=False)
    assert not utfall.ok
    assert "ANNET innhold" in utfall.feilet[0][1]
    assert boetter["lager"]["maal"][noekkel] == PNG, "maalet skal staa uroert"


def test_feilet_opplasting_telles_ikke_som_kopiert(session, boetter):
    noekkel = "feilanalyse/rejected/2026/08/03/1-ee.jpg"
    _rad(session, noekkel, JPEG, boetter)
    boetter["feil"]["put"] = "AccessDenied"

    utfall = _kopier(session, boetter, toerrkjoer=False)
    assert not utfall.ok and utfall.kopiert == []
    assert boetter["lager"]["kilde"][noekkel] == JPEG


def test_avvikende_tilbakelesing_telles_ikke_som_kopiert(session, boetter,
                                                        monkeypatch):
    """PUT sa ok, men det som kom tilbake var noe annet."""
    from app.services import objstore

    noekkel = "feilanalyse/rejected/2026/08/03/1-ff.jpg"
    _rad(session, noekkel, JPEG, boetter)

    ekte_legg = objstore.legg_i

    def _legg_som_forvansker(b, n, data, ct):
        ekte_legg(b, n, data[:-1], ct)      # mistet en byte underveis

    monkeypatch.setattr(objstore, "legg_i", _legg_som_forvansker)
    utfall = _kopier(session, boetter, toerrkjoer=False)

    assert not utfall.ok and utfall.kopiert == []
    assert "lest tilbake" in utfall.feilet[0][1]


def test_403_paa_maalet_er_ikke_et_fravaer(session, boetter):
    """
    Et 403 skal aldri behandles som «objektet finnes ikke» - da ville jobben
    lastet opp paa nytt i det uendelige og meldt suksess. Samme forveksling som
    BackupKeys.resolve gikk i.
    """
    noekkel = "feilanalyse/rejected/2026/08/03/1-gg.jpg"
    _rad(session, noekkel, JPEG, boetter)
    boetter["feil"]["get_maal"] = "AccessDenied"

    utfall = _kopier(session, boetter, toerrkjoer=False)
    assert not utfall.ok
    assert "kunne ikke lese maalet" in utfall.feilet[0][1]
    assert noekkel not in boetter["lager"]["maal"], "skal ikke ha lastet opp"


def test_samme_bucket_avvises(session, boetter):
    from app.services import bucketflytt

    with pytest.raises(RuntimeError, match="samme bucket"):
        bucketflytt.kopier(session, boetter["kilde"], boetter["kilde"],
                           skriv=lambda s: None)


def test_ukonfigurert_bucket_avvises(session, boetter):
    from app.services import bucketflytt, objstore

    tom = objstore.Bucket(endpoint="", bucket="", access_key_id="",
                          secret_access_key="")
    with pytest.raises(RuntimeError, match="ikke konfigurert"):
        bucketflytt.kopier(session, tom, boetter["maal"], skriv=lambda s: None)


def test_rader_uten_objekt_hoppes_over(session, boetter):
    """En rad uten `object_key` har ingenting i noen bucket."""
    from app.models import FailedAnalysis, FailedTag

    session.add(FailedAnalysis(status_code=3, confidence=0.1,
                               core_version="1.0.0", tag=FailedTag.rejected))
    session.commit()
    utfall = _kopier(session, boetter, toerrkjoer=False)

    assert utfall.ok and utfall.kopiert == []
