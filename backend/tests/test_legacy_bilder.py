"""
AAP-B11: flytting av gamle `image_legacy`-rader til R2.

Verktoeyet kjoeres én gang, mot produksjonsdata, og sletter den eneste kopien av
et bilde naar det er ferdig. Derfor er det den destruktive rekkefoelgen testene
her bryr seg om: at basen ALDRI toemmes uten at objektet er lest tilbake og er
identisk. Resten - opptelling, toerrkjoering - er sekundaert.
"""
import pytest

JPEG = b"\xff\xd8\xff\xe0" + b"\x01" * 200
PNG = b"\x89PNG\r\n\x1a\n" + b"\x02" * 50


@pytest.fixture()
def r2(monkeypatch, db_url):
    """R2 konfigurert, med en bucket i minnet og en valgfri feil."""
    monkeypatch.setenv("R2_ENDPOINT", "https://konto.r2.cloudflarestorage.com")
    monkeypatch.setenv("R2_BUCKET", "bestefar-bilder")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "AKIDEXAMPLE")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "hemmelig-testnoekkel")

    from app.config import settings
    from app.services import objstore

    settings.cache_clear()
    bucket: dict[str, bytes] = {}
    oppsett = {"put_feiler": None, "hent_gir": None}

    def _legg(cfg, noekkel, data, content_type):
        if oppsett["put_feiler"]:
            raise objstore.LagringFeilet(oppsett["put_feiler"])
        bucket[noekkel] = data

    def _hent(cfg, noekkel):
        if oppsett["hent_gir"] is not None:
            return oppsett["hent_gir"]
        return bucket[noekkel]

    monkeypatch.setattr(objstore, "legg", _legg)
    monkeypatch.setattr(objstore, "hent", _hent)
    return {"bucket": bucket, "oppsett": oppsett}


def _rad(session, data=JPEG, tag=None):
    from app.models import FailedAnalysis, FailedTag

    fa = FailedAnalysis(status_code=3, confidence=0.5, core_version="1.0.0",
                        tag=tag or FailedTag.rejected, image_legacy=data)
    session.add(fa)
    session.commit()
    return fa.id


def _flytt(session, **kw):
    from app.config import settings
    from app.services import legacy_bilder

    return legacy_bilder.flytt(session, settings(), skriv=lambda s: None, **kw)


def test_toerrkjoering_roerer_ingenting(session, r2):
    from app.models import FailedAnalysis

    rad = _rad(session, JPEG)
    utfall = _flytt(session)          # toerrkjoer=True er standard

    assert utfall.flyttet == [rad]
    assert utfall.byte_flyttet == len(JPEG)
    assert r2["bucket"] == {}, "toerrkjoering skal ikke laste opp"
    session.expire_all()
    assert session.get(FailedAnalysis, rad).image_legacy == JPEG


def test_flytter_og_toemmer_kolonnen(session, r2):
    from app.models import FailedAnalysis

    rad = _rad(session, JPEG)
    utfall = _flytt(session, toerrkjoer=False)

    assert utfall.ok and utfall.flyttet == [rad]
    fa = session.get(FailedAnalysis, rad)
    assert fa.image_legacy is None
    assert r2["bucket"][fa.object_key] == JPEG
    assert fa.object_key.endswith(".jpg")


def test_noekkelen_dateres_etter_innsendingen(session, r2):
    """Datoen i noekkelen er da donasjonen kom inn, ikke da den ble flyttet."""
    from datetime import datetime, timezone

    from app.models import FailedAnalysis

    rad = _rad(session, PNG)
    fa = session.get(FailedAnalysis, rad)
    fa.submitted_at = datetime(2026, 3, 4, 9, 0, tzinfo=timezone.utc)
    session.commit()

    _flytt(session, toerrkjoer=False)
    session.expire_all()
    assert "/2026/03/04/" in session.get(FailedAnalysis, rad).object_key


def test_feilet_opplasting_beholder_bildet(session, r2):
    from app.models import FailedAnalysis

    rad = _rad(session, JPEG)
    r2["oppsett"]["put_feiler"] = "AccessDenied"
    utfall = _flytt(session, toerrkjoer=False)

    assert not utfall.ok
    assert utfall.feilet[0][0] == rad
    fa = session.get(FailedAnalysis, rad)
    assert fa.image_legacy == JPEG, "bildet er den eneste kopien - skal staa"
    assert fa.object_key is None


def test_avvikende_tilbakelesing_beholder_bildet(session, r2):
    """
    PUT svarte ok, men GET ga noe annet. Da er basen fortsatt den eneste
    paalitelige kopien, og den skal ikke toemmes.
    """
    from app.models import FailedAnalysis

    rad = _rad(session, JPEG)
    r2["oppsett"]["hent_gir"] = b"noe helt annet"
    utfall = _flytt(session, toerrkjoer=False)

    assert not utfall.ok
    fa = session.get(FailedAnalysis, rad)
    assert fa.image_legacy == JPEG
    assert fa.object_key is None


def test_ukjent_filtype_hoppes_over(session, r2):
    from app.models import FailedAnalysis

    rad = _rad(session, b"ikke et bilde i det hele tatt")
    utfall = _flytt(session, toerrkjoer=False)

    assert utfall.ok, "en ukjent blob er ikke en feil, bare noe vi lar ligge"
    assert utfall.flyttet == [] and utfall.hoppet_over[0][0] == rad
    assert session.get(FailedAnalysis, rad).image_legacy is not None


def test_stort_bilde_flyttes_selv_om_det_er_over_opplastingsgrensa(session, r2):
    """
    MAX_UPLOAD_BYTES er en regel for MOTTAKET. Én av produksjonsradene er 11 MB,
    fra tiden foer grensen fantes, og skal flyttes som de andre.
    """
    from app.config import settings
    from app.models import FailedAnalysis

    stort = JPEG + b"\x00" * (settings().max_upload_bytes + 1000)
    rad = _rad(session, stort)
    utfall = _flytt(session, toerrkjoer=False)

    assert utfall.flyttet == [rad]
    assert session.get(FailedAnalysis, rad).image_legacy is None


def test_en_feilet_rad_stopper_ikke_de_andre(session, r2):
    from app.models import FailedAnalysis

    ok1 = _rad(session, JPEG)
    ukjent = _rad(session, b"xxx")
    ok2 = _rad(session, PNG)

    utfall = _flytt(session, toerrkjoer=False)

    assert utfall.flyttet == [ok1, ok2]
    assert [r for r, _ in utfall.hoppet_over] == [ukjent]
    session.expire_all()
    assert session.get(FailedAnalysis, ok2).image_legacy is None


def test_uten_r2_nekter_aa_kjoere(session, db_url):
    """Ingen halvveis modus: uten R2 er det ingenting aa flytte til."""
    from app.config import settings
    from app.services import legacy_bilder

    settings.cache_clear()
    with pytest.raises(RuntimeError, match="ikke konfigurert"):
        legacy_bilder.flytt(session, settings(), toerrkjoer=False)
