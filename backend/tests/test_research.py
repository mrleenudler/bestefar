"""
Forskning (backend_spec §7). Tre sperrer: aktiveringsflagg, pseudonym-
hemmelighet og samtykke.
"""
import pytest
from conftest import AUTH

RECORD = {"session_ref": "oekt-1", "captured_at": "2026-08-01T10:15:00",
          "result_type": "training", "payload": {"note": "test"}}


@pytest.fixture()
def paa(db_url, monkeypatch):
    """Slaar paa forskningsinnsamling for denne testen.

    Avhenger av db_url slik at den kjoerer FOERST og setter flagget av;
    ellers ville standardverdien derfra overskrive dette.
    """
    monkeypatch.setenv("RESEARCH_ENABLED", "true")


def test_avslaatt_som_standard(client):
    """§7/§9: innsamling krever personvernerklaering og avklart DPIA-behov."""
    r = client.post("/v1/research/consent", json={"consent_type": "training"},
                    headers=AUTH)
    assert r.status_code == 503
    assert "DPIA" in r.json()["detail"]


def test_uten_pseudonym_hemmelighet_lagres_ingenting(paa, monkeypatch, client):
    monkeypatch.setenv("RESEARCH_PSEUDONYM_SECRET", "")
    from app.config import settings
    settings.cache_clear()
    r = client.post("/v1/research/consent", json={"consent_type": "training"},
                    headers=AUTH)
    assert r.status_code == 503


def test_innsending_krever_samtykke(paa, client):
    assert client.post("/v1/research/records", json=RECORD, headers=AUTH).status_code == 403


def test_samtykke_deretter_innsending(paa, client):
    assert client.post("/v1/research/consent", json={"consent_type": "training"},
                       headers=AUTH).status_code == 201
    assert client.post("/v1/research/records", json=RECORD, headers=AUTH).status_code == 201


def test_samtykke_kan_trekkes_tilbake_og_fornyes(paa, client):
    client.post("/v1/research/consent", json={"consent_type": "training"}, headers=AUTH)

    assert client.delete("/v1/research/consent/training", headers=AUTH).status_code == 200
    assert client.post("/v1/research/records", json=RECORD, headers=AUTH).status_code == 403

    r = client.post("/v1/research/consent", json={"consent_type": "training"}, headers=AUTH)
    assert r.json()["status"] == "fornyet"
    assert client.post("/v1/research/records", json=RECORD, headers=AUTH).status_code == 201


def test_samtykke_gjelder_kun_sin_egen_resultattype(paa, client):
    """Trening og jakt har ulik personvernprofil og deles hver for seg (§7)."""
    client.post("/v1/research/consent", json={"consent_type": "training"}, headers=AUTH)
    jakt = {**RECORD, "result_type": "hunt"}
    assert client.post("/v1/research/records", json=jakt, headers=AUTH).status_code == 403


def test_forskningsraden_inneholder_ingen_konto_id(paa, client, session):
    client.post("/v1/research/consent", json={"consent_type": "training"}, headers=AUTH)
    client.post("/v1/research/records", json=RECORD, headers=AUTH)

    from conftest import USER_ID
    from app.models import ResearchRecord
    row = session.query(ResearchRecord).one()
    assert row.pseudonym_id != USER_ID
    assert USER_ID not in row.pseudonym_id


# --------------------------------------------------------------------
# Delingsvalg og filtrering av jaktdata (§7)
# --------------------------------------------------------------------

JAKT = {"session_ref": "jakt-1", "captured_at": "2026-09-21T07:30:00",
        "result_type": "hunt",
        "payload": {"species": "elg", "shot_situation": "staaende dyr",
                    "shooting_position": "STAAENDE", "distance_m": 85,
                    "lat": 60.1234, "lon": 10.5678,
                    "kommune": "Kongsberg", "fylke": "Buskerud",
                    "injury": "skadeskutt", "tracking_distance_m": 400}}


def _jakt_samtykke(client):
    assert client.post("/v1/research/consent", json={"consent_type": "hunt"},
                       headers=AUTH).status_code == 201


def test_ingenting_deles_som_standard(paa, client):
    """§7: valgene starter avslaatt. Et samtykke til jakt-typen betyr ikke at
    hvert felt i den er delt."""
    assert client.get("/v1/research/sharing", headers=AUTH).json() == {
        "share_species": False, "share_date": False,
        "share_shot_situation": False, "share_injury_data": False,
        "position_granularity": "none"}

    _jakt_samtykke(client)
    r = client.post("/v1/research/records", json=JAKT, headers=AUTH)
    assert r.status_code == 201
    assert r.json()["stored_fields"] == []


def test_bare_valgte_felt_lagres(paa, client, session):
    client.put("/v1/research/sharing", headers=AUTH,
               json={"share_species": True, "share_shot_situation": True})
    _jakt_samtykke(client)
    client.post("/v1/research/records", json=JAKT, headers=AUTH)

    from app.models import ResearchRecord
    lagret = session.query(ResearchRecord).one().payload_json
    assert set(lagret) == {"species", "shot_situation", "shooting_position",
                           "distance_m"}


def test_skadedata_krever_sin_egen_bryter(paa, client, session):
    """§7: skadedata er private som standard. Alle de ANDRE bryterne paa
    endrer ikke det - de er skilt fra hverandre med vilje, fordi «jeg skjoet
    staaende paa 85 meter» og «dyret ble skadeskutt» ikke er samme opplysning."""
    client.put("/v1/research/sharing", headers=AUTH,
               json={"share_species": True, "share_date": True,
                     "share_shot_situation": True,
                     "position_granularity": "exact"})
    _jakt_samtykke(client)
    client.post("/v1/research/records", json=JAKT, headers=AUTH)

    from app.models import ResearchRecord
    lagret = session.query(ResearchRecord).one().payload_json
    assert "injury" not in lagret
    assert "tracking_distance_m" not in lagret


def test_skadedata_lagres_naar_bryteren_er_paa(paa, client, session):
    """Ettersoeksdata er den mest verdifulle delen av materialet, og skal
    kunne deles av den som vil."""
    client.put("/v1/research/sharing", headers=AUTH,
               json={"share_injury_data": True})
    _jakt_samtykke(client)
    r = client.post("/v1/research/records", json=JAKT, headers=AUTH)
    assert set(r.json()["stored_fields"]) == {"injury", "tracking_distance_m"}

    from app.models import ResearchRecord
    lagret = session.query(ResearchRecord).one().payload_json
    # Bryteren aapner BARE skadedata - art og sted staar fortsatt av.
    assert "species" not in lagret
    assert "kommune" not in lagret


def test_ukjent_felt_droppes(paa, client, session):
    """Tillatelsesliste, ikke forbudsliste: feltinnholdet er ikke endelig
    avklart, og et nytt felt skal ikke vaere delt til noen forbyr det."""
    client.put("/v1/research/sharing", headers=AUTH,
               json={"share_species": True})
    _jakt_samtykke(client)
    client.post("/v1/research/records", headers=AUTH,
                json={**JAKT, "payload": {"species": "elg",
                                          "helt_nytt_felt": "hemmelig"}})

    from app.models import ResearchRecord
    assert set(session.query(ResearchRecord).one().payload_json) == {"species"}


@pytest.mark.parametrize("grovhet,forventet", [
    ("none", set()),
    ("fylke", {"fylke"}),
    ("kommune", {"kommune", "fylke"}),
    ("exact", {"lat", "lon", "kommune", "fylke"}),
])
def test_posisjonsgrovhet_styrer_stedsfelt(paa, client, session, grovhet, forventet):
    client.put("/v1/research/sharing", headers=AUTH,
               json={"position_granularity": grovhet})
    _jakt_samtykke(client)
    client.post("/v1/research/records", headers=AUTH,
                json={**JAKT, "payload": {k: JAKT["payload"][k]
                                          for k in ("lat", "lon", "kommune", "fylke")}})

    from app.models import ResearchRecord
    assert set(session.query(ResearchRecord).one().payload_json) == forventet


def test_uten_datosamtykke_beholdes_bare_aaret(paa, client, session):
    _jakt_samtykke(client)
    client.post("/v1/research/records", json=JAKT, headers=AUTH)

    from app.models import ResearchRecord
    lagret = session.query(ResearchRecord).one().captured_at
    assert (lagret.year, lagret.month, lagret.day) == (2026, 1, 1)


def test_med_datosamtykke_beholdes_datoen(paa, client, session):
    client.put("/v1/research/sharing", headers=AUTH, json={"share_date": True})
    _jakt_samtykke(client)
    client.post("/v1/research/records", json=JAKT, headers=AUTH)

    from app.models import ResearchRecord
    lagret = session.query(ResearchRecord).one().captured_at
    assert (lagret.month, lagret.day) == (9, 21)


def test_treningsdata_filtreres_ikke(paa, client, session):
    """§7 gir ingen felt-for-felt-valg for trening - samtykket gjelder hele
    resultattypen."""
    client.post("/v1/research/consent", json={"consent_type": "training"},
                headers=AUTH)
    client.post("/v1/research/records", headers=AUTH,
                json={**RECORD, "payload": {"note": "test", "hva som helst": 1}})

    from app.models import ResearchRecord
    assert set(session.query(ResearchRecord).one().payload_json) == {
        "note", "hva som helst"}
