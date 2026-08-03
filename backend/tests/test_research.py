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
