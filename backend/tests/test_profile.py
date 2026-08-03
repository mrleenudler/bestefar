"""Profil og delingsvalg (backend_spec §1, §3)."""
from conftest import AUTH


def test_hent_profil_gir_gyldig_public_id(client):
    p = client.get("/v1/profile", headers=AUTH).json()
    from app.services import ids
    assert ids.is_valid(p["public_id"])


def test_endre_navn_godkjennes(client):
    r = client.put("/v1/profile", json={"display_name": "Ola Nordmann"}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["display_name"] == "Ola Nordmann"
    assert r.json()["display_name_status"] == "approved"


def test_norske_tegn_og_tegnsetting_godtas(client):
    for navn in ["Kåre Øståsen", "Ær-Å_B.C'D", "Bjørn 2"]:
        r = client.put("/v1/profile", json={"display_name": navn}, headers=AUTH)
        assert r.status_code == 200, navn


def test_navn_normaliseres(client):
    r = client.put("/v1/profile", json={"display_name": "  Ola    Nordmann "},
                   headers=AUTH)
    assert r.json()["display_name"] == "Ola Nordmann"


def test_ulovlige_tegn_avvises(client):
    """Klientfilteret er bekvemmelighet; en modifisert klient kan sende hva
    som helst, saa regelen maa haandheves her."""
    for navn in ["Ola 😀", "字", "Ola<script>"]:
        r = client.put("/v1/profile", json={"display_name": navn}, headers=AUTH)
        assert r.status_code == 422, navn


def test_for_langt_navn_avvises(client):
    r = client.put("/v1/profile", json={"display_name": "A" * 25}, headers=AUTH)
    assert r.status_code == 422


def test_tomt_navn_avvises(client):
    assert client.put("/v1/profile", json={"display_name": "   "},
                      headers=AUTH).status_code == 422


def test_blokkert_ord_avvises_og_lagres_ikke(client, monkeypatch):
    from app.config import settings
    monkeypatch.setenv("DISPLAY_NAME_BLOCKLIST", "stygt,verre")
    settings.cache_clear()

    client.put("/v1/profile", json={"display_name": "Ola"}, headers=AUTH)
    r = client.put("/v1/profile", json={"display_name": "Et stygt navn"}, headers=AUTH)
    assert r.status_code == 422
    # Avvist navn skal ikke ligge lagret - da kan det heller ikke lekke.
    assert client.get("/v1/profile", headers=AUTH).json()["display_name"] == "Ola"


def test_blocklist_ser_gjennom_omskriving(client, monkeypatch):
    from app.config import settings
    monkeypatch.setenv("DISPLAY_NAME_BLOCKLIST", "stygt")
    settings.cache_clear()
    for forsok in ["S-t-y-g-t", "STYGT", "stÿgt"]:
        r = client.put("/v1/profile", json={"display_name": forsok}, headers=AUTH)
        assert r.status_code == 422, forsok


def test_delingsvalg_er_av_som_standard(client):
    valg = client.get("/v1/profile/sharing", headers=AUTH).json()
    assert set(valg.values()) == {False}


def test_delingsvalg_kan_endres_enkeltvis(client):
    client.put("/v1/profile/sharing", json={"share_shots_total": True}, headers=AUTH)
    valg = client.get("/v1/profile/sharing", headers=AUTH).json()
    assert valg["share_shots_total"] is True
    assert valg["share_avg_score"] is False


def test_uten_innlogging_svarer_501(client):
    assert client.get("/v1/profile").status_code == 501
