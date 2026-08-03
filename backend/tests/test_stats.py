"""Treningsresultater (backend_spec §5)."""
from conftest import AUTH, USER_ID

SERIE = {
    "id": "aaaaaaaa-0000-0000-0000-000000000001",
    "ts": "2026-08-01T10:15:00",
    "weapon_id": "vaapen-1",
    "ammo_name": "Norma 9,7g",
    "distance_m": 100,
    "position": "LIGGENDE",
    "modifier": "ANLEGG",
    "season_key": "2026",
    "shots": [
        {"r_rel": 0.6, "theta": 1.2, "decimal": 10.4, "integer": 10},
        {"r_rel": 1.2, "theta": 2.0, "decimal": 9.8, "integer": 9},
    ],
}


def _put(client, body=None):
    body = body or SERIE
    return client.put(f"/v1/stats/series/{body['id']}", json=body, headers=AUTH)


def test_serie_lagres_med_utregnet_sum(client):
    r = _put(client)
    assert r.status_code == 200
    assert r.json() == {"id": SERIE["id"], "created": True}

    rows = client.get("/v1/stats/series", headers=AUTH).json()
    assert len(rows) == 1
    assert rows[0]["sum_decimal"] == 20.2
    assert rows[0]["sum_integer"] == 19
    assert rows[0]["shot_count"] == 2
    assert rows[0]["position"] == "LIGGENDE"


def test_opplasting_er_idempotent(client):
    """Klienten koer usendte serier og kan sende samme serie flere ganger."""
    _put(client)
    r = _put(client)
    assert r.json()["created"] is False
    assert len(client.get("/v1/stats/series", headers=AUTH).json()) == 1


def test_ny_opplasting_erstatter_treffene(client):
    _put(client)
    rettet = {**SERIE, "corrected": True,
              "shots": [{"r_rel": 0.6, "theta": 1.2, "decimal": 10.4, "integer": 10}]}
    _put(client, rettet)

    rows = client.get("/v1/stats/series", headers=AUTH).json()
    assert len(rows) == 1
    assert rows[0]["shot_count"] == 1
    assert rows[0]["sum_decimal"] == 10.4
    assert rows[0]["corrected"] is True


def test_annen_brukers_serie_er_usynlig(client):
    _put(client)
    annen = {"X-Debug-User-Id": "99999999-2222-3333-4444-555555555555"}
    assert client.get("/v1/stats/series", headers=annen).json() == []
    # ...og kan ikke overskrives - vi avsloerer ikke at ID-en finnes.
    r = client.put(f"/v1/stats/series/{SERIE['id']}", json=SERIE, headers=annen)
    assert r.status_code == 404


def test_filtrering_paa_sesong(client):
    _put(client)
    _put(client, {**SERIE, "id": "aaaaaaaa-0000-0000-0000-000000000002",
                  "season_key": "2025"})
    assert len(client.get("/v1/stats/series?season_key=2026", headers=AUTH).json()) == 1


def test_uten_innlogging_svarer_501(client):
    """Auth kommer i fase 3; til da skal endepunktet ikke vaere brukbart."""
    assert client.get("/v1/stats/series").status_code == 501
    assert USER_ID  # brukt av de andre testene via AUTH
