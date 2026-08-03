"""Backup / dataoverfoering (backend_spec §2)."""
from conftest import AUTH

BLOB = b"\x00\x01kryptert innhold som serveren ikke kan lese\xff"
ANNEN = {"X-Debug-User-Id": "99999999-2222-3333-4444-555555555555"}


def _put(client, blob=BLOB, ts="2026-08-01T12:00:00", headers=AUTH, **q):
    params = {"client_ts": ts, "schema_version": 1, "device_id": "telefon-1", **q}
    return client.put("/v1/backup", content=blob, params=params, headers=headers)


def test_last_opp_og_hent_ned_gir_identiske_bytes(client):
    r = _put(client)
    assert r.status_code == 200
    assert r.json()["bytes"] == len(BLOB)

    ned = client.get("/v1/backup", headers=AUTH)
    assert ned.status_code == 200
    assert ned.content == BLOB
    assert ned.headers["x-backup-device-id"] == "telefon-1"
    assert ned.headers["x-backup-schema-version"] == "1"


def test_ny_opplasting_erstatter_forrige(client):
    _put(client)
    _put(client, b"nyere", ts="2026-08-02T12:00:00")
    assert client.get("/v1/backup", headers=AUTH).content == b"nyere"


def test_metadata_uten_aa_laste_ned_bloben(client):
    _put(client)
    meta = client.get("/v1/backup/meta", headers=AUTH)
    assert meta.status_code == 200
    assert meta.json()["bytes"] == len(BLOB)
    assert meta.json()["device_id"] == "telefon-1"


def test_ingen_backup_gir_404(client):
    assert client.get("/v1/backup", headers=AUTH).status_code == 404
    assert client.get("/v1/backup/meta", headers=AUTH).status_code == 404


def test_utdatert_enhet_far_ikke_overskrive(client):
    """En gammel telefon som synker foerst etter maaneder skal ikke kunne
    viske ut alt som er logget siden. Serveren ser ikke inn i bloben, saa
    §2-regelen om last-write-wins per post-ID hjelper ikke her."""
    _put(client, b"nytt", ts="2026-08-02T12:00:00")

    r = _put(client, b"gammelt", ts="2026-07-01T12:00:00")
    assert r.status_code == 409
    assert client.get("/v1/backup", headers=AUTH).content == b"nytt"


def test_force_overstyrer_vernet(client):
    """«Gjenopprett fra denne enheten» er et bevisst brukervalg."""
    _put(client, b"nytt", ts="2026-08-02T12:00:00")
    r = _put(client, b"gammelt", ts="2026-07-01T12:00:00", force=True)
    assert r.status_code == 200
    assert client.get("/v1/backup", headers=AUTH).content == b"gammelt"


def test_samme_tidsstempel_godtas(client):
    """Retry etter et avbrutt kall skal ikke avvises."""
    _put(client)
    assert _put(client).status_code == 200


def test_tom_backup_avvises(client):
    assert _put(client, b"").status_code == 422


def test_for_stor_backup_avvises(client, monkeypatch):
    from app.config import settings
    monkeypatch.setenv("MAX_BACKUP_BYTES", "32")
    settings.cache_clear()
    assert _put(client, b"x" * 33).status_code == 413
    assert _put(client, b"x" * 32).status_code == 200


def test_backup_er_isolert_per_bruker(client):
    _put(client)
    assert client.get("/v1/backup", headers=ANNEN).status_code == 404

    _put(client, b"annen brukers blob", headers=ANNEN)
    assert client.get("/v1/backup", headers=AUTH).content == BLOB


def test_sletting(client):
    _put(client)
    assert client.delete("/v1/backup", headers=AUTH).status_code == 204
    assert client.get("/v1/backup", headers=AUTH).status_code == 404
    # Sletting av noe som ikke finnes skal ikke feile.
    assert client.delete("/v1/backup", headers=AUTH).status_code == 204


def test_uten_innlogging_svarer_501(client):
    assert client.get("/v1/backup").status_code == 501
    assert client.put("/v1/backup", content=BLOB,
                      params={"client_ts": "2026-08-01T12:00:00"}).status_code == 501
