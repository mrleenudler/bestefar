"""
Flyktig kunngjoering av felt dyr (§3).

Det som MAA holde: ingenting om art eller sted skal bli liggende igjen paa
serveren. Hele poenget med lloesningen er at jaktloggen forblir i den
klient-krypterte bloben.
"""
import pytest
from conftest import AUTH, USER_ID

B = {"X-Debug-User-Id": "22222222-2222-3333-4444-555555555555"}


@pytest.fixture()
def sendte(client, monkeypatch) -> list:
    from app.services import push

    kall: list = []

    def falsk(cfg, tokens, title, body, data=None):
        kall.append({"tokens": list(tokens), "title": title, "body": body})
        return len(tokens), []

    monkeypatch.setattr(push, "send", falsk)
    return kall


def _venner(client):
    """AUTH og B blir venner, og B registrerer en enhet."""
    b_id = client.get("/v1/profile", headers=B).json()["id"]
    client.post("/v1/friends/request", json={"user_id": b_id}, headers=AUTH)
    req = client.get("/v1/friends/requests", headers=B).json()[0]
    client.post("/v1/friends/respond",
                json={"request_id": req["request_id"], "accept": True}, headers=B)
    client.put("/v1/devices", json={"push_token": "venn-token"}, headers=B)


def _slaa_paa(client):
    client.put("/v1/profile/sharing", json={"share_kills": True}, headers=AUTH)


def test_kunngjoering_naar_vennene(client, sendte):
    _venner(client)
    _slaa_paa(client)

    r = client.post("/v1/hunts/announce",
                    json={"species": "et villsvin", "kommune": "Molde"},
                    headers=AUTH)
    assert r.status_code == 200
    assert len(sendte) == 1
    assert sendte[0]["tokens"] == ["venn-token"]
    assert "villsvin" in sendte[0]["body"]
    assert "Molde" in sendte[0]["body"]


def test_krever_delingsvalget(client, sendte):
    _venner(client)
    r = client.post("/v1/hunts/announce", json={"species": "en elg"},
                    headers=AUTH)
    assert r.status_code == 403
    assert sendte == [], "Ingenting skal sendes uten samtykke"


def test_ingenting_lagres_om_dyret(client, sendte, session):
    """Kolonnen skal bare inneholde et tidsstempel. Art og sted lagret paa
    serveren ville gjenskapt jaktloggen i klartekst."""
    _venner(client)
    _slaa_paa(client)
    client.post("/v1/hunts/announce",
                json={"species": "et villsvin", "kommune": "Molde"},
                headers=AUTH)

    from app.models import PendingMessage, User
    assert session.query(PendingMessage).count() == 0, \
        "Kunngjoeringen er flyktig - ingen koerad"

    rad = session.get(User, USER_ID)
    assert rad.hunt_announced_at is not None
    lagret = " ".join(str(v) for v in vars(rad).values() if v is not None)
    assert "villsvin" not in lagret
    assert "Molde" not in lagret


def test_gjentatt_kunngjoering_bremses(client, sendte):
    _venner(client)
    _slaa_paa(client)
    assert client.post("/v1/hunts/announce", json={"species": "en elg"},
                       headers=AUTH).status_code == 200
    r = client.post("/v1/hunts/announce", json={"species": "en elg til"},
                    headers=AUTH)
    assert r.status_code == 429
    assert len(sendte) == 1


def test_uten_venner_er_det_ingen_feil(client, sendte):
    _slaa_paa(client)
    r = client.post("/v1/hunts/announce", json={"species": "en elg"},
                    headers=AUTH)
    assert r.status_code == 200
    assert r.json()["devices_notified"] == 0
    assert sendte == []


def test_ikke_godkjent_navn_lekker_ikke(client, sendte, session):
    """Et navn som ikke har passert moderasjonen skal ikke kunne snike seg ut
    i en push (§3)."""
    _venner(client)
    _slaa_paa(client)

    from app.models import NameStatus, User
    rad = session.get(User, USER_ID)
    rad.display_name = "Stygtnavn"
    rad.display_name_status = NameStatus.pending
    session.commit()

    client.post("/v1/hunts/announce", json={"species": "en elg"}, headers=AUTH)
    assert "Stygtnavn" not in sendte[0]["body"]


def test_uten_innlogging_svarer_401(client):
    assert client.post("/v1/hunts/announce",
                       json={"species": "en elg"}).status_code == 401
