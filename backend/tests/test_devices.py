"""
Push-registrering og utsending (backend_spec §11, fase 8).

Det viktigste her er ikke at push virker, men at det ALDRI er kritisk:
meldingskoeen er garantien, push er bekvemmeligheten. Flere av testene sjekker
derfor at en feilende eller frakoblet FCM ikke velter lagoperasjonen.
"""
import pytest
from conftest import AUTH

B = {"X-Debug-User-Id": "22222222-2222-3333-4444-555555555555"}

ENHET = {"push_token": "fcm-token-1", "platform": "android",
         "app_version": "0.15", "model": "Pixel 8"}


@pytest.fixture()
def sendte(client, monkeypatch) -> list:
    """Fanger push-utsendingene i stedet for aa gaa mot FCM."""
    from app.services import push

    kall: list = []

    def falsk(cfg, tokens, title, body, data=None):
        kall.append({"tokens": list(tokens), "title": title, "body": body,
                     "data": data or {}})
        return len(tokens), []

    monkeypatch.setattr(push, "send", falsk)
    return kall


# --------------------------------------------------------------------
# Registrering
# --------------------------------------------------------------------

def test_registrerer_enhet(client):
    r = client.put("/v1/devices", json=ENHET, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["platform"] == "android"


def test_registrering_er_idempotent(client, session):
    client.put("/v1/devices", json=ENHET, headers=AUTH)
    client.put("/v1/devices", json={**ENHET, "app_version": "0.16"}, headers=AUTH)

    from app.models import Device
    rader = session.query(Device).all()
    assert len(rader) == 1, "Samme token skal ikke gi to rader"
    assert rader[0].app_version == "0.16", "Siste registrering skal vinne"


def test_token_foelger_med_til_ny_konto(client, session):
    """Samme telefon, ny innlogging: varsler til den forrige brukeren skal
    IKKE fortsette aa havne paa en telefon som naa er noen andres."""
    client.put("/v1/devices", json=ENHET, headers=AUTH)
    client.put("/v1/devices", json=ENHET, headers=B)

    from app.models import Device
    rad = session.query(Device).one()
    assert rad.user_id != client.get("/v1/profile", headers=AUTH).json()["id"]


def test_listen_roeper_ikke_push_tokenet(client):
    client.put("/v1/devices", json=ENHET, headers=AUTH)
    rader = client.get("/v1/devices", headers=AUTH).json()
    assert len(rader) == 1
    assert rader[0]["model"] == "Pixel 8"
    assert "push_token" not in rader[0]


def test_ser_bare_egne_enheter(client):
    client.put("/v1/devices", json=ENHET, headers=AUTH)
    assert client.get("/v1/devices", headers=B).json() == []


def test_avregistrering_er_idempotent(client, session):
    client.put("/v1/devices", json=ENHET, headers=AUTH)
    assert client.post("/v1/devices/unregister", json={"push_token": "fcm-token-1"},
                       headers=AUTH).status_code == 204
    # Ukjent token skal ogsaa gi 204 - avregistrering skal aldri feile.
    assert client.post("/v1/devices/unregister", json={"push_token": "finnes-ikke"},
                       headers=AUTH).status_code == 204

    from app.models import Device
    assert session.query(Device).count() == 0


def test_kan_ikke_avregistrere_andres_enhet(client, session):
    """Uten eierfilteret kunne hvem som helst skrudd av varslene til en annen
    ved aa gjette et token."""
    client.put("/v1/devices", json=ENHET, headers=AUTH)
    assert client.post("/v1/devices/unregister", json={"push_token": "fcm-token-1"},
                       headers=B).status_code == 204

    from app.models import Device
    assert session.query(Device).count() == 1, "Enheten skal staa uroert"


def test_uten_innlogging_svarer_401(client):
    assert client.put("/v1/devices", json=ENHET).status_code == 401
    assert client.get("/v1/devices").status_code == 401


# --------------------------------------------------------------------
# Kobling mot §11-varslene
# --------------------------------------------------------------------

def _lag_med_medlem(client):
    lag = client.post("/v1/teams", json={"name": "Ringsaker jaktlag", "kind": "jakt"},
                      headers=AUTH).json()
    inv = client.post(f"/v1/teams/{lag['id']}/invite",
                      json={"email_or_phone": "kari@example.com"},
                      headers=AUTH).json()
    client.post("/v1/teams/join", json={"token": inv["url"].rsplit("/", 1)[1]},
                headers=B)
    return lag


def test_navneendring_gir_push_til_medlemmene(client, sendte):
    lag = _lag_med_medlem(client)
    client.put("/v1/devices", json={"push_token": "medlem-token"}, headers=B)
    sendte.clear()

    client.put(f"/v1/teams/{lag['id']}", json={"name": "Nytt navn"}, headers=AUTH)

    assert len(sendte) == 1
    assert sendte[0]["tokens"] == ["medlem-token"]
    assert "Nytt navn" in sendte[0]["body"]
    # Klienten trenger aa vite HVA varselet gjelder for aa kunne aapne riktig
    # skjerm naar brukeren trykker paa det.
    assert sendte[0]["data"]["team_id"] == lag["id"]
    assert sendte[0]["data"]["kind"]


def test_koeen_faar_meldingen_selv_uten_enheter(client, sendte, session):
    """Ingen registrert enhet er ikke en feil - koeen baerer meldingen."""
    lag = _lag_med_medlem(client)
    client.put(f"/v1/teams/{lag['id']}", json={"name": "Nytt navn"}, headers=AUTH)

    assert sendte == [], "Ingen enheter => ingen push-kall"
    assert client.get("/v1/messages", headers=B).json(), "Koeen skal ha meldingen"


def test_doede_tokens_ryddes_bort(client, monkeypatch, session):
    """FCM sier fra naar en app er avinstallert. Rydder vi ikke, vokser
    tabellen med adresser vi aldri naar - og hver av dem koster et kall av
    push-budsjettet ved neste varsel."""
    from app.services import push
    monkeypatch.setattr(push, "send",
                        lambda cfg, tokens, *a, **k: (0, list(tokens)))

    lag = _lag_med_medlem(client)
    client.put("/v1/devices", json={"push_token": "doedt-token"}, headers=B)
    client.put(f"/v1/teams/{lag['id']}", json={"name": "Nytt navn"}, headers=AUTH)

    from app.models import Device
    assert session.query(Device).count() == 0


def test_push_feil_velter_ikke_lagoperasjonen(client, monkeypatch):
    """push.send skal aldri kaste. Skulle den likevel gjoere det, er det
    viktigere at navneendringen gaar gjennom enn at varselet naar fram."""
    from app.services import push

    def eksploderer(*a, **k):
        raise RuntimeError("FCM er nede")

    monkeypatch.setattr(push, "send", eksploderer)
    lag = _lag_med_medlem(client)
    client.put("/v1/devices", json={"push_token": "medlem-token"}, headers=B)

    r = client.put(f"/v1/teams/{lag['id']}", json={"name": "Nytt navn"},
                     headers=AUTH)
    assert r.status_code == 200
    assert client.get("/v1/messages", headers=B).json()


# --------------------------------------------------------------------
# push.py uten Firebase konfigurert
# --------------------------------------------------------------------

def test_uten_konfigurasjon_logges_push_bare(db_url):
    from app.config import settings
    from app.services import push

    cfg = settings()
    assert push.backend_name(cfg) == "log"
    assert push.send(cfg, ["a", "b"], "Tittel", "Tekst") == (0, [])


def test_health_rapporterer_push(client):
    assert client.get("/health").json()["push"] == "log"


def test_ugyldig_tjenestekonto_slaar_av_push(db_url, monkeypatch):
    """Feil i JSON-en skal gi «av», ikke en kraesj ved oppstart."""
    monkeypatch.setenv("FCM_SERVICE_ACCOUNT_JSON", "{ikke json")
    from app.config import settings
    settings.cache_clear()
    from app.services import push
    assert push.backend_name(settings()) == "log"
