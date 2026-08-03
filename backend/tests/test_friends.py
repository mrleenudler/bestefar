"""Venner: soek, forespoersel/aksept og utgaaende filtrering (§3, §3.1)."""
from conftest import AUTH

B = {"X-Debug-User-Id": "22222222-2222-3333-4444-555555555555"}
C = {"X-Debug-User-Id": "33333333-2222-3333-4444-555555555555"}


def _oppsett(client, headers, navn, **profil):
    client.put("/v1/profile", json={"display_name": navn, **profil}, headers=headers)
    return client.get("/v1/profile", headers=headers).json()


def _bli_venner(client, a=AUTH, b=B):
    b_profil = client.get("/v1/profile", headers=b).json()
    r = client.post("/v1/friends/request", json={"user_id": b_profil["id"]}, headers=a)
    req = client.get("/v1/friends/requests", headers=b).json()[0]
    client.post("/v1/friends/respond",
                json={"request_id": req["request_id"], "accept": True}, headers=b)
    return r


# --------------------------------------------------------------------
# Soek (§3.1)
# --------------------------------------------------------------------

def test_soek_paa_bruker_id(client):
    p = _oppsett(client, B, "Kari")
    r = client.get("/v1/users/search", params={"q": p["public_id"]}, headers=AUTH)
    assert r.status_code == 200
    assert r.json()["display_name"] == "Kari"


def test_soek_paa_telefon(client):
    _oppsett(client, B, "Kari", phone="+4791234567")
    r = client.get("/v1/users/search", params={"q": "+4791234567"}, headers=AUTH)
    assert r.status_code == 200


def test_ikke_findable_er_usynlig(client):
    p = _oppsett(client, B, "Kari", phone="+4791234567", findable=False)
    assert client.get("/v1/users/search", params={"q": p["public_id"]},
                      headers=AUTH).status_code == 404
    assert client.get("/v1/users/search", params={"q": "+4791234567"},
                      headers=AUTH).status_code == 404


def test_tastefeil_i_id_slaas_ikke_opp(client):
    """Sjekksifferet fanger den foer vi roerer databasen - og den skal ikke
    telle som et mislykket soek."""
    r = client.get("/v1/users/search", params={"q": "BF-1111-1111"}, headers=AUTH)
    assert r.status_code == 422


def test_gjentatte_bom_paa_telefon_gir_karantene(client):
    """§3.1: 5 mislykkede telefonsoek paa ett doegn -> karantene."""
    for _ in range(5):
        assert client.get("/v1/users/search", params={"q": "+4790000000"},
                          headers=AUTH).status_code == 404
    r = client.get("/v1/users/search", params={"q": "+4790000000"}, headers=AUTH)
    assert r.status_code == 429
    assert "sperret_til" in r.json()["detail"]


def test_karantene_stopper_ogsaa_gyldige_soek(client):
    p = _oppsett(client, B, "Kari")
    for _ in range(5):
        client.get("/v1/users/search", params={"q": "+4790000000"}, headers=AUTH)
    assert client.get("/v1/users/search", params={"q": p["public_id"]},
                      headers=AUTH).status_code == 429


def test_treff_teller_ikke_som_bom(client):
    p = _oppsett(client, B, "Kari")
    for _ in range(10):
        assert client.get("/v1/users/search", params={"q": p["public_id"]},
                          headers=AUTH).status_code == 200


def test_karantene_overlever_omstart(client, session):
    """Telleren ligger i databasen, ikke i minnet - ellers er den ingen
    karantene paa en flermaskins-oppsett."""
    for _ in range(5):
        client.get("/v1/users/search", params={"q": "+4790000000"}, headers=AUTH)
    from app.models import QuarantineScope, SearchQuarantine
    rader = session.query(SearchQuarantine).filter(
        SearchQuarantine.scope == QuarantineScope.account).all()
    assert len(rader) == 1 and rader[0].quarantined_until is not None


# --------------------------------------------------------------------
# Forespoersel og aksept (§3)
# --------------------------------------------------------------------

def test_ingen_data_deles_for_aksept(client):
    _oppsett(client, B, "Kari")
    b_id = client.get("/v1/profile", headers=B).json()["id"]
    client.post("/v1/friends/request", json={"user_id": b_id}, headers=AUTH)
    assert client.get("/v1/friends", headers=AUTH).json() == []
    assert client.get("/v1/friends", headers=B).json() == []


def test_aksept_gjor_begge_til_venner(client):
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari")
    _bli_venner(client)
    assert [v["display_name"] for v in client.get("/v1/friends", headers=AUTH).json()] == ["Kari"]
    assert [v["display_name"] for v in client.get("/v1/friends", headers=B).json()] == ["Ola"]


def test_avslag_deler_ingenting(client):
    _oppsett(client, B, "Kari")
    b_id = client.get("/v1/profile", headers=B).json()["id"]
    client.post("/v1/friends/request", json={"user_id": b_id}, headers=AUTH)
    req = client.get("/v1/friends/requests", headers=B).json()[0]
    client.post("/v1/friends/respond",
                json={"request_id": req["request_id"], "accept": False}, headers=B)
    assert client.get("/v1/friends", headers=AUTH).json() == []


def test_bare_mottaker_kan_svare(client):
    _oppsett(client, B, "Kari")
    _oppsett(client, C, "Per")
    b_id = client.get("/v1/profile", headers=B).json()["id"]
    client.post("/v1/friends/request", json={"user_id": b_id}, headers=AUTH)
    req = client.get("/v1/friends/requests", headers=B).json()[0]

    for hvem in (AUTH, C):
        r = client.post("/v1/friends/respond",
                        json={"request_id": req["request_id"], "accept": True},
                        headers=hvem)
        assert r.status_code == 404


def test_krysset_forespoersel_blir_vennskap(client):
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari")
    a_id = client.get("/v1/profile", headers=AUTH).json()["id"]
    b_id = client.get("/v1/profile", headers=B).json()["id"]

    client.post("/v1/friends/request", json={"user_id": b_id}, headers=AUTH)
    r = client.post("/v1/friends/request", json={"user_id": a_id}, headers=B)
    assert r.json()["status"] == "accepted"


def test_kan_ikke_legge_til_seg_selv(client):
    a_id = client.get("/v1/profile", headers=AUTH).json()["id"]
    assert client.post("/v1/friends/request", json={"user_id": a_id},
                       headers=AUTH).status_code == 422


def test_fjern_venn(client):
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari")
    _bli_venner(client)
    b_id = client.get("/v1/profile", headers=B).json()["id"]
    assert client.delete(f"/v1/friends/{b_id}", headers=AUTH).status_code == 204
    assert client.get("/v1/friends", headers=AUTH).json() == []
    assert client.get("/v1/friends", headers=B).json() == []


# --------------------------------------------------------------------
# Utgaaende filtrering (§3)
# --------------------------------------------------------------------

SERIE = {"ts": "2026-08-01T10:00:00", "distance_m": 100, "position": "LIGGENDE",
         "season_key": "2026",
         "shots": [{"r_rel": 0.6, "theta": 1.2, "decimal": 10.0, "integer": 10},
                   {"r_rel": 1.2, "theta": 2.0, "decimal": 9.0, "integer": 9}]}


def _serie(client, headers, nr):
    sid = f"bbbbbbbb-0000-0000-0000-{nr:012d}"
    client.put(f"/v1/stats/series/{sid}", json={**SERIE, "id": sid}, headers=headers)


def test_ingen_felt_deles_uten_samtykke(client):
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari", phone="+4791234567", home_kommune="Ringsaker")
    _serie(client, B, 1)
    _bli_venner(client)

    venn = client.get("/v1/friends", headers=AUTH).json()[0]
    assert venn["display_name"] == "Kari"          # deles alltid (§3)
    for felt in ("phone", "home_kommune", "shots_total", "avg_score", "trend"):
        assert felt not in venn


def test_valgte_felt_deles(client):
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari", phone="+4791234567", home_kommune="Ringsaker")
    _serie(client, B, 1)
    client.put("/v1/profile/sharing",
               json={"share_phone": True, "share_home_kommune": True,
                     "share_shots_total": True, "share_avg_score": True},
               headers=B)
    _bli_venner(client)

    venn = client.get("/v1/friends", headers=AUTH).json()[0]
    assert venn["phone"] == "+4791234567"
    assert venn["home_kommune"] == "Ringsaker"
    assert venn["shots_total"] == 2
    assert venn["avg_score"] == 9.5


def test_deaktivering_nuller_delte_felt(client):
    """§3: naar deling slaas av, forsvinner feltet umiddelbart - serveren
    filtrerer utgaaende, saa det ligger ingen kopi hos vennen."""
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari", phone="+4791234567")
    client.put("/v1/profile/sharing", json={"share_phone": True}, headers=B)
    _bli_venner(client)
    assert "phone" in client.get("/v1/friends", headers=AUTH).json()[0]

    client.put("/v1/profile/sharing", json={"share_phone": False}, headers=B)
    assert "phone" not in client.get("/v1/friends", headers=AUTH).json()[0]


def test_trend_krever_nok_data(client):
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari")
    client.put("/v1/profile/sharing", json={"share_trend": True}, headers=B)
    for nr in range(1, 6):
        _serie(client, B, nr)
    _bli_venner(client)

    # Under 10 serier gir ingen trend - to serier ville vaert stoey presentert
    # som innsikt.
    assert client.get("/v1/friends", headers=AUTH).json()[0]["trend"] is None

    for nr in range(6, 11):
        _serie(client, B, nr)
    assert client.get("/v1/friends", headers=AUTH).json()[0]["trend"] == 0.0


def test_uvurdert_navn_eksponeres_ikke(client, session):
    """Er navnet ikke godkjent, ser andre en noeytral plassholder (§3)."""
    _oppsett(client, AUTH, "Ola")
    _oppsett(client, B, "Kari")
    _bli_venner(client)

    from app.models import NameStatus, User
    b = session.get(User, B["X-Debug-User-Id"])
    b.display_name_status = NameStatus.pending
    session.commit()

    assert client.get("/v1/friends", headers=AUTH).json()[0]["display_name"] \
        == "Ukjent skytter"


def test_uten_innlogging_svarer_501(client):
    assert client.get("/v1/friends").status_code == 501
    assert client.get("/v1/users/search", params={"q": "BF-7Q4K-9F2M"}).status_code == 501
