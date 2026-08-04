"""Lag: oppretting, invitasjon, lederskap og avstemning (§4, §11)."""
from conftest import AUTH

B = {"X-Debug-User-Id": "22222222-2222-3333-4444-555555555555"}
C = {"X-Debug-User-Id": "33333333-2222-3333-4444-555555555555"}

LAG = {"name": "Ringsaker jaktlag", "kind": "jakt", "lat": 60.9, "lon": 10.9}


def _uid(client, headers):
    return client.get("/v1/profile", headers=headers).json()["id"]


def _lag(client, headers=AUTH, **over):
    return client.post("/v1/teams", json={**LAG, **over}, headers=headers).json()


def _med_medlem(client, headers=B):
    """Oppretter lag med AUTH som leder og `headers` som medlem."""
    lag = _lag(client)
    inv = client.post(f"/v1/teams/{lag['id']}/invite",
                      json={"email_or_phone": "kari@example.com"},
                      headers=AUTH).json()
    client.post("/v1/teams/join", json={"token": inv["url"].rsplit("/", 1)[1]},
                headers=headers)
    return lag


# --------------------------------------------------------------------
# Oppretting (§4)
# --------------------------------------------------------------------

def test_oppretter_lag_med_seg_selv_som_leder(client):
    lag = _lag(client)
    assert lag["member_count"] == 1
    assert lag["leaders"] == [_uid(client, AUTH)]
    assert lag["has_leader"] is True


def test_opprett_for_leder_gir_lag_uten_leder(client):
    """§4: «opprett for leder» - oppretteren er medlem, men ikke leder."""
    lag = _lag(client, i_am_leader=False)
    assert lag["has_leader"] is False
    assert lag["member_count"] == 1


def test_egen_bruker_er_alltid_i_medlemslista(client):
    lag = _lag(client)
    detalj = client.get(f"/v1/teams/{lag['id']}", headers=AUTH).json()
    assert _uid(client, AUTH) in [m["user_id"] for m in detalj["members"]]


def test_ikke_medlem_ser_ikke_laget(client):
    """Vi avslorer ikke at laget finnes for noen som staar utenfor."""
    lag = _lag(client)
    assert client.get(f"/v1/teams/{lag['id']}", headers=B).status_code == 404


# --------------------------------------------------------------------
# Naerliggende lag (§4)
# --------------------------------------------------------------------

def test_naerliggende_sorteres_etter_avstand(client):
    _lag(client, name="Naert", lat=60.90, lon=10.90)
    _lag(client, name="Lengre", lat=60.99, lon=10.99)
    treff = client.get("/v1/teams/near", params={"lat": 60.9, "lon": 10.9},
                       headers=AUTH).json()
    assert [t["name"] for t in treff] == ["Naert", "Lengre"]
    assert treff[0]["distance_m"] < treff[1]["distance_m"]


def test_de_tre_naermeste_er_alltid_med(client):
    """§4: ellers ville en bruker i et tynt befolket omraade faatt tom liste."""
    for i in range(4):
        _lag(client, name=f"Fjernt {i}", lat=59.0 + i, lon=5.0)
    treff = client.get("/v1/teams/near",
                       params={"lat": 60.9, "lon": 10.9, "r": 1000},
                       headers=AUTH).json()
    assert len(treff) == 3


def test_lag_uten_koordinater_utelates(client):
    _lag(client, name="Uten sted", lat=None, lon=None)
    assert client.get("/v1/teams/near", params={"lat": 60.9, "lon": 10.9},
                      headers=AUTH).json() == []


# --------------------------------------------------------------------
# Invitasjon (§4)
# --------------------------------------------------------------------

def test_epost_invitasjon_gir_lenke_og_kvittering(client):
    lag = _lag(client)
    r = client.post(f"/v1/teams/{lag['id']}/invite",
                    json={"email_or_phone": "kari@example.com"}, headers=AUTH)
    assert r.status_code == 201
    assert r.json()["target_kind"] == "email"
    assert r.json()["delivery_status"] == "sent"
    assert r.json()["url"].startswith("http")


def test_invitasjonen_sendes_til_den_inviterte(client, monkeypatch):
    """Regresjonsvern: mailer.send() var skrevet for §10, der mottakeren alltid
    er utviklerens innboks. Uten `to=` havnet lag-invitasjonene der i stedet
    for hos den inviterte, og `delivery_status: sent` saa helt riktig ut."""
    from app.routers import teams as teams_router
    sendt: list = []
    monkeypatch.setattr(teams_router.mailer, "send",
                        lambda cfg, subject, body, reply_to=None, to=None:
                        sendt.append(to))

    lag = _lag(client)
    client.post(f"/v1/teams/{lag['id']}/invite",
                json={"email_or_phone": "kari@example.com"}, headers=AUTH)
    assert sendt == ["kari@example.com"]


def test_telefonnummer_normaliseres_og_melder_at_sms_mangler(client):
    lag = _lag(client)
    r = client.post(f"/v1/teams/{lag['id']}/invite",
                    json={"email_or_phone": "912 34 567"}, headers=AUTH).json()
    assert r["target_kind"] == "phone"
    assert r["target"] == "+4791234567"
    # SMS er utsatt til v2 - klienten skal faa vite det, og faar lenken uansett.
    assert r["delivery_status"] == "failed"
    assert "v2" in r["delivery_error"]
    assert r["url"].startswith("http")


def test_ugyldig_mottaker_avvises(client):
    lag = _lag(client)
    for verdi in ["ikke en adresse", "kari@", "12345"]:
        r = client.post(f"/v1/teams/{lag['id']}/invite",
                        json={"email_or_phone": verdi}, headers=AUTH)
        assert r.status_code == 422, verdi


def test_invitasjon_gir_medlemskap(client):
    lag = _med_medlem(client)
    detalj = client.get(f"/v1/teams/{lag['id']}", headers=B).json()
    assert detalj["member_count"] == 2


def test_redirect_velger_butikk_etter_user_agent(client):
    lag = _lag(client)
    inv = client.post(f"/v1/teams/{lag['id']}/invite",
                      json={"email_or_phone": "kari@example.com"},
                      headers=AUTH).json()
    sti = "/i/" + inv["url"].rsplit("/", 1)[1]

    android = client.get(sti, headers={"User-Agent": "Mozilla/5.0 (Linux; Android 14)"},
                         follow_redirects=False)
    assert android.status_code == 302
    assert "play.google.com" in android.headers["location"]

    ios = client.get(sti, headers={"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17)"},
                     follow_redirects=False)
    assert ios.status_code == 302


def test_ugyldig_token_avslorer_ingenting(client):
    """Lenken deles i aapne kanaler; svaret maa ikke skille gyldig fra ugyldig."""
    r = client.get("/i/finnes-ikke", follow_redirects=False)
    assert r.status_code == 302


# --------------------------------------------------------------------
# Navneendring og medlemmer (§11)
# --------------------------------------------------------------------

def test_navneendring_varsler_medlemmene(client):
    lag = _med_medlem(client)
    client.put(f"/v1/teams/{lag['id']}", json={"name": "Nytt navn"}, headers=AUTH)

    meldinger = client.get("/v1/messages", headers=B).json()
    navn = [m for m in meldinger if m["kind"] == "team_renamed"]
    assert len(navn) == 1
    assert "Nytt navn" in navn[0]["body"]
    # Lederen som gjorde endringen skal ikke varsle seg selv.
    assert not [m for m in client.get("/v1/messages", headers=AUTH).json()
                if m["kind"] == "team_renamed"]


def test_bare_leder_kan_endre_navn(client):
    lag = _med_medlem(client)
    assert client.put(f"/v1/teams/{lag['id']}", json={"name": "Kupp"},
                      headers=B).status_code == 403


def test_fjernet_medlem_varsles(client):
    lag = _med_medlem(client)
    b_id = _uid(client, B)
    assert client.delete(f"/v1/teams/{lag['id']}/members/{b_id}",
                         headers=AUTH).status_code == 204
    meldinger = client.get("/v1/messages", headers=B).json()
    assert any(m["kind"] == "removed_from_team" for m in meldinger)


def test_meldinger_kvitteres(client):
    lag = _med_medlem(client)
    client.put(f"/v1/teams/{lag['id']}", json={"name": "Nytt navn"}, headers=AUTH)
    meldinger = client.get("/v1/messages", headers=B).json()
    client.post("/v1/messages/ack", json={"ids": [m["id"] for m in meldinger]},
                headers=B)
    assert client.get("/v1/messages", headers=B).json() == []


# --------------------------------------------------------------------
# Lederskap (§11)
# --------------------------------------------------------------------

def test_lederskap_krever_bekreftelse(client):
    """Ingen skal vaakne opp som lagleder uten aa ha sagt ja."""
    lag = _med_medlem(client)
    b_id = _uid(client, B)
    r = client.post(f"/v1/teams/{lag['id']}/leaders/{b_id}", headers=AUTH)
    assert r.json()["status"] == "avventer_bekreftelse"
    assert client.get(f"/v1/teams/{lag['id']}", headers=AUTH).json()["leaders"] == [
        _uid(client, AUTH)]

    client.post(f"/v1/teams/{lag['id']}/leaders/confirm", headers=B)
    ledere = client.get(f"/v1/teams/{lag['id']}", headers=AUTH).json()["leaders"]
    assert set(ledere) == {_uid(client, AUTH), b_id}   # §4: flere ledere mulig


def test_bekreftelse_uten_tilbud_avvises(client):
    lag = _med_medlem(client)
    assert client.post(f"/v1/teams/{lag['id']}/leaders/confirm",
                       headers=B).status_code == 404


# --------------------------------------------------------------------
# Lederavstemning (§11)
# --------------------------------------------------------------------

def _lag_uten_leder(client):
    lag = _lag(client, i_am_leader=False)
    inv = client.post(f"/v1/teams/{lag['id']}/invite",
                      json={"email_or_phone": "kari@example.com"},
                      headers=AUTH).json()
    client.post("/v1/teams/join", json={"token": inv["url"].rsplit("/", 1)[1]},
                headers=B)
    return lag


def test_avstemning_krever_at_laget_mangler_leder(client):
    lag = _lag(client)
    assert client.post(f"/v1/teams/{lag['id']}/election",
                       headers=AUTH).status_code == 409


def test_avstemning_varsler_alle_og_har_syv_dagers_frist(client):
    lag = _lag_uten_leder(client)
    r = client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    assert r.status_code == 201

    status = client.get(f"/v1/teams/{lag['id']}/vote-status", headers=AUTH).json()
    assert status["outcome"] == "pending"
    assert 6 * 86400 < status["seconds_left"] <= 7 * 86400
    assert any(m["kind"] == "election_started"
               for m in client.get("/v1/messages", headers=B).json())


def test_stemme_kan_endres(client):
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    a_id, b_id = _uid(client, AUTH), _uid(client, B)

    client.post(f"/v1/teams/{lag['id']}/vote", json={"candidate_id": a_id},
                headers=AUTH)
    r = client.post(f"/v1/teams/{lag['id']}/vote", json={"candidate_id": b_id},
                    headers=AUTH)
    assert r.json()["my_vote"] == b_id
    assert r.json()["votes"] == {b_id: 1}


def test_enstemmighet_avslutter_tidlig(client):
    """§11: enstemmighet avslutter avstemningen foer fristen."""
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    b_id = _uid(client, B)

    client.post(f"/v1/teams/{lag['id']}/vote", json={"candidate_id": b_id},
                headers=AUTH)
    r = client.post(f"/v1/teams/{lag['id']}/vote", json={"candidate_id": b_id},
                    headers=B)
    assert r.json()["outcome"] == "elected"
    assert r.json()["elected_user_id"] == b_id
    assert client.get(f"/v1/teams/{lag['id']}", headers=AUTH).json()["leaders"] == [b_id]


def test_kandidat_maa_vaere_medlem(client):
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    client.get("/v1/profile", headers=C)          # oppretter bruker C
    assert client.post(f"/v1/teams/{lag['id']}/vote",
                       json={"candidate_id": _uid(client, C)},
                       headers=AUTH).status_code == 422


def test_uavgjort_ved_frist_gir_ingen_leder(client, session):
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    a_id, b_id = _uid(client, AUTH), _uid(client, B)
    client.post(f"/v1/teams/{lag['id']}/vote", json={"candidate_id": a_id}, headers=AUTH)
    client.post(f"/v1/teams/{lag['id']}/vote", json={"candidate_id": b_id}, headers=B)

    # Flytt fristen bakover i tid i stedet for aa vente sju dager.
    from datetime import timedelta
    from app.models import TeamElection, utcnow
    valg = session.query(TeamElection).one()
    valg.closes_at = utcnow() - timedelta(seconds=1)
    session.commit()

    status = client.get(f"/v1/teams/{lag['id']}/vote-status", headers=AUTH).json()
    assert status["outcome"] == "expired"
    assert client.get(f"/v1/teams/{lag['id']}", headers=AUTH).json()["has_leader"] is False


# --------------------------------------------------------------------
# Fjern inaktiv lagleder (§11)
# --------------------------------------------------------------------

def test_aktiv_leder_kan_ikke_utfordres(client):
    lag = _med_medlem(client)
    r = client.post(f"/v1/teams/{lag['id']}/leader-challenge", headers=B)
    assert r.status_code == 409
    assert r.json()["detail"] == "Lagleder er ikke inaktiv. Ta kontakt."


def _gjor_leder_inaktiv(client, session):
    from datetime import timedelta
    from app.models import User, utcnow
    a = session.get(User, AUTH["X-Debug-User-Id"])
    a.last_seen_at = utcnow() - timedelta(days=60)
    session.commit()


def test_inaktiv_leder_varsles_og_faar_syv_dager(client, session):
    lag = _med_medlem(client)
    _gjor_leder_inaktiv(client, session)
    r = client.post(f"/v1/teams/{lag['id']}/leader-challenge", headers=B)
    assert r.status_code == 201
    assert any(m["kind"] == "leader_challenged"
               for m in client.get("/v1/messages", headers=AUTH).json())


def test_leder_som_logger_paa_avbryter_prosessen(client, session):
    lag = _med_medlem(client)
    _gjor_leder_inaktiv(client, session)
    client.post(f"/v1/teams/{lag['id']}/leader-challenge", headers=B)

    client.get("/v1/profile", headers=AUTH)      # lederen bruker appen
    status = client.get(f"/v1/teams/{lag['id']}/leader-challenge", headers=B).json()
    assert status["outcome"] == "cancelled_leader_active"
    assert client.get(f"/v1/teams/{lag['id']}", headers=B).json()["has_leader"] is True


def test_frist_uten_paalogging_fjerner_lederstatus(client, session):
    lag = _med_medlem(client)
    _gjor_leder_inaktiv(client, session)
    client.post(f"/v1/teams/{lag['id']}/leader-challenge", headers=B)

    from datetime import timedelta
    from app.models import LeaderChallenge, utcnow
    ch = session.query(LeaderChallenge).one()
    ch.deadline_at = utcnow() - timedelta(seconds=1)
    session.commit()

    status = client.get(f"/v1/teams/{lag['id']}/leader-challenge", headers=B).json()
    assert status["outcome"] == "leader_demoted"
    detalj = client.get(f"/v1/teams/{lag['id']}", headers=B).json()
    assert detalj["has_leader"] is False
    # §11: lederen forblir MEDLEM.
    assert AUTH["X-Debug-User-Id"] in [m["user_id"] for m in detalj["members"]]


def test_uten_innlogging_svarer_401(client):
    assert client.get("/v1/teams").status_code == 401
    assert client.get("/v1/messages").status_code == 401
