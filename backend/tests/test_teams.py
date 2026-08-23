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


# --------------------------------------------------------------------
# «Hvem er jeg i dette laget» (issue #14)
# --------------------------------------------------------------------
#
# Uten disse to feltene kan klienten verken merke «(Du)» eller avgjoere om den
# skal vise «Rediger lag» eller «Velg leder». Alternativet - aa matche
# visningsnavn - feiler stille i lag der to personer heter det samme, og da paa
# den maaten at feil person faar lederknappene.

def _public_id(client, headers):
    return client.get("/v1/profile", headers=headers).json()["public_id"]


def test_my_role_sier_hva_jeg_er(client):
    lag = _med_medlem(client)
    som_leder = client.get(f"/v1/teams/{lag['id']}", headers=AUTH).json()
    som_medlem = client.get(f"/v1/teams/{lag['id']}", headers=B).json()

    assert som_leder["my_role"] == "leader"
    assert som_medlem["my_role"] == "member"


def test_my_role_staar_ogsaa_paa_lista(client):
    """Listeskjermen skal slippe aa hente detaljer for aa vite hva den tilbyr."""
    _med_medlem(client)
    assert client.get("/v1/teams", headers=AUTH).json()[0]["my_role"] == "leader"
    assert client.get("/v1/teams", headers=B).json()[0]["my_role"] == "member"


def test_my_role_er_none_for_den_som_staar_utenfor(client):
    """`/near` viser lag man ikke er medlem av. Da er svaret ingen rolle."""
    _lag(client)
    naer = client.get("/v1/teams/near?lat=60.9&lon=10.9", headers=C).json()
    assert naer and naer[0]["my_role"] is None


def test_medlemslista_baerer_public_id(client):
    """
    `public_id` er den eneste bruker-ID-en klienten har sett - den kommer i
    innloggingssvaret. Uten den i `members[]` finnes det ingen rad aa peke paa.
    """
    lag = _med_medlem(client)
    detalj = client.get(f"/v1/teams/{lag['id']}", headers=AUTH).json()
    per_public = {m["public_id"]: m for m in detalj["members"]}

    assert _public_id(client, AUTH) in per_public
    assert _public_id(client, B) in per_public
    assert per_public[_public_id(client, AUTH)]["role"] == "leader"
    # Den interne ID-en staar fortsatt, fordi mutasjonsrutene tar imot DEN.
    assert per_public[_public_id(client, B)]["user_id"] == _uid(client, B)


# --------------------------------------------------------------------
# App Links: /.well-known/assetlinks.json (§4)
# --------------------------------------------------------------------
#
# Google henter denne fila for aa avgjoere om appen faar aapne
# invitasjonslenken i stedet for nettleseren. Formkravene er eksterne, saa de
# testes: en fil som er «nesten riktig» verifiserer ikke, og symptomet er at
# lenken stille aapner nettleseren.

def test_assetlinks_har_formen_google_krever(client):
    r = client.get("/.well-known/assetlinks.json")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")

    doc = r.json()
    assert isinstance(doc, list) and len(doc) == 1      # toppnivaa er en LISTE
    assert doc[0]["relation"] == ["delegate_permission/common.handle_all_urls"]

    maal = doc[0]["target"]
    assert maal["namespace"] == "android_app"
    assert maal["package_name"] == "no.bestefar.app"
    # Release OG debug: App Links verifiserer mot signeringssertifikatet til
    # det INSTALLERTE bygget, saa uten debug-avtrykket kan et debug-bygg ikke
    # brukes til aa teste lenken. Debug-avtrykket skal ut ved lansering
    # (AAP-E8) - faller denne testen paa 1 i stedet for 2, er det trolig det
    # som er gjort, og da skal tallet her ned og ikke avtrykket inn igjen.
    assert len(maal["sha256_cert_fingerprints"]) == 2
    for avtrykk in maal["sha256_cert_fingerprints"]:
        # 32 byte som store hex-par med kolon mellom.
        assert len(avtrykk.split(":")) == 32, avtrykk


def test_assetlinks_krever_ikke_innlogging(client):
    """Verifisereren har ingen tokens. En 401 her ville vaert usynlig hos oss."""
    assert client.get("/.well-known/assetlinks.json").status_code == 200


def test_flere_avtrykk_kan_settes_uten_utrulling(client, monkeypatch):
    """
    Release, debug og - hvis Play App Signing er paa - Plays eget avtrykk maa
    kunne staa samtidig. Uten Plays avtrykk slutter lenkene aa virke i det
    oeyeblikket appen distribueres derfra.
    """
    from app.config import settings

    monkeypatch.setenv("ANDROID_CERT_FINGERPRINTS", "AA:BB, CC:DD")
    settings.cache_clear()
    try:
        avtrykk = client.get("/.well-known/assetlinks.json").json()[0]["target"]
        assert avtrykk["sha256_cert_fingerprints"] == ["AA:BB", "CC:DD"]
    finally:
        settings.cache_clear()


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


# --------------------------------------------------------------------
# Kvorum, absolutt frist og overkjoerte meldinger (Â§11-avklaringer 2026-08-09)
# --------------------------------------------------------------------

def _forfall(session, team_id, sekunder_siden=60):
    """Flytter fristen bakover saa avstemningen er forfalt uten aa vente."""
    from datetime import timedelta

    from app.models import TeamElection, utcnow
    valg = session.query(TeamElection).filter(
        TeamElection.team_id == team_id).order_by(
        TeamElection.opened_at.desc()).first()
    valg.closes_at = utcnow() - timedelta(seconds=sekunder_siden)
    session.commit()
    return valg


def test_kvorum_laases_ved_start_ikke_ved_avgjoerelse(client, session):
    """
    Kjernen i kvorumsregelen: medlemstallet maales NAAR avstemningen starter.
    Maalte vi ved avgjoerelse, kunne den som starter avstemningen senke
    terskelen ved aa fjerne medlemmer mens den paagaar.
    """
    lag = _lag_uten_leder(client)                      # AUTH + B = 2 medlemmer
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)

    status = client.get(f"/v1/teams/{lag['id']}/vote-status", headers=AUTH).json()
    assert status["member_count_at_open"] == 2
    assert status["quorum"] == 1                       # ceil(2 * 0.25)


def test_kvorum_rundes_opp_og_har_ingen_unntak_for_smaa_lag(client):
    """25 % av tre er 0,75 - og kvorumet er da 1, ikke 0."""
    lag = _lag(client, i_am_leader=False)              # ett medlem
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    status = client.get(f"/v1/teams/{lag['id']}/vote-status", headers=AUTH).json()
    assert status["member_count_at_open"] == 1
    assert status["quorum"] == 1


def test_under_kvorum_ved_fristen_gir_expired(client, session, monkeypatch):
    """
    For faa stemmer gir samme utfall som uavgjort: expired. Ingen leder kaares
    paa et mindretall, og laget kan proeve igjen med Ã©n gang.
    """
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)

    # Hev kvorumet kunstig: lat som laget hadde tolv medlemmer ved start, saa
    # kravet blir tre stemmer og den ene under er for lite.
    from app.models import TeamElection
    valg = session.query(TeamElection).filter(
        TeamElection.team_id == lag["id"]).one()
    valg.member_count_at_open = 12
    session.commit()

    client.post(f"/v1/teams/{lag['id']}/vote",
                json={"candidate_id": _uid(client, AUTH)}, headers=AUTH)
    _forfall(session, lag["id"])

    status = client.get(f"/v1/teams/{lag['id']}/vote-status", headers=AUTH).json()
    assert status["quorum"] == 3
    assert status["votes_cast"] == 1
    assert status["outcome"] == "expired"
    assert status["elected_user_id"] is None


def test_ingen_sperrefrist_etter_expired(client, session):
    """Et lederloest lag skal ikke hindres i aa proeve igjen med Ã©n gang."""
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    _forfall(session, lag["id"])
    # Foerste kall etter fristen avgjoer den late avstemningen.
    assert client.get(f"/v1/teams/{lag['id']}/vote-status",
                      headers=AUTH).json()["outcome"] == "expired"

    assert client.post(f"/v1/teams/{lag['id']}/election",
                       headers=AUTH).status_code == 201


def test_fristen_er_absolutt_stemme_etter_frist_avvises(client, session):
    """
    Lat avgjoerelse er en implementasjonsdetalj, ikke en utsettelse. Den foerste
    som ser paa avstemningen etter fristen faar RESULTATET - avstemningen
    reaapnes ikke fordi ingen saa paa den i tide.
    """
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    _forfall(session, lag["id"], sekunder_siden=3 * 86400)

    r = client.post(f"/v1/teams/{lag['id']}/vote",
                    json={"candidate_id": _uid(client, AUTH)}, headers=AUTH)
    assert r.status_code == 404                        # ingen paagaaende avstemning
    assert client.get(f"/v1/teams/{lag['id']}/vote-status",
                      headers=AUTH).json()["outcome"] == "expired"


def test_overkjoert_koemelding_leveres_ikke(client, session):
    """
    Â«Avstemningen er aapen i 7 dagerÂ» skal ikke dukke opp i koen etter at
    avstemningen er avgjort. Klienten henter koen ved appstart, saa den kunne
    ellers blitt vist ni dager for sent - rett over resultatet.
    """
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    assert any(m["kind"] == "election_started"
               for m in client.get("/v1/messages", headers=B).json())

    _forfall(session, lag["id"])
    client.get(f"/v1/teams/{lag['id']}/vote-status", headers=AUTH)

    koe = client.get("/v1/messages", headers=B).json()
    assert not any(m["kind"] == "election_started" for m in koe)


def test_resultatet_leveres_selv_om_varselet_ble_overkjoert(client, session):
    """Det er varselet om at noe PAAGAAR som annulleres, ikke utfallet."""
    lag = _lag_uten_leder(client)
    client.post(f"/v1/teams/{lag['id']}/election", headers=AUTH)
    a_id = _uid(client, AUTH)
    client.post(f"/v1/teams/{lag['id']}/vote",
                json={"candidate_id": a_id}, headers=AUTH)
    client.post(f"/v1/teams/{lag['id']}/vote",
                json={"candidate_id": a_id}, headers=B)   # enstemmig

    kinds = [m["kind"] for m in client.get("/v1/messages", headers=B).json()]
    assert "leader_elected" in kinds
    assert "election_started" not in kinds

