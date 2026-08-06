"""
Innlogging (backend_spec §1).

Google og Apple testes med en erstattet verifikator: en ekte test ville krevd
et gyldig, ferskt ID-token fra leverandoeren, som ikke kan lages i CI. Selve
verifiseringen (signatur, iss, aud) er PyJWTs ansvar; det vi tester her er det
VI har ansvar for - at en verifisert identitet blir riktig konto, at
kontosammenslaaing skjer paa verifisert e-post, og at ingenting slipper
gjennom uten konfigurasjon.
"""
import re

import pytest


def _kode(sendte: list) -> str:
    treff = re.search(r"\b(\d{6})\b", sendte[-1][1])
    assert treff, f"Fant ingen kode i e-posten: {sendte[-1][1]}"
    return treff.group(1)


@pytest.fixture()
def sendte(client, monkeypatch) -> list:
    """Fanger e-post i stedet for aa sende den. Returnerer (mottaker, tekst)."""
    from app.routers import auth as auth_router
    ut: list = []

    def fange(cfg, subject, body, reply_to=None, to=None):
        ut.append((to, body))

    monkeypatch.setattr(auth_router.mailer, "send", fange)
    return ut


@pytest.fixture()
def falsk_google(monkeypatch):
    """Erstatter Google-verifiseringen. Kalles med (sub, epost, verifisert)."""
    from app.routers import auth as auth_router
    from app.services import oidc

    def sett(sub: str, epost: str | None, verifisert: bool = True) -> None:
        ident = oidc.Identitet(provider=oidc.Provider.google, subject=sub,
                               email=epost, email_verifisert=verifisert)
        monkeypatch.setattr(auth_router.oidc, "verifiser_google",
                            lambda cfg, token: ident)

    return sett


# --------------------------------------------------------------------
# E-post med engangskode
# --------------------------------------------------------------------

def test_epostinnlogging_gir_tokens(client, sendte):
    assert client.post("/v1/auth/email/start",
                       json={"email": "ola@example.com"}).status_code == 202
    assert sendte[-1][0] == "ola@example.com", "Koden maa gaa til brukeren"

    r = client.post("/v1/auth/email/verify",
                    json={"email": "ola@example.com", "code": _kode(sendte)})
    assert r.status_code == 200
    data = r.json()
    assert data["is_new"] is True
    assert data["token_type"] == "Bearer"
    assert data["access_token"] and data["refresh_token"]


def test_access_token_gir_tilgang(client, sendte):
    client.post("/v1/auth/email/start", json={"email": "kari@example.com"})
    data = client.post("/v1/auth/email/verify",
                       json={"email": "kari@example.com",
                             "code": _kode(sendte)}).json()

    h = {"Authorization": f"Bearer {data['access_token']}"}
    r = client.get("/v1/profile", headers=h)
    assert r.status_code == 200
    assert r.json()["id"] == data["user_id"]


def test_nytt_navn_er_ferdig_moderert(client, sendte):
    """
    Navnet som utledes av e-postadressen er allerede kjoert gjennom
    moderasjonen, saa statusen maa si `approved`. Sto den paa standardverdien
    `pending`, viste sharing.friend_view «Navn under vurdering» til venner og
    lagkamerater for hver eneste nye konto - og ingenting satte den noen gang,
    med mindre brukeren tilfeldigvis lagret profilen sin paa nytt.
    """
    client.post("/v1/auth/email/start", json={"email": "nina@example.com"})
    data = client.post("/v1/auth/email/verify",
                       json={"email": "nina@example.com",
                             "code": _kode(sendte)}).json()

    h = {"Authorization": f"Bearer {data['access_token']}"}
    profil = client.get("/v1/profile", headers=h).json()
    assert profil["display_name"] == "nina"
    assert profil["display_name_status"] == "approved"


def test_andre_gangs_innlogging_gir_samme_konto(client, sendte):
    client.post("/v1/auth/email/start", json={"email": "per@example.com"})
    foerste = client.post("/v1/auth/email/verify",
                          json={"email": "per@example.com",
                                "code": _kode(sendte)}).json()
    client.post("/v1/auth/email/start", json={"email": "per@example.com"})
    andre = client.post("/v1/auth/email/verify",
                        json={"email": "per@example.com",
                              "code": _kode(sendte)}).json()
    assert andre["user_id"] == foerste["user_id"]
    assert andre["is_new"] is False


def test_feil_kode_avvises_og_brenner_forsoek(client, sendte):
    client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    for _ in range(5):
        r = client.post("/v1/auth/email/verify",
                        json={"email": "ola@example.com", "code": "000000"})
        assert r.status_code == 401
    # Sekssifret kode taaler ikke ubegrenset proeving.
    r = client.post("/v1/auth/email/verify",
                    json={"email": "ola@example.com", "code": "000000"})
    assert r.status_code == 429


def test_brukt_kode_kan_ikke_brukes_om_igjen(client, sendte):
    client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    kode = _kode(sendte)
    assert client.post("/v1/auth/email/verify",
                       json={"email": "ola@example.com", "code": kode}).status_code == 200
    assert client.post("/v1/auth/email/verify",
                       json={"email": "ola@example.com", "code": kode}).status_code == 401


def test_ukjent_adresse_roeper_ingenting(client, sendte):
    """Svaret maa vaere likt for kjent og ukjent adresse (§3.1-tankegang):
    ellers er endepunktet et oppslagsverk over hvem som bruker appen."""
    a = client.post("/v1/auth/email/start", json={"email": "ny@example.com"})
    client.post("/v1/auth/email/verify",
                json={"email": "ny@example.com", "code": _kode(sendte)})
    b = client.post("/v1/auth/email/start", json={"email": "ny@example.com"})
    assert a.status_code == b.status_code == 202
    assert a.json() == b.json()


def test_for_mange_koder_ratebegrenses(client, sendte):
    # Grensen leses fra konfigurasjonen. Hardkodet tall her betyr at en
    # justering av kvoten gir en rod test i stedet for et svar paa om
    # begrensningen virker.
    from app.config import settings
    grense = settings().email_code_rate_per_hour

    for _ in range(grense):
        assert client.post("/v1/auth/email/start",
                           json={"email": "spam@example.com"}).status_code == 202
    assert client.post("/v1/auth/email/start",
                       json={"email": "spam@example.com"}).status_code == 429


# --------------------------------------------------------------------
# Google / Apple
# --------------------------------------------------------------------

def test_google_uten_klient_id_er_ikke_konfigurert(client):
    """Uten GOOGLE_CLIENT_IDS kan vi ikke sjekke `aud`, og da skal vi ikke
    slippe noen inn."""
    r = client.post("/v1/auth/google", json={"id_token": "hva som helst"})
    assert r.status_code == 503


def test_google_gir_konto(client, falsk_google):
    falsk_google("google-sub-1", "ola@example.com")
    r = client.post("/v1/auth/google", json={"id_token": "x"})
    assert r.status_code == 200
    assert r.json()["is_new"] is True

    r2 = client.post("/v1/auth/google", json={"id_token": "x"})
    assert r2.json()["user_id"] == r.json()["user_id"]
    assert r2.json()["is_new"] is False


def test_verifisert_epost_kobles_til_samme_konto(client, sendte, falsk_google):
    """Samme person med Google og e-postkode skal ha ÉN konto."""
    client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    via_epost = client.post("/v1/auth/email/verify",
                            json={"email": "ola@example.com",
                                  "code": _kode(sendte)}).json()

    falsk_google("google-sub-2", "ola@example.com", verifisert=True)
    via_google = client.post("/v1/auth/google", json={"id_token": "x"}).json()
    assert via_google["user_id"] == via_epost["user_id"]
    assert via_google["is_new"] is False


def test_uverifisert_epost_kobles_ikke(client, sendte, falsk_google):
    """Ellers kunne en konto hos en slapp leverandoer, opprettet med en annen
    persons adresse, overta kontoen deres."""
    client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    via_epost = client.post("/v1/auth/email/verify",
                            json={"email": "ola@example.com",
                                  "code": _kode(sendte)}).json()

    falsk_google("google-sub-3", "ola@example.com", verifisert=False)
    via_google = client.post("/v1/auth/google", json={"id_token": "x"}).json()
    assert via_google["user_id"] != via_epost["user_id"]
    assert via_google["is_new"] is True


# --------------------------------------------------------------------
# Fornyelse og utlogging
# --------------------------------------------------------------------

def test_refresh_gir_nytt_par_og_roterer(client, sendte):
    client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    foerste = client.post("/v1/auth/email/verify",
                          json={"email": "ola@example.com",
                                "code": _kode(sendte)}).json()

    r = client.post("/v1/auth/refresh",
                    json={"refresh_token": foerste["refresh_token"]})
    assert r.status_code == 200
    nytt = r.json()
    assert nytt["refresh_token"] != foerste["refresh_token"]
    assert nytt["user_id"] == foerste["user_id"]


def test_gjenbrukt_refresh_token_tilbakekaller_alt(client, sendte):
    client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    foerste = client.post("/v1/auth/email/verify",
                          json={"email": "ola@example.com",
                                "code": _kode(sendte)}).json()
    nytt = client.post("/v1/auth/refresh",
                       json={"refresh_token": foerste["refresh_token"]}).json()

    # Det gamle tokenet dukker opp igjen: enten en kopi paa avveie eller et
    # dobbeltkjoert forsoek. Vi kan ikke skille, saa alt tilbakekalles.
    assert client.post("/v1/auth/refresh",
                       json={"refresh_token": foerste["refresh_token"]}).status_code == 401
    assert client.post("/v1/auth/refresh",
                       json={"refresh_token": nytt["refresh_token"]}).status_code == 401


def test_utlogging_er_idempotent(client, sendte):
    client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    data = client.post("/v1/auth/email/verify",
                       json={"email": "ola@example.com",
                             "code": _kode(sendte)}).json()

    for _ in range(2):
        assert client.post("/v1/auth/logout",
                           json={"refresh_token": data["refresh_token"]}).status_code == 204
    assert client.post("/v1/auth/logout",
                       json={"refresh_token": "finnes-ikke"}).status_code == 204
    assert client.post("/v1/auth/refresh",
                       json={"refresh_token": data["refresh_token"]}).status_code == 401


# --------------------------------------------------------------------
# Tokenverifisering
# --------------------------------------------------------------------

def test_tullete_token_avvises(client):
    for verdi in ["Bearer noe.som.ikke.er.et.token", "Basic abc", "Bearer "]:
        r = client.get("/v1/profile", headers={"Authorization": verdi})
        assert r.status_code == 401, verdi


def test_token_signert_med_feil_noekkel_avvises(client):
    """Signaturen MAA sjekkes - ellers kan hvem som helst skrive sin egen
    `sub` og bli hvem som helst."""
    import jwt
    from app.models import utcnow
    falskt = jwt.encode({"sub": "11111111-2222-3333-4444-555555555555",
                         "iss": "bestefar-api",
                         "exp": int(utcnow().timestamp()) + 3600},
                        "feil-noekkel-som-ogsaa-er-minst-32-tegn", algorithm="HS256")
    r = client.get("/v1/profile", headers={"Authorization": f"Bearer {falskt}"})
    assert r.status_code == 401


def test_utloept_token_avvises(client, sendte):
    import jwt
    from app.models import utcnow
    from app.config import settings
    utloept = jwt.encode({"sub": "11111111-2222-3333-4444-555555555555",
                          "iss": "bestefar-api",
                          "exp": int(utcnow().timestamp()) - 10},
                         settings().jwt_secret, algorithm="HS256")
    r = client.get("/v1/profile", headers={"Authorization": f"Bearer {utloept}"})
    assert r.status_code == 401


@pytest.mark.parametrize("hemmelighet", ["", "for-kort"])
def test_daarlig_signeringsnoekkel_svarer_503(client, monkeypatch, hemmelighet):
    """Tom noekkel kan vi ikke signere med, og en kort HMAC-noekkel kan brutes
    - da kan hvem som helst utstede tokens for hvilken som helst bruker."""
    monkeypatch.setenv("JWT_SECRET", hemmelighet)
    from app.config import settings
    settings.cache_clear()
    r = client.post("/v1/auth/email/start", json={"email": "ola@example.com"})
    assert r.status_code == 503
