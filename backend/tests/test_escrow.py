"""
Frivillig noekkeldeponering for backup-bloben (backend_spec §2, §13).

Det som maa vaere sant her, er ikke at endepunktet virker - det er at det
IKKE virker naar noe mangler: uten server-hemmelighet lagres ingenting, og
etter kontosletting finnes ingenting. Deponeringen er det eneste stedet
serveren har noe som kan aapne den klient-krypterte kopien.
"""
import base64

import pytest
from conftest import AUTH, USER_ID

MATERIALE = base64.b64encode(b"32 byte noekkelmateriale herfra").decode()
ANNEN = {"X-Debug-User-Id": "99999999-2222-3333-4444-555555555555"}


@pytest.fixture()
def paa(db_url, monkeypatch):
    """Konfigurerer hemmeligheten. Avhenger av db_url slik at den kjoerer
    foerst - ellers ville miljoeoppsettet der overskrive dette."""
    monkeypatch.setenv("BACKUP_ESCROW_SECRET", "test-escrow-hemmelighet")


def _put(client, materiale=MATERIALE, headers=AUTH):
    return client.put("/v1/backup/key-escrow", json={"key_material": materiale},
                      headers=headers)


def test_uten_hemmelighet_lagres_ingenting(client):
    """503, ikke klartekst. En manglende driftsinnstilling skal ikke kunne
    ende med noekler liggende lesbart i basen."""
    assert _put(client).status_code == 503
    assert client.get("/v1/backup/key-escrow", headers=AUTH).status_code == 503


def test_deponer_og_hent(paa, client):
    r = _put(client)
    assert r.status_code == 200
    assert r.json()["escrowed"] is True

    ut = client.get("/v1/backup/key-escrow", headers=AUTH)
    assert ut.status_code == 200
    assert ut.json()["key_material"] == MATERIALE


def test_materialet_ligger_kryptert_i_basen(paa, client, session):
    """Hele poenget med aa kryptere i ro: en databasedump alene skal ikke gi
    noekkelen. Hemmeligheten ligger som Fly-secret, ikke i basen."""
    _put(client)
    from app.models import BackupKeyEscrow
    lagret = session.query(BackupKeyEscrow).one().material
    assert b"noekkelmateriale" not in lagret
    assert base64.b64decode(MATERIALE) not in lagret


def test_ny_deponering_erstatter_forrige(paa, client):
    _put(client)
    ny = base64.b64encode(b"nytt materiale etter bytte av kode").decode()
    _put(client, ny)
    assert client.get("/v1/backup/key-escrow", headers=AUTH).json()["key_material"] == ny


def test_ingen_deponering_gir_404(paa, client):
    assert client.get("/v1/backup/key-escrow", headers=AUTH).status_code == 404


def test_sletting_er_idempotent(paa, client):
    _put(client)
    assert client.delete("/v1/backup/key-escrow", headers=AUTH).status_code == 204
    assert client.get("/v1/backup/key-escrow", headers=AUTH).status_code == 404
    assert client.delete("/v1/backup/key-escrow", headers=AUTH).status_code == 204


def test_sletting_virker_uten_hemmelighet(paa, client, monkeypatch):
    """Aa slutte aa deponere skal ALLTID gaa gjennom. Et 503 her ville laast
    brukeren inne i et valg hen vil ut av."""
    _put(client)
    monkeypatch.setenv("BACKUP_ESCROW_SECRET", "")
    from app.config import settings
    settings.cache_clear()
    assert client.delete("/v1/backup/key-escrow", headers=AUTH).status_code == 204


def test_rotert_hemmelighet_gir_503_ikke_soppel(paa, client, monkeypatch):
    """Materialet er uleselig, og da skal svaret si det - ikke levere ut byte
    som klienten proever aa dekryptere bloben med."""
    _put(client)
    monkeypatch.setenv("BACKUP_ESCROW_SECRET", "en helt annen hemmelighet")
    from app.config import settings
    settings.cache_clear()
    r = client.get("/v1/backup/key-escrow", headers=AUTH)
    assert r.status_code == 503
    assert "gjenopprettingskoden" in r.json()["detail"]


# --------------------------------------------------------------------
# Utskiftning av hemmeligheten
# --------------------------------------------------------------------

def _skift(monkeypatch, ny: str, gammel: str = "test-escrow-hemmelighet"):
    monkeypatch.setenv("BACKUP_ESCROW_SECRET", ny)
    monkeypatch.setenv("BACKUP_ESCROW_SECRET_OLD", gammel)
    from app.config import settings
    settings.cache_clear()


def test_gammel_hemmelighet_aapner_fortsatt_raden(paa, client, monkeypatch):
    """Uten dette er en utskiftning et stup: alt deponert materiale blir
    uleselig i samme oeyeblikk hemmeligheten byttes."""
    _put(client)
    _skift(monkeypatch, "ny-escrow-hemmelighet")
    r = client.get("/v1/backup/key-escrow", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["key_material"] == MATERIALE


def test_raden_krypteres_om_ved_lesing(paa, client, session, monkeypatch):
    """Utskiftningen migrerer seg selv etter hvert som radene leses, i stedet
    for aa kreve en engangsjobb ingen husker aa kjoere."""
    _put(client)
    from app.models import BackupKeyEscrow
    foer = session.get(BackupKeyEscrow, USER_ID).key_check

    _skift(monkeypatch, "ny-escrow-hemmelighet")
    client.get("/v1/backup/key-escrow", headers=AUTH)

    session.expire_all()
    rad = session.get(BackupKeyEscrow, USER_ID)
    assert rad.key_check != foer

    # Og naa aapnes den av den NYE alene - den gamle kan fjernes.
    monkeypatch.setenv("BACKUP_ESCROW_SECRET_OLD", "")
    from app.config import settings
    settings.cache_clear()
    assert client.get("/v1/backup/key-escrow",
                      headers=AUTH).json()["key_material"] == MATERIALE


def test_omkryptering_flytter_ikke_deponeringstidspunktet(paa, client,
                                                          monkeypatch):
    """`updated_at` skal bety «sist brukeren deponerte», ikke «sist serveren
    stelte med raden»."""
    deponert = _put(client).json()["updated_at"]
    _skift(monkeypatch, "ny-escrow-hemmelighet")
    assert client.get("/v1/backup/key-escrow",
                      headers=AUTH).json()["updated_at"] == deponert


def test_health_melder_fra_om_feil_hemmelighet(paa, client, monkeypatch):
    """En feilsatt hemmelighet skal dukke opp i overvaakningen, ikke hos en
    bruker midt i en gjenoppretting."""
    assert client.get("/health").json()["escrow"] == "ok"
    _put(client)
    assert client.get("/health").json()["escrow"] == "ok"

    # Byttet ut UTEN aa beholde den gamle - det verste tilfellet.
    monkeypatch.setenv("BACKUP_ESCROW_SECRET", "en helt annen hemmelighet")
    from app.config import settings
    settings.cache_clear()
    assert client.get("/health").json()["escrow"] == "1 rader paa annen hemmelighet"


def test_health_sier_av_uten_hemmelighet(client):
    assert client.get("/health").json()["escrow"] == "av"


def test_noekkel_id_roeper_ikke_hemmeligheten(paa, client, session):
    """Fingeravtrykket er HMAC over en KONSTANT streng. Det skal kunne ligge i
    basen uten aa si noe om hva hemmeligheten er."""
    _put(client)
    from app.models import BackupKeyEscrow
    kcv = session.get(BackupKeyEscrow, USER_ID).key_check
    assert kcv and "test-escrow-hemmelighet" not in kcv
    assert len(kcv) == 16


def test_deponering_er_isolert_per_bruker(paa, client):
    _put(client)
    assert client.get("/v1/backup/key-escrow", headers=ANNEN).status_code == 404


def test_ugyldig_base64_avvises(paa, client):
    assert _put(client, "ikke gyldig base64!!").status_code == 422


def test_for_stort_materiale_avvises(paa, client, monkeypatch):
    """Endepunktet skal ikke kunne brukes som en ekstra lagringsplass."""
    monkeypatch.setenv("MAX_ESCROW_BYTES", "32")
    from app.config import settings
    settings.cache_clear()
    assert _put(client, base64.b64encode(b"x" * 33).decode()).status_code == 413
    assert _put(client, base64.b64encode(b"x" * 32).decode()).status_code == 200


def test_meta_forteller_om_kopien_kan_gjenopprettes_uten_kode(paa, client):
    """En ny telefon skal kunne vite dette FOER den laster ned 16 MB."""
    client.put("/v1/backup", content=b"kryptert blob",
               params={"client_ts": "2026-08-01T12:00:00"}, headers=AUTH)
    assert client.get("/v1/backup/meta", headers=AUTH).json()["escrowed"] is False

    _put(client)
    assert client.get("/v1/backup/meta", headers=AUTH).json()["escrowed"] is True


def test_sletting_av_bloben_beholder_noekkelen(paa, client):
    """«Slett sikkerhetskopien» fjerner bloben, ikke innstillingen. Ellers
    ville neste opplasting stilltiende vaert udeponert med bryteren paa."""
    _put(client)
    client.delete("/v1/backup", headers=AUTH)
    assert client.get("/v1/backup/key-escrow", headers=AUTH).status_code == 200


def test_kontosletting_fjerner_deponeringen(paa, client, session):
    """§9/§2: det eneste serveren har hatt som kan aapne bloben, skal vaere
    borte naar kontoen er slettet."""
    _put(client)
    assert client.delete("/v1/account", headers=AUTH).status_code == 200

    from app.models import BackupKeyEscrow
    assert session.get(BackupKeyEscrow, USER_ID) is None


def test_uten_innlogging_svarer_401(paa, client):
    assert client.get("/v1/backup/key-escrow").status_code == 401
    assert client.put("/v1/backup/key-escrow",
                      json={"key_material": MATERIALE}).status_code == 401
