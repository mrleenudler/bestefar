"""
Kontosletting (backend_spec §9).

Det som gjoer denne verdt aa teste noeye: brukerraden slettes IKKE, den
toemmes. En test som bare sjekket «finnes raden?» ville derfor passert selv om
telefonnummeret laa igjen.
"""
import pytest
from conftest import AUTH, USER_ID


@pytest.fixture()
def forskning_paa(db_url, monkeypatch):
    monkeypatch.setenv("RESEARCH_ENABLED", "true")


SERIE = {
    "id": "aaaaaaaa-0000-0000-0000-000000000001",
    "ts": "2026-08-01T10:15:00",
    "weapon_id": "vaapen-1",
    "distance_m": 100,
    "position": "LIGGENDE",
    "modifier": "ANLEGG",
    "season_key": "2026",
    "shots": [{"r_rel": 0.6, "theta": 1.2, "decimal": 10.4, "integer": 10}],
}


def test_sletting_toemmer_persondata(client, session):
    client.put("/v1/profile", headers=AUTH,
               json={"display_name": "Ola", "phone": "+4791234567",
                     "birth_year": 1975, "home_kommune": "Kongsberg"})

    assert client.delete("/v1/account", headers=AUTH).status_code == 200

    from app.models import User
    rad = session.get(User, USER_ID)
    # Raden MAA staa igjen: public_id skal ikke kunne gjenbrukes av en ny
    # konto, ellers ser en venn plutselig en fremmed.
    assert rad is not None
    assert rad.public_id
    assert rad.deleted_at is not None
    assert rad.phone is None
    assert rad.birth_year is None
    assert rad.home_kommune is None
    assert rad.findable is False
    assert "Ola" not in rad.display_name


def test_sletting_fjerner_treningsdata_og_backup(client, session):
    client.put(f"/v1/stats/series/{SERIE['id']}", json=SERIE, headers=AUTH)
    client.put("/v1/backup", content=b"kryptert", headers=AUTH,
               params={"client_ts": "2026-08-01T12:00:00", "schema_version": 1,
                       "device_id": "telefon-1"})

    client.delete("/v1/account", headers=AUTH)

    from app.models import Backup, Series, Shot
    assert session.query(Series).count() == 0
    assert session.query(Shot).count() == 0, "Treffene skal foelge serien"
    assert session.query(Backup).count() == 0


def test_slettet_konto_gir_ikke_tilgang(client):
    """Access-tokenet kan ikke tilbakekalles, saa `deleted_at` MAA sjekkes ved
    hvert kall - ellers har den slettede kontoen tilgang en time til."""
    client.delete("/v1/account", headers=AUTH)
    assert client.get("/v1/profile", headers=AUTH).status_code == 401


def test_sletting_legger_inn_sletteanmodning_til_forskning(forskning_paa, client, session):
    client.post("/v1/research/consent", json={"consent_type": "training"},
                headers=AUTH)

    r = client.delete("/v1/account", headers=AUTH)
    assert r.json()["research_deletion_requested"] is True

    from app.models import ResearchConsent, ResearchDeletionRequest
    anmodning = session.query(ResearchDeletionRequest).one()
    # Anmodningen skal IKKE inneholde konto-ID-en - da hadde §7-adskillelsen
    # vaert brutt av selve slettingen.
    assert USER_ID not in anmodning.pseudonym_id
    assert session.query(ResearchConsent).one().revoked_at is not None


def test_sletting_uten_pseudonym_hemmelighet(client, monkeypatch, session):
    """Uten hemmeligheten kunne det aldri vaert lagret forskningsdata fra
    denne brukeren, saa det er ingenting aa be om - men kontoen skal
    fortsatt slettes."""
    monkeypatch.setenv("RESEARCH_PSEUDONYM_SECRET", "")
    from app.config import settings
    settings.cache_clear()

    r = client.delete("/v1/account", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["research_deletion_requested"] is False

    from app.models import ResearchDeletionRequest, User
    assert session.query(ResearchDeletionRequest).count() == 0
    assert session.get(User, USER_ID).deleted_at is not None


def test_sletting_fjerner_lagmedlemskap(client, session):
    client.post("/v1/teams", headers=AUTH,
                json={"name": "Testlaget", "kind": "hunt"})
    client.delete("/v1/account", headers=AUTH)

    from app.models import TeamMember
    assert session.query(TeamMember).count() == 0
