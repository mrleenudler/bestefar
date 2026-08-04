import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


def _reset_app_modules() -> None:
    """Settings er lru_cache-et og engine ligger paa modulnivaa - begge maa
    bort mellom tester saa hver test faar sin egen database."""
    for mod in [m for m in list(sys.modules) if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]


@pytest.fixture()
def db_url(tmp_path, monkeypatch) -> str:
    """Tom database per test, bygget av Alembic - ikke create_all.

    Da gaar testene gjennom noeyaktig den samme veien som produksjon, og en
    modellendring uten migrasjon slaar ut her i stedet for ved deploy.

    Settes TEST_DATABASE_URL (CI kjoerer mot Postgres) brukes den i stedet for
    en SQLite-fil, og basen rulles tilbake foer hver test. Postgres er det
    eneste stedet skjemaet `research` finnes pa ekte, saa driftsjekken i
    test_migrations.py kan bare kjoere der.
    """
    external = os.environ.get("TEST_DATABASE_URL")
    url = external or f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    monkeypatch.setenv("AUTO_CREATE_TABLES", "false")
    monkeypatch.setenv("ENV", "dev")
    monkeypatch.setenv("FEEDBACK_TO", "")
    monkeypatch.setenv("RESEARCH_ENABLED", "false")
    monkeypatch.setenv("RESEARCH_PSEUDONYM_SECRET", "test-hemmelighet")
    _reset_app_modules()

    from alembic import command
    from alembic.config import Config

    cfg = Config(str(BACKEND / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND / "migrations"))
    if external:
        command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    # Alembics env.py importerer app.config og varmer opp lru_cache-en i
    # settings(). Nullstill paa nytt, saa fixtures som kjoerer ETTER denne
    # (f.eks. `paa` i test_research.py) faar sine miljoevariabler lest.
    _reset_app_modules()
    yield url

    # Lukk forbindelsene testen aapnet. `_reset_app_modules` kaster bare
    # modulen ut av sys.modules; engine-en lever videre med poolen sin og
    # holder socketene aapne til GC en gang tar den. Mot Postgres i CI gikk
    # basen tom for tilkoblinger midt i suiten.
    mod = sys.modules.get("app.db")
    if mod is not None and getattr(mod, "_engine", None) is not None:
        mod._engine.dispose()


@pytest.fixture()
def client(db_url):
    from fastapi.testclient import TestClient

    from app import db as database
    from app.main import app

    database._engine = None
    database._sessionmaker = None
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def session(db_url):
    from app.db import db
    s = next(db())
    try:
        yield s
    finally:
        s.close()


# Fast bruker-ID for de midlertidige debug-headerne (deps.current_user).
USER_ID = "11111111-2222-3333-4444-555555555555"
AUTH = {"X-Debug-User-Id": USER_ID}
