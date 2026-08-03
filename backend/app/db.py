"""
Databasetilkobling. Engine lages LATT (foerste bruk), ikke ved import: da kan
prosessen starte - og /health svare - selv om databasen er nede eller
DATABASE_URL mangler. Viktig paa Fly, der en helsesjekk som feiler under
oppstart gir en deploy-loop uten forklaring.
"""
import logging
from collections.abc import Iterator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from .config import settings

log = logging.getLogger(__name__)

_engine: Engine | None = None
_sessionmaker: sessionmaker[Session] | None = None


def normalize_url(url: str) -> str:
    """Bruk psycopg3-driveren; Supabase/Fly oppgir bare `postgresql://`."""
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://"):
        url = "postgresql+psycopg://" + url[len("postgresql://"):]
    return url


def schema_translate_map(url: str) -> dict[str, str | None] | None:
    """
    Forskningstabellene ligger i skjemaet `research` (backend_spec §0/§7).
    SQLite har ikke skjemaer, saa der oversettes `research` til None og
    tabellene havner i hovedbasen - navnene kolliderer ikke.
    """
    from .models import RESEARCH_SCHEMA
    if url.startswith("sqlite"):
        return {RESEARCH_SCHEMA: None}
    return None


def engine() -> Engine:
    global _engine, _sessionmaker
    if _engine is None:
        cfg = settings()
        url = normalize_url(cfg.database_url)
        kwargs: dict = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        translate = schema_translate_map(url)
        if translate is not None:
            kwargs["execution_options"] = {"schema_translate_map": translate}
        _engine = create_engine(url, **kwargs)
        _sessionmaker = sessionmaker(bind=_engine)
        if cfg.auto_create_tables:
            # Bekvemmelighet lokalt. I produksjon eier Alembic skjemaet
            # (AUTO_CREATE_TABLES=false i fly.toml).
            from .models import RESEARCH_SCHEMA, Base
            if not url.startswith("sqlite"):
                with _engine.begin() as conn:
                    conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{RESEARCH_SCHEMA}"'))
            Base.metadata.create_all(_engine)
    return _engine


def db() -> Iterator[Session]:
    """FastAPI-avhengighet."""
    engine()
    assert _sessionmaker is not None
    s = _sessionmaker()
    try:
        yield s
    finally:
        s.close()


def ping() -> bool:
    try:
        with engine().connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception as exc:                       # noqa: BLE001 - rapporteres i /health
        log.warning("Databasen svarer ikke: %s", exc)
        return False
