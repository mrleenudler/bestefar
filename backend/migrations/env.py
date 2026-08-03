"""
Alembic-miljoe.

To ting er spesielle for dette prosjektet:
  1. Tilkoblingsstrengen kommer fra DATABASE_URL (app.config), ikke fra
     alembic.ini - hemmeligheter skal aldri ligge i repoet.
  2. Forskningstabellene ligger i skjemaet `research`. Paa Postgres opprettes
     skjemaet av foerste migrasjon; paa SQLite (tester) oversettes navnet til
     None via schema_translate_map, akkurat som i app/db.py.
"""
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine

from app.config import settings
from app.db import normalize_url, schema_translate_map
from app.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

URL = normalize_url(settings().database_url)


def run_migrations_offline() -> None:
    context.configure(url=URL, target_metadata=target_metadata,
                      literal_binds=True, compare_type=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    kwargs: dict = {}
    translate = schema_translate_map(URL)
    if translate is not None:
        kwargs["execution_options"] = {"schema_translate_map": translate}
    connectable = create_engine(URL, **kwargs)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
