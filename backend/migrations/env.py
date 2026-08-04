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
from app.db import alembic_compare_opts, normalize_url, schema_translate_map
from app.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

URL = normalize_url(settings().database_url)
COMPARE_OPTS = alembic_compare_opts()


def run_migrations_offline() -> None:
    context.configure(url=URL, target_metadata=target_metadata,
                      literal_binds=True, **COMPARE_OPTS,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    kwargs: dict = {}
    translate = schema_translate_map(URL)
    if translate is not None:
        kwargs["execution_options"] = {"schema_translate_map": translate}
    connectable = create_engine(URL, **kwargs)

    try:
        with connectable.connect() as connection:
            context.configure(connection=connection, target_metadata=target_metadata,
                              **COMPARE_OPTS)
            with context.begin_transaction():
                context.run_migrations()
    finally:
        # MAA lukkes. Ved deploy spiller det ingen rolle - prosessen doer
        # like etterpaa - men testene kjoerer env.py to ganger per test
        # (downgrade + upgrade), og en udisponert pool holder socketen
        # aapen. Mot Postgres i CI gikk basen tom for tilkoblinger midtveis
        # i suiten, med «connection failed» i stedet for en testfeil.
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
