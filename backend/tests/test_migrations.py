"""
Vokter mot drift mellom SQLAlchemy-modellene og Alembic-migrasjonene:
etter `upgrade head` skal autogenerate ikke finne noe aa gjoere.

Feiler denne testen har noen endret en modell uten aa lage migrasjon.
Fiks: `alembic revision --autogenerate -m "..."`.

Kjoerer bare mot Postgres. Paa SQLite oversettes skjemaet `research` til None
ved kjoering, men refleksjonen sammenlignes fortsatt mot modellenes
`schema='research'` - da rapporterer autogenerate en forskjell som ikke
finnes. CI kjoerer suiten mot Postgres (TEST_DATABASE_URL), som ogsaa er det
skjemaet produksjon faktisk bruker.
"""
import pytest


def test_modeller_og_migrasjoner_er_i_takt(db_url):
    if db_url.startswith("sqlite"):
        pytest.skip("Krever Postgres - se modulens docstring")

    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from sqlalchemy import create_engine

    from app.db import (alembic_compare_opts, normalize_url,
                        schema_translate_map)
    from app.models import Base

    url = normalize_url(db_url)
    kwargs = {}
    translate = schema_translate_map(url)
    if translate is not None:
        kwargs["execution_options"] = {"schema_translate_map": translate}
    engine = create_engine(url, **kwargs)

    # Samme innstillinger som migrations/env.py bruker - se docstringen der.
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts=alembic_compare_opts())
            diff = compare_metadata(ctx, Base.metadata)
    finally:
        engine.dispose()

    assert diff == [], f"Modellene og migrasjonene spriker: {diff}"
