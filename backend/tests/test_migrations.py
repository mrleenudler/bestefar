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

    from app.db import normalize_url, schema_translate_map
    from app.models import Base

    url = normalize_url(db_url)
    kwargs = {}
    translate = schema_translate_map(url)
    if translate is not None:
        kwargs["execution_options"] = {"schema_translate_map": translate}
    engine = create_engine(url, **kwargs)

    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        diff = compare_metadata(ctx, Base.metadata)

    assert diff == [], f"Modellene og migrasjonene spriker: {diff}"
