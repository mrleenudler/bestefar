"""kunngjoering av felt dyr

Legger til users.hunt_announced_at (§3).

Bare et tidsstempel. Art og sted lagres ALDRI - jaktloggen ligger i den
klient-krypterte bloben (§2), og en kolonne med «hva ble felt hvor» ville
gjenskapt den i klartekst paa serveren. Kolonnen finnes utelukkende for aa
kunne bremse gjentatte kunngjoeringer.

Haandskrevet, ikke autogenerert: autogenerate mot SQLite melder hele
`research`-skjemaet som en diff, siden skjemaet ikke finnes der.

Revision ID: c93f2a1d7b60
Revises: a7c14b93d502
"""
import sqlalchemy as sa
from alembic import op

revision = "c93f2a1d7b60"
down_revision = "a7c14b93d502"
branch_labels = None
depends_on = None

TZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.add_column("users", sa.Column("hunt_announced_at", TZ, nullable=True))


def downgrade() -> None:
    op.drop_column("users", "hunt_announced_at")
