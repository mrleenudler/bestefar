"""kvorum paa lederavstemning + overkjoerte koemeldinger

To kolonner, begge fra §11-avklaringene 2026-08-09:

`team_elections.member_count_at_open` - medlemstallet da avstemningen startet.
Kvorumet (25 %, rundet opp) regnes av DETTE tallet og ikke av medlemstallet ved
avgjoerelse. Maalte vi ved avgjoerelse, kunne terskelen senkes ved aa fjerne
medlemmer mens avstemningen paagaar - og den som starter en avstemning i et
lederloest lag er ofte den samme som kan fjerne folk.

server_default "0" fordi rader fra foer kolonnen fantes ikke kan vite hva
medlemstallet var. `kvorum()` gir 0 for dem, altsaa ingen kvorumskrav: en
avstemning som allerede loeper skal ikke kunne bli ugyldig av en migrasjon.

`pending_messages.superseded_at` - satt naar meldingen er overkjoert av et
senere utfall. Klienten henter koen ved APPSTART, saa «avstemningen er aapen i 7
dager» kan bli hentet ni dager senere og vist rett over «avstemningen er
avsluttet». Filtreringen skjer server-side (routers/messages.py), fordi det er
serveren som vet at utfallet finnes.

Haandskrevet, ikke autogenerert: autogenerate mot SQLite melder hele
`research`-skjemaet som en diff, siden skjemaet ikke finnes der.

Revision ID: b4c81f0d3a97
Revises: f1a7d3c8e206
"""
import sqlalchemy as sa
from alembic import op

revision = "b4c81f0d3a97"
down_revision = "f1a7d3c8e206"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("team_elections",
                  sa.Column("member_count_at_open", sa.Integer(),
                            nullable=False, server_default="0"))
    op.add_column("pending_messages",
                  sa.Column("superseded_at", sa.DateTime(timezone=True),
                            nullable=True))


def downgrade() -> None:
    op.drop_column("pending_messages", "superseded_at")
    op.drop_column("team_elections", "member_count_at_open")
