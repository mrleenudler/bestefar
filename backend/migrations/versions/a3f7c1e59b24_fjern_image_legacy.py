"""fjern image_legacy

Kolonnen holdt feilanalyse-bilder i databasen mens R2-opplastingen ikke fantes
(B-32). Opplastingen kom 2026-08-15 (B-44), og de fem radene som laa igjen ble
flyttet til R2 samme dag (B-48, AAP-B11) - verifisert ved at `GET /health` gikk
fra «"bilder":"r2 (5 gamle rader i basen)"» til «"bilder":"r2"». Kolonnen er
tom, og §6 sier at bilder aldri skal ligge i databasen.

VINDUET UNDER DEPLOY. `release_command` kjoerer denne migrasjonen FOER den nye
versjonen slippes til, saa i noen sekunder kjoerer gammel kode mot et skjema
uten kolonnen. Foelgen er avgrenset: bare `failed_analyses` er beroert, og bare
INSERT-en i `POST /v1/failed-analyses` - den lister kolonnen og vil feile med
500 i vinduet. Klienten behandler 500 som `retryable` og sender donasjonen paa
nytt, saa ingenting gaar tapt. `/health` taaler det, fordi tellingen der allerede
staar i en try/except.

DOWNGRADE gir deg kolonnen tilbake, men ikke innholdet. Bildene ligger i R2 og
er referert med `object_key`; det er den veien tilbake som finnes.

Haandskrevet, ikke autogenerert: autogenerate mot SQLite melder hele
`research`-skjemaet som en diff, siden skjemaet ikke finnes der.

Revision ID: a3f7c1e59b24
Revises: c7e2b91f4d08
"""
import sqlalchemy as sa
from alembic import op

revision = "a3f7c1e59b24"
down_revision = "c7e2b91f4d08"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("failed_analyses", "image_legacy")


def downgrade() -> None:
    op.add_column("failed_analyses",
                  sa.Column("image_legacy", sa.LargeBinary(), nullable=True))
