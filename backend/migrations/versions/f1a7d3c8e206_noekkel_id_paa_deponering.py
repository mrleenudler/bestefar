"""noekkel-id paa deponering

Legger til `backup_key_escrow.key_check`: et fingeravtrykk av hemmeligheten
raden er kryptert med (HMAC over en fast streng - ingenting om selve
hemmeligheten). Uten den kunne en feilsatt BACKUP_ESCROW_SECRET bare oppdages
den dagen en bruker proevde aa gjenopprette kopien sin, og det er verst
tenkelige tidspunkt. Naa teller /health hvor mange rader som ligger paa en
annen hemmelighet enn den som staar.

server_default "" fordi eksisterende rader ikke vet hvilken hemmelighet de
ligger paa. De rapporteres som avvikende, og retter seg selv ved foerste
lesing (routers/backup.py krypterer om naar den gamle hemmeligheten maatte til).

Haandskrevet, ikke autogenerert: autogenerate mot SQLite melder hele
`research`-skjemaet som en diff, siden skjemaet ikke finnes der.

Revision ID: f1a7d3c8e206
Revises: e5b0c72a94d1
"""
import sqlalchemy as sa
from alembic import op

revision = "f1a7d3c8e206"
down_revision = "e5b0c72a94d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("backup_key_escrow",
                  sa.Column("key_check", sa.String(length=32), nullable=False,
                            server_default=""))


def downgrade() -> None:
    op.drop_column("backup_key_escrow", "key_check")
