"""noekkeldeponering og skadedata

To ting fra samme runde:

1. `backup_key_escrow` (§2/§13): FRIVILLIG deponering av noekkelen til
   backup-bloben. `material` er kryptert i ro med en server-hemmelighet som
   IKKE ligger i basen, saa en dump alene gir ingen noekler. Tabellen er det
   eneste stedet serveren har noe som kan aapne bloben.

2. `research_sharing_preferences.share_injury_data` (§7): skadedata gaar fra
   «ingen bryter, altsaa aldri» til egen opt-in. server_default er false, saa
   eksisterende rader beholder dagens oppfoersel - en migrasjon skal ikke
   utvide noens deling.

Haandskrevet, ikke autogenerert: autogenerate mot SQLite melder hele
`research`-skjemaet som en diff, siden skjemaet ikke finnes der.

Revision ID: e5b0c72a94d1
Revises: c93f2a1d7b60
"""
import sqlalchemy as sa
from alembic import op

revision = "e5b0c72a94d1"
down_revision = "c93f2a1d7b60"
branch_labels = None
depends_on = None

TZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "backup_key_escrow",
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("material", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", TZ, nullable=False),
        sa.Column("updated_at", TZ, nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.add_column("research_sharing_preferences",
                  sa.Column("share_injury_data", sa.Boolean(), nullable=False,
                            server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("research_sharing_preferences", "share_injury_data")
    op.drop_table("backup_key_escrow")
