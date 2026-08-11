"""identitet paa oekten

`auth_sessions.identity_id` - hvilken AuthIdentity oekten ble startet med.

Tokenparet svarer naa med `email`, saa klienten kan vise «Logget inn som
ola@example.com». Adressen kan IKKE leses ut av ID-tokenet klientside:
kontosammenslaaing skjer paa verifisert e-post (backend_spec §1), saa kontoen
kan vaere knyttet til en annen adresse enn den man nettopp logget inn med - og
da ville skjermen loeyet i akkurat det tilfellet den finnes for.

Ved fornyelse finnes ikke ID-tokenet lenger, derfor maa oekten huske det.

SET NULL og ikke CASCADE: forsvinner identiteten, skal ikke brukeren logges ut -
hen mister bare adressen i visningen. NULL ogsaa for oekter startet foer denne
kolonnen fantes; svaret blir da `email: null`, som klienten maa taale uansett
(Apple med «Skjul e-postadressen min» gir ingen adresse).

Haandskrevet, ikke autogenerert: autogenerate mot SQLite melder hele
`research`-skjemaet som en diff, siden skjemaet ikke finnes der.

Revision ID: c7e2b91f4d08
Revises: b4c81f0d3a97
"""
import sqlalchemy as sa
from alembic import op

revision = "c7e2b91f4d08"
down_revision = "b4c81f0d3a97"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite kan ikke legge til en FK paa en eksisterende tabell med ALTER, saa
    # batch_alter_table brukes - den bygger tabellen paa nytt der. No-op-omvei
    # paa Postgres, som tar ALTER direkte.
    with op.batch_alter_table("auth_sessions") as batch:
        batch.add_column(sa.Column("identity_id", sa.Integer(), nullable=True))
        batch.create_foreign_key("fk_auth_sessions_identity", "auth_identities",
                                 ["identity_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    with op.batch_alter_table("auth_sessions") as batch:
        batch.drop_constraint("fk_auth_sessions_identity", type_="foreignkey")
        batch.drop_column("identity_id")
