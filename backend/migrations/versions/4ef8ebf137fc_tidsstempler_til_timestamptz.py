"""tidsstempler til timestamptz

Alle tidsstempelkolonner gaar fra TIMESTAMP til TIMESTAMP WITH TIME ZONE.

Bakgrunn: `utcnow()` gir en tidssone-bevisst datetime, men en naiv kolonne
leverer en naiv datetime tilbake. Alt som SAMMENLIGNER en ny verdi med en
lagret - karantenevinduer (§3.1), «er denne backupen eldre enn den lagrede?»
(§2) - kastet da «can't subtract offset-naive and offset-aware datetimes», og
bare paa de kodestiene som faktisk sammenligner. Modellene bruker naa
`UtcDateTime`, som i tillegg normaliserer naive verdier inn til UTC.

Skrevet for haand: autogenerate maa kjoere mot Postgres for aa se
typeforskjellen, og mot SQLite produserer den bare stoey fra
schema_translate_map. Lista under er generert fra Base.metadata.

Migrasjonen er en no-op paa SQLite, som ikke har en egen tidsstempeltype -
konverteringen haandteres der av UtcDateTime.

Revision ID: 4ef8ebf137fc
Revises: d21e5f8ac782
Create Date: 2026-08-03
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = '4ef8ebf137fc'
down_revision: str | None = 'd21e5f8ac782'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# (tabell, skjema, kolonne, nullable)
KOLONNER = [
    ('feedback', None, 'received_at', False),
    ('feedback', None, 'forwarded_at', True),
    ('consents', 'research', 'granted_at', False),
    ('consents', 'research', 'revoked_at', True),
    ('deletion_requests', 'research', 'requested_at', False),
    ('deletion_requests', 'research', 'completed_at', True),
    ('records', 'research', 'captured_at', False),
    ('records', 'research', 'received_at', False),
    ('search_quarantines', None, 'window_started_at', False),
    ('search_quarantines', None, 'quarantined_until', True),
    ('search_quarantines', None, 'updated_at', False),
    ('users', None, 'display_name_reviewed_at', True),
    ('users', None, 'created_at', False),
    ('users', None, 'updated_at', False),
    ('users', None, 'last_seen_at', True),
    ('users', None, 'deleted_at', True),
    ('auth_identities', None, 'created_at', False),
    ('backups', None, 'client_ts', True),
    ('backups', None, 'updated_at', False),
    ('devices', None, 'created_at', False),
    ('devices', None, 'last_seen_at', False),
    ('failed_analyses', None, 'submitted_at', False),
    ('friendships', None, 'created_at', False),
    ('friendships', None, 'responded_at', True),
    ('research_sharing_preferences', None, 'updated_at', False),
    ('series', None, 'ts', False),
    ('series', None, 'created_at', False),
    ('series', None, 'updated_at', False),
    ('sharing_preferences', None, 'updated_at', False),
    ('teams', None, 'created_at', False),
    ('teams', None, 'updated_at', False),
    ('leader_challenges', None, 'opened_at', False),
    ('leader_challenges', None, 'deadline_at', False),
    ('leader_challenges', None, 'resolved_at', True),
    ('pending_messages', None, 'created_at', False),
    ('pending_messages', None, 'delivered_at', True),
    ('team_elections', None, 'opened_at', False),
    ('team_elections', None, 'closes_at', False),
    ('team_elections', None, 'resolved_at', True),
    ('team_invites', None, 'created_at', False),
    ('team_invites', None, 'sent_at', True),
    ('team_invites', None, 'accepted_at', True),
    ('team_members', None, 'joined_at', False),
    ('team_votes', None, 'cast_at', False),
]


def _konverter(med_tidssone: bool) -> None:
    if op.get_bind().dialect.name == "sqlite":
        return
    for tabell, skjema, kolonne, nullable in KOLONNER:
        op.alter_column(
            tabell, kolonne, schema=skjema, nullable=nullable,
            existing_nullable=nullable,
            type_=sa.DateTime(timezone=med_tidssone),
            existing_type=sa.DateTime(timezone=not med_tidssone),
            # Eksisterende naive verdier ER UTC (skrevet av utcnow()).
            postgresql_using=f"{kolonne} AT TIME ZONE 'UTC'" if med_tidssone else None)


def upgrade() -> None:
    _konverter(True)


def downgrade() -> None:
    _konverter(False)
