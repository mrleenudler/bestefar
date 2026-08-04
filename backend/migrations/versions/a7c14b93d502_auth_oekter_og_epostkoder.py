"""auth-oekter og e-postkoder

Fase 3 (backend_spec §1): to nye tabeller.

  auth_sessions       én rad per innlogget enhet. Holder SHA-256 av
                      refresh-tokenet, aldri tokenet selv.
  email_login_codes   engangskoder for e-postinnlogging, ogsaa hashet.

Skrevet for haand av samme grunn som 4ef8ebf137fc: autogenerate mot SQLite
rapporterer hele `research`-skjemaet som en forskjell (schema_translate_map
oversetter det bort ved kjoering, men modellene bruker det fortsatt), og det
ville forurenset migrasjonen med tabeller som allerede finnes.

Tidsstempelkolonnene bruker sa.DateTime(timezone=True) - se 4ef8ebf137fc for
hvorfor alt skal vaere tidssone-bevisst.

Revision ID: a7c14b93d502
Revises: 4ef8ebf137fc
Create Date: 2026-08-04
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'a7c14b93d502'
down_revision: str | None = '4ef8ebf137fc'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

TZ = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        'auth_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.String(length=36), nullable=False),
        sa.Column('refresh_hash', sa.String(length=64), nullable=False),
        sa.Column('user_agent', sa.String(length=200), nullable=False),
        sa.Column('created_at', TZ, nullable=False),
        sa.Column('last_used_at', TZ, nullable=False),
        sa.Column('expires_at', TZ, nullable=False),
        sa.Column('revoked_at', TZ, nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_auth_sessions_user_id'), 'auth_sessions',
                    ['user_id'], unique=False)
    # Unik: hashen er oppslagsnoekkelen ved fornyelse, og to oekter kan ikke
    # dele token.
    op.create_index(op.f('ix_auth_sessions_refresh_hash'), 'auth_sessions',
                    ['refresh_hash'], unique=True)

    op.create_table(
        'email_login_codes',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('code_hash', sa.String(length=64), nullable=False),
        sa.Column('attempts', sa.Integer(), nullable=False),
        sa.Column('created_at', TZ, nullable=False),
        sa.Column('expires_at', TZ, nullable=False),
        sa.Column('consumed_at', TZ, nullable=True),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_email_login_codes_email'), 'email_login_codes',
                    ['email'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_email_login_codes_email'), table_name='email_login_codes')
    op.drop_table('email_login_codes')
    op.drop_index(op.f('ix_auth_sessions_refresh_hash'), table_name='auth_sessions')
    op.drop_index(op.f('ix_auth_sessions_user_id'), table_name='auth_sessions')
    op.drop_table('auth_sessions')
