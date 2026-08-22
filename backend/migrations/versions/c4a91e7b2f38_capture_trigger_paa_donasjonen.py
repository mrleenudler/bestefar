"""capture_trigger paa feilanalyse-donasjonen

Klienten fikk fra v0.29 en tidsgrense paa auto-capture: utloeser ikke gatingen
innen 7 sekunder, tas gjeldende ramme og analyseres likevel. Lykkes analysen da,
er det ikke en feil - det er maaledata som sier at gatingen var for streng, og
det er nettopp observasjonen AAP-K1 mangler. Uten et merke i donasjonen er den
ikke skillbar fra en helt ordinaer scan.

EGET FELT, IKKE EN NY TAG-VERDI (issue #11, B-53). `tag` svarer paa hva
donasjonen VISER (ocr_match / ocr_mismatch / rejected), `capture_trigger` paa
HVORDAN bildet ble tatt. De to er ortogonale - en timeout-capture kan ende som
hvilken som helst av de tre - saa en «timeout»-verdi i `tag` ville overskrevet
OCR-utfallet, og skulle begge deler bevares maatte enumet dobles for hver nye
capture-aarsak.

KOLONNEN ER NULLABLE, OG NULL BETYR «KLIENTEN SA DET IKKE». Ikke «auto». Fra
v0.29 finnes timeout-capture i klienten, men feltet ble bevisst ikke sendt foer
formen var avtalt, saa donasjonene fra det vinduet ER dels timeout-utloeste uten
aa kunne si det. En default paa `auto` ville stemplet dem som gatede, og da
ville den foerste maalingen AAP-K1 hviler paa, vaert bygget paa en verdi ingen
har oppgitt.

Ingen backfill, av samme grunn: det finnes ikke en riktig verdi aa fylle inn.

Revision ID: c4a91e7b2f38
Revises: b8d24a0f5c17
"""
import sqlalchemy as sa
from alembic import op

revision = "c4a91e7b2f38"
down_revision = "b8d24a0f5c17"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Samme form som `tag` i d21e5f8ac782: VARCHAR + CHECK, ikke en native
    # Postgres-type (B-33). Konstruksjonen skrives ut slik modellen skriver
    # den, saa `test_migrations.py` ikke finner drift.
    op.add_column(
        "failed_analyses",
        sa.Column("capture_trigger",
                  sa.Enum("auto", "timeout", name="capturetrigger",
                          native_enum=False, length=16),
                  nullable=True))


def downgrade() -> None:
    op.drop_column("failed_analyses", "capture_trigger")
