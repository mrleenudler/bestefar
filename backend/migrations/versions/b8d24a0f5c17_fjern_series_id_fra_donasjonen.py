"""fjern series_id og user_id fra feilanalyse-donasjonen

`failed_analyses.series_id` var den ENESTE koblingen mellom et donert bilde og
en konto: ID-en er den samme som serien lagres under i `/v1/stats`, saa den som
har basen kunne slaa opp hvem bildet kom fra. Endepunktet krever ikke
innlogging, og raden er ikke merket med konto - men koblingen fantes likevel, og
den gjorde bildene til personopplysninger (`personvernerklaring.txt` 2.7,
AAPENT PUNKT 6).

Eieravklaring 2026-08-20: koblingen skal FJERNES VED INNSENDING, ikke
tidsbegrenses. En slettefrist utsetter koblingen; her opphoerer den.

INGENTING LESER KOLONNEN. Den ble skrevet av `POST /v1/failed-analyses` og lest
av ingen - verifisert med soek i hele backend-treet foer den ble fjernet. Den
hadde til og med en indeks (`ix_failed_analyses_series_id`) uten et eneste
oppslag. Kalibreringsmaterialet ligger i `detected_scores`/`ocr_scores` og
`tag`, som er selvstendige; ingen analyse mister noe.

KOLONNEN DROPPES, den nulles ikke. De radene som allerede finnes baerer sin
series_id, og en «vi slutter aa skrive den»-loesning ville latt dem staa igjen
som personopplysninger. Da ville AAPENT PUNKT 6 ikke kunne lukkes.

MAALT I PRODUKSJON 2026-08-21, foer migrasjonen: 9 rader i `failed_analyses`,
6 av dem med `series_id`, 0 med `user_id`. Tallene staar her fordi de ikke kan
skrives i ettertid - etter denne migrasjonen finnes ikke kolonnene aa telle.
Det er seks koblinger til en konto som forsvinner, ikke null og ikke elleve.

USER_ID DROPPES I SAMME MIGRASJON. Den er tom (0 av 9) og har aldri vaert satt
av endepunktet, saa den er ingen personopplysning i dag. Men den er en ferdig
lagt fremmednoekkel til `users` paa en tabell hvis hele poeng naa er at radene
IKKE kan knyttes til en konto, og en kobling som allerede er koblet opp er
noe annet enn en som maa lages. Fjernes den, kreves en ny migrasjon for aa
gjenaapne spoersmaalet - og det er riktig terskel for aa gjoere donasjonene
identifiserbare igjen.

VINDUET UNDER DEPLOY. `release_command` kjoerer migrasjonen FOER den nye
versjonen slippes til, saa i noen sekunder kjoerer gammel kode mot et skjema
uten kolonnene. Bare INSERT-en i `POST /v1/failed-analyses` treffes; den lister
dem og feiler med 500. Klienten behandler 500 som `retryable` og sender
donasjonen paa nytt, saa ingenting gaar tapt. Samme avveining som a3f7c1e59b24.
Den andre leseren av tabellen, `services/bucketflytt.py`, velger bare
`object_key` og bryr seg ikke om hvilke andre kolonner som finnes.

INGEN KRAV OM ÉN MASKIN. Migrasjonen kjoerer én gang, i sin egen release-maskin,
ikke én gang per app-maskin - saa to (eller en oppvaaknende) maskin endrer ikke
hva som kjoeres. En DROP COLUMN i Postgres er en katalogendring uten omskriving
av tabellen, saa laasen er kortvarig.

EN GAMMEL KLIENT KAN FORTSETTE AA SENDE FELTET. FastAPI ignorerer skjemafelt
ruten ikke erklaerer, saa donasjonen gaar gjennom som foer og verdien naar
aldri basen. Det er med vilje: et 4xx her ville vaert ikke-`retryable` og
dermed stille tap av donasjonen hos en klient vi ikke kan oppdatere i samme
oeyeblikk.

DOWNGRADE gir deg kolonnene tilbake, men de er tomme. `series_id`-verdiene er
borte, og det er hele hensikten - de kan ikke gjenskapes. `user_id` var tom fra
foer, saa den taper ingenting.

Haandskrevet, ikke autogenerert: autogenerate mot SQLite melder hele
`research`-skjemaet som en diff, siden skjemaet ikke finnes der.

Revision ID: b8d24a0f5c17
Revises: a3f7c1e59b24
"""
import sqlalchemy as sa
from alembic import op

revision = "b8d24a0f5c17"
down_revision = "a3f7c1e59b24"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index(op.f("ix_failed_analyses_series_id"),
                  table_name="failed_analyses")
    op.drop_column("failed_analyses", "series_id")
    # Fremmednoekkelen foelger kolonnen; SQLite krever batch-modus for begge.
    with op.batch_alter_table("failed_analyses") as batch:
        batch.drop_column("user_id")


def downgrade() -> None:
    op.add_column("failed_analyses",
                  sa.Column("series_id", sa.String(length=36), nullable=True))
    op.create_index(op.f("ix_failed_analyses_series_id"), "failed_analyses",
                    ["series_id"], unique=False)
    with op.batch_alter_table("failed_analyses") as batch:
        batch.add_column(sa.Column("user_id", sa.String(length=36),
                                   nullable=True))
        batch.create_foreign_key("fk_failed_analyses_user_id", "users",
                                 ["user_id"], ["id"], ondelete="SET NULL")
