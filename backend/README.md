# Bestefar backend

FastAPI-backend, se `../backend_spec.md` for kravene og
`../Plan_for_backend_implementering.odt` for faseplanen.

Ansvarsomraadene er ATSKILTE (kravspec §5):

1. **`/v1/stats`** — brukerens egne resultatdata (statistikk-sync).
   Standard: kun treffdata, ingen bilder. Bildelagring er brukerstyrt opsjon.
2. **`/v1/failed-analyses`** — opt-in innsending av feilede/lav-konfidens
   analyser (bilde + metadata) for CV-forbedring.
3. **`/v1/research`** — forskningsdata, STRUKTURELT ADSKILT (egne tabeller,
   pseudonym skytter-ID, eksplisitt samtykke). Kravspec §6.
4. **`/v1/feedback`** — melding fra bruker til utvikler (backend_spec §10).
5. **`/v1/backup`** — klient-kryptert blob med logg og innstillinger (§2).
   Serveren lagrer bytes den ikke kan lese. `PUT` avviser en blob hvis
   `client_ts` er eldre enn den lagrede (409) — se docstringen i
   `app/routers/backup.py` for hvorfor det vernet maa ligge server-side.
6. **`/v1/profile`**, **`/v1/users/search`**, **`/v1/friends`** — profil,
   delingsvalg, brukersoek og vennskap (§3, §3.1).

Tre ting styrer venne-delen:

- **Soek gir kun eksakt treff** paa bruker-ID eller telefon, og kun for
  `findable`-brukere. Fritekstsoek paa navn ville gjort brukerbasen listbar.
- **Karantenen ligger i databasen** (`services/quarantine.py`), ikke i minnet
  som `ratelimit.py`: den skal overleve omstart og gjelde paa tvers av Flys
  maskiner. Bare MISLYKKEDE soek telles - treff er normal bruk.
- **Filtreringen er utgaaende og server-side** (`services/sharing.py`). Da er
  «deaktivering nuller delte felt» en garanti, ikke noe en modifisert klient
  kan omgaa.

## Tidsstempler

Alle `Mapped[datetime]`-kolonner bruker `UtcDateTime` (se
`app/models/base.py`) via `type_annotation_map` paa `Base` - ikke oppgi
`DateTime` eksplisitt i `mapped_column`. Typen garanterer at verdier er
tidssone-bevisste i Python og UTC i basen, og tolker naive verdier inn (typisk
ISO-tid fra klienten uten offset) som UTC. Uten den kaster enhver
SAMMENLIGNING mot en lagret verdi «can't subtract offset-naive and
offset-aware datetimes». Bruk `models.as_utc()` paa datoer som kommer inn som
query-parametere.

`/health` rapporterer database- og e-poststatus, og svarer 200 saa lenge
prosessen lever (se `app/routers/health.py` for hvorfor).

## Lokal kjoering

Backenden har sitt EGET virtuelt miljoe, `backend/.venv` — ikke prosjektets
`.venv` (den er for CV-pipelinen og har cv2/scipy).

```powershell
cd backend
..\.venv\Scripts\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
copy .env.example .env          # fyll inn ved behov
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
.\.venv\Scripts\python.exe -m pytest tests -q
```

SQLite som standard (`bestefar.db`); sett `DATABASE_URL` for Postgres.
Interaktiv API-dok paa `/docs` naar `ENV != prod`.

## Drift (backend_spec §0.1)

| | |
|---|---|
| Vert | Fly.io, app `bestefar-api`, region `ams` (EU/EOES) |
| Database | Supabase Postgres (`Bestefar_base`, EU) |
| Objektlagring | Cloudflare R2 — for feilanalyse-bilder (fase 6) |
| CI/CD | GitHub Actions: push til `main` → `flyctl deploy` |

Secrets settes som Fly secrets, aldri i repoet (samme prinsipp som
`gradle.properties` for signeringsnoekkelen):

```powershell
flyctl secrets set DATABASE_URL="postgresql://postgres:<passord>@db.<ref>.supabase.co:5432/postgres" -a bestefar-api
flyctl secrets set FEEDBACK_TO="<utviklerens e-post>" -a bestefar-api
```

E-postvideresending (§10) er leverandoer-agnostisk: sett `RESEND_API_KEY`
ELLER `SMTP_HOST`+`SMTP_USER`+`SMTP_PASSWORD`. Uten noen av delene lagres
meldingen i databasen og logges — den gaar aldri tapt.

For automatisk deploy fra GitHub trengs repo-secret `FLY_API_TOKEN`:

```powershell
flyctl tokens create deploy -a bestefar-api
```

## Datamodell

`app/models/` er delt etter ansvar:

| Modul | Innhold |
|---|---|
| `user.py` | konto, identitet, delingsvalg, backup, misbruksvern (§1, §2, §3.1) |
| `training.py` | serier og treff (§5) — speiler `SeriesRecord`/`Shot` i `Model.kt` |
| `social.py` | venner og lag, invitasjoner, lederavstemning, meldingskoe (§3, §4, §11) |
| `ops.py` | feilanalyse (§6) og feedback (§10) |
| `research.py` | forskningsdata i EGET skjema — ingen FK til brukertabellene (§7) |

Tre valg som er verdt aa kjenne til:

- **Forsknings-ID avledes, den lagres ikke.** `services/pseudonym.py` regner
  HMAC-SHA256(hemmelighet, user_id). En oppslagstabell ville vaert nettopp den
  reversible koblingen §7 forbyr. Hemmeligheten maa aldri roteres uten en plan
  for eksisterende forskningsdata.
- **Serie-ID er klientens egen UUID.** Opplasting blir da idempotent, slik at
  klienten trygt kan sende en koet serie flere ganger (§5).
- **Enum-kolonner er VARCHAR + CHECK** (`native_enum=False`), ikke Postgres
  ENUM: billigere aa utvide, og lesbart i Supabases admin-UI.

Konkret feltinnhold i forskningsdatasettet er fortsatt IKKE avklart — se
`TODO(eier)` i `app/models/research.py`.

## Migrasjoner

```powershell
cd backend
.\.venv\Scripts\python.exe -m alembic upgrade head
.\.venv\Scripts\python.exe -m alembic revision --autogenerate -m "hva du endret"
```

I produksjon kjoerer `alembic upgrade head` som `release_command` i `fly.toml`,
altsaa FOER den nye versjonen slippes til. Feiler migrasjonen, stanses
deployen og forrige versjon blir staaende.

`AUTO_CREATE_TABLES` (`create_all`) finnes bare som bekvemmelighet lokalt og er
`false` i produksjon — Alembic eier skjemaet.

`tests/test_migrations.py` sammenligner modellene med basen etter
`upgrade head`, saa en modellendring uten migrasjon feiler i CI i stedet for
ved deploy. Den kjoerer bare mot Postgres: skjemaet `research` finnes ikke paa
SQLite. CI kjoerer derfor suiten to ganger — SQLite og Postgres.

## Ikke aktiver uten videre

`/v1/research` skal ikke tas i bruk mot ekte brukere foer personvernerklaering
foreligger og behovet for DPIA er avklart (backend_spec §7/§9). Klienten har
samme sperre via `Dialogs.RESEARCH_ENABLED`.
