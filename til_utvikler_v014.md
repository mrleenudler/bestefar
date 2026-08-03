# Til utvikler — v0.14 (backend-kobling, runde 1)

## Tilbakemeldingen denne runden

Ingen `musingsUI.txt`-runde. Oppgaven ble stilt direkte i chat:

> «I've (or you, rather, in a different instance) have started implementing the
> backend. Is there anything we can do in the UI at the same time, or should we
> wait for the backend to complete?»

Svaret var: ikke vent — men det nyttige arbeidet er ikke nye skjermer, det er
**klientsiden av ledningen**. Appen hadde null nettverkskode. Alt av
backend-funksjonalitet står og faller på det laget, så det er den kritiske stien.

Deretter: «yeah» — bygg det.

## Hva som ble gjort

### 1. `Api.kt` — transportlaget (nytt)
`HttpURLConnection`, ingen nytt bibliotek. Basis-URL fra `BuildConfig.API_BASE_URL`
(release `https://bestefar-api.fly.dev`, debug `http://10.0.2.2:8000`), overstyrbar
i felt via DevTools. Klartekst-HTTP kun i debug (`src/debug/AndroidManifest.xml`).

Det viktigste valget her er **feilklassifiseringen**. Appen er offline-først, så
et mislykket kall er normaltilstanden, ikke en feil. `retryable` = `code == 0`
(nådde aldri fram), 408, 429, ≥ 500. Alt annet er permanent, og køen kaster
elementet i stedet for å prøve i evig tid.

### 2. `Sync.kt` — opplastingskøen (nytt)
Filbasert kø i `filesDir/dev_uploads`, ett par `{seriesId}_{tag}.jpg` + `.json`.
Filbasert med vilje: den overlever appdrap, omstart og flymodus uten database.
Tømmes ved appstart og på «Send nå» — ikke som bakgrunnsjobb. Det er nok for
offline-først og sparer oss for WorkManager. Trenger vi sending mens appen er
lukket, er periodisk WorkManager neste steg.

Sidecar-formatet er bumpet til v2 og bærer nå alt endepunktet trenger:
`status_code`, `confidence`, `core_version`, `tag`, `series_id`, `detected`, `ocr`.

### 3. Tre plassholdere som løy, er nå ekte
- `queueDevImage()` skrev filer ingen sendte → sender nå.
- «N usendte opplastinger i kø» i øktoppsummeringen var en teller som kun kunne
  vokse → teller nå det som faktisk kan sendes, og skjules når køen er tom.
- «Send bildet til feilanalyse» på avvist-skjermen viste en kvittering og
  deaktiverte seg selv uten å sende noe → sender nå faktisk bildet.

### 4. «Melding til utvikler» → `POST /v1/feedback`
Den ene funksjonen som kunne gå ende-til-ende i dag, siden endepunktet med vilje
ikke krever innlogging. E-post beholdes som fallback når nettet er nede.

### 5. Avanserte innstillinger
Ny bryter «Last kun opp på wifi» (default på) og «Send bilder til feilanalyse nå»
med levende køstatus. Ny DevTools-oppføring «API-adresse».

## Verifisert

Begge endepunktene testet mot produksjon med nøyaktig den wire-formen klienten
produserer, 2026-08-02:

| Kall | Svar |
|---|---|
| `GET /health` | `200 {"status":"ok","env":"prod","database":"ok","mailer":"log"}` |
| `POST /v1/feedback` | `202 {"id":1,"status":"mottatt"}` |
| `POST /v1/failed-analyses` | `201 {"id":1}` |

De to testradene (`feedback` id 1, `failed_analyses` id 1) står i databasen og kan
slettes. `mailer: log` betyr at ingen e-post ble sendt ut — meldingen ligger kun
i basen, som designet.

Release-bygg OK, `dist\Bestefar-0.14.apk`.

## Én ting du bør se på når kontoen kommer

`detected` i sidecar-en er alltid poengene **CV-kjernen** ga, ikke det som vises.
Dette var nær ved å bli feil: `applyScores()` overskriver `record.shots` med
OCR-poengene *før* donasjonen køes, så en `ocr_match`-donasjon ville hatt
OCR-poengene i begge felt — altså null informasjon om hva kjernen faktisk så.
Det er nettopp den differansen dataene skal brukes til å kalibrere.

## Fortsatt ikke koblet inn (venter på fase 3, konto)

- `/v1/stats` — ruteren importerer allerede `deps.current_user`, så den er
  skrevet men ikke testbar. `SeriesRecord.uploaded` står urørt.
- `/v1/research` — sperret i begge ender (`Dialogs.RESEARCH_ENABLED` i klienten,
  personvernerklæring/DPIA i backend_spec §7).
- Venner, lag, backup, push. Front-end-skjelettene finnes allerede
  (`VennerActivity`, `LagActivity`, `TeamPageActivity`) og er bevisst ikke
  koblet på noe.

Når auth lander er det eneste nye en token-header i `Api.request()` — ikke et
helt transportlag.

## Åpent spørsmål til deg

`core_version` i donasjonene er foreløpig appens `versionName` («0.14»).
CV-kjernen eksponerer ingen egen versjon over FFI. Skal vi legge til en
`bf_version()` i `bestefar_ffi.h`, slik at feilanalyse-dataene kan knyttes til
nøyaktig hvilken kjerne som produserte dem? Det blir vanskelig å tolke gamle
donasjoner uten.

---

# Backend — fase 1 og 2 (samme runde, annen instans)

Skrevet av backend-instansen. Klientdelen over og dette er to ender av samme
ledning; feltnavnene er verifisert mot hverandre nederst.

## Fase 0 — beslutninger

Planens fase 0 ba om fem valg. Fire var i praksis avgjort av kontoene som
allerede var opprettet; det femte utsatte jeg med vilje.

| Valg | Landet på |
|---|---|
| Vert | Fly.io, app `bestefar-api`, region `ams` (EU/EØS) |
| Database | **Supabase** (`Bestefar_base`) — prosjektet var allerede opprettet |
| Objektlagring | Cloudflare R2 (ikke koblet inn ennå, hører til §6) |
| Telefon-OTP | utsatt til v2, som spec-en allerede sier |
| Secrets | Fly secrets |

**E-postleverandør er bevisst ikke valgt.** Videresendingen i §10 er
leverandør-agnostisk: `RESEND_API_KEY` → Resend, `SMTP_HOST` → SMTP, ellers kun
logging. Meldingen lagres uansett i databasen først. Påkoblingen senere er én
secret. Det er dette `mailer: log` i `/health` betyr.

**Domene er ikke kjøpt.** `bestefar-api.fly.dev` holder til OAuth-redirect-URLene
skal registreres (fase 3).

## Fase 1 — skjelett, CI/CD, første endepunkt

FastAPI delt i routere, Dockerfile, `fly.toml` med scale-to-zero og helsesjekk.
GitHub Actions: `ci.yml` kjører testene, ny `deploy-backend.yml` deployer ved
push til `main`. Backenden har sitt **eget** virtuelle miljø, `backend/.venv` —
prosjektets `.venv` er CV-pipelinen og er urørt.

`POST /v1/feedback` ble første reelle endepunkt, slik planen foreslo: ingen
auth, ingen datamodell-avhengighet, altså en ekte ende-til-ende-test av kjeden.

To valg tatt for å unngå kjente driftsfeller:

1. **Databasetilkoblingen lages ved første bruk, ikke ved import.** Ellers dør
   prosessen under oppstart hvis databasen er nede, og Fly gir en deploy-loop
   uten forklaring. `/health` svarer alltid 200; databasetilstanden står i
   kroppen.
2. **`/health` sier «feilkonfigurert» hvis `ENV=prod` og `DATABASE_URL` er
   SQLite.** `SELECT 1` mot en flyktig containerfil svarer nemlig helt fint — og
   hadde rapportert «ok» for en database som forsvinner ved omstart.

## Fase 2 — datamodell og migrasjoner

23 tabeller som dekker §1–§11. `app/models.py` er splittet til pakken
`app/models/` (`user`, `training`, `social`, `ops`, `research`).

Treningsmodellen speiler `SeriesRecord`/`Shot` i `Model.kt` felt for felt, så
synk blir en direkte mapping. Den gamle `Shooter`/`Session`-modellen fra
CV-kravspec-en er erstattet av serier som henger på brukeren.

Tre valg verdt å kjenne til:

- **Forsknings-ID avledes, den lagres ikke.** HMAC-SHA256(server-hemmelighet,
  bruker-UUID). En oppslagstabell ville vært nettopp den reversible koblingen
  §7 forbyr. Prisen: hemmeligheten kan ikke roteres uten å bryte koblingen til
  allerede innsamlede forskningsdata.
- **Serie-ID er klientens egen UUID**, så `PUT /v1/stats/series/{id}` er
  idempotent — klienten kan trygt sende en køet serie flere ganger.
- **Enum-kolonner er VARCHAR + CHECK**, ikke Postgres-native ENUM. Native enum
  krever egne `ALTER TYPE`-migrasjoner og finnes per skjema.

Forskningstabellene ligger i et eget Postgres-skjema (`research`) fra dag én —
adskillelsen er dyr å ettermontere. SQLite har ikke skjemaer, så i tester
oversettes navnet bort.

**Migrasjonene kjører som `release_command` ved deploy**, altså i egen maskin
før den nye versjonen slippes til. Feiler de, stanses deployen og forrige
versjon blir stående. `tests/test_migrations.py` sammenligner modellene med
basen, så en modellendring uten migrasjon feiler i CI i stedet for ved deploy.
Den sjekken krever Postgres, så CI kjører suiten to ganger — SQLite og Postgres.

## Sperrer som står med vilje

- **Forskning er av** (`RESEARCH_ENABLED=false`), på linje med
  `Dialogs.RESEARCH_ENABLED`. §7/§9 gjør personvernerklæring og en avklaring av
  DPIA-behovet til en *forutsetning* for at innsamling kan starte. Endepunktet
  svarer 503 til flagget snus.
- **Auth finnes ikke ennå.** Alt som krever bruker svarer 501 i produksjon.
  Lokalt kan brukeren angis med `X-Debug-User-Id` — den headeren er død i prod,
  ikke bare frarådet. Det er derfor `/v1/stats` er skrevet, men ikke testbar fra
  appen ennå.
- **Bilder ligger fortsatt i databasen** (`image_legacy`), ikke i R2, stikk i
  strid med §6. Inntil R2 er koblet inn avviser endepunktet filer over 8 MB.
  Kolonnen skal fjernes når opplastingen er på plass.

## Svar på §3.1 («8–10 tegn, 9 base32-tegn»)

Implementert med **8 signifikante tegn** (7 tilfeldige + sjekksiffer), vist som
`BF-XXXX-XXXX` — altså nøyaktig eksempelet i spec-en. Det gir ~3,4 · 10¹⁰
ID-er: ett tegn kortere enn teksten nevner, men fortsatt langt over det
gjettingsargumentet krever, og lettere å lese opp i telefonen. Sjekksifferet er
`sum mod 32` i samme alfabet; Crockfords egen mod-37-variant ville trukket inn
fire symboler utenfor alfabetet. Innlesing folder I/L→1 og O→0, så de vanligste
lesefeilene godtas i stedet for å avvises.

## Svar på det åpne spørsmålet om `core_version`

**Ja, legg til `bf_version()` i `bestefar_ffi.h`.** Begrunnelsen er sterkere enn
«fint å ha»: hele poenget med §6-donasjonene er å kalibrere kjernen, og en
måling uten å vite hvilken kjerne som produserte den kan ikke brukes til det.
Appens `versionName` er en dårlig stedfortreder, siden UI-runder bumper den uten
at kjernen er rørt — da ser det ut som kjernen endret seg når den ikke gjorde
det. Kolonnen `core_version` finnes allerede og tar imot hva som helst, så dette
er en ren kjerne-endring. Ført opp i `backend_spec.md` §8.

## Verifisert mot produksjon

| Kall | Svar |
|---|---|
| `GET /health` | `{"status":"ok","env":"prod","database":"ok","mailer":"log"}` |
| `POST /v1/feedback` | `202` — rad lagret, `forwarded_at` satt |
| `POST /v1/failed-analyses` (klientens feltnavn) | `201` |
| samme med `tag=tull` | `422` — som `Sync.kt` korrekt behandler som DROP |

Feltnavnene i `Sync.kt` (`status_code`, `confidence`, `core_version`, `tag`,
`series_id`, `detected_scores`, `ocr_scores`, filfelt `image`) matcher
endepunktet eksakt.

Databasen i Supabase har 23 tabeller, `research`-skjemaet står separat med sine
tre. Testradene (`feedback` id 1–2, `failed_analyses` id 1–2) kan slettes.

## Kjent avvik

Rate-limiteren på `/v1/feedback` teller **i minnet, per maskin**. Fly kjører to
maskiner, så den reelle grensen er 10/time, ikke 5. Telleren må uansett flyttes
til databasen for §3.1-karantenen (som skal overleve omstart) — tabellen
`search_quarantines` ligger klar.

## Gjenstår før fase 3

- `FLY_API_TOKEN` som repo-secret, ellers kjører ikke auto-deploy.
- Google OAuth-klient (Google Cloud Console) og Apple-utviklerkonto — begge
  krever nettleser og kan ikke settes opp fra terminalen.
- Eget domene for redirect-URLene.
