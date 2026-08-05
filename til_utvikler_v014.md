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

## Fase 4 — backup/sync (§2)

Bygget før fase 3, siden auth er blokkert på kontooppsett du må gjøre i
nettleser. Endepunktene henger på `current_user`, så de er ferdige i det
innloggingen lander.

`PUT /v1/backup`, `GET /v1/backup`, `GET /v1/backup/meta`, `DELETE /v1/backup`.
Bloben sendes rå (`application/octet-stream`) med metadata som
query-parametere — det sparer base64-påslaget på ~33 % på den største
nyttelasten vi håndterer. Grense 16 MB. `/meta` finnes for at «har jeg noe å
gjenopprette?» på en ny telefon ikke skal kreve nedlasting av hele bloben.

**Ett tillegg til spec-en, som du bør være enig i:** serveren avviser en `PUT`
der `client_ts` er eldre enn den lagrede (409). §2 sier last-write-wins per
post-ID, men den regelen kan bare håndheves *klient-side* — serveren ser ikke
inn i den krypterte bloben, den lagrer bytes den ikke kan lese. Uten dette
vernet kunne en telefon som synker for første gang på måneder viske ut alt som
er logget siden. `?force=true` overstyrer, for det bevisste valget
«gjenopprett fra denne enheten». Likt tidsstempel godtas, slik at en retry
etter et avbrutt kall ikke feiler.

12 tester dekker rundturen, vernet, isolasjon mellom brukere og
størrelsesgrensen. Verifisert live: alle backup-rutene svarer 501 i produksjon,
også med `X-Debug-User-Id` — den headeren er død i prod, ikke bare frarådet.

## Fase 5 — venner (§3, §3.1)

`GET/PUT /v1/profile` og `/v1/profile/sharing`, `GET /v1/users/search`,
`POST /v1/friends/request` og `/respond`, `GET /v1/friends` og
`/friends/requests`, `DELETE /v1/friends/{id}`.

**Søk gir kun eksakt treff** på bruker-ID eller telefonnummer, og bare for
`findable`-brukere. Fritekstsøk på navn ville gjort hele brukerbasen listbar,
så det finnes ikke. Er sjekksifferet i en ID feil, svarer vi 422 *uten* å slå
opp — det er en tastefeil, ikke et forsøk på å finne noen, og den skal ikke
telle som bom.

**Karantenen ligger i databasen**, ikke i minnet slik rate-limiteren på
`/v1/feedback` gjør. En karantene som forsvinner ved omstart — eller som bare
gjelder én av Flys to maskiner — er ingen karantene. 5 bom på telefonsøk gir
1 døgn, gjentakelse gir 7, som spec-en sier. Bare bom telles.

**Filtreringen er utgående og server-side.** Klienten får aldri et felt den
ikke har lov til å vise, så «deaktivering nuller delte felt» er en garanti og
ikke noe en modifisert app kan omgå. Visningsnavn deles alltid — men bare når
moderasjonen har godkjent det; ellers ser andre «Ukjent skytter».

**Navnemoderasjon:** tegnsett og lengde speiler `Ui.nameFilters()` i klienten
(klientfilteret er bekvemmelighet, ikke sikkerhet), pluss en ordliste satt med
`DISPLAY_NAME_BLOCKLIST`. Lista er tom som standard — en hardkodet norsk
banneordliste ville vært både ufullstendig og umulig å vedlikeholde fra repoet.
Sammenligningen skjer på en foldet form, så «S-t-y-g-t» og «stÿgt» fanges
også. Avvist navn lagres ikke i det hele tatt.

### To ting du må ta stilling til

**1. `kills[]` i §3-modellen kan ikke leveres.** Jaktloggen ligger inne i den
klient-krypterte backup-bloben (§2), som serveren ikke kan lese. Skal felte dyr
deles med venner, må jaktposter synkes som egne rader — det er en
spec-avklaring, ikke noe jeg kan implementere meg ut av. `avgScore` og `trend`
er derimot på plass, regnet fra treningsseriene.

**2. `trend` trenger en definisjon fra deg.** Spec-en sier bare
«snitt/utvikling». Jeg har valgt: snitt per skudd i de 5 siste seriene minus de
5 foregående, og `null` før det finnes 10 serier — et «trendtall» fra to serier
ville vært støy presentert som innsikt. Vindusstørrelsen er valgt fordi en
serie er 5–10 skudd og 5 serier dekker en typisk økt.

### En feil som var verdt å finne

Tester på karantenen avdekket at `utcnow()` ga tidssone-bevisste tidsstempler
mens kolonnene var naive. Alt som *sammenligner* en ny verdi mot en lagret —
karantenevinduer, «er denne backupen eldre enn den lagrede?» — kastet
`can't subtract offset-naive and offset-aware datetimes`. Det slo bare ut på de
kodestiene som faktisk sammenligner, så resten hadde ligget og ventet.

Fikset i roten: alle tidsstempler bruker nå typen `UtcDateTime`, som garanterer
tidssone-bevisste verdier i Python og UTC i basen, og tolker naive verdier inn
som UTC. Migrasjon `4ef8ebf137fc` konverterer kolonnene til `timestamptz`.
Den måtte skrives for hånd: autogenerate ser ikke typeforskjellen mot SQLite.

74 tester grønne.

## Fase 6 — lag (§4, §11)

`POST/GET /v1/teams`, `GET /v1/teams/near`, `GET/PUT /v1/teams/{id}`,
`POST /v1/teams/{id}/invite`, `POST /v1/teams/join`,
`DELETE /v1/teams/{id}/members/{id}`, lederskap med bekreftelse, avstemning,
utfordring av inaktiv leder, `GET/POST /v1/messages`, og redirect-URLen
`GET /i/{token}`.

**Fristene avgjøres lat.** Både avstemningen og inaktiv-leder-utfordringen
løper i 7 dager, men appen har ingen jobbkjører. I stedet avgjøres de første
gang noen spør etter dem. Utfallet blir det samme — en avstemning ingen spør
etter, har heller ingen som venter på svaret — og vi slipper en scheduler som
må driftes. Når push lander i fase 8 bør et periodisk kall legges inn, så
varselet går ut på fristen og ikke ved neste besøk.

**Invitasjonslenken svarer likt uansett om tokenet finnes.** Den deles i åpne
kanaler, og et svar som skilte gyldig fra ugyldig ville gjort den til et
oppslagsverk over hvilke lag som eksisterer. Redirecten leser User-Agent og
sender til Play eller App Store.

**SMS-invitasjon melder ærlig fra.** Siden SMS er utsatt til v2, får en
telefoninvitasjon `delivery_status: failed` med en forklaring — *og lenken
vedlagt*, så klienten kan dele den med ACTION_SEND i stedet. Telefonnumre
normaliseres til E.164; norske 8-sifrede numre får +47, alt annet må oppgis
med landkode siden vi ikke kan gjette landet.

**Overføring av lederskap krever bekreftelse** fra den valgte. Ingen skal våkne
opp som lagleder uten å ha sagt ja. Enstemmighet avslutter en avstemning tidlig,
som spec-en sier; uavgjort ved fristen gir `expired` framfor en leder kåret på
terningkast, og laget kan starte en ny.

**Meldingskøen kvitteres, ikke slettes.** `POST /v1/messages/ack` markerer raden
som levert. En klient som krasjer mellom henting og visning mister da ikke
meldingen for godt.

To ting du bør vite: `GET /v1/teams/near` leser alle lag med koordinater og
sorterer i Python — det holder lenge, men må byttes til PostGIS eller en
geohash-kolonne når tabellen vokser. Og `User.last_seen_at` oppdateres nå maks
én gang i timen per bruker, siden §11 trenger «har lederen brukt appen siste
måned?».

104 tester grønne. Ingen ny migrasjon — lag-tabellene lå allerede i
initial-migrasjonen fra fase 2.

---

## Fase 3 — innlogging (§1)

Dette er den runden som endrer noe for klienten. Fram til nå har alt som
krever bruker svart `501`; nå svarer det `401`, og det finnes en ekte vei inn.

### Det klienten må gjøre

| Endepunkt | Inn | Ut |
|---|---|---|
| `POST /v1/auth/google` | `{"id_token": "..."}` | tokenpar |
| `POST /v1/auth/apple` | `{"id_token": "..."}` | tokenpar |
| `POST /v1/auth/email/start` | `{"email": "..."}` | `202` |
| `POST /v1/auth/email/verify` | `{"email": "...", "code": "123456"}` | tokenpar |
| `POST /v1/auth/refresh` | `{"refresh_token": "..."}` | nytt tokenpar |
| `POST /v1/auth/logout` | `{"refresh_token": "..."}` | `204` |

Tokenparet er:

```json
{"access_token": "...", "refresh_token": "...", "token_type": "Bearer",
 "expires_in": 3600, "user_id": "...", "public_id": "BF-7Q4K-9F2M",
 "display_name": "Ola", "is_new": true}
```

Alt annet tar `Authorization: Bearer <access_token>`. I `Api.kt` er det én
header å legge på — som forutsatt da transportlaget ble skrevet.

### Tre ting som påvirker klientkoden

**Access-tokenet varer én time, refresh-tokenet 90 dager.** Får du `401` på et
vanlig kall, prøv `refresh` én gang og gjenta kallet. Feiler *den* også, er
brukeren logget ut på ekte og må inn i innloggingsskjermen — ikke prøv i loop.

**Refresh-tokenet roterer.** Hvert kall til `/refresh` gir et nytt og
ugyldiggjør det gamle, så det må lagres på nytt hver gang. Og viktigere:
sender du det *samme* refresh-tokenet to ganger, tolkes det som et token på
avveie og **alle** brukerens økter tilbakekalles. To parallelle kall som begge
prøver å fornye vil altså logge brukeren ut. Serialiser fornyelsen — én mutex
rundt refresh, ikke ett kall per kø-element.

**`is_new: true`** betyr at kontoen ble opprettet nå. Det er signalet til å
tilby «gjenopprett fra backup» (`GET /v1/backup/meta`) i stedet for å starte
tomt.

### Det du ikke trenger å gjøre

Ingenting i `Sync.kt` sin feilklassifisering endres. `401` er fortsatt
permanent for et køelement — men *etter* at fornyelsen over er forsøkt.

### Én ting jeg vil advare mot

Ikke lagre `access_token` i `SharedPreferences` uten videre. Det er en time
gyldig og kan ikke tilbakekalles. `refresh_token` bør inn i `EncryptedSharedPreferences`
eller Keystore — det er 90 dager gyldig og er i praksis brukerens konto.

### Hva som gjenstår før dette virker i produksjon

Fire ting, og alle er eierens å skaffe:

1. **`JWT_SECRET`** som Fly-secret — minst 32 tegn (`openssl rand -base64 48`).
   Uten den svarer `/v1/auth/*` 503.
2. **`GOOGLE_CLIENT_IDS`** fra Google Cloud Console. Android- og iOS-klienten
   har hver sin, og begge må stå der (kommaseparert) — de sjekkes mot `aud` i
   ID-tokenet.
3. **`APPLE_CLIENT_IDS`** — krever Apple-utviklerkonto.
4. **En e-postleverandør** (`RESEND_API_KEY` eller SMTP). Uten den blir
   innloggingskoden bare logget, ikke sendt.

Google og Apple kan settes uavhengig; leverandøren som mangler klient-ID
svarer 503, de andre virker.

### En feil fra forrige runde, funnet nå

`mailer.send()` hadde ingen mottakerparameter — den var skrevet for §10, der
mottakeren alltid er utviklerens innboks. Da lag-invitasjonene i fase 6
gjenbrukte den, gikk **invitasjonene til utviklerens innboks** i stedet for
til den inviterte. Typesystemet fanget det ikke, og testen sjekket bare at
`delivery_status` ble `sent`. Fikset, og testen sjekker nå mottakeren.

123 tester grønne. Én ny migrasjon (`a7c14b93d502`): `auth_sessions` og
`email_login_codes`.

---

## Fase 7 — forskningsdata (§7) og kontosletting (§9)

Endepunktet fra fase 2 viste seg å være halve jobben. Modellens egen docstring
sa at jakt-payloaden «filtreres før innsending» — men ingenting gjorde det, og
brukerens valg (`ResearchSharingPreference`) hadde ingen endepunkter i det hele
tatt. Et samtykke til jakt-typen delte i praksis *alt* klienten sendte.

### Nytt

- `GET/PUT /v1/research/sharing` — `{share_species, share_date,
  share_shot_situation, position_granularity}`. Alt av som standard,
  `position_granularity: "none"`.
- `DELETE /v1/account` — §9.

### Filtreringen skjer på serveren, og den er streng

Payloaden for `result_type: "hunt"` går gjennom en **tillatelsesliste**. Sender
du en nøkkel som ikke står der, forsvinner den stille. Det er med vilje:
feltinnholdet er ikke endelig avklart, og med en forbudsliste ville hvert nytt
felt vært delt som standard helt til noen kom på å forby det.

Kanoniske nøkler i dag:

| Valg | Nøkler som slipper gjennom |
|---|---|
| `share_species` | `species` |
| `share_shot_situation` | `shot_situation`, `shooting_position`, `position_modifier`, `distance_m`, `rest_used` |
| `position_granularity: exact` | `lat`, `lon`, `kommune`, `fylke` |
| `position_granularity: kommune` | `kommune`, `fylke` |
| `position_granularity: fylke` | `fylke` |
| `position_granularity: none` | ingen |

Merk at grovheten velger *hvilke felt* som lagres, ikke hvor mye koordinatene
avrundes. Serveren har ingen kommunegrenser å slå opp i, og «kommune» er et
navn — ikke et antall desimaler. Klienten kjenner stedet og sender navnet.

To ting til: **uten `share_date` beholdes bare året** (datoen settes til 1.
januar), og **skadedata lagres aldri** — de har ingen bryter, og «private som
standard» uten en måte å slå dem på betyr aldri.

Svaret fra `POST /v1/research/records` inneholder nå `stored_fields`. Bruk den
til å vise brukeren hva som faktisk ble delt, i stedet for å påstå noe annet.
Treningsdata filtreres ikke — §7 gir ingen felt-for-felt-valg for dem.

### Kontosletting

`DELETE /v1/account` tømmer brukerskjemaet med det samme: serier, treff,
backup, venner, lagmedlemskap, stemmer, enheter og innlogginger. **Brukerraden
slettes ikke, den tømmes** — `public_id` må stå igjen så den ikke gjenbrukes
av en ny konto, ellers ser en venn som har ID-en lagret plutselig en fremmed.

Forskningsskjemaet røres ikke herfra. Radene er pseudonymiserte, og §7 forbyr
koblingen tilbake — det er hele poenget. I stedet legges det inn en
sletteanmodning på pseudonymet, og samtykkene trekkes tilbake med én gang.
Svaret sier `research_deletion_requested: true/false`.

For klienten: etter et vellykket kall er alle tokens verdiløse (`401` ved
neste kall, også innenfor access-tokenets time), og lokal state bør nullstilles
uten å prøve `refresh`.

### Det som gjenstår er ikke kode

`RESEARCH_ENABLED` står fortsatt av, og skal stå av til personvernerklæringen
foreligger og DPIA-behovet er avklart. I tillegg finnes det ingen jobbkjører
som faktisk tømmer `research.deletion_requests` — det er et driftsansvar, og
`completed_at` er kolonnen det kvitteres i.

141 tester grønne. Ingen ny migrasjon — tabellene lå allerede i
initial-migrasjonen fra fase 2.

## Innlogging er live i produksjon

Fase 3 og 7 er deployet. `JWT_SECRET` og Resend er satt opp, og hele
e-postflyten er røyktestet mot `https://bestefar-api.fly.dev` med en ekte
adresse: kode sendt, konto opprettet, access-token gir profiltilgang,
refresh roterer, gjenbruk av et brukt refresh-token tilbakekaller alle økter,
utlogging er idempotent.

Google og Apple er fortsatt ikke konfigurert (`GOOGLE_CLIENT_IDS` /
`APPLE_CLIENT_IDS` er tomme), så `/v1/auth/google` og `/v1/auth/apple` svarer
**503**. E-postkode er altså den eneste innloggingen som virker nå, men den
virker fullt ut og kan bygges mot.

### To ting klienten MÅ håndtere

**`/logout` gjør ikke access-tokenet ugyldig.** Verifisert mot produksjon:
etter 204 fra `/logout` svarer `GET /v1/profile` fortsatt 200 med det samme
access-tokenet. Det er ikke en feil — tokenet er en statsløs JWT og kan ikke
tilbakekalles, derfor er levetiden bare 60 minutter. Men **klienten må selv
slette begge tokenene ved utlogging.** Gjør den ikke det, har en utlogget
bruker full tilgang i opptil en time.

**Ikke kjør to `/refresh` parallelt.** Refresh-tokenet roteres ved hver bruk,
og et allerede brukt token som dukker opp igjen tolkes som en lekkasje — da
tilbakekalles *alle* brukerens økter og hen må logge inn på nytt. To samtidige
kall logger altså ut brukeren. Serialiser fornyelsen bak én lås.

### `code` må sendes som streng

`{"code": 42731}` gir 422. Koden er sekssifret og kan starte med null
(`"042731"`); som JSON-tall forsvinner nullen. Bekreftet mot produksjon.

### Rettet: nye kontoer fikk «Navn under vurdering»

Visningsnavnet som utledes av e-postadressen er allerede kjørt gjennom
moderasjonen, men `display_name_status` ble stående på standardverdien
`pending`. `sharing.friend_view` viser bare navnet når statusen er `approved`,
så **hver nyopprettede konto framsto som «Navn under vurdering» for venner og
lagkamerater** — permanent, siden `PUT /v1/profile` var det eneste stedet
statusen noen gang ble satt. Rettet i `_ny_bruker`; regresjonstest
`test_nytt_navn_er_ferdig_moderert`.

Kontoer opprettet før denne rettelsen ligger fortsatt med `pending` i basen.
Det gjelder foreløpig bare testkontoen min, men merk det hvis dere ser det.

## Fase 8: push (§11)

### Enhetsregistrering

```
PUT  /v1/devices              {"push_token": "...", "platform": "android",
                               "app_version": "0.15", "model": "Pixel 8"}
GET  /v1/devices              -> liste, UTEN push_token
POST /v1/devices/unregister   {"push_token": "..."}  -> 204
```

`PUT` er idempotent — kall det ved **hver** oppstart og hver gang Firebase
roterer tokenet. Dere trenger ikke holde rede på om tokenet er nytt.

`platform` er `android` eller `ios` og defaulter til `android`.

**Kall `/unregister` ved utlogging**, sammen med `/v1/auth/logout`. Gjør dere
ikke det, fortsetter telefonen å få varsler for en konto som er logget ut.

Logger en annen bruker inn på samme telefon, flyttes enheten automatisk til den
nye kontoen — ellers ville varsler til forrige bruker havnet på en telefon som
nå er noen andres.

### Hva dere får i varselet

Ved siden av `notification.title`/`body` følger en `data`-blokk:

```json
{"kind": "team_renamed", "team_id": "..."}
```

`kind` er den samme som i meldingskøen, så trykk på varselet kan åpne riktig
skjerm direkte. Alle verdier er **strenger** — FCM tillater ikke annet.

### Push erstatter ikke meldingskøen

Dette er den viktigste setningen i hele avsnittet. `GET /v1/messages` er
fortsatt garantien for at en melding når fram; push er bare det som når
brukeren mens appen er lukket. Serveren dropper bevisst push når det tar for
lang tid, og et varsel som feiler logges og glemmes.

**Klienten må derfor fortsatt hente køen ved oppstart, akkurat som før.** Bygg
ikke noe som antar at push kom fram.

### Ikke slått på ennå

`/health` viser `"push": "log"` til `FCM_SERVICE_ACCOUNT_JSON` er satt som
Fly-secret. Endepunktene virker uansett — dere kan registrere enheter og bygge
hele flyten nå, varslene sendes bare ikke ut. Når den er satt, viser `/health`
`"push": "fcm"`.
