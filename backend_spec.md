# Bestefar — Backend-spesifikasjon (utkast v0.1)

Underlag for backend som dekker funksjonene UI-laget (v0.6) bygger front-end for,
men som ikke kan fullføres uten konto + server. Bygger på `bestefar_CV-kjerne_spec.md`
§5–§6 og den eksisterende FastAPI-skissen i `backend/` (tre atskilte ansvar).

Status: **utkast** — feltdefinisjoner merket `TODO(eier)` der forskningsinnholdet
ikke er avklart.

## 0. Prinsipper (arvet fra spec)
- **Strukturell adskillelse:** treningsdata, feilanalyse-bilder og forskningsdata i
  separate tabeller/lagre. Forskning bruker pseudonym skytter-ID + samtykketabell.
- **Offline-først:** klienten fungerer uten backend; alt køes og synkes opportunistisk.
- **Samtykke styrer alt:** ingen deling uten eksplisitt, tilbaketrekkbart samtykke.

## 1. Konto og identitet
Trengs for backup, venner og lag.
- **Innlogging:** Google / Apple / e-post ved lansering (v1, OAuth/OIDC).
  Telefonnummer (OTP) **utsatt til v2** — krever betalt SMS-leverandør
  (Twilio/Vonage o.l.) og legger til en kostnads-/integrasjonslinje som ikke
  er nødvendig for MVP.
- **Bruker-ID:** intern UUID. **Pseudonym forsknings-ID** avledet separat (ikke
  reversibelt koblet i forskningslageret).
- **Profil:** visningsnavn (≤ 24 tegn, latinske bokstaver inkl. æ/ø/å + tall/
  mellomrom/enkel tegnsetting — speiler klientens filter), fødselsår
  (egenrapportert), hjemkommune (valgfri), findable-flagg.
- **Endepunkter:** `POST /v1/auth/*`, `GET/PUT /v1/profile`.

*Implementert 2026-08-04 (fase 3):*

| Endepunkt | Inn | Ut |
|---|---|---|
| `POST /v1/auth/google` | `{id_token}` | tokenpar + `is_new` |
| `POST /v1/auth/apple` | `{id_token}` | tokenpar + `is_new` |
| `POST /v1/auth/email/start` | `{email}` | 202 `{status:"sendt"}` |
| `POST /v1/auth/email/verify` | `{email, code}` | tokenpar + `is_new` |
| `POST /v1/auth/refresh` | `{refresh_token}` | nytt tokenpar |
| `POST /v1/auth/logout` | `{refresh_token}` | 204 |

Tokenparet er `{access_token, refresh_token, token_type:"Bearer", expires_in,
user_id, public_id, display_name}`. Alle andre endepunkter tar
`Authorization: Bearer <access_token>`.

- **To ulike tokentyper, med vilje.** Access-tokenet er et kortlivet JWT
  (HS256, 60 min) som verifiseres med signatur alene — ingen databaseoppslag
  per forespørsel. Prisen er at det ikke kan tilbakekalles; derfor er
  levetiden kort, og `deleted_at` sjekkes ved hvert kall. Refresh-tokenet er
  32 tilfeldige byte, lagret som SHA-256 (`auth_sessions`), og *kan*
  tilbakekalles.
- **Rotasjon med tyverideteksjon:** hver bruk av refresh-tokenet gir et nytt
  og merker det gamle brukt. Dukker et allerede brukt token opp igjen,
  tilbakekalles *alle* brukerens økter. Vi kan ikke skille en kopi på avveie
  fra et dobbeltkjørt forsøk, og da er det trygge valget riktig.
- **Kontosammenslåing på verifisert e-post.** Samme person med Google én gang
  og e-postkode neste gang skal ha én konto. Uverifisert e-post kobles
  *aldri* — ellers kunne en konto opprettet med en annen persons adresse hos
  en slapp leverandør overta kontoen deres.
- **`aud` sjekkes alltid.** Uten `GOOGLE_CLIENT_IDS`/`APPLE_CLIENT_IDS` svarer
  den leverandøren 503. Et gyldig Google-token utstedt til en *annen* app
  skal ikke gi tilgang her.
- **E-postkode:** seks siffer, 15 min, maks 5 forsøk, maks 5 koder per adresse
  per time (telles i basen, ikke i minnet — Fly kjører to maskiner).
  `/email/start` svarer alltid 202, også for ukjent adresse: et svar som
  skilte kjent fra ukjent ville gjort endepunktet til et oppslagsverk over
  hvem som bruker appen.
- **`JWT_SECRET` må være ≥ 32 tegn** (RFC 7518 §3.2). Kortere avvises med 503
  — en HMAC-nøkkel som kan brutes lar hvem som helst utstede tokens for hvem
  som helst.
- **`X-Debug-User-Id`** finnes fortsatt, men *bare* utenfor produksjon. Den
  gjør at testene for §2–§11 slipper å sette opp Google-klienter.
- **Telefon-OTP** er fortsatt utsatt til v2 (ingen SMS-leverandør).

## 2. Backup / dataoverføring (løser «mister loggen»)
Problem: appdata forsvinner ved avinstaller/reinstall uten konto.
- **Sync:** `PUT /v1/backup` (kryptert blob: serier + jaktlogg + innstillinger,
  klient-kryptert), `GET /v1/backup`. Konfliktløsning: last-write-wins per post-ID
  (postene har allerede UUID + `ts`).
- *Implementert 2026-08-03:* bloben sendes som rå `application/octet-stream` med
  metadataene som query-parametere (sparer base64-påslaget på vår største
  nyttelast). `GET /v1/backup/meta` gir metadata uten å laste ned bloben, slik at
  «har jeg noe å gjenopprette?» på en ny telefon er et lite kall.
  **Tillegg til konfliktløsningen:** serveren avviser en `PUT` der `client_ts` er
  eldre enn den lagrede (409). Last-write-wins per post-ID kan bare håndheves
  klient-side — serveren ser ikke inn i den krypterte bloben — så uten dette
  vernet kunne en telefon som synker første gang på måneder viske ut alt som er
  logget siden. `?force=true` overstyrer ved et bevisst brukervalg
  («gjenopprett fra denne enheten»). Grense: 16 MB.
- **«Flytt til ny telefon»:** kryptert eksportfil (klient) ELLER gjenoppretting fra
  konto-backup. Nøkkel avledet fra bruker-hemmelighet.
- **Android Auto Backup** dekker oppdateringer; konto-backup dekker reinstall/bytte.

## 3. Venner (front-end finnes i `VennerActivity`)
- **Modell:** `Friend { id, displayName, teamIds[], phone?, homeKommune?,
  shotsTotal?, shotsSeason?, avgScore?, trend?, kills[] }`. Klienten har `nickAlias`
  lokalt (redigert visningsnavn) — deles ikke.
- **Søk/legg til:** `GET /v1/users/search?q=` (kun findable-brukere), by bruker-ID,
  eller QR (QR = bruker-ID/redirect-URL). Vennskap krever **aksept** hos mottaker.
  `POST /v1/friends/request`, `POST /v1/friends/respond`.
- **Deling:** hver bruker velger felt som deles med venner (visningsnavn alltid).
  Server filtrerer utgående data etter delerens valg. Deaktivering nuller delte felt.
- **Telefonnummer delt** → klienten viser ring/SMS-ikon (håndteres klient-side når
  `phone` finnes).
- **Sensur av visningsnavn:** moderasjon server-side før navnet eksponeres for andre
  (regelsett + evt. manuell kø). Avvist navn deles ikke; brukeren varsles.
- *Implementert 2026-08-03:* regelsettet håndhever tegnsett og lengde (speiler
  `Ui.nameFilters()` — klientfilteret er bekvemmelighet, ikke sikkerhet) pluss en
  ordliste satt med `DISPLAY_NAME_BLOCKLIST`. Ordlista sammenlignes på en foldet
  form (uten aksenter, tegnsetting og store bokstaver), så «S-t-y-g-t» ikke
  slipper unna. Avvist navn **lagres ikke i det hele tatt** — da kan det heller
  ikke lekke. Er navnet ikke godkjent, eksponeres «Ukjent skytter» for andre.
  Den manuelle køen krever en admin-flate som ikke finnes ennå; navn som passerer
  regelsettet godkjennes derfor direkte.
- **Åpent punkt — `kills[]` kan ikke leveres:** jaktloggen ligger inne i den
  klient-krypterte backup-bloben (§2), som serveren ikke kan lese. Skal felte dyr
  deles med venner, må jaktposter synkes som egne rader. Det er en
  spec-avklaring, ikke en implementasjonsdetalj. `avgScore` og `trend` er
  implementert fra treningsseriene; `trend` er foreløpig definert som snitt per
  skudd i de 5 siste seriene minus de 5 foregående, og er `null` før det finnes
  10 serier.

## 4. Lag (jaktlag/skytterlag)
- **Modell:** `Team { id, name, kind(jakt|skytter), memberCount, location?, leaders[] }`.
- **Nærliggende lag:** `GET /v1/teams/near?lat=&lon=&r=50000` → inntil 20, sortert
  etter avstand; de 3 nærmeste alltid med uansett avstand.
- **Oppretting + roller:** «jeg er leder» / «opprett for leder» / «be leder opprette».
  Flere ledere mulig.
- **Invitasjon:** ACTION_SEND-intent (ingen kontakttillatelse) med **redirect-URL** i
  EXTRA_TEXT; server leser User-Agent → riktig butikk (Play/App Store). Samme URL som
  QR for e-post/SMS-invitasjon. `POST /v1/teams/{id}/invite { emailOrPhone }` med
  validering (identifiser e-post vs telefon) og **leveringskvittering/-feil** tilbake.
- *Implementert 2026-08-03:* `GET /i/{token}` leser User-Agent og svarer 302 til
  Play/App Store. Den svarer likt **uansett om tokenet finnes** — lenken deles i
  åpne kanaler, og et svar som skilte gyldig fra ugyldig ville gjort den til et
  oppslagsverk over hvilke lag som eksisterer. Telefonnumre normaliseres til
  E.164 (norske 8-sifrede får +47). Siden SMS er utsatt til v2, får en
  telefoninvitasjon `delivery_status: failed` **med lenken vedlagt**, slik at
  klienten kan dele den via ACTION_SEND i stedet.
- `GET /v1/teams/near` leser alle lag med koordinater og sorterer i Python. Det
  holder lenge, men må byttes til PostGIS eller en geohash-kolonne når tabellen
  vokser.

## 5. Treningsresultater (utvidelse av `/v1/stats`)
- Per serie: `{ id, ts, weaponId, distanceM, position, modifier, shots[], corrected,
  seasonKey }`. Klienten køer usendte (uploaded=false).
- Brukes til venners delte statistikk (snitt/utvikling) og til brukerens egen backup.

## 6. Feilanalyse-bilder / OCR-donasjon (utvidelse av `/v1/failed-analyses`)
- Klienten køer i `filesDir/dev_uploads/{seriesId}_{tag}.jpg|json` når brukeren har
  godtatt bildedeling (oppstartsvindu 2) eller per-innsending.
- **Endepunkt:** `POST /v1/failed-analyses` (multipart: bilde + JSON med detekterte
  poeng, OCR-poeng, `tag` ∈ {ocr_match, ocr_mismatch, rejected}).
- **Lagring:** bilder lagres i S3-kompatibel objektlagring (Cloudflare R2), ikke i
  selve databasen — kun metadata/JSON lagres relasjonelt.
- Formål: kalibrere OCR-heuristikken og CV-kjernen (bl.a. **over-deteksjon av treff**,
  se §8).

## 7. Forskningsdata (`/v1/research`, strukturelt adskilt)
- To resultattyper: **trening** og **jakt**, som separate modeller.
- Jakt-deling styres av brukerens valg: `{ vilt?, dato?, posisjon(grovhet)?,
  skuddsituasjon? }`. Skadedata private som standard.
- Samtykketabell: `{ pseudonymId, type(trening|jakt), granted_at, revoked_at? }`.
- *Implementert 2026-08-02:* tabellene ligger i et eget Postgres-**skjema**
  (`research.consents`, `research.records`, `research.deletion_requests`) uten
  fremmednøkler til brukertabellene. Pseudonymet **avledes** med
  HMAC-SHA256(server-hemmelighet, bruker-UUID) i stedet for å lagres i en
  oppslagstabell — en slik tabell ville vært nettopp den reversible koblingen
  denne paragrafen forbyr. Konsekvens: hemmeligheten kan ikke roteres uten å
  bryte koblingen til allerede innsamlede forskningsdata.
- Endepunktet er sperret av flagget `RESEARCH_ENABLED` (av som standard), på
  linje med `Dialogs.RESEARCH_ENABLED` i klienten.
- `# TODO(eier): konkret feltinnhold for forskning ikke endelig avklart` (jf. kjerne-spec §6).
- **Forutsetning før aktivering i produksjon:** personvernerklæring må foreligge,
  og det må avklares om Datatilsynet krever DPIA (sannsynlig, gitt forskningsformålet
  og sensitiviteten i jaktdata). Endepunktet kan bygges teknisk før dette, men reell
  innsamling fra brukere venter til det juridiske er på plass.

## 8. CV-kjerne-oppgaver (ikke backend, men avdekket i felt)
Notat til kjerne-repoet (versjonert bump av pinnen ved endring):
- **Over-deteksjon av treff** (flere merker enn reelt): undersøk om kamerabevegelse /
  multieksponering gir doble merker. Tiltak å vurdere: strammere auto-capture-
  stabilitet, og dedup av treff som ligger nærmere enn X ringavstander i `hits`/`overlap`.
- **OCR i kjernen:** UI-et bruker foreløpig ML Kit on-device. Vurder om skjerm-OCR bør
  flyttes til kjernen for konsistens og for å utnytte skjermens kjente layout.
- **`bf_version()` i `bestefar_ffi.h`:** klienten sender i dag appens `versionName`
  som `core_version` i §6-donasjonene. Det er en dårlig stedfortreder — UI-runder
  bumper versjonen uten at kjernen er rørt, så det ser ut som kjernen endret seg
  når den ikke gjorde det. Donasjonenes formål er å kalibrere kjernen, og en måling
  uten å vite hvilken kjerne som produserte den kan ikke brukes til det. Kolonnen
  `core_version` finnes allerede og tar imot hva som helst; dette er en ren
  kjerne-endring.

## 9. Sikkerhet / personvern
- All PII kryptert i ro og i transitt. Forsknings-ID ikke reversibelt koblet til konto.
- Sletting: `DELETE /v1/account` (lokalt + sletteanmodning via pseudonym-ID for
  forskningslageret).
- **Personvernerklæring + DPIA:** se §7 — forutsetning for aktivering av
  forskningsdatainnsamling, ikke bare en implementasjonsdetalj.

## 3.1 Bruker-ID og misbruksvern (musingsUI runde 5-spørsmål)
Designsvar på eierens spørsmål om venne-ID og søk:
- **Unik bruker-ID:** kort, håndskrivbar streng. Forslag: 8–10 tegn fra et
  forvekslingssikkert alfabet (Crockford base32: utelater I/L/O/U), f.eks.
  `BF-7Q4K-9F2M`. 9 base32-tegn ≈ 34 bit ≈ 68 mrd. ID-er — rikelig kapasitet for
  internasjonal spredning, samtidig som den er lett å lese opp/skrive ned. Vises
  også som QR (redirect-URL, jf. §4).
- **Feiltastingsvern:** ID-en har innebygd sjekksiffer (siste tegn), så åpenbare
  tastefeil avvises uten oppslag. Rommet er stort nok til at gjetting er
  upraktisk (forventet ~10⁴ reelle brukere mot 10¹⁰ mulige ID-er).
  *Implementert 2026-08-02* (`backend/app/services/ids.py`): **8 signifikante
  tegn** — 7 tilfeldige + sjekksiffer — vist som `BF-XXXX-XXXX`, altså akkurat
  eksempelet over. Det gir ~3,4 · 10¹⁰ ID-er; ett tegn kortere enn de «9 tegn»
  teksten nevner, men fortsatt langt over det gjettingsargumentet krever, og
  lettere å lese opp. Sjekksifferet er `sum mod 32` i samme alfabet (ikke
  Crockfords mod-37-variant, som ville trukket inn symboler utenfor alfabetet).
  Innlesing folder I/L→1 og O→0, så vanlige lesefeil godtas.
- **Søk etter bruker:** kun brukere med `findable=true` er søkbare.
  - Telefonsøk: hard rate-limit per konto/enhet — 5 mislykkede telefonsøk på én
    dag → karantene. Anbefaling: **1 dag** karantene ved første overtredelse,
    eskalerende til 7 dager ved gjentakelse. (Balanserer bruksvennlighet mot
    enumereringsangrep på telefonnumre, som er personsensitive.)
  - ID-søk: lavere risiko (ID er ikke PII), men samme prinsipp med mildere terskel.
  - **IP-heuristikk:** rate-limit også per IP/subnett for å stoppe automatiserte
    søk fra én kilde; CAPTCHA ved terskel. Logg kun aggregert, ikke per-søk-PII.
- **Personvern:** vennskap krever gjensidig aksept; ingen data deles før aksept.
  Telefonnummer deles kun hvis brukeren selv har krysset det av.

## 0.1 Infrastruktur (tillegg til §0)
Konkretisering av vertsvalg og driftsoppsett for backend-MVP:
- **Vert:** Fly.io, region Amsterdam (EU) — lav driftsbyrde, gratis-tier dekker
  lav trafikk i tidlig fase. Alternativ: Google Cloud Run (scale-to-zero, betal
  kun for faktisk bruk) hvis trafikkmønsteret er svært sporadisk.
  *Etablert 2026-08-02:* app `bestefar-api`, `https://bestefar-api.fly.dev`.
- **Database:** Postgres. **Valgt: Supabase** (EU-region, prosjekt
  `Bestefar_base`) — administrasjons-UI på kjøpet. Forskningsdata i
  egne tabeller/skjema i samme database — ikke fysisk separat database
  (jf. §0 strukturell adskillelse).
- **Objektlagring:** Cloudflare R2 (S3-kompatibelt API, gratis egress) for
  feilanalyse-bilder (§6, oppdatert). Bilder lagres aldri i selve databasen.
- **Secrets:** Fly secrets (eller tilsvarende hos valgt vert) — samme prinsipp
  som `gradle.properties`-mønsteret for signeringsnøkkelen (aldri i
  versjonskontroll).
- **CI/CD:** GitHub Actions, push til `main` → automatisk deploy.
- **Domene:** eget domene for API-endepunkt og OAuth-redirect-URLer.
- **Region/GDPR-begrunnelse:** EU/EØS-hosting valgt for å forenkle
  personvern-compliance — unngår overføringsmekanismer for tredjelandsoverføring
  som ellers kreves ved USA-only-tjenester (jf. Schrems II).

## 10. Direkte melding til utvikler (musingsUI runde 5)
Eier ønsket å slippe å åpne e-postapp. Krever backend:
- `POST /v1/feedback { subject, body, appVersion, deviceModel, userId? }` →
  serveren videresender til utviklerens innboks (eller sak-system).
- Klienten bruker foreløpig ACTION_SENDTO (mailto) med `subject` som e-post-Subject.
  Bytt til endepunktet når backend finnes.

## 11. Lag-medlemskap, lederskap og varsler (musingsUI runde 6)
UI-et (TeamPageActivity) bygger front-end for dette; alt reelt krever backend.
- **Medlemskap:** `Team.members[]`, `Team.leaders[]` (flere ledere mulig).
  Egen bruker vises alltid i lagets medlemsliste.
- **Invitasjoner:** «Inviter medlemmer» → kontaktliste/e-post/telefon (jf. §4).
- **Endre lagnavn:** varsel til alle medlemmer: «Navneendring — Lagleder for X har
  endret navnet på laget til Y». Vises som første melding ved neste app-åpning
  (klienten trenger en «pending messages»-kø hentet ved oppstart).
- **Fjern medlem:** varsel til den fjernede: «Lagleder har fjernet deg fra X».
- **Overfør lederskap:** velg eksisterende medlem → bekreftelse hos valgt leder.
- **Velg leder (ingen leder):** avstemning med 7-dagers nedtelling (vises som
  dager → timer < 24 → minutter < 120). Push til alle medlemmer. Stemmer kan
  endres til fristen; enstemmighet avslutter tidlig. `POST /v1/teams/{id}/vote`,
  `GET /v1/teams/{id}/vote-status`.
- **Fjern inaktiv lagleder:** hvis leder har brukt appen siste måned → «Lagleder
  er ikke inaktiv. Ta kontakt.» Ellers push til leder + 7-dagers timer; logger
  leder på, avbrytes; ellers mister leder lederstatus (forblir medlem), og laget
  kan velge ny leder.
- **Push-varsler:** krever FCM/APNs-registrering per enhet.
- *Implementert 2026-08-03:* meldingskøen er `GET /v1/messages` +
  `POST /v1/messages/ack`. Kvittering **markerer** raden som levert i stedet for
  å slette den, så en klient som krasjer mellom henting og visning ikke mister
  meldingen. Køen erstatter ikke push — push når brukeren mens appen er lukket,
  køen er garantien for at meldingen når fram til slutt.
- **Frister avgjøres lat, ikke av en bakgrunnsjobb:** både avstemningen og
  inaktiv-leder-utfordringen har 7-dagers frist, men appen har ingen jobbkjører.
  De avgjøres første gang noen spør etter dem. Utfallet blir det samme — en
  avstemning ingen spør etter, har ingen som venter på svaret — men når push
  (fase 8) er på plass bør et periodisk kall legges inn, så varselet går ut på
  fristen og ikke ved neste besøk.
- Uavgjort ved fristen gir `expired`, ikke en leder kåret på terningkast; laget
  kan starte en ny avstemning. Overføring av lederskap krever **bekreftelse** fra
  den valgte — ingen skal våkne opp som lagleder uten å ha sagt ja.

## 12. Klientens transportlag (Android, v0.14)
Klientsiden av kontrakten. Bygget mot fase 1-endepunktene og verifisert mot
`https://bestefar-api.fly.dev` 2026-08-02 (`/v1/feedback` → 202, `/v1/failed-analyses` → 201).

- **`Api.kt`** — `HttpURLConnection`, ingen nettverksbibliotek. Basis-URL fra
  `BuildConfig.API_BASE_URL` (release: `https://bestefar-api.fly.dev`, debug:
  `http://10.0.2.2:8000`), overstyrbar i felt via DevTools → «API-adresse».
  Klartekst-HTTP er tillatt **kun** i debug (`src/debug/AndroidManifest.xml`).
- **Feilklassifisering:** `retryable` = `code == 0` (nådde aldri fram), 408,
  429, ≥ 500. Alt annet (400/413/422) er permanent — køen kaster elementet i
  stedet for å prøve i evig tid. Serveren bør derfor svare 4xx på data den
  aldri vil kunne ta imot, og 5xx/429 på alt som kan gå bra senere.
- **`Sync.kt`** — filbasert kø i `filesDir/dev_uploads`, ett par
  `{seriesId}_{tag}.jpg` + `.json` per innsending. Tømmes ved appstart og på
  «Send nå» i Avanserte innstillinger. Ikke WorkManager: sending mens appen er
  lukket er ikke et krav ennå.
- **Sidecar-format v2** (feltnavnene mappes 1:1 til multipart-feltene):
  ```json
  {"v":2,"series_id":"<uuid>","tag":"ocr_match|ocr_mismatch|rejected",
   "status_code":0,"confidence":0.83,"core_version":"0.14",
   "detected":[10.4,9.8],"ocr":[10.4,9.9]}
  ```
  `detected` er alltid poengene **CV-kjernen** ga, også når OCR har overskrevet
  visningen — ellers ville en `ocr_match`-donasjon ikke si noe om hva kjernen så.
- **`confidence = -1.0`** betyr *ukjent*: sendes for kø-filer skrevet før v0.14
  (format v1 hadde bare `detected` + `tag`). Behandle som «ikke målt», ikke som
  lav konfidens.
- **`core_version`** er foreløpig appens `versionName`. CV-kjernen eksponerer
  ingen egen versjon over FFI ennå; når den gjør det, skal den brukes her.
- **Kun wifi** (`Store.uploadWifiOnly`, default på): køen er fullskala-JPEG-er.
  «Send nå» overstyrer valget, men ikke «er vi på nett i det hele tatt».
- **`/v1/feedback`** er koblet inn i «Melding til utvikler». Feiler kallet
  (annet enn 429) faller klienten tilbake til `mailto:` — meldingen skal aldri
  gå tapt fordi nettet er nede. 429 gir en egen melding, siden e-post-fallback
  der bare ville laget duplikater.
- **Ikke koblet inn ennå:** `/v1/stats` (krever `deps.current_user`) og
  `/v1/research` (sperret av `Dialogs.RESEARCH_ENABLED`). `SeriesRecord.uploaded`
  står fortsatt urørt og venter på konto (§1).
