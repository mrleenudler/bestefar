# Backendens kontrakt utad

**Eier: backend-instansen.** Dette er hva klienten og de andre områdene kan
stole på — statuskoder, idempotens, grenser, og hva vi *ikke* lagrer.

**Hva som hører hjemme her:** garantier andre kan bygge på. **Ikke** hvorfor det
ble slik — det er beslutninger, og de bor i `BESLUTNINGER.md`. Samme skille som
`android/KONTRAKT.md`.

**Eierskap:** en invariant eies av den som **håndhever** den, ikke den som må
adlyde. `retryable` står derfor i `android/KONTRAKT.md` selv om vi må rette oss
etter den; tyverideteksjonen ved refresh står her selv om klienten må unngå å
utløse den.

Alt under er verifisert mot koden 2026-08-07. Filreferansene er der du kan
etterprøve det.

---

## 0. Statuskoder — hva vi lover om klassen

Klienten kaster permanent avviste elementer og prøver de midlertidige på nytt
(`android/KONTRAKT.md` §1). Vi retter oss etter det:

- **4xx betyr «send aldri dette igjen».** 400/413/422 brukes kun på nyttelast vi
  aldri vil kunne ta imot.
- **429 og 5xx betyr «prøv igjen senere».** Kvoter, sperrefrister og
  driftsfeil svarer alltid i den klassen.
- **503 betyr «funksjonen er ikke slått på på serveren»**, ikke «du gjorde
  noe galt». Brukes når en secret mangler: `JWT_SECRET`,
  `GOOGLE_CLIENT_IDS`/`APPLE_CLIENT_IDS`, `BACKUP_ESCROW_SECRET`,
  `RESEARCH_PSEUDONYM_SECRET`.

`GET /health` svarer **alltid 200** så lenge prosessen lever. Databasens
tilstand står i kroppen, ikke i statuskoden — ellers ville en midlertidig
DB-feil fått Fly til å rulle tilbake en frisk deploy.
Felter: `status`, `env`, `database`, `mailer`, `push`, `escrow`.

## 1. Innlogging og økt

`services/tokens.py`, `routers/auth.py`, `deps.py`.

**To tokentyper, og de oppfører seg ulikt:**

| | Access | Refresh |
|---|---|---|
| Form | JWT HS256, `iss: bestefar-api` | 32 tilfeldige byte, ugjennomsiktig |
| Levetid | `ACCESS_TOKEN_TTL_MINUTES`, standard 60 | `REFRESH_TOKEN_TTL_DAYS`, standard 90 |
| Lagres hos oss | nei | kun SHA-256 av det |
| Kan tilbakekalles | **nei** | ja |

- **Access-tokenet kan ikke tilbakekalles.** `/v1/auth/logout` rører det ikke;
  det lever ut levetiden sin. Det finnes ingen sperreliste.
- **Refresh roterer ved hver bruk.** Den gamle raden merkes brukt, en ny
  utstedes i samme svar.
- **Gjenbruk av et allerede brukt refresh-token tilbakekaller *alle* brukerens
  økter.** Vi kan ikke skille en kopi på avveie fra et dobbeltkjørt kall.
  Klienten garanterer at fornyelser er serialiserte
  (`android/KONTRAKT.md` §6) — uten den garantien logger normal bruk brukeren ut.
- **`deleted_at` sjekkes ved hvert kall**, på begge innloggingsveier.
- **`/v1/auth/logout` er idempotent** — ukjent eller allerede tilbakekalt token
  gir også 204.

**401 er normal trafikk herfra, ikke et angrepsmønster.** Klienten fornyer
**reaktivt**: den holder ikke øye med `expires_in`, men lar tokenet gå ut, får
et 401, fornyer og prøver forespørselen om igjen nøyaktig én gang
(`android/KONTRAKT.md` §6). Med 60 minutters levetid betyr det at hver aktive
enhet produserer minst ett 401 i timen, jevnt fordelt over døgnet.

Konsekvensen er en regel, ikke bare en observasjon: **bygg aldri
misbruksdeteksjon eller rate-limiting på antall 401.** Den ville slått ut på de
mest aktive brukerne først. Skal noe telles for å finne misbruk, tell
mislykkede *innlogginger* (`/v1/auth/email/verify`, `/google`, `/apple`) — de
har ingen legitim grunn til å gjenta seg.

**Tokenparet** som returneres av `/google`, `/apple`, `/email/verify` og
`/refresh`: `{access_token, refresh_token, token_type: "Bearer", expires_in,
user_id, public_id, display_name}`. Innloggingene legger dessuten på `is_new`.

**`aud` verifiseres alltid** mot `GOOGLE_CLIENT_IDS` / `APPLE_CLIENT_IDS`. Tom
liste ⇒ 503 for den leverandøren. Et gyldig Google-token utstedt til en *annen*
app gir ikke tilgang her.

**Kontosammenslåing skjer kun på verifisert e-post.** Uverifisert adresse kobles
aldri til en eksisterende konto.

### E-postkode

- **`/v1/auth/email/start` svarer alltid 202**, også for ukjent adresse.
  Svaret skiller aldri kjent fra ukjent.
- Svarkropp: `{status: "sendt", resend_after_seconds, expires_in_minutes}`.
  Les fristene herfra — ikke hardkod dem.
- **429 med `Retry-After`** innen sperrefristen (`EMAIL_CODE_RESEND_COOLDOWN_SECONDS`,
  standard 60 s). Ingen e-post sendes da.
- **429 uten `Retry-After`** når timekvoten er brukt opp
  (`EMAIL_CODE_RATE_PER_HOUR`, standard 10 per adresse). Kvoten teller
  *forespørsler*, ikke leveranser.
- Koden er seks siffer, gyldig `EMAIL_CODE_TTL_MINUTES` (15), maks
  `EMAIL_CODE_MAX_ATTEMPTS` (5) forsøk. Feil kode ⇒ 401, brukt opp ⇒ 429.
- **Utsendingsfeil røpes ikke.** Feiler e-posten, svarer vi fortsatt 202 —
  alt annet ville vært et signal om adressen finnes.

## 2. Backup

`routers/backup.py`.

- **Bloben er ugjennomsiktig.** Vi validerer ikke innholdet og gjør ingen
  konfliktløsning inne i den.
- **`PUT /v1/backup` avviser eldre `client_ts` med 409**, med kroppen
  `{melding, lagret_client_ts, innsendt_client_ts}`. **Likt** `client_ts`
  godtas — en omprøving etter et avbrutt kall skal ikke feile.
  `?force=true` overstyrer.
- **Grensen er 16 MB** (`MAX_BACKUP_BYTES`). Sjekkes både på `Content-Length` og
  på faktisk lest kropp ⇒ 413. Tom kropp ⇒ 422.
- **`GET /v1/backup`** returnerer `application/octet-stream` med metadata i
  hoder: `X-Backup-Schema-Version`, `X-Backup-Device-Id`, `X-Backup-Client-Ts`,
  `X-Backup-Updated-At`.
- **`GET /v1/backup/meta`** gir `{bytes, schema_version, device_id, client_ts,
  updated_at, escrowed}` uten å laste ned bloben. 404 hvis ingenting er lagret.
- **`DELETE /v1/backup` fjerner bloben, ikke deponeringen.** 204 også når det
  ikke fantes noe.

**405 på `GET /v1/backup/meta` og `GET /v1/backup/key-escrow` i loggene er
gamle klienter, ikke rutefeil.** Begge rutene er og har vært `GET`. Klientfeilen
er rettet i v0.19; 405-ene avtar etter hvert som appene oppdateres. Ikke bruk
dem som grunnlag for å endre rutene.

### Nøkkeldeponering

- **`PUT /v1/backup/key-escrow`** er idempotent; en ny PUT erstatter materialet.
  Svar: `{escrowed: true, updated_at}`.
- **`GET`** gir `{key_material, updated_at}`, eller 404 hvis ingenting er
  deponert. **`updated_at` betyr «sist brukeren deponerte»** — en intern
  omkryptering flytter den ikke.
- **`DELETE` virker uten at hemmeligheten er konfigurert**, og er idempotent.
  Veien ut av valget kan ikke feile.
- **Uten `BACKUP_ESCROW_SECRET` svarer PUT og GET 503.** Vi lagrer heller
  ingenting enn nøkler i klartekst.
- **Er materialet uleselig, svarer GET 503** — aldri byte klienten kunne finne
  på å dekryptere bloben med.
- Grenser: `key_material` må være gyldig base64 (ellers 422), maks
  `MAX_ESCROW_BYTES` = 512 byte dekodet (ellers 413).

## 3. Feilanalyse-mottaket

`routers/failed_analyses.py`. **Vi eier feltnavnene i multipart-skjemaet.**

| Multipart-felt | Type | Merknad |
|---|---|---|
| `status_code` | int | påkrevd |
| `confidence` | float | påkrevd; `-1.0` = ukjent |
| `core_version` | str | påkrevd |
| `tag` | enum | `ocr_match` \| `ocr_mismatch` \| `rejected`, standard `rejected` |
| `series_id` | str? | valgfri |
| `detected_scores` | JSON-liste som streng | standard `""` |
| `ocr_scores` | JSON-liste som streng | standard `""` |
| `image` | fil | påkrevd |

**Merk at feltnavnene *ikke* er identiske med sidecar-JSON-ens nøkler:** sidecar
har `detected` og `ocr`, skjemaet her har `detected_scores` og `ocr_scores`, og
sidecarens `v` sendes ikke. Kartleggingen skjer i klienten (`Sync.kt`).
`android/KONTRAKT.md` §2 påstår «1:1» og tar feil — meldt som issue med label
`ui`.

- **`tag`-enumet er vårt** (`models/base.py`, `FailedTag`). En ukjent verdi gir
  422 fra FastAPI-valideringen.
- **Endepunktet krever ikke innlogging.** Donasjonen henger på
  bildedelings-samtykket, ikke på konto.
- Ugyldig JSON i poengfeltene ⇒ 422. Bilde over `MAX_UPLOAD_BYTES` (8 MB) ⇒ 413.
- Svar: `201 {id}`.

## 4. Enheter og push

`routers/devices.py`, `services/push.py`.

- **`PUT /v1/devices` er idempotent på `push_token`, ikke på bruker.** Finnes
  tokenet under en annen konto, **flytter raden seg** til den som registrerer nå.
  Svar: `{id, platform}`.
- **`GET /v1/devices` returnerer aldri `push_token`.**
- **`POST /v1/devices/unregister` rører bare egne enheter**, og gir 204 også for
  et ukjent token.
- Alle tre krever innlogging.
- **Vi sender en `notification`-blokk** pluss `data` og
  `android.priority: "high"`. Bytter vi til ren `data`-melding, varsler vi først
  — det flytter kontrollen over utseendet fra manifestet til klientkode.
- **Alle `data`-verdier sendes som strenger**, og `None` fjernes. FCM avviser
  tall og `null`.
- **Døde tokens slettes** (`UNREGISTERED`, `INVALID_ARGUMENT`, `NOT_FOUND`).
- **`PUSH_BUDGET_SECONDS` (6 s) avbryter resten av en utsending.** Sjekken skjer
  **før hvert kall**, aldri under, og timeouten per kall er 5 s — så budsjettet
  tåler nøyaktig ett tregt kall. Målt med ti mottakere: 5 s per kall ⇒ **2
  forsøkt, 8 aldri forsøkt, 10 s veggtid**; 0,2 s per kall ⇒ alle ti på 2 s.
  Øvre veggtid er `budsjett + timeout` = 11 s, ikke 6. Se issue #3.

  **Avbruddet er en utsettelse, ikke tap** — klienten henter køen ved appstart
  fra og med v0.19. Se §9 for hva den garantien faktisk dekker.
- **Uten `FCM_SERVICE_ACCOUNT_JSON` logges push bare**, og `/health` sier
  `"push": "log"`. Verdien godtas både som rå JSON og base64.

## 4.1 Meldingskøen

`routers/messages.py`, `models/social.py` (`PendingMessage`).

Køen er **garantien** for §11-varsler; push er den raske leveringen. Begge
krever innlogging, og en bruker ser bare sine egne meldinger.

**`GET /v1/messages`** returnerer en JSON-liste med uleverte meldinger, eldste
først (`created_at` stigende). Tom liste er 200, ikke 404.

| Felt | Type | Merknad |
|---|---|---|
| `id` | **`int`** | Autoinkrement. **Ikke** UUID — bryter med mønsteret ellers i API-et, men er den faktiske typen. |
| `kind` | `string`, ≤ 32 tegn | **Fri streng, ikke enum.** Se under. |
| `title` | `string`, ≤ 120 tegn | |
| `body` | `string` | Ubegrenset (`TEXT`). |
| `team_id` | `string \| null` | UUID når meldingen gjelder et lag, ellers `null`. |
| `created_at` | `string` | ISO-8601 med UTC-offset. |

**De to avvikene er reelle og bevisste å dokumentere, ikke å skjule:**

- **`id` er et heltall mens `team_id` er en streng-UUID eller `null`.** En klient
  som antar samme type i begge, kompilerer og feiler stille. `id` er
  primærnøkkelen i en tabell bare vi eier og som aldri deles på tvers av
  installasjoner; `team_id` er en fremmednøkkel til en ID klienten også kjenner.
- **`kind` er en fri `String(32)`, ikke et enum.** Serveren validerer den ikke,
  og listen kan vokse uten API-versjonering. De sju verdiene som finnes i koden
  i dag:

  | `kind` | Sendes til | Utløses av |
  |---|---|---|
  | `team_renamed` | alle andre medlemmer | lagnavn endret |
  | `removed_from_team` | den fjernede | medlem fjernet |
  | `leadership_offered` | den utpekte | lederskap tilbudt, krever bekreftelse |
  | `election_started` | alle medlemmer | lederavstemning åpnet, 7 dager |
  | `leader_challenged` | lederen alene | utfordring av inaktiv leder, 7 dager |
  | `leader_elected` | alle medlemmer | avstemning avgjort |
  | `leader_demoted` | alle medlemmer | leder mistet rollen ved frist |

  **Klienten må derfor behandle en ukjent `kind` som gyldig** og falle tilbake
  på å vise `title` + `body`, som alltid er ferdig formulert tekst på norsk. En
  klient som `switch`-er uttømmende på `kind` vil før eller siden møte en verdi
  den ikke kjenner. Trenger dere en lukket liste, er det en **ny** avtale — ikke
  noe dere kan lese ut av dagens verdier.

**`POST /v1/messages/ack`** tar `{"ids": [int, ...]}` og svarer **204**.

- **Kvittering markerer, den sletter ikke.** Raden får `delivered_at`, så en
  klient som krasjer mellom henting og visning ikke mister meldingen for godt.
- **Idempotent og tolerant:** ukjente ID-er, andres ID-er og allerede kvitterte
  ID-er ignoreres uten feil. Tom liste er også 204.
- Det finnes **ingen delvis-suksess-respons**. Vil dere vite hva som faktisk ble
  kvittert, hent køen på nytt.

**Meldingen er ferdig formulert av serveren.** `title` og `body` er norsk
brukertekst med æøå, ikke nøkler klienten skal oversette. Endrer vi ordlyden,
endres den for alle klienter samtidig — det er tilsiktet.

## 5. Forskningsdata

`routers/research.py`, `services/research_filter.py`, `services/pseudonym.py`.

- **Sperret av `RESEARCH_ENABLED`** (av som standard) og av at
  `RESEARCH_PSEUDONYM_SECRET` finnes — uten den 503.
- **Skytter-ID er et avledet pseudonym**, aldri en oppslagstabell. Det finnes
  ingen vei fra en forskningsrad tilbake til en konto uten hemmeligheten.
- **Jakt-payload filtreres server-side mot en tillatelsesliste.** Ukjente nøkler
  droppes stille. Kanoniske nøkler:
  - `share_species` → `species`
  - `share_shot_situation` → `shot_situation`, `shooting_position`,
    `position_modifier`, `distance_m`, `rest_used`
  - `share_injury_data` → `wounded`, `injury`, `hit_placement`, `shots_fired`,
    `tracking_distance_m`, `tracking_time_min`, `dog_used`, `recovered`
- **Posisjonsgrovheten velger hvilke felt som lagres**, ikke hvor mye
  koordinater avrundes: `exact` → `lat`/`lon`/`kommune`/`fylke`, `kommune` →
  `kommune`/`fylke`, `fylke` → `fylke`, `none` → ingenting.
- **Uten datosamtykke beholdes bare året** (1. januar).
- **Treningsdata filtreres ikke** — §7 gir ingen felt-for-felt-valg for dem.
- Svaret inneholder `stored_fields`, så klienten kan vise hva som faktisk ble
  delt.

## 6. Venner, søk og lag

- **Søk gir kun eksakt treff** på `public_id` eller telefon, og kun for
  `findable`-brukere. Ingen fritekstsøk på navn.
- **Sjekksifferfeil i en bruker-ID gir 422 uten oppslag**, og teller ikke som
  bom mot karantenen.
- **Karantenen ligger i databasen** (`services/quarantine.py`), så den overlever
  omstart og gjelder på tvers av begge Fly-maskinene. Bare *mislykkede* søk
  telles.
- **Utgående deling filtreres server-side** (`services/sharing.py`).
  «Deaktivering nuller delte felt» er en garanti, ikke en klientdetalj.
- **Et avvist visningsnavn lagres ikke i det hele tatt.** Andre ser «Ukjent
  skytter».
- **`GET /i/{token}` svarer 302 til butikken uansett om tokenet finnes.**
  Lenken deles i åpne kanaler og skal ikke kunne brukes til å sjekke hvilke lag
  som eksisterer.
- **Telefoninvitasjon gir `delivery_status: failed` med lenken vedlagt** — det
  finnes ingen SMS-leverandør (ÅP-E9), så klienten deler den selv.

## 7. Kunngjøring av felt dyr

`routers/hunts.py`.

- **Ingenting lagres om hva som ble felt eller hvor.** Kun
  `users.hunt_announced_at` — et tidsstempel. Varselet er flyktig; når det ikke
  fram, er det borte.
- **Teksten er `«{navn} har felt {species}{ i kommune}.»`** Bøyningen og den
  ubestemte artikkelen ligger i `species`-strengen klienten sender — serveren
  legger ikke til noe.
- Krever `share_kills` (403 uten). Maks én kunngjøring per 5 minutter (429).
- Er visningsnavnet ikke godkjent, brukes «En venn».
- Svar: `{status, message, devices_notified}` — antall **enheter**, ikke venner.

## 8. Kontosletting

`routers/account.py`.

- **Brukerskjemaet tømmes med det samme.** Serier, treff, backup, deponert
  nøkkel, venner, lagmedlemskap, stemmer, enheter og økter er borte når kallet
  returnerer.
- **Brukerraden slettes ikke, den tømmes.** `public_id` blir stående så den
  ikke kan gjenbrukes.
- **Forskningsskjemaet røres ikke.** Det legges inn en sletteanmodning på
  pseudonymet, og alle samtykker trekkes tilbake med én gang.
  **Anmodningen tømmes ikke av noen jobb** — ÅP-E2.

## 9. Kjente unøyaktigheter

Ærlighet om hva kontrakten *ikke* holder:

- **Kø-garantien gjelder fra oppstart til oppstart, ikke i sanntid.**
  *Rettet 2026-08-08:* klienten henter køen fra og med v0.19 (`Messages.kt`,
  issue #4 lukket) — påstanden om at den ikke gjorde det, sto her i ett døgn
  etter at den sluttet å være sann.

  Det som fortsatt er verdt å vite: hentingen skjer **ved appstart**, ikke
  løpende. En melding som oppstår mens appen står åpen, vises først ved neste
  oppstart om ikke pushen når fram. Push dekker altså det vinduet, og er ikke
  bare «rask levering» der.

  Klienten kvitterer **etter visning**, én melding om gangen, og gjør ingenting
  hvis kvitteringen feiler — meldingen kommer igjen. Vår `delivered_at`-modell
  (markér, ikke slett) er det som gjør den toleransen mulig.
- **Push-budsjettet tåler ett tregt kall**, og begrenser ikke svartiden det
  finnes for å begrense. Målte tall i §4. Issue #3. Nå som køen leses, er et
  avbrutt budsjett en **utsettelse** til neste appstart, ikke tap — men for de
  §11-meldingene som har en 7-dagers frist, er en utsettelse på ubestemt tid
  fortsatt ikke gratis.
- **`/v1/feedback`-kvoten er per maskin.** `ratelimit.py` teller i minnet, og
  Fly kjører to maskiner, så den reelle grensen er 10/time, ikke 5. ÅP-B9.
- **`GET /v1/teams/near` sorterer i Python.** Holder på dagens datamengde. ÅP-B6.
- **Fristene i §11 avgjøres lat**, første gang noen spør — ikke på selve
  fristen. ÅP-B7.
- **Feilanalyse-bilder ligger i databasen**, ikke i R2 som §6 foreskriver.
  Kolonnen heter `image_legacy` nettopp fordi det er midlertidig. ÅP-B5.
