# Klientens kontrakt mot backend

**Eier: UI-instansen.** Klienten håndhever disse reglene, derfor eier den
teksten. `backend_spec.md` og `android/ARCHITECTURE.md` skal **peke hit**, ikke
gjenta — en regel som står tre steder, blir før eller siden tre ulike regler.

**Hva som hører hjemme her:** det andre kan stole på. Ytre formater, statuskoder,
hva som må sendes. **Ikke** hvorfor klienten valgte som den gjorde — det er
beslutninger, og de bor i `android/ARCHITECTURE.md`. Serveren lagrer bytes den
ikke kan lese, og bryr seg ikke om hvordan nøkkelen ble til.

Opprettet 2026-08-07 som del av dokumentsplitten (se `docs/ARCHITECTURE.md`),
med tekst flyttet fra `backend_spec.md` §12/§13 og det gamle
`docs/ARCHITECTURE.md`.

Alt under er **verifisert mot koden 2026-08-07** — `Api.kt`, `Sync.kt`,
`Backup.kt`, `BackupKeys.kt`, `Auth.kt`, `Push.kt`, `Announce.kt` — og mot
mottakssiden i `backend/app/routers/`. Filreferansene er der du kan etterprøve
det. Det som ikke holder, står i §9; det er ikke en liste over ting vi skammer
oss over, men den delen av kontrakten ingen skal bygge på.

---

## 1. Feilklassifisering — hva køen prøver på nytt

`retryable` = `code == 0` (nådde aldri fram), 408, 429, ≥ 500. Alt annet
(400/413/422) er permanent — køen kaster elementet i stedet for å prøve i evig
tid. Serveren bør derfor svare 4xx på data den aldri vil kunne ta imot, og
5xx/429 på alt som kan gå bra senere.

Håndheves i `Api.kt`. Konsekvensen for backend er at et *midlertidig* problem
som svares med 400 fører til stilltiende datatap hos brukeren, og et *permanent*
problem som svares med 500 gir en kø som aldri tømmes.

## 2. Sidecar-format v2 — feilanalyse-innsending

Køen på disk er ett par `filesDir/dev_uploads/{seriesId}_{tag}.jpg|json` per
innsending. JSON-en er **klientens eget filformat**, ikke skjemaet som går over
ledningen:

```json
{"v":2,"series_id":"<uuid>","tag":"ocr_match|ocr_mismatch|rejected",
 "status_code":0,"confidence":0.83,"core_version":"0.14",
 "detected":[10.4,9.8],"ocr":[10.4,9.9],"capture_trigger":"auto|timeout"}
```

**Seks felter går videre under samme navn, to skifter navn, og to sendes
ikke.** `Sync.kt` mot `POST /v1/failed-analyses`:

| Sidecar-JSON | Multipart-felt | Merknad |
|---|---|---|
| `status_code` | `status_code` | påkrevd av serveren |
| `confidence` | `confidence` | påkrevd av serveren |
| `core_version` | `core_version` | påkrevd av serveren |
| `tag` | `tag` | serveren har default `rejected` |
| `series_id` | `series_id` | valgfri |
| `detected` | **`detected_scores`** | JSON-array serialisert som streng |
| `ocr` | **`ocr_scores`** | mangler `ocr` i fila, sendes `[]` |
| `v` | *— sendes ikke* | se under |
| `capture_trigger` | `capture_trigger` | **utelates når køfila ikke har den** — se under |
| *(bildefila)* | `image` | `multipart/form-data`, `image/jpeg`, påkrevd |

Feltnavnene i skjemaet eies av **`backend/KONTRAKT.md` §3**; tabellen her sier
bare hva klienten legger i dem.

**`v` er klientintern og krysser ikke grensa.** Den finnes fordi køen kan
inneholde filer skrevet av en eldre app — formatet var v1 fram til v0.14, med
bare `detected` + `tag` — men den leses ikke av noen, heller ikke av senderen.
Toleransen for gamle filer ligger i at hvert felt hentes med en defaultverdi
(`confidence` → `-1.0`, `core_version` → `"ukjent"`), ikke i et versjonsvalg.
Skal serveren noen gang få vite hvilket format fila hadde, er det en **ny**
avtale mellom begge parter, ikke et felt som «egentlig skulle vært med».

**`capture_trigger` krysser grensa fra v0.35** (issue #11 lukket 2026-08-22).
Feltet skiller et bilde gatingen slapp gjennom (`auto`) fra ett tidsgrensen
tvang fram (`timeout`, v0.29 — `CaptureActivity.CAPTURE_TIMEOUT_MS`). Det ble
skrevet i køfila fra v0.29, men først sendt nå.

**Det utelates når køfila ikke har det, og det er ikke en detalj.** Serveren
lagrer et utelatt felt som NULL, og **NULL betyr «ikke oppgitt» — ikke `auto`**
(`backend/KONTRAKT.md` §3). Køen kan inneholde filer skrevet av v0.29–v0.34,
der klienten *hadde* timeout-capture men ikke sendte feltet. De radene skal
forbli ubesvarte. Å fylle dem med `auto` ville vært å påstå noe vi ikke vet, i
nøyaktig det datasettet ÅP-K1 skal måles på.

**Hvorfor det ikke ble en ny tagg:** `tag` svarer på *hva donasjonen viser*,
`capture_trigger` på *hvordan bildet ble tatt*. De er ortogonale — en
timeout-capture kan ende som hvilken som helst av de tre taggene, og en enkelt
`timeout`-verdi ville overskrevet OCR-utfallet. Backend landet på et eget felt.

- **`tag` ∈ {`ocr_match`, `ocr_mismatch`, `rejected`}.**
- **`ocr_mismatch` dekker to ulike avvik, og lengdene skiller dem.** Fram til
  v0.26 hadde `detected` og `ocr` alltid like mange elementer, og taggen betydde
  bare at *verdiene* spriket. Fra v0.27 sendes også antalls-avvik, fordi de er
  det mest interessante materialet vi har:
  `len(detected) > len(ocr)` er **over-deteksjon** (kjernen så merker som ikke er
  skudd), `len(detected) < len(ocr)` er **skjulte treff** (to skudd i samme
  hull). Ingen av dem har et eget felt — retningen leses av lengdene.
- **`rejected` med `status_code = 0` er klientens avvisning, ikke kjernens.**
  Kjernen svarte OK; klienten forkastet serien fordi `detected` inneholdt flere
  treff enn en serie kan ha skudd (10) og OCR ikke ga noen fasit å korrigere mot.
  `status_code != 0` er som før kjernens egen avvisning, og da er `detected` tom.
  Skal du finne over-deteksjonsbilder å kalibrere mot, er det disse to du vil ha:
  `rejected` med `status_code = 0`, og `ocr_mismatch` med
  `len(detected) > len(ocr)`.
- **`detected` er alltid poengene CV-kjernen ga**, også når OCR har overskrevet
  visningen — ellers ville en `ocr_match`-donasjon ikke si noe om hva kjernen så.
- **`confidence = -1.0`** betyr *ukjent*, ikke lav konfidens.
- **`core_version`** er CV-kjernens egen versjon, hentet med `bf_version()` over
  JNI (`BestefarCore.version`). Fram til 2026-08-06 var den appens
  `versionName` — eldre innsendinger bærer den verdien.
- **Bildet er kameraets originalfil, fra v0.28.** Ikke en omkoding: `takePicture`
  gir rå JPEG-bytes, de skrives uendret til cache, og `Sync.queue` kopierer fila
  byte for byte. **Til og med v0.27 var det en andre generasjons JPEG** —
  `bmp.compress(JPEG, 92)` av de samme pikslene — så eldre donasjoner i R2 bærer
  ett ekstra sett komprimeringsartefakter. Skal de brukes som treningsdata
  sammen med nye, er det en forskjell som må vites om, ikke oppdages.
- **Bildet er i SENSORORIENTERING; kjernen analyserte det rotert.** Rotasjonen
  (0/90/180/270) krysser ikke ledningen i dag — den finnes bare i klientens
  intent. Det gjør ikke poengene tvetydige, for `detected` og `ocr` er radier og
  dermed rotasjonsuavhengige; det betyr bare at et donert bilde kan ligge på
  siden i forhold til det kjernen så. Trenger treningen den eksakte
  orienteringen, er det et nytt felt og en **ny avtale** med backend, ikke noe
  klienten kan legge til alene.
- **Bildegrensen er 8 MiB** (`max_upload_bytes`), ikke de 16 fra §3. Et 413 er
  ikke `retryable`, så et for stort bilde kastes ut av køen ved første forsøk.
  **Marginen er liten fra v0.28:** originalfilene er 6–7 MB der omkodingen var
  ~3 MB. Klienten sjekker grensa *før* den køer (`Sync.kt`), så utfallet er en
  `Log.w`-linje og ingen donasjon — ikke en 413. Et telefonkamera med større
  sensor kan dermed slutte å bidra uten at noen ser det. ÅP-U16.
- **Endepunktet krever ikke innlogging.** Donasjonen henger på
  bildedelings-samtykket, ikke på kontoen, og køen tømmes også for en bruker som
  aldri har logget inn.

**Bruker du feil navn på poengfeltene, merker du det ikke.** `detected_scores`
og `ocr_scores` har defaultverdi `""` hos serveren, så en sender som holder seg
til sidecar-navnene får **201 Created med tomme poenglister** — ikke en 422.
Bildet lagres, donasjonen er verdiløs, og ingen part ser en feil. Det er grunnen
til at tabellen over står her og ikke bare i koden.

## 3. Sikkerhetskopiens blob — bytene serveren ikke kan lese

**Ytre format** (`Backup.kt`):
`"BFBK" | 1 B versjon | 16 B salt | 12 B IV | AES-256-GCM (tag 128 bit)`.

Alt etter de fire magiske bytene og versjonsbyten er ugjennomsiktig. Ingenting av
innholdet kan valideres server-side. Hvordan nøkkelen utledes, er en
klientbeslutning som ikke angår serveren — `android/ARCHITECTURE.md`.

**Det ene serveren må vite om nøkkelen:** den er brukerens, og **serveren kan
ikke hjelpe** en bruker som har mistet den. Den ene ærlige måten å hjelpe på er
at brukeren uttrykkelig gir oss nøkkelen — se `backend_spec.md` §2.1. Ikke bygg
noen annen vei inn; en «gjenopprett kopien min» uten det samtykket finnes det
ingen implementasjon av som er sann.

**Opplasting** er `PUT /v1/backup` med bloben som rå `application/octet-stream`
og metadataene i query-strengen — ikke base64 i JSON, som ville lagt ~33 % på den
største nyttelasten appen har.

- **`client_ts` er ISO-8601 med `Z`** fra og med v0.20 — `2026-08-08T09:14:22.317Z`.
  Til og med v0.19 var det epoke-millisekunder som heltall; serveren tar imot
  begge, men skjemaet deklarerer `date-time`, og ms-varianten hvilte på at
  parseren tolker store heltall som millisekunder framfor sekunder. Verdien er
  tidspunktet **snapshotet ble tatt**, ikke da opplastingen var ferdig.
  Serveren sammenligner den bare mot sin egen lagrede verdi, aldri mot sin egen
  klokke, så klokkeskjev mellom telefoner er den eneste kilden til feil her.
- **`schema_version`** er `Backup.SNAPSHOT_VERSION`, altså formatversjonen på
  *innholdet* i bloben. Sendes fra og med v0.20; eldre klienter sendte den ikke,
  så alt som ligger lagret fra før har serverens default `1` — som tilfeldigvis
  er riktig, siden formatet ikke har endret seg.
- **`?force=true` settes kun når brukeren har svart ja** på «overskriv den nyere
  kopien». 409 vises som en egen dialog, ikke som en feil.
- **`app_version` sendes ikke.** Til og med v0.19 gjorde klienten det, men
  serveren tar ikke imot en slik parameter og ukjente query-parametere
  forsvinner stille. Appversjonen ligger i bloben (`app`), der den kan leses av
  den som har nøkkelen.
- **`device_id` sendes ikke**, så `/meta` svarer alltid `device_id: ""`. Se §9.

**Grense:** 16 MiB (`max_backup_bytes`). Serveren sjekker `Content-Length`
*før* den leser kroppen, så et for stort opplastingsforsøk avvises uten at
bytene går over ledningen.

## 4. Nøkkeldeponering — hva `key_material` inneholder

`key_material` er base64 av ugjennomsiktige byte (≤ 512 byte). Serveren bryr seg
ikke om det er en nøkkel eller en gjenopprettingskode, og skal ikke tolke dem.

**503 betyr «ikke slått på på serveren»**, og vises som det — ikke som en feil
brukeren har gjort. Bryteren går tilbake til av hvis `PUT` ikke svarer 2xx, så
klienten aldri står og påstår at nøkkelen er deponert når den ikke er det.

**Veien ut kan ikke feile.** Slår brukeren bryteren av, settes valget lokalt av
med det samme, og `DELETE` sendes etterpå uten at utfallet leses. Et
personvernvalg man må ha dekning for å komme ut av, er ikke et valg — men
konsekvensen er at server og klient kan komme i utakt, se §9.

**`escrowed` i `/meta` er fasit, ikke den lokale bryteren** (fra v0.22). Bryteren
ligger i app-preferansene og forsvinner ved reinstallasjon; serverens felt
overlever telefonen. Før v0.22 hoppet klienten over deponeringen når den lokale
bryteren sto av — altså i nøyaktig det scenarioet deponeringen finnes for, en ny
telefon. Nå styrer `escrowed` om vi henter materialet, og et vellykket oppslag
setter den lokale bryteren tilbake på.

## 5. Gravsteiner sendes, de utelates ikke

En slettet serie eller jaktpost forsvinner ikke fra det som går over ledningen —
den sendes som gravstein. Uten den finnes ingen forskjell på «har aldri
eksistert» og «brukeren slettet den», og last-write-wins per post-ID kan ikke
håndheves ved gjenoppretting.

I bloben ligger `series`/`hunts` derfor **rå**, altså inkludert de slettede.
**Når §5-synken kommer, gjelder det samme der: slettede poster sendes som
gravstein, ikke utelates.**

Hvordan gravsteinene er representert internt, og hvordan visningskoden skjermes
fra dem, står i `android/ARCHITECTURE.md`.

## 6. Øktoppførsel serveren kan regne med

Serverens tyverideteksjon (`backend_spec.md` §1, §14) hviler på disse. Uten dem
ser normal bruk ut som et tokentyveri, og brukeren logges ut overalt.

- **Aldri to `/v1/auth/refresh` parallelt med samme token.** Fornyelse er
  serialisert bak én lås, og en tråd som ventet sjekker om tokenet allerede er
  fornyet før den prøver selv.
- **Én omprøving etter 401, ikke flere.** Auth-kallene selv — og
  avregistreringen i utloggingssekvensen — sendes uten omprøving, så et 401 fra
  `/refresh` ikke utløser en ny fornyelse i ring. Lykkes fornyelsen og svaret
  likevel er 401, gir klienten opp: da er det ikke tokenet som er problemet.
- **Fornyelse skjer reaktivt, ikke i forkant.** Klienten kaller `/refresh`
  først når et kall har fått 401 — den har en `needsRefresh`-vurdering, men
  ingen bakgrunnsjobb som fornyer på forhånd. **Et jevnt innslag av 401 er
  altså normal trafikk fra denne klienten**, ikke et angrepsmønster: hver
  telefon som har ligget stille i over en time gir ett før den fornyer.
- **Utlogging skjer i fast rekkefølge:** `POST /v1/devices/unregister` først —
  mens access-tokenet fortsatt virker — så `POST /v1/auth/logout`, og deretter
  slettes begge tokenene lokalt uansett utfall, også offline. En utlogging som
  avbrytes av dårlig dekning skal ikke etterlate en telefon som fortsetter å få
  varsler.

Hvorfor det er løst slik, og hvor tokenene ligger: `android/ARCHITECTURE.md`.

## 7. Levering av §11-varsler: køen og pushen

Backenden legger varselet i køen **før** pushen sendes, og køen er garantien —
pushen er bare rask levering. Fra og med v0.19 leser klienten begge, så den
arbeidsdelingen holder i praksis og ikke bare i teksten.

### 7.1 Meldingskøen — det som faktisk garanterer levering

**`GET /v1/messages` hentes ved hver appstart**, i `MainActivity.onCreate`,
parallelt med oppstartsvinduene. Uten konto sendes kallet ikke i det hele tatt.
Et mislykket kall — offline, 401, ubrukelig svar — behandles som «ingen
meldinger»: køen blir stående på serveren og hentes ved neste oppstart. En
oppstartsjobb i en offline-først app skal aldri vise en feil og aldri hindre at
appen starter.

**`POST /v1/messages/ack` sendes når meldingen er VIST**, aldri ved henting.
Serveren markerer raden med `delivered_at` i stedet for å slette den nettopp for
å tåle en klient som forsvinner imellom, og den toleransen er verdiløs hvis
klienten kvitterer for tidlig. Klienten kvitterer per melding, etter hvert som
brukeren klikker seg gjennom dem. Feiler kvitteringen, gjøres ingenting: **en
melding kan altså vises to ganger**, og det er den billige feilen i dette valget.

Skjemaet er lest ut av `backend/app/routers/messages.py:24–54` og
`backend/app/models/social.py:146–163` — `backend/KONTRAKT.md` har ingen seksjon
for køen (issue #5). To ting bryter med mønsteret ellers og er verdt å gjenta:
**`id` er et heltall, mens `team_id` er en streng-UUID eller `null`**, og
**`kind` er en fri streng på 32 tegn, ikke et enum**. Klienten lagrer `kind` som
streng nettopp fordi backenden skal kunne legge til en åttende meldingstype uten
en klientutgivelse.

Meldingene vises som fullskjermsvinduer, én om gangen, i serverens rekkefølge
(eldste først), etter at oppstartsvinduene er unnagjort. Tittel og brødtekst
vises **ordrett slik serveren sendte dem** — klienten reparerer dem ikke.

### 7.2 Enheten, og hva en push må inneholde for å bli sett

**Registrering skjer ved hver oppstart**, ikke én gang. `PUT /v1/devices` kalles
fra `BestefarApp.onCreate` hver gang appen starter med en konto innlogget, og på
nytt hver gang Firebase roterer tokenet. Klienten fører ikke noe «allerede
registrert»-flagg — et slikt flagg kan bli feil uten at noen merker det, og
`PUT` er idempotent nettopp for dette. Serveren skal altså regne med gjentatte,
uendrede innmeldinger fra den samme telefonen.

Kroppen er `{push_token, platform: "android", app_version, model}`, der
`app_version` er **appens** `versionName` (ikke `core_version` fra §2) og
`model` er `"<MANUFACTURER> <MODEL>"` avkortet til 64 tegn. Uten innlogging
hentes FCM-tokenet ikke i det hele tatt — et varsel er alltid til noen.

**Meldingen må ha en `notification`-blokk.** Ligger appen i bakgrunnen, er det
Android selv som tegner varselet ut fra den blokken; klientens
`onMessageReceived` kjøres bare i forgrunnen. En melding som *bare* har `data`,
er derfor **usynlig for brukeren** i det vanligste tilfellet, uten at noen part
ser en feil. `data`-felter (`kind`, `team_id`) leses av klienten, men styrer
ennå ingen ruting — alle varsler åpner forsiden.

## 8. Kunngjøring av felling — bøyningen ligger i `species`

`POST /v1/hunts/announce` med `{species, kommune?}`. **`species` er ubestemt form
med artikkel** — «en elg», «et rådyr» — fordi serveren limer sammen
«{navn} har felt {art} i {sted}.» og norsk artikkelvalg ikke kan avledes av en
enum. Klienten eier bøyningen, og viser brukeren nøyaktig den setningen vennene
får, før den sendes.

`kommune` er **fritekst brukeren kan rette eller tømme i dialogen**, ikke
profilens hjemkommune. Er feltet tomt, sendes det ikke.

Serveren lagrer ingenting om hva som ble felt eller hvor; jaktloggen bor i den
klient-krypterte bloben og skal fortsette å gjøre det.

## 9. Kjente unøyaktigheter

Ærlighet om hva kontrakten *ikke* holder:

- **Køen hentes bare ved appstart, ikke ved `onResume`.** En melding som kommer
  mens appen ligger åpen i bakgrunnen, vises først neste gang appen startes helt
  — med mindre pushen når fram. Backend kan regne med at meldingen leveres, men
  ikke med hvor raskt.
- **Ingen ruting på `kind`.** Både `data["kind"]` i pushen og `kind` i køen leses
  og lagres, men alle meldinger ender på forsiden, og en melding som ber om en
  handling («Bekreft i appen», «Avstemningen er åpen i 7 dager») viser bare
  teksten. Lag- og vennesidene er fortsatt lokale skjeletter uten
  server-kobling, så en dyplenke ville landet på en skjerm som ikke kjenner
  `team_id`. Det er en bevisst utsettelse, ikke en glipp — men **§11-flytene som
  krever et svar fra brukeren, er ikke fullførbare i klienten ennå.**
- **`device_id` på backupen er alltid tom.** Serveren tar imot parameteren og
  eksponerer den i `/meta` og i `X-Backup-Device-Id`, men klienten har ingen
  stabil installasjons-ID å sette der og sender den ikke. «Hvilken telefon
  lastet opp dette?» er derfor ubesvart — og det er nøyaktig spørsmålet man
  stiller den dagen to enheter har overskrevet hverandre. ÅP-U13.
  (`app_version` og `schema_version` var samme klasse feil og er rettet i
  v0.20.)
- **Deponeringen kan bli stående etter at brukeren skrudde den av.** `DELETE`
  sendes uten at svaret leses (§4), så en avslåing gjort offline etterlater
  nøkkelmaterialet på serveren mens klienten viser bryteren som av. Klienten
  prøver ikke på nytt. Vil man vite hva serveren faktisk har, er `escrowed` i
  `GET /v1/backup/meta` fasit — ikke bryteren.
- **§5 gjelder bare bloben.** Gravsteiner sendes i sikkerhetskopien, som er det
  eneste som faktisk går over ledningen i dag. Serie-synken (`/v1/stats`) er
  ikke bygget, så setningen om at «det samme gjelder der» er en intensjon, ikke
  en oppførsel noen kan observere. Hvem som eier sannheten når den kommer, er
  fortsatt åpent — ÅP-B4.
- **`retryable` beskytter ikke mot en tom donasjon.** Klassifiseringen i §1
  skiller «prøv igjen» fra «gi opp», men den ser bare statuskoden. Et 201 med
  tomme poenglister (§2) passerer som suksess, og køelementet slettes.
- **Klienter til og med v0.18 sendte GET som POST.** `Api.send` åpnet alltid
  utstrømmen, og `HttpURLConnection` gjør da en GET om til en POST. `GET
  /v1/backup/meta` og `GET /v1/backup/key-escrow` traff derfor serveren som POST
  og fikk 405. Rettet i v0.19, men **APK-er i felt gjør det fortsatt**, så 405 på
  de to rutene er en gammel klient, ikke et angrep eller en rutefeil.
  `PUT`/`DELETE` var ikke rammet — bare `GET`.
- **`GET /v1/backup/meta` ble tatt i bruk først i v0.19.** Kallet har eksistert
  siden v0.15 uten en eneste kaller, og var ødelagt hele tiden av punktet over.
  Fra v0.19 gater det gjenopprettingen: 404 stopper flyten før brukeren blir
  bedt om en kode, og `client_ts` vises i bekreftelsen. Serveren kan altså
  regne med at ruten treffes fra v0.19 og framover, og knapt nok før det.
- **Ingen push er verifisert mottatt.** Hele kjeden i §7 er bygget og bygger
  grønt, men den er ikke kjørt ende-til-ende på en enhet — `FCM_SERVICE_ACCOUNT_JSON`
  står ikke som Fly-secret ennå. Formen på meldingen er lest ut av
  `services/push.py`, ikke observert.
- **`Login`-flyten er ikke prøvd med Google i debug.** Debug-keystorens SHA-1 er
  ikke registrert i Firebase-prosjektet, så Google-innlogging feiler i
  debug-bygg. E-postkoden er uavhengig av signeringsnøkkel og virker begge veier.

---

## Hvor resten står

| Tema | Eier |
|---|---|
| Endepunktene selv, tokens, kvoter | `backend_spec.md` |
| Statuskoder, idempotens, grenser og feltnavn på serversiden | `backend/KONTRAKT.md` |
| Klientens interne arkitektur og begrunnelser | `android/ARCHITECTURE.md` |
| CV-kontrakten (`BfResult`, statuskoder, `BF_MAX_HITS`) | `core/KONTRAKT.md` |
| Skjermflyt | `docs/flytskjema.md` |
