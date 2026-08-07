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
 "detected":[10.4,9.8],"ocr":[10.4,9.9]}
```

**Seks felter går videre under samme navn, to skifter navn, og ett sendes
ikke.** `Sync.kt:132–142` mot `POST /v1/failed-analyses`:

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

- **`tag` ∈ {`ocr_match`, `ocr_mismatch`, `rejected`}.**
- **`detected` er alltid poengene CV-kjernen ga**, også når OCR har overskrevet
  visningen — ellers ville en `ocr_match`-donasjon ikke si noe om hva kjernen så.
- **`confidence = -1.0`** betyr *ukjent*, ikke lav konfidens.
- **`core_version`** er CV-kjernens egen versjon, hentet med `bf_version()` over
  JNI (`BestefarCore.version`). Fram til 2026-08-06 var den appens
  `versionName` — eldre innsendinger bærer den verdien.
- **Bildegrensen er 8 MiB** (`max_upload_bytes`), ikke de 16 fra §3. Et 413 er
  ikke `retryable`, så et for stort bilde kastes ut av køen ved første forsøk.
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

- **`client_ts` er epoke-millisekunder som heltall**, ikke ISO-tid. Det er
  klientens veggklokke i det øyeblikket opplastingen starter — altså etter at
  bloben er bygget, som tar et kvart sekund eller to i nøkkelutledningen.
  Serveren sammenligner den bare mot sin egen lagrede verdi, aldri mot sin egen
  klokke, så klokkeskjev mellom telefoner er den eneste kilden til feil her.
- **`?force=true` settes kun når brukeren har svart ja** på «overskriv den nyere
  kopien». 409 vises som en egen dialog, ikke som en feil.
- **`app_version`** sendes med, men **ingen leser den** — serveren tar ikke imot
  en slik parameter. Se §7.

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

## 7. Enheten, og hva en push må inneholde for å bli sett

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

- **Meldingskøen leses ikke.** `backend/KONTRAKT.md` og backendens invariant om
  at «køen er garantien, push er bekvemmeligheten» stemmer på serversiden, men
  klienten henter aldri `/v1/messages`. I dag er push derfor den **eneste**
  leveringsveien: et varsel som gikk tapt fordi telefonen var av, er tapt for
  brukeren også. Backend kan altså ikke regne med at et køet varsel når fram.
- **`app_version` på `PUT /v1/backup` treffer ingenting.** Serveren tar imot
  `client_ts`, `schema_version`, `device_id` og `force`; klienten sender
  `client_ts`, `app_version` og `force`. Ukjente query-parametere ignoreres
  stille, så verdien forsvinner. Motsatt vei: `schema_version` og `device_id`
  sendes aldri, så `GET /v1/backup/meta` svarer alltid `device_id: ""` og
  `schema_version: 1`, uansett hvilken telefon som lastet opp. Feltene finnes,
  men de er tomme løfter. ÅP-U13.
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
