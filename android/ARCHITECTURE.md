# Arkitektur — Android-klienten (`android/`)

**Underlag:** `bestefar_UI_spec-v0-4.md` (UI/UX) og `backend_spec.md` (kontrakt).
Skjermflyten står i `docs/flytskjema.md`; kontrakten mot serveren i
`android/KONTRAKT.md`; CV-kjernen i `core/ARCHITECTURE.md`.

Dette er **beslutningsloggen for klienten** — hvorfor lagene ser ut som de gjør,
og hva som ble vraket underveis. Flyttet ordrett ut av `docs/ARCHITECTURE.md`
2026-08-07 da dokumentet ble splittet etter arbeidsområde; se
`docs/ARCHITECTURE.md` for kartet over splitten.

## Klientens nettverkslag (v0.14–v0.15)

Appen er **offline-først**: alt virker uten nett, og et mislykket kall er
normaltilstanden, ikke en feil.

- `Api.kt` — `HttpURLConnection`, ingen nytt bibliotek. Én enkelt-tråds kø, så
  opplastinger går i rekkefølge og ikke parallelt mot en gratis-tier.
  Basis-URL fra `BuildConfig.API_BASE_URL`, overstyrbar i DevTools.
  `Authorization: Bearer` settes fra `Auth.accessToken(ctx)` når den er satt
  (fram til v0.16 fra `Store.authToken`, som nå bare finnes som
  `legacyAuthToken` for engangsmigreringen).
  **Feilklassifiseringen (`retryable`) eies av `KONTRAKT.md` §1** — den er en
  avtale med serveren, ikke en intern detalj.
- `Sync.kt` — filbasert kø i `filesDir/dev_uploads` mot `/v1/failed-analyses`.
  Filbasert med vilje: overlever appdrap, omstart og flymodus uten database.
  Formatet på køfilene står i `KONTRAKT.md` §2.
- `Backup.kt` — klient-kryptert sikkerhetskopi (backend_spec §2). Serveren
  lagrer bytes den ikke kan lese, så serialisering, kryptering og
  gjenoppretting er **helt** klientside og testbart uten server. Blobens ytre
  format — det serveren ser — står i `KONTRAKT.md` §3. Klartekst er JSON:
  `{v, app, ts, prefs, series[], hunts[]}`.

  **Nøkkelavledningen er valget som betyr noe her**, og den er ikke en del av
  kontrakten: PBKDF2-HMAC-SHA256, 210 000 runder, over en generert
  gjenopprettingskode på 20 tegn (Crockford-base32 minus I/L/O/U ⇒ 100 bit).
  Bevisst **ikke** et brukervalgt passord: angriperen har hele bloben og kan
  gjette offline, så ingen server kan bremse ham — og et passord folk velger
  selv, har langt under 100 bit. Prisen er at koden må oppbevares, og
  konsekvensen står i UI-et: mister brukeren koden, er kopien tapt.

  Selve problemet er nøkkelen, ikke kryptoen. En nøkkel som bare finnes på
  telefonen er verdiløs i akkurat det scenarioet kopien finnes for — telefonen
  er borte. Derfor de tre lagene i `BackupKeys.kt` under.

## Økt og hemmeligheter (v0.16)

- `Secrets.kt` — AES-256-GCM-nøkkel i **Android Keystore**, chiffertekst i en
  **egen** prefs-fil (`bestefar_secrets`). Egen fil er ikke kosmetikk:
  `Store.exportPrefs()` er generisk over hele `bestefar_ui`, så alt som legges
  der havner i sikkerhetskopien. Tokener og gjenopprettingskoden hører ikke
  hjemme i en kopi som skal flyttes til en annen telefon.
  `androidx.security:security-crypto` ble vraket — den er avviklet av Google, og
  Keystore er grunnmuren den sto på uansett.
- `Auth.kt` — access-token (statsløs JWT, 60 min, kan ikke tilbakekalles) +
  refresh-token (roteres ved hver bruk). `refresh()` er `@Synchronized` fordi to
  parallelle fornyelser ville sett ut som en tokenlekkasje for serveren og
  tilbakekalt alle brukerens økter. `logout()` avregistrerer enheten for push
  **først** (mens tokenet virker), tilbakekaller så refresh-tokenet, og sletter
  begge lokalt uansett utfall. `Api` prøver på nytt etter 401 nøyaktig én gang.
- `BackupKeys.kt` — nøkkelen til kopien i tre lag: lokalt → **Block Store**
  (kun når `isEndToEndEncryptionAvailable()`) → frivillig deponering hos
  serveren (`/v1/backup/key-escrow`, av som standard). Koden er nødutgangen.
  Det som deponeres er **gjenopprettingskoden som ASCII, base64-kodet** — men
  serveren skal ikke vite det, og gjør det ikke: for den er `key_material`
  ugjennomsiktige byte (`KONTRAKT.md` §4). Sender vi en avledet nøkkel i stedet
  senere, er det en klientendring alene.
- `Lock.kt` — `BiometricPrompt` foran jaktloggen, av som standard, fem minutters
  frist. En dør, ikke kryptering; loggen ligger like lesbar på disk.
- `Login.kt` (v0.17) — veien *inn* i en økt. **Credential Manager**
  (`GetSignInWithGoogleOption`), ikke den utfasede `GoogleSignInClient`. Den
  eksplisitte knappeflyten er valgt framfor den filtrerte bunnarken, fordi
  bunnarken feiler for en bruker som aldri har logget inn før — nettopp den
  brukeren en «logg inn»-knapp finnes for. Callback-API-et brukes så
  innloggingen ikke drar inn coroutines i en kodebase uten dem.
  `BuildConfig.GOOGLE_WEB_CLIENT_ID` leses ved bygg ut av
  `google-services.json` (`client_type: 3`) — to kopier av samme ID kommer i
  utakt, og symptomet er et gyldig token backenden avviser.
- `Messages.kt` (v0.19) — meldingskøen, den *andre* halvdelen av §11.
  Arbeidsdelingen er backendens: køen er garantien, pushen er rask levering. Til
  v0.18 leste klienten bare pushen, så garantien fantes ikke — en bruker uten
  varseltillatelse fikk ingenting. Køen hentes ved appstart og vises **etter**
  oppstartsvinduene, ikke oppå dem; derfor `onStartupOverlaysDone` i
  `MainActivity`, som hver utgang av oppstartskjeden må kalle.
  **Kvitteringen sendes etter visning**, ikke ved henting — serveren markerer i
  stedet for å slette for å tåle en klient som krasjer imellom, og kvitterer vi
  for tidlig, kaster vi bort nettopp den toleransen. Prisen er at en melding kan
  vises to ganger. `kind` lagres som `String`, ikke som enum: backenden skal
  kunne legge til en meldingstype uten en klientutgivelse.
- **`Api.send` med `GET` var i praksis `POST` til og med v0.18.** `request()`
  åpnet alltid utstrømmen, og `HttpURLConnection` gjør en GET om til POST i det
  øyeblikket den åpnes — arvet JDK-oppførsel, ikke en Android-detalj. `GET
  /v1/backup/meta` og `GET /v1/backup/key-escrow` fikk derfor 405, og
  405-grenen i `AvansertActivity.confirmEscrow` behandlet symptomet som «ikke
  konfigurert». Fikset ved å bare åpne utstrømmen når det finnes en kropp å
  skrive. **Lærdommen er den generelle:** en HTTP-metode som settes ett sted og
  overstyres et annet, feiler stille — og en feilkode som blir «håndtert» før
  den er forstått, sementerer feilen.
- `Push.kt` / `PushService.kt` (v0.18) — FCM. `register()` kalles ved hver
  oppstart (idempotent `PUT /v1/devices`) og er en no-op uten konto.
  **Merk asymmetrien:** backenden sender en `notification`-blokk, så Android
  tegner varselet selv når appen er i bakgrunnen — `onMessageReceived` kjøres
  kun i forgrunnen. Bakgrunnstilfellet styres derfor av
  `default_notification_*`-meta-data i manifestet, forgrunnstilfellet av kode.
  Begge må stemme, ellers forsvinner enten halvparten av varslene eller
  utseendet deres. Kanalen opprettes i `BestefarApp` fordi den må finnes før
  det første varselet, også når appen ikke har kjørt kode.

**Soft-delete** (v0.15): `SeriesRecord.deletedAt` / `HuntRecord.deletedAt`.
Sletting setter tidsstempel i stedet for å fjerne raden — uten gravsteinen har
klienten ingenting å fortelle backenden, og en gjenoppretting ville lagt inn
igjen det brukeren slettet. `allSeries()`/`allHunts()` filtrerer dem bort, så
visningskoden er uendret; `allSeriesRaw()`/`allHuntsRaw()` gir hele sannheten
til synk og sikkerhetskopi. Hva som må sendes over ledningen: `KONTRAKT.md` §5.

## Broen til CV-kjernen

`app/src/main/cpp/jni_bridge.cpp` ligger i dette området, men er **forbruker** av
kjernens C-API (`core/include/bestefar/bestefar_ffi.h`). Endres headeren, eier
kjernen endringen og klienten følger etter — meld issue med label `kjerne`, ikke
rediger `core/` (rot-`CLAUDE.md` §2.1).

Kotlin-datastrukturene i `BestefarCore.kt` speiler C-structene. Kontrakten selv —
statuskoder, `BfResult`-feltene, `BF_MAX_HITS` — er dokumentert i
`core/ARCHITECTURE.md`, «CV-kontrakt».
