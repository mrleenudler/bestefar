# android/ — arbeidsinstruks

**Les rot-`CLAUDE.md` først**, særlig §7 (feller og lærdommer). Eierskapsregelen,
issue-flyten, PowerShell-syntaks, ISO-datoer og de delte filene gjentas ikke her.

Denne fila dekker bare det som er særegent for UI-området: `android/`, `UI/`,
`dist/`.

## Bygg

```powershell
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
cd android
.\gradlew assembleDebug
.\gradlew assembleRelease
```

`JAVA_HOME` **må** peke på Android Studios JBR. Uten den plukker Gradle en JDK
som ikke matcher AGP-en, og feilmeldingen sier ikke det.

`opencvDir` i `gradle.properties` må peke på et utpakket OpenCV Android SDK
(`…/sdk/native/jni`) — bruk skråstreker, backslash er escape-tegn i
`.properties`. NDK-en bygger kjernen fra `core/` via
`app/src/main/cpp/CMakeLists.txt`, så et brudd i `core/src` stopper
`assembleDebug` selv om ingenting i Kotlin er rørt.

**Release-signering** leses fra `android/keystore.properties` +
`android/bestefar-release.keystore`. Begge er **gitignorert** og skal aldri
committes. Mangler de, bygges release usignert i stedet for å feile.
Mistes keystoren, kan appen ikke oppdateres på enheter som allerede har den —
ta backup utenfor repoet.

**APK-er sendes ut fra `dist\`**, som også er gitignorert. `assembleRelease`
skriver til `app\build\outputs\apk\release\`; kopiering til `dist\` er et
manuelt steg, og `versionCode`/`versionName` i `app/build.gradle.kts` bumpes i
samme runde. Kopier med **absolutte stier** — står du i `android/`, blir en
relativ sti til `android\app\build\…` til `android\android\app\build\…`.

**CI bygger ikke klienten.** `android`-jobben i `.github/workflows/ci.yml` står
med `if: false` — se ÅP-U12. En grønn CI-kjøring sier ingenting om at Kotlin
kompilerer; det gjør bare `gradlew assembleDebug` lokalt.

### Byggoppsettet har tre feller som ikke feiler tydelig

- **AGP 9 har innebygd Kotlin.** `org.jetbrains.kotlin.android` skal *ikke* stå
  i `plugins {}`, og `kotlinOptions {}` er erstattet av
  `kotlin.compilerOptions {}`.
- **`buildConfig = true` må stå eksplisitt.** `BuildConfig` er av som standard
  fra AGP 8, og vi trenger den til `API_BASE_URL` og `GOOGLE_WEB_CLIENT_ID`.
- **`google-services.json` må ligge i `app/`.** Pluginen feiler byggingen om den
  mangler, og det er riktig: et push-bygg uten prosjektkonfigurasjon ville vært
  stille ødelagt. Web-klient-ID-en (`client_type: 3`) leses ut av *den* fila ved
  bygg — ikke dupliser den inn i `gradle.properties`, to kopier kommer i utakt
  og symptomet er et gyldig token backenden avviser.

## Hvor tingene står

| Fil | Hva |
|---|---|
| `../bestefar_UI_spec.md` | Kravene, slik de ble skrevet som v0.4. **Ikke omskrevet siden** — les den som hva appen skulle bli. Du eier hele den. |
| `android/CHANGELOG.md` | Hva hver runde faktisk endret, v0.6→. Nye runder føres inn HER. |
| `android/ARCHITECTURE.md` | Beslutningsloggen: nettverkslag, Keystore/økt, de tre nøkkellagene, soft-delete, broen til kjernen |
| `android/KONTRAKT.md` | Det du garanterer utad — `retryable`, sidecar-formatet, blobens ytre format, gravsteiner, øktgarantiene |
| `android/README.md` | Oppsett for en ny maskin + hvilke filer som gjør hva |
| `../docs/flytskjema.md` §2–§3 | Skjermflyten og varselflyten som mermaid, avledet fra kode |
| `../contracts/openapi.json` | **Maskinlesbar kontrakt for FORESPØRSLER**, generert fra backend-koden med CI-sjekk mot drift. Bygg mot den, ikke mot speccen. Les `contracts/README.md` først — den lister fem ting fila ikke dekker. |
| `../backend_spec.md` §12–§17 | Klientsiden av kontrakten. Backend eier §0–§11; **rediger kun §12–§17**, og les regionen på nytt rett før du skriver. |
| `../AAPNE_PUNKTER.md` | Det som ikke kan besluttes i kode. UI-punktene er ÅP-U* |
| `../til_utvikler_v##.md` | Tilbakemelding per runde. **Delt fil** — legg til nederst, aldri `Write` |

Det finnes **ingen** `android/BESLUTNINGER.md`. `android/ARCHITECTURE.md` er
beslutningsloggen; en parallell fil ville bare drive fra den.

## Invarianter — omgjøres ikke uten at det står i ARCHITECTURE.md hvorfor

1. **Offline-først.** Alt appen kan gjøre uten konto, virker uten konto og uten
   nett. Et mislykket kall er normaltilstanden, ikke en feil å vise fram. Ingen
   funksjon får kreve innlogging som ikke *er* om noen andre.
2. **`retryable`-klassifiseringen er en avtale, ikke en detalj.** `code == 0`,
   408, 429, ≥ 500 prøves på nytt; alt annet kastes. Endrer du den, endrer du
   backendens forpliktelser — `KONTRAKT.md` §1, og meld det.
3. **Aldri to `/v1/auth/refresh` parallelt.** `Auth.refresh` er `@Synchronized`,
   og det er sikkerhetskritisk: serveren leser gjenbruk av et rotert
   refresh-token som en lekkasje og tilbakekaller alle brukerens økter.
4. **Utlogging sletter tokenene lokalt uansett svar.** Access-tokenet er en
   statsløs JWT som ikke kan tilbakekalles — `/v1/auth/logout` gjør den ikke
   ugyldig. Uten den lokale slettingen har en utlogget bruker full tilgang i
   opptil en time.
5. **Hemmeligheter i `Secrets` (Keystore), aldri i `bestefar_ui`.**
   `Store.exportPrefs()` er generisk over hele den prefs-fila, så alt som legges
   der havner i sikkerhetskopien. En kopi som inneholder sin egen nøkkel er en
   sirkel. `androidx.security:security-crypto` er avviklet — Keystore direkte.
6. **Sletting er soft-delete.** `deletedAt` settes, raden blir stående.
   `allSeries()`/`allHunts()` filtrerer for visning; `…Raw()` gir sannheten til
   synk og backup. Uten gravsteinen kan ikke backenden skille «har aldri
   eksistert» fra «brukeren slettet den».
7. **Serveren skal ikke kunne lese bloben.** Kryptering, serialisering og
   gjenoppretting er helt klientside. Deponering av nøkkelen
   (`/v1/backup/key-escrow`) er det ene unntaket, den er **av som standard**, og
   hjelpeteksten sier rett ut hva den betyr.
8. **Funksjonalitet som pauses, får ett navngitt flagg** som alle inngangene
   sjekker (`Dialogs.RESEARCH_ENABLED`). Ikke slett kode, og ikke la én inngang
   stå igjen ubeskyttet.
9. **`/v1/research` aktiveres ikke mot ekte brukere** før personvernerklæringen
   finnes og DPIA-spørsmålet er avklart (`backend_spec.md` §7/§9).
10. **Deler du et personvernvalg opp, slår du det ikke sammen igjen.** «Jeg
    skjøt stående på 85 meter» og «dyret ble skadeskutt og aldri funnet» er ikke
    samme opplysning å dele om seg selv.
11. **En teller eller knapp som gjelder backend, skal ha en sender.** Tre
    plassholdere løy i månedsvis: en kø ingen tømte, en teller som bare kunne
    vokse, og en knapp som viste kvittering uten å sende. Bygger du flaten,
    bygg ledningen — eller la være å vise flaten.

## Hva de andre eier

Du leser gjerne koden deres. Du redigerer den ikke — issue med label
`kjerne`/`backend` (rot-`CLAUDE.md` §2.1, form: issue #1 og #2).

| Eies av | Hva det betyr for deg |
|---|---|
| **kjerne** — `core/include/bestefar/bestefar_ffi.h` | CV-kontrakten. `app/src/main/cpp/jni_bridge.cpp` ligger hos deg, men er **forbruker**: endres headeren, eier kjernen endringen og du følger etter. Statuskoder, `BfResult`-feltene og `BF_MAX_HITS` står i `core/KONTRAKT.md` — ikke gjenta dem her. |
| **kjerne** — `bf_version()` | `core_version` i §6-donasjonene kommer derfra (`BestefarCore.version`), ikke fra appens `versionName`. De to følger ikke hverandre, og skal ikke gjøre det. |
| **backend** — `backend/KONTRAKT.md` §3 | Feltnavnene i multipart-skjemaet for `/v1/failed-analyses`. Sidecar-filen på disk er din; **navnene på ledningen er deres** — se `KONTRAKT.md` §2 her. |
| **backend** — `backend_spec.md` §0–§11 | Endepunkter, tokens, kvoter, grenser. Trenger du en rute eller et felt, meld issue — ikke bygg mot en rute du selv har «foreslått» uten å lese speccen på nytt først; den kan allerede finnes med et annet navn. |
| **backend** — Fly-secrets | Alt som må settes med `flyctl secrets set` settes av utvikler. Du ber om det, genererer det ikke, og skriver det aldri ut. |

## Arbeidsetikette i dette området

- **Meld aldri en runde ferdig uten at `assembleDebug` faktisk har kjørt grønt i
  denne økten.** Ingen automatikk fanger et Kotlin-brudd for deg (ÅP-U12).
- **Brukertekst har æøå, kodekommentarer og logg er ASCII-translitterert.**
  Grensen har lekket før — sjekk hvilken side du står på. Datoer i UI er norske,
  datoer i dokumenter er ISO.
- **Strenger hører i `strings.xml`**, også de du «bare» trenger én gang. Alt
  brukeren ser skal kunne leses samlet av noen som vurderer tonen.
- **Sier brukeren at noe «ikke ble implementert» og koden finnes:** sjekk om de
  kjørte en eldre APK fra `dist\`, eller om feltet bare vises betinget.
- **Ny bryter i Avanserte innstillinger: skriv defaultverdien i KDoc-en**, og
  velg den bevisst. Personvernbrytere er av som standard — men «privat som
  standard» uten en bryter er «aldri», og det er en annen feil.
- **Skal du kalle et endepunkt: slå det opp i `contracts/openapi.json`.**
  Rutenavn, verb, feltnavn, typer, hva som er valgfritt og hvilke lengdegrenser
  som gjelder står der, generert fra koden. **Grensene skal håndheves i
  klienten** — et felt som er for langt gir 422, og 422 er ikke `retryable`, så
  det ser ut som en helt annen feil enn den er. Spriker skjemaet og klienten,
  er det en feil: rett klienten, eller meld issue med label `backend`. Ikke
  tilpass deg stille.
- **Ingen «Nytt i v0.NN»-seksjoner i spec eller flytskjema.** Det mønsteret gikk
  fire stadier og gjorde begge filene uleselige forfra; de ble ryddet
  2026-08-08. Runden føres i `android/CHANGELOG.md`, og
  `docs/flytskjema.md` **endres** til å vise den nye nåtilstanden. Et dokument
  som beskriver appen, og et som forteller hva som skjedde — aldri i samme fil.
