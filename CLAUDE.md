# CLAUDE.md — Bestefar

Ett repo, tre arbeidsområder, **tre parallelle Claude Code-instanser**. Denne
filen er kontrakten mellom dem. Les den før du rører noe.

Prosjektet er en app for rifleskyting og jakt: on-device CV analyserer bilder av
en Kongsberg elektronisk skive, klienten viser skyteevne og jaktlogg, og en liten
backend dekker konto, sikkerhetskopi, venner/lag og forskningsdata.

---

## 1. Katalogkart og eierskap

```
Bestefar/
├── core/                    KJERNE  C++17 CV-kjerne (OpenCV), ren C-FFI
│   ├── include/bestefar/    KJERNE  offentlige headere + AutoCaptureParams
│   ├── src/                 KJERNE  porterte moduler (screen, rings, hits, …)
│   ├── cli/                 KJERNE  desktop-CLI for verifisering mot C-settet
│   └── ARCHITECTURE.md      KJERNE  beslutningslogg: port, CV-kontrakt, FFI
├── ios/                     KJERNE  Swift-skjelett mot samme C-header (stub)
├── *.py  (rot)              KJERNE  Python-referansen — fasit for porten
├── Testsett/, hits_truth.txt KJERNE oracle-data (10/10 PASS er kravet)
│
├── backend/                 BACKEND FastAPI + SQLAlchemy + Alembic
│   ├── app/                 BACKEND routers, models, services
│   ├── migrations/          BACKEND håndskrevne Alembic-revisjoner
│   ├── tests/               BACKEND pytest (SQLite + Postgres)
│   └── fly.toml, Dockerfile BACKEND drift mot Fly.io
│
├── android/                 UI      Kotlin-app, CameraX, JNI-bro
│   ├── app/src/main/cpp/    UI*     jni_bridge.cpp — se merknad under
│   ├── ARCHITECTURE.md      UI      beslutningslogg: nettverk, økt, nøkler
│   └── KONTRAKT.md          UI      wire-kontrakten mot backend (se §4)
├── UI/                      UI      ikoner, silhuetter, SVG-kilder
├── dist/                    UI      APK-er som sendes ut (versjonsbump ved ny)
│
├── contracts/               DELT    openapi.json — GENERERT, ikke rediger (se §4)
├── docs/                    DELT    kart over dokumentene + bygg/CI, flytskjema
├── *_spec.md                DELT    de tre spesifikasjonene (se §4)
├── til_utvikler_v##.md      DELT    tilbakemelding per runde (se §4)
├── musings*.txt             EIER    skrives av utvikler — ikke skriv til dem
├── AAPNE_PUNKTER.md         DELT    alt som ikke kan besluttes av en instans
└── .github/workflows/       DELT    én CI-jobb per område (se §4)
```

| Område | Eier disse | Instansen kalles |
|---|---|---|
| **kjerne** | `core/`, `ios/`, Python-referansen i rot, `Testsett/` | CV-kjerne-instansen |
| **backend** | `backend/`, `.github/workflows/deploy-backend.yml` | backend-instansen |
| **ui** | `android/`, `UI/`, `dist/` | UI-instansen |

**Grensetilfellet `android/app/src/main/cpp/jni_bridge.cpp`:** filen ligger i
UI-området, men den er *forbruker* av kjernens C-API. Endres `bestefar_ffi.h`,
er det kjernen som eier endringen og UI som må følge etter — meld det som issue
begge veier, ikke rett i den andres fil.

---

## 2. Grunnregler

### 2.1 Du redigerer ALDRI et annet områdes kode

Ingen unntak — heller ikke for «det er bare én linje», «det er åpenbart feil»
eller «jeg fikser det raskere selv». To instanser som skriver i samme fil gir en
delt git-indeks og tapt arbeid, og det har allerede skjedd i dette repoet.

Trenger du en endring hos naboen: **opprett et GitHub-issue** med riktig label.

```powershell
gh issue create --label backend --title "409 mangler Retry-After" --body "..."
gh issue create --label kjerne  --title "bf_version() bør bumpes" --body "..."
gh issue create --label ui      --title "les /v1/messages ved oppstart" --body "..."
```

Skriv issuet slik at mottakeren kan handle uten å gjette:

- **Hva du trenger**, ikke hvordan du ville løst det.
- **Hvorfor** — hvilken oppførsel hos deg som blokkeres.
- **Referanse**: fil og linje (`backend/app/routers/backup.py:88`) eller
  spec-paragraf (`backend_spec.md §2.1`).
- **Hva du gjør i mellomtiden** hvis du har en midlertidig vei rundt.

Er du selv mottaker: lukk issuet med commit-meldingen (`Fixes #12`), og skriv
hva som faktisk ble gjort i `til_utvikler_v##.md`.

**Lesing er alltid lov.** Du *skal* lese andres kode for å forstå kontrakten —
det er skriving som er forbudt.

### 2.2 Alle shell-kommandoer skrives i PowerShell-syntaks

Utviklermiljøet er Windows med PowerShell. Bash-syntaks feiler her.

```powershell
# Riktig
Get-ChildItem backend\app -Recurse -Filter *.py
$env:TEST_DATABASE_URL = "postgresql://..."
if (Test-Path .env) { Remove-Item .env }
```

- `&&` og `||` finnes ikke i Windows PowerShell 5.1 — bruk `;` eller `if ($?)`.
- Korte kommandoer. Lange script-blokker og subexpressions utløser
  bekreftelsesdialog hver gang.
- Heredoc-syntaks fra bash virker ikke; bruk `@'…'@` med `'@` i kolonne 0.

### 2.3 Alle datoer skrives ISO

`2026-08-07`. Ikke `7. august 2026`, ikke `07.08.2026`, ikke `8/7/26`.

Gjelder overalt: spec-filer, `til_utvikler_v##.md`, arkitekturdokumentene,
commit-meldinger, kodekommentarer, issue-tekster.

Unntaket er **tekst brukeren ser i appen** — der gjelder norsk datoformat, fordi
det er UI-språk og ikke dokumentasjon.

---

## 3. Språk og tegnsett

- **Dokumentasjon og brukertekst: æøå.** Alt brukeren ser skal ha ekte norske
  tegn.
- **Kodekommentarer og loggutskrifter: ASCII-translitterering** (`aa`, `oe`,
  `ae`). Dette er en kodebase-konvensjon, ikke et systemkrav — og den har lekket
  ut i brukertekst før. Sjekk hvilken side av grensen du står på.

---

## 4. Delte filer — hvem skriver hva

| Fil | Regel |
|---|---|
| `bestefar_CV-kjerne_spec.md` | Kjernen eier. Andre melder issue. |
| `backend_spec.md` | Backend eier §0–§11; UI eier §12–§17 (klientsiden). Rediger kun din del, og les regionen på nytt rett før du skriver — den andre instansen kan ha endret filen. |
| `bestefar_UI_spec.md` | UI eier. **Kravene slik de ble skrevet som v0.4, ikke omskrevet siden** — endringene bor i `android/CHANGELOG.md`. Het `bestefar_UI_spec-v0-4.md` til 2026-08-08. |
| `docs/flytskjema.md` | Beskriver appen slik den **er nå**, avledet fra kode. Endres appen, endres diagrammet — ingen «Nytt i v0.NN»-seksjoner. |
| `til_utvikler_v##.md` | **Delt per runde.** Legg til en seksjon nederst med områdenavn i overskriften. Overskriv aldri. Én fil per runde, høyeste nummer er gjeldende. |
| `AAPNE_PUNKTER.md` | Alle skriver. Legg til når du oppdager noe som ikke kan besluttes i kode; stryk aldri et punkt uten at det faktisk er avklart av eier. **Navnet er med `AA`, ikke `Å`** — PowerShell 5.1 mangler æøå når filnavn sendes videre til `git.exe`, så `git mv`/`git commit <sti>` feiler på den. Ikke «rett» det tilbake. |
| `musings.txt`, `musingsUI.txt`, `musings_backend.txt` | **Eierens filer. Ikke skriv til dem.** Svar hører hjemme i `til_utvikler_v##.md`. |
| `contracts/openapi.json` | **Generert fra backend-koden, aldri redigert for hånd.** Backend eier generatoren (`backend/tools/gen_openapi.py`); alle leser fila. CI feiler hvis den er utdatert. Den er **ikke uttømmende** — les `contracts/README.md` før du bygger mot den. |
| `.github/workflows/ci.yml` | Én jobb per område (`core`, `android`, `backend`). Rør kun din egen jobb. |
| `docs/ARCHITECTURE.md` | Kart over hvor arkitekturteksten bor, + bygg/CI som gjelder alle tre. Beslutningsloggene ligger hos områdene (`core/ARCHITECTURE.md`, `android/ARCHITECTURE.md`) — det opprettes ingen tredje. |

### Én tekst, ett sted

Fakta som gjelder to områder skrives **ett** sted, og det andre stedet peker dit.
En regel som står to steder blir før eller siden to ulike regler.

| Tema | Eier |
|---|---|
| Wire-kontrakten klient↔server: `retryable`, sidecar v2 + `tag`-enumet, blob-formatet, `key_material`, gravsteiner | `android/KONTRAKT.md` |
| Endepunkter, tokens, kvoter, lagring | `backend_spec.md` |
| CV-kontrakten: `BfResult`, statuskoder, `BF_MAX_HITS`, pikselformater | `core/ARCHITECTURE.md` |
| Katalogkart og eierskap | denne fila |
| Feller og lærdommer | denne fila, §7 |
| Skjermflyt og CV-flyt (mermaid) | `docs/flytskjema.md` |

**Prinsippet som avgjør hvem som eier hva:** en invariant eies av den som
**håndhever** den, ikke den som må adlyde. `retryable` eies av klienten selv om
backend må rette seg etter den; tyverideteksjonen ved refresh eies av serveren
selv om klienten må unngå å utløse den.

*`backend_spec.md` §14–§16 ble redusert til pekere 2026-08-07 og er ikke lenger
en dublett.*

### Filene du skal kjenne ved navn

Denne fila er den ene alle leser. Resten må du gå til selv:

| Fil | Hva den er |
|---|---|
| `core/ARCHITECTURE.md` | Kjernens beslutningslogg: modultabell, numerikk-avvik, CV-kontrakt, MPI-beslutningen, auto-capture, FFI, verifisering |
| `android/ARCHITECTURE.md` | Klientens beslutningslogg: nettverkslag, Keystore/økt, nøkkellagene, soft-delete, broen til kjernen |
| `android/CLAUDE.md` | Klientens arbeidsinstruks: bygg og signering, invarianter, hva de andre eier |
| `android/CHANGELOG.md` | Hva hver klientrunde faktisk endret, v0.6→. Nye runder føres inn her, ikke i spec eller flytskjema. |
| `android/KONTRAKT.md` | Det andre kan stole på: `retryable`, sidecar v2 + `tag`-enumet, blobens ytre format, gravsteiner, øktgarantiene — og §9, det kontrakten ikke holder |
| `docs/ARCHITECTURE.md` | Kart over hvor arkitekturteksten bor + bygg/CI for alle tre |
| `docs/flytskjema.md` | Mermaid: CV-flyten (auto-capture, analyse, statuskoder) og skjermflyten |
| `AAPNE_PUNKTER.md` | Alt som ikke kan besluttes i kode — `TODO(eier)`, ukalibrerte verdier, åpne spec-punkter, med punkt-ID |
| `backend/CLAUDE.md` | Backendens arbeidsinstruks: bygg, invarianter, hva de andre eier |
| `backend/KONTRAKT.md` | Det backend garanterer utad: statuskoder, idempotens, grenser, hva vi ikke lagrer |
| `backend/BESLUTNINGER.md` | Backendens beslutningslogg, med en egen liste over valg som mangler dokumentert begrunnelse |
| `backend_spec.md` | Endepunktene, tokens, kvoter, lagring, personvern |
| `bestefar_CV-kjerne_spec.md` | Kravspec for kjernen |
| `bestefar_UI_spec.md` | UI/UX-spec + endringslogg per runde |

---

## 5. Miljø

```powershell
# CV-pipelinen (cv2, scipy):
.venv\Scripts\python.exe

# Backenden (fastapi, sqlalchemy, psycopg) — EGEN venv, ikke bland dem:
backend\.venv\Scripts\python.exe

# Gradle trenger Android Studios JBR:
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"

# gh er installert, men ikke på PATH i skall som startet før installasjonen:
& "C:\Program Files\GitHub CLI\gh.exe"      # sjekk med `where.exe gh`
```

- **`python`, `py` og `python3` finnes ikke.** De peker på Microsoft
  Store-stubben («Python was not found») eller på en tolk uten `cv2`/`scipy`.
  **Denne fella kostet en hel sesjon.** Symptomet er lumsk: du «kjører» et
  script, melder at fila er oppdatert, men ingenting ble skrevet — brukeren ser
  den gamle fila. **Sjekk alltid at scriptet faktisk skrev forventet utdata før
  du melder OK.**
- Det finnes en tredje venv, `.Bestefar\`, som mangler `scipy`. Ikke bruk den.
- I Bash-verktøyet må stier som sendes til `python.exe` skrives `C:/Users/...` —
  den er en Windows-binær og forstår ikke `/c/Users/...`.
- Konsollen er cp1252. Unngå ikke-ASCII (λ, δ) i `print`.
- Doble anførselstegn *inne i* argumenter til native exe brekker
  argument-parsingen i PowerShell 5.1. Bruk `-F melding.txt` framfor
  `git commit -m "…"`.
- **Filnavn skal være ASCII.** PowerShell 5.1 mangler æøå når filnavn sendes
  videre til `git.exe` — `git mv` svarer «fatal: bad source». Innholdet skal ha
  norske tegn; filnavnet skal ikke.

**Hemmeligheter settes av utvikler selv**, med
`flyctl secrets set NAVN=verdi -a bestefar-api`. Passord, tokens og nøkler skal
aldri inn i chat-loggen. En instans som trenger en secret satt, ber om det — den
genererer den ikke og skriver den ikke ut.

**Testsuiten kjøres aldri mot produksjonsdatabasen.** `conftest` gjør
`alembic downgrade base`, som dropper alle tabeller.

---

## 6. Verifisering før du melder noe ferdig

```powershell
# Kjerne
cmake --build core\build
.venv\Scripts\python.exe verify_cset_cpp.py     # krav: 10/10

# Backend
backend\.venv\Scripts\python.exe -m pytest backend\tests -q

# UI
cd android; .\gradlew assembleDebug
```

Rapporter utfallet slik det faktisk ble. Feiler noe, si det med utdraget.

---

## 7. Feller og lærdommer

Destillert fra `CLAUDE_CONTEXT.txt`, som ble slettet 2026-08-07 — den var en
dokumentasjonskanal ingen instans leste automatisk. Hele den gamle fila med alle
rundenotatene ligger i git-historikken (`git show <commit>^:CLAUDE_CONTEXT.txt`).
Det som sto der om *hvorfor koden ser ut som den gjør*, er flyttet til
beslutningsloggene; det som står her, er feil vi kan gjøre om igjen.

**Fører du opp noe nytt:** skriv det som en regel, ikke som en hendelse. Og
slett punktet når det ikke lenger kan skje.

### 7.1 Når tre instanser deler én arbeidskopi

- **Git-indeksen er felles.** `git add` + `git commit` i to steg har allerede
  ført til at en hel runde havnet i en annen instans' commit — pushet, og da
  kan den ikke skrives om. Bruk `git commit <sti> … -F melding.txt` med
  **eksplisitte stier på selve commit-kommandoen**. Tidsvinduet mellom `add` og
  `commit` er hele problemet.
- **Delte filer redigeres samtidig.** Edit-verktøyet melder «modified since
  read». Ta det varselet alvorlig: gjør små, målrettede endringer og les
  regionen på nytt rett før du skriver noe som avhenger av konteksten rundt.
- **Bruk aldri Write på en delt fil.** En full overskriving av
  `til_utvikler_v0NN.md` slettet en annen instans' seksjon som måtte skrives på
  nytt. Sjekk om fila finnes (`git ls-files`) og legg til nederst.
- **Les speccen på nytt før du skriver kode mot et endepunkt du selv har
  «foreslått».** Det kan allerede være bygget, med en annen rute enn din.
- **Ikke kjør `flyctl` fra to instanser samtidig.** `~/.fly/config.yml` ble
  skrevet full av NUL-bytes og måtte gjenopprettes med `flyctl auth login`.
- **Eierens filer endrer seg underveis.** Får du `FileNotFoundError` på en fil
  du nettopp listet: list på nytt.

### 7.2 Ting som ødelegger ekte data

- **Kjør aldri testsuiten mot produksjonsdatabasen.** `conftest` gjør
  `alembic downgrade base`, som dropper alle tabeller.
- **Sjekk hva som ligger der før du skriver til en produksjonskonto.** En
  testblob ble lagt i utviklerens egen backup uten at `GET /v1/backup/meta` var
  sjekket først. 409-vernet stopper bare *eldre* innsendinger.
- **Rydd aldri i samme kjøring som produserer data du kan trenge igjen.** Et
  `rm` på slutten av scriptet som skrev og leste filene kjørte selv om lesingen
  feilet — og engangskoden var brukt opp.
- **Hemmeligheter settes av utvikler selv**, med
  `flyctl secrets set NAVN=verdi -a bestefar-api`. Passord, tokens og nøkler skal
  aldri inn i chat-loggen. Trenger du en secret satt, be om det — ikke generer
  den og ikke skriv den ut.

### 7.3 Designregler som er kjøpt dyrt

- **Lager du et enkeltpunkt, bygg utskiftningsveien i samme runde.**
  `BACKUP_ESCROW_SECRET` var uerstattelig, og advarselen var en kommentar som sa
  «roteres ALDRI uten en plan». En advarsel i en kommentar er ikke et tiltak.
- **En feil som først oppdages når en bruker rammes, er en designfeil.** En
  feilsatt escrow-hemmelighet ville vist seg første gang noen prøvde å
  gjenopprette kopien sin. Derfor nøkkel-ID på raden og status i `/health`.
- **«Privat som standard» uten en bryter er «aldri».** Skadedata var
  implementert som «ingen bryter, aldri lagret», og det gjorde den mest
  verdifulle delen av forskningsmaterialet umulig å samle inn. Standardverdien
  `False` oppfyller kravet; umulighet er noe annet.
- **Slå aldri sammen delingsvalg med ulik sosial kostnad.** «Jeg skjøt stående
  på 85 meter» og «dyret ble skadeskutt og aldri funnet» er ikke samme
  opplysning å dele om seg selv.
- **En sletting skal ikke i stillhet skru av en innstilling**, og veien *ut* av
  et personvernvalg skal aldri kunne feile på en driftsinnstilling.
- **Et dokument beskriver enten nåtilstanden eller historien, aldri begge.**
  «Nytt i v0.NN» på slutten av en fil er billig å skrive og gjør fila uleselig
  forfra: `bestefar_UI_spec.md` fikk tretten slike lag og `docs/flytskjema.md`
  fem, og for å vite hva som gjaldt måtte du lese alt og holde styr på hvilket
  tillegg som overstyrte hvilket. Ryddet 2026-08-08 til `android/CHANGELOG.md` og
  `backend/CHANGELOG.md`. Endrer du oppførselen, **endre beskrivelsen** — og før
  runden i endringsloggen.
- **Endres speccen, skriv hvorfor — ikke stryk det gamle.** §13 forbød et
  «gjenopprett kopien min»-endepunkt; deponering ble likevel bygget. Forbudet
  gjaldt å hjelpe *uten* nøkkelen, så setningen fikk stå og endringen ble
  forklart. En spec der gamle forbud bare forsvinner, kan ingen stole på.
- **Funksjonalitet som skal pauses: bruk ett navngitt flagg** som alle
  inngangene sjekker (`Dialogs.RESEARCH_ENABLED`, `RESEARCH_ENABLED`). Ikke
  slett kode.
- **Når en teller eller knapp gjelder backend, sjekk at det finnes en sender.**
  Tre plassholdere i UI-et løy i månedsvis: en kø ingen tømte, en teller som bare
  kunne vokse, og en knapp som viste kvittering uten å sende.
- **En feilkode som får en forklaring før den får en årsak, blir stående.**
  `AvansertActivity` behandler `503 || 404 || 405` som «deponering er av på
  serveren». 503 er dokumentert; 404 og 405 er gjetninger uten kjent
  opphav — og en gren som allerede *har* en forklaring, er et sted en ekte
  årsak kan gjemme seg uten å bli lett etter. Grener du på en statuskode du ikke
  har fremkalt med vilje, skriv ned hvordan den oppstår. Kan du ikke det, er den
  ikke håndtert; den er skjult.
- **En fallback-kjede som behandler en feil som et fravær, skjuler feilen.**
  `BackupKeys.resolve` prøver lokalt → Block Store → deponering, og returnerte
  tom streng både når deponeringen var tom og når kallet aldri nådde fram. Da
  `Api.send` gjorde GET om til POST og deponeringen svarte 405 hver gang, så det
  ut som «ingen nøkkel deponert» i tre versjoner — brukeren ble bedt om
  gjenopprettingskoden, altså nøyaktig det funksjonen fantes for å slippe. Skill
  «fant ingenting» fra «fikk ikke svar», i det minste i loggen.
- **Kode uten kaller blir aldri verifisert.** `Backup.meta()` var skrevet,
  dokumentert og ødelagt fra dag én. Ingen oppdaget det, fordi ingen kalte den.
  Bygger du et endepunkt du «skal bruke senere», er det ikke bygget.
- **Mistenk stille suksess like mye som synlig feil.** Tre feil i dette repoet
  hører til samme klasse — noe galt som ser vellykket ut for begge parter:
  feil feltnavn i multipart ga **201 med tomme poenglister**; en `data`-only
  push blir **aldri tegnet** når appen er i bakgrunnen; en GET som ble POST fikk
  et 405 ingen så. Ingen av dem hadde en feilmelding å lete etter. Når du kobler
  til noe nytt, verifiser at den *riktige* veien virker — fravær av feil er ikke
  bevis.

### 7.4 Arbeidsmåte

- **Sjekk API-et i modulen før du kaller det**, også i moduler du har lest før.
  `database.session()` finnes ikke; `db.py` har `db()` som FastAPI-avhengighet.
- **Sjekk ruten i routeren før du skriver testen.** Fire tester feilet på 405
  fordi `PATCH` ble gjettet der ruten er `PUT`.
- **Ikke les en datert layout-tabell som fasit.** `docs/ARCHITECTURE.md`
  (2026-07-05) beskrev en beslutningslogg som aldri har eksistert, utelot fire
  kataloger, og kalte backenden «FastAPI + SQLite … tre routere» da den var
  Postgres med fjorten. Verifiser mot `git ls-files`.
- **Når brukeren sier «X ble ikke implementert» men koden finnes:** sjekk om de
  testet en eldre APK, eller om feltet bare vises betinget.
- **Når du innfører en tidsgrense, sjekk hvem som allerede kaller endepunktet i
  løkke.** Sperrefristen på «send ny kode» brøt ni innloggingstester.
- **Les grensen fra `settings()` i tester.** En test som hardkodet kvoten 5
  feilet da kvoten ble hevet til 10 — på kvoten, ikke på oppførselen.

### 7.5 Områdespesifikke feller

Detaljene bor i beslutningsloggene; dette er stikkordene som er verdt å huske
på tvers.

**Kjerne** — Python-referansen er fasit, og de to bevisste avvikene (alle
kantpunkter i stedet for subsampling; rekalibrerte NCC-terskler) er begrunnet i
`core/ARCHITECTURE.md`. Feilsøkingsmetoden som virker: dump C++-cropen, kjør
Python-stadiene på den, binærsøk i stadiekjeden.

**Backend** — legger du til en forelder og et barn i samme flush *uten* en
`relationship()` mellom dem, må du `flush()` imellom; SQLite sier ingenting
(`PRAGMA foreign_keys` er av som standard) og feilen dukker først opp i CI mot
Postgres. Lager du en engine i en test eller i `env.py`, må du `dispose()` den.
Autogenerate kan ikke brukes: mot SQLite rapporterer den hele
`research`-skjemaet som slettet. Naiv vs. aware datetime er løst med
`UtcDateTime` i `models/base.py` — ikke oppgi `mapped_column(DateTime, …)`.

**Klient** — `Ui.themeColor` må slå opp ColorStateList-attributter (særlig
`android:textColorPrimary`); er noe tintet med tekstfargen «usynlig», mistenk
theme-attr-oppløsning først. En KDoc som inneholder `/*` (f.eks. en sti med
`/v1/auth/*` i backticks) åpner en nestet kommentar og gjør resten av fila til
kommentar — feilmeldingen peker aldri på årsaken. `targetSdk` 36 gjør
`statusBarColor` til en no-op; appen tegner systemlinjene selv.
`androidx.security:security-crypto` er avviklet — Keystore direkte. AGP 9s lint
kan krasje i sin egen UAST-kode («this is a bug in lint») på en Java-getter lest
som Kotlin-egenskap i en lokal variabel — kall getteren eksplisitt i stedet for å
skru av `lintVital`, som ville skjult alle framtidige ekte funn.
