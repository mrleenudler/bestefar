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
│   └── cli/                 KJERNE  desktop-CLI for verifisering mot C-settet
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
│   └── app/src/main/cpp/    UI*     jni_bridge.cpp — se merknad under
├── UI/                      UI      ikoner, silhuetter, SVG-kilder
├── dist/                    UI      APK-er som sendes ut (versjonsbump ved ny)
│
├── docs/                    DELT    ARCHITECTURE.md, flytskjema.md
├── *_spec.md                DELT    de tre spesifikasjonene (se §4)
├── til_utvikler_v##.md      DELT    tilbakemelding per runde (se §4)
├── musings*.txt             EIER    skrives av utvikler — ikke skriv til dem
├── CLAUDE_CONTEXT.txt       DELT    feillogg på tvers av instanser
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

Gjelder overalt: spec-filer, `til_utvikler_v##.md`, `CLAUDE_CONTEXT.txt`,
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
| `backend_spec.md` | Backend eier §0–§11; UI eier §12–§16 (klientsiden). Rediger kun din del, og les regionen på nytt rett før du skriver — den andre instansen kan ha endret filen. |
| `bestefar_UI_spec-v0-4.md` | UI eier. |
| `til_utvikler_v##.md` | **Delt per runde.** Legg til en seksjon nederst med områdenavn i overskriften. Overskriv aldri. Én fil per runde, høyeste nummer er gjeldende. |
| `CLAUDE_CONTEXT.txt` | Alle skriver. Før inn feil du gjorde som burde vært husket. Les den etter komprimering. |
| `AAPNE_PUNKTER.md` | Alle skriver. Legg til når du oppdager noe som ikke kan besluttes i kode; stryk aldri et punkt uten at det faktisk er avklart av eier. **Navnet er med `AA`, ikke `Å`** — PowerShell 5.1 mangler æøå når filnavn sendes videre til `git.exe`, så `git mv`/`git commit <sti>` feiler på den. Ikke «rett» det tilbake. |
| `musings.txt`, `musingsUI.txt`, `musings_backend.txt` | **Eierens filer. Ikke skriv til dem.** Svar hører hjemme i `til_utvikler_v##.md`. |
| `.github/workflows/ci.yml` | Én jobb per område (`core`, `android`, `backend`). Rør kun din egen jobb. |

---

## 5. Miljø

```powershell
# Python finnes ikke på PATH. Backend-venv:
C:\Users\mrlee\Desktop\Bestefar\backend\.venv\Scripts\python.exe

# Gradle trenger Android Studios JBR:
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
```

I Bash-verktøyet må stier som sendes til `python.exe` skrives `C:/Users/...` —
den er en Windows-binær og forstår ikke `/c/Users/...`.

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
