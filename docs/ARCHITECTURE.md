# Arkitektur — kart og fellesdeler

**Opprinnelig status:** Første Fable-iterasjon, 2026-07-05.
**Splittet og verifisert mot koden:** 2026-08-07.

Dette dokumentet var én arkitekturbeskrivelse for hele repoet. Med tre instanser
som jobber i hver sin del, ble det en fil tre parter måtte redigere samtidig for
å holde à jour — og den drev fra koden. Innholdet er derfor **flyttet ordrett**
ut til det området som eier det. Her står bare det som gjelder alle tre.

## Hvor innholdet gikk

| Opprinnelig avsnitt | Nå i |
|---|---|
| Repo-layout | rot-`CLAUDE.md` (hele kartet m/eierskap) + `core/ARCHITECTURE.md` (kjernens undertre) |
| C++-kjernen: modulinndeling, numerikk-avvik, minne/tråder | `core/ARCHITECTURE.md` |
| CV-kontrakt (`AnalyzeResult`, `BfImage`, statuskoder) | `core/ARCHITECTURE.md` |
| MPI-firkanten — beslutning | `core/ARCHITECTURE.md` |
| Auto-capture (`FrameProbe`, `AutoCapture`) | `core/ARCHITECTURE.md` |
| FFI-strategi | `core/ARCHITECTURE.md` |
| Verifisering av porten | `core/ARCHITECTURE.md` |
| Klientens nettverkslag (`Api`, `Sync`, `Backup`) | `android/ARCHITECTURE.md` |
| Økt og hemmeligheter (`Secrets`, `Auth`, `BackupKeys`, `Lock`, `Login`, `Push`) | `android/ARCHITECTURE.md` |
| Soft-delete | `android/ARCHITECTURE.md` |
| Backend (FastAPI-routerne) | `backend_spec.md` §0.1 — sto der allerede, i nyere form |
| Bygg/CI | **her**, under |

Kontraktsteksten mellom klient og server — feilklassifisering, sidecar-format,
blob-format — eies av `android/KONTRAKT.md`. Ingen andre dokumenter skal gjenta
den.

## Om beslutningsloggen

Den gamle repo-layouten beskrev `docs/` som «Dette dokumentet + beslutningslogg»,
som om beslutningsloggen var en egen fil. **Den har aldri eksistert.**
Beslutningene har hele tiden ligget *inne i* arkitekturdokumentet — MPI-firkanten,
de bevisste numerikk-avvikene, valget av Credential Manager framfor
`GoogleSignInClient`, hvorfor `androidx.security:security-crypto` ble vraket.

Det er en god ordning, og den videreføres: **`core/ARCHITECTURE.md` er
beslutningsloggen for kjernen, `android/ARCHITECTURE.md` for klienten.** Det
opprettes ingen tredje logg. Skriv beslutningen der koden den gjelder bor, med
begrunnelsen intakt — det er begrunnelsen som er verdt noe når noen om et halvt
år lurer på om valget kan reverseres.

`docs/` inneholder etter dette:

- `ARCHITECTURE.md` (denne) — kart + bygg/CI
- `flytskjema.md` — mermaid-diagrammer over CV-flyten og skjermflyten, avledet
  fra koden i v0.18

## Bygg/CI

- `core/`: CMake ≥3.22. Desktop: MSYS2/MinGW eller Linux (apt libopencv-dev).
  Mobil: samme CMakeLists konsumeres av NDK.
- GitHub Actions: (1) core-build + enhetstester på ubuntu; C-sett-verifisering
  kjøres lokalt (testbildene er for store for CI-artefakter), (2) android
  assembleDebug, (3) backend pytest.

**Rettelser 2026-08-07 (verifisert mot `.github/workflows/`):**

- **Android-jobben kjører fra 2026-08-08** (ÅP-U12 lukket). `if: false` er
  fjernet, OpenCV Android SDK lastes ned og bufres, og jobben bygger
  `assembleDebug` med wrapperen — kun debug, siden release krever
  signeringsnøkkelen som ligger utenfor repoet. Jobben sjekker utfallet og ikke
  bare exit-koden: APK-en skal finnes og inneholde `libbestefar_jni.so`.

  **NDK-bygget er ikke dobbeltarbeid mot core-jobben.** De to konsumerer samme
  `core/CMakeLists.txt`, men med hver sin toolchain: core-jobben bygger for
  x86_64 med systemets GCC/libstdc++ mot `libopencv-dev`, android-jobben for
  arm64-v8a med NDK-ens Clang og libc++ mot OpenCV Android SDK, med `minSdk 26`
  som API-nivå. Det som brekker i den ene og ikke i den andre er reelt:
  transitive `#include`-er libstdc++ drar inn og libc++ ikke, 32/64-bits
  antakelser, og OpenCV-moduler som finnes i apt-pakken men ikke i
  Android-SDK-en. I tillegg dekker android-jobben Kotlin-kompilering, ressurs-
  og manifestlenking, som core-jobben ikke rører.
- **Backend-jobben kjører pytest to ganger:** først mot SQLite, så mot Postgres
  16 som service-container. Begge trengs — Postgres er produksjonsdialekten, og
  skjemaet `research` finnes bare der (SQLite har ikke skjemaer), så
  driftsjekken mellom modeller og migrasjoner kan bare kjøre på Postgres.
- **Det finnes en andre workflow**, `deploy-backend.yml`: push til `main` som
  rører `backend/**` kjører testene mot Postgres og deployer så til Fly.io med
  `flyctl deploy --remote-only`. Krever repo-secreten `FLY_API_TOKEN`.
  `concurrency: deploy-backend` med `cancel-in-progress: false` hindrer to
  deployer i å overlappe.
- **Begge workflowene pakker pytest-utdata som `::error`-annotasjon ved feil.**
  Jobbloggen svarer 403 uten token — også for offentlige repoer — mens
  annotasjoner er åpne. Uten innpakningen må loggen eksporteres for hånd hver
  gang noe feiler. Bare halen sendes (3000 tegn): GitHub kutter lange
  annotasjoner fra slutten, så et stort utdrag skjuler nettopp oppsummeringen.

### Lokal verifisering

```powershell
# Kjerne
cmake --build core\build
.venv\Scripts\python.exe verify_cset_cpp.py            # krav: 10/10

# Backend
backend\.venv\Scripts\python.exe -m pytest backend\tests -q

# Klient (bygges ikke i CI - se over)
$env:JAVA_HOME = "C:\Program Files\Android\Android Studio\jbr"
cd android; .\gradlew assembleDebug
```
