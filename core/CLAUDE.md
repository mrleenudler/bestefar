# core/ — arbeidsinstruks

**Les rot-`CLAUDE.md` først**, særlig §7 (feller og lærdommer). Eierskapsregelen,
issue-flyten, PowerShell-syntaks, ISO-datoer og de delte filene gjentas ikke her.

Denne fila dekker bare det som er særegent for kjerneområdet.

## Bygg og verifisering

Kjernen bygges og verifiseres **på desktop, uten mobil-toolchain** — det er
hovedgrepet som gjør porten etterprøvbar uten en Android-enhet.

```powershell
$env:PATH = "C:\tools\msys64\mingw64\bin;" + $env:PATH
cmake -S core -B core\build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build core\build
.venv\Scripts\python.exe verify_cset_cpp.py     # oracle for porten: krav 10/10
```

`.venv` i repo-roten er CV-pipelinen (`cv2`, `scipy`) — Python-referansen som
`verify_cset_cpp.py` sjekker C++-kjernen mot. Ikke bland den med
`backend\.venv`. `python`/`py`/`python3` uten sti finnes ikke i dette miljøet
(peker på Store-stubben) — bruk alltid `.venv\Scripts\python.exe`.

Pacman-pakker installert i MSYS2 (`mingw-w64-x86_64-opencv/cmake/ninja`) — bare
`mingw64\bin` trenger å stå på `PATH` for at kjørbare filer skal finne
OpenCV-DLL-ene sine.

## NDK-konsum (Android)

`android/app/src/main/cpp/CMakeLists.txt` gjør `add_subdirectory(...)` rett inn
i `core/CMakeLists.txt` — samme kildetre, ingen forhåndsbygget binær eller
pakket artifact går mellom områdene. Kjernen bygges dermed **to ganger**: én
gang her på desktop for verifisering mot C-settet, én gang av Gradle/NDK som
del av `assembleDebug`/`assembleRelease`. Endrer du noe i `core/src` eller
`core/include`, er det andre bygget (Android) som avgjør om endringen faktisk
virker i appen — desktop-verifiseringen sjekker kun analyseresultatet, ikke at
NDK-bygget lykkes. `gradlew assembleDebug` er UI-instansens jobb å kjøre, men
dukker feilen opp i kode du eier (`core/`), er det din å rette.

`android/app/src/main/cpp/jni_bridge.cpp` ligger fysisk i UI-området, men er
ren *forbruker* av `bestefar_ffi.h`. Endrer du headeren, er det du som eier
endringen; UI må følge etter — meld issue begge veier (rot-`CLAUDE.md` §1),
ikke rett i `jni_bridge.cpp` selv.

## Hvor tingene står

| Fil | Hva |
|---|---|
| `../bestefar_CV-kjerne_spec.md` | Kravspecen. Du eier hele den. |
| `core/ARCHITECTURE.md` | Beslutningsloggen: modultabell, numerikk-avvik fra Python, CV-kontrakten, MPI-firkant-beslutningen, auto-capture, FFI-strategi |
| `core/KONTRAKT.md` | Det du garanterer utad — `bestefar_ffi.h`, statuskoder, pikselformater, `BF_MAX_HITS` |
| `../docs/flytskjema.md` §1 | CV-flyten (auto-capture, analyse, statuskoder) som mermaid |
| `../AAPNE_PUNKTER.md` | Det som ikke kan besluttes i kode. Ukalibrerte terskler har egne punkt-ID-er (ÅP-K*) |
| `../docs/ARCHITECTURE.md` | Bygg/CI for alle tre områder |

Det finnes **ingen** `core/BESLUTNINGER.md`. `core/ARCHITECTURE.md` er
beslutningsloggen; en parallell fil ville bare drive fra den.

## Invarianter — omgjøres ikke uten at det står i ARCHITECTURE.md hvorfor

1. **Python-pipelinen i repo-roten er fasit.** `core/` porteres fra den og
   verifiseres mot samme oracle (`Testsett/C1–C10` + `hits_truth.txt`). Et avvik
   fra Python er kun gyldig når det er et bevisst, målt og dokumentert valg
   (se `core/ARCHITECTURE.md` "Numerikk-avvik fra Python") — ikke en tilfeldig
   driftet implementasjonsdetalj.
2. **Ingen exceptions over FFI-grensen.** Alt fanges i C-laget
   (`bf_analyze`/`bf_autocapture_feed`) og mappes til en statuskode. Et JNI-lag
   som ikke forventer en C++-exception, krasjer appen.
3. **`bestefar_ffi.h` er ren C**, ingen C++-typer, POD-structs. Det er
   kontrakten mot både JNI og en fremtidig Swift-bro; et C++-typebrudd der
   låser deg til én plattform.
4. **`BfResult`/`BfFrameProbe` er faste structs, ikke variable formater.**
   `BF_MAX_HITS` er et tak, ikke en foreslått default — treff utover det
   forsvinner stille i `bf_analyze`, de feiler ikke.
5. **`AutoCaptureParams`-terskler er UKALIBRERTE startverdier med vilje.**
   Ikke rund dem til "penere" tall eller fjern UKALIBRERT-merkingen uten en
   faktisk feltmåling bak — se `AAPNE_PUNKTER.md` ÅP-K1.
6. **`docs/flytskjema.md` §1 og `core/ARCHITECTURE.md` skal si det samme om
   statuskoder.** Oppdager du at flytskjemaet lyver (som `BF_REJECTED_NO_SCREEN`
   som aldri emitteres), ret begge — det er én kontrakt beskrevet to steder, og
   det stedet som lyver blir noens fasit.

## Hva de andre eier

Du leser gjerne koden deres. Du redigerer den ikke — issue med label
`ui`/`backend` (rot-`CLAUDE.md` §2.1).

| Eies av | Hva det betyr for deg |
|---|---|
| **ui** — `android/app/src/main/cpp/jni_bridge.cpp`, `BestefarCore.kt` | Forbrukere av `bestefar_ffi.h`. Endrer du protokollen (feltrekkefølge i `BfFrameProbe`/`BfResult`, nye statuskoder), er det UI som må oppdatere serialiseringen — meld issue, ikke rett i deres filer, selv om det "bare" er å legge til ett tall i en array. |
| **backend** — `backend_spec.md` §8 | CV-kjerne-oppgaver oppdaget i felt, men som ikke er backendens å løse, samles der som notat til deg. Sjekk den paragrafen når du starter en runde. |
| **backend** — `backend/KONTRAKT.md` §3 | `core_version`-kolonnen i §6-donasjonene tar imot `bf_version()`-strengen din uendret og validerer ikke formen — du kan endre versjonsskjemaet fritt uten at det er en backend-endring. |

## Verifiserings-etikette

- **Meld aldri "10/10 PASS" uten å ha faktisk kjørt scriptet i denne økten.**
  `.venv\Scripts\python.exe` som "kjører" uten feil, men uten synlig ny utdata,
  er symptomet på at `python`/`py` traff Store-stubben i stedet — se rot-`CLAUDE.md` §5.
- **En kommentar i en header er kode.** Endrer du en default i `config.h`, sjekk
  om `bestefar_ffi.h` beskriver samme default i en kommentar — de to driver fra
  hverandre stille (issue #1 er akkurat dette).
- **`core/build` er ikke inkrementelt trygt på tvers av CMake-cache-endringer.**
  Endrer du `CMakeLists.txt` (nye kilder, nye lenkede biblioteker), slett
  `core\build` og kjør `cmake -S core -B core\build` på nytt før du stoler på
  et rødt/grønt resultat.
