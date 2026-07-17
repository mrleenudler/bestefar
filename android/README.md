# Bestefar Android

Native Android-app rundt den delte C++-kjernen (`../core`).

## Bygge

1. Installer Android Studio med NDK (r26+) og CMake 3.22.1 (SDK Manager).
2. Last ned [OpenCV Android SDK](https://opencv.org/releases/) (4.x), pakk ut.
3. Sett `opencvDir` i `gradle.properties` til
   `<utpakket>/OpenCV-android-sdk/sdk/native/jni`.
4. Åpne `android/` i Android Studio og bygg (`assembleDebug`).

Kjernen bygges av NDK-en via `app/src/main/cpp/CMakeLists.txt`, som inkluderer
`core/CMakeLists.txt` direkte — samme kode som desktop-verifiseres mot C-settet
(`verify_cset_cpp.py`, krav 10/10).

## Flyt (UI-iterasjon, bestefar-spec-v0-4.md)

Skall med seks faner (Våpen, Avstand, Stilling, Innsikt, Jakt, Meny) og stor
sentrert «Scan serie»-knapp. Capture-løkka: scan → auto-capture (UKALIBRERTE
terskler i `core/include/bestefar/config.h` `AutoCaptureParams`) → analyse på
enheten → stillingsprompt → resultatkort (skiveplott, korrigering,
klikk-forslag) → OK/avslutt økt.

UI-lag (alt i `app/src/main/java/no/bestefar/app/`, programmatiske views):

- `Model.kt` / `Store.kt` — lokal JSON/prefs-persistens (offline-først)
- `Stats.kt` — P(dødelig), frekvensspråk, maks hold, klikk-forslag.
  MERK: `RING_STEP_CM` og dødelig-sone-radiene er PLASSHOLDERE (spec §10.1)
- `MainActivity` — skall/faner; `OktFragment` — hjem/øktflate
- `InnsiktFragment` — kompetanseoversikt + kapabilitetskart
- `JaktFragment` / `HuntLogActivity` — jaktmodus + hurtiglogg (3 steg)
- `ResultActivity` / `SummaryActivity` — resultatkort + øktoppsummering
- `ProfilActivity` — skytterprofil, våpenkartotek, samtykker, sletting
- `OnboardingActivity` — tre skjermbilder ved første start

## Kalibrering av auto-capture

Tersklene i `AutoCaptureParams` er startverdier og SKAL kalibreres mot faktisk
maskinvare (kravspec §4). Eksponer dem gjerne i en debug-meny før felttesting.
