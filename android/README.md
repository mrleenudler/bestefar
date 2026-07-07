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

## Flyt (første iterasjon, kravspec §7)

Start → kameraskjerm med skanner-ramme (auto-capture: stabilitet + kvalitet,
UKALIBRERTE terskler i `core/include/bestefar/config.h` `AutoCaptureParams`)
→ analyse på enheten → resultatskjerm → OK → Start.

## Kalibrering av auto-capture

Tersklene i `AutoCaptureParams` er startverdier og SKAL kalibreres mot faktisk
maskinvare (kravspec §4). Eksponer dem gjerne i en debug-meny før felttesting.
