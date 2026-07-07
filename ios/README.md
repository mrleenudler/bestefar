# Bestefar iOS — stub

iOS prioriteres ETTER Android (kravspec §1). Kjernen er klar for iOS:
`core/include/bestefar/bestefar_ffi.h` er ren C og kalles direkte fra Swift.

## Plan (neste iterasjon)

1. Bygg `bestefar_core` som XCFramework:
   - CMake toolchain for iOS (arm64 + simulator), OpenCV via
     offisiell `opencv2.framework` eller vcpkg.
   - Modulemap som eksponerer `bestefar_ffi.h`.
2. SwiftUI-skall med samme 3-skjerms flyt som Android
   (Start -> AVFoundation-kamera + auto-capture via `bf_autocapture_feed`
   paa `AVCaptureVideoDataOutput`-frames (Y-plan) -> resultat -> OK).
3. Gjenbruk BfAutoCaptureParams-kalibreringen fra Android-felttesting som
   startpunkt, men RE-kalibrer mot iOS-kameraets karakteristikk (kravspec §4).

Ingen Swift-kode i denne iterasjonen — kontrakten (C-headeren) er leveransen.
