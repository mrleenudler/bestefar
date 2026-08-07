/* Kjernens EGEN versjon — uavhengig av appens versionName (build.gradle.kts).
 * Bump denne naar en endring i core/ paavirker analyse eller auto-capture,
 * UANSETT om appens versionName ogsaa bumpes samme runde (den gjoer det som
 * regel, siden UI- og kjerne-endringer ofte havner i samme commit — men de to
 * tallene maaler ikke det samme og skal IKKE holdes synkronisert med vilje).
 * Se bestefar_ffi.h: bf_version(). Formaalet er §6-donasjonene i
 * backend_spec.md: en kalibreringsmaaling er verdiloes uten aa vite hvilken
 * kjerne som produserte den. */
#ifndef BESTEFAR_VERSION_H
#define BESTEFAR_VERSION_H

#define BESTEFAR_CORE_VERSION "1.0.0"

#endif /* BESTEFAR_VERSION_H */
