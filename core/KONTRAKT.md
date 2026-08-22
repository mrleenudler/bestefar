# Kjernens kontrakt utad

**Eier: kjerne-instansen.** Dette er hva JNI-broen, en fremtidig Swift-bro og
backenden (via `core_version` i donasjonene) kan stole på: `bestefar_ffi.h`,
statuskodene, `BfImage`-formatene, resultat-skjemaene og `BF_MAX_HITS`.

**Hva som hører hjemme her:** garantier andre kan bygge på — feltnavn, typer,
verdiområder, hva som ALDRI skjer. **Ikke** hvorfor det ble slik eller hvilke
avveininger som ligger bak — det er beslutninger, og de bor i
`core/ARCHITECTURE.md`. Samme skille som `android/KONTRAKT.md` og
`backend/KONTRAKT.md` bruker.

**Eierskap:** en invariant eies av den som **håndhever** den, ikke den som må
adlyde. `bestefar_ffi.h` står derfor her selv om JNI-broen (`jni_bridge.cpp`)
fysisk ligger i `android/` — kjernen definerer strukturene, UI serialiserer dem.

Alt under er verifisert mot koden 2026-08-07 (`bestefar_ffi.h`, `types.h`,
`analyze.cpp`). Filreferansene er der du kan etterprøve det.

---

## 1. `bf_analyze` — statuskoder

`BfResult.status` er én av:

| Kode | Verdi | Betyr |
|---|---|---|
| `BF_OK` | 0 | Analyse lyktes, `hits[]`/`sum_*` er gyldige |
| `BF_REJECTED_NO_SCREEN` | 1 | Definert, **emitteres aldri** — se §5 |
| `BF_REJECTED_NO_RINGS` | 2 | Fant ingen poengringer |
| `BF_REJECTED_INVALID_TARGET` | 3 | Kalibreringen ble avvist (delta/ringtall/residual) |
| `BF_REJECTED_NO_HITS` | 4 | Ingen treff funnet på skiva |
| `BF_ERROR_BAD_INPUT` | 100 | `image`/`out` var null, eller bildet var tomt |
| `BF_ERROR_INTERNAL` | 101 | Uventet C++-exception fanget i FFI-laget |

**`bf_analyze` kaster aldri over FFI-grensen.** Enhver feil — også en du ikke
har sett før — kommer ut som `BF_ERROR_INTERNAL`, ikke som en krasjende app.

**`message` er ikke en stabil, maskinlesbar verdi.** Det er norsk
ASCII-translitterert fritekst ment for logg/debug (rot-`CLAUDE.md` §3), og for
`BF_ERROR_INTERNAL` kan den inneholde en rå C++-exception-melding
(`e.what()`, `analyze.cpp:166`) som ikke er skrevet for et sluttbrukeraudiens.
Match aldri på innholdet i `message`; match på `status`. Vis den aldri direkte
til bruker — se §5 for konsekvensen når UI-siden gjorde nettopp dette.

## 2. `BfImage` — pikselformater kjernen tar imot

Fire formater, ikke flere: `BF_FMT_GRAY8` (0), `BF_FMT_BGR8` (1),
`BF_FMT_RGBA8` (2), `BF_FMT_NV21` (3). Android `YUV_420_888` er **ikke** et eget
format — klienten pakker det til NV21 før kallet.

`BfImage.stride` er bytes per rad for **plan 0 only**. For NV21 forutsetter
kjernen kontinuerlig NV21-pakking av UV-planet umiddelbart etter Y-planet i
samme buffer; den tar ikke separate plan-pekere.

## 3. `BfResult` — hva som faktisk krysser grensen

`BfHit` per treff: `x_px, y_px` (inputbildets koordinater), `r_rel` (ringavstander
fra kalibrert senter), `theta` (radianer), `decimal`, `integer`, `detect_score`.
**Akse- og rammekonvensjonen for `r_rel`/`theta`/`x_px`/`y_px` — inkludert at
`theta` deler akse men ikke nødvendigvis ramme med `x_px`/`y_px` — står i
`bestefar_ffi.h` selv** (issue #12); gjentas ikke her for å unngå at de to
driver fra hverandre slik `stability_frames`-kommentaren gjorde (§5).

`BfResult` selv: `status`, `message[256]` (§1), `n_hits`, `hits[BF_MAX_HITS]`,
`sum_decimal`, `sum_integer`, `confidence`, `n_rings`, `ring_resid_frac`,
`timestamp_ms` (ekko av innsendt tidsstempel, ikke kjernens egen klokke).

**`BF_MAX_HITS` = 32 er et hardt tak, ikke en foreslått default.** Flere treff
enn det forsvinner stille — `bf_analyze` kutter ved 32, den feiler ikke og den
varsler ikke i `status`. En konsument som teller "reelle" treff mot `n_hits`
alene, kan derfor telle for lavt på et veldig fullt skudd-bilde uten at noe
signaliserer det.

**Bare tre konfidens-delfelt krysser grensen: `confidence`, `n_rings`,
`ring_resid_frac`.** C++-structen `bestefar::Confidence` har i tillegg
`mean_hit_score` og `screen_used`, men de er **ikke** i `BfResult` — de finnes
kun internt. Trenger UI eller backend et av dem, er det en ABI-utvidelse av
`bestefar_ffi.h`, ikke noe som allerede ligger og venter i strukturen.

## 4. `bf_version()`

Returnerer en semver-streng (`BESTEFAR_CORE_VERSION`,
`core/include/bestefar/version.h`) — **statisk, eid av kjernen, IKKE `free()`
den.** Bevisst uavhengig av appens `versionName`: bump kun når en endring i
`core/` faktisk påvirker analyse eller auto-capture, ikke i takt med
UI-utgivelser. Formålet er å vite hvilken kjerne som produserte en
§6-donasjon (`backend_spec.md` §8) — ikke å speile appversjoner.

## 5. Kjente unøyaktigheter

Ærlighet om hva kontrakten *ikke* holder:

- **`BF_REJECTED_NO_SCREEN` emitteres aldri.** Koden er reservert i `types.h`
  og `bestefar_ffi.h`, men finner ikke `rectify_to_screen` en skjerm i bildet,
  faller `analyze_target` alltid gjennom til helbilde-analyse i stedet
  (`analyze.cpp:133–158`) — den returnerer aldri status 1. En konsument som har
  bygget spesifikk UI-tekst for denne koden, viser den aldri. Se
  `docs/flytskjema.md` §1b.
- **`AutoCaptureParams`-defaultene er UKALIBRERTE startverdier**, ikke
  produksjonsmålte terskler — `bf_autocapture_default_params()` gir deg et
  førstepass, ikke en garanti om at auto-capture trigger ved riktig
  bildekvalitet på en gitt telefon. Se `AAPNE_PUNKTER.md` ÅP-K1.
- **`stability_frames`-kommentaren i `bestefar_ffi.h` var feil til 2026-08-07**
  (sa 24, faktisk default 6 siden v0.12) — rettet, men er et konkret eksempel
  på at en kommentar i denne fila kan drifte fra `config.h` uten at noe bygg
  varsler det. Stol på `config.h`s faktiske defaultverdi, ikke bare
  header-kommentaren, hvis de to noensinne ser ut til å være uenige igjen.
- **iOS-broen er et skjelett.** `bestefar_ffi.h` er skrevet for å være
  plattformnøytral C, men det finnes i denne iterasjonen ingen reell
  Swift-konsument som har verifisert kontrakten fra den siden — kun en stub
  (`ios/`). Formen er ikke feltprøvd på iOS.

---

## Hvor resten står

| Tema | Eier |
|---|---|
| Hvorfor porten ser ut som den gjør, numerikk-avvik fra Python, MPI-firkant-beslutningen | `core/ARCHITECTURE.md` |
| Auto-capture-tilstandsmaskinen og hvorfor tersklene er som de er | `core/ARCHITECTURE.md`, `config.h` |
| CV-flyten og skjermflyten som mermaid | `docs/flytskjema.md` |
| Hvordan JNI-broen serialiserer disse structene til Kotlin | `android/app/src/main/cpp/jni_bridge.cpp`, `BestefarCore.kt` |
| `core_version` i donasjonene | `backend/KONTRAKT.md` §3, `backend_spec.md` §8 |
