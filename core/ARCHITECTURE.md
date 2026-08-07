# Arkitektur — CV-kjernen (`core/`)

**Status:** Første Fable-iterasjon, 2026-07-05. Verifisert mot koden 2026-08-07.
**Underlag:** `bestefar_CV-kjerne_spec.md` (tidl. `kravspec.md`; omtales som
«kravspec» i kommentarer og under). Dette dokumentet fyller ut valgene
kravspec §2 delegerer.

Dette er **beslutningsloggen for kjernen** — hvorfor porten ser ut som den gjør,
og hva som bevisst ikke ble portert. Flyttet ordrett ut av `docs/ARCHITECTURE.md`
2026-08-07 da dokumentet ble splittet etter arbeidsområde; se
`docs/ARCHITECTURE.md` for kartet over splitten.

Klientsiden ligger i `android/ARCHITECTURE.md`, kontrakten mot backend i
`android/KONTRAKT.md`, bygg/CI i `docs/ARCHITECTURE.md`.

## Repo-layout

Hele katalogkartet med eierskap står i rot-`CLAUDE.md`. Kjernens del:

```
core/                  Delt C++17 CV-kjerne (kun OpenCV som avhengighet)
├── include/bestefar/  Offentlige headere (C++-API + ren C FFI)
├── src/               Porterte moduler (se tabell under)
├── cli/               Desktop-CLI for verifisering mot C-settet
└── CMakeLists.txt
*.py (repo-rot)        Python-referansen (urørt; fortsatt oracle via verify_cset.py)
```

Python-pipelinen forblir fasit. `core/` porteres fra den og verifiseres mot
samme oracle (`Testsett/C1–C10` + `hits_truth.txt`, krav: 10/10 PASS) via en
desktop-CLI. **Kjernen skal alltid kunne bygges og testes uten mobil-toolchain**
— det er hovedgrepet som gjør porten verifiserbar.

## C++-kjernen

### Modulinndeling (speiler Python-referansen)

| C++-modul        | Python-kilde       | Merknad |
|------------------|--------------------|---------|
| `config.h`       | config.py          | Typet struct, defaults = config.py-verdiene |
| `preprocess`     | preprocess.py      | downscale/gray/blur/Scharr/akse-suppresjon |
| `outer_circle`   | Bestefar.py + voting/histogram/points | Pass1+Pass2 gradient-voting. **NMS/hysterese/radial-varians-raffinering er IKKE portert** (`outermost_ring_refine_enable=False` i produksjon — død kode) |
| `screen`         | screen.py          | Kontrast-ROI, blob, quad, IRLS-linjer, kant-snapping, rektifisering |
| `rings`          | rings.py           | Polar kalibrering + harmonisk senterraffinering. **`ring_comb_v2`-stien (scipy-prominens) er IKKE portert** (av i produksjon) |
| `perspective`    | perspektiv.py      | 6-param rektifisering. scipy `least_squares(method='lm')` → håndrullet Levenberg–Marquardt med diagonal-skalering (6 param, ~10³ residualer) |
| `circles`        | circles.py         | Stemmekart-detektor (median→Scharr→retningsgating→radius-stemmer→NMS) |
| `hits`           | hits.py            | **Kun enhetlig detektor-sti** (`hit_unified_detector=True`; Hough-stien er legacy og ikke portert) |
| `overlap`        | overlap.py         | Månesigd-NCC for delvis skjulte treff + sentrum-sveip (`matchTemplate` m/maske) |
| `scoring`        | scoring.py         | Desimalpoeng m/lokal ringgeometri, trunkering |
| `analyze`        | Bestefar.py        | Orkestrering: screen→outer→rings→persp→hits→scoring |
| `autocapture`    | (ny, §4 i kravspec)| FrameProbe + trigger-tilstandsmaskin |
| `ffi`            | (ny)               | Ren C-API for JNI/Swift |

*Verifisert 2026-08-07:* alle modulene finnes i `core/src/`. `preprocess` er
header-only (`preprocess.h`, ingen `.cpp`).

### Numerikk-avvik fra Python (bevisste)

- **RNG:** numpy PCG64(seed 42) → `std::mt19937_64(42)`. Stemme-par-samplingen
  er statistisk, ikke bitvis, ekvivalent. Oraclet (heltall antall+sum) er
  korrekthetskriteriet, ikke flyttallslikhet.
- **np.percentile / gaussian_filter1d / find_peaks:** implementert i
  `numpy_compat.h` med numpy-semantikk (lineær interpolasjon; reflect-kant;
  lokale maksima m/høydeterskel).
- **np.linalg.lstsq / eigh:** `cv::solve(DECOMP_SVD)` / `cv::eigen`.

### Minne og tråder

- `cv::Mat` eier alt bildeminne; ingen egne allokatorer.
- Én analyse = én tråd (OpenCVs interne parallellisme får stå på).
  Analysen er en engangsjobb per capture — ikke latenskritisk.
- FrameProbe (auto-capture) er derimot per-frame på kamerastrømmen:
  den arbeider KUN på nedskalert bilde (~480px) og gjør ROI + skarphet +
  eksponering — budsjett ~10 ms på moderne mobil-CPU.
- Ingen exceptions over FFI-grensen: alt fanges i C-laget og mappes til statuskode.

### CV-kontrakt (kravspec §3)

`AnalyzeResult` (C-struct, stabil ABI):

- `status`: OK / REJECTED_NO_SCREEN / REJECTED_NO_RINGS / REJECTED_INVALID_TARGET /
  REJECTED_NO_HITS / ERROR_*  (Python-ValueError-ene mappet til enum)
- `hits[]`: per treff `{ r_rel, theta }` (relativ polar: r i ringavstander fra
  senter, theta i radianer) + `{x_px, y_px}` i inputbildet + `decimal`, `integer`
- `sum_decimal`, `sum_integer`
- `confidence` ∈ [0,1] + delfelter (`n_rings`, `ring_resid_frac`, `mean_hit_score`,
  `screen_ok`). Interim-heuristikk til OCR-sjekken kommer (kravspec-merknad).
- `timestamp_ms`: ekko av innsendt capture-tidsstempel (forskningskrav §6).

Inputformat: `BfImage { data, w, h, stride, format }`, format ∈ {GRAY8, BGR8,
RGBA8, NV21, YUV_420_888-planer}. Kjernen konverterer internt til BGR/gray.

**Rettelser 2026-08-07 (verifisert mot `types.h` og `bestefar_ffi.h`):**

- **Bare tre av konfidens-delfeltene krysser FFI-grensen.** `BfResult` har
  `confidence`, `n_rings` og `ring_resid_frac`. `mean_hit_score` og
  `screen_used` — feltet heter `screen_used`, ikke `screen_ok` — finnes kun i
  C++-structen `bestefar::Confidence` og er ikke synlige for klienten.
- **Fire pikselformater, ikke fem.** `BF_FMT_GRAY8 | BGR8 | RGBA8 | NV21`.
  «YUV_420_888-planer» er ikke et eget format: Androids `YUV_420_888` pakkes til
  NV21 før den sendes inn.
- **`BF_MAX_HITS` er 32.** Resultatstructen har en fast treffarray, ikke en
  liste. Taket sto ikke i kontrakten før.
- **`bf_version()` finnes** (lagt til 2026-08-06): returnerer en semver-streng
  fra `core/include/bestefar/version.h` (`BESTEFAR_CORE_VERSION`), bevisst
  UAVHENGIG av appens `versionName`. Bump den når en endring i `core/` påvirker
  analyse eller auto-capture. Strengen er statisk og eid av kjernen — ikke
  `free()` den. Formålet er å vite hvilken kjerne som produserte en §6-donasjon
  (`backend_spec.md` §8).
- **`BF_REJECTED_NO_SCREEN` emitteres aldri.** Koden er definert i `types.h`,
  men `analyze_target` returnerer den ikke: finner ikke `rectify_to_screen` en
  skjerm, faller vi alltid gjennom til helbilde-analyse. Se `docs/flytskjema.md`
  §1b for hele statustabellen.

### MPI-firkant (cluster-senter) — beslutning

**Ikke wiret inn i denne iterasjonen.** Begrunnelse (data fra 2026-07-04/05-øktene):
signalet er for svakt til å stå alene (NCC 0.24–0.36, ikke separerbart fra
ringer/tall); med tyngdepunkt-prior følger deteksjonen prioren og gir null
uavhengig informasjon; målt offset ≤0.15·delta på ALLE C-bilder inkl. C3 (som
har et faktisk skjult treff). Marginalverdien er dermed ~0 på tilgjengelig data,
i tråd med kravspec §3 («prioriteres ikke nå»). Resultat-skjemaet har et
`aux`-utvidelsespunkt slik at markøren kan legges til senere uten ABI-brudd.
Referanseimplementasjon beholdes i `_vis_mpi_square.py`.

*Merk 2026-08-07:* `aux`-utvidelsespunktet er en kommentar i `types.h`
(`AnalyzeResult`), ikke et felt. C-structen `BfResult` har ingen tilsvarende
åpning, så et senere tillegg vil kreve en ABI-beslutning der.

## Auto-capture (kravspec §4)

To adskilte vurderinger, to adskilte klasser:

1. **`FrameProbe`** (per frame, i kjernen): gjenbruker kontrast-ROI-en fra
   screen-modulen på nedskalert frame. Returnerer
   `{ roi_found, quad[4], sharpness (Laplacian-varians i ROI), exposure_lo/hi
   (histogram-klipp-andeler), coverage (quad innenfor frame m/margin) }`.
2. **`AutoCapture`** (tilstandsmaskin, i kjernen, plattformnøytral): mates med
   FrameProbe-resultater; trigger når (a) quad-hjørnene har holdt seg innenfor
   `stability_max_move_frac` i `stability_frames` påfølgende frames, og
   (b) kvalitetskravene er oppfylt i samme vindu.

**Alle terskler ligger i `AutoCaptureParams` og er merket UKALIBRERT** —
startverdier er satt for å være konservative og SKAL kalibreres mot faktisk
maskinvare (kravspec-krav; ikke funnet på fra teori). Se `AAPNE_PUNKTER.md`
ÅP-K1.

**Rettelser 2026-08-07 (verifisert mot `bestefar_ffi.h`):**

- **FrameProbe har vokst.** Feltene heter `clip_lo_frac`/`clip_hi_frac`, ikke
  `exposure_lo/hi`, og structen har i tillegg `stable`, `quality_ok`,
  `should_capture`, `screen_width_frac`, `size_ok`, `bull_width_frac`,
  `glare_frac` og `glare_ok`. Størrelses- og gjenskinnsportene kom til etter
  felttestene i 2026-07 og er en del av trigger-beslutningen, ikke bare
  diagnostikk.
- **Holdevinduet er 6 frames, ikke 24.** `config.h:164` har
  `stability_frames = 6` etter capture-first-omleggingen i v0.12
  (`docs/flytskjema.md` §1a): bildet tas i det øyeblikket kriteriene er oppfylt,
  og UI-et spilles av etterpå. De 24 fantes bare fordi den gamle flyten glødet
  grønt *før* capture, så ett dårlig frame nullstilte vinduet.
  **Kommentaren i `bestefar_ffi.h:75` sier fortsatt «default 24 (~0.8 s
  holdevindu)» og er feil.** Meldt som issue med label `kjerne` 2026-08-07 —
  headeren er kjernens kode og rettes der, ikke herfra.

## FFI-strategi

- **Ett rent C-header** (`bestefar_ffi.h`): opaque handle, POD-structs, ingen
  C++-typer. Kjernen bygges som statisk lib; C-laget som tynt shim.
- **Android:** JNI-bro (`android/app/src/main/cpp/jni_bridge.cpp`) mot C-API-et.
  Kotlin-datastrukturer speiler C-structene. Bygges med NDK + CMake
  (`externalNativeBuild`), OpenCV Android SDK som prefab/AAR-avhengighet.
- **iOS:** samme C-header via modulemap i et XCFramework; Swift kaller C direkte.
  (Stub i denne iterasjonen.)

`jni_bridge.cpp` ligger fysisk i UI-området, men er forbruker av dette
API-et. Endres `bestefar_ffi.h`, eier kjernen endringen og UI følger etter —
meld issue begge veier (rot-`CLAUDE.md` §1).

## Verifisering av porten

```
.venv\Scripts\python.exe verify_cset.py            # Python-fasit: 10/10
core/build/bestefar_cli Testsett/C1.jpg            # C++-kjerne, JSON ut
.venv\Scripts\python.exe verify_cset_cpp.py        # samme oracle mot CLI: krav 10/10
```
