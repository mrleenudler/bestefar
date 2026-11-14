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

**Rettelse 2026-08-21 (issue #12 — funnet på enhet, meldt av UI-instansen):**

`theta`s fortegnskonvensjon sto udokumentert (`bestefar_ffi.h:39` sa bare
«radianer»). UI antok matematisk konvensjon (y oppover) i god tro og
speilvendte `TargetView`: et treff over senter ble tegnet under. Målt mot
`bestefar_cli Testsett/C1.jpg` med senter og skala som ukjente (minste
kvadrat, begge fortegn):

| Antatt konvensjon | RMS | k (px/ringsteg) |
|---|---|---|
| y **nedover** (bildekoordinater) | **1,2 px** | **91,9** |
| y oppover (matematisk) | 70,5 px | 58,4 |

Kalibrert `delta_px` for bildet var 94,0 — bare den første stemmer, og
skalaen faller ut som uavhengig bekreftelse på samme konvensjon. Kjernen
gjorde alltid det riktige (`scoring.cpp:13`: `atan2(dy, dx)` med
`dy = y − calib.center.y` i bildekoordinater); feilen lå i rendereren og er
rettet der (`android/CHANGELOG.md` v0.30). **`theta = atan2(y_px − cy,
x_px − cx)`, samme aksekonvensjon som `x_px`/`y_px`** (x mot høyre, y
nedover — ikke matematisk): et treff over senter har `sin(theta) < 0`. Nå
skrevet inn i `bestefar_ffi.h`.

**Presisering utover det issue #12 spurte om, funnet ved samme gjennomgang:**
«samme aksekonvensjon» er ikke det samme som «samme ramme». `r_rel`/`theta`
regnes i **analyserammen** — `score_hit` kalles på `calib`/hit-koordinater
*før* de transformeres tilbake (`analyze.cpp:80–91`) — mens `x_px`/`y_px` er
det **samme punktet transformert gjennom den inverse** av skjerm-/
perspektivrettingen (`transform_point_inverse`, to steg: perspektiv i
`analyze_core`, deretter skjerm-crop i `analyze_target`). Retting er
`cv::warpPerspective` — en generell projeksjon, ikke vinkelbevarende — så de
to rammene er **numerisk identiske kun når ingen retting ble utført** (typisk
for et rett-forfra-bilde, og for et allerede rektifisert testbilde som
`C1.jpg` — derfor stemte målingen over så godt). For et skråtthold-bilde der
rettingen faktisk gjør noe, divergerer de. **Ikke rekonstruer `x_px`/`y_px`
fra `(senter, r_rel, theta)`** — de er to uavhengige avlesninger av samme
treff, ikke et polar/kartesisk-par i én ramme. Ingen konsument gjør dette i
dag (`TargetView` bruker bare `r_rel`/`theta` mot sitt eget canvas-senter,
ikke `x_px`/`y_px` — se `android/app/src/main/java/no/bestefar/app/Views.kt`),
så dette er en presisering av kontrakten, ikke en feil noen har truffet.

**`BfFrameProbe.quad` hadde samme klasse mangel** (ingen akse-/rammeangivelse
utover «frame-koordinater») og er rettet i samme runde: x mot høyre, y
nedover, i **framen slik klienten sendte den inn** til
`bf_autocapture_feed` — ikke den nedskalerte `probe_max_side`-oppløsningen
proben arbeider på internt. `frame_probe()` skalerer boksen tilbake før den
rapporteres (`autocapture.cpp:142–147`); det var allerede riktig i koden, bare
usagt i kontrakten.

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
.venv\Scripts\python.exe _probe_robustness_cpp.py  # perturbasjonsmatrise (se under)
```

## Robusthetsrunden 2026-11-11 (kjerne 1.1.0)

Ingen nye bilder fantes; alt under er målt mot `Testsett/` og syntetiske
perturbasjoner/negativer. Tre ting ble endret, og tre ting ble prøvd og
forkastet — de forkastede står her fordi de ellers blir foreslått igjen.

### Støy-vern i treffdeteksjonen (endret, begge implementasjoner)

`Testsett/T9.jpg` (bevegelsesuskarpt apparatfoto) avslørte et hull:
C++-kjernen kalibrerte «ringer» i smøret (delta=40,6, 5 ringer, gate OK) og
returnerte **140 treff, sum 900, status OK, confidence 0,50**.
Python-referansen avviste samme bilde — men bare fordi dens
subsample-lotteri i ytre-sirkel landet et annet sted; det var flaks, ikke
vern. Målt skille:

| | kandidater fra stemmekartet | stemmer per topp / marker_r |
|---|---|---|
| gyldige bilder (C/T-sett) | 3–10 | 0,72–2,44 |
| T9 | 200 | 0,63–1,40 |

Per-topp-styrke overlapper — intet absolutt gulv kan skille støy fra ekte.
**Antallet skiller perfekt**, og det er også fysisk begrunnet: en
Kongsberg-serie er maks 20 skudd (panelrad 11–20 finnes, se `Ta.jpg`).
Derfor: `hit_max_candidates = 40` (kandidater fra stemmekartet) og
`hit_max_hits = 24` (etter dedup/overlapp-pass); brudd →
`REJECTED_INVALID_TARGET` («treffdeteksjonen fant stoy, ikke markoerer»).
Implementert i både `hits.py` (ValueError) og `hits.cpp`
(`noise_rejected`-utparameter). Eneste endrede verdikt i hele Testsett:
T9 OK→avvist. 9 ikke-skive-negativer (beskåret bakgrunn + syntetisk
vitrineskap/sofa): 0 fantomtreff. Bieffekt: `BF_MAX_HITS`-trunkeringen (32)
kan ikke lenger inntreffe i praksis (24 < 32) — se `core/KONTRAKT.md` §3.

### Perturbasjonsmatrisen (`_probe_robustness_cpp.py`)

Syntetiske varianter av C-settet (perspektivvipp ±10/±20°, rotasjon,
nedskalering, gjenskinnsflekk, lysgradient, gamma, defokus) kjørt mot
C++-CLI-en og skåret mot samme fasit. Perturbasjonen endrer ikke fasiten.
Status 2026-11-11: **113/150**; rotasjon ±15° og gamma 1,6 er 10/10;
svakest er vipp 20° (2–6/10) og halv skala (4/10). Det viktige er *hvordan*
det feiler: nesten alle avvik er trygge avvisninger eller ±1 treff — ikke
gale summer. Matrisen er referansen enhver robusthetsendring skal måles mot.

### Overlapp-passet har et gulv for markørstørrelse (endret, begge impl.)

Matrisen avslørte at halv skala ga **+1 fantomtreff** på fire bilder
(C4/C6/C7/C10: 6 treff der fasit er 5), alle fra overlapp-passet. Målt: ved
marker_r ≈ 19 px scorer falske NCC-treff 0,28–0,50 — uskillelig fra C3s
ekte skjulte treff (0,47 ved marker_r ≈ 39 px, der falske målte ≤ 0,12) —
og heller ikke avstand til anker skiller (ekte 0,59·marker_r, fantomer
0,26–0,57). Samtidig finner passet *ikke* det ekte skjulte treffet på den
skalaen (C3 halv skala mangler det uansett): templaten har for få piksler
til å være diskriminerende. Passet bidro altså kun med fantomer der. Ny
`hit_overlap_min_marker_px = 24` (≈ delta 58) hopper over passet under
gulvet. Effekt: halv skala 4/10 → 7/10, null endring ellers, begge orakler
10/10. Sentrum-sveipet beholdes uendret (dets 0,96-NCC er ekte treff, og en
større eksklusjonsradius ville truet C5s tette 10,2-par).

### Forkastet etter måling (ikke gjør om igjen uten nye data)

1. **«Beste iterat»-fallback i `calibrate_and_refine`** (bruk iteratet med
   flest sporede ringer når en senere iterasjon feiler): reddet C6 +20°
   korrekt, men netto null på matrisen og konverterte trygge avvisninger til
   **feilsummer** (C1 tilt_x+20: 4 treff sum 37, fasit 5/46; C9 gamma0,6:
   23 treff sum 118, fasit 3/25 — like under 24-taket). En gal sum er verre
   enn en avvisning.
2. **Ellipsitets-adaptivt sporingsbånd** (grunnbånd + k2-amplitude·rho):
   skapte feilsummene C2/C7 tilt_x+20 via halv-avstands-løsninger som
   passerte gaten, og ødela kalibreringer som konvergerte før (C1 rot-15,
   C5 tilt_y-10, C3 gamma0,6 — alle korrekte uten, avvist med).
3. **Delta-kontinuitet mellom iterasjoner** (behold forrige delta når
   autokorrelasjonen hopper >30 %): blokkerte også *legitim* gjenoppretting
   fra et halv-avstands-førstegjett — Python-C9 falt fra PASS til avvist.

Ringløkka er derfor bevisst identisk med Python-referansen. Lærdommen på
tvers: på marginale bilder er gaten (`validate_calibration`) ikke sterk nok
til å skille en reddet-men-gal kalibrering fra en ekte — så mekanismer som
«redder» kalibreringer flytter feil fra synlig (avvist) til usynlig (gal
sum). Vipp >15° forblir en kjent svakhet; riktig vei videre er trolig å
gjøre *gaten* skarpere (f.eks. bull-størrelse mot delta), ikke å redde
flere kalibreringer.

### Auto-capture: størrelsesgatene rekalibrert + innholdsport (ny)

Feltobservasjonene i ÅP-K1 (vitrineskap utløste; skjermbilde av skive
utløste aldri) fikk målbare forklaringer:

- **Falsk negativ:** `min_screen_width_frac = 0,90` avviste 7 av 10
  C-bilder som analysen scorer feilfritt (målt 0,59–0,96). Satt til
  **0,55**. `min_bull_width_frac` 0,10 → **0,07** (tre feltfoto med bull
  0,08–0,09 analyserer OK; 0,07 er den geometriske nedre grensen som alt
  sto i kommentaren).
- **Falsk positiv:** ingen gate så på *innhold* — kvalitetsgatene (skarphet/
  eksponering/gjenskinn) passerte samtlige 56 Testsett-bilder OG alle
  negativer. Ny port: `ring_evidence` = antall smale, sammenhengende avvik
  fra glattet trend i radialprofilen (vinkelsnitt over 48 retninger) rundt
  bull-komponentens sentroid; `content_ok` krever ≥ `min_ring_evidence`
  (2,0, UKALIBRERT). Er bull-komponenten > 0,35 av framebredden er den ikke
  en bull men en mørk skivevariant (Ta/Tc/Td) eller noe annet — da skannes
  komponentens *indre* rundt samme sentroid.

Målt på hele Testsett + 9 negativer: alle 44 sanne apparatskjermer passerer
innhold+størrelse unntatt tre mørke/små randtilfeller (Tc, Td,
PXL_…164413401) som faller til klientens 8-sekunders tidsgrense; av
negativene er det **kun den syntetiske vitrineskap-mocken som fortsatt
passerer alle gater** — porten krymper vitrineklassen, den lukker den ikke.
Feltkalibreringen i ÅP-K1 står.

**Forkastet underveis (målt):** (a) radialprofil rundt *skjermblob-senteret*
som fallback — telte tabell-/bezelkanter og ga ring ≥ 2 på ALT, inkl.
reinsdyr og romfoto; sentroid-forankringen er det som gjør målet
ringspesifikt. (b) *Median* over vinkler i stedet for snitt — teoretisk
riktigere mot sektorstrukturer, men en fullskutt bull (C3, fasitbilde) har
markører som bryter ringlinjene i mange retninger, og medianen mistet
ringene der (ring=1) uten å stoppe vitrineskapet bedre.

FFI: `BfFrameProbe` fikk `ring_evidence`/`content_ok`, `BfAutoCaptureParams`
fikk `min_ring_evidence` — begge bakerst i structen (se `core/KONTRAKT.md`
§4b). `BESTEFAR_CORE_VERSION` bumpet til 1.1.0.

### Kutt-porten: skjermblobben skal ikke beroere framekanten (2026-11-12)

Eier meldte at T-settet scoret galt, trolig fordi bezelen manglet paa en
kant. Panel-fasiten ble lest fra fotoene (`hits_truth_tset.txt`, IKKE
eier-verifisert) og bekrefter det: **alle gale/avviste T-resultater (T2
avvist, T4 7/63 mot fasit 9/82, T6 55 mot 54) er foto der skjermen kuttes
av framekanten; begge helt innrammede (T7 10/86, T8 7/66) scorer riktig.**
Mekanismen: skjermdeteksjonen snapper fire kantlinjer, og en side uten
bezel har ingen kant aa snappe til — crop-en blir gal, og resultatet kan
bli en gal sum, ikke bare en avvisning.

`coverage`-maalet kan ikke fange dette: en klippet blob gir en klippet
boks, og boks-mot-margin-snittet ser fint ut (T2 maalte dek=0,98). Det
paalitelige signalet er at blobben BEROERER framekanten (<=1 px paa
probe-skala): en komplett skjerm har alltid moerk bezel mellom lys blob og
kant. Nytt probefelt `edge_cut`, inngaar i `quality_ok`; param
`block_edge_cut` (default paa) for feltkalibrering.

Prisen, maalt paa Testsett: porten blokkerer ogsaa stramt-men-komplett
innrammede foto (bl.a. C2/C3/C8/C9/C10, som analyserer perfekt) — 20 av 56
bilder trigger naa, mot 52 foer. Det er villet: testfotoene er haandtatt
uten skannerramma, og i live bruk er svaret at brukeren rygger litt til
hele skjermen er inne — alternativet er en capture som kan gi gal sum.
Statisk enkeltbilde kan uansett ikke maale stabilitetsvinduet.

### Kutt-porten v2: maalomraadet i framen, ikke hele skjermen (2026-11-13)

v1 over levde ett doegn. Eier meldte tilbake at C-blokkeringene var for
strenge (tape paa skjermen og gjenskinn kan koble blobben til framekanten,
og stram innramming er normal), og foreslo en bedre invariant: **bullkanten
er 7-ringen og 1-ringen ligger paa en fast ratio fra samme senter** — bruk
den til aa regne ut en riktig margin. Det er fast geometri (bull = 5,7δ i
diameter → r_bull = 2,85δ; 1-ringen = 10δ = 3,51·r_bull), ikke en tunbar
terskel.

v2: `edge_cut` settes naar sirkelen 3,5·r_bull rundt bull-sentroiden gaar
utenfor framen. Begrunnelsen er skarpere enn v1s: gaar 1-ring-sonen utenfor
fotoet, kan en markoer der ligge HELT utenfor bildet — et manglende treff i
summen uten noen feilmelding aa oppdage det paa.

To attribusjonskorrigeringer fra eier-verifiseringen av panel-fasiten
(`hits_truth_tset.txt`, 2026-11-14) hoerer til her: T4 har et HELT skjult
9,1-treff, saa detekterbart maal er 8/73 — kjernens 7/63 mangler ett treff
i 10,x-klyngen (C3-klassen overlapp, ikke noedvendigvis kuttet), og T6
(+1 poeng) er presisjon. Det kausalt sikre kutt-tilfellet er T2, der selve
ringomraadet er kuttet (→ NO_RINGS). v1s «alle gale/avviste er kuttede
foto» var korrelasjon, ikke aarsak — derav v2s maalomraade-formulering.

Maalt klarering (avstand bullsenter → naermeste kant, i r_bull): C2 3,62 og
C10 5,66 slipper; T3 3,49 og T5 2,90 blokkeres — de scoret riktig kun fordi
treffene satt sentralt — og T2 2,60 blokkeres. Resultat: **alle 10
C-bilder trigger igjen** (26 av 56 totalt), T2 forblir blokkert, og som
sidegevinst blokkeres reinsdyr- og moerkvariant-skjermene (bull-
segmenteringen gir der en diger «bull», og 3,5·r gaar alltid utenfor) —
flater kjernen uansett ikke kan score. Uten funnet bull felles ingen
kutt-dom; stoerrelsesgaten tar det tilfellet.
