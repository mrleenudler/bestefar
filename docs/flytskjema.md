# Flytskjema

To flyter: **CV-kjernen** (`core/`, C++17 bak en ren C-FFI) og **UI-et**
(`android/`, Kotlin med programmatiske views). De møtes ett sted — JNI-kallet
`BestefarCore.analyze()` i `CaptureActivity`.

Diagrammene er avledet fra koden slik den står i v0.17, ikke fra spesifikasjonen.
Stadienavn er de faktiske funksjonsnavnene, så de kan søkes opp direkte.

---

## 1. CV-kjernen

### 1a. Auto-capture — `core/src/autocapture.cpp`

Kjører på hver live-frame fra CameraX, ~10 ms budsjett. Alle terskler er
**ukalibrerte** startverdier (kravspec §4).

```mermaid
flowchart TD
    A["Live-frame fra CameraX<br/>YUV_420_888 · Y-planet = gråbilde"] --> B["Nedskaler til 480 px<br/>probe_max_side"]
    B --> C["apparatus_roi<br/>lokalt std-kart, hindrer lekkasje til bakgrunn"]
    C --> D{"ROI funnet?"}
    D -->|"nei"| T1["«Beveg telefonen slik at<br/>skjermen passer i rammen»"]
    D -->|"ja"| E["screen_blob<br/>hysterese-terskling + morph-close"]
    E --> F["Måling per frame"]
    F --> G1["skarphet<br/>Laplacian-varians"]
    F --> G2["eksponering<br/>clip_lo_frac / clip_hi_frac"]
    F --> G3["dekning<br/>coverage"]
    F --> G4["gjenskinn<br/>glare_frac"]
    F --> G5["størrelse<br/>screen_width_frac · bull_width_frac"]
    G1 --> Q{"quality_ok"}
    G2 --> Q
    G3 --> Q
    G4 --> Q
    G5 --> S{"size_ok"}
    Q --> H["Historikk over 6 frames<br/>stability_frames"]
    S --> H
    H --> I{"Stabil OG kvalitet i<br/>alle 6 frames?"}
    I -->|"nei"| T2["Statushint:<br/>gjenskinn · lys/fokus"]
    I -->|"ja"| J["should_capture = 1"]
    J --> K["takeStillAndAnalyze<br/>bildet tas STILLE"]
    K --> L["Grønn ramme + «Klar!» 0,4 s<br/>→ hvit blits"]
    K --> M["→ analyse, se 1b"]
```

**Capture-first** (felttest skytebanen, v0.12): bildet tas i det øyeblikket
kriteriene er oppfylt, og UI-et spilles av *etterpå* mens analysen allerede
kjører. Holdevinduet ble kuttet fra 24 til 6 frames — 24 fantes bare fordi den
gamle flyten glødet grønt *før* capture, og ett dårlig frame nullstilte vinduet.

### 1b. Analyse — `core/src/analyze.cpp`

```mermaid
flowchart TD
    IN["bf_analyze BfImage<br/>GRAY8 · BGR8 · RGBA8 · NV21"] --> E0{"Tomt bilde?"}
    E0 -->|"ja"| ERR["ERROR_BAD_INPUT"]
    E0 -->|"nei"| SR["rectify_to_screen"]

    subgraph SCREEN ["Finn og rett ut apparatskjermen · screen.cpp"]
        direction TB
        S1["apparatus_roi<br/>kontrast-ROI, 480 px grå, σ = 3,5"]
        S2["screen_blob<br/>hysterese + morph-close + største komponent"]
        S3["konveks innhylling<br/>fyller konkaviteten fra bro-lekkasjer"]
        S4["rough_quad<br/>approxPolyDP → TL, TR, BR, BL"]
        S5["refine_from_contour<br/>IRLS: Huber → Tukey, hjørner = linjeskjæring"]
        S6["snap_line_to_edge × 4<br/>YTTERSTE sterke gradient, ikke argmax"]
        S7["Perspektivwarp → skjermutklipp"]
        S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7
    end

    SR --> S1
    SR -.->|"skjerm ikke funnet"| FULL["Helbilde-analyse<br/>samme kjerne på hele fotoet"]
    S7 --> C1
    FULL --> C1

    subgraph CORE ["analyze_core · kalibrering, treff, poeng"]
        direction TB
        C1["detect_outer_circle<br/>gradient-stemming, ALLE kantpunkter, 1M par"]
        C2["calibrate_and_refine<br/>polarwarp → radiell gradient → autokorr-spacing<br/>→ ringprogresjon → harmonisk tilpasning"]
        C3["validate_calibration"]
        C4["fit_rectification + rekalibrering<br/>beholdes kun hvis den nye også er gyldig"]
        C5["detect_hits<br/>circles.cpp over HELE skiva, NCC + overlap"]
        C6["score_hit per treff<br/>r_rel · theta · decimal · integer"]
        C7["Konfidens<br/>n_rings · ring_resid_frac · mean_hit_score"]
        C1 --> C2 --> C3 --> C4 --> C5 --> C6 --> C7
    end

    C1 -.->|"ingen kantpunkter"| R1["REJECTED_NO_RINGS"]
    C2 -.->|"ingen poengringer"| R1
    C3 -.->|"ikke gyldig skive"| R2["REJECTED_INVALID_TARGET"]
    C5 -.->|"ingen treff"| R3["REJECTED_NO_HITS"]
    C7 --> OK["BF_OK<br/>treff sortert synkende + sum + konfidens"]
    OK --> BACK["transform_point_inverse<br/>koordinater tilbake til originalfotoet"]
```

**Statuskoder** (`bestefar_ffi.h`) — UI-et bruker `status != OK` som det harde
signalet «dette kan jeg ikke score», og det er dette som utløser re-scan-dialogen.

| Kode | Verdi | Utløses av |
|---|---|---|
| `BF_OK` | 0 | Gyldig kalibrering og minst ett treff |
| `BF_REJECTED_NO_SCREEN` | 1 | *Definert, men ikke emittert* — se merknad |
| `BF_REJECTED_NO_RINGS` | 2 | `detect_outer_circle` eller `calibrate_and_refine` feilet |
| `BF_REJECTED_INVALID_TARGET` | 3 | `validate_calibration` avviste skiva |
| `BF_REJECTED_NO_HITS` | 4 | Ingen treff funnet (`gate_require_hits`) |
| `BF_ERROR_BAD_INPUT` | 100 | Tomt bilde |
| `BF_ERROR_INTERNAL` | 101 | Uventet exception |

**Merknad om `NO_SCREEN`:** koden er definert i `types.h`, men `analyze_target`
returnerer den aldri. Finner ikke `rectify_to_screen` en skjerm, faller vi
*alltid* gjennom til helbilde-analyse. Flagget `analyze_screen_fallback`
(default `false`) gjelder kun tilfellet «skjerm funnet, men crop-analysen
feilet» — da returneres crop-resultatets egen avvisningskode.

---

## 2. UI-flyten

```mermaid
flowchart TD
    subgraph OPPSTART ["Oppstart · MainActivity"]
        direction TB
        START(["App-start"]) --> W1{"Introvindu sett?<br/>eller dev-flagg?"}
        W1 -->|"vis"| I1["Oppstartsvindu"]
        W1 -->|"hopp over"| W2
        I1 --> W2{"Spør om bildedeling?<br/>første gang · ny sesong · dev-flagg"}
        W2 -->|"ja"| I2["«Vil du dele bilder der<br/>appen gjør en dårlig jobb?»"]
        W2 -->|"nei"| TUT
        I2 --> TUT{"Veiledning sett?"}
        TUT -->|"nei"| I3["Veiledning · 4 steg"]
        TUT -->|"ja"| HOME
        I3 --> HOME
    end

    HOME["Hovedskjerm<br/>Avstand · Innsikt · Meny + Scan"]

    subgraph SCANFLYT ["Scan-flyten"]
        direction TB
        SCAN["CaptureActivity<br/>eneste LIGGENDE skjerm"]
        SCAN --> GALS["Lagre JPEG i Bilder/Bestefar<br/>hvis saveScansMode ≠ ALDRI"]
        GALS --> ANA["BestefarCore.analyze — JNI"]
        ANA --> SIDE["JSON-sidecar i app-mappen"]
        SIDE --> RES["Resultatkort · ResultActivity"]
        RES --> ST{"status"}
        ST -->|"avvist"| RSC["«Bildet ble ikke korrekt analysert.<br/>Scan bildet på ny.»<br/>Avbryt · Scan på ny<br/>+ ⌷ Send bildet til feilanalyse"]
        ST -->|"OK"| GAL{"Første scan?"}
        GAL -->|"ja"| GALQ["«Ønsker du at skjermbildene skal<br/>lagres i bildearkivet ditt?»<br/>Nei · Alle · De beste<br/>→ 🎛 «Kan endres i Avanserte innstillinger»"]
        GAL -->|"nei"| POS
        GALQ --> POS["Stillingsvelger<br/>liggende · sittende · knestående · stående<br/>+ anlegg / reim"]
        POS --> SI{"Ser innskutt skjevt ut?<br/>Stats.looksMiscalibrated"}
        SI -->|"ja"| SIQ["«Er dette innskyting?»<br/>kan forkaste serien"]
        SI -->|"nei"| OCR
        SIQ --> OCR["OcrVerifier · ML Kit<br/>leser apparatets poengliste"]
        OCR -->|"Match ≤ 0,2"| P1["OCR-poeng i SKJERMREKKEFØLGE"]
        OCR -->|"Inconclusive"| P2["Detekterte poeng, stigende"]
        OCR -->|"Mismatch > 0,2"| P3["OCR øverst · «Identifiserte treff» nederst<br/>Forkast · Lagre leste poeng"]
        SAVE{"Lagre"}
        P1 --> SAVE
        P2 --> SAVE
        P3 --> SAVE
        SAVE --> DUP{"Lik forrige serie?<br/>poeng < 0,05 OG treffpunkter ≤ 0,1 p"}
        DUP -->|"ja"| DUPQ["«Lik forrige serie»<br/>lagre likevel?"]
        DUP -->|"nei"| STORE
        DUPQ --> STORE["Store.addSeries → series.json"]
        STORE --> JM{"Tredje serie?"}
        JM -->|"ja"| JMD["«Mitt jaktmål»"]
    end

    HOME --> SCAN
    RSC --> SCAN
    JM -->|"nei"| HOME
    JMD --> HOME

    subgraph NAV ["Navigasjon"]
        direction TB
        DIST["Avstand<br/>nedtrekk"]
        INS["Innsikt<br/>7-raders matrise:<br/>stilling × vilt × vinkel × hold"]
        MENY["Meny<br/>nedtrekk"]
        MENY --> M1["Min profil"]
        MENY --> M2["Jakt"]
        MENY --> M3["Venner"]
        MENY --> M4["Serier · Serielogg"]
        M4 --> TRD["📈 Trendgraf øverst<br/>x = dato, 2 jaktår · y = 20-skudds snitt<br/>ingen framskriving"]
        MENY --> M5["Send melding"]
        MENY --> M6["Veiledning"]
        M1 --> M1a["Mine lag"]
        M1 --> LOG{"Konto"}
        LOG -->|"utlogget"| LOGA["Fortsett med Google<br/>Credential Manager · SIWG"]
        LOG -->|"utlogget"| LOGB["Kode på e-post<br/>/email/start → /email/verify"]
        LOG -->|"innlogget"| LOGC["Navn · Bruker-ID · Logg ut"]
        LOGA --> LOGT["POST /v1/auth/google<br/>→ egne tokens i Secrets"]
        LOGB --> LOGT
        M1 --> M1b["🎛 Avanserte innstillinger<br/>våpen · bildearkiv Aldri/Alle/De beste ·<br/>bildedeling · venstrehånd ·<br/>sikkerhetskopi · utviklermeny"]
        M1b --> M1c["Sikkerhetskopi<br/>Sikkerhetskopier nå · Gjenopprett ·<br/>Vis gjenopprettingskode (nødutgang)"]
        M1b --> M1d["🔑 Gjenopprett uten kode (av)<br/>🔒 Krev opplåsing for jaktloggen (av)"]
        M2 --> LOCK{"🔒 Krev opplåsing?<br/>av som standard"}
        LOCK -->|"på"| BIO["BiometricPrompt<br/>biometri ELLER skjermlås<br/>5 min frist · avvist = bli stående"]
        LOCK -->|"av"| M2a
        BIO --> M2a["Logg jaktskudd<br/>«Detaljert visning» av → enkel side<br/>på → to sider"]
        BIO --> M2b["Se registrerte skudd → Rediger"]
        LOCK -->|"av"| M2b
        M2a --> ANN{"Vellykket felling?"}
        ANN -->|"ja"| ANNP["Forhåndsvis varselet<br/>«Ola har felt et villsvin i Molde.»<br/>Send · Ikke send"]
        M3 --> M3a["Lagside"]
    end

    HOME --> DIST
    HOME --> INS
    HOME --> MENY
```

### Tre ting verdt å merke seg i flyten

- **OCR har presedens.** Finnes OCR-poeng, er det de som vises — usortert, i den
  rekkefølgen de står på apparatskjermen. Totalen regnes av det som vises, så
  liste og sum aldri spriker.
- **To uavhengige kvalitetsporter.** `status != OK` fanger «jeg fikk ikke
  kalibrert skiva»; OCR-uenighet fanger den vanskeligere klassen der analysen
  *lyktes* men leste feil — typisk opp-ned-bildene.
- **Alt er offline.** `Store` skriver `series.json` / `hunts.json` i appens egen
  filkatalog. Appen virker uten konto; venner/lag er front-end-skjelett (se
  `backend_spec.md`). Nett brukes kun til feilanalyse-køen og sikkerhetskopien.

### Nytt i v0.15

- **Sletting er soft-delete.** `deleteSeries`/`deleteHunts` setter `deletedAt`
  i stedet for å fjerne raden. Visningskoden ser ingen forskjell
  (`allSeries()`/`allHunts()` filtrerer), men `…Raw()` beholder gravsteinen så
  en gjenoppretting ikke legger inn igjen det brukeren har slettet.
- **Sikkerhetskopien er klient-kryptert.** `Backup.build()` → AES-256-GCM med
  nøkkel utledet fra en generert gjenopprettingskode; serveren lagrer bytes den
  ikke kan lese. Mister brukeren koden, er kopien tapt — og det står i dialogen.
- **🎛-ikonet.** Overalt der «Avanserte innstillinger» nevnes ligger det et
  equalizer-ikon som åpner siden direkte (`Ui.advancedIcon`).

### Nytt i v0.16

- **Nøkkelen til kopien søkes opp i tre lag** (`BackupKeys.resolve`): lokalt →
  Block Store → deponering hos serveren. Brukeren blir bedt om
  gjenopprettingskoden *bare* når ingen av dem har noe. Derfor spør
  «Sikkerhetskopier nå» ikke lenger om noe i det hele tatt.
- **Jaktloggen kan låses**, resten av appen aldri. Låsen ligger foran begge
  inngangene i Jakt-menyen, med fem minutters frist. Avvist opplåsing lukker
  ingenting — brukeren blir stående.
- **Felling-varselet forhåndsvises.** Setningen vennene får bygges i klienten
  (`Announce.speciesPhrase` eier bøyningen) og vises ordrett før den sendes.
- **Utlogging** (`Auth.logout`) er tre steg i fast rekkefølge: avregistrer
  enheten for push → tilbakekall refresh-tokenet → slett begge tokenene lokalt.
  Siste steg skjer uansett, også offline.

### Nytt i v0.17

- **Innlogging finnes.** Min profil → Konto → `LoggInnActivity`. Google via
  Credential Manager, eller sekssifret kode på e-post. Appen ber aldri om
  innlogging uoppfordret, og skjermen sier eksplisitt at alt annet virker uten
  konto.
- **Ingen data går tapt ved inn- eller utlogging.** Serier, jaktlogg og
  innstillinger er lokale og røres ikke av noen av delene.
