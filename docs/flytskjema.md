# Flytskjema

Tre flyter: **CV-kjernen** (`core/`, C++17 bak en ren C-FFI), **UI-et**
(`android/`, Kotlin med programmatiske views), og **varselveien** mot backend.
De to første møtes ett sted — JNI-kallet `BestefarCore.analyze()` i
`CaptureActivity`.

Diagrammene er avledet fra koden slik den står i v0.19, ikke fra spesifikasjonen.
Stadienavn er de faktiske funksjonsnavnene, så de kan søkes opp direkte.

**Dette dokumentet beskriver appen slik den er nå.** Hva som ble bygget når,
står i `android/CHANGELOG.md` — ikke skriv «Nytt i v0.NN»-seksjoner inn her
igjen.

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
    A -.->|"første frame armer"| TO["Tidsgrense 8 s<br/>CaptureActivity · klientside"]
    TO -->|"gatingen har ikke utløst"| K
```

**Capture-first** (felttest skytebanen, v0.12): bildet tas i det øyeblikket
kriteriene er oppfylt, og UI-et spilles av *etterpå* mens analysen allerede
kjører. Holdevinduet ble kuttet fra 24 til 6 frames — 24 fantes bare fordi den
gamle flyten glødet grønt *før* capture, og ett dårlig frame nullstilte vinduet.

**Tidsgrensen** (v0.29) er den stiplede veien i diagrammet, og den er
**klientside** — `bf_analyze` er en egen FFI-inngang som tar piksler, så en
capture uten `should_capture` går til nøyaktig samme analyse uten å røre
tilstandsmaskinen i `autocapture.cpp`. Utløser ikke gatingen innen 8 sekunder
fra første gatede frame, tas gjeldende ramme likevel. Brukeren ser ingen
nedtelling og ingen forskjell: samme grønne ramme, samme blits.

Grunnen er at de to utfallene «gatingen slapp aldri noe gjennom» og «kjernen
kjørte og feilet» så helt like ut — begge ga *ingenting*. Lykkes analysen etter
en timeout, er tersklene for strenge (ÅP-K1); feiler den, er det en ordinær
feilet analyse. Donasjonen bærer `capture_trigger` i sidecaren for å skille
dem. **Serveren tar imot feltet fra 2026-08-22** (issue #11 lukket, B-53):
`capture_trigger` ∈ {`auto`, `timeout`} som eget multipart-felt ved siden av
`tag`. **Klienten sender det ikke ennå**, så inntil den gjør det står feltet
tomt på alle rader — og tomt betyr «ikke oppgitt», ikke `auto`.

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
        START -.->|"parallelt, med konto"| MSG["Messages.fetch<br/>GET /v1/messages"]
        W1 -->|"vis"| I1["Oppstartsvindu"]
        W1 -->|"hopp over"| W2
        I1 --> W2{"Spør om bildedeling?<br/>første gang · ny sesong · dev-flagg"}
        W2 -->|"ja"| I2["«Vil du dele bilder der<br/>appen gjør en dårlig jobb?»"]
        W2 -->|"nei"| TUT
        I2 --> TUT{"Veiledning sett?"}
        TUT -->|"nei"| I3["Veiledning · 4 steg"]
        TUT -->|"ja"| DONE
        I3 --> DONE["onStartupOverlaysDone"]
        MSG -.-> DONE
        DONE --> MQ{"Ventende beskjeder?"}
        MQ -->|"ja"| MQ1["Én om gangen · eldste først<br/>tittel · tekst · tidspunkt<br/>→ ack ETTER visning"]
        MQ -->|"nei"| HOME
        MQ1 --> HOME
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
        OCR -->|"Inconclusive"| MAX{"Flere enn 10<br/>detekterte treff?"}
        MAX -->|"nei"| P2["Detekterte poeng, stigende"]
        MAX -->|"ja"| TMH["«Appen fant N treff.<br/>En serie er høyst 10 skudd»<br/>Forkast · Scan på ny<br/>+ ⌷ Send bildet til feilanalyse<br/>serien kan ikke lagres"]
        OCR -->|"Mismatch > 0,2"| P3["OCR øverst · «Identifiserte treff» nederst<br/>Forkast · Lagre leste poeng"]
        OCR -->|"CountMismatch<br/>flere detekterte enn OCR-poeng"| P4["Over-deteksjon forklart<br/>Lagre leste poeng = overtallige treff fjernes"]
        OCR -->|"CountMismatch<br/>færre detekterte enn OCR-poeng"| P5["Skjulte treff forklart<br/>Lagre leste poeng = skuddet lagres<br/>uten posisjon, uten merke på skiva"]
        SAVE{"Lagre"}
        P1 --> SAVE
        P2 --> SAVE
        P3 --> SAVE
        P4 --> SAVE
        P5 --> SAVE
        SAVE --> DUP{"Lik forrige serie?<br/>poeng < 0,05 OG treffpunkter ≤ 0,1 p"}
        DUP -->|"ja"| DUPQ["«Lik forrige serie»<br/>lagre likevel?"]
        DUP -->|"nei"| STORE
        DUPQ --> STORE["Store.addSeries → series.json"]
        STORE --> JM{"Tredje serie?"}
        JM -->|"ja"| JMD["«Mitt jaktmål»"]
    end

    HOME --> SCAN
    RSC --> SCAN
    TMH --> SCAN
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
        LOGT --> LOGN{"is_new fra serveren?"}
        LOGN -->|"ja"| LOGB1["«Ta vare på loggen din?»<br/>etter varseldialogen"]
        LOGB1 -->|"Ja"| LOGB2["Vis gjenopprettingskoden<br/>avkryssing kreves · kan ikke avbrytes"]
        LOGB2 --> LOGB3["PUT /v1/backup"]
        LOGN -->|"nei"| LOGM["GET /v1/backup/meta"]
        LOGM -->|"404 · offline · 5xx"| LOGS["ingenting —<br/>ingen dialog uten en kopi å tilby"]
        LOGM -->|"200"| LOGR["«Kopien ble laget 7. august …<br/>dette ERSTATTER alt»"]
        LOGR -->|"Ja"| LOGK{"escrowed?"}
        LOGK -->|"ja"| LOGK1["nøkkelen hentes fra serveren<br/>ingen kode etterspørres"]
        LOGK -->|"nei"| LOGK2["lokalt → Block Store<br/>→ tast koden"]
        LOGK1 --> LOGRD["GET /v1/backup → dekrypter"]
        LOGK2 --> LOGRD
        M1 --> M1b["🎛 Avanserte innstillinger<br/>våpen · bildearkiv Aldri/Alle/De beste ·<br/>bildedeling · venstrehånd ·<br/>sikkerhetskopi · utviklermeny"]
        M1b --> M1c["Sikkerhetskopi<br/>Sikkerhetskopier nå · Gjenopprett ·<br/>Vis gjenopprettingskode (nødutgang)"]
        M1c --> RST["Gjenopprett:<br/>GET /v1/backup/meta først"]
        RST -->|"404"| RST0["«Ingen sikkerhetskopi funnet»<br/>stopper her, spør ikke om kode"]
        RST -->|"200"| RST1["«Kopien ble laget 7. august kl. 09:14.<br/>Dette ERSTATTER alt lokalt.»"]
        RST1 --> RST2["BackupKeys.resolve<br/>lokalt → Block Store → deponering"]
        RST2 -->|"fant nøkkel"| RST3["GET /v1/backup → dekrypter → replaceAll"]
        RST2 -->|"ingen kilde"| RST4["Tast gjenopprettingskoden"]
        RST4 --> RST3
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
- **Antallet er en egen kontroll, ikke en del av verdi-sammenligningen.** En
  serie er 0–10 skudd; kjernens `BF_MAX_HITS` (32) er en bufferstørrelse. Er
  antallet detekterte treff og antallet OCR-poeng ulikt, forteller *retningen*
  hvilken feil det er — flere detekterte er over-deteksjon (treffet finnes ikke,
  OCR kan ikke redde poengene), færre er skjulte treff (OCR-presedens gir riktig
  sum). Uten OCR-fasit er 10 en hard grense: serien kan ikke lagres.
- **Alt er offline.** `Store` skriver `series.json` / `hunts.json` i appens egen
  filkatalog. Appen virker uten konto; venner/lag er front-end-skjelett (se
  `backend_spec.md`). Nettet brukes til feilanalyse-køen, sikkerhetskopien,
  innloggingen og beskjedene — aldri til noe brukeren trenger for å skyte en
  serie.
- **Sletting er soft-delete.** `deletedAt` settes, raden blir stående.
  Visningskoden ser ingen forskjell (`allSeries()`/`allHunts()` filtrerer), men
  `…Raw()` beholder gravsteinen så en gjenoppretting ikke legger inn igjen det
  brukeren har slettet.
- **Beskjedene kommer sist i oppstarten, ikke oppå den.** Hentingen starter
  parallelt med oppstartsvinduene, men resultatet holdes til
  `onStartupOverlaysDone`. Hver utgang av oppstartskjeden må kalle den — en gren
  som glemmer det, gir en bruker som aldri ser beskjedene sine.

---

## 3. Varsler og beskjeder — to veier, én garanti

Backenden legger §11-varsler i en kø **før** den sender push. Køen er
garantien; pushen er rask levering. Klienten leser begge.

```mermaid
flowchart TD
    subgraph ADR ["Adressen · Push.kt"]
        direction TB
        APP["Appstart<br/>BestefarApp"] --> CH["Push.ensureChannel<br/>«Venner og lag»"]
        CH --> REG{"Innlogget?"}
        REG -->|"nei"| NOOP["ingenting —<br/>et varsel er alltid til noen"]
        REG -->|"ja"| TOK["FirebaseMessaging.token"]
        TOK --> PUT["PUT /v1/devices<br/>idempotent · hver oppstart"]
        LOGIN["Vellykket innlogging"] --> ASK["Forklaring → systemdialog<br/>POST_NOTIFICATIONS (API 33+)"]
        ASK --> PUT
    end

    SRV["Backend: varselet oppstår"] --> KOE["Rad i pending_messages<br/>lagres FØRST"]
    SRV --> SND["push.send til kjente enheter"]

    subgraph RASK ["Rask levering · PushService.kt"]
        direction TB
        SND --> FG{"App i forgrunnen?"}
        FG -->|"ja"| SVC["onMessageReceived<br/>bygger varselet selv"]
        FG -->|"nei"| SYS["Android tegner det<br/>default_notification_* i manifestet"]
        SVC --> TAP["Trykk → forsiden"]
        SYS --> TAP
    end

    subgraph GARANTI ["Garantien · Messages.kt"]
        direction TB
        KOE --> GET["Neste appstart:<br/>GET /v1/messages"]
        GET --> VIS["Vis én om gangen<br/>etter oppstartsvinduene"]
        VIS --> ACK["POST /v1/messages/ack<br/>ETTER visning"]
        ACK --> MARK["Serveren setter delivered_at<br/>— sletter ikke raden"]
    end
```

- **Pushen kan mislykkes uten at beskjeden går tapt.** Avslått varseltillatelse,
  rotert FCM-token, telefonen av, oppbrukt push-budsjett — beskjeden ligger
  fortsatt i køen og vises ved neste appstart.
- **Meldingen må ha en `notification`-blokk.** Ligger appen i bakgrunnen, er det
  Android selv som tegner varselet ut fra den; `onMessageReceived` kjøres bare i
  forgrunnen. En ren `data`-melding ville vært usynlig i det vanligste
  tilfellet, uten at noen part ser en feil.
- **Kvitteringen kommer etter visningen.** Serveren markerer i stedet for å
  slette, nettopp for å tåle en klient som forsvinner imellom. Prisen er at en
  beskjed kan vises to ganger — den billige feilen av de to.
- **Nei til varsler er et gyldig svar.** Enheten registreres likevel, så varsler
  som skrus på senere virker med én gang — og køen virker uansett.
- **Ingen ruting på `kind`.** Alle varsler og beskjeder ender på forsiden.
  Venne- og lagsidene er lokale skjeletter uten server-kobling, så en dyplenke
  ville landet på en skjerm som ikke kjenner `team_id`. Følgen er at beskjeder
  som ber om et svar («Bekreft i appen») **når fram, men ikke kan besvares**.

---

## Historikken

Hva hver runde endret, står i `android/CHANGELOG.md`. Denne fila beskriver
appen slik den er **nå** — seksjonene «Nytt i v0.15» til «Nytt i v0.19» ble
flyttet ordrett dit 2026-08-08, fordi diagrammene hadde blitt stående på v0.14
med fem lag lapper under. Endrer du appen, endrer du diagrammet.

