# Til utvikler — UI-runde 10 (v0.13)

Kopi av alle tilbakemeldingene fra `musingsUI.txt` og hvordan de er løst.
APK: `dist\Bestefar-0.13.apk` (versionCode 13).

---

## Duplikate serier

> Det ser ut til at hvis poengene er like, defineres seriene som like. (Fra
> erfaringer med opp-ned bildene) Serier må også ha samme treffpunkter med en
> liten margin (<= 0,1p) for at vi skal definere dem som like og vise
> lik-serie dialogen.

- `ResultActivity.isIdentical()` krever nå BÅDE like poeng (< 0,05) OG like
  treffpunkter. Treffene pares grådig (nærmeste ubrukte treff), og hvert par må
  ligge innenfor **0,1 poeng**.
- `rRel` er oppgitt i **ringsteg**, og ett ringsteg = ett poeng, så margen
  oversettes direkte: avstand i planet `(rRel·cosθ, rRel·sinθ)` ≤ 0,1.
- Derfor treffer den også nettopp opp-ned-tilfellet: et speilvendt bilde gir
  *identiske radier* (samme poeng) men helt andre treffpunkter.

## Forskning legges i bakgrunnen

> Vi legger den i bakgrunnen så lenge · Skjul dialogen i oppstart · Endre «Del
> med forskning» checkbox i logg jaktskudd til «Detaljert visning» · Lås
> checkboxen i Avanserte innstillinger til unchecked.

- Nytt flagg `Dialogs.RESEARCH_ENABLED = false`. `maybeResearchConsent()` og
  `maybeHuntConsent()` returnerer umiddelbart, så ingen forskningsdialog dukker
  opp i noen flyt. All kode står urørt bak flagget — sett det til `true` for å
  slå funksjonaliteten på igjen.
- Checkboxen i Logg jaktskudd heter nå **«Detaljert visning»**. Siden den ikke
  lenger har noe med forskning å gjøre, er koblingen til vilttypen «Annet»
  fjernet — den kan nå brukes for alt vilt.
- Bryteren i Avanserte innstillinger er **låst av** (`isEnabled = false`) med
  hintet «Forskningsdeling er midlertidig satt på pause mens innsamlingen
  ferdigstilles.»

## Svart statuslinje i lys visning

> Telefonens statusikoner må ha svart bakgrunn, også i lys visning.

- `statusBarColor`/`navigationBarColor` = svart og `windowLightStatusBar` =
  false i både `values/themes.xml` og `values-night/themes.xml`.
- **Merk fella:** med `targetSdk 35+` tvinges edge-to-edge og `statusBarColor`
  blir ignorert. Vi setter derfor også
  `android:windowOptOutEdgeToEdgeEnforcement=true`. Da tegner systemet baren
  selv, og `Ui.applyInsets()` blir automatisk en no-op (innsets = 0) — ingen
  layoutendringer var nødvendige.

## OCR / poengskjerm

> Når OCR-resultater finnes, brukes de direkte i poengvisningen, og tar
> presedens over identifiserte treff. Da skal poengene ikke være sortert etter
> verdi, men stå i den rekkefølgen de er oppført på skjermen. Identifiserte
> treff vises nederst på skjermen, i.e. de to visningene bytter plass når det
> ikke er samsvar mellom dem.

- Poengkolonnen viser nå `ocrScores` når de finnes — **usortert**, i den
  rekkefølgen OCR-en leste dem av apparatskjermen. `OcrVerifier` sorterte aldri
  returverdien; det var visningen (`ocr.sorted()`) som gjorde det.
- Totalen under skiva regnes av det som **vises**, så poengliste og sum aldri
  spriker.
- Ved uenighet legges «Identifiserte treff:» nederst, under snitt-blokken.
- Ny tekst: **«Appen klarte ikke å se poengene riktig.\nVil du lagre serien
  likevel?»**, knapp **«Lagre leste poeng»**.

## Merking i Serier flyttet skjermen

> Når serier lengre ned på listen merkes, hopper skjermen til start for hver ny
> som merkes. Skjermen må ikke flyttes.

- Årsak: hver avkryssing kalte `renderList()`, som bygget hele ScrollView-en på
  nytt → scroll-posisjonen nullstilt.
- Radene, søppelbøtta og bunnknappene holdes nå som felt. Merking kaller
  `updateSelectionUi()`, som kun setter radbakgrunn og bytter synlighet på
  knappene. Full ombygging skjer bare når selve lista endres (sletting).

## Oppstart

> Den skal dukke opp første gang appen åpnes. → Når «Vis oppstartsmelding hver
> gang» er valgt i Utviklermodus, skal meldingen for bildedeling også vises.

- `maybeDonateThenTutorial()` har fått `store.alwaysShowStartup -> true` som
  første betingelse, så dev-flagget nå også tvinger fram bildedelings-vinduet
  (ikke bare intro-vinduet).

## Skjermorientering

> Vi låser alle skjermbildene til vertikal visning. Bare Scan skal være
> horisontal.

- `android:screenOrientation="portrait"` på alle aktiviteter i manifestet;
  `CaptureActivity` beholder `sensorLandscape`.

## Innsikt

> Nå passer ikonene mye bedre, men avstandene skal gå så langt ned at 200 m står
> rett til høyre for vilt-posisjon valgene … Vi ønsker også at alt innholdet skal
> passe på én skjerm. Pass på at de auto-skaleres. Legg inn silhuetter for [rein]

- Rammen er bygget om til **7 like høye rader**: stillingsraden øverst, og en
  kropp på 6 rader. Vinkelraden (vilt-posisjonsvalgene) er flyttet **inn i**
  venstrekolonnen som rad 6 — da faller hold-kolonnens 6. knapp (**200 m**)
  nøyaktig ved siden av den. Tidligere lå vinkelraden *under* hele kroppen, og
  200 m havnet én rad for høyt.
- Radhøyden regnes ut fra skjermhøyden (`(høyde − 250dp) / 7`, klemt til
  34–66 dp), og cellebredden av skjermbredden, slik at alt får plass på én
  skjerm på både små og store telefoner. Tekststørrelsen på viltraden og
  avstandsknappene skalerer med radhøyden.
- **Villrein** har fått egne silhuetter for alle tre vinkelvalgene:
  `ic_rein_front`, `ic_rein_side` og `ic_rein_skraa`. Villrein er dermed den
  første arten med en **egen skrå-silhuett** — elg og villsvin mangler fortsatt
  den og faller tilbake på side-varianten sin.

## Scan-bilde

> Nå ser jeg det er blyanten som blokkerer plassen til høyre. Fjern den. Vi lar
> OCR ta den rollen, og heller be om en re-capture om analysen feiler.
> → Kan vi anbefale re-scan når bildet ikke blir korrekt analysert, eller trenger
> vi et cue fra CV-kjernen?

- Blyanten (`ic_edit`-knappen per poenglinje) er fjernet. Poengene står nå fritt
  midtstilt under «Poeng:».
- **Svar på spørsmålet: vi trenger ikke noe nytt cue.** Vi har allerede to:
  1. `result.status != OK` — analysens harde kvalitetsport («jeg fikk ikke
     kalibrert ringene / fant ikke skiva»). Det er dette som utløser den nye
     dialogen «Bildet ble ikke korrekt analysert. Scan bildet på ny.»
     ‹Avbryt›/‹Scan›.
  2. **OCR-uenighet** — det er cue-et for den vanskeligere klassen: analysen
     *lyktes* men leste feil (typisk opp-ned-bildene). Der beholder vi
     ‹Forkast›/‹Lagre leste poeng›, siden vi faktisk har fasiten fra skjermen.
  Et tredje signal fra kjernen (f.eks. en lav-konfidens-flagg) ville først bli
  interessant hvis vi ser tilfeller som verken porten eller OCR fanger.

## Logg jaktskudd — «Rediger»-animasjonen

> La skriftstørrelsen vokse til det dobbelte, og la den bevege seg opp mot
> høyre. (Går det bra, eller er det begrensninger i hva vi kan gjøre?)

- **Går fint.** Tallet skalerer til `2×`, flytter seg +70 dp i x og −70 dp i y og
  fader til 0 over 550 ms, med akselererende interpolator (som om det suser
  vekk). Pivot er satt til venstre/nedre hjørne så veksten går oppover-høyre.
- Én ting måtte fikses: foreldrene klippet animasjonen. `clipChildren = false` og
  `clipToPadding = false` på både kolonnen og NestedScrollView-en i dialogen.

## Avanserte innstillinger — lagring av scan

> Valg om å lagre scannede bilder i bildearkivet. Etter første scan, må det åpnes
> en dialog: «Ønsker du at skjermbildet skal lagres i bildearkivet ditt?»
> ‹Ja›‹Nei› → Ny dialog: «Du kan endre dette valget i «Avanserte innstillinger»»

- Ny bryter **«Lagre scannede bilder i bildearkivet»** i Avanserte innstillinger
  (`Store.saveScansToGallery`, default **på**).
- Spørsmålet stilles én gang, etter første **vellykkede** scan, før stillings-
  valget — så det ikke kommer midt i lagringsflyten. Deretter
  informasjonsdialogen om hvor valget endres.
- Siden CaptureActivity lagrer *før* vi rekker å spørre, sender den nå
  galleri-URI-en videre; svarer brukeren «Nei», slettes også bildet fra den
  første scanen.
- Rejected scans hopper over spørsmålet — der ber vi i stedet om et nytt bilde.

---

## Ikke implementert (med vilje)

- **Venner-blokka** i `musingsUI.txt` lå inne i `<ignore>`-tagger.
- Alt under «Alt nedenfor skal ignoreres», inkludert teksten i «Mitt jaktmål»-
  dialogen (den lå under skillelinjen denne runden).
