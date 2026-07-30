# Til utvikler — musingsUI runde 8 (v0.10)

Kopi av tilbakemeldingene fra `musingsUI.txt` og hvordan de er løst.

## Rotårsak funnet: usynlige silhuetter/tekst (flere runder)
`Ui.themeColor()` returnerte `tv.data` rått. For `android.R.attr.textColorPrimary`
peker den oppløste verdien til en **ColorStateList**, ikke en farge-int — `tv.data`
blir da en RESSURS-ID som tolket som ARGB gir en tilfeldig, ofte «usynlig» farge.
Det var årsaken til at alt som ble tintet med tekstfargen (vilt-silhuetter i lys
OG mørk, valgt piktogram i mørk, viltnavn-tekst, jaktlogg-silhuetter i mørk)
forsvant — gjennom flere runder. `colorPrimary` virket fordi den oppløses til en
direkte farge-int. `themeColor` håndterer nå begge: farge-int OG ressurs-referanse
(color/ColorStateList via ContextCompat/ResourcesCompat). Én fiks løser samtlige
synlighetspunkter under.

## Globalt
- **Topplinje 3 % opp.** Marginen justert 5 % → 2 % av skjermhøyden.
- **Forskningsdialog:**
  - Auto-kryss av delings-checkboxene skjer nå KUN når forskning er aktivert
    (`consentResearch == "ja"`).
  - **«Del med forskning»** lagt som egen bryter i Avanserte innstillinger
    (på = «ja», av = «aldri»; 18-årsgate ved påslag).
  - **Ny sesong → nytt spørsmål:** `researchConsentSeason` lagres ved svar; ved ny
    sesong spørres det på nytt (også for dem som allerede deler).
- **Oppstartsmelding:** bildedelings-spørsmålet popper ikke lenger ved oppstart.
  Valget bor nå i Avanserte innstillinger som **«Del bilder med utvikler»**
  (på = «ja», av = «nei»). Intro-vinduet vises fremdeles kun første gang / når
  dev-flagget «Vis oppstartsmelding hver gang» er på.

## Innsikt
- **(i)** byttet fra SVG-ikon til UTF-8-glyf «ⓘ» (tekstfarget, 40 dp, klikkbar) —
  garantert synlig.
- **Silhuetter/tekst synlige** i lys og mørk (themeColor-fiksen): vilt-silhuett
  sort i lys / tekstfarge i mørk (grå kun uten data), valgt piktogram tekstfarget i
  mørk, viltnavn i default tekstfarge.
- **Ramme-flukt:** de tre første stillingene venstrejustert (liggende rett over
  «forfra»-viltet nederst), stående skjøvet av en spacer helt til høyre så den står
  i loddrett flukt med hold-kolonnen.
- **Luft** lagt mellom stående-ikonet og «25 m» (6 dp, likt mellomrommet mellom
  avstandene).
- **Avstandsknappene** er nå like store som stilling-/vilt-ikoncellene (60×54 dp),
  jevnt mellomrom, så høyre-aksen binder seg pent til resten.

## Stillingsvelger (etter scan)
- **«Liggende» ikke lenger sammentrykt:** ikonet tegnes med FIT_CENTER i en
  fast-høyde/variabel-bredde-boks (MaterialButton.iconSize tvang kvadrat). Brede
  ikoner får være litt lengre, aspektforholdet beholdes.
- **«## skudd»** teller nå faktiske SKUDD (ikke serier) —
  `Store.shotsCountByPosition()`.

## Serier
- **«Lukk»**-knapp nederst til høyre i serie-lista; lista får bunn-luft så den ikke
  scroller bak knappen.

## Poeng-visning
- **«Poeng:»** (overskrift over poenglista) har nå samme størrelse som poengene og
  er i fet — både i resultatkortet (19sp) og serie-karusellen (18sp).
- **«Mitt gjennomsnitt for denne stillingen»** brytes nå kun på stilling (ikke
  avstand/hjelpemiddel) og vises som:
  «Mitt gjennomsnitt for denne stillingen / Denne sesongen: X / Totalt Y»,
  oppdateres løpende (gjeldende serie regnes med selv før lagring).

## Logg jaktskudd
- **«Felling var vellykket»** er nå UAVKRYSSET som default (både forenklet flyt og
  side 2). Står den uavkrysset ved lagring, spør «Var fellingen vellykket?»
  <Nei><Ja> og lagrer deretter riktig status.
- **Rediger → «Dyret løp»:** velges «Bom» eller «Dyret ble ikke funnet», fades
  tallet elegant ut, slettes og feltet låses. Gjenåpnes feltet når et annet utfall
  velges.
- **Mørk modus:** jaktlogg-silhuettene tintes nå faktisk til tekstfargen
  (themeColor-fiksen) — både i lista, detaljen og posisjonsvelgeren.

## Venner
- **Klikk på jaktlag** åpner nå den fulle lagsiden (samme som fra Min profil):
  du selv står i medlemslista, og lagleder får «Rediger lag».

## Fremdeles backend-avhengig (skjelett på enheten)
Lagleder-avstemning (7-dagers nedtelling, push, enstemmig-avslutt), «Velg leder»,
«Fjern inaktiv lagleder», invitasjoner og kryssbruker-varsler. Se `backend_spec.md`
§4/§11. Forsknings-re-samtykke per sesong er inntil videre klientstyrt
(`researchConsentSeason`).
