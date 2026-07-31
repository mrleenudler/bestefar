# Til utvikler — felttest skytebanen (v0.12)

Kopi av tilbakemeldingene fra `musings.txt` (2026-07, felttest på skytebanen)
og hvordan de er løst. APK: `dist\Bestefar-0.12.apk`.

## «Det er altfor vanskelig å få scannet» → capture-first
> Gatingen er for sensitiv … Teksten endres til: «Beveg telefonen slik at
> skjermen passer i rammen.» Når kriteriene er oppfylt, tas bildet umiddelbart,
> MEN vi viser den grønne rammen med «Klar!» i 0,4 s, og viser deretter
> ta-bilde-animasjonen.

- **Flyten er snudd:** bildet tas nå **stille i det øyeblikket** kriteriene er
  oppfylt. Deretter spilles UI-et: grønn ramme + «Klar!» i 0,4 s → klassisk
  hvit blits. Analysen kjører allerede mens dette vises.
- **Holdevinduet er kuttet fra 24 til 6 frames** (~0,2 s). 24-vinduet krevde
  kvalitet+størrelse i *hvert* av 24 frames (ett dårlig frame nullstilte), og
  fantes bare fordi den gamle flyten glødet grønt *før* capture. Det behovet er
  borte; 6 frames vokter kun mot sveipebevegelse.
- Hovedteksten er nå alltid «Beveg telefonen slik at skjermen passer i rammen.»
  Kun reelle kvalitetsblokkere gir egne hint: gjenskinn («Gjenskinn på skjermen
  — prøv en annen vinkel») og lys/fokus («Hold kamera nærmere / bedre lys»).
  Ingen grønn ramme før bildet er tatt.

## Opp-ned-bilde
> Scan-skjermen tillater at telefonen ligger begge veier, men resultatet kan
> bli opp-ned.

- **Rotårsak:** `sensorLandscape` roterer 180° (landskap ↔ omvendt landskap)
  **uten** at activityen gjenskapes → `ImageCapture.targetRotation` ble stående
  på rotasjonen fra oppstart → stillbildet fikk feil `rotationDegrees`, og
  kjernen så et opp-ned-bilde.
- **Fiks:** en `OrientationEventListener` følger telefonens fysiske orientering
  kontinuerlig og oppdaterer `targetRotation` (standard CameraX-mønster).
  Begge liggende retninger gir nå riktig vei opp.

## Ny scan-ramme
> Brukerne forstår ikke hvordan telefonen skal holdes …

Rammen er nå en egen view (`ScanFrameView`) som tegner hele skjermlayouten:

1. sirkel som omslutter det **hvite skiveområdet** + sirkel som omslutter den
   **sorte bullen**
2. rektangel om **poenglista** (høyre side) + rektangel **rett under** om
   **oppsummeringen** (snitt/S-10/SUM/TOT/Xm/Ym)
3. ytre ramme **3× så tykk** (9 dp mot før 3 dp)
4. alt **matt-gjennomsiktig** (hvit, 40 % alfa; grønn ved «Klar!»)
5. **Geometrien er målt, ikke gjettet:** fraksjoner målt på rektifiserte
   C-bilder (`_probe_frame_geometry*.py`): senter (0.415, 0.420), hvit skive
   r=0.304·B, bull r=0.121·B, tabellfelt x 0.752–0.990, liste y 0.016–0.516,
   oppsummering y 0.516–0.824.

**Verifiser selv:** `Visualiseringer/outputs/scan_frame_mock.png` viser rammen
tegnet over C1/C5/C9 (oransje og kraftig der, kun for synlighet — posisjonene
er identiske med appen). Sjekk spesielt bull-sirkelen mot C5.

## Glare
> Kan vi identifisere når det er glare på skjermen?

- Ja: gjenskinn måles nå per frame som **største sammenhengende mettede flekk**
  inne i skjermblobben (morfologisk åpning fjerner punktmetning fra
  LED/markører; helt utbrent skjerm fanges allerede av klipp-taket).
- Flekk > 5 % av skjermarealet blokkerer capture og gir statushintet om
  gjenskinn. Terskelen er **UKALIBRERT** (bevisst permissiv); målt verdi vises
  i debug-overlayet (`glare=…`) slik at den kan kalibreres fra felt.
- Dette avviser bildet *før* det tas. Analysens harde kvalitetsport består
  som før uendret.

## Teknisk
- FFI/JNI-probeprotokollen er utvidet 13 → 15 verdier (`glare_frac`,
  `glare_ok`) — `jni_bridge.cpp` og `BestefarCore.kt` holdt i synk.
- Desktop-kjernen bygget og **verify_cset_cpp.py: 10/10 PASS** (ingen
  regresjon i analysen; endringene berører kun capture-siden).
- `scan_frame.xml`/`scan_frame_active.xml` (drawables) er fjernet.

## CV-notater (fra musings, parkert/besvart i chat)
- MPI-firkanten: «Ignored for now» — C6-avviket, forklaring av min(V,H)-bildet
  og utdyping av offset-reframingen er besvart i sesjonens oppsummering.
- Strukturell validering av topp-N ble forsøkt 2026-07-05 og **feilet som
  diskriminator** (tall/ringer scorer like høyt som ekte firkant) — se
  CLAUDE_CONTEXT.txt før evt. nytt forsøk.
