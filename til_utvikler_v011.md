# Til utvikler — musingsUI runde 9 (v0.11)

Kopi av tilbakemeldingene fra `musingsUI.txt` og hvordan de er løst.

## Oppstart
- **Bildedelings-popup gjeninnført ved oppstart:** vises FØRSTE gang appen åpnes,
  og deretter én gang neste sesong hvis deling ikke er valgt
  (`shareDevImagesSeason`). Valget kan uansett styres fra Avanserte innstillinger
  («Del bilder med utvikler»). Intro-vinduet vises som før (første gang / dev-flagg),
  og bildedelingen kommer nå etter intro.

## Avanserte innstillinger
- Fjernet setningen «Du blir spurt på nytt ved ny sesong» fra forsknings-hintet.
- **«Fjern inaktiv lagleder»:** jaktlag-velger-dialogen («For hvilket jaktlag …»)
  vises nå KUN når flere lag har inaktiv lagleder; ellers toast **«Ingen inaktive
  lagledere funnet»**. Inaktivitet krever aktivitetsdata per lagleder = backend
  (§11), så lista er tom i skjelettet (⇒ alltid toast inntil videre).

## Innsikt
- **Ikon-flukt:** alle fire stillinger står nå inntil hverandre til venstre med kun
  en smal stripe (2 dp) imellom; stående havner i 4. kolonne — rett over hold-
  kolonnen. Matrisebredden er låst til de tre første kolonnene, så hold-kolonnen
  lander presist under stående og «kobler» seg til resten. Avstandsknappene er like
  store som ikoncellene (60×54).
- **Mørk visning:** silhuetter/piktogrammer tintes nå med den **varme lysebrune**
  fargen (colorPrimary = #D8B79B), ikke den nøytrale tekstfargen. Lys visning er
  fortsatt sort.
- **Nye viltsilhuetter:** Elg (side + front) og Villsvin (side + front) portert fra
  `UI/*_silhuett_*.svg` til vector drawables; Innsikt bruker art-spesifikk silhuett
  (elg/villsvin har ikke skrå-variant, så skrå bruker deres side-silhuett).

## Scan-bilde (resultatkort)
- Poengene er nå **midtstilt** under «Poeng:» (ikke venstrejustert).
- **Tettere** poeng-rader (kompakt blyant, liten radhøyde).
- Lagt til «:» bak «Mitt gjennomsnitt for denne stillingen».

## Serier
- Lista **avsluttes nå over** «Lukk»-knappen: scroll-området har bunnmarg, så
  seriene klippes ved knappens overkant i stedet for å scrolle bak den.

## Logg jaktskudd (Rediger)
- Ved «Bom» / «Dyret ble ikke funnet» er avstandsfeltet nå **umiddelbart blokkert**
  (deaktivert + ikke fokuserbart), så det ikke går an å skrive inn avstand.
- Animasjonen **blåser opp fontstørrelsen** (1,7×) samtidig som tallet fader ut —
  mer oppmerksomhetsfangende enn ren utfading.

## Fremdeles backend-avhengig (skjelett på enheten)
Lagleder-avstemning / «Velg leder» (7-dagers nedtelling, push, enstemmig-avslutt),
inaktiv-lagleder-deteksjon, invitasjoner og kryssbruker-varsler. Se
`backend_spec.md` §4/§11.
