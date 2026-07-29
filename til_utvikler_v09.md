# Til utvikler — musingsUI runde 7 (v0.9)

Kopi av tilbakemeldingene fra `musingsUI.txt` og hvordan de er løst. Flere punkter
var «ikke implementert i forrige runde» — årsaken er notert der den er funnet.

## Globalt
- **Topplinje 5 % ned.** Årsak funnet: `column.addView(view, MATCH_PARENT, WRAP_CONTENT)`
  lagde nye `LayoutParams` og nullet `topMargin`. Nå 1-args `addView(view)`, så
  marginen (`heightPixels * 0.05`) faktisk beholdes.
- **Forskningsdialog.** Vises to steder: (a) Ja/Ikke nå/Aldri-popup etter N serier
  (fra resultatflyten), (b) delings-checkbokser fra jaktloggen. (b) er nå
  **auto-krysset** første gang. Teksten var allerede oppdatert i runde 6.
- **Oppstartsmelding av/på.** «Vil du dele bilder …»-vinduet kan slås av i Avanserte
  innstillinger (`Store.startupDonateAsk`, default på).
- **Popup-modell.** Liste-popuper som manglet Avbryt fikk `setNegativeButton(Avbryt)`
  som «Rediger lag».

## Innsikt (ombygd)
- **(i)** er nå eksplisitt 36 dp, tema-tintet, t.h. for tittelen.
- **Silhuetter/piktogrammer:** valgt = sort (lys) / tekstfarge (mørk); uvalgt = grå.
  Presentasjonssilhuetten er tekstfarget (grå kun når data mangler). Viltnavnet har
  default tekstfarge.
- **«To loddrette striper over jeger-ikoner»:** kom av at hele ImageButton (med
  bakgrunn/ramme) ble skalert 1,8× (sittende). Nå ligger rammen på en FrameLayout av
  fast størrelse, og bare det indre bildet skaleres.
- **Størrelser:** FIT_CENTER i cellen normaliserer allerede SVG-ene, så de store
  enum-skalaene doblet opp. Innsikt bruker egne skalaer (liggende 1,0; sittende 0,75;
  knestående 0,8; stående 1,0). De opprinnelige −65 %/+80 %-forslagene gjelder
  **stillingsvelgeren etter scan** — der er ikonene nå homogenisert (fast iconSize).
- **Ramme rundt matrisen:** jeger-stilling høyrejustert (stående over hold-kolonnen),
  hold-kolonne like bred som stående-cellen og høy nok til å ramme matrisen (6 celler
  ≈ 5 rader), dyr-vinkling venstrejustert nederst (til venstre for 200 m). Hold-
  knappene viser tresifret tall **og** «m» (redusert inset så «100 m» ikke kuttes).
- JJJJ/A/VT/DDD-arrangementet og «Grønn/rød i fet» var merket `<ignore>`.

## Poeng-visning + karusell
- **«Poeng:»** står nå foran totalen (`Poeng: 72,5 (70)`). «Poeng:»-overskriften over
  poenglista var allerede der.
- **«… med anlegg/reim»** vises (var allerede kodet; verifisert — krever at man
  faktisk valgte hjelpemiddel i stillingsvelgeren).
- **Dato** «8. mars 2026    08:38» (fire mellomrom for luft) vises nå på resultatkortet.
- **«Mitt gjennomsnitt for denne stillingen»** (uten «for sittende», uten KI).
- **Skive:** fjernet ekstra hvit ytre halvring (maxR 10,5 → 10).
- **Karusell** i Serielogg: rik visning (skive + «Poeng:»-liste + total + snitt) med
  piler i endene + OK. Datoen animeres når **dagen** bytter: starter ~3× og ~20 % ned
  til høyre, zoomer til default på 1 s, ikke-blokkerende, avbrytes ved nytt piltrykk.
- **Dummy-scan:** 10 skudd.

## Profil / Venner / Jaktlag
- **Profil:** «+ legg til nytt lag» er outlined (ikke uthevet). Lag-rekkefølgen følger
  `sortOrder` (samme som Venner).
- **Profil-lag-medlemskap:** «Slett lag» håndterer: eneste medlem → slett stille;
  flere medlemmer → forlat; eneste leder → oppløs for alle / overfør + forlat.
- **Jaktlag:** medlemmer er klikkbare → karusell der pilene går **helt rundt** + OK.
  Lagleder(e) merkes «(Lagleder)» øverst, ellers alfabetisk. «Slett lag» er lagt til
  i Rediger lag.
- **Venner:** dobbelt innrykk for medlemmer i lag.
- **Utvikler → «Legg til venn»:** navn + velg lag, genererer en venn med 5 serier á
  10 skudd (50 øvelsesskudd) for å teste venne-/lag-UI.

## Logg jaktskudd
- **Felling-checkbox** vises nå også i den forenklede (forsknings-av) flyten, med
  samme bekreftelsesdialog ved uavkrysset.
- **Posisjon logges også** når forskningsdeling er av.
- **Skuddlogg:** «Utfall: Dødelig» + «Felling vellykket» var redundant — ved dødelig
  vises kun felling-status, i samme font som resten.
- **Rediger:** etikett foran hvert felt; eksisterende tekst forhåndsutfylt og
  redigerbar; **dato redigerbar**; utfallsknappene stablet loddrett; «Skade» vist som
  «Ettersøk»; «Dyret ble ikke funnet» kun når Ettersøk er valgt.
- **Mørk modus:** stilling-silhuettene i jaktloggen tintes til tekstfarge.

## Fremdeles backend-avhengig (skjelett på enheten)
Lagleder-avstemning (7-dagers nedtelling, push, enstemmig-avslutt), «Fjern inaktiv
lagleder», invitasjoner og kryssbruker-varsler ved navnebytte/fjerning. Se
`backend_spec.md` §4/§11.
