# Tilbakemeldinger til utvikler — v0.8 (musingsUI runde 6)

✅ implementert · 🟡 delvis/skjelett · ⛔ utsatt (backend) · ✍️ spec.

## Hovedskjerm
- ✅ Kun antall øvelsesskudd + Scan-knapp; serieliste, øvelsesforslag og
  øktoppsummering fjernet fra hovedskjermen.
- ✅ Øverste menylinje 5 % ned (beholdt fra runde 5).

## Samtykke / jaktmål
- ✅ Ny forskningstekst («Hjelp oss å lære …»).
- ✅ «Mitt jaktmål» og «Bidra til forskning» kommer aldri på samme serie;
  jaktmål først (etter 3 serier), forskning tidligst 2 serier senere.
- ✅ Radioknapp-bug fikset: RadioButtons manglet id-er, så forrige valg ikke ble
  deaktivert. Nå gis hver en id og RadioGroup styrer enkeltvalg.

## Innsikt
- ✅ (i): «Grønn»/«rød» i fet + farge; ikonet følger app-tekstfargen (synlig i
  mørk modus); (i) flyttet opp til «Innsikt»-tittelen.
- ✅ Stilling-valgene er nå ikoner (ikke tekst), med ramme; vilt-vinkling likeså.
  Mørk modus: ikoner får tekstfargen, uvalgte en dus variant.
- ✅ Viltnavn i default tekstfarge; andel + prosent beholder grønn/rød.
- ✅ «øv på stillingen» lysere grå i mørk, mørkere i lys.
- ✅ Layout: JJJJ (jeger) øverst, VT-rader, DDD (vilt) nederst, hold-kolonne t.h.
  Jeger-/vilt-/holdknapper like store; «100 m» får plass på én linje.
- ⚠️ Tallene er fortsatt basert på PLASSHOLDER-radier/`RING_STEP_CM`.

## Profil
- ✅ Lys modus default (beholdt).
- ✅ «+ legg til nytt lag» som knapp.
- ✅ «Mitt jaktmål» flyter naturlig over to linjer, tall bold+større, «Endre» t.h.
- ✅ Klikk på et jaktlag åpner laget (TeamPage), ikke «legg til»-menyen.

## Jaktlag (TeamPage — front-end-skjelett)
- ✅ Navn øverst, «Inviter medlemmer», medlemsliste (egen bruker + venner),
  «Rediger lag» / «Velg leder» nederst t.v., «Lukk» nederst t.h.
- ✅ Rediger lag: Endre lagnavn (lokalt), Fjern medlemmer, Overfør lederskap
  (velg medlem + bekreft).
- ⛔ Medlemskap, roller, invitasjoner, **ledervalg-avstemning med 7-dagers timer,
  push-varsler** og «navneendring/fjernet»-meldinger krever backend
  (backend_spec.md §4 + nytt §11). Bygget som skjelett med toasts/notater.
- 🟡 «Fjern inaktiv lagleder» lagt i Avanserte innstillinger (skjelett).

## Legg til jaktlag (LagActivity)
- ✅ Fullsides; «Mine lag» som innrykkede knapper; klikk åpner laget.
  «Opprett nytt lag» uten «+» i overskriften; roller som knapper; ingen
  invite-toast. «Venter på backend-implementering».

## Jakt
- ✅ «Tilbake» lukker menyen og går til hovedsiden.

## Stilling-ikoner
- ✅ Homogenisert visuell størrelse via per-stilling skala (Liggende 0.35,
  Sittende 1.8, Knestående 0.8, Stående 1.0) — brukt i Innsikt.

## Poeng-visning (resultatkort)
- ✅ Genererte testserier er nå 10 skudd.
- ✅ «Poeng:» foran poenglista.
- ✅ Stilling vises «… med anlegg» / «… med reim».
- ✅ «Langsiktig …» → «Mitt gjennomsnitt for denne stillingen: X» uten KI.
- ✅ Skiveringene fikset: 6 hvite + 5 sorte ringer (inkl. inner/ytter-tier),
  svart blink når ring 7 — riktig telt både innenfra og utenfra.

## Melding til utvikler
- ✅ Tittel + tekst legges i selve mailto-URIen (subject/body) — mer pålitelig.
- ⛔ Direkte sending uten e-postapp krever backend (backend_spec §10).

## Venner
- ✅ Egen bruker vises også i lagene; venner i flere lag føres opp under hvert;
  kun venner uten felles lag vises utenfor lagene.
- ✅ Flytt opp/ned: «Avbryt» til høyre; flytt-funksjonen fikset (sortOrder
  normaliseres, så bytte får effekt).
- ✅ «Jeg vil dele»-pilene kraftigere; menyen åpen som default.
- ✅ «La venner finne meg»-varsel stabilt (rebuild ved endring) med ny tekst.

## Logg jaktskudd
- ✅ «Felling var vellykket»-checkbox; uavkrysset → «Var fellingen vellykket?»
  <Nei>/<Ja>. Logges som vellykket/mislykket. (Kun vellykkede fellinger deles
  med venner — filtrering skjer i backend når deling implementeres.)
- ✅ «Dyret løp» godtar 0 m, men ikke sifre etter en enslig 0.
- ✅ «Skriv inn omtrent hvor langt dyret løp».

## Se registrerte skudd
- ✅ Tilbake-knapp (hardware) i detalj går til lista.
- ✅ Store, høye piler (samme stil overalt hvor vi blar); fjernes i endene mens
  OK står fast.
- ✅ Nye skudd på samme dato legges øverst (created-tiebreaker).
- ✅ «Rediger» endrer all info inkl. utfall/ettersøk/ikke funnet.
- ✅ Silhuetter tinter riktig i mørk modus.
- ✅ Flervalg → «Slett alle» nederst t.h. + bekreftelsespopup (Avbryt → hovedsiden).

## Spec
- ✅ UI-spec §14, backend_spec §11 (lag-avstemning/push), til_utvikler_v08.md.
