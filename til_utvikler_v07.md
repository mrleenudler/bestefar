# Tilbakemeldinger til utvikler — v0.7 (musingsUI runde 5)

Kopi av alle tilbakemeldinger fra `musingsUI.txt` (runde 5) med status.
✅ implementert · 🟡 delvis/skjelett · ⛔ utsatt (krever backend) · ✍️ spec.

## Globalt
- ✅ **Toast-kø-bug:** ny toast avbryter forrige (`Ui.toast`), så en kø av
  toasts ikke blokkerer appen. Brukt i jaktloggen (der bugen viste seg).
- ✅ **Identiske serier:** sammenligner nå KUN med forrige serie —
  «Denne serien er helt lik forrige serie. Vil du lagre likevel?» <Lagre>/<Avbryt>.

## Oppstart / hovedskjerm
- ✅ «Vi trenger hjelp …»-dialogen (og velkomst) er nå fullskjerm-overlegg som
  dekker Scan serie-knappen; luft mellom setningene og stor luft ned til valgene.
- ✅ Øverste menylinje flyttet 5 % ned.

## Innsikt (stor omlegging)
- ✅ Nytt klikk på Innsikt-knappen lukker den (går hjem).
- ✅ **Matrise:** fem vilttyper (rader) rammet inn av jeger-stilling (øverst,
  sittende forhåndsvalgt), dyr-vinkling (nederst, bredside forhåndsvalgt) og
  skuddhold (kolonne t.h.: 25/50/75/100/150/200, 100 forhåndsvalgt).
- ✅ Vilt-silhuett i fast ramme, skalert med hold (25 m ≈ fyller/klippes, 200 m ≈
  halv ramme). Hjort brukes som placeholder for både V og D.
- ✅ Tekst «X av N / Z %» grønn når jaktmålet er nådd, rød ellers.
- ✅ Mangler øvelsesskudd i stillingen → grå silhuett + «øv på stillingen».
- ✅ (i) med forklaringstekst (grønn/rød ordene fargelagt).
- ⚠️ Radien-tabell og `RING_STEP_CM` er fortsatt PLASSHOLDERE (spec §10) —
  tallene er ikke kalibrerte.

## Meny
- ✅ «Send melding» → «Melding til utvikler». Tittel blir e-postens Subject.
- ⛔ **Direkte sending** (uten e-postapp) krever backend/SMTP — se backend_spec §10.
  Foreløpig brukes ACTION_SENDTO (mailto) til mrleenudler@gmail.com.
- ✅ Klikk utenfor åpnet meny lukker den (både Avstand og Meny).

## Profil
- ✅ Visningsnavn tillater latinske tegn inkl. æ/ø/å (ikke bare ASCII), lengdecap 24.
- ✅ **Tema:** default LYS (startet feilaktig i mørk før). Meny-overskrift «Velg
  visningsprofil». Knappen viser gjeldende modus.
- ✅ «Mine jaktlag og skytterlag» liste + «+ legg til …» → fullsides Lag-meny.
- ✅ «Mitt jaktmål»: måltallet 2 pt større + bold; «Hvorfor er noen tall røde?»
  flyttet til (i) høyrejustert på overskriftslinjen.
- ✅ «Avanserte innstillinger» er nå egen knapp → undermeny (Mine våpen, Flytt,
  Slett, Venstrehåndsmodus, Utvikler).
- 🟡 **Venstrehåndsmodus:** speiler UI horisontalt (LAYOUT_DIRECTION_RTL på
  aktivitetsroten). Enkел implementasjon; kan finpusses per skjerm ved behov.
- ✅ **Utvikler-meny** (flagg `DevTools.ENABLED` slår den av):
  - «Generer serie»: semi-tilfeldig serie vektet mot midten/areal (rundt 6–8),
    lagres som normal serie.
  - «Dummy scan»: fabrikkerer et treffsett og åpner ResultActivity (uten kamera).
    (Brukte ikke Real 1.jpg direkte — bildet er ikke pakket i appen; fabrikkert
    sett er en stabilere test. Kan endres hvis du vil pakke bildet i assets.)
  - «Vis oppstartsmelding hver gang».

## Lag (fullsides, front-end-skjelett)
- ✅ «Mine lag» som innrykkede knapper; klikk velger → Opprett-knapp blir «Rediger
  lag». Roller som knapper; ingen invite-toast. «Venter på backend-implementering».
- 🟡 Slett/forlat/oppløs/overfør styres av `Team.memberCount` lokalt; ekte
  medlemskap/roller/nærliggende-lag krever backend (backend_spec §4).

## Jakt
- ✅ «Tilbake»-knapp nederst t.h.; «Logg jaktskudd» ikke lenger uthevet.

## Logg jaktskudd
- ✅ «Annet» krever tekst (> 1 tegn), toast ellers.
- ✅ «Informasjonen kan redigeres senere» (fjernet «i menyen»).
- ✅ «Avstand» → «Skuddhold». Ikke tillat 0 som første siffer.
- ✅ Posisjon-overskrift fjernet fra side 2; toast ber om «hvor omtrent langt dyret løp».
- ✅ Bomskudd eller «Dyret ble ikke funnet» krever ikke tall.
- ✅ Blokkerings-bug fikset (toast-kø + posisjonskrav fjernet fra validering).

## Se registrerte skudd
- ✅ Klikk-og-hold merker flere → Slett øverst t.h., Avbryt nederst t.v.
- ✅ Større piler (⟸ ⟹), Rediger øverst t.h., OK fast plassert (flytter seg ikke;
  samme posisjon når én pil mangler).

## Venner (front-end-skjelett)
- ✅ Lag som innrykkede knapper; klikk → lagside med medlemmer (lag som overskrift).
- ✅ Ny opp/ned-knapp (etter `up_down_arrows_model.jpg`) t.h. → popup «flytt opp/
  ned/avbryt». «<=»-kollaps ved siden av for å skjule medlemmer.
- ✅ «Jeg vil dele:» har «<=»-kollaps (ned-pil for å utvide igjen).
- ✅ Etikettendringer: «Antall øvelsesskudd (anbefalt)», «Gjennomsnittlig score»,
  «Min utvikling», «Mine jaktlag og skytterlag», «Min hjemkommune», «Telefon»
  nederst.
- ✅ «Lagre» lukker og går til HOVEDsiden (ikke menysiden).
- ✍️ **Bruker-ID / misbruk:** designspørsmål besvart i backend_spec §3.1
  (kort håndskrivbar ID, internasjonal kapasitet, rate-limit på søk,
  telefon-søk-karantene, IP-heuristikk).

## Spec-arbeid
- ✅ `bestefar_UI_spec-v0-4.md` §12 oppdatert med runde 5.
- ✅ `backend_spec.md` utvidet (§3.1 bruker-ID/misbruk, §10 direkte melding).
- ✅ Denne fila (`til_utvikler_v07.md`).
