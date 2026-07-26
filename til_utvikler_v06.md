# Tilbakemeldinger til utvikler — v0.6 (musingsUI runde 4)

Kopi av alle tilbakemeldinger fra `musingsUI.txt` med status og notater. Ordnet
etter tema. **Status:** ✅ implementert · 🟡 delvis / skjelett · ⛔ utsatt (krever
kjerne/backend) · ✍️ spec-arbeid.

## Avklaringer (besvart av eier før implementering)
- **Mitt jaktmål-rater:** erstattet med **1 av 7 / 13 / 20 / 50** (13 = nasjonalt snitt).
- **OCR:** implementert nå med Google ML Kit (on-device).
- **Venner/lag:** front-end bygget som skjelett; backend beskrevet i `backend_spec.md`.
- **Leveranse:** alt klient-side i én runde + tre spec-dokumenter (denne + UI-spec + backend-spec).

## Ikoner
- ✅ Nye stillingsikoner (liggende/sittende/knestående/stående) portert fra
  `UI/skytestilling-*.svg` → `ic_stilling_*`.
- ✅ Hjort-silhuetter (`hjort-front/side/skrå`) → `ic_hjort_*`, brukt i jaktloggens
  posisjonsvalg og «Se registrerte skudd».
- 🟡 Øvrige viltfigurer mangler (`UI/Ikon-prompt-*.txt` er prompter for å lage dem).
  Hjort brukes på alle arter inntil videre.
- ✅ `menu_icon_stats`: innhold i forstørrelsesglasset fjernet (jf. `UI/notater til ikoner.txt`).

## Scan / resultat
- ✅ **OCR-finpussing** (ML Kit): leser skjermens poengtall og sammenligner med
  detekterte treff. Avvik ≤ 0,2 → sømløs oppdatering (+ køing av bilde til utvikler
  hvis bildedeling er aktiv). Avvik > 0,2 → «Appen klarte ikke se treffene riktig …»
  med Forkast / Lagre med poengene på skjermen, og bilde-donasjonsdialog hvis
  bildedeling ikke er aktiv.
  - ⚠️ **UKALIBRERT heuristikk:** skjermlayouten er ikke modellert; vi trekker ut
    desimaltall i [0, 10.9] og matcher antall mot detekterte. Treffsikkerhet MÅ
    felttestes. Se `OcrVerifier.kt`.
  - ⛔ **Backend:** faktisk opplasting av dev-bilder er ikke koblet (køes lokalt i
    `filesDir/dev_uploads`). Se `backend_spec.md`.
- ✅ **Bare liggende bilder** godkjennes i gatingen (`CaptureActivity`:
  `sensorLandscape` + portrett-frames droppes).
- ✅ **Innskyting:** sesongens (og våpenets) første serie sjekkes for tydelig
  feilkalibrering (senter vs. spredning). Dialog «Er våpenet innskutt?» Ja→registrer,
  Nei→forkast. Senere på dagen: hvis dagens to første serier har omtrent samme bias
  → samme dialog, Nei → «Skal vi lagre de to siste seriene?».
  - ⚠️ `RING_STEP_CM` er fortsatt plassholder; miscalibration-detektoren bruker
    skalafritt forhold bias/spredning, så den er robust mot dette.
- ✅ **Identiske serier:** varsel «Lik en tidligere serie» ved lagring.
- ⛔ **Flere treff enn reelt** (feltobservasjon): sannsynlig kamerabevegelse /
  multieksponering. Dette er et **CV-kjerne**-spørsmål (ikke løsbart i UI-laget).
  Anbefaling i `backend_spec.md` / kjerne-notat: stram inn auto-capture-stabilitet
  og undersøk dedup av nære treff i `hits`/`overlap`.
- ✅ Resultatkort ellers som før (skive + stigende poeng + blyant + Ikke lagre/OK).

## Startskjerm
- ✅ Ikoner for våpen/jakt/stilling fjernet fra baren; kun **Avstand, Innsikt, Meny**.
  Stilling promptes kun etter scan; våpen ligger i Profil, jakt i menyen.
- ✅ «## øvelsesskudd denne sesongen» øverst.
- ✅ Stillingsprompt: fire stillinger vertikalt med ikoner + antall skudd; hjelpemidler
  (anlegg/reim) horisontalt som radio-toggler («uten» = deaktivert). Benk fjernet.

## Tema
- ✅ Lys brunfarge erstatter Material3-lilla (`colors.xml` / `themes.xml` + `-night`).
- ✅ Lys/mørk/system-veksler øverst til høyre i Min profil.

## Meny
- ✅ Meny-ikon 10 % mindre. Profil→**Min profil** (øverst), **Søk** nederst.
  Historikk fjernet, «Avansert statistikk»→**Mer statistikk** flyttet nederst i
  Innsikt, «Om appen» fjernet (→ oppstartsmelding), **Deling** fjernet (ligger nå i
  Profil/Venner), «Send melding»→**Gi tilbakemelding til utvikler**, «Rediger våpen»→
  **Legg til våpen** flyttet til Profil/Avanserte innstillinger, **Jakt** lagt til.

## Optikk / ammunisjon / kalkulator
- ✅ Fjernet for å forenkle. `KalkulatorActivity` og optikk-/ammo-delene av
  våpendialogen er utkommentert/fjernet med notat; `OpticProfile`/`clickSuggestion`
  beholdt utkommentert for evt. gjeninnføring.

## Om appen → oppstartsmelding
- ✅ Vindu 1 (velkomst, to avsnitt) + vindu 2 (bildedeling: Nei takk/Godkjenn).
  Vises første gang og på nytt hvis `STARTUP_MSG_VERSION` bumpes.
- ✅ Tutorial omdøpt «Hvordan bruke appen» med omskrevne tekster (Velkommen, Scan
  serie, Velg skytestilling, Innsikt). «Hopp over»→«Avbryt».

## Profil
- ✅ Visningsnavn (var Navn/Kallenavn), fødselsår gated 2–120, «Legg til jaktlag
  eller skytterlag», «La venner finne meg» (default av), fortløpende lagring (ingen
  Lagre-knapp), «Mitt jaktmål» med ny ordlyd + rater + (i), «Dele med venner» fjernet,
  Avanserte innstillinger (Mine våpen, Flytt, Slett).
- 🟡 **Lag-oppretting:** rolleflyt (leder / for leder / be leder) som skjelett;
  invitasjon via kontaktliste / e-post / telefon og nærliggende-lag-listing krever
  backend (`backend_spec.md`).
- ⛔ **Forskning + alder:** valgt **egenrapportert alder** (ingen bevis-krav) — endelig
  avklaring hører hjemme i forskningsprotokollen (REK/SIKT).

## Venner
- 🟡 Front-end-skjelett: legg-til-flyt (søk/ID/QR), delingsvalg (visningsnavn perma),
  lag med venner gruppert under (offset) + øvrige venner, flytt lag opp/ned, gråing av
  lista når deling er deaktivert, endre visningsnavn på venn (alias). Tegnbegrensning
  + ASCII på visningsnavn.
- ⛔ Ekte vennedata, invitasjoner, QR og deling krever backend.
- ✍️ **Brukernavnsensur:** anbefalt som backend-/moderasjonssteg (navnet finnes bare
  for andre via server). Ikke klient-side nå.

## Jakt
- ✅ Menyvalg «Jakt» med to knapper: Registrer jaktskudd / Se registrerte skudd
  (scan-knappen vises ikke her).
- ✅ Datovelger (kalender, «3. mars 2026», dagens dato forhåndsvalgt).
- ✅ Checkbox «Del med forskning» (av = forenklet visning), deaktiveres ved «Annet».
- ✅ Posisjon under viltknappene med rød pin; avstand som tekstboks; «Informasjonen kan
  redigeres senere» under Neste; toast ber om både vilt og avstand.
- ✅ Side 2: tre hjort-silhuetter for «Velg dyrets posisjon»; «Dyret løp ca X m»;
  Ettersøk/Bomskudd på linjen under; «Dyret ble ikke funnet» under Ettersøk.
- ✅ «Se registrerte skudd»: liste (vilt-ikon, stedsnavn, dato) + pil venstre/høyre
  (løper ikke rundt).

## Diverse
- ✅ Skjermtastatur-konflikt: alle dropdown-paneler og lange dialoger er scrollbare.
- ✅ Alle fritekstfelt: stor forbokstav som default.
- 🟡 **Data over reinstall:** oppdatering bevarer data (Android Auto Backup). Full
  overlevelse ved avinstaller+reinstall krever konto/eksport — se `backend_spec.md`.
- ✍️ (i)-ikoner: hvilke elementer som skal ha info-ikon avgjøres løpende (utviklers
  vurdering, jf. eierens note).

## Ikke implementert (eksplisitt utsatt av eier)
- Jegeropplæring.no-faktasnutter/lenker.
- Kobling mot statens app / Jegerprofil.
- Tips + Kunnskapsbank.
