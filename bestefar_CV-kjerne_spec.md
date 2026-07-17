# Kravspesifikasjon — Målskive-app for rifleskyting

**Status:** Utkast til første Fable-iterasjon
**Formål med dokumentet:** Underlag for en Claude Code Fable-økt som skal utforme arkitektur og skrive et første kjørende skjelett. Dokumentet definerer *hva* som skal bygges og hvilke kontrakter som er faste; det overlater bevisst arkitektur- og implementasjonsvalg der Fable har et reelt avveiningsrom til Fable selv.

---

## 1. Oversikt

Appen tar bilder av en digital målskive for rifleskyting, kjører computer vision on-device for å analysere treff, og lagrer resultatet. Kjernen er en CV-oppgave; UI og statistikk legges oppå. En liten backend støtter datasync, feilrapportering og forskningsdatainnsamling.

Prosjektet er selvstendig og har ingen kobling til andre systemer.

### Målbilde for denne iterasjonen

Fable skal, med den eksisterende Python/OpenCV-pipelinen som **referanse**, utforme og skrive:

1. En delt CV-kjerne i C++ (bygget på OpenCV), portert fra Python-referansen.
2. Native mobil-skjelett for Android og iOS (Prioriter Android først) som kaller inn i kjernen via FFI.
3. En liten backend med tre atskilte ansvarsområder (se §5).

Fable skal **ikke** portere Python-koden 1:1 uten vurdering; den skal bruke den som referanse for algoritmene og etablere en C++-arkitektur som er riktig for on-device bruk på mobil.

---

## 2. Arkitektur — fast og åpent

### Fast (besluttet, skal ikke omgjøres i økta)

- **On-device analyse.** CV-en kjører på enheten, ikke mot en backend.
- **Delt C++-kjerne på OpenCV**, med tynt native UI per plattform. Grunnen er at kamera og bildeprosessering er kjernen i appen, at OpenCVs C++-API er en direkte overgang fra den eksisterende Python-koden, og at den feilutsatte CV-logikken da ligger ett sted og verifiseres én gang.
- **Native mobil-apper** (Android/iOS) framfor delt UI-rammeverk, begrunnet i tett kobling mot native kamera og auto-capture på live kamerastrøm.
- **Auto-capture** — appen avgjør selv når et bilde skal fanges, framfor manuelt utløst. Vi ser for oss en modell etter scannere for QR-kode, dokument eller kredittkort, med en ramme som brukeren skal holde apparaturen innenfor, før et bilda tas automatisk.

### Åpent (Fable skal bestemme, dette er hva kvoten betales for)

- Intern arkitektur i C++-kjernen (modulinndeling, minnehåndtering, trådmodell).
- FFI-strategi mot Kotlin (JNI) og Swift (C-header), og hvordan kjernen pakkes for begge plattformer.
- Backend-teknologi og hvordan de tre backend-jobbene struktureres.
- Bygg-/CI-oppsett for et prosjekt med delt native kjerne og to plattform-targets.

---

## 3. CV-kjernen — kontrakt

Kontrakten mellom kjerne og resten av appen er det stabile grensesnittet. Den er definert her fordi den styrer både UI- og backend-laget; alt annet i de lagene kan endres uten å røre kjernen.

### Eksisterende pipeline (referanse, allerede skrevet i Python/OpenCV)

- Detektering og crop av skjermen
- Deteksjon av senter ved gradient voting
- Justering av senter og skew ved polarbilde og graftilpasning
- Deteksjon av treffunkter (flere steg)
- Deteksjon av cluster senter ikon (Ikke ferdig testet, men prioriteres ikke nå på grunn av lav marginal verdi og tidspress)

### Input til kjernen

- Et kamerabilde (stillbilde valgt av auto-capture, se §4) av målskiven.
- Format/oppløsning/fargerom: **må spesifiseres** ut fra hva native kamera leverer per plattform — Fable definerer et plattform-uavhengig internt bildeformat kjernen tar imot.

### Output fra kjernen

Per analysert bilde:

- Treffkoordinater (forslag: relative polarkoordinater).
- Poeng per treff, med desimaler, som vist på apparatskjermen. 
- Et **kvalitets-/konfidensmål** for analysen, slik at UI kan skille sikre resultater fra tvilsomme, og slik at feilede analyser kan rutes til feilinnsending (§5). (Merk: Kvalitetsmålet skal være at beregnede poeng ikke matcher maskinens score, men OCR er ikke implementert enda. Det skal ikke prioriteres akkruat nå.)
- Tidsstempel (kreves av forskningsdelen, §6).

Det eksakte output-skjemaet defineres av Fable, men **må** inneholde de fire punktene over.

---

## 4. Auto-capture — ny funksjonalitet

Auto-capture er **ikke skrevet ennå**. Den skal spesifiseres som ny funksjonalitet i denne iterasjonen, men bygger på eksisterende deler.

### Gjenbruk

Den kontrast-drevne ROI-en som allerede identifiserer apparaturens ramme, gjenbrukes som grunnlag for auto-capture. Auto-capture er i praksis en **trigger-beslutning** lagt oppå ramme-deteksjonen: gitt at rammen er funnet i en live kamerastrøm.

### To atskilte kriterier

Auto-capture trenger to logisk uavhengige vurderinger, som ikke må blandes sammen:

1. **Stabilitet** — ROI-en må holde seg i ro over flere påfølgende rammer, ellers fanges bildet midt i en bevegelse.
2. **Bildekvalitet** — omfatter fokus/skarphet, eksponering, og at hele skiven er innenfor rammen.

### Åpne parametere (skal IKKE finnes på fra teori)

Terskelverdiene for begge kriteriene — hvor mange rammer stabilitet krever, hvor stramt skarphetskravet er, hvor stor andel av skiven som må være synlig — **kan ikke bestemmes teoretisk**. De må kalibreres mot faktisk maskinvare. Fable skal eksponere disse som eksplisitte, konfigurerbare parametere med rimelige startverdier merket som *ukalibrerte*, ikke hardkode terskler som om de var kjente.

---

## 5. Backend — tre atskilte ansvarsområder

En liten backend er nødvendig. Den har tre distinkte jobber med ulike krav; de skal holdes adskilt.

1. **Statistikk-sync.** Lett. Brukerens egne resultatdata (se §3-output) synkes for statistikk-laget. Kun data for statistikken lagres som standard.

2. **Innsending av feilede analyser.** Bilder der analysen feiler eller får lav konfidens kan sendes inn (opsjon). Dette henger sammen med forbedring av CV-en over tid. Bildelagring generelt er en opsjon brukeren styrer; standard er kun statistikkdata.

3. **Forskningsdatainnsamling.** Se §6. Skal være strukturelt adskilt fra brukerens egne statistikkdata.

---

## 6. Forskningsdata — struktur og separasjon

Forskningsformålet er å studere effekten av øvelsesskyting og å bevisstgjøre jegerne på deres ferdighetsnivå. Resultater fra jakt skal  kunne rapporteres inn på sikt, men UI for dette er ikke en prioritet.

### Konsekvenser for datamodellen

- **Tidsserie, ikke enkeltskudd i vakuum.** Effekt av øvelsesskyting over tid krever at utfall er tidsstemplet og koblet til økt/serie, slik at statistikk-laget kan aggregere per skytter over tid.
- **Per-skytter-aggregering.** Ferdighetsnivå-bevisstgjøring forutsetter at data kan grupperes per skytter og følges over tid.
- **To resultattyper.** Treningsresultater og jaktresultater er ulike datatyper med potensielt ulike felter, og — viktig — trolig ulik personvern-/samtykkeprofil. Jaktdata er mer sensitivt enn treningsdata.

### Krav til separasjon og samtykke

- Forskningsdata skal være **strukturelt adskilt** fra brukerens egne statistikkdata fra dag én. Dette er dyrt å ettermontere, derfor er det med her. Begge gruppene skal lagre både øvings- og jaktdata, men brukerens egne data skal ha muligheten for å lagre bildene i tillegg til bare treffpunktene.
- Innsamling til forskning krever **eksplisitt samtykkeflyt**, adskilt fra ordinær app-bruk.
- Fable skal ikke løse de juridiske/personvernmessige spørsmålene, men datamodellen skal designes slik at samtykke, separasjon og de to resultattypene er reflektert i strukturen.

### Åpent punkt (må defineres av deg, ikke av Fable)

Det **konkrete feltinnholdet** i forskningsdatasettet er ikke avklart ennå. Fable skal etablere strukturen (økt, skytter, tidsstempel, resultattype trening/jakt) og la selve feltdefinisjonene være et tydelig merket, utfyllbart punkt.

---

## 7. Statistikk og UI — ikke spesifisert ennå

Statistikk- og UI-laget er ikke utformet. Arbeidet så langt har vært på CV-delen. Appens første iterasjon skal kun ha en knapp for "start", capture skjermen, og en resultatskjerm. Når man trykker OK på resultatskjermen, kommer man tilbake til Start.

Dette er ikke et hinder for denne iterasjonen: CV-kjernen og backend-datamodellen er det som skal formes først, og de avhenger av CV-kontrakten (§3) og forskningsstrukturen (§6), ikke av UI-en. CV-kontrakten er det stabile grensesnittet UI-laget senere bygges mot.

---

## 8. Oppsummert til Fable-økta

**Bygg:** delt C++/OpenCV-kjerne (portert fra Python-referanse), native Android/iOS-skjelett som kaller kjernen via FFI, liten backend med tre atskilte jobber.

**Fast:** on-device analyse, auto-capture, delt C++-kjerne, native apper, CV-kontraktens fire output-felter, strukturell separasjon av forskningsdata.

**Fable bestemmer:** kjernens interne arkitektur, FFI-strategi, backend-teknologi og -struktur, byggoppsett.

**Skal ikke finnes på:** auto-capture-terskler (kalibreres mot maskinvare), konkret feltinnhold i forskningsdatasettet (defineres av eier).
