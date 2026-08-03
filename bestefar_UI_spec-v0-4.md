# Bestefar — UI/UX-spesifikasjon v0.4

> **Addendum v0.6 (musingsUI runde 2–4).** Seksjonene under er den opprinnelige
> v0.4-speccen. Implementasjonen har siden utviklet seg gjennom tre
> musingsUI-runder; gjeldende UI er oppsummert i «§12 Endringslogg v0.6» nederst,
> og full tilbakemeldingskopi ligger i `til_utvikler_v06.md`. Ved konflikt
> gjelder §12 og faktisk kode (`android/app/src/main/java/no/bestefar/app/`).

Åpne punkter i seksjon 10.
Bruk placeholdere for grafikk som ikke er triviell å implementere.
Instruksjonen kan være uklar. Bruk egen dømmekraft, eller spør om veiledning.

## 1. Formål og designprinsipper

Appen har to formål som i praksis er allierte: den skal gi jegeren et presist bilde av egen skyteevne — hvilke hold og situasjoner som er forsvarlige, og hvilke som bør unngås — og den skal samle strukturerte data til forskning på effekten av øvelsesskyting på jaktutfall. Det som gjør dataene forskbare, er det samme som gir jegeren god egeninnsikt; den eneste reelle konflikten er friksjon, løst med kontekstarv og forhåndsvalg.

**Bruksmodell (styrende):** Appen brukes ved øvelsesskyting — ved planlegging før jakt og ved tilbakeblikk etterpå. Den er aldri fremme i skuddøyeblikket. Verdien ligger i at jegeren gjennom trening etablerer en internalisert grense for akseptable skuddsituasjoner, og tar *regelen* med seg i felt. Dette er for-tanke satt i system.

Prinsipper:
- **Kontekstarv i tre nivåer.** Våpen (+ eventuell ammunisjon) bekreftes én gang per dag. Dersom bare ett våpen er registrert, skal det være default og ikke promptes. Avstand er default 100m, men kan enkelt endres. Skytestilling settes per serie som et prompt etter skivescan. 
- **Verdi før forpliktelse.** Full funksjonalitet uten konto og uten samtykker. Forskningssamtykke og tilbud om online konto tilbys etter fem skutte serier. Valgene er Ja, Ikke nå, Aldri. Dersom Ikke nå velges, tilbys valget på nytt hver tiende serie. Valgene kan alltid endres i settings. Forskningssamtykke kan trekkes tilbake. 
- **Jegerens språk, ikke statistikerens.** Frekvens («9 av 10»), ikke desimaltall eller tekniske mål, i alle beslutnings- og kompetanseflater. Usikkerhet kan vises som spenn der den er stor. Tekniske mål (σ, R95, MOA) finnes kun som "mer statistikk" i settings, og i forskningseksport.
- **Ett forsvarlighetsbegrep.** Brukeren velger akseptabel skadeskytingsrate; appen bruker samme kriterium overalt og sier aldri «trygt».
- **Ikonspråk.** Mest mulig ikoner (stillinger, arter, vinkling, utfall, vær), alltid med i-tegn som åpner forklaring, og tekstalternativer (WCAG). Funksjonalitet for tekst på alle tilgjengelige språk med tilfredsstillende dekning for fagterminologien.
- **Offline-først.** Alt fungerer uten dekning; opplastinger køes.

## 2. Navigasjon

Seks faner: **Våpen**, **Avstand**, **Stilling**, **Innsikt**, **jakt**, **Meny**. Samtykke- og datastyring under Profil i Meny.

Stor sentrert knapp: "scan serie"


### Dags- og øktoppsett
- **Våpen** promptes én gang per dag, etter første serie, deretter forhåndsvalgt og synlig som valg. Med bare ett registrert våpen velges det automatisk uten prompt. Ved ammunisjonssplitt (seksjon 6) bekreftes ammunisjon i samme dagsprompt.
- **Avstand** 100m som standard. 
- **Skive:** Kongsberg digital skive, fast skala. Ingen skivevalg i v1.

### Capture-løkke
Per serie: Scan knapp → auto-capture → stillingsprompt → analyse → resultatkort → ok/avslutt. Stillingsprompten viser fire hovedstillinger (liggende, sittende, knestående, stående) med modifikator-chips (uten / anlegg / reim); siste modifikator huskes per hovedstilling. **Benk** er egen inngang ved innskyting. Innstilling «ikke spør — manuell stilling» erstatter prompten med synlig chip. I øvelsesmodus er stillingen forhåndsvalgt.

### Resultatkort
- Skuddene plottet på skivegjengivelse, med **korrigering av skuddmerker** (blyant-ikon for å justere poeng og desimaler med popup som på en timer innstilling). Korrigerte analyser tilbys sendt til feilanalysekanalen (mikrosamtykke per innsending).
- Poengsum.
- Langsiktig gjennomsnitt og spredning 
- **Klikk-forslag** kun når offsetet er skjelnbart fra støy (per akse > ~2σ̂/√n); ellers «innenfor støy — ikke juster». Krever klikkverdi på optikken. Anvendte justeringer logges som hendelser på oppsettet.
- Status: teller / teller ikke i evidensgrunnlaget (benk teller ikke i kompetanse).

### Øktoppsummering
Serier med stilling-ikon, poeng og frekvensbudskap; endringer i kompetanse; usendte opplastinger i kø.

## 4. Flate: Jakt

### Jaktmodus
Valgfritt dagsoppsett: art(er), våpen, stedslogging av/på (huskes). Aktiv jaktmodus gir ett-trykks, forhåndsutfylt hurtiglogg.

### Hurtiglogg — tre steg, hanskevennlig
1. **Art** (ikoner: elg, hjort, villrein, rådyr, villsvin; «annet» finnes, holdes utenfor analyser).
2. **Hold**, deretter **vinkling** presentert som dyr i sirkel (BH-undersøkelsens format), med valgfri chip **«i bevegelse»**.
3. **Utfall** — tre kategorier: **dødelig** (operasjonalisert som «dyret løp kortere enn x meter», x artsavhengig), **skade**, **bom**.

Systemhentede metadata (dato, tid, posisjon når aktiv) legges på automatisk. Øvrige BH-parametere legges til senere.

Mål: under et halvt minutt, helt uten dekning.

### Totrinns utfall og ettersøk
Umiddelbart utfall kan oppdateres etter ettersøk (felt/avlivet, funnet død, friskmeldt, ikke gjenfunnet). Appen følger opp med ett stille spørsmål ved neste åpning eller dagen etter — nøytralt, aldri gjentatt purring. Skadedata er private som standard, aldri i delt kontekst.

### Stedsdata
Presis posisjon lagres **lokalt** når påslått (kartvisning over egne jaktskudd). Deling til forskning i grovere oppløsning valgt i samtykket. Vær hentes automatisk (MET) når posisjon finnes — også ved økter.

## 5. Flate: Innsikt

Åpner kompetanseoversikten; kapabilitetskart bak segmentkontroll. Begge skjult til første økt — for nye brukere og ved hver sesongstart.

### Kompetanseoversikt (primærvisning)
Sidevendt dyrefigur for utvalg av arter(norsk storvilt). Rotasjonspiler på hver side veksler vinkling; holdvelger til høyre. Figuren skalerer *subtilt* med hold — utelukkende som ekstra visuelt signal om at holdet er endret; **farge** (andel dødelige treff på lang sikt. skala fra rødt til grønt, der brukerens valgte skadeskytingsrate er gult) bærer hovedbudskapet, ikke størrelse.

Modellen: gitt homogen spredning fra siktepunktet (målt på papir) og radien på dødelig sone ved valgt art × vinkling, beregnes andelen skudd som havner innenfor. Vinkling virker ved å krympe den projiserte dødelige sonen; **minste halvakse** er den bindende radien (ikke gjennomsnitt — den korte veien ut av sonen avgjør).

To utfall vises:
- **Dødelig treff** — innenfor radien.
- **Skade** — alt annet. (Rene bom eksisterer i virkeligheten, men er på papir umulig å skille fra skade og slås derfor sammen; jf. seksjon 8.)

Presentasjon i frekvens og **ren gevinstramme**: «Feller rent ~9 av 10.» Tapssetningen er bevisst utelatt — appen er aldri med i skuddøyeblikket, så ingen impulskontroll tapes, og dobbel formulering blir clutter som senker bruksterskelen. Brukerens valgte skadeskytingsrate brukes som grense, og **maks forsvarlig hold** leses av per stilling. Utestet stilling: «ikke testet». Bevegelige mål utenfor modellen (veikart); «i bevegelse»-skudd vises separat.

### Kapabilitetskart (sekundærvisning) [Usikker på denne - gir den mening?]
Matrise avstand × stilling med artsvelger. Celletall i frekvens (andel dødelige treff) med spenn ved lite data. Farge fra det felles forsvarlighetskriteriet (rate + minstekrav til P(dødelig)), så kart og kompetanseoversikt aldri motsier hverandre. Ekstrapolerte celler: tilde-prefiks + stiplet ramme. Utestede: «ikke testet» som inngang til øvelsesmotoren. Celledetalj: spenn, antall skudd, dato for siste måling, underliggende serier.

### Sesong og historikk
Statistikk nullstilles per sesong = jaktåret 1. april–31. mars. Kompetanse og kart regnes på inneværende jaktår. Historikk i Meny viser tidligere sesonger som frosne kart, pluss utviklingskurver over sesonger og egne jaktutfall. **Presentasjon av langsiktig tilstand:** vis gjeldende langsiktig gjennomsnitt, og la nye serier falle inn i det bildet — ingen «du ligger over/under»-dom per serie (én serie er for støyende til å dømmes; vis tilstand og usikkerhet, ikke avvik-fra-forventning). 

**«År uten skadeskudd»** [Denne hører kanskje til kompetansekortet?] (når jaktskuddvolum finnes): et gripbart, personlig tidsmål. To designgrep gjør det ærlig uten matematisk clutter:
- *Vis raten i tidsformat, ikke inversen.* Regn ikke 1/p (som eksploderer og hopper ved lave rater), men vis den stabilt estimerte raten skalert med brukerens skuddvolum: «omtrent hvert X-te jaktår for deg». Samme underliggende p, langt roligere tall.
- *Skrittvis presisjon i stedet for terskel.* Ingen ventetid på tom skjerm: tidlig vises en grov kategori («sjelden» / «av og til» / «ofte»), som smalner til et tallanslag etter hvert som skuddene samler seg. De grove kategoriene tåler mye estimeringsusikkerhet og er derfor ærlige fra dag én.
- En **(i)** forklarer bevegelsen på jegerspråk — «anslaget bygger på hvor mange skudd du har logget; jo flere, jo sikrere» — uten konfidensintervall. Plasseres i refleksjon/historikk, aldri som et grønt lys i planleggingen.

### Øvelsesmotoren 
-> Denne bør fungere som et popup-forslag, men bare dersom det er behov for spesifikk øving, og med fornuftige mellomrom, slik at den ikke blir masete. Den kan for eksempel trigges etter en serie der brukeren har valgt en vanlig øvelse fremfor en nyttig øvelse. "OK" setter innstillingene for brukeren.
En øvelse = stilling + avstand(norm 100m). **Kjerneprinsipp — lukk trening/felt-gapet:** motoren sammenligner brukerens *treningsfordeling* av stillinger mot hans *jakt-/jaktformfordeling*, og prioriterer stillingen med størst underdekning. Speiler brukerens egen inkonsistens tilbake («skutt 40 % av jaktskuddene sittende, trent 5 %») heller enn å foreskrive normativt. Inntil brukerens egne data er robuste, brukes generell jaktstatistikk som forhåndsinformasjon for hva som er relevante stillinger og hold.

Prompt-logikk: (1) relevant + null data → «bør etableres»; (2) relevant + tynt/eldet → «bør bekreftes»; (3) irrelevant → nevnes ikke uoppfordret. Relevans fra jaktform/jaktlogg, tynnhet fra skuddantall. Aldri prompt på fravær alene. Forslag vises som kort på Økt-flaten og som «test denne» fra celler/stillinger.

## 6. Flate: Profil

- **Skytterprofil:** fødselsår (kun for 18-årsgrense på forskningssamtykke), skytterlag, jaktlag (mulighet for å være med i flere lag) valgt **skadeskytingsrate-grense** med forklaring; endring virker umiddelbart og logges.
- **Våpenkartotek:** våpen, optikk med klikkverdi (cm/klikk@100 m).(i) forklarer at appen kan gi klikkforslag når den er utfylt, **ammunisjonssplitt av/på per våpen** (av som standard). På: hvert våpen+optikk+ammo-oppsett får egen MTP-historikk, men med opsjon for å slå dem sammenfor statistikk pr våpen, klikklogg og evidensgrunnlag; ammo bekreftes i dagsprompt. Av: ammo som valgfri metadata (forskningskovariat). Bryteren virker fremover.
- **Data og samtykke:** to separate samtykker (trening/jakt), stedsgranularitet for deling, «flytt til ny telefon» (kryptert eksportfil med historikk + pseudonym-ID), Android Auto Backup-status, sletting (lokalt + sletteanmodning via pseudonym-ID).
- **Hjelp:** ikonforklaring, forklaring av skadeskytingsrate/forsvarlighetskriterium, om forskningsprosjektet, kilder for dødelig-sone-radier.
- Mulighet for innlogging via Google/Apple/Email/telefonnummer for backup av profil

Én skytter per installasjon antas.

## 7. Onboarding og samtykkeflyt

Tre korte skjermbilder (hva appen gjør, ikonspråket, «alt lagres lokalt(hvis ikke konto/backup ønskes)») → rett i første økt. Etter tredje fullførte økt: forskningssamtykke, med konkret visning av hva som ville blitt delt. Jaktsamtykke tilsvarende ved første bruk av jaktloggen. Avslag endrer ingen funksjonalitet; samtykke kan gis/trekkes når som helst. Forskningssamtykke krever 18 år.

## 8. Statistikk- og modellnotater (UX-relevante)

- **To situasjoner, holdes fra hverandre:**
  1. *Kompetanse (papir):* skiveresultat → P(dødelig) vs. P(skade). To utfall, ingen silhuett. «Skade» = alt utenfor dødelig radius; rene bom slås sammen med skade fordi de på papir ikke kan skilles, og fordi et hold der bom er mer sannsynlig enn treff uansett er uforsvarlig. Beslutningsstøtte trenger ett tall, ikke to.
  2. *Jaktlogg (observert):* dødelig / skade / bom som tre kategorier. «Dødelig» = «dyret løp kortere enn 100 meter»  — et observerbart feltkriterium. Skille skade/bom er reelt her (ettersøksplikt utløses av skade).
- **Referansepunkt:** kompetanse måles om siktepunktet. Jaktrelevant spredning er √(σ² + offset²). MTP brukes kun der presisjon er riktig størrelse: klikk-forslag og benk.
- **Homogen spredning fra siktepunktet antas** (sirkulær normal). P(dødelig) = massen innenfor dødelig radius ved valgt vinkel.
- **Framing:** prospektteoriens risikovillighet-i-tapsdomene forutsetter akutt sanntidsbeslutning. Appen er alltid for-tanke/refleksjon, aldri sanntid, så den effekten er uansett svak — og siden dobbel framing gir clutter som senker bruksterskelen, brukes **ren gevinstramme** på kompetansekortet. Tapsformulering er bevisst utelatt; valget bør ikke reverseres uten å ta hensyn til bruksterskelen.
- **Benkens særrolle:** måler våpen+ammo (egenspredning, om MTP); utenfor kompetanse, brukes til å dekomponere skytter- fra våpenkomponent.
- **Aldring:** intervall vokser med tid siden siste måling; punktestimat står.

## 9. Ikoner og tilgjengelighet

Ikonsett for stillinger (silhuetter med modifikatormerker), arter, vinkling (vinklede dyr), utfall. Tekstalternativ per ikon; i-tegn på alle ikonskjermer.  Utendørsbruk: høy kontrast, mørk modus (kan det sensorstyres? Bør i tilfelle også være valg).

## 10. Åpne punkter

1. **Dødelig-sone-radius per art × vinkling** — tabellverdier (minste halvakse ved hver vinkel), kildebelagt. Dette er nå hele sonedatajobben (redusert fra dobbel silhuett til én radiustabell).
2. **Vinkeltaksonomi venter på primærkilde** — provisorisk 30°/60° + side/front/bak beholdt, men *ikke verifisert*. Hentes fra BH-materialet før implementasjon; skytter-oppgitt-vinkel er en kjent målefeilkilde å håndtere i forskningsdesignet.
3. **Navn og språk** — bokmål i v1; app-navn avklares før butikklansering. Støtte for systemspråk / andre relevante språk.
4. **Veikart:** papirskive-støtte med auto-gjenkjenning; dyrefigur-CV; bevegelige mål i modellen; personlige målsettinger per stilling; **gamifisering av øvingen** (bør vurderes på sikt — utmerkelser, progresjon, streaks e.l. for å øke treningsfrekvens; må balanseres mot at appen ikke skal oppmuntre til uforsvarlige skudd eller gjøre skadeskyting til «poengtap» som frister til underrapportering).

## 11. Avhengighet til CV-kjernen (teknisk)

UI-prosjektet skal **ikke forke** OpenCV/C++-kjernen. Konsumer den som **pinnet avhengighet** — eget repo, tagget versjon, hentet via submodule eller pakket artefakt — så UI bygger mot en kjent kjerne-commit uten en divergerende kopi å vedlikeholde. Må kjernen endres for UI-ens skyld, gjøres det som en versjonert endring i kjerne-repoet med bump av pinnen — bevisst og sporbart, ikke drift i en fork. Instruks til Claude Code: legg kjernen inn som pinnet submodule/artefakt på gitt tag, bygg UI mot det. Kobling til kjernen gjøres via hovedskjermens "Scan" knapp.

## 12. Endringslogg v0.6 (musingsUI runde 2–4)

Denne seksjonen overstyrer eldre beskrivelser der de er i konflikt.

### Navigasjon (endret fra §2)
- **Tre ikonknapper øverst:** Avstand, Innsikt, Meny. (Våpen, Jakt og Stilling er
  IKKE lenger i baren.) Hvitt motiv på sort med grå ramme; valgt = fylt/markert ramme.
  Meny-ikonet er 10 % mindre. Liggende: 60 % bredde; stående: 80 %.
- **Avstand** og **Meny** åpner som dropdown-paneler (trykk igjen lukker; kun avstand
  lukkes ved klikk utenfor). **Innsikt** er fullskjerm.
- **Scan serie**-knappen ligger i hovedflaten (nedre halvdel; full bredde liggende).
- **Stilling** velges KUN som prompt etter hvert scan (ikke i noen meny). Fire
  stillinger vertikalt med egne ikoner + antall skudd per stilling; hjelpemidler
  (anlegg/reim) horisontalt som radio-toggler («uten» = deaktivert). **Benk fjernet.**
- Hovedflate viser «## øvelsesskudd denne sesongen» øverst.

### Meny
Min profil (øverst) · Jakt · Venner · Mine serier · Gi tilbakemelding til utvikler ·
Hvordan bruke appen · Søk (nederst). Deling, Historikk, Om appen og optikk-kalkulator
er fjernet. «Mer statistikk» ligger nederst i Innsikt.

### Resultat/scan
- **OCR-finpussing** av poeng (ML Kit, on-device, UKALIBRERT heuristikk): ≤ 0,2 avvik
  → sømløs oppdatering; > 0,2 → «kunne ikke se treffene» med Forkast / Lagre med
  skjermpoeng, + bilde-donasjonsdialog.
- **Innskyting:** kalibreringssjekk på sesongens/dagens første serie(r).
- **Identiske serier:** varsel ved lagring. **Bare liggende bilder** gates.

### Profil
Visningsnavn, fødselsår (2–120), «Legg til jaktlag eller skytterlag», «La venner finne
meg», fortløpende lagring, tema-veksler (lys/mørk/system) øverst til høyre, «Mitt
jaktmål» (rater 1 av 7/13/20/50; 13 = nasjonalt snitt) med (i). Avanserte innstillinger:
Mine våpen (+ Legg til våpen), Flytt til ny telefon, Slett alle data. Optikk/ammo fjernet.

### Jakt (menyvalg)
To knapper: Registrer jaktskudd / Se registrerte skudd. Datovelger («3. mars 2026»).
«Del med forskning»-checkbox (av = forenklet visning). Posisjon m/rød pin under
viltknappene. Side 2: tre viltsilhuetter (hjort brukes på alle inntil videre) for
«Velg dyrets posisjon» + «Dyret løp ca X m» + Ettersøk/Bomskudd.

### Venner (front-end-skjelett)
Legg til venn (søk/ID/QR) · delingsvalg (visningsnavn perma) · lag med venner gruppert
under (offset) · flytt lag opp/ned · gråing når deling er av · endre visningsnavn (alias,
≤ 24 tegn ASCII). Ekte data/invitasjoner krever backend — se `backend_spec.md`.

### Tema
Lys brunfarge erstatter Material3-lilla; egen dag/natt-palett.

### Tutorial / oppstart
Oppstartsmelding (vindu 1 velkomst, vindu 2 bildedeling), vist første gang / ved
`STARTUP_MSG_VERSION`-bump. Tutorial = «Hvordan bruke appen» (Velkommen · Scan serie ·
Velg skytestilling · Innsikt), med «Avbryt»-knapp.

## 13. Endringslogg v0.7 (musingsUI runde 5)

Overstyrer eldre beskrivelser ved konflikt.

- **Innsikt** er nå en **matrise**: fem vilttyper (rader) rammet av jeger-stilling
  (topp), dyr-vinkling (bunn) og skuddhold (høyre kolonne). Silhuett skalerer med
  hold; frekvenstekst grønn = jaktmål nådd, rød = ikke, grå «øv på stillingen» der
  stilling mangler data. (i) forklarer fargene. Nytt klikk på Innsikt lukker den.
- **Meny:** «Melding til utvikler» (tittel = Subject); klikk utenfor lukker menyen.
- **Profil:** visningsnavn tillater latinske tegn (æøå); tema default lys, veksler
  «Velg visningsprofil», knapp viser gjeldende modus; «Mine jaktlag og skytterlag»
  liste; «Mitt jaktmål» tall bold+større + (i); «Avanserte innstillinger» egen
  knapp → undermeny (våpen, flytt, slett, venstrehåndsmodus, Utvikler-meny med
  Generer serie / Dummy scan / Vis oppstartsmelding hver gang).
- **Lag:** fullsides meny (Mine lag som innrykkede knapper, Rediger lag).
- **Jakt:** «Tilbake»-knapp; «Logg jaktskudd» ikke uthevet.
- **Logg jaktskudd:** «Skuddhold» (ikke Avstand), «Annet» krever tekst, ingen
  ledende 0, toaster avbryter hverandre, Bomskudd/«ikke funnet» krever ikke tall.
- **Se registrerte skudd:** klikk-og-hold flervalg-sletting, store piler,
  Rediger, fast OK-plassering.
- **Venner:** lag som innrykkede knapper → lagside med medlemmer; opp/ned-popup +
  «<=»-kollaps; delings-etiketter oppdatert; «Lagre» går til hovedsiden.
- **Oppstart:** fullskjerm-overlegg som dekker Scan-knappen; menylinje 5 % ned.
- **Optikk/ammo/kalkulator:** forblir fjernet (runde 4).

## 14. Endringslogg v0.8 (musingsUI runde 6)

- **Hovedskjerm:** kun antall øvelsesskudd + Scan-knapp (all annen clutter fjernet).
- **Samtykke:** jaktmål og forskning aldri på samme serie (jaktmål først,
  forskning tidligst 2 serier senere); ny forskningstekst; radioknapp-fiks.
- **Innsikt:** (i) på tittellinjen (fet grønn/rød, tema-tintet ikon); stilling- og
  vilt-valg som ikoner med ramme; viltnavn i default farge, tall grønn/rød; mørk-
  modus-farger; layout JJJJ / VT-rader + hold-kolonne / DDD; like store knapper.
- **Poeng-visning:** «Poeng:»-etikett; «… med anlegg/reim»; «Mitt gjennomsnitt for
  denne stillingen» uten KI; skiveringer korrigert (6 hvite + 5 sorte).
- **Profil:** «+ legg til nytt lag» som knapp; jaktmål over to linjer m/«Endre»;
  jaktlag-klikk åpner laget (TeamPage).
- **Jaktlag (TeamPage):** navn, Inviter medlemmer, medlemsliste (m/egen bruker),
  Rediger lag / Velg leder, Lukk — front-end-skjelett (avstemning/push = backend).
- **Jakt:** Tilbake går til hovedsiden.
- **Melding til utvikler:** subject/body i mailto-URIen.
- **Venner:** egen bruker i lag; venner i flere lag under hvert; flytt opp/ned
  fikset (Avbryt t.h.); delings-piler kraftigere + åpen som default; findable-
  varsel stabilt m/ny tekst.
- **Logg jaktskudd:** «Felling var vellykket»-checkbox + bekreftelse; «dyret løp»
  godtar 0 m (ikke sifre etter enslig 0); toast-tekst justert.
- **Se registrerte skudd:** tilbake → lista; store faste piler; nyeste samme-dag
  øverst; Rediger endrer alt; mørk-modus-silhuetter; flervalg «Slett alle» + popup.

## 15. Endringslogg v0.9 (musingsUI runde 7)

- **Topplinje:** flyttes faktisk ~5 % ned (2-args `addView(w,h)` nullet marginen
  før — nå 1-args så `topMargin` beholdes).
- **Oppstart:** «Vil du dele bilder …»-vinduet kan slås av i Avanserte innstillinger.
  Forskningens delings-checkbokser er auto-krysset første gang.
- **Popuper:** felles modell (setItems + Avbryt) på lister som manglet Avbryt.
- **Innsikt (ombygd):** (i) tydelig synlig (36 dp) t.h. for tittelen; valgt piktogram
  = sort (lys) / tekstfarge (mørk), uvalgt = grå; presentasjonssilhuett tekstfarget;
  viltnavn i default farge. «Stripe»-bug fikset (rammen ligger på en FrameLayout,
  bare bildet skaleres). Egne Innsikt-skalaer (liggende opp, sittende ned). Ramme:
  jeger-stilling høyrejustert over hold-kolonnen, dyr-vinkling venstrejustert under,
  hold-kolonne like bred som stående-ikonet og høy nok til å ramme matrisen, med
  tresifret tall + «m». Stillings-homogeniseringen (−65 %/+80 %) hører til
  stillingsvelgeren *etter scan*, ikke Innsikt.
- **Poeng-visning:** «Poeng:» foran totalen; dato «8. mars 2026    08:38» (luft
  mellom dato/tid); «Mitt gjennomsnitt for denne stillingen»; skivas ekstra hvite
  ytre halvring fjernet (maxR = 10). **Karusell** over viste serier i Serielogg med
  piler + OK; datoen animeres (3× → default, ~20 % ned/høyre, 1 s, avbrytbar) når
  dagen bytter. Dummy-scan gir 10 skudd.
- **Profil:** «+ legg til nytt lag» outlined (ikke uthevet); lag-rekkefølge følger
  Venner (sortOrder). Lag-«Slett»: eneste medlem → slettes; flere → forlat; eneste
  leder → oppløs for alle / overfør + forlat.
- **Jaktlag:** medlemmer klikkbare → karusell (piler går helt rundt + OK); lagleder
  merket «(Lagleder)» øverst, ellers alfabetisk; «Slett lag» i Rediger lag.
- **Venner:** dobbelt innrykk for lagmedlemmer.
- **Utvikler:** «Legg til venn» (navn + lag, 5 serier á 10 skudd).
- **Logg jaktskudd:** felling-checkbox også i forenklet (forsknings-av) flyt;
  posisjon logges også der; mørk-modus-tintede stilling-silhuetter. Skuddloggen viser
  kun «Felling vellykket/mislykket» (ikke «Utfall: Dødelig») i samme font. Rediger:
  etikett foran hvert felt, redigerbar dato, utfallsknapper stablet loddrett, «Skade»
  vist som «Ettersøk», «Dyret ble ikke funnet» kun ved Ettersøk.

## 16. Endringslogg v0.10 (musingsUI runde 8)

- **Fargeoppslag (rotårsak):** `Ui.themeColor` løser nå opp ColorStateList-attributter
  (særlig `android:textColorPrimary`) korrekt — tidligere ble ressurs-ID-en tolket som
  farge, så alt tintet med tekstfargen var «usynlig». Fikser vilt-silhuetter (lys/mørk),
  valgt piktogram i mørk, viltnavn-tekst og jaktlogg-silhuetter i mørk i én operasjon.
- **Topplinje:** margin 5 % → 2 % (3 % opp).
- **Forskning:** «Del med forskning»-bryter i Avanserte innstillinger (18-årsgate);
  auto-kryss kun når forskning er aktivert; nytt samtykke-spørsmål hver ny sesong
  (`researchConsentSeason`).
- **Oppstart:** bildedelings-spørsmålet popper ikke lenger; flyttet til «Del bilder med
  utvikler»-bryter i Avanserte innstillinger. Intro-vindu kun første gang / dev-flagg.
- **Innsikt:** (i) = UTF-8-glyf «ⓘ». Ramme-flukt: tre første stillinger venstrejustert
  (liggende over «forfra»-vilt), stående skjøvet helt til høyre over hold-kolonnen; luft
  mellom stående og 25 m; avstandsknapper like store som ikoncellene (60×54).
- **Stillingsvelger:** «Liggende» beholder aspekt (FIT_CENTER-ImageView i boks, ikke
  `iconSize`-kvadrat); «## skudd» teller skudd (`shotsCountByPosition`), ikke serier.
- **Serier:** «Lukk»-knapp nederst t.h., lista scroller ikke bak den.
- **Poeng:** «Poeng:»-overskrift i fet, samme størrelse som poengene. «Mitt gjennomsnitt
  for denne stillingen» brytes kun på stilling: «Denne sesongen: X / Totalt Y».
- **Logg jaktskudd:** «Felling var vellykket» uavkrysset default (→ bekreftelsesdialog
  ved uavkrysset). Rediger: «Bom»/«ikke funnet» fader ut, sletter og låser «Dyret løp».
- **Venner:** klikk på jaktlag åpner full lagside (deg selv i lista + «Rediger lag» for
  lagleder), samme som fra Min profil.

## 17. Endringslogg v0.11 (musingsUI runde 9)

- **Oppstart:** bildedelings-popup vises igjen — første gang, og én gang neste sesong
  hvis deling ikke er valgt (`shareDevImagesSeason`). Kommer etter intro-vinduet.
- **Avanserte innstillinger:** fjernet «spurt på nytt hver sesong»-hintet. «Fjern
  inaktiv lagleder» viser jaktlag-velger kun ved flere inaktive; ellers toast «Ingen
  inaktive lagledere funnet» (inaktivitet = backend §11, tom liste i skjelettet).
- **Innsikt:** alle fire stillinger inntil hverandre til venstre (smal 2 dp-stripe),
  stående i 4. kolonne over hold-kolonnen; matrisebredde låst til tre kolonner så
  hold-kolonnen lander under stående. Avstandsknapper = ikoncellenes størrelse.
  Mørk-modus-silhuetter tintes med varm lysebrun (colorPrimary #D8B79B). Nye
  art-spesifikke silhuetter for Elg og Villsvin (side + front; skrå bruker side).
- **Resultatkort:** poeng midtstilt under «Poeng:», tettere rader, «:» bak «Mitt
  gjennomsnitt for denne stillingen».
- **Serier:** lista klippes ved overkanten av «Lukk»-knappen (scroll-bunnmarg).
- **Logg jaktskudd (Rediger):** «Bom»/«ikke funnet» blokkerer avstandsfeltet
  umiddelbart (deaktivert + ikke fokuserbart); animasjonen blåser opp fonten (1,7×)
  mens tallet fader ut.

## 18. Endringslogg v0.13 (musingsUI runde 10)

(v0.12 var capture-runden fra felttesten på skytebanen — se `til_utvikler_v012.md`;
den rørte ikke UI-spesifikasjonen utenom scan-skjermen.)

- **Skjermorientering:** ALLE skjermbilder er låst til portrett. Kun `CaptureActivity`
  (Scan) er liggende (`sensorLandscape`).
- **Systemlinjer:** status- og navigasjonslinjen har svart bakgrunn med lyse ikoner i
  BÅDE lys og mørk visning. Appen melder seg ut av edge-to-edge-tvangen (targetSdk 35+
  ignorerer ellers `statusBarColor`), så systemet tegner baren selv.
- **Forskning er satt på pause** (`Dialogs.RESEARCH_ENABLED = false`): ingen
  forskningsdialoger i flyten, og bryteren «Del med forskning» i Avanserte innstillinger
  er låst av med forklarende hint. Checkboxen «Del med forskning» i Logg jaktskudd heter
  nå **«Detaljert visning»** og er rent et visningsvalg — den gjelder også for «Annet».
- **Duplikatsjekk:** to serier regnes bare som like hvis BÅDE poengene (< 0,05) OG
  treffpunktene stemmer. Treffene pares grådig, og hvert par må ligge innenfor
  **0,1 poeng** (rRel er i ringsteg, så ett ringsteg = ett poeng). Opp-ned-bilder gir
  identiske poeng men speilvendte treffpunkter, og meldes ikke lenger som duplikat.
- **Poengvisning (resultatkort):**
  - Blyanten per poenglinje er **fjernet** — OCR har overtatt korreksjonsrollen, og
    blyanten blokkerte plassen til høyre for de midtstilte poengene.
  - Finnes OCR-poeng, **tar de presedens** og vises i **skjermrekkefølge** (ikke sortert
    på verdi). Totalen følger det som vises.
  - Ved uenighet bytter visningene plass: OCR-poengene øverst, «Identifiserte treff:»
    nederst. Tekst: «Appen klarte ikke å se poengene riktig. Vil du lagre serien
    likevel?» med knappene «Forkast» / **«Lagre leste poeng»**.
- **Avvist analyse:** resultatskjermen anbefaler nytt bilde — «Bildet ble ikke korrekt
  analysert. Scan bildet på ny.» med «Avbryt» / «Scan». Signalet er analysens egen
  kvalitetsport (`status != OK`); ingen nytt cue fra CV-kjernen trengs.
- **Serier:** merking av flere serier oppdaterer kun radbakgrunn og knapper — lista
  bygges ikke om, så skjermen står stille (før hoppet den til toppen for hver merking).
- **Oppstart:** bildedelings-vinduet vises første gang appen åpnes, én gang neste sesong
  hvis deling ikke er valgt, og **alltid** når «Vis oppstartsmelding hver gang» er på i
  utviklermodus.
- **Innsikt:** rammen er nå 7 like høye rader som **autoskaleres etter skjermhøyden** så
  alt får plass på én skjerm: stillingsraden øverst + 6 rader i kroppen (5 vilttyper +
  vinkelraden). Hold-kolonnen har nøyaktig 6 knapper, så **200 m står rett til høyre for
  vinkel-/vilt-posisjonsvalgene**. Tekst- og knappestørrelser følger radhøyden. Nye
  silhuetter for **villrein** i alle tre vinkler (front/side/skrå) — første art med
  egen skrå-silhuett; elg og villsvin bruker fortsatt side-varianten på skrå.
- **Logg jaktskudd (Rediger):** tallet i «Dyret løp» vokser til **dobbel** størrelse og
  flyr **opp mot høyre** mens det fader ut (0,55 s). Foreldrene har `clipChildren=false`
  slik at animasjonen ikke klippes.
- **Avanserte innstillinger:** ny bryter **«Lagre scannede bilder i bildearkivet»**.
  Etter første scan spør appen «Ønsker du at skjermbildet skal lagres i bildearkivet
  ditt?» ‹Ja›/‹Nei›, etterfulgt av «Du kan endre dette valget i «Avanserte
  innstillinger»». Svarer man Nei, slettes også bildet fra den scanen.

## 19. Endringslogg v0.14 (backend-kobling, runde 1)

Første runde der appen faktisk snakker med en server. Alt annet fungerer som før
uten nett — offline-først er ikke svekket, bare supplert.

- **Melding til utvikler** går nå til `POST /v1/feedback` i stedet for å åpne
  e-postappen. Kvitteringer: «Sender …» → «Takk! Meldingen er sendt.» Feiler
  kallet, åpnes e-postappen som før («Fikk ikke kontakt. Åpner e-post i stedet.»).
  Er man rate-limitet (429) sier appen det rett ut i stedet for å lage duplikater
  via e-post. Tom melding avvises før sending.
- **Opplastingskøen er reell.** Bilder til feilanalyse (`ocr_match`,
  `ocr_mismatch`, `rejected`) ligger i en filbasert kø og sendes til
  `POST /v1/failed-analyses` ved appstart og på «Send nå». Tidligere ble de bare
  skrevet til disk og ble liggende for alltid.
- **«Send bildet til feilanalyse»** på avvist-skjermen sender nå faktisk bildet.
  Før viste knappen bare en kvittering og deaktiverte seg selv. Knappen er et
  eksplisitt samtykke for akkurat det bildet, uavhengig av den generelle
  bildedelings-bryteren, og vises bare når det finnes et bilde å sende.
- **Avanserte innstillinger** har fått:
  - **«Last kun opp på wifi»** (default på) — køen er fullskala-JPEG-er og skal
    ikke spise mobildata på skytebanen. «Send nå» overstyrer valget.
  - **«Send bilder til feilanalyse nå»** med levende status: «N bilde(r) venter
    på å bli sendt» / «Ingenting venter. Sist sendt \<dato tid\>». Knappen er
    deaktivert når køen er tom.
- **Øktoppsummering:** kø-linja teller nå det som faktisk kan sendes (bilder), og
  skjules når køen er tom. Den gamle telleren summerte serier + jaktlogg, som
  ingen kunne sende, og kunne derfor bare vokse.
- **Utviklermeny:** «API-adresse» lar en peke appen mot en annen backend (lokal
  maskin, staging) uten å bygge på nytt. Tom verdi = innebygd adresse.
- Ingen nye tillatelser brukeren merker: `INTERNET` og `ACCESS_NETWORK_STATE`
  krever ikke samtykke. Ingen data forlater telefonen uten at brukeren har sagt ja
  til bildedeling eller selv trykket «Send».
