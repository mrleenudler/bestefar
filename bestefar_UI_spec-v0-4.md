# Bestefar — UI/UX-spesifikasjon v0.4


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
