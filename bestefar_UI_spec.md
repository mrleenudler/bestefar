# Bestefar — UI/UX-spesifikasjon

> **§1–§11 beskriver appen slik den er i dag (v0.20).** De ble skrevet om fra
> v0.4-teksten 2026-08-08; fram til da beskrev de en app som ikke fantes.
> Kildene var koden, `docs/flytskjema.md` og `android/KONTRAKT.md`, i den
> rekkefølgen.
>
> **§8 og §10 er ikke omskrevet, og skal ikke være det.** De er modellnotater og
> åpne punkter — resonnement og ubesvarte spørsmål, ikke beskrivelser av en
> flate. De gjelder uendret.
>
> Hva som ble bygget runde for runde, står i `android/CHANGELOG.md`.
> Fila het `bestefar_UI_spec-v0-4.md` fram til 2026-08-08.

Bruk placeholdere for grafikk som ikke er triviell å implementere.
Instruksjonen kan være uklar. Bruk egen dømmekraft, eller spør om veiledning.

## 1. Formål og designprinsipper

Appen har to formål som i praksis er allierte: den skal gi jegeren et presist bilde av egen skyteevne — hvilke hold og situasjoner som er forsvarlige, og hvilke som bør unngås — og den skal samle strukturerte data til forskning på effekten av øvelsesskyting på jaktutfall. Det som gjør dataene forskbare, er det samme som gir jegeren god egeninnsikt; den eneste reelle konflikten er friksjon, løst med kontekstarv og forhåndsvalg.

**Bruksmodell (styrende):** Appen brukes ved øvelsesskyting — ved planlegging før jakt og ved tilbakeblikk etterpå. Den er aldri fremme i skuddøyeblikket. Verdien ligger i at jegeren gjennom trening etablerer en internalisert grense for akseptable skuddsituasjoner, og tar *regelen* med seg i felt. Dette er for-tanke satt i system.

Prinsippene under er de opprinnelige. **De har styrt hver eneste runde og gjelder
uendret** — det var teksten rundt dem som var utdatert, ikke de.

- **Kontekstarv i tre nivåer.** Våpen bekreftes én gang per dag, og bare når
  brukeren har registrert mer enn ett — med ett våpen velges det uten prompt.
  Avstand er default 100 m og kan endres i nedtrekket. Skytestilling settes per
  serie, som et prompt etter skivescan.
- **Verdi før forpliktelse.** Full funksjonalitet uten konto og uten samtykker.
  Appen ber aldri om innlogging uoppfordret. Samtykker kan alltid endres og
  trekkes tilbake. *Forskningssamtykket er for tiden pauset — se §7.*
- **Jegerens språk, ikke statistikerens.** Frekvens («9 av 10»), ikke desimaltall
  eller tekniske mål, i alle beslutnings- og kompetanseflater. Usikkerhet kan
  vises som spenn der den er stor. Tekniske mål (σ, R95, MOA) finnes kun bak
  «Mer statistikk», og i forskningseksport.
- **Ett forsvarlighetsbegrep.** Brukeren velger akseptabel skadeskytingsrate
  (default 1 av 20); appen bruker samme kriterium overalt og sier aldri «trygt».
- **Ikonspråk.** Mest mulig ikoner (stillinger, arter, vinkling, utfall), alltid
  med i-tegn som åpner forklaring, og tekstalternativer (WCAG). Flerspråklighet
  er ikke bygget — bokmål i v1, se §10.3.
- **Offline-først.** Alt fungerer uten dekning; opplastinger køes. Nettet brukes
  til feilanalysekøen, sikkerhetskopien, innloggingen og beskjedkøen — aldri til
  noe brukeren trenger for å skyte en serie.

## 2. Navigasjon

**Tre faner** i en linje øverst: **Avstand**, **Innsikt**, **Meny**. Hvitt motiv
på sort med grå ramme; valgt fane er markert. Liggende orientering gir baren
60 % bredde, stående 80 %.

- **Avstand** og **Meny** åpner som nedtrekkspaneler — trykk på fanen igjen
  lukker. Klikk utenfor lukker avstandspanelet.
- **Innsikt** er fullskjerm.

Under baren ligger hovedflaten med **antall øvelsesskudd denne sesongen** øverst
og den store, sentrerte **«Scan serie»**-knappen i nedre halvdel.

Våpen, Stilling og Jakt er **ikke** faner. Våpen bekreftes i dagsprompten,
stilling velges etter hver scan, og Jakt ligger i menyen. Grunnen er at ingen av
dem er et sted man navigerer til — de er ledd i en flyt.

**Menyen**, i rekkefølge: Profil · Jakt · Venner · Serier · Melding til utvikler ·
Hvordan bruke appen · Søk. Søket er et fritekstfelt over de samme skjermene, med
nøkkelord per oppføring.

Samtykke- og datastyring ligger under Profil → Avanserte innstillinger.

## 3. Flate: Økt

### Dags- og øktoppsett
- **Våpen** bekreftes én gang per dag, og **bare når mer enn ett våpen er
  registrert**. Med ett våpen velges det automatisk, uten prompt.
- **Avstand** 100 m som standard, endres i nedtrekket. Egendefinert avstand kan
  settes på en av knappene.
- **Skive:** Kongsberg elektronisk skive, fast skala. Ingen skivevalg i v1.
- **Ammunisjon spørres ikke om.** Feltene finnes i datamodellen, men det er
  ingen flate for dem — se §6.

### Capture-løkke
Per serie: **Scan serie → auto-capture → analyse på enheten → resultatkort →
stillingsvelger → innskytingssjekk → OCR-kontroll → lagre**.

Merk rekkefølgen: **stillingen velges etter at resultatet er vist**, ikke før.
Skuddene er alt skutt når appen kommer fram, så det er ingenting å spare på å
spørre først — og et resultatkort på skjermen er en bedre påminnelse om hvilken
stilling serien faktisk ble skutt fra enn et tomt spørsmål.

`CaptureActivity` er appens eneste **liggende** skjerm. Auto-capture tar bildet i
det øyeblikket kriteriene er oppfylt, og spiller av bekreftelsen (grønn ramme,
«Klar!», hvit blits) *etterpå*, mens analysen allerede kjører. Tersklene er
ukalibrerte startverdier — se `core/ARCHITECTURE.md`.

Stillingsprompten viser fire stillinger — **liggende, sittende, knestående,
stående** — med hjelpemidlene **anlegg** og **reim** som radio-toggler. «Uten» er
ikke et eget valg, men den tilstanden der ingen av de to er valgt; trykk på en
aktiv knapp slår den av igjen. Siste hjelpemiddel huskes per stilling. Velgeren
viser antall skudd per stilling denne sesongen.

**Det finnes ingen benk-inngang.** Stillingsenumet har fire verdier, og benk er
ikke en av dem. §8 beskriver hvilken rolle benk *ville* hatt i modellen; den
rollen er ikke bygget.

Innstillingen «ikke spør — manuell stilling» erstatter prompten med et synlig
valg.

### Resultatkort
- Skuddene plottet på en skivegjengivelse, med poengsum og desimaler.
- **OCR-kontroll**, etter stillingsvalget. `OcrVerifier` (ML Kit, on-device)
  leser apparatets egen poengliste og sammenligner med kjernens treff:
  - **Match** (avvik ≤ 0,2): OCR-poengene vises, i *skjermrekkefølge* — usortert.
  - **Uavklart:** kjernens poeng vises, stigende.
  - **Uenighet** (> 0,2): OCR øverst, «Identifiserte treff» nederst, og valget
    Forkast / Lagre leste poeng.
  **OCR har presedens når den finnes**, og totalen regnes av det som vises, så
  liste og sum aldri spriker.
- **To uavhengige kvalitetsporter:** kjernens statuskode fanger «jeg fikk ikke
  kalibrert skiva» og utløser re-scan-dialogen; OCR-uenighet fanger den
  vanskeligere klassen der analysen *lyktes* men leste feil.
- **Innskytingssjekk:** ser sesongens eller dagens første serier skjevt innskutt
  ut, spør appen «Er dette innskyting?», og serien kan forkastes.
- **Duplikatvarsel** ved lagring når serien er praktisk talt lik den forrige.
- **Klikk-forslag** kun når offsetet er skjelnbart fra støy (per akse
  > ~2σ̂/√n); ellers «innenfor støy — ikke juster». Krever at klikkverdien på
  optikken er fylt ut.

Ved avvist analyse: «Bildet ble ikke korrekt analysert. Scan bildet på ny.» med
Avbryt / Scan på ny, og tilbud om å sende bildet til feilanalyse.

### Øktoppsummering
Serier med stillingsikon, poeng og frekvensbudskap; usendte opplastinger i kø.

## 4. Flate: Jakt

Nås fra Meny → Jakt, med to innganger: **Registrer jaktskudd** og **Se
registrerte skudd**.

### Valgfri lås foran loggen
Innstillingen «Krev opplåsing for jaktloggen» (av som standard) legger
`BiometricPrompt` foran begge inngangene — biometri eller skjermlås, med fem
minutters frist etter vellykket opplåsing. Avvist opplåsing lukker ingenting;
brukeren blir stående på forrige skjerm.

**Dette er en dør foran skjermen, ikke kryptering.** Jaktloggen ligger like
lesbar på disk. Bryteren skjules helt på enheter uten biometri eller skjermlås —
en bryter som ikke kan virke, skal ikke stå der og se ut som en mulighet.

### Jaktmodus
Valgfritt dagsoppsett: art(er), våpen, stedslogging av/på (huskes). Aktiv
jaktmodus gir en forhåndsutfylt hurtiglogg.

### Hurtiglogg — tre steg, hanskevennlig
1. **Art** (ikoner: elg, hjort, villrein, rådyr, villsvin; «annet» finnes og
   holdes utenfor analyser).
2. **Hold**, deretter **vinkling** som dyr i sirkel (BH-undersøkelsens format) —
   bredside, skrå 30°, skrå 60°, forfra, bakfra. Valgfri chip **«i bevegelse»**.
3. **Utfall** — tre kategorier: **dødelig** (operasjonalisert som «dyret løp
   kortere enn x meter»), **skade**, **bom**.

Innstillingen «Detaljert visning» styrer om loggingen skjer på én enkel side
eller to. Systemhentede metadata (dato, tid, posisjon når stedslogging er på)
legges på automatisk.

Mål: under et halvt minutt, helt uten dekning.

### Kunngjøring av felling
Etter en vellykket felling tilbyr appen å varsle vennene. **Setningen vises
ordrett før den sendes** — «Ola har felt et villsvin i Molde.» — og stedet er et
felt brukeren kan rette eller tømme. Et delingsvalg der man må gjette hva som
deles, er ikke et valg.

Bøyningen bygges i klienten, ikke på serveren: norsk artikkelvalg («en elg», «et
rådyr») kan ikke avledes av en enum. Serveren lagrer ingenting om hva som ble
felt eller hvor. Krever konto; uten innlogging tilbys det ikke.

### Totrinns utfall og ettersøk
Umiddelbart utfall kan oppdateres etter ettersøk (felt/avlivet, funnet død,
friskmeldt, ikke gjenfunnet). Appen følger opp med **ett stille spørsmål** — kun
for skadeskudd, tidligst to timer etter registreringen, og **aldri mer enn én
gang per post**. Nøytralt, uten purring. Skadedata er private som standard,
aldri i delt kontekst.

### Stedsdata
Presis posisjon lagres **lokalt** når stedslogging er på, og vises som
stedsnavn på posten. Deling til forskning i grovere oppløsning velges i
samtykket. **Kartvisning over egne jaktskudd er ikke bygget**, og heller ikke
værhenting (MET) — begge står på veikartet, §10.4.

## 5. Flate: Innsikt

Én fullskjermsvisning: **kompetanseoversikten**, som en matrise over fem
viltarter. Ingen segmentkontroll, og **intet kapabilitetskart** — se under.

### Kompetanseoversikt
Én rad per art (elg, hjort, villrein, rådyr, villsvin). Radene rammes inn av
jegersilhuetten for den valgte stillingen, og vinklingen velges per art med
skråstilte dyrefigurer. Holdvelgeren ligger til høyre. Figuren skalerer *subtilt*
med hold — utelukkende som ekstra visuelt signal om at holdet er endret; **farge**
(andel dødelige treff på lang sikt, rødt til grønt, der brukerens valgte
skadeskytingsrate er gult) bærer hovedbudskapet, ikke størrelse.

Er en stilling utestet, står det «øv på stillingen» i stedet for et tall.

Modellen: gitt homogen spredning fra siktepunktet (målt på papir) og radien på
dødelig sone ved valgt art × vinkling, beregnes andelen skudd som havner
innenfor. Vinkling virker ved å krympe den projiserte dødelige sonen; **minste
halvakse** er den bindende radien (ikke gjennomsnitt — den korte veien ut av
sonen avgjør). *Sonefaktorene per vinkel er plassholdere — §10.1 og §10.2.*

To utfall vises:
- **Dødelig treff** — innenfor radien.
- **Skade** — alt annet. (Rene bom eksisterer i virkeligheten, men er på papir umulig å skille fra skade og slås derfor sammen; jf. seksjon 8.)

Presentasjon i frekvens og **ren gevinstramme**: «Feller rent ~9 av 10.»
Tapssetningen er bevisst utelatt — appen er aldri med i skuddøyeblikket, så ingen
impulskontroll tapes, og dobbel formulering blir clutter som senker
bruksterskelen. Brukerens valgte skadeskytingsrate brukes som grense, og **maks
forsvarlig hold** leses av per stilling.

Nederst ligger **«Mer statistikk»**: de tekniske målene (σ) for sesongen, eller
«Ingen data denne sesongen». Det er det eneste stedet i appen tekniske mål vises.

### Kapabilitetskartet er ikke bygget
Matrisen avstand × stilling med artsvelger, ekstrapolerte celler og celledetalj
finnes ikke. Spørsmålet fra v0.4-teksten — *gir den mening ved siden av
kompetanseoversikten?* — er fortsatt ubesvart og ligger som **ÅP-U6** i
`AAPNE_PUNKTER.md`. Det er ikke utsatt av tidsnød, men fordi to visninger av
samme tall må kunne begrunnes hver for seg.

### Sesong og historikk
Statistikk nullstilles per sesong = **jaktåret 1. april–31. mars**. Kompetansen
regnes på inneværende jaktår.

Historikken bor i **Serier** (Meny → Serier), ikke i en egen historikkflate.
Øverst der ligger **trendgrafen**: x-aksen er dato over inntil to jaktår — når
det tredje begynner, faller det eldste ut — og y-aksen er et rullende snitt over
de siste 20 skuddene. Er det skutt mer enn 20 skudd på én dag, brukes dagens eget
snitt for det punktet. Y-aksen skaleres slik at avstanden fra laveste datapunkt
til bunnen aldri blir mindre enn 25 % av aksehøyden; en flat kurve skal se flat
ut, ikke dramatisk.

**Den siste økten framskrives ikke.** Er det rullende vinduet ennå ikke fullt,
tegnes siste punkt hult i stedet for å gjettes ferdig. Et beregnet punkt er ikke
til å skille fra et målt, og det er nettopp der brukeren ser hardest.

**Presentasjon av langsiktig tilstand:** vis gjeldende langsiktig gjennomsnitt,
og la nye serier falle inn i det bildet — ingen «du ligger over/under»-dom per
serie (én serie er for støyende til å dømmes; vis tilstand og usikkerhet, ikke
avvik-fra-forventning).

Frosne kart per tidligere sesong finnes ikke — de forutsetter kapabilitetskartet.

### «År uten skadeskudd» — ikke bygget

Hvor det hører hjemme er fortsatt ubesvart (**ÅP-U7**). Designet under står fordi
det er gjennomtenkt, ikke fordi det finnes:

Et gripbart, personlig tidsmål (når jaktskuddvolum finnes). To designgrep gjør det ærlig uten matematisk clutter:
- *Vis raten i tidsformat, ikke inversen.* Regn ikke 1/p (som eksploderer og hopper ved lave rater), men vis den stabilt estimerte raten skalert med brukerens skuddvolum: «omtrent hvert X-te jaktår for deg». Samme underliggende p, langt roligere tall.
- *Skrittvis presisjon i stedet for terskel.* Ingen ventetid på tom skjerm: tidlig vises en grov kategori («sjelden» / «av og til» / «ofte»), som smalner til et tallanslag etter hvert som skuddene samler seg. De grove kategoriene tåler mye estimeringsusikkerhet og er derfor ærlige fra dag én.
- En **(i)** forklarer bevegelsen på jegerspråk — «anslaget bygger på hvor mange skudd du har logget; jo flere, jo sikrere» — uten konfidensintervall. Plasseres i refleksjon/historikk, aldri som et grønt lys i planleggingen.

### Øvelsesmotoren — ikke bygget

`Store.practicePosition` finnes som felt, men ingenting skriver til det, og
ingen dialog foreslår en øvelse. Designet under er uendret og fortsatt gyldig
som mål:

Den bør fungere som et popup-forslag, men bare dersom det er behov for spesifikk øving, og med fornuftige mellomrom, slik at den ikke blir masete. Den kan for eksempel trigges etter en serie der brukeren har valgt en vanlig øvelse fremfor en nyttig øvelse. "OK" setter innstillingene for brukeren.
En øvelse = stilling + avstand(norm 100m). **Kjerneprinsipp — lukk trening/felt-gapet:** motoren sammenligner brukerens *treningsfordeling* av stillinger mot hans *jakt-/jaktformfordeling*, og prioriterer stillingen med størst underdekning. Speiler brukerens egen inkonsistens tilbake («skutt 40 % av jaktskuddene sittende, trent 5 %») heller enn å foreskrive normativt. Inntil brukerens egne data er robuste, brukes generell jaktstatistikk som forhåndsinformasjon for hva som er relevante stillinger og hold.

Prompt-logikk: (1) relevant + null data → «bør etableres»; (2) relevant + tynt/eldet → «bør bekreftes»; (3) irrelevant → nevnes ikke uoppfordret. Relevans fra jaktform/jaktlogg, tynnhet fra skuddantall. Aldri prompt på fravær alene. Forslag vises som kort på Økt-flaten og som «test denne» fra celler/stillinger.

## 6. Flate: Profil

Meny → Profil. Alt lagres fortløpende; ingen «Lagre»-knapp.

- **Skytterprofil:** visningsnavn, fødselsår (kun for 18-årsgrensen på
  forskningssamtykke), lag (skytterlag/jaktlag, flere mulig), «La venner finne
  meg» (**av** som standard).

  **Visningsnavnet er det vennene ser**, og det modereres på serveren. Er man
  innlogget, sendes navnet når feltet forlates, og svaret er endelig med det
  samme: godkjent og lagret, eller avvist med serverens egen begrunnelse vist i
  feltet. **Ingen «venter på moderasjon»-tilstand vises** — regelsettet svarer
  ja eller nei, og den manuelle køen finnes ikke (ÅP-B8). Et avvist navn blir
  ikke stående i feltet, fordi serveren ikke lagret det.

  Uten konto er navnet bare et lokalt kallenavn på venne- og lagskjermene.

  **E-postadressen skal aldri brukes som visningsnavn.** Navnet deles med
  venner, og lokaldelen av en adresse er ofte nok til å finne adressen. Ved
  Google-innlogging settes startnavnet nå fra `name` i det verifiserte
  ID-tokenet, med lokaldelen og «Skytter» som reserve.

  **Kontoer opprettet før dette beholder navnet de fikk.** Serveren rører ikke
  visningsnavnet ved senere innlogginger — å overskrive det ville stille
  tilbakestilt et navn brukeren selv har satt. Har du en konto med lokaldelen
  som navn, endrer du den i feltet over.
- **Mitt jaktmål:** den valgte **skadeskytingsrate-grensen**, som en setning i
  jegerspråk («1 av 20») med (i)-forklaring. Endring virker umiddelbart.
  Tilbys automatisk etter tredje serie.
- **Konto:** innlogging med Google (Credential Manager) eller sekssifret kode på
  e-post. Innlogget vises kontonavnet, bruker-ID og «Logg ut».
  Apple-innlogging er ikke bygget.

  **Under kontonavnet står «Logget inn med Google som ola@gmail.com».** Hele
  adressen, siden samme lokaldel finnes hos flere leverandører, og vist **kun
  for brukeren selv** — dette er en kontoidentifikator, ikke et navn.

  Adressen er den økten ble startet med, hentet fra innloggingssvaret. Klienten
  leser den ikke ut av ID-tokenet: kontosammenslåing på verifisert adresse
  (`backend_spec.md` §1) gjør at kontoen kan være knyttet til en annen adresse
  enn den man nettopp logget inn med, og linja ville løyet i akkurat det
  tilfellet den finnes for.

  Vet ikke serveren adressen — Apple med skjult e-post, eller en økt startet før
  feltet fantes — vises bare leverandøren. Mangler begge, står det ingenting. **Appen ber aldri om innlogging uoppfordret**, og skjermen sier
  eksplisitt at alt annet virker uten konto.

  **Er kontoen nyopprettet**, tilbys en sikkerhetskopi med det samme — etter
  varseldialogen, aldri samtidig med den. Det er det ene øyeblikket appen vet
  at brukeren har data lokalt og ingen kopi noe sted.

  **Fantes kontoen fra før**, spør appen serveren om det ligger en kopi der, og
  tilbyr *gjenoppretting* i stedet — med datoen kopien ble laget. Finnes det
  ingen kopi, eller svarer ikke serveren, **vises ingenting**. En bruker uten
  kopi trenger ikke å få vite at funksjonen finnes akkurat da, og et oppslag som
  ikke nådde fram vet ingenting.
- **Visningsprofil:** lys / mørk / system, øverst til høyre. Default lys.
- **Avanserte innstillinger** (egen skjerm, 🎛-ikon fører dit fra alle steder
  som nevner den).

### Avanserte innstillinger

- **Mine våpen** — våpen og optikk med klikkverdi (cm/klikk@100 m). (i)
  forklarer at appen kan gi klikkforslag når den er utfylt.
  **Ammunisjonssplitt er ikke bygget:** feltene finnes i datamodellen, men det
  er ingen flate for dem, og dagsprompten spør ikke om ammunisjon.
- **Flytt til ny telefon** — ikke bygget; knappen sier fra om det.
- **Slett alle data** — lokal sletting, bak en bekreftelse med STOP-ikon.
- **Sikkerhetskopi** — «Sikkerhetskopier nå», «Gjenopprett», «Vis
  gjenopprettingskode». Se under.
- **Del med forskning** — bryter, **pauset**, se §7.
- **Del bilder der appen gjør en dårlig jobb** — feilanalysekøen, med teller og
  «Send nå».
- **Kun wifi for opplasting** — **på** som standard: køen består av
  fullskala-JPEG-er, og en jeger på skytebanen skal ikke bruke opp mobildata
  uten å ha bedt om det.
- **Lagre skannede skjermbilder i bildearkivet** — tre valg, ikke av/på:
  *Aldri*, *Alle*, *De beste*. «De beste» = blant de 25 % beste i samme
  stilling, eller beste serie noensinne.
- **Gjenopprett uten kode** — **på** som standard fra v0.26. Se under.
- **Krev opplåsing for jaktloggen** — **av** som standard, §4.
- **Venstrehåndsmodus.**
- **Utviklermeny** (kun i debug-bygg).

### Sikkerhetskopi og nøkkelforvaltning

Kopien er **klient-kryptert**: serveren lagrer bytes den ikke kan lese. Nøkkelen
søkes opp i tre lag, i denne rekkefølgen:

1. **Lokalt** på telefonen.
2. **Google Play Block Store** — følger med til den nye telefonen ved
   gjenoppretting. Brukes bare når innholdet er ende-til-ende-kryptert; er det
   ikke det, lagres ingenting der. En nøkkel som ligger lesbar hos en tredjepart
   er dårligere enn ingen nøkkel der.
3. **Deponering hos oss** — bryteren «Gjenopprett uten kode», **på som
   standard** fra v0.26. **Dette er det eneste tilfellet der vi kan lese kopien
   din**, og teksten sier det rett ut både ved første kopi og i innstillingene.

   Begrunnelsen for at den er på: brukerne er jegere, ikke teknologer, og
   forventningen etter innlogging er at dataene er trygge. Sikkerhet som må
   velges, blir ikke valgt. Slår brukeren den **av**, blir gjenopprettingskoden
   eneste vei tilbake, og da vises koden der og da med krav om avkryssing —
   det som endret seg er ikke koden, men hva som skjer hvis den mistes.

Den 20-tegns **gjenopprettingskoden er nødutgangen**, ikke hovedveien. Brukeren
blir bedt om å *taste* den bare når ingen av de tre lagene har noe.

**Koden vises én gang med krav om bekreftelse — første gang.** Da kan dialogen
ikke avbrytes, og knappen er deaktivert til brukeren har krysset av for at koden
er skrevet ned et annet sted enn på telefonen. Teksten sier hva som går tapt:
kopien er tapt også for oss, uten koden. Etterpå — når brukeren selv ber om å se
koden fra Avanserte innstillinger — er avkryssingen borte; da er den ren
friksjon.

**Koden kan hentes fram igjen** under Avanserte innstillinger → Sikkerhetskopi →
«Vis gjenopprettingskoden», så lenge brukeren har telefonen. Det som ikke kan
hentes igjen, er koden til en telefon som er borte — og det er hele grunnen til
at den skal skrives ned et annet sted.

«Gjenopprett» spør serveren først om det finnes noe å hente, og viser **når
kopien ble laget** før den erstatter alt lokalt.

### Data og samtykke

To separate samtykker (trening/jakt) og stedsgranularitet for deling ligger i
samtykkedialogene, §7. Sletting er lokal wipe; sletteanmodning via pseudonym-ID
sendes av backenden ved kontosletting.

### Hjelp

Ikonforklaring, forklaring av skadeskytingsrate og forsvarlighetskriteriet, og
merknad om at dødelig-sone-radiene er provisoriske (§10.1).

Én skytter per installasjon antas.

## 7. Onboarding og samtykkeflyt

### Oppstart

Ingen egne intro-skjermer. Ved appstart, i rekkefølge:

1. **Oppstartsvinduet** — hva appen gjør. Vises første gang, og på nytt hvis
   teksten endres (versjonsnummer på meldingen).
2. **«Vil du dele bilder der appen gjør en dårlig jobb?»** — første gang, og
   deretter én gang per sesong så lenge deling ikke er valgt.
3. **Veiledningen** — fire steg som overlegg, hvis den ikke er sett. Kan også
   åpnes når som helst fra Meny → «Hvordan bruke appen».
4. **Ventende beskjeder** fra serveren, hvis brukeren har konto, én om gangen.

Rekkefølgen er ikke tilfeldig: beskjedene hentes *parallelt* med de tre første,
men holdes til de er unnagjort. Et nettverkssvar skal ikke legge seg oppå
veiledningen.

### Samtykker

- **Jaktmålet** (skadeskytingsraten) tilbys etter tredje serie.
- **Forskningssamtykket** tilbys etter fem serier, og deretter hver tiende ved
  «Ikke nå». Valgene er Ja / Ikke nå / Aldri. Ved ny sesong spørres det på nytt,
  også av dem som allerede deler. Aldri i samme økt som jaktmålet, og tidligst
  to serier etter.
- **Jaktsamtykket** tilbys ved første bruk av jaktloggen, med delingsvalg for
  hva som deles.
- **Forskningssamtykke krever 18 år**, og fødselsåret spørres om der og da hvis
  det mangler.

Avslag endrer ingen funksjonalitet. Samtykke kan gis og trekkes når som helst.

**Skadedata deles bak sin egen bryter**, atskilt fra resten. «Jeg skjøt stående
på 85 meter» og «dyret ble skadeskutt og aldri funnet» er ikke samme opplysning
å dele om seg selv, og de skal derfor ikke ha samme bryter.

### Forskningsdelingen er pauset

Hele forskningsflyten — begge samtykkedialogene og delingsbryteren — er slått av
med ett navngitt flagg (`Dialogs.RESEARCH_ENABLED = false`). Ingen dialoger
vises, bryteren er deaktivert med en forklaring, og ingenting sendes.

Grunnen er ikke teknisk: den skal ikke aktiveres mot ekte brukere før
personvernerklæringen finnes og DPIA-spørsmålet er avklart. Koden er ikke
slettet, og alle inngangene sjekker det samme flagget — funksjonalitet som skal
pauses, pauses ett sted.

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

Ikonsett for stillinger (silhuetter, skalert per stilling så de får lik visuell
tyngde), arter (side, front og skrå per art), utfall og faner. Tekstalternativ
per ikon; i-tegn på ikonskjermene.

Utendørsbruk: høy kontrast, og lys/mørk/system som eget valg med **lys som
standard**. Sensorstyrt mørk modus er ikke bygget — **ÅP-U8**.

**Venstrehåndsmodus** speiler grensesnittet horisontalt.

Advarsler bruker et varselikon; **irreversible handlinger bruker STOP-ikonet**,
ikke det samme trekantikonet som en advarsel man kan angre.

Appen tegner systemlinjene selv, i både lys og mørk visning — `targetSdk` 36
gjør `statusBarColor` til en no-op.

## 10. Åpne punkter

1. **Dødelig-sone-radius per art × vinkling** — tabellverdier (minste halvakse ved hver vinkel), kildebelagt. Dette er nå hele sonedatajobben (redusert fra dobbel silhuett til én radiustabell).
2. **Vinkeltaksonomi venter på primærkilde** — provisorisk 30°/60° + side/front/bak beholdt, men *ikke verifisert*. Hentes fra BH-materialet før implementasjon; skytter-oppgitt-vinkel er en kjent målefeilkilde å håndtere i forskningsdesignet.
3. **Navn og språk** — bokmål i v1; app-navn avklares før butikklansering. Støtte for systemspråk / andre relevante språk.
4. **Veikart:** papirskive-støtte med auto-gjenkjenning; dyrefigur-CV; bevegelige mål i modellen; personlige målsettinger per stilling; **gamifisering av øvingen** (bør vurderes på sikt — utmerkelser, progresjon, streaks e.l. for å øke treningsfrekvens; må balanseres mot at appen ikke skal oppmuntre til uforsvarlige skudd eller gjøre skadeskyting til «poengtap» som frister til underrapportering).

## 11. Avhengighet til CV-kjernen (teknisk)

UI-prosjektet **forker ikke** OpenCV/C++-kjernen — men den konsumeres heller
ikke som pinnet avhengighet, slik denne paragrafen krevde i v0.4.

**Slik er det i dag:** kjernen ligger i samme repo, under `core/`, og
`android/app/src/main/cpp/CMakeLists.txt` gjør `add_subdirectory` rett inn i
`core/CMakeLists.txt`. Samme kildetre, ingen forhåndsbygget binær og ingen
versjonspinne mellom områdene. Kjernen bygges dermed to ganger: én gang på
desktop for verifisering mot C-settet, og én gang av NDK-en som del av
`assembleDebug`.

Kravet om pinnet submodule er **ikke oppfylt, og det er ikke besluttet om det
skal være det** — se `AAPNE_PUNKTER.md` **ÅP-U11**. Monorepoet gir at et brudd i
`core/src` stopper klientbygget med det samme, som er en fordel; prisen er at
det ikke finnes noen versjon å peke på når noe går galt i felt.

**Grensen mot kjernen:** `jni_bridge.cpp` ligger fysisk i UI-området, men er
*forbruker* av `core/include/bestefar/bestefar_ffi.h`. Endres headeren, eier
kjernen endringen og klienten følger etter. Selve CV-kontrakten — statuskoder,
`BfResult`, `BF_MAX_HITS`, pikselformater — står i `core/KONTRAKT.md` og gjentas
ikke her.

Koblingen til kjernen skjer via hovedskjermens **«Scan serie»**-knapp, gjennom
`BestefarCore.analyze()` i `CaptureActivity`.

Kjernen har sin egen versjon (`bf_version()`), uavhengig av appens
`versionName`. Den følger med i feilanalyse-donasjonene, så det er mulig å se
hvilken kjerne som produserte et resultat.


## 12. Om dette dokumentet

Seksjonene §12–§24 var tretten endringslogger fra v0.6 til v0.19, lagt til én per
runde. De er **flyttet ordrett** til `android/CHANGELOG.md` 2026-08-08.

**§1–§11 ble skrevet om til nåtid samme dag.** Fram til da beskrev de v0.4 — seks
faner der det er tre, «Benk» som egen inngang, optikk-kalkulator i menyen — og
den som leste forfra møtte en app som ikke fantes. Rangordningen som sto her,
med §1–§11 nederst, var en innrømmelse av det: seksjonene var sist i køen og lå
likevel først i fila.

**Nå beskriver §1–§11 appen slik den er.** Rangordningen er derfor tatt ut.
Dokumentene sier ulike ting fordi de har ulike jobber, ikke fordi ett av dem er
utdatert:

| Spørsmål | Fil |
|---|---|
| Hva appen gjør og hvorfor den er slik | **denne fila**, §1–§11 |
| Hva som skjer når, steg for steg | `docs/flytskjema.md` |
| Hva andre kan stole på over ledningen | `android/KONTRAKT.md` |
| Hvorfor lagene ser ut som de gjør, hva som ble vraket | `android/ARCHITECTURE.md` |
| Hva hver runde endret | `android/CHANGELOG.md` |

**Koden er fortsatt fasit** — ikke fordi denne fila er mistenkt, men fordi den er
skrevet av mennesker og koden er det som kjører. Finner du et sprik, er det en
feil som skal rettes her, ikke noteres.

**Det som ikke er bygget, står som ikke bygget.** Kapabilitetskartet,
øvelsesmotoren, «år uten skadeskudd», ammunisjonssplitt, kartvisning, værdata og
den pinnede kjerne-avhengigheten er alle nevnt der de hører hjemme, med et
åpent-punkt-nummer der det finnes ett. En spec som tier om det ubygde, leses som
om alt er bygget.

**§8 og §10 er ikke omskrevet.** §8 er modellnotater — resonnement om statistikk
og framing — og §10 er eierens åpne punkter. Ingen av dem beskriver en flate, og
begge gjelder uendret. Merk at §8 omtaler benkens rolle i modellen; benk er
**ikke** en stilling i appen (§3), og bulletpunktet beskriver hva den ville vært
brukt til dersom den kom.
