# ÅPNE PUNKTER

Alt i de tre spesifikasjonene som **ikke kan besluttes i kode**: `TODO(eier)`,
eksplisitt åpne spec-punkter, og verdier som er merket ukalibrerte eller
provisoriske.

**Punktene står her fordi de ikke kan finnes på.** En terskel som skal kalibreres
mot maskinvare, en radius som skal kildebelegges, et feltinnhold som skal
defineres av prosjekteier — ingen av dem blir riktigere av at en instans gjetter
en plausibel verdi. Denne fila foreslår derfor ingen verdier. Den sier hva som
mangler, hvor det står, og hvem som kan avgjøre det.

**Unntaket er seksjon F**, som ble lagt til 2026-08-22. Den holder utestående
arbeid fra en feltrunde — ting som *kan* besluttes i kode, men ikke er gjort.
De står her for å ha en ID å lukkes fra. Blandes de sammen med A–E, mister
begge kategorier mening: en ukalibrert terskel og en uimplementert knapp er
ikke samme slags åpent punkt.

Sist gjennomgått: **2026-08-07**.

**Bruk:** referer punkt-ID i issues (`ÅP-K3`) og i `til_utvikler_v##.md`. Legg
til nye punkter når du oppdager dem. Stryk aldri et punkt uten at eier faktisk
har avklart det — flytt det da til «Avklart» nederst med dato.

**ID-ene:** `K` kjerne, `B` backend, `U` klient, `E` krever eier, `D` delt
(treffer alle tre og passer derfor ikke i eierskapsmodellen).

Kilder: `bestefar_CV-kjerne_spec.md` (kjerne-spec), `backend_spec.md`,
`bestefar_UI_spec.md` (UI-spec).

---

## A. Ukalibrerte verdier

Verdier som står i koden i dag, med startverdier som er merket som gjetninger.
De virker — men ingen vet om de er riktige, og det er ikke lest av noen måling.

### ÅP-K1 — Auto-capture-tersklene · label `kjerne`
> «Terskelverdiene for begge kriteriene … **kan ikke bestemmes teoretisk**. De må
> kalibreres mot faktisk maskinvare.»
> — kjerne-spec §4, linje 89–91; gjentatt i §8, linje 145

Gjelder: hvor mange rammer stabilitet krever, hvor stramt skarphetskravet er,
hvor stor andel av skiven som må være synlig.

Står i koden som: `core/include/bestefar/config.h:181` (`min_bull_width_frac`),
`:185` (`max_glare_frac`), `core/include/bestefar/bestefar_ffi.h:72,82,83`,
samlet i `AutoCaptureParams` (`docs/ARCHITECTURE.md:109`).

Krever: målesesjon med faktisk telefon mot faktisk skive. Feltrunden i
`til_utvikler_v012.md:63` er nærmeste vi har, og satte bevisst permissive verdier.

**Feltobservasjon 2026-08-21 — falsk positiv:** auto-capture utløste på et
**vitrineskap**. Observert av eier under en annen test (frikoblingen, ÅP-B13), ikke
under en kalibreringsrunde — så den er et enkelttilfelle uten måletall ved
siden av seg, og den sier ikke *hvilket* av kriteriene som slapp den gjennom.

Det er første registrerte tegn på at de bevisst permissive verdiene har en
kostnad, og det er en kostnad som treffer brukeren: en utløsning uten skive
koster irritasjon og tillit. Notert her fordi det er her
tersklene diskuteres; materialet hører også hjemme i ÅP-U14. **Eies av kjernen
— backend har bare ført observasjonen inn.**

**Feltobservasjon 2026-08-21 — falsk negativ, motsatt vei:** et **skjermbilde av
en skive** utløste aldri capture. Sammen med vitrineskapet betyr det at
tersklene bommer i begge retninger, og de to feilene kan ikke rettes med samme
justering.

**Merknad om forholdene bak begge observasjonene over (fra eier):** dette er
ikke opptak av en fysisk skive på bane. Begge er fotografier av det gamle
testsettet og av en PC-skjerm. Vitrineskapet og et separat funn — 17
fantomtreff detektert i et bilde av en **sofa** — er tatt under de samme
forholdene. Det svekker ikke observasjonene som *tegn på at tersklene bommer i
begge retninger* (det er de fortsatt), men en terskel som skulle vært
kalibrert mot banelys, ekte blenk fra en opplyst skjerm sett fra flere
avstander/vinkler, og faktisk skivedybde, er **ikke** kalibrert mot det bare
fordi den er målt mot skjermbilder — lysforhold, blenk og manglende dybde
skiller de to. Gjelder alle feltobservasjoner fra eier «den siste tiden»
(datostemplet her, ikke bare disse to) inntil en runde faktisk gjøres på bane.

**Klienten fikk en tidsgrense i v0.29, og den er en måleordning — ikke en
løsning.** Utløser ikke gatingen innen 8 sekunder, tar `CaptureActivity`
gjeldende ramme og analyserer den likevel (`android/CHANGELOG.md` v0.29).
Hensikten er å skaffe punktet denne saken mangler: fram til nå ga en falsk
negativ *ingen* observasjon, og var derfor ikke til å skille fra «kjernen kjørte
og feilet». En analyse som **lykkes** etter timeout ville vært et direkte bevis
på at tersklene var for strenge for akkurat det motivet, med bilde ved siden av
seg.

**Men det er ikke observert, og hensikten er derfor ikke innfridd (2026-08-22).**
Tidsgrensen er verifisert som *mekanisme* — den utløser, tar bildet og kjører
analysen. Den utløste bare i tilfeller der **ingen skive var vist**. Hvert
forsøk med et faktisk skivemotiv endte i ordinær auto-capture før de 8 sekundene
var gått, så tidsgrensen har aldri sluppet gjennom et skivebilde gatingen
avviste. Det ene den ble bygget for å fange, er altså fortsatt ufanget.

En tidligere versjon av dette punktet antydet at timeout-bildet hadde scoret en
reell skive. **Det er feil, og strykes her framfor å bli stående som en
observasjon andre kan bygge på.**

Merk hva det gjør med bevisverdien: en timeout som utløser på en vegg eller et
tilfeldig motiv sier ingenting om tersklene — den sier bare at det ikke var noen
skive der. Materialet som er verdt noe, er en timeout **med** skive i bildet, og
det finnes ikke ennå. Den falske negativen fra 2026-08-21 (skjermbildet av en
skive) lot seg ikke gjenskape.

To ting den ikke gjør: den rører ikke tersklene, og den hjelper ikke mot falske
positive — et tilfeldig motiv utløser fortsatt gatingen innen 8 sekunder.

**Åpent, og eid av kjernen:** selve kalibreringen. **Åpent hos UI:** de 8
sekundene er valgt, ikke målt, og hører hjemme i samme målesesjon. (7 s var
første verdi; hevet til 8 s 2026-08-21 etter at 7 ble sett for kort — men se
merknaden om forholdene over: heller ikke den justeringen er gjort på bane.)
**Blokkert:** merket kommer ikke fram til basen ennå — `capture_trigger` ligger
i sidecaren på disk, men feltet på ledningen er ikke avtalt (**issue #11**,
label `backend`). Til det er på plass må timeout-utløste donasjoner skilles ut
for hånd, eller ikke i det hele tatt.

### ÅP-U1 — OCR-heuristikkens avviksterskel · label `ui`
> «OCR-finpussing av poeng (ML Kit, on-device, **UKALIBRERT heuristikk**): ≤ 0,2
> avvik → sømløs oppdatering; > 0,2 → «kunne ikke se treffene»»
> — UI-spec §12, linje 166

Skjermlayouten på Kongsberg-apparatet er ikke modellert
(`android/…/OcrVerifier.kt:14`). Grensa 0,2 er valgt, ikke målt.

**Fikk større konsekvens i v0.27.** Heuristikken plukker alle desimaltall i
[0, 10.9] fra hele skjermbildet, og *antallet* tall styrer nå en dialog: leser
den ett tall for mye eller for lite, får brukeren et avviksvarsel der det før ble
stille forkastet som `Inconclusive`. Grensa på 10 tall siler bort de groveste
tilfellene, men et enkelt streiftall (vind, avstand, serienummer) gjør en
korrekt serie til et «skjulte treff»-varsel. **Felttesten som mangler er nå
todelt:** treffer terskelen 0,2, og treffer *antallet*. Blir varselet hyppig og
falskt, er svaret å modellere hvor på skjermen poenglista står — ikke å heve
terskelen.

### ÅP-U2 — `RING_STEP_CM` · label `ui`
Ringavstanden på Kongsberg-skiva i cm per ringsteg er plassholder
(`android/…/Stats.kt:20`). Verdien går inn i klikkforslag og spredningsmål, så
den forplanter seg til alt som presenteres i cm.

### ÅP-B1 — Karantenetersklene for telefonsøk · label `backend`
> «5 mislykkede telefonsøk på én dag → karantene. **Anbefaling:** 1 dag karantene
> ved første overtredelse, eskalerende til 7 dager ved gjentakelse.»
> — backend_spec §3.1, linje 390–396

Speccen oppgir dette som en anbefaling, ikke som en besluttet verdi. Samme
avsnitt nevner IP-heuristikk og CAPTCHA ved terskel uten å tallfeste noen av dem.

---

## B. Feltinnhold og datamodell

### ÅP-K2 — Konkret feltinnhold i forskningsdatasettet · `TODO(eier)`
> «Det **konkrete feltinnholdet** i forskningsdatasettet er ikke avklart ennå.
> Fable skal etablere strukturen … og la selve feltdefinisjonene være et tydelig
> merket, utfyllbart punkt.»
> — kjerne-spec §6, linje 123–125; gjentatt §8, linje 145

Speilet i `backend_spec.md:324`
(`# TODO(eier): konkret feltinnhold for forskning ikke endelig avklart`) og
markert tre steder i koden: `backend/app/models/research.py:42`,
`backend/app/services/research_filter.py:15`,
`backend/app/routers/research.py:154`.

Konsekvens i dag: `research_filter.py` er en **tillatelsesliste** nettopp fordi
lista ikke er endelig — ukjente nøkler droppes stille i stedet for å bli delt som
standard. Når feltlista foreligger, er det den filen som utvides.

### ÅP-B2 — Skadedata: navn og feltliste stemmer ikke med eieravklaringen · label `backend`
> «**EIERAVKLARING 2026-08-06:** … 1. Egen bryter i `ResearchSharingPreference`
> (forslag: `share_wound_data`), av som standard. 2. `outcome`, `follow_up` og
> `ran_m` inn på tillatelseslista.»
> — backend_spec §7, linje 300–320

Implementert er en bryter som heter `share_injury_data`, med feltlista `wounded`,
`injury`, `hit_placement`, `shots_fired`, `tracking_distance_m`,
`tracking_time_min`, `dog_used`, `recovered` (backend_spec §7, linje 290–299).

Åpent: om de tre feltene i eieravklaringen er de samme opplysningene under andre
navn, eller tre felter som mangler. Krever at ÅP-K2 lander — feltnavn kan ikke
harmoniseres mot en liste som ikke finnes.

### ÅP-U3 — Dødelig-sone-radius per art × vinkling · label `ui`
> «Tabellverdier (minste halvakse ved hver vinkel), **kildebelagt**. Dette er nå
> hele sonedatajobben.»
> — UI-spec §10 punkt 1, linje 135

Uten disse tallene har hele Innsikt-flaten ingen modell bak fargene. UI-spec §9
lister «kilder for dødelig-sone-radier» som eget hjelpepunkt — kildene skal altså
vises til brukeren, ikke bare brukes.

### ÅP-U4 — Vinkeltaksonomien er ikke verifisert · label `ui`
> «Provisorisk 30°/60° + side/front/bak beholdt, men *ikke verifisert*. Hentes fra
> BH-materialet før implementasjon; skytter-oppgitt-vinkel er en kjent
> målefeilkilde å håndtere i forskningsdesignet.»
> — UI-spec §10 punkt 2, linje 136

### ÅP-U5 — «Dyret løp kortere enn x meter» — er x artsavhengig eller 100? · label `ui`
UI-spec §4 (linje 60) sier utfallet «dødelig» er operasjonalisert som «dyret løp
kortere enn x meter, **x artsavhengig**». UI-spec §8 (linje 122) sier «dyret løp
kortere enn **100 meter**». De to kan ikke begge være riktige, og kriteriet går
rett inn i jaktloggen og i forskningsdataene.

### ÅP-B13 — Skivebildene: koblingen fjernes, og vellykkede scans lagrer koordinater · label `backend`

**Retningen er avklart av eier 2026-08-20.** Den erstatter en tidsfrist på
skivebildene — 3 år ble diskutert og forkastet, fordi en frist utsetter
koblingen i stedet for å fjerne den, og fordi den krever en ryddejobb som ikke
finnes (ÅP-E2 er den samme mangelen ett annet sted).

**Del 1 er LUKKET 2026-08-21. Del 2 står åpen, og holder punktet åpent.**

Del 1 ble ikke lukket av at koden var skrevet, men av en observasjon: en ekte
donasjon kom inn i produksjon 2026-08-21 etter frikoblingen — `failed_analyses`
gikk fra 9 til 10 rader, mot et skjema uten `series_id` og uten `user_id`. En
donasjon *kan* dermed ikke bære en kobling til en konto; det er lest i basen,
ikke sluttet fra koden. `personvernerklaring.txt` ÅPENT PUNKT 6 er lukket på
samme observasjon.

### Målingen fra før frikoblingen — den kan ikke gjøres om igjen

Talt i produksjonsdatabasen 2026-08-21, rett før migrasjon `b8d24a0f5c17`:

| | |
|---|---|
| Rader i `failed_analyses` | **9** |
| Av dem med `series_id` | **6** |
| Av dem med `user_id` | **0** |

Det er altså **seks koblinger til en konto som forsvant** — ikke null, og ikke
elleve. Tallene står her fordi de er umulige å hente i ettertid: etter
migrasjonen finnes ikke kolonnene å telle.

**Avviket mot tallene i ÅP-E11, som ser ut som de teller det samme:** de gjør
det ikke, og de har tre ulike kilder.

- **9** er rader i databasen, målt 2026-08-21.
- **7** («sju objekter, 16 096 622 byte») er *objekter i R2*, målt av
  kopieringsjobben 2026-08-18. En rad uten `object_key` — en donasjon der
  opplastingen feilet — har ingen objekt å telle.
- **11** er ingen telling i det hele tatt. Det er **rad-ID-en** til
  verifikasjonsdonasjonen i ÅP-E11 steg 3 («rad 11»), og ID-er er en sekvens,
  ikke en beholdning. At ID 11 er utdelt mens bare 9 rader finnes, er ventet:
  `POST /v1/failed-analyses` gjør `flush()` for å få ID-en *før* opplastingen,
  og ruller raden tilbake hvis R2 svarer feil — sekvensverdien er da brukt opp.
  Nettopp det skjedde gjentatte ganger under de fire feilkonfigurasjonene
  2026-08-18 (ÅP-B12).

At «11» ble lest som et radtall, skjedde i en samtale og ikke i denne fila —
men det er verdt å ha skrevet ned hvorfor de tre tallene ikke skal stemme
overens.

### Det som skal bygges

1. ~~**`series_id` skal ikke lagres på donasjonen.**~~ **LUKKET 2026-08-21
   (B-52).** Ruten tar ikke lenger imot feltet, og migrasjon `b8d24a0f5c17`
   dropper kolonnen — også for radene som alt fantes, siden en «vi slutter å
   skrive den»-løsning ville latt seks eksisterende koblinger stå igjen.
   `user_id` på samme tabell ble droppet i samme migrasjon: den var tom, men
   en ferdig oppkoblet fremmednøkkel til `users` er en mulighet det nå kreves
   en ny migrasjon for å gjenåpne. **Ingen klientendring var nødvendig** —
   FastAPI ignorerer skjemafelt ruten ikke erklærer, og et 4xx ville vært
   ikke-`retryable` og dermed stille tap av donasjonen.
2. **Vellykkede scans skal ikke lagre bildet i det hele tatt**, bare
   treffkoordinatene (f.eks. polarkoordinater per poengenhet). Det gjelder
   taggen `ocr_match` (`android/KONTRAKT.md` §2), altså donasjonen som sendes
   når kjernen og OCR var enige. **Ikke bygget.**

**Del 2 er en kontraktendring, og halve jobben ligger hos klienten.** `image`
er påkrevd i ruten, og koordinater finnes ikke som felt; klienten køer sidecar
og JPEG (`android/KONTRAKT.md` §2–§3) og må slutte å legge ved bildet for
`ocr_match`. Meldt som **issue #10** med label `ui` 2026-08-21, med forslag om å
gjenbruke `{r_rel, theta, decimal, integer}` fra `/v1/stats` (`models/training.py`,
`Shot`) — det er kjernens §3-output, allerede kontraktfestet, og svarer til
«polarkoordinater i poengenheter». Serverhalvdelen bygges ikke før feltnavnene
er avtalt: et endepunkt uten kaller blir aldri verifisert (rot-`CLAUDE.md` §7.3).

**Konsekvensen som bør ses før det bygges:** `ocr_match`-bildene er det eneste
materialet som viser kjernen på en scan som *gikk bra*. ÅP-U14 trenger
over-deteksjon, og den ligger i `rejected` og `ocr_mismatch`, så den rammes
ikke — men muligheten til å etterprøve en vellykket analyse mot bildet
forsvinner. Hva koordinatene skal inneholde for at det fortsatt skal være
kalibreringsmateriale, er en del av oppgaven.

---

## C. Åpne designspørsmål fra eier

Punkter der eier selv har satt spørsmålstegn i speccen.

### ÅP-U6 — Gir kapabilitetskartet mening? · label `ui`
> «### Kapabilitetskart (sekundærvisning) **[Usikker på denne - gir den mening?]**»
> — UI-spec §5, linje 87

Ikke bygget. Innsikt-matrisen (v0.7, UI-spec §13) dekker deler av samme behov, så
spørsmålet er om kartet fortsatt har en rolle.

### ÅP-U7 — Hvor hører «år uten skadeskudd» hjemme? · label `ui`
> «**«År uten skadeskudd»** [Denne hører kanskje til kompetansekortet?]»
> — UI-spec §5, linje 93

Speccen sier samtidig at målet skal plasseres «i refleksjon/historikk, aldri som
et grønt lys i planleggingen» (linje 96) — de to utsagnene trekker i hver sin
retning.

### ÅP-U8 — Skal mørk modus kunne sensorstyres? · label `ui`
> «mørk modus (kan det sensorstyres? Bør i tilfelle også være valg)»
> — UI-spec §9, linje 131

I dag: lys/mørk/system som eksplisitt valg (UI-spec §12, «Profil»).

### ÅP-B3 — Er `trend` riktig størrelse? · label `backend` + `ui`
> «Merk at dette er en **retning** (en differanse), ikke et løpende snitt. Skal
> klienten vise nivået over de siste tjue skuddene, er det `avgScore` med et
> vindu — et annet felt og et annet delingsvalg.»
> — backend_spec §3, linje 213–220

Klienten har siden bygget en trendgraf på rullende 20-skudds-snitt (UI-spec §21,
«Serier»). Åpent: om `trend`-feltet fortsatt skal være en differanse, og om en
venn skal kunne se grafen — sistnevnte krever i så fall et nytt delingsvalg og et
nytt endepunkt.

### ÅP-B4 — Hvem eier sannheten når serie-synken kommer? · label `backend` + `ui`
> «**Serie-synk-køen er bevisst IKKE bygget ennå.** To parallelle synkveier over
> samme data (blob + per-post `/v1/stats`) bør ikke finnes før det er avgjort
> hvilken som eier sannheten. **Forslag:** bloben eier «alt mitt», `/v1/stats`
> eier det som skal kunne deles/aggregeres.»
> — backend_spec §13, linje 534–537

Forslaget er ikke bekreftet. Beslutningen må tas før §5-synken bygges, ikke etter.

### ÅP-U9 — Når tilbys forskningssamtykke? · label `ui`
Tre ulike svar i samme dokument:
- «etter fem skutte serier» (UI-spec §1, linje 21)
- «etter tredje fullførte økt» (UI-spec §7, linje 116)
- «jaktmål først, forskning tidligst 2 serier senere» (UI-spec §14, v0.8)

Klienten kan bare implementere én av dem.

### ÅP-U10 — Veikart, ikke besluttet omfang · label `ui`
> «Papirskive-støtte med auto-gjenkjenning; dyrefigur-CV; bevegelige mål i
> modellen; personlige målsettinger per stilling; **gamifisering av øvingen** …
> må balanseres mot at appen ikke skal oppmuntre til uforsvarlige skudd eller
> gjøre skadeskyting til «poengtap» som frister til underrapportering.»
> — UI-spec §10 punkt 4, linje 138

Parkert som veikart. Står her for at det ikke skal bli glemt, ikke fordi det
haster.

---

## D. Krever eier — juridikk, kontoer og drift

Ingen av disse kan lukkes av en instans. De blokkerer reell funksjonalitet.

### ÅP-E1 — Personvernerklæring + DPIA før forskningsdata samles inn
> «Personvernerklæring må foreligge, og det må avklares om Datatilsynet krever
> DPIA (sannsynlig, gitt forskningsformålet og sensitiviteten i jaktdata).
> Endepunktet kan bygges teknisk før dette, men **reell innsamling fra brukere
> venter til det juridiske er på plass**.»
> — backend_spec §7, linje 325–328; gjentatt §9, linje 369–370

Holdes av `RESEARCH_ENABLED` (backend) og `Dialogs.RESEARCH_ENABLED` (klient),
begge av. ÅP-B2 gjør ikke vurderingen mindre nødvendig — et mer sensitivt felt
gjør den mer nødvendig.

### ÅP-E2 — Ingen rutine tømmer `research.deletion_requests`
> «Selve slettingen i forskningslageret er et **manuelt/driftsansvar** — det
> finnes ingen jobbkjører som tømmer `research.deletion_requests`. `completed_at`
> er kolonnen den kvitteres i.»
> — backend_spec §9, linje 366–368

En bruker som har slettet kontoen sin, har fått en anmodning lagt inn — ikke
data slettet. Det er en forpliktelse med en frist, ikke et teknisk løst punkt.

### ÅP-E5 — Apple: utviklerkonto og verifisert domene
> «Krever Apple Developer Program + verifisert domene, fordi Sign in with Apple på
> Android går via web-flyten. Services ID-en er verdien.»
> — backend_spec §1 og §15, linje 582–586

Endepunktet er bygget og røres ikke. Klientknappen finnes ikke.

### ÅP-E7 — Eget domene for API og OAuth-redirect
> «Domene: eget domene for API-endepunkt og OAuth-redirect-URLer.»
> — backend_spec §0.1, linje 416

I dag `bestefar-api.fly.dev`. Domenet `jegeropplæring.no` finnes; e-postadresser
må bruke punycode-formen (`xn--jegeropplring-cgb.no`).

### ÅP-E8 — App-navn og språkstøtte før butikklansering
> «Bokmål i v1; **app-navn avklares før butikklansering**. Støtte for systemspråk
> / andre relevante språk.»
> — UI-spec §10 punkt 3, linje 137

### ÅP-E9 — Telefon-OTP krever betalt SMS-leverandør
> «Telefonnummer (OTP) **utsatt til v2** — krever betalt SMS-leverandør
> (Twilio/Vonage o.l.)»
> — backend_spec §1, linje 19–21

Konsekvens i dag: telefoninvitasjoner til lag får `delivery_status: failed` med
lenken vedlagt, så klienten kan dele den via ACTION_SEND (backend_spec §4).

---

## E. Uavklart teknisk retning

Ikke gjeld — beslutninger som mangler.

### ~~ÅP-B12 — `/health` sier «r2» også når lagringen er ubrukelig~~ · **LUKKET 2026-08-18**

Bygget som B-51. **Ingen av de to foreslåtte utfallene ble valgt.**
Feilkonfigurasjon teller *ikke* som «ikke konfigurert» — det ville vært å
behandle en feil som et fravær, nøyaktig fellen `BackupKeys.resolve` gikk i
(rot-`CLAUDE.md` §7.3), og `objstore` har forbudt det for opplastingsveien
siden B-45. Det ble heller ikke et eget felt, for da hadde `bilder` fortsatt
sagt `"r2"` mens lagringen ikke virket, og det er selve påstanden som var
usann.

I stedet fikk `bilder` en **tredje verdi**, som `database` allerede hadde:
`r2` | `ikke konfigurert (§6)` | `feilkonfigurert (<hva>)`. Bare et helt tomt
oppsett er «av»; halvveis satte secrets er «feilkonfigurert», med navnene på de
manglende i teksten. `POST /v1/failed-analyses` avviser det feilkonfigurerte
tilfellet før kroppen leses og logger hvilken tilstand det var; `kan_brukes()`
er definert som «`tilstand()` sier `r2`», så mottaket og `/health` ikke kan
komme i utakt igjen.

Sjekkene er de som er **alltid** feil, og de kjører uten nettverk: linjeskift
eller mellomrom rundt en verdi, `R2_ENDPOINT` som ikke er en URL eller har sti,
`R2_BUCKET` med skråstrek, `R2_ACCESS_KEY_ID` som ikke er 32 tegn. Den tredje
produksjonsfeilen — token uten EU-jurisdiksjon — er **fortsatt bare synlig ved
et faktisk kall**, og `tools/r2_check.py` kjører nå de lokale sjekkene først så
den ikke bruker en rundtur på å oppdage noe som var åpenbart.

Se `backend/BESLUTNINGER.md` B-51 for det som ble forkastet og hvorfor.

### ÅP-B10 — 34 av 48 svar er utypet, så kontrakten beskriver dem ikke · label `backend`
Målt 2026-08-08 ved generering av `contracts/openapi.json`. **De 14 rutene
Android-klienten faktisk kaller, har fått `response_model`** (lista i
`contracts/README.md`). De øvrige 34 er annotert `-> dict`, og FastAPI kan bare
utlede `{"type": "object", "additionalProperties": true}`.

Konsekvensen for de 34: den innsjekkede kontrakten forteller hvilke ruter som
finnes og hva de tar imot, men ikke hva de svarer med. `backend/KONTRAKT.md`
beskriver svarene i prosa; ingenting holder de to i takt.

Det gjenstående er venne-, lag- og forskningsflatene, `/v1/stats`, `/v1/profile`
og `/health` — altså det klienten ennå bare har front-end-skjelett for.

Åpent: om resten skal få modeller, og når. **Det er ikke gratis.** En
`response_model` filtrerer svaret, så et felt som i dag sniker seg med,
forsvinner uten at noe feiler. `PUT /v1/profile` er det konkrete eksempelet:
den legger på `advarsel` *bare* når et visningsnavn venter på moderasjon, og en
modell som glemmer det feltet ville fjernet meldingen brukeren skal se — stille,
og bare i det ene tilfellet en test sjelden dekker. Rekkefølgen bør derfor følge
hva klienten tar i bruk, ikke hva som er raskest å skrive.

### ÅP-B9 — Feedback-kvoten teller i minnet, per maskin · label `backend`
> «Holder for MVP med én maskin. Ved flere Fly-maskiner er telleren per maskin —
> den reelle grensen blir da N x limit.»
> — `backend/app/ratelimit.py`, modul-docstring

`FEEDBACK_RATE_PER_HOUR` er 5, men Fly kjører to maskiner, så den faktiske
grensen er 10 per IP per time. E-postkodene ble flyttet til basen av nettopp
denne grunnen; `/v1/feedback` ble stående igjen.

Åpent: om grensen skal flyttes til basen slik `services/quarantine.py` og
e-postkodene allerede er, eller om 10/time er en akseptabel grense — i så fall
bør tallet i konfigurasjonen si det, ikke halvparten av det.

Merk at selve verdien 5 heller ikke har noen dokumentert begrunnelse
(`backend/BESLUTNINGER.md`, «uten dokumentert begrunnelse»).

### ÅP-B6 — `GET /v1/teams/near` sorterer i Python · label `backend`
> «Det holder lenge, men må byttes til PostGIS eller en geohash-kolonne når
> tabellen vokser.»
> — backend_spec §4, linje 239–241

Åpent: hvilken av de to, og ved hvilket antall lag.

### ÅP-B14 — Ingenting håndhever lagringsfristene · label `backend`

Eier besluttet 2026-08-20 en oppbevaringstid per kategori driftsdata.
**Verdiene står i `personvernerklaring.txt` ÅPENT PUNKT 2 og skal ikke
gjentas her** — to kopier av en frist blir før eller siden to ulike frister.

Det som mangler er alt: det finnes ingen ryddejobb i koden, og ingen av
fristene håndheves. Brukte engangskoder, tilbakekalte økter,
karanteneoppføringer med IP, leverte meldinger, ubesvarte invitasjoner,
avstemninger og meldinger til utvikleren blir stående til kontoen slettes —
og de tre siste også etter det.

**Derfor kan ÅPENT PUNKT 2 i erklæringen ikke lukkes av at tallene er
bestemt.** Erklæringen sier hva som skal skje, ikke hva som skjer, og en
erklæring som lover sletting uten at noe sletter er verre enn ingen — det
står allerede i den fila, og det er fortsatt sant. Punktet her lukkes når en
jobb faktisk kjører og det kan vises hva den slettet.

Åpent: hva som kaller den (samme spørsmål som ÅP-B7 — se vurderingen der),
og om slettingen skal være hard eller en anonymisering for de kategoriene som
har verdi i aggregat.

Merk at skivebildene **ikke** hører hjemme her: de løses ved frikobling
(ÅP-B13), ikke ved en frist.

### ÅP-B7 — Frister avgjøres lat · label `backend`
> «Nå som push (fase 8) er på plass **bør et periodisk kall legges inn**, så
> varselet går ut på fristen og ikke ved neste besøk.»
> — backend_spec §11, linje 460–465

Gjelder lederavstemning og inaktiv-leder-utfordring, begge med 7-dagers frist.
Åpent: hva som skal kalle det — Fly cron, ekstern pinger, eller noe annet.

**Ryddejobben i ÅP-B14 trenger det samme (vurdert 2026-08-20).** Vurderingen,
uten at noe er valgt:

- **Mekanismen er én oppgave.** Begge trenger nøyaktig det samme: noe som kjører
  uten at en bruker har spurt om noe. Å bygge to måter å bli kalt på ville gitt
  to steder å glemme, og valget mellom Fly cron, ekstern pinger og noe annet er
  det samme valget i begge tilfeller. Det er også dette valget som blokkerer
  begge — ingen av jobbene kan skrives ferdig uten å vite hva som kaller dem.
- **Jobbene er to oppgaver**, og de hører til hver sin risikoklasse. ÅP-B7 gjør
  en avgjørelse tidligere enn den ellers ville blitt gjort; kjører den for ofte
  eller to ganger, skjer det ingenting galt, og verste utfall er et varsel som
  kommer sent. ÅP-B14 **sletter**, og en feil der er ugjenkallelig og treffer
  data brukeren ikke kan skaffe igjen. Den trenger ting den andre ikke trenger:
  tørrkjøring, tall per kategori før og etter, en øvre grense per kjøring, og en
  logglinje som kan siteres i erklæringen. Å slå dem sammen til én funksjon
  ville gitt slettingen samme letthet som avstemningen.
- **Hastene er ulike.** ÅP-B7 er en brukeropplevelse. ÅP-B14 holder
  `personvernerklaring.txt` ÅPENT PUNKT 2 åpent, og det punktet står mellom
  utkastet og en erklæring som kan publiseres.

Konklusjonen som *ikke* er trukket: hvilken mekanisme. Den er fortsatt det
åpne i dette punktet, og bør avgjøres én gang for begge.

### ~~ÅP-U12 — Klienten bygges ikke av noen automatikk~~ · **LUKKET 2026-08-08**

Android-jobben kjører nå. `if: false` er fjernet, SDK-nedlastingen er kablet
inn, og jobben bygger `assembleDebug` med wrapperen.

**Valget som ble tatt:** last ned OpenCV-SDK-en i jobben og buffer den, framfor
en forhåndsbygget container. Bufferen er ikke en optimalisering — en jobb som er
treig nok blir slått av, og en avslått jobb var tilstanden punktet handlet om.
Tre ting bufres: OpenCV-SDK-en (nøkkel på versjonsnummeret, så en bump ikke
leverer feil versjon i stillhet), NDK og CMake under SDK-roten (den posten som
avgjør om jobben tar tre eller femten minutter), og Gradle-cachen via
`gradle/actions/setup-gradle`.

**Kun debug.** Release krever signeringsnøkkelen, som ligger utenfor repoet og
ikke hører hjemme i CI.

**Jobben er verifisert som en ekte port**, ikke bare grønn: `assembleDebug` ble
kjørt lokalt med en innført Kotlin-feil og med en innført C++-feil, og feilet
begge ganger. I tillegg sjekker jobben utfallet — at APK-en finnes og at
`lib/arm64-v8a/libbestefar_jni.so` ligger i den. Faller `add_subdirectory` av
`core/` ut av byggingen, fanges det der og ikke av exit-koden alene.

**Første kjøring ble rød på riktig måte.** SDK-en ble pakket ut i workspace, og
`setup-gradle` sin wrapper-validering fant `samples/gradle/wrapper/gradle-wrapper.jar`
inne i den. Valideringen har rett — en fremmed wrapper-JAR i kildetreet er
nøyaktig det den skal fange — så løsningen var å pakke SDK-en ut i
`~/opencv-android-sdk`, utenfor treet, ikke å slå av sjekken.

**Målt kjøretid:** 17m39s kald buffer (`31265758984`), **2m31s varm**
(`31295862290`). Kald buffer inntreffer bare når SDK- eller NDK-versjonen
bumpes, så 2m31s er tallet som gjelder i praksis — og det er lavt nok til at
ingen får noen grunn til å slå jobben av igjen. Det var hele risikoen ved å
åpne dette punktet.

`android/gradlew` er sjekket inn med modus `100644`, og `./gradlew` feiler med
«Permission denied» på Linux. Jobben gjør derfor `chmod +x` selv. Å sette
kjørbar-biten i git ble prøvd og forkastet: utvikleren jobber på Windows, der
git ikke fanger den biten, så den ville falt av igjen ved neste endring uten at
noen kunne se det. Ett steg i workflowen er billigere enn en regel ingen merker
at de bryter.

Se `docs/ARCHITECTURE.md`, «Bygg/CI», og `android/CLAUDE.md`.

### ÅP-U13 — `device_id` på backupen er alltid tom · label `ui`
Serveren tar imot `device_id` på `PUT /v1/backup` og eksponerer den både i
`GET /v1/backup/meta` og i `X-Backup-Device-Id`. Klienten sender den ikke, så
feltet er alltid `""` (`contracts/openapi.json`, `Backup.kt`).

Ingenting er ødelagt — 409-vernet hviler bare på `client_ts`. Men feltet later
som om det bærer informasjon, og «hvilken telefon lastet opp dette?» er
nøyaktig spørsmålet man stiller den dagen to enheter har overskrevet hverandre.

Åpent: **hva ID-en skal være.** En installasjons-UUID identifiserer
installasjonen og overlever ikke reinstall — som kanskje er riktig, siden det
er nettopp den nye installasjonen som skal gjenkjennes som «en annen enhet». Et
modellnavn er ikke unikt. En Android-ID er enhetsbundet og mer personidentifi-
serende enn resten av det vi sender. Valget er ikke tatt, og det er et
personvernvalg like mye som et teknisk et.

*Delvis avklart 2026-08-08 (v0.20):* `app_version` ble droppet — serveren tar
ikke imot den — og `schema_version` sendes nå fra `Backup.SNAPSHOT_VERSION`.
Se `android/KONTRAKT.md` §3 og §9.

### ÅP-B8 — Manuell moderasjonskø for visningsnavn mangler flate · label `backend`
> «Den manuelle køen krever en admin-flate som ikke finnes ennå; navn som passerer
> regelsettet **godkjennes derfor direkte**.»
> — backend_spec §3, linje 199–201

Speccen forutsetter «regelsett + evt. manuell kø». Bare halvparten finnes.

**Og halvparten som finnes har en rest som ser ut som mer enn den er**
(oppdaget 2026-08-11): `routers/profile.py` legger et `advarsel`-felt på svaret
fra `PUT /v1/profile` — «Navnet vises for andre når moderasjonen har godkjent
det». Feltet settes bare når `moderation.review` returnerer `pending`, og
**den returnerer aldri `pending`**: den gir `approved` eller `rejected`, og
ingenting annet. Grenen er død kode.

Det betyr at en klient som bygger mot svaret, kan komme til å implementere en
«navnet ditt er til vurdering»-tilstand som aldri inntreffer. Feltet fjernes
ikke nå — det er riktig den dagen køen finnes — men det skal ikke leses som en
fungerende mekanisme før den gjør det.

**Falske positiver har ingen brukervei** (2026-08-12): ordlista er ikke lenger
tom (B-43), og en delstrengsmatch på foldet form kan ramme et ekte navn — særlig
på tvers av fornavn og etternavn, der mellomrommet er borte. Veien ut er
`DISPLAY_NAME_ALLOWLIST`, som krever at en operatør setter en miljøvariabel.
Brukeren får bare «velg et annet navn», og har ingen måte å si fra på. Det er
den samme manglende flaten som over.

### ÅP-K3 — Konfidensmålet er en interim-heuristikk · label `kjerne`
> «Kvalitetsmålet skal være at beregnede poeng ikke matcher maskinens score, men
> **OCR er ikke implementert enda**. Det skal ikke prioriteres akkurat nå.»
> — kjerne-spec §3, linje 67

`confidence` er i dag en heuristikk over ringpasning og treffskår
(`docs/ARCHITECTURE.md:79`). Det egentlige kriteriet krever OCR i kjernen —
backend_spec §8 spør om skjerm-OCR bør flyttes fra ML Kit i klienten til kjernen,
og det spørsmålet er ikke besvart.

### ÅP-K4 — MPI-firkanten (cluster-senter-ikonet) · label `kjerne`
> «Ikke ferdig testet, men prioriteres ikke nå på grunn av lav marginal verdi og
> tidspress.»
> — kjerne-spec §3, linje 54

Målt til ~0 marginalverdi på tilgjengelig data (`docs/ARCHITECTURE.md:87`).
Resultat-skjemaet har et `aux`-utvidelsespunkt så den kan legges til uten
ABI-brudd. Åpent: om den noen gang skal wires inn.

### ÅP-U14 — Over-deteksjon dedupliseres ikke i kjernen · label `kjerne` + `ui`
> «Doble merker ved multieksponering, mistenkt kamerabevegelse.»
> — `backend_spec.md` §8

Klienten håndhever fra v0.27 at en serie er 0–10 skudd, og bruker OCR-poengene
til å luke ut overtallige treff (`android/CHANGELOG.md` v0.27). Det er en
kontroll, ikke en løsning: klienten kan bare *forkaste* et treff den vet er
feil, og uten OCR-fasit må hele serien forkastes.

Den riktige fiksen ligger i kjernen, som er den eneste som ser hullene og kan
slå sammen to merker som er samme skudd. Den krever testbilder av faktisk
over-deteksjon, og de finnes ikke i `Testsett/` i dag.

**Materialet er nå på vei inn:** donasjonene med tagg `rejected` +
`status_code = 0`, og `ocr_mismatch` der `len(detected) > len(ocr)`, er begge
over-deteksjon med bilde (`android/KONTRAKT.md` §2). Åpent: hvor mange bilder
som trengs før terskelen kan kalibreres, og om dedupliseringen skal skje i
`hits`-modulen eller som et etterfilter.

**Én type falsk positiv er observert i felt, og den skjer før kjernen ser noe
som helst:** auto-capture utløste på et vitrineskap 2026-08-21. Den hører til
tersklene i ÅP-K1, der observasjonen står — nevnt her fordi materialet fra en
slik utløsning havner i den samme donasjonsstrømmen, og «kjernen fant merker
som ikke er skudd» ser likt ut i basen enten motivet var en skive eller ikke.

**Fra v0.29 kommer det en ny sort donasjon inn i samme strøm, og den må kunne
skilles ut.** Tidsgrensen på auto-capture (ÅP-K1) gjør at bilder gatingen aldri
ville sluppet gjennom, nå analyseres og kan bli donasjoner. Det er med hensikt —
de er det eneste materialet som finnes om falske negative — men de er *ikke*
representative for hva kjernen ser i drift, og et treningssett som blander dem
inn ukjent er verre enn ett uten dem. Klienten skriver `capture_trigger` i
sidecaren, men **feltet krysser ikke ledningen ennå** (issue #11). Inntil det
gjør det, kan donasjoner fra tidsgrensen ikke filtreres bort server-side.

**Bildene er treningsdata, og fra v0.28 er de førstegenerasjons.** Til og med
v0.27 ble køfila skrevet med `bmp.compress(JPEG, 92)`, altså en omkoding av de
pikslene kjernen fikk — en detektor finstilt på dem ville vært finstilt mot
artefakter den aldri møter i drift. Nå køes kameraets originalbytes. **Det
skiller de sju bildene som allerede ligger i R2 fra alle senere**, og
forskjellen må være kjent når settet brukes. Se ÅP-U16 for prisen (filstørrelse)
og ÅP-U15 for personopplysninger i stevnebilder.

### ÅP-U15 — Skivebilder fra stevne kan inneholde navn og medlemsnummer · label `ui` + `backend`

Donasjonene til `/v1/failed-analyses` er fotografier av apparatskjermen. På
øvingsskiver viser skjermen poeng og lite annet, og **det tilfellet er avklart:
de sju bildene som ligger i R2 i dag inneholder ingen personopplysninger.**

I stevnesammenheng er det ikke gitt. Skiven kan logges inn på med personlig
konto via QR, og squadding-lister inkluderer medlems-ID — altså kan skytterens
**navn og medlemsnummer** stå på skjermen og havne i et bilde brukeren donerer
for å hjelpe med å kalibrere en detektor. Samtykketeksten («Vi vil gjerne bruke
dette bildet til å forbedre appen») dekker bildet, ikke det som måtte stå på
det.

Åpent, og skal ikke løses i kode før det er avklart: om stevnebruk i det hele
tatt er et scenario appen skal støtte for donasjoner, og i så fall om svaret er
å beskjære skjermområdet, gjøre samtykket kontekstavhengig, eller å la være å
tilby donasjon når appen ikke kan vite hva som står på skjermen. Merk at dette
er en annen avveining enn resten av personvernvalgene i appen: her er
opplysningen ikke noe brukeren *oppgir*, men noe som følger med et bilde de tror
de gir bort noe annet med.

### ÅP-U16 — Originalbytene er 6–7 MB mot en grense på 8 MB · label `ui`

Fra v0.28 køes kameraets originalfil i stedet for en omkoding (ÅP-U14 krever at
treningsdataene er det kjernen faktisk så). Prisen er marginen:
`MAX_UPLOAD_BYTES` er 8 MiB, og originalene er 6–7 MB på telefonen dette er
prøvd på.

Klienten sjekker grensa før den køer, så en for stor fil gir ingen 413 — den
gir en `Log.w` og ingen donasjon. Det er riktig oppførsel (en 413 er ikke
`retryable` og ville uansett kastet elementet), men det betyr at en telefon med
større sensor kan slutte å bidra helt stille.

**Regnes ikke som løst før det er målt på flere telefoner.** Det som må vites:
hvor stor originalfila faktisk er på et utvalg kameraer, og om 8 MiB da er nok.
Blir svaret nei, er valget mellom å heve serverens grense (backend eier tallet)
og å gå tilbake til omkoding for de telefonene det gjelder — og det siste
gjeninnfører nettopp problemet ÅP-U14 trenger at vi unngår.

### ÅP-U11 — Speccen sier pinnet submodule, repoet er monorepo · label `ui` + `kjerne`
> «UI-prosjektet skal **ikke forke** OpenCV/C++-kjernen. Konsumer den som
> **pinnet avhengighet** — eget repo, tagget versjon, hentet via submodule eller
> pakket artefakt.»
> — UI-spec §11, linje 142

Faktisk struktur er ett monorepo der `android/` bygger `core/` direkte via
CMake/NDK (`docs/ARCHITECTURE.md`, «Repo-layout»). Intensjonen bak kravet —
ingen divergerende kopi, versjonert endring med bump — er ivaretatt av
`BESTEFAR_CORE_VERSION` i `core/include/bestefar/version.h`. Åpent: om speccen
skal rettes til å beskrive monorepoet, eller om oppdelingen fortsatt er målet.

---

## F. Utestående fra feltrunden på v0.28 (musingsUI «v022»)

**Denne seksjonen er av en annen art enn A–E.** Punktene over står her fordi de
*ikke kan besluttes i kode* — en terskel som må måles, et feltinnhold eieren må
definere. Punktene under **kan** besluttes i kode; de er ikke gjort. De ble
flyttet hit fra `musingsUI.txt` 2026-08-22 for at de skulle ha en ID og et sted
å bli lukket fra, i stedet for å ligge som løpende tekst i eierens fil.

Kilden er eierens gjennomkjøring av v0.28. Ingen UI-runde har rørt dem siden:
v0.29 og v0.30 gjaldt scan-vinduet og skivevisningen. Der noe er verifisert mot
koden, står det ved punktet — resten er rapportert, ikke ettergått.

### ÅP-U17 — Statuslinja er fortsatt lys i lys visning · label `ui`

Rapportert av eier på v0.28: «Statuslinjen øverst på telefonen er fremdeles lys
i lys visning. Der feilet forsøket på å gjøre den mørk.»

**Implementasjonen finnes.** `Ui.paintSystemBars` (runde 12) legger en svart
flate bak hver systemlinje og setter `isAppearanceLightStatusBars = false`, og
den kalles for *alle* aktiviteter via `ActivityLifecycleCallbacks` i
`BestefarApp.kt:28`. `values/themes.xml:21-25` setter i tillegg
`windowOptOutEdgeToEdgeEnforcement`, svart `statusBarColor` og
`windowLightStatusBar=false`.

Dette er altså **ikke en manglende funksjon, men en virkningsløs**, og det er
tredje forsøk på samme sak (runde 10, runde 12, og nå). Åpent: hvorfor flaten
ikke slår ut på eierens enhet. Merk `Ui.kt:135-137` — høyden settes fra
innsettene, og er innsettene 0 fordi opt-out-en *fortsatt* virker på den
enheten, blir flaten 0 px høy og hele mekanismen en no-op. Det er den første
hypotesen å måle.

Relatert: `TODO`-en om `windowOptOutEdgeToEdgeEnforcement` nederst i
`musingsUI.txt` — flagget er Googles midlertidige opt-out og forsvinner.

### ÅP-U18 — «Slett alle data»: ikon, plassering og knapperekkefølge · label `ui`

Tre ting i samme dialog, fra eier:

- STOP-ikonet skal være **større** og stå **over** overskriften, ikke ved siden.
- «Avbryt» og «Slett alle data» skal **bytte plass**. Begrunnelsen eieren gir er
  den som betyr noe: *det kritiske valget skal aldri være den naturlige
  knappen.*
- Varseltrekanten som brukes ved sletting **andre steder** i appen skal være
  **gul med sort kontur** (i dag rød, `Ui.warningDialog`).

Merk at `Ui.kt:164-169` allerede sier at trekanten kun skal brukes på
destruktive valg, «ellers slites ikonet ut og slutter å bety noe» — fargeskiftet
skal ikke undergrave den regelen.

### ÅP-U19 — Innloggingsflaten: fire punkter · label `ui`

- **Google-'G' bak «Fortsett med Google».** Verifisert mangler: knappen settes
  med ren tekst (`LoggInnActivity.kt:102,126`), og det finnes ingen
  `ic_google`-drawable i `res/`.
- **Tilbake-knapp nederst til høyre** — både på innloggingsskjermen og på den
  innloggede visningen. Samme krav er meldt for «Legg til lag»-dialogen.
- **Kontoen som vises må være permanent**, basert på e-post eller Google-konto,
  ikke visningsnavnet brukeren kan endre. Merk at `Store.neverBackedUp`
  (`Store.kt:666-671`) allerede holder `accountEmail`/`accountProvider` utenfor
  sikkerhetskopien nettopp fordi de hører til *denne* telefonens innlogging —
  feltet finnes altså, det er visningen som er feil.
- **Er QR-koden for å legge til venner aktiv?** Eierens spørsmål, ikke ettergått.

### ÅP-U20 — Gjenoppretting henter ikke tilbake tre profilfelter · label `ui`

Eier rapporterer at **fødselsår**, **jaktlag** og **«la venner finne meg»** ikke
kom tilbake etter gjenoppretting, mens «Mitt jaktmål» gjorde det.

**Delvis ettergått, og de tre har trolig ikke samme årsak:**

- `birthYear` (`Store.kt:143`) og `findable` (`Store.kt:255`) ligger i prefs, og
  `exportPrefs` er **generisk over hele prefs-fila** med bare fem unntak
  (`neverBackedUp`). De *skal* altså være med i bloben. Hvorfor de likevel ikke
  kom tilbake, er ikke funnet. Kandidater: kopien ble tatt før feltene var satt,
  eller det ble testet mot en eldre APK.
- **Jaktlag er noe annet.** Det finnes ikke i `Store`-prefsene i det hele tatt —
  lagmedlemskap er serverside. En gjenoppretting av en lokal blob kan derfor
  *ikke* gjenopprette laget, og det er ikke en feil i backupen, men et hull i
  flyten: etter innlogging må klienten hente medlemskapet fra serveren.

Skal lukkes med en faktisk observasjon, ikke med en kodelesing.

### ÅP-U21 — Dialoger uten OK-knapp, og manglende tilbake-knapper · label `ui`

«Ønsker du at …»-dialogen ved første dummy-scan har **ingen OK-knapp**. Eier
skriver: «Det tyder på at denne mangler et annet sted i flyten også» — så
oppgaven er å finne mønsteret, ikke bare det ene tilfellet.

Samme familie: «Legg til lag»-dialogen mangler tilbake-knapp nederst til høyre
(se også ÅP-U19).

### ÅP-U22 — Dialoger kan avbrytes ved trykk utenfor · label `ui`

To krav fra eier:

1. **«Lagre nøkkel»-dialogen skal ikke kunne avbrytes** ved trykk utenfor —
   brukeren må ta et aktivt valg.
2. **Lag en liste** over alle dialoger som i dag kan avbrytes slik.

**Verifisert at dette er reelt:** `setCancelable(false)` forekommer bare **7
ganger** i hele klienten (`Dialogs.kt` 3, `ResultActivity.kt` 4). Alle øvrige
dialoger er avbrytbare med et trykk utenfor.

Dette henger sammen med en observasjon som allerede har kostet noe: et trykk
utenfor deponeringsdialogen sendte `DELETE /v1/backup/key-escrow`
(produksjonslogg 2026-08-15 10:51:25, fikk 401), stikk i strid med at v0.26 sier
at en avbrutt dialog skal la bryteren stå urørt og ikke sende noe. **En avbrutt
dialog som utfører en handling er den farlige varianten**, og den skal
kartlegges først.

### ÅP-U23 — Startmeldingene kommer alle på én gang etter «slett alle data» · label `ui`

Eier: «Etter at alle data er slettet, popper alle startmeldingene våre opp på en
gang.» Hypotesen eieren selv foreslår — at alle har nådd triggerne sine, men
ikke er flagget som vist — er ikke ettergått.

I samme åndedrag: **det bør finnes et oppstartsvalg `<Start>` /
`<Fortell meg om appen>`**, som er en egen liten flate og ikke bare en fiks på
utløsningen.

### ÅP-U24 — Seriegrafens førsteakse skal være tid · label `ui`

Førsteaksen skal vise **datoer (dd.mm.åå) kun der det er registrert skudd**, men
plassert med **riktig innbyrdes avstand** i tid — ikke jevnt fordelt. To års
tidsrom er ~730 mulige punkter, og bare de med data får etikett.

Dekker data mindre enn to år, skal aksen **strekkes ut** så punktene blir
lesbare; skaleringsfaktoren er ikke bestemt og overlates til utvikler.

Beslektet, og fortsatt åpent fra en tidligere runde: trendvisningens andreakse
(rullende snitt over 20 skudd vs. dagsgjennomsnitt når det skytes mer enn 20 på
én dag), og hva som skal gjøres med siste økt — framskrive og justere, eller
vente. Se `musingsUI.txt`, avsnittet «Trendvisning», som står igjen der.

### ÅP-U25 — Visningsnavn: lagring ved tilbake, og moderasjonsforsinkelsen · label `ui`

- **Navnet forsvinner om tilbake-knappen trykkes.** Det skal lagres ved tilbake,
  eventuelt fortløpende per tegn.
- **Moderasjonen slår inn for sent.** Eier skrev «Fuck» uten at noe skjedde før
  hen gikk ut av profilsiden og inn igjen — da var navnet endret til «Mr».
  Neste forsøk ble avvist umiddelbart med varsel, som er riktig oppførsel.
  Åpent: om dette bare er forsinkelse, eller om koden må endres.

**«Mr» er trolig samme sak som ÅP-U26** — se der.

### ÅP-U26 — Innloggingsnavnet blir «Mr» · label `ui`

Ved innlogging uten gjenoppretting ble visningsnavnet bare «Mr». Eier spør om
det ikke burde vært Google-kontoens visningsnavn.

Verdt å merke seg at dette har vært oppe før fra motsatt kant: **issue #7**
(`backend`, lukket) het «Google-innlogging bruker e-postens lokaldel som
visningsnavn; name-kravet forkastes». «Mr» ser ut som lokaldelen av eierens
e-postadresse, så symptomet kan ha overlevd lukkingen — eller ha kommet tilbake
på klientsiden. Sjekk hva `/v1/auth`-svaret faktisk inneholder før det rettes i
klienten.

### ÅP-U27 — Småting med tekst og tid · label `ui`

- **«Mitt jaktmål»-dialogen etter første serie skal hete «Velg ditt jaktmål».**
  Verifisert ikke gjort: `strings.xml:358` `jaktmaal_title` er fortsatt «Mitt
  jaktmål». Merk at samme streng brukes som overskrift i profilen
  (`ProfilActivity.kt:259`), der «Mitt jaktmål» er riktig — så dette krever to
  strenger, ikke en endring av den ene.
- **Toasten ved innsending av avvist scan vises for kort** til å rekke å leses,
  og kommer sent. Tidsvinduet må økes.
- **Gjenopprettingskoden:** vis `######` i tekstfeltet, og oppgi tidsgrensen som
  «15 min».

### ÅP-U28 — Gjenopprettingstilbudet tar ikke hensyn til lokal tilstand · label `ui`

Det største punktet i runden, og eierens egen instruks i sin helhet.

Ved `is_new = false` og en kopi på serveren tilbys gjenoppretting umiddelbart
etter innlogging, **uten å se på om brukeren har data lokalt**. Situasjonen som
avdekket det: en bruker med lokale serier logger inn og får tilbud om å erstatte
dem med en eldre kopi. Det riktige der er å tilby en **ny kopi**, ikke en
gjenoppretting.

**Undersøk først:**

- Hva skjer faktisk ved avbryt? v0.25 la inn at en kopi ikke får erstatte data
  uten bekreftelse, med avbryt som standardvalg — **men den koden er aldri
  kjørt.** Verifiser at lokale data står urørt etter avbrutt gjenoppretting.
- Sammenlignes kopiens `client_ts` mot lokal tilstand i det hele tatt, eller er
  det bare 200/404 fra `/meta` som styrer?

**Implementer så en avgjørelse basert på begge sider:**

| Lokalt | Kopi | Tilbud |
|---|---|---|
| ingen data | finnes | gjenoppretting (dagens oppførsel, riktig) |
| data finnes | eldre | tilby **ny kopi**; gjenoppretting mulig, men ikke foreslått |
| data finnes | nyere | vis begge sider og la brukeren velge, **avbryt som standard** |

«Eldre/nyere» er ikke hele bildet — en nyere kopi kan inneholde *mindre* enn det
lokale hvis den ble tatt fra en annen telefon. **Tallene fra `Backup.les`
(N serier, M jaktposter) er det som faktisk sier hva som står på spill, og de
bør vises før noe overskrives.**

Dette er samme lærdom som allerede står i rot-`CLAUDE.md` §7.3 — «en destruktiv
operasjon skal lese ferdig før den skriver» — og punktet er ikke lukket før
beslutningen ligger *mellom* lesing og skriving.

**Teksten skal også være en annen når tilbudet utløses av innlogging:**
«Eksisterende konto funnet! Gjenopprett?» og «Kontoen ble lagret (fortsetter som
før)».

Se ÅP-U22 for deponeringsdialogen, som hører til samme flate.

### ÅP-U29 — Lagene finnes bare lokalt, så ingen lag-rute kan kalles · label `ui`

**Dette blokkerer hele lag- og venneflaten, og ble oppdaget da punkt 4
(«Overfør laget») skulle kobles 2026-08-22.**

Backend har 22 ruter under `/v1/teams/*` og `/v1/friends/*`. Klienten kaller
**ingen** av dem — verifisert ved å liste hvert `Api.*`-kall i klienten:
19 kall totalt, og det eneste i denne familien er `/v1/messages`.

Årsaken er ikke at kallene mangler, men at **identitetene mangler**:

- `Team.id` er en lokal UUID fra `Store.newId()`. Klienten har aldri kalt
  `POST /v1/teams`, så laget finnes ikke på serveren og ID-en betyr ingenting
  der.
- `Team` har `memberCount: Int` — et *tall*, ikke en medlemsliste. Det finnes
  ingen medlems-ID-er å sende som `{member_id}`.
- «Medlemmer» utledes lokalt av `store.friends().filter { t.id in it.teamIds }`,
  og `Friend.id` er også lokal.
- Ingen rolle lagres per medlem; `Team.hasLeader: Boolean` er alt.

**Konsekvensen for rekkefølgen:** ruter som tar `{team_id}` eller `{member_id}`
— overføring, fjern medlem, invitasjon, avstemning, inaktiv-leder — kan ikke
kobles før laget er opprettet på serveren og medlemslisten hentes derfra.
`POST /v1/teams` og `GET /v1/teams/{id}` er altså ikke ett punkt blant flere;
de er forutsetningen for resten.

Åpent: om lokale lag skal migreres opp ved første innlogging, eller om de skal
regnes som noe annet enn serverlag. Det er en eierbeslutning — en stille
opprettelse av alle lokale lag på serveren ved innlogging er ikke åpenbart
riktig, og en bruker som har laget «Test» tre ganger vil ikke ha tre lag.

### ÅP-U30 — Delingsmodellen: hvem ser hva · label `ui` + `backend`

**Eierbeslutning 2026-08-22. Besluttet, ikke bygget — hører til steg (4).**

- **Venner ser resultater.**
- **Lagmedlemmer ser navn og medlemskap, ikke resultater.**
- **Ingen bryter og ingen nivåer.** Vil du dele med noen i laget, blir dere
  venner. Det er hele mekanismen.
- **Lagleder ser ikke mer enn andre medlemmer.** §11 gir *administrative*
  fullmakter — invitere, endre navn, fjerne medlem, overføre lederskap — og
  ikke innsyn.

**Ett unntak, og det følger av inaktiv-leder-utfordringen:** laget kan se om
lederen er inaktiv, som **ja/nei**. Ikke når hen sist var pålogget, ikke noe
tall — bare den ene boolske opplysningen utfordringen trenger for å kunne
avvises («Lagleder er ikke inaktiv. Ta kontakt.»).

**Dette skal inn i personvernerklæringen.** `personvernerklaring.txt` er ikke
UI-eid, så den er ikke rørt herfra; punktet er at et lagmedlems
aktivitetsstatus blir synlig for laget, og at det er en ny opplysningstype å
beskrive.

Merk hva beslutningen rydder bort: den fjerner behovet for delingsnivåer per
lag, per venn eller per felt. Det er også grunnen til at den er verdt å skrive
ned — «ingen bryter» er et valg, og uten begrunnelsen ved siden av vil noen før
eller siden foreslå en bryter igjen.

### ÅP-U31 — Venner er serverbaserte og unike per `public_id` · label `ui` + `backend`

**Eierbeslutning 2026-08-22. Hører til steg (4).**

- Vennskap ligger på **serveren**, ikke lokalt, og er **unikt per `public_id`**.
- **Vennskap og lagmedlemskap er uavhengige relasjoner.** Man kan være det ene
  uten det andre, i begge retninger.
- **E-postadresser deles ALDRI automatisk** — verken med lag eller med venner.

Konsekvens for dagens kode, som må ryddes i steg (4): `Friend` er i dag en rent
lokal post med lokal `id`, og `VennerActivity` grupperer venner etter
`Friend.teamIds`. Begge deler forutsetter at vennskap og medlemskap er samme
sak, som beslutningen sier at de ikke er. `DevTools` fabrikkerer også venner
direkte inn i et lokalt lag.

Se ÅP-U29: så lenge medlemslista kommer fra serveren og vennelista er lokal, er
det to ulike mengder som ikke kan sammenlignes. De kan ikke slås sammen før
venner faktisk er serverbaserte.

### ÅP-U32 — Medlemsprofil i lagvisningen · label `ui`

**Eierbeslutning 2026-08-22. Hører til steg (4).**

Klikk på et lagmedlem **som ikke er venn** åpner en profil med:

- visningsnavn
- bruker-ID
- mulighet til å sette et **lokalt kallenavn** (se ÅP-U33)
- en **«Legg til som venn»**-knapp

Medlemmer som **allerede er venner** vises i lagvisningen med **antall
øvelsesskudd** ved siden av navnet — som er delingsmodellen i ÅP-U30 gjort
synlig: resultater følger vennskapet, ikke medlemskapet.

Avhenger av ÅP-U29 (medlemslista må komme fra serveren) og av **issue #14** —
uten å vite hvem *jeg* er i `members[]` kan visningen ikke skille «meg» fra «et
medlem jeg kan legge til som venn», og ville tilbudt brukeren å bli venn med seg
selv.

### ÅP-U33 — Identitet og gjenkjennelse ved navnebytte · label `ui`

**Eierbeslutning 2026-08-22, delvis. Hører til steg (4).**

- **Lokalt kallenavn per person**, lagret på egen enhet, **deles med ingen**.
  Dette er **hovedløsningen** på at folk skal kjenne igjen hverandre når
  visningsnavn endres.

Feltet finnes allerede i klienten som `Friend.nickAlias`, men bare for venner og
bare lokalt i dagens lokale vennemodell. I steg (4) må det gjelde enhver person
man møter i appen — også et lagmedlem som ikke er venn (ÅP-U32).

- **`public_id` vises diskret i medlemslista som fasit** — under visningsnavnet.
  Formålet er at to personer med samme navn kan skilles, og at en navneendring
  ikke skjuler hvem det er.

Rollefordelingen mellom de to: **kallenavnet er hovedløsningen**, `public_id` er
**oppslagsverket når kallenavnet ikke er satt**. Den skal derfor være lesbar,
ikke fremtredende.

**Konsekvensen er vurdert og akseptert, ikke oversett:** `public_id` er også det
man oppgir for å legge til en venn. Å vise den i medlemslista betyr at **alle i
laget kan sende meg en venneforespørsel**. Det er greit — vi er i samme lag — men
det er et valg, og det skal stå som et valg. Den som senere vurderer å vise
`public_id` et sted *utenfor* laget, må ta stilling til det samme på nytt, og
der er svaret ikke gitt.

Merk sammenhengen med ÅP-U30: laget ser navn og medlemskap, ikke resultater. En
venneforespørsel fra et lagmedlem er derfor veien fra det ene til det andre, og
`public_id` i medlemslista er det som gjør den veien farbar uten at noen må
utveksle noe utenfor appen.

### ÅP-E12 — Skal appen be om innlogging når sesong 2 starter? · krever eier

UI-speccen slår fast: «En jeger som bare vil scanne skiver skal ikke føle at hen
har hoppet over noe. **Ingenting i appen ber om innlogging uoppfordret.**»

Eier stiller spørsmålet selv: sikkerhetskopi av resultater er viktig, og det
krever konto. Skal appen da gi et prompt når sesong 2 starter?

Det er en reell motsetning mellom invariant 1 i `android/CLAUDE.md`
(offline-først, ingen funksjon krever innlogging som ikke *er* om noen andre) og
det å verne data brukeren har samlet i et helt år. **Kan ikke avgjøres i kode**
— det er et produktvalg om hvor påtrengende appen har lov til å være, og det er
eierens.

---

## Avklart

Punkter som er lukket. Beholdes med dato så de ikke tas opp igjen.

| Punkt | Avklart | Utfall |
|---|---|---|
| `bf_version()` over FFI | 2026-08-06 | Løst: `BESTEFAR_CORE_VERSION` i `core/include/bestefar/version.h`, uavhengig av appens `versionName`. backend_spec §8. |
| `kills[]` som liste i venne-modellen | 2026-08-05 | Kan ikke leveres. Løst som flyktig push-kunngjøring (`POST /v1/hunts/announce`); ingenting om art eller sted lagres. backend_spec §3. |
| Definisjonen av `trend` | 2026-08-05 | Siste ~20 skudd minus de ~20 foregående, talt i skudd. Se ÅP-B3 for det som fortsatt er åpent. |
| Skadedata i forskningsdata | 2026-08-06 | Skal kunne lagres, bak egen bryter, av som standard. Se ÅP-B2 for navn/feltliste. |
| Utskiftning av `BACKUP_ESCROW_SECRET` | 2026-08-06 | `_OLD`-fallback med omkryptering ved lesing + nøkkel-ID i `/health`. Siste del lukket som ÅP-E6. |
| ÅP-E3 — `FCM_SERVICE_ACCOUNT_JSON` | 2026-08-10 | **Satt i produksjon.** Verifisert ved at `GET /health` svarer `"push":"fcm"` og ikke `"push":"log"` — `push.backend_name()` gir `fcm` bare når tjenestekonto-JSON-en lot seg lese. `FCM_PROJECT_ID` ble aldri nødvendig; `push.project_id()` leser den fra JSON-en. Merk hva testen *ikke* dekker: at nøkkelen er lesbar er ikke det samme som at FCM godtar den. Det viser seg først når et varsel faktisk sendes. |
| ÅP-E4 — `GOOGLE_CLIENT_IDS` | 2026-08-10 | **Satt i produksjon.** Verifisert ved at `POST /v1/auth/google` med et ugyldig `id_token` svarer **401 «Ugyldig Google-token»** og ikke 503 — altså at `aud`-sjekken faktisk kjører i stedet for å bli hoppet over som ukonfigurert. Testen sier at secreten er lest; den sier ikke at *verdien* er riktig web-klient-ID. Det viser seg først ved en ekte innlogging fra klienten. **Den innloggingen er gjort 2026-08-15:** bruker logget inn med Google fra v0.27 og gjenopprettet en kopi tatt med v0.25 — en gjennomført gjenoppretting krever gyldig tokenpar, så verdien er riktig. Dermed er også ÅP-E4 helt lukket, ikke bare delvis. Se `android/ARCHITECTURE.md`. |
| ÅP-B5 — Cloudflare R2 er betalt for og ubrukt | 2026-08-15 | **Koblet inn.** Speccen avgjorde valget: §6 og §0.1 sier begge at bildene ligger i objektlagring, så «avvikle oppsettet» ville krevd en spec-endring og ikke en kodebeslutning. `POST /v1/failed-analyses` laster opp via `app/services/objstore.py` (SigV4 uten boto3) og lagrer bare `object_key`. Backend B-44/B-45/B-46. Gamle `image_legacy`-rader er ikke flyttet — se ÅP-B11. |
| ÅP-D1 — CI-actionene kjørte på avviklet Node | 2026-08-16 | **Bumpet, alle tre områdene i én commit etter avtale med eier** (`a020dc3`) — den ene endringen eierskapsmodellen ikke dekker, notert i commit-meldingen og i toppen av `ci.yml` så de to andre instansene ikke lurer. `checkout` v4→v7, `setup-python` v5→v7, `setup-java` v4→v5, `cache` v4→v6, `setup-gradle` v4→v6, `setup-flyctl` `@master`→`@1.6`. **Runtime ble lest fra `runs.using` i hver actions `action.yml` på den taggen vi peker på**, ikke antatt: `setup-flyctl@1.5`, som GitHub melder som «latest release», kjører node20 — å pinne til den ville låst oss på runtimen som avvikles, og `@master` var allerede node24. Verifisert på to måter, siden grønn kjøring og fravær av advarsel er to ulike påstander: alle fem jobbene `success` (CI 31940266418, Deploy backend 31940266433), **og** null node20-annotasjoner mot fire i kjøringen før (31879818643). Pinningen av `setup-flyctl` var eget avsnitt i punktet og er lukket samtidig. |
| ÅP-B11 — gamle bilder i `image_legacy` | 2026-08-15 | **Flyttet til R2, ikke kastet** (eieravklaring: de er materialet ÅP-U14 mangler). Kjørt med `tools/migrate_legacy_images.py --utfoer` på Fly-maskinen: fem rader, 13 byte til 3 844 036 byte, 10 509 298 byte til sammen, alle JPEG, alle med innsendingsdatoen i nøkkelen. Hver rad ble lest tilbake og sammenlignet byte for byte før `image_legacy` ble tømt (B-48). Verifisert ved at `GET /health` gikk fra `"bilder": "r2 (5 gamle rader i basen)"` til `"bilder": "r2"`. Merk to ting for ettertiden: «11 MB-raden» fantes ikke — det var totalen, og ingen rad var over 8 MB-grensen — og rad 2 er 13 byte, altså en avkortet innsending uten bildeinnhold. Fire av de fem er `ocr_mismatch`. |
| ÅP-E10 — R2-secretene og signeringen | 2026-08-15 | **Virker mot ekte R2.** Secretene sto i Fly fra før (punktet påstod først det motsatte, og ble rettet). Verifisert ved å kjøre `tools/r2_check.py` **inne på Fly-maskinen**, der secretene faktisk er: `flyctl ssh console -a bestefar-api -C "python tools/r2_check.py"` svarte `PUT: ok` / `GET: ok (64 byte, identisk)` / `DELETE: ok` mot bucketen `bestefar-scan-failures`. Det er hele rundturen, ikke bare at verdiene lot seg lese — `/health` sier `"bilder": "r2"` uansett om Cloudflare avviser signaturen. Verktøyet ligger i imaget nettopp derfor (B-47). Merk hva testen *ikke* dekker: at en ekte donasjon fra klienten går gjennom hele veien; det viser seg først når `POST /v1/failed-analyses` svarer 201 med en `object_key`. |
| ÅP-E11 — R2-secretene måtte byttes til den EU-bundne bucketen | 2026-08-20 | **Gammel bucket tømt og slettet; all lagring går til `bestefar-scan-failures-eur` (jurisdiksjon `eu`).** Steg 1–3 ble gjort 2026-08-18: secretene byttet, sju objekter (16 096 622 byte) kopiert med uendret nøkkel og lest tilbake byte for byte, og en ekte donasjon fra appen verifisert i den nye bucketen med **0 byte i den gamle** — altså at også skrivingen, ikke bare kopieringen, går dit. **Merk hva tallene her teller:** «sju objekter» er objekter i R2, ikke rader i basen (en donasjon der opplastingen feilet har ingen objekt), og «rad 11» er en rad-**ID**, ikke et antall. Databasen hadde 9 rader da den ble talt 2026-08-21 — se ÅP-B13, som forklarer hvorfor de tre tallene ikke skal stemme overens. Steg 4 er gjort 2026-08-20: `bestefar-scan-failures` er ikke bare tømt, den er slettet, så det finnes ikke lenger en bucket uten jurisdiksjonsbinding å komme i skade for å skrive til. Fire feilkonfigurasjoner måtte rettes underveis uten at noe hos oss fanget dem — det ga ÅP-B12/B-51. Detaljene: `backend/BESLUTNINGER.md` B-50, `backend/CHANGELOG.md` 2026-08-16. |
| ÅP-E6 — kopi av `BACKUP_ESCROW_SECRET` utenfor Fly | 2026-08-10 | **Utført av eier.** Verdien finnes nå lagret et annet sted enn i Fly secrets, så den kan gjenopprettes hvis Fly mister den eller den overskrives ved et uhell. Hvor kopien ligger, står ikke her og skal ikke stå her. Dermed er alle tre tiltakene i backend_spec §2.1 på plass. |
