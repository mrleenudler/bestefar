# ÅPNE PUNKTER

Alt i de tre spesifikasjonene som **ikke kan besluttes i kode**: `TODO(eier)`,
eksplisitt åpne spec-punkter, og verdier som er merket ukalibrerte eller
provisoriske.

**Punktene står her fordi de ikke kan finnes på.** En terskel som skal kalibreres
mot maskinvare, en radius som skal kildebelegges, et feltinnhold som skal
defineres av prosjekteier — ingen av dem blir riktigere av at en instans gjetter
en plausibel verdi. Denne fila foreslår derfor ingen verdier. Den sier hva som
mangler, hvor det står, og hvem som kan avgjøre det.

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

### ÅP-E11 — R2-secretene må byttes til den EU-bundne bucketen
Den gamle bucketen `bestefar-scan-failures` har *location hint* `EEUR` uten
jurisdiksjonsbinding — et hint er ingen garanti, og «Eastern Europe» dekker land
utenfor EØS. `bestefar-scan-failures-eur` er opprettet med jurisdiksjon `eu`.
**Steg 1 og 2 er gjort 2026-08-18.** Secretene er byttet, og kopieringen er
kjørt: **sju objekter, 16 096 622 byte** — ikke fem. De to ekstra er ekte
donasjoner som kom inn etter at R2 ble koblet på, og de ble med fordi jobben
leser nøklene fra databasen og ikke fra en telling. Verifisert ved at en ny
tørrkjøring etterpå melder alle sju som «allerede der», altså lest fra målet og
sammenlignet byte for byte. `r2_check.py` mot den nye bucketen: PUT/GET/DELETE
ok.

Fire feil måtte rettes underveis, og **ingen av dem ble fanget av noe hos oss** —
alle tre første svarte `/health` `"bilder":"r2"` på: `R2_ENDPOINT` med bucketnavn
i stien (signaturen dekket en annen sti enn forespørselen), API-token uten
EU-jurisdiksjon, og Access Key ID satt til plassholderteksten. Se ÅP-B12.

**Steg 3 er gjort 2026-08-18.** En ekte donasjon fra appen (rad 11,
`submitted_at` 18:53:55Z, `feilanalyse/rejected/2026/08/18/11-…jpg`) ligger i
`bestefar-scan-failures-eur` med 3 289 511 byte — og **0 byte i den gamle
bucketen**, altså ikke der i det hele tatt. Den var heller ikke blant de sju
kopierte objektene. Skrivingen går derfor direkte til den EU-bundne bucketen,
ikke bare kopieringen.

**Det som gjenstår er steg 4: den gamle bucketen er ikke tømt.**

Opprinnelig rekkefølge, som **rekkefølgen er en del av tiltaket**:

1. Bytt til den nye bucketen. Jurisdiksjonsbundne buckets har **eget
   endepunkt**, så `R2_ENDPOINT` må endres, ikke bare `R2_BUCKET`:

   ```powershell
   flyctl secrets set R2_ENDPOINT="https://<konto-id>.eu.r2.cloudflarestorage.com" R2_BUCKET="bestefar-scan-failures-eur" -a bestefar-api
   flyctl secrets set R2_KILDE_ENDPOINT="https://<konto-id>.r2.cloudflarestorage.com" R2_KILDE_BUCKET="bestefar-scan-failures" -a bestefar-api
   ```

   Er API-tokenet bucket-begrenset og ikke kontoomfattende, trengs også
   `R2_KILDE_ACCESS_KEY_ID` og `R2_KILDE_SECRET_ACCESS_KEY` — symptomet er
   `AccessDenied` fra den ene av dem.
2. Kjør kopieringen **etterpå**, ikke før: da skriver tjenesten allerede til den
   nye bucketen, og kilden kan ikke få nye objekter mens jobben går.
3. Verifiser med `tools/r2_check.py` og en **ekte donasjon** fra appen.
4. Først da: tøm den gamle bucketen. Egen runde, egen beslutning.

Punktet står til steg 4 er gjort. `personvernerklaring.txt` kan ikke si at
bildene ligger i EU før den gamle bucketen er tom (ÅPENT PUNKT 10 der).

---

## E. Uavklart teknisk retning

Ikke gjeld — beslutninger som mangler.

### ÅP-B12 — `/health` sier «r2» også når lagringen er ubrukelig · label `backend`
Målt 2026-08-18, under byttet til den EU-bundne bucketen (ÅP-E11). Fire forsøk,
tre ulike feilkonfigurasjoner, og `GET /health` svarte `"bilder": "r2"` gjennom
alle sammen — mens produksjonen ikke kunne skrive et eneste bilde og hver
donasjon fikk 503.

`objstore.er_konfigurert()` sjekker bare at de fire verdiene er ikke-tomme. To
av de tre feilene er trivielt sjekkbare uten nettverk:

- **Access Key ID som ikke er 32 tegn** er alltid feil. R2 sier det selv:
  «Credential access key has length 9, should be 32».
- **`R2_ENDPOINT` med sti i seg** er alltid feil for oss — koden legger selv på
  `/{bucket}/{nøkkel}`, så en sti i endepunktet gir dobbelt bucketnavn og en
  signatur som dekker en annen sti enn forespørselen.

Den tredje (token uten EU-jurisdiksjon) kan bare oppdages ved et faktisk kall,
og er `r2_check.py` sitt område.

Åpent: om feilkonfigurasjon skal telle som «ikke konfigurert» — da svarer
`/health` `ikke konfigurert (§6)` og donasjonene får 503 med en logglinje som
sier hva som er galt — eller om det skal være et eget felt. Det første er
enklest og gjør at feilen ser ut som det den er; motforestillingen er at
«ikke konfigurert» da dekker to ulike tilstander.

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

### ÅP-B7 — Frister avgjøres lat · label `backend`
> «Nå som push (fase 8) er på plass **bør et periodisk kall legges inn**, så
> varselet går ut på fristen og ikke ved neste besøk.»
> — backend_spec §11, linje 460–465

Gjelder lederavstemning og inaktiv-leder-utfordring, begge med 7-dagers frist.
Åpent: hva som skal kalle det — Fly cron, ekstern pinger, eller noe annet.

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
| ÅP-E6 — kopi av `BACKUP_ESCROW_SECRET` utenfor Fly | 2026-08-10 | **Utført av eier.** Verdien finnes nå lagret et annet sted enn i Fly secrets, så den kan gjenopprettes hvis Fly mister den eller den overskrives ved et uhell. Hvor kopien ligger, står ikke her og skal ikke stå her. Dermed er alle tre tiltakene i backend_spec §2.1 på plass. |
