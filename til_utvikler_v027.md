# Til utvikler — v0.27

---

# backend — `/health` skiller «av» fra «i stykker» (ÅP-B12 lukket, B-51)

Bakgrunnen er byttet til den EU-bundne bucketen dagen før. Det tok fire forsøk
å få secretene riktige, og `GET /health` svarte `"bilder": "r2"` gjennom alle
sammen — mens produksjonen ikke kunne skrive et eneste bilde og hver donasjon
fikk 503. Feltet sa altså «lagringen er koblet på» når det eneste den visste
var at fire strenger ikke var tomme.

## Hva som ble valgt

Punktet stilte to muligheter: la feilkonfigurasjon telle som «ikke
konfigurert», eller lage et eget felt. **Ingen av dem ble valgt.**

Det første er å behandle en feil som et fravær — nøyaktig det
`BackupKeys.resolve` gjorde hos klienten, der en 405 så ut som «ingen nøkkel
deponert» i tre versjoner. `objstore` har forbudt den formen for
opplastingsveien siden R2 ble koblet inn; å innføre den i diagnostikken ville
vært å bygge inn feilen vi allerede har betalt for. Det andre lar `bilder`
fortsette å si `"r2"` mens lagringen ikke virker, og det er selve påstanden som
var usann.

I stedet fikk feltet en **tredje verdi** — samme tredeling som `database`
allerede hadde:

| `bilder` | Betyr |
|---|---|
| `r2` | Ingenting vi kan se uten å spørre Cloudflare er galt |
| `ikke konfigurert (§6)` | **Ingen** av de fire verdiene er satt — funksjonen er av |
| `feilkonfigurert (…)` | Verdier er satt, og noe er beviselig galt |

Halvveis satte secrets er «feilkonfigurert», ikke «av», og navnene på de
manglende står i teksten. Bare et helt tomt oppsett er «av», fordi det er en
normal tilstand i utvikling.

## Hva som sjekkes

Bare det som er **alltid** feil, og alt sammen uten et nettverkskall:

- mellomrom eller linjeskift rundt en verdi — usynlig i Fly-panelet, gir
  `SignatureDoesNotMatch`, og ser ut som feil nøkkel
- `R2_ENDPOINT` som ikke er en URL, eller som har sti i seg (koden legger selv
  på `/{bucket}/{nøkkel}`, så signaturen dekker da en annen sti enn
  forespørselen)
- `R2_BUCKET` med skråstrek
- `R2_ACCESS_KEY_ID` som ikke er 32 tegn — R2 sier det selv: «Credential access
  key has length 9, should be 32»

To av de tre produksjonsfeilene fra 2026-08-18 fanges av dette. **Den tredje —
token uten EU-jurisdiksjon — gjør det ikke**, og kan ikke gjøre det: den viser
seg først når Cloudflare svarer. Det er fortsatt `tools/r2_check.py` sitt
ærend, og scriptet kjører nå de lokale sjekkene først, så det ikke bruker en
rundtur på å oppdage noe som var åpenbart.

Ingen sjekk med mulige falske positiver er tatt med. En slik sjekk ville stengt
en fungerende bucket, og det er verre enn tilstanden dette retter.

## Hva mer som endret seg

- `POST /v1/failed-analyses` avviser det feilkonfigurerte tilfellet på samme
  måte som det ukonfigurerte: **503 før kroppen leses**, og en logglinje som
  sier hvilken av dem det var. Klienten får samme svar og samme oppførsel som
  før — den kan ikke gjøre noe med hvordan serveren er satt opp, og skal ikke
  få vite det.
- `kan_brukes()` er definert som «`tilstand()` sier `r2`». To uavhengige
  sjekker på samme spørsmål var måten uenigheten mellom `/health` og mottaket
  oppsto på i utgangspunktet.
- `bucketflytt.kopier()` nekter å starte på et oppsett som umulig kan virke.
  Uten det går jobben i gang og feiler på hvert eneste objekt med et 403 som
  ikke sier hvorfor.
- `objstore.backend_name()` er **slettet**. Den hadde ingen kallere og svarte
  `"database"` — et sted bildene ikke har ligget siden kolonnen ble fjernet.

Verdiene skrives aldri ut, bare navnene: teksten går både i `/health`, som er
åpen, og i loggen, og to av de fire er hemmeligheter.

## Verifisert

- `backend\.venv\Scripts\python.exe -m pytest backend\tests -q` →
  **245 passed, 1 skipped** (den skippede er `test_migrations.py`, som bare
  kjører mot Postgres). 11 av dem er nye.
- De nye testene dekker: at et riktig oppsett gir `None`, at hver av de fem
  feilformene fanges og navngir *hvilken* variabel som er gal uten å røpe
  verdien, at feilkonfigurert R2 gir 503 og ingen rad i basen, at `/health`
  skiller feilkonfigurert fra av, og at halvveis satt ikke leses som av.
- `tools/gen_openapi.py` → `contracts/openapi.json` uendret. `/health` er
  utypet (`-> dict`), så svarets *form* står ikke i kontrakten uansett (ÅP-B10).
- Kjørt for hånd mot de fire faktiske feilformene fra 2026-08-18; utskriften er
  den som står i tabellen over.

## Ikke verifisert

- ~~Ikke deployet~~ **Deployet og lest 2026-08-19.** `Deploy backend` 32216294319
  → `success`, og `curl https://bestefar-api.fly.dev/health` svarer 200 med
  `"bilder":"r2"`. Sjekkene stenger altså ikke den bucketen vi faktisk bruker.
  **Det observasjonen ikke sier:** at det er den nye koden som svarer. Med et
  friskt oppsett svarer gammel og ny kode likt på dette feltet — forskjellen er
  bare synlig når oppsettet *er* galt. Den påstanden hviler på at deployen gikk
  gjennom, ikke på `/health`.
- Ingenting er kjørt mot ekte R2 i denne runden. Sjekkene er lokale med vilje,
  og den delen som krever et ekte kall er uendret.

## Hva som gjenstår rundt R2

ÅP-E11 steg 4 — den gamle bucketen `bestefar-scan-failures` er **ikke tømt**.
`personvernerklaring.txt` kan ikke si at bildene ligger i EU før den er det.
Det er en egen runde og en egen beslutning, og den er din.


---

# UI — v0.29: tidsgrense på auto-capture, og kalibreringsvisningen ut

## 1. Tidsgrensen

Utløser ikke gatingen innen **7 sekunder** fra første gatede frame, tas
gjeldende ramme og analysen kjøres på den. Ingen nedtelling, ingen visuell
forskjell — samme grønne «Klar!»-ramme og samme blits som en vanlig capture.

Armes ved første frame som faktisk gates, ikke i `onCreate`: da spiser verken
kameraets oppstartstid eller en tillatelsesdialog av vinduet. Portrett-frames er
allerede droppet på det punktet, så en timeout-capture er liggende som en vanlig
capture. Vinner gatingen kappløpet, avlyses nedtellingen; feiler selve capturen,
armes den på nytt så brukeren ikke mister tidsgrensen for resten av økten.

## 2. Nei, dette krevde ingen kjerneendring — og omgår ikke tilstandsmaskinen

Du ba meg avgjøre dette selv og stoppe hvis svaret var nei. Svaret er ja, det er
UI alene, og grunnen er verdt en linje:

`bf_analyze` er en **egen FFI-inngang** som tar piksler og et tidsstempel
(`core/include/bestefar/bestefar_ffi.h:62`). Den har aldri gått gjennom
`BfAutoCapture` — `takeStillAndAnalyze` kalte den direkte også før. Gatingen
avgjorde bare *når* den ble kalt. Tidsgrensen er derfor en andre utløser til
nøyaktig samme kall, ikke en vei rundt tilstandsmaskinen: `autocapture.cpp` er
urørt, ingen probe blir løyet om, og ingen terskel er rørt.

## 3. Tag-verdien: backend eier den, og jeg har ikke gjettet

`backend/KONTRAKT.md` §3 er tydelig: «`tag`-enumet er vårt (`models/base.py`,
`FailedTag`). En ukjent verdi gir 422 fra FastAPI-valideringen.» Meldt som
**issue #11**, label `backend`. Ingenting nytt sendes før de har svart.

**Merk at begge de nærliggende snarveiene svikter stille**, som er grunnen til
at jeg ikke tok noen av dem:

- En gjettet `tag`-verdi gir 422. 422 er ikke `retryable`, så køelementet
  droppes — og det ville rammet nøyaktig de donasjonene dette handler om.
- Et ekstra multipart-felt er verre: ukjente skjemafelt ignoreres, serveren
  svarer 201, og feltet forsvinner uten at noen part ser en feil.

I mellomtiden skriver klienten `capture_trigger` ∈ {`auto`, `timeout`} i
sidecar-fila på disk (klientens eget filformat), slik at køen bærer
opplysningen den dagen feltet er avtalt. At det **ikke** krysser grensa står
eksplisitt i `android/KONTRAKT.md` §2 og i en kommentar på selve sendestedet.

**Én ting jeg vil ha ditt syn på, og som ligger i issuet:** du skrev «ny
tag-verdi». Jeg har foreslått et eget felt i stedet, fordi `tag` svarer på *hva
donasjonen viser* (`ocr_match`/`ocr_mismatch`/`rejected`) mens timeout svarer på
*hvordan bildet ble tatt*. De er ortogonale — en timeout-capture kan ende som
hvilken som helst av de tre. En enkelt `timeout`-verdi ville overskrevet
OCR-utfallet, og skal begge bevares blir det `timeout_ocr_match`,
`timeout_ocr_mismatch`, `timeout_rejected`. Backend står fritt til å velge; jeg
bygger mot det som lander.

## 4. Kalibreringsvisningen er fjernet

`debugText` er ute av `activity_capture.xml`. Verdiene logges fortsatt per frame
til logcat (tag `BestefarCapture`, linjene som starter med `probe roi=`), så
ingenting er tapt for feilsøking — det var visningen over kamerabildet som ikke
hørte hjemme i en app folk bruker.

## Verifisert

- `.\gradlew assembleDebug` → **BUILD SUCCESSFUL**, kjørt i denne økten.
- Timeout-stien er **ikke kjørt på enhet**. Se under.

## Ikke verifisert — dette må du teste

- **Selve timeout-capturen er ikke sett virke.** Den enkleste prøven: åpne Scan
  og pek telefonen på noe som *ikke* utløser gatingen (en vegg, et papirark) og
  la den ligge. Etter 7 sekunder skal det komme grønn ramme, blits og en
  resultatskjerm — mest sannsynlig en avvisning, og det er riktig utfall.
- **Den interessante prøven er skjermbildet av en skive**, det som aldri utløste
  capture. Går den nå gjennom analysen og gir poeng, er det svaret ÅP-K1 har
  ventet på — si fra, for da er tersklene beviselig for strenge.
- **7 sekunder er valgt, ikke målt.** Kjennes det langt eller kort på banen, er
  tallet ett tegn å endre (`CaptureActivity.CAPTURE_TIMEOUT_MS`). Det hører
  hjemme i samme kalibrering som tersklene.

## Det tidsgrensen ikke fikser

Den fjerner falske negative fra å være usynlige. Den gjør **ingenting** med
falske positive — vitrineskapet utløser fortsatt gatingen innen 7 sekunder, og
ser i basen ut som en helt vanlig capture. Det er fortsatt ÅP-K1, og det er
fortsatt kjernens.



---

# UI — to rettelser før overlevering (v0.30)

## 1. Tidsgrensen: 7 → 8 sekunder

`CaptureActivity.CAPTURE_TIMEOUT_MS`. Begrunnelsen i KDoc-en er oppdatert til å
si at 7 s ble verifisert for kort på enhet 2026-08-21, ikke bare at tallet er
valgt. Tallet er fulgt opp i dokumentene som beskriver **nåtilstanden**:
`docs/flytskjema.md` og ÅP-K1.

**`android/CHANGELOG.md` er derimot rullet tilbake til å si 7 for v0.29**, og de
to rettelsene er ført som en egen v0.30-seksjon. v0.29 *ble* sendt ut med 7
sekunder og med speilvendingen inne — den APK-en ligger i `dist\`. En
endringslogg som skriver om hva en utsendt versjon inneholdt, er ikke lenger en
endringslogg. Samme grunn til at seksjonen over i denne fila fortsatt sier 7:
`til_utvikler` er historikk, ikke nåtilstand.

## 2. Speilvendt skivevisning — feilen lå i rendereren, ikke i dataene

Du ba meg avgjøre *først* om det var opptegningen eller koordinatene. Det var
opptegningen, og her er prøven.

**Konvensjonen er ikke dokumentert noe sted.** `bestefar_ffi.h:39` sier i sin
helhet `double theta; /* radianer */`. Det kunne altså ikke leses — bare måles.

Jeg kjørte `bestefar_cli Testsett/C1.jpg`, som gir `x, y` (originalbildets
koordinater), `r_rel` og `theta` per treff, og løste med minste kvadrat for
senter og skala med begge fortegn:

| Antatt konvensjon | RMS | k (px per ringsteg) |
|---|---|---|
| y **nedover** (bildekoordinater) | **1,2 px** | **91,9** |
| y oppover (matematisk) | 70,5 px | 58,4 |

Kalibrert `delta_px` for bildet var 94,0. Bare den første stemmer, og den
stemmer godt.

Så `theta = atan2(y_px − cy, x_px − cx)` — samme koordinatsystem som `x_px,
y_px`, som headeren allerede kaller «i inputbildets koordinater». Det er også
nøyaktig hva `scoring.cpp:13` gjør. **Dataene er riktige og internt
konsistente.**

Feilen sto i `Views.kt`, og den hadde til og med skrevet seg selv ned:
KDoc-en sa «theta-konvensjon **antatt** matematisk (x=r·cosθ, y opp)», og
opptegningen speilvendte y for å passe til antakelsen. Siden skjermens y også
peker nedover, er riktig avbildning ren identitet — begge aksene skal ha pluss.
Én linje: `cy - …` → `cy + …`.

**Ingen kjerneendring, og ingenting kompensert i UI.** Hadde jeg snudd fortegnet
her *og* kjernen hadde vært gal, ville bildet blitt riktig og dataene forblitt
feil — og de dataene går til backend som forskningsmateriale.

### Jeg sjekket om samme antakelse fantes andre steder

`Stats.kt` bruker også `cos(theta)`/`sin(theta)`, men er **upåvirket**:
`biasAndSpread` bruker bare vektorens *lengde*, og `biasVector` konsumeres kun
av `similarBias`, som regner differanse og lengder. Speiles begge vektorene
likt, endres ingen av delene. `ResultActivity`s duplikatsjekk regner avstand
mellom to treff og er også fortegns-invariant.

Speilingen var altså isolert til `TargetView` alene.

### Meldt videre: issue #12 (`kjerne`)

Kjernen gjør det riktige, men kontrakten sier ikke hva den gjør, og det er
årsaken til at feilen kunne oppstå. Bedt om én linje i `bestefar_ffi.h` med
konvensjonen skrevet ut. Verdt det fordi feilen er av den snille sorten å se
på: en klient som gjetter matematisk konvensjon får riktig radius, riktig
spredning og riktig gruppestørrelse — bare speilvendt. Det er ingen feilmelding
å lete etter, og den sto i tre versjoner. `theta` går nå også til backend som
forskningsdata, så det er én konsument til som skal tolke den.

## Verifisert

- `.\gradlew assembleDebug` → **BUILD SUCCESSFUL**, kjørt i denne økten.
- Fortegnskonvensjonen: målt mot `Testsett/C1.jpg` som over, ikke antatt.

## Ikke verifisert

- **Skivevisningen er ikke sett på enhet etter rettingen.** Prøven er et treff
  du vet ligger høyt: det skal nå tegnes høyt. Det er verdt ett blikk før
  overlevering, siden det er den ene endringen som er rent visuell.
- 8 sekunder er fortsatt valgt, ikke målt — men nå justert på en observasjon.


---

# backend — issue #11 og #14 besvart, og en sovende feil i utsendingen

## 1. `capture_trigger` som eget felt (issue #11, B-53)

UI-vurderingen holder, og den er nå bygget slik: `capture_trigger` ∈
{`auto`, `timeout`}, valgfritt multipart-felt ved siden av `tag`.

Begrunnelsen er ført i `BESLUTNINGER.md` B-53 så den ikke må gjenoppdages:
`tag` svarer på hva donasjonen *viser*, `capture_trigger` på hvordan bildet ble
*tatt*. En timeout-capture kan ende som hvilken som helst av de tre taggene, så
en `timeout`-verdi i `tag` ville overskrevet OCR-utfallet — og skulle begge
bevares, måtte enumet dobles for hver nye capture-årsak.

**Ett sted valgte jeg annerledes enn forslaget:** feltet har ingen default.
Utelates det, lagres `NULL`, og **`NULL` betyr «klienten sa det ikke» — ikke
`auto`**. Grunnen er v0.29-vinduet: der finnes timeout-capture i klienten, men
feltet ble bevisst ikke sendt. En default på `auto` ville stemplet nettopp de
donasjonene som gatede, og det er disse radene ÅP-K1 skal måles på. Ingen
backfill, av samme grunn — det finnes ikke en riktig verdi å fylle inn.

**Enumet er vårt, med den konsekvensen dere allerede kjenner fra `tag`:** en
ukjent verdi gir 422, og 422 er ikke `retryable`. Nye verdier rulles derfor ut
på serveren *før* klienten sender dem. Det står i `models/base.py` ved siden av
enumet, der noen kommer til å legge til den neste.

Klar til å kobles på. Feltnavnet er `capture_trigger` i multiparten, samme navn
som i sidecar-fila deres.

## 2. «Hvem er jeg i dette laget» (issue #14, B-54)

**To felt, fordi det er to spørsmål** — ett av dem alene løser ikke begge:

| Felt | Hvor | Svarer på |
|---|---|---|
| `my_role` | **alle** lagsvar, også `GET /v1/teams` og `/near` | `"leader"` \| `"member"` \| `null` |
| `members[].public_id` | `GET /v1/teams/{id}` | hvilket element som er meg |

`my_role` ligger med vilje også på **listesvaret**, der `members[]` ikke er med
— dere nevnte at listeskjermen ellers må hente detaljer for å vite hva den skal
tilby. `null` er et normalt svar i `/near`, der lag man står utenfor vises.

**`public_id` og ikke et ekko av `user_id`**, som var deres foretrukne form:
`public_id` er den dere allerede har fra innloggingssvaret, og det er formen
venne-modellen skal bruke (ÅP-U31, «unik per `public_id`»). Bygger dere
gjenkjenningen på den nå, står den seg gjennom steg 4.

**`members[].user_id` står fortsatt, og skal fortsatt brukes til kallene.**
`DELETE …/members/{id}`, `POST …/leaders/{id}` og `candidate_id` i avstemningen
tar imot den interne ID-en. Å bytte `members[]` alene ville gjort lista
ubrukelig som kilde til nettopp de kallene. Hele flaten skal over på
`public_id`, men det er én endring i samme runde som venne-modellen — ikke seks
små — og den varsles.

**Kort sagt:** gjenkjenning på `public_id`, gating på `my_role`, kall med
`user_id`. `has_leader` skal ikke lenger brukes til gating; den sier bare at
laget *har* en leder.

## 3. En sovende feil i `mailer.py`, rettet

`msg["To"]` i SMTP-grenen var hardkodet til utviklerinnboksen, uansett
mottaker. `send_message` tar konvolutt-adressen fra To-hodet, så
innloggingskoder og lag-invitasjoner ville gått til utvikleren i stedet for
brukeren. Resend-grenen ble rettet da `to` ble innført; denne ble stående.

Produksjonen bruker Resend, så feilen har aldri truffet noen — den ville våknet
den dagen SMTP ble tatt i bruk, og da på innloggingskoder.

Ny `tests/test_mailer.py` kjører **begge** leverandørgrenene mot det samme
kravet. Testen er verifisert som en ekte port, ikke bare grønn: jeg satte den
gamle linjen tilbake og så den feile.

## Verifisert

- `backend\.venv\Scripts\python.exe -m pytest backend\tests -q` → se
  commit-meldingen for tallet. 11 nye tester.
- `upgrade head → downgrade base → upgrade head` mot fersk SQLite, siden CI
  ellers prøver downgrade-veien først (mot Postgres).
- `contracts/openapi.json` regenerert i samme commit — `capture_trigger` er i
  multipart-skjemaet.

## Ikke verifisert

- **Ingen ekte donasjon med `capture_trigger` er sett**, og ingen klient sender
  feltet ennå. At det tas imot er testet; at det kommer inn, er det ikke.
- **`my_role` er ikke sett fra klienten.** Testene dekker leder, medlem og
  ikke-medlem, men ingen har tegnet en lagside på svaret.

---

# backend — invitasjonslenken: hvor «item not found» faktisk kom fra (B-55)

## Diagnosen

Feilen var ikke i noen av de tre delene vi mistenkte. Kjeden, sporet med et
**oppdiktet** token:

```
$ curl -sI https://bestefar-api.fly.dev/i/etTokenSomIkkeFinnes
HTTP/1.1 302 Found
location: https://play.google.com/store/apps/details?id=no.bestefar.app&invite=...

$ curl -sI "https://play.google.com/store/apps/details?id=no.bestefar.app"
404
```

«item not found» kommer fra **Google Play**, fordi appen ikke er publisert
(ÅP-E8). At et token som umulig kan finnes ga nøyaktig samme 302, er beviset
for at feilen ikke er tilstandsavhengig: `/i/{token}` leser aldri tokenet og
rører aldri databasen. Det er tilsiktet — §4 sier at svaret skal være likt for
gyldig og ugyldig token — og det er også derfor den var lik før og etter
innlogging.

**Invitasjonen ble lagret.** `routers/teams.py:225–248` legger raden inn *før*
utsendingen og committer etterpå i begge grener, så en e-post som kom fram
betyr en rad med `delivery_status = sent`. Merk at dette er utledet fra koden,
ikke observert i basen — en `select` mot produksjon står igjen som det som
faktisk ville sett raden.

## Den feilen som ville overlevd publisering

Selv med appen installert åpnes lenken i nettleseren og går videre til
butikken. Tre ting manglet, og de har hver sin eier:

| | Eier | Status |
|---|---|---|
| `/.well-known/assetlinks.json` (var 404) | backend | **gjort, B-55** |
| `VIEW`/`BROWSABLE` intent-filter for domenet | UI | issue #15 |
| En kaller for `POST /v1/teams/join` | UI | issue #15 |

Den siste er verdt å merke seg: endepunktet finnes, virker og er dokumentert —
og har aldri hatt en kaller. Klienten kan *sende* invitasjoner, men har ingen
vei fra et token til medlemskap.

## Om fila

Bygges av konfigurasjonen i stedet for å ligge statisk i repoet, fordi flere
avtrykk er normalen: release, debug, og **Play sitt eget signeringsavtrykk hvis
Play App Signing er på**. I det siste tilfellet er avtrykket vi har nå bare
opplastingsnøkkelen, og lenkene ville sluttet å virke i det øyeblikket appen
kom i butikken — en feil som først viser seg hos brukerne.
`ANDROID_CERT_FINGERPRINTS` retter det med én `flyctl secrets set`, uten
utrulling.

Avtrykket står som **standardverdi i koden**, ikke som en påkrevd secret: et
sertifikatavtrykk er offentlig av natur (hele poenget med fila er at Google
skal kunne lese den), mens en tom standardverdi ville gitt en fil som
verifiserer for ingen — uten at noe feiler hos oss.

## Debug-bygg trenger noe eget

`autoVerify` bruker signeringssertifikatet til **det installerte bygget**, så et
debug-bygg verifiserer ikke mot release-avtrykket. Det feiler stille: lenken
åpner nettleseren, uten feilmelding. Enten testes det på release-APK, eller så
legges debug-avtrykket inn ved siden av. Status på enheten:
`adb shell pm get-app-links no.bestefar.app`.

## Verifisert

- `pytest backend\tests -q` → **261 passed, 1 skipped**. 3 nye tester, som
  dekker Googles formkrav: toppnivå som liste, riktig `relation`,
  `Content-Type: application/json`, ingen innlogging, og at flere avtrykk kan
  settes.
- `contracts/openapi.json` uendret — ruta er `include_in_schema=False`, som
  `/i/{token}`.

## Ikke verifisert

- **Ingen App Links-verifisering er kjørt.** Fila har riktig form, men at
  Google faktisk godtar den for dette domenet, viser seg først når klienten har
  et `autoVerify`-filter og en APK er installert. Til da er dette en fil som
  ser riktig ut.
- **Ikke lest utenfra i produksjon** før denne runden er utrullet.
