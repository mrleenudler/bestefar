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

