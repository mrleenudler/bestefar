# Til utvikler — v0.15 (musingsUI runde 12)

## Tilbakemeldingen denne runden

Kopiert fra `musingsUI.txt`:

> **Image donate:** Oppstartsmeldingen er litt flat. Prøv å legge til en stor
> takk/be emoji over teksten. Tror du forresten emojien vil trigge gamle,
> uutdannede, rasistiske vestlendinger? Bør vi velge noe annet?
>
> **Lys visning:** Telefonens statusikoner må ha svart bakgrunn, også i lys
> visning. (Ikke implementert)
>
> **«Bildet ble ikke korrekt analysert»:** `<Scan>` → `<Scan på ny>`. Lag
> knapp/ring rundt `<Send bildet til feilanalyse>`.
>
> **Innsikt:** Legg inn silhuetter for [rådyr]
>
> **Scan-bilde:** (La poengene være midt-aligned med «Poeng:», ikke
> venstre-aligned) Rettelse: La både «Poeng:» og poenglisten være aligned med
> høyresiden av skjermen.
>
> «Ønsker du at skjermbildet(→«Skjermbildene») skal lagres i bildearkivet
> ditt?» Valg: Nei, Alle, De beste
>
> **Avanserte innstillinger:** Når «avanserte innstillinger» blir referert til,
> vis også et slider/equalizer ikon som også kan åpne avanserte innstillinger.
> Gi meg en liste over alle sider der ikonet blir lagt til. Valg om å lagre
> scannede bilder i bildearkivet → Endre til valg: Aldri, Alle, De beste
>
> **Logg jaktskudd:** «Rediger» for skudd: Fjern linjen fra animasjonen, behold
> bare tallet. Når man sletter skudd, legg inn et advarsel-ikon i advarselen. Er
> det flere dialoger som bør ha et advarsel-ikon?
>
> **Soft-delete:** «Jeg er litt usikker på hva dette konkret betyr. Betyr det at
> vi soft-deleter på klientsiden også? Det bør ikke være noe problem, det er
> ikke data som tar noe plass.»
>
> **Backup-blobben:** «Dette må du utdype. Mener du vi skal bygge den nå? Kjør
> på.»
>
> **Serie-synk-kø:** «Hva betyr det?»

---

## Svar på de tre spørsmålene

### Soft-delete — ja, klientsiden er hele poenget

Det er nettopp klientsiden det handler om. I dag fjerner `deleteSeries()` raden
fra `series.json`. Da finnes det ingen forskjell mellom «denne serien har aldri
eksistert» og «brukeren slettet den» — begge deler ser like ut: fravær.

Det er greit så lenge telefonen er eneste kopi. I det backupen slås på, blir det
et datatap-problem med motsatt fortegn: du sletter en serie, tar en
gjenoppretting, og serien er tilbake. Serveren hadde jo en kopi, og ingenting
fortalte den at den var slettet.

Løsningen er en **gravstein**: raden blir stående med `deletedAt` satt.
`allSeries()` og `allHunts()` filtrerer dem bort, så all visningskode er
uendret — men `allSeriesRaw()` gir hele sannheten til synk og backup.

Grunnen til at dette hastet: gjør vi det først når synken bygges, er slettinger
gjort i mellomtiden allerede tapt. Det finnes ingen måte å gjenskape dem i
ettertid. Og du har rett i at plass ikke er et argument — en gravstein er noen
hundre byte.

### Backup-blobben — bygget nå

Ja, den er bygget denne runden (`Backup.kt`). Det som gjorde den byggbar uten
backend er dette: **serveren lagrer bytes den ikke kan lese.** Den ser bare en
ugjennomsiktig binærklump. Alt det som er reelt vanskelig — hva som skal med,
hvordan det serialiseres, kryptering, nøkkelhåndtering, gjenoppretting — er
klientside og kunne bygges og testes i dag.

Det virkelig vanskelige er ikke kryptoen, det er **nøkkelen**. En nøkkel som bare
finnes på telefonen er verdiløs i akkurat det scenarioet kopien eksisterer for:
telefonen er borte. Så nøkkelen må kunne gjenskapes fra noe brukeren har
utenfor telefonen.

Valget landet på en **generert gjenopprettingskode** på 20 tegn (100 bit), vist
én gang med beskjed om å skrive den ned. Nøkkelen utledes med PBKDF2-HMAC-SHA256
(210 000 runder). Brukervalgt passord ble bevisst vraket: en kopi kryptert med
«Bestefar1» gir falsk trygghet, og her finnes ingen server som kan bremse
gjetting — angriperen har hele bloben.

Prisen er ærlig og står i UI-et: **mister du koden, er kopien tapt.** Det følger
direkte av at serveren ikke kan lese bloben. Alternativet ville vært at vi kunne
lese dataene dine, og det er nettopp det §2 ikke vil.

Koden lagres på telefonen etter at den er vist. Ellers måtte brukeren taste 20
tegn ved hver kopi, og da tar ingen kopi. Den beskytter mot at *backenden* kan
lese dataene, ikke mot noen som allerede har telefonen ulåst i hånda — det er
den trusselen §2 faktisk adresserer.

Verifiserbart uten server og uten innlogging: **DevTools → «Test sikkerhetskopi
(rundtur)»** krypterer dagens data, dekrypterer dem igjen, sammenligner, og
sjekker at feil kode blir avvist. Den rører aldri dataene dine.

### Serie-synk-kø — hva det betyr

`PUT /v1/stats/series/{id}` er endepunktet som legger én treningsserie inn på
kontoen din. To ting gjør det trygt å køe:

**Serie-ID-en er klientens egen UUID.** Serveren lager ikke sin egen. Sender du
samme serie to ganger, skriver den over seg selv i stedet for å bli to serier.
Det kalles idempotent, og det er det som gjør en kø mulig i det hele tatt —
telefonen kan sende i blinde uten å vite om forrige forsøk kom fram.

**Køen kan bygges før innloggingen finnes.** `SeriesRecord.uploaded` står
allerede i modellen og har aldri blitt satt til noe. Køen ville vært: marker
usendte serier, send dem når nettet finnes, sett `uploaded = true`. Nøyaktig
samme mønster som `Sync.kt` bruker for feilanalysebildene.

Den er **ikke** bygget denne runden — backupen dekker det samme behovet (dataene
dine overlever telefonen), og to parallelle synkveier mot samme data bør ikke
bygges før vi vet hvilken av dem som skal eie sannheten. Det er en avklaring mot
backend-instansen, ikke noe jeg bør avgjøre alene.

### Emojien

🙏 er lagt inn. På spørsmålet ditt: det rasistiske sporet holder ikke — emojien
har ingen hudtone som standard (den er gul), og den kommer fra japansk
«vær så snill / takk», ikke fra noen kulturkrets folk har meninger om.

Den reelle innvendingen er en annen: mange leser 🙏 som **bønn**, og det er en
litt underlig tone i en jaktapp. Emojien ligger derfor i sin egen streng,
`startup_donate_emoji`, så den byttes uten kodeendring. Alternativer som fungerer
like godt: 📷 (det er faktisk bilder vi ber om), ❤️, eller 🎯. Si fra, så bytter
jeg — det er én linje.

---

## Sider der equalizer-ikonet er lagt til

Du ba om lista. Ikonet er `ic_settings_sliders`, og det er lagt inn tre steder —
alle stedene «Avanserte innstillinger» faktisk nevnes i appen i dag:

| Sted | Form | Klikkbart |
|---|---|---|
| **Min profil** — knappen «Avanserte innstillinger» | ikon foran teksten på knappen | ja (knappen gikk dit fra før) |
| **Scan-resultat** — dialogen «Du kan endre dette valget i «Avanserte innstillinger»» | ikon til venstre for teksten | **ja** — ikonet åpner siden |
| **Avanserte innstillinger** — sidens egen tittel | ikon foran overskriften | nei (vi er allerede framme) |

Det siste er med vilje: symbolet må knyttes til siden det fører til, ellers er
det bare et ikon. `Ui.advancedIcon()` er hjelperen — nye henvisninger til siden
skal bruke den, så listen holder seg fullstendig.

---

## Advarselsikon — hvilke dialoger

Ja, det var flere. Regelen jeg har brukt: ikonet er for valg der **noe forsvinner
og ikke kan hentes tilbake**. Bekreftelser som bare er et veivalg får det ikke —
ellers slites ikonet ut og slutter å bety noe.

Fikk ikon (`Ui.warningDialog()`):

- **Slett skudd** i Registrerte skudd (`delete_all_confirm`) — det du ba om
- **Slett alle data** i Avanserte innstillinger — den farligste i hele appen
- **Slett serier** i Serieloggen
- **Innskyting: «Ikke lagre»** — den sletter også dagens *første* serie, ikke
  bare den du står i. Det var ikke tydelig fra teksten alene.
- **Overskriv nyere sikkerhetskopi** (ny, se under)

Fikk det bevisst **ikke**: «lik serie — lagre likevel?» (ingenting går tapt),
og alle samtykke-/delingsdialogene.

---

## Hva som ellers ble gjort

### Svart statuslinje i lys visning — rotårsaken var targetSdk 36

Koden fantes fra runde 10 og var riktig da. Den sluttet å virke fordi
`targetSdk` gikk til 36: **Android ignorerer `windowOptOutEdgeToEdgeEnforcement`
for apper som targeter SDK 36.** Da tvinges edge-to-edge, `statusBarColor` blir
en no-op, og systemlinjene tegnes rett oppå appbakgrunnen. I mørk visning merkes
det ikke — bakgrunnen er nesten svart fra før. I lys visning står de hvite
systemikonene på lys brunt og blir uleselige. Nøyaktig det du så.

Dette var forutsagt i `musingsUI.txt`s egen TODO-liste («da må vi tegne baren
selv»). Så nå gjør vi det: `Ui.paintSystemBars()` legger en svart flate bak hvert
systeminnsett. Registrert én gang i `BestefarApp` via `ActivityLifecycleCallbacks`
framfor i tolv aktiviteter — en ny aktivitet som glemte kallet ville fått
uleselige ikoner i lys visning uten at noen oppdaget det før i felt.

På enheter der opt-out-en fortsatt virker er innsettene 0, og flatene blir 0
høye. Da er dette en no-op i stedet for dobbelt-tegning.

### Rådyr-silhuetter

Front, skrå og side er portert. `Species.RAADYR` falt tidligere gjennom til
`else ->` og tegnet **hjort** — feil art, og den arten der forveksling faktisk
betyr noe: dødelig sone på rådyr er under halvparten så stor.

En felle verdt å kjenne: sidevisningen din er **potrace-eksport**, ikke samme
format som de andre. Den maler et hvitt bakgrunnsrektangel og tegner hull som
hvite paths oppå. Vi tinter drawablene i UI-koden, så «hvit» finnes ikke som
farge der — males alt svart, blir ikonet en solid klump. `_convert_species_svgs.py`
oppdager nå varianten, kaster bakgrunnsrektangelet og slår resten sammen til én
path med `evenOdd`, som er nettopp hull-semantikken potrace koder med.

**Sjekk denne på telefonen.** Hullogikken er verifisert på bounding-bokser (ett
lite hull på 30×88 px, som forventet), men om beina overlapper kroppen kan
`evenOdd` i teorien lage et hull som ikke skal være der. Det ser man bare.

### «De beste» — definisjonen

Valget kan ikke avgjøres ved fangst: poengene finnes først etter analysen.
Derfor lagrer vi alltid til galleriet når valget ikke er «Aldri», og
`ResultActivity` rydder bort bildet igjen hvis serien ikke kvalifiserer.

Definisjonen er valgt så den kan forklares i én setning i innstillingene:
**blant de 25 % beste i samme stilling, eller beste serie noensinne.** Stilling
er med fordi en liggende serie ellers ville utkonkurrert alt stående for alltid.
Med under fire serier i stillingen kvalifiserer alt — det er riktig; da er alle
bildene de beste vi har.

Den gamle av/på-bryteren migreres: på → «Alle», av → «Aldri».

### Rediger-animasjonen

«Linjen» var EditText-ens egen understrek. Den er en del av *viewet*, så den ble
skalert og flyttet sammen med tallet — det så ut som en strek som fløy avgårde.
Bakgrunnen fjernes nå mens animasjonen står på og settes tilbake etterpå.

Én detalj som måtte med: `animate().cancel()` kaller **aldri** `withEndAction`.
Uten at understreken også gjenopprettes i avbrudds-stiene, ville feltet mistet
den for godt hvis brukeren klikket raskt.

---

## Verifisert

- `compileDebugKotlin` grønt.
- Release-bygg OK, `dist\Bestefar-0.15.apk`.
- Blobformatet: rundtur i minnet (DevTools), inkludert at feil kode avvises.

**Ikke verifisert (krever telefon):** svart statuslinje i lys visning, rådyr-
silhuettene, høyrejustert poengliste, animasjonen uten understrek.

**Ikke verifisert (krever innlogging):** opp-/nedlasting av sikkerhetskopien.
`/v1/backup` svarer 401 til klienten har et token. Selve blobben er ferdig; det
som mangler er `Store.authToken`, som transportlaget allerede leser.

---

## Til deg

**Fase 3 landet mens vi snakket.** Backend-instansen committet innlogging
(Google, Apple, e-postkode) i `7363879` i dag. Det jeg skrev tidligere i dag om
at «alt som krever bruker svarer 501» stemmer ikke lenger — det er 401 nå, altså
«logg inn», ikke «finnes ikke».

Det som gjenstår hos deg, fra backend-instansens egen liste:

- `GOOGLE_CLIENT_IDS` (Google Cloud Console)
- `APPLE_CLIENT_IDS` (Apple-utviklerkonto)
- `JWT_SECRET` som Fly-secret
- En ekte e-postleverandør (`RESEND_API_KEY`/`SMTP`) før e-postkoder virker

Uten disse kan ingen logge inn, og da er sikkerhetskopien, serie-synken, venner
og lag fortsatt utilgjengelige uansett hvor ferdig koden er på begge sider.

**Åpent fra forrige runde, fortsatt ubesvart:** `bf_version()` i
`bestefar_ffi.h`. Backend-instansen har svart ja og ført den i `backend_spec.md`
§8; det er en ren kjerne-endring og krever ingenting av backenden.
