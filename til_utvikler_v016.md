# Til utvikler — v0.16 (musingsUI runde 13)

## Tilbakemeldingen denne runden

Kopiert fra `musingsUI.txt`:

> **Warning ikon:** La ikonet for «Slett alle data» være et STOP ikon, ikke warning.
>
> **Innskyting:** «Ikke lagre» — den sletter også dagens første serie, ikke bare
> den du står i. Det var ikke tydelig fra teksten alene. *Kan du beskrive litt
> nærmere hva som skjer her?*
>
> **To ting klienten MÅ håndtere**
> `/logout` gjør ikke access-tokenet ugyldig. Verifisert mot produksjon: etter
> 204 fra `/logout` svarer `GET /v1/profile` fortsatt 200 med det samme tokenet.
> Ikke en feil — tokenet er en statsløs JWT og kan ikke tilbakekalles, derfor er
> levetiden bare 60 minutter. Klienten må selv slette begge tokenene, og kalle
> `POST /v1/devices/unregister` samtidig. Ellers har en utlogget bruker full
> tilgang i opptil en time og fortsetter å få varsler.
>
> Ikke kjør to `/refresh` parallelt. Refresh-tokenet roteres ved hver bruk, og
> et allerede brukt token som dukker opp igjen tolkes som en lekkasje — da
> tilbakekalles alle brukerens økter. To samtidige kall logger altså ut
> brukeren. Serialiser fornyelsen bak én lås.
>
> **Backend Fase 7:** Det bør være mulig å lagre skadedata for forskning. Det er
> sentral informasjon.
>
> **Push varsel felling:** Backend anbefaler «Vis teksten for brukeren før dere
> sender, så hen ser hva vennene får.» Eks: Sender «Ola har felt et villsvin i
> Molde»
>
> **Trendvisning:** Jeg vil ikke ha den som at avvik fra snittet, men som en
> graf som viser hvor skytteren ligger nå. Førsteaksen skal være tid (dato) opp
> til 2 år. Når tredje sesong/jaktår startes, fjernes det første fra grafen.
> Andreaksen er 20-skudds rolling average. Avpass aksen slik at avstanden fra
> laveste datapunkt til sentrum aldri blir mindre enn 25 % av andreaksen.
> *Nå ser jeg også at 20-skudds rolling average ikke går sammen med visningen
> min. La oss bruke gjennomsnittet for dagen, dersom det skytes mer enn 20 skudd
> på en dag. Hva er beste løsning for siste økt? Fremskrive trenden og justere
> når neste serie kommer?*
>
> **Sikkerhetskopi — nøkkelforvaltning** (erstatter dagens 20-tegns kode som
> primærmekanisme): nøkkelen i Block Store med `shouldBackUpToCloud=true` når
> `isEndToEndEncryptionAvailable()`; koden degraderes til nødutgang; nytt valg
> «Sikkerhetskopi: Gjenopprett uten kode»; nytt valg «Sikkerhet: Krev opplåsing
> for jaktloggen»; brukeren forblir permanent innlogget, refresh serialiseres,
> tokens i EncryptedSharedPreferences.

---

## Svar på spørsmålene

### «Kan du beskrive litt nærmere hva som skjer» — innskyting

Kort: dialogen dukker opp når appen tror du **skyter inn** våpenet, og
alternativet «Nei, ikke lagre» kaster **to** serier, ikke én.

Hele mekanikken, i rekkefølge (`ResultActivity.sightInCheck`):

1. Etter hver scan sjekker `Stats.looksMiscalibrated(shots)` om treffbildet ser
   ut som et uinnskutt våpen. Kriteriet er skalafritt: **gruppas senter** må
   ligge mer enn 1,5 ganger **spredningen** unna siktepunktet, og mer enn to
   ringavstander. En tett gruppe langt oppe til venstre slår ut; en spredt
   gruppe rundt blinken gjør det ikke. Det er nettopp forskjellen på «siktet er
   feil» og «jeg skjøt dårlig».
2. Er dette **sesongens første serie med dette våpenet**, spør vi rett ut: «Er
   våpenet innskutt?»
   - *Ja* → serien lagres normalt.
   - *Nei* → serien forkastes med én gang. Den er innskyting, ikke trening, og
     den ville trukket ned statistikken din for en feil du allerede har rettet.
3. Er dette **dagens andre serie**, og dagens første hadde *samme skjevhet*
   (`Stats.similarBias` — samme retning og størrelse på bias), spør vi igjen. Da
   er det ikke tilfeldig; du har skutt to serier med det samme feiljusterte
   siktet.
   - *Ja, den er innskutt* → begge beholdes.
   - *Nei* → **her kommer dialogen du spør om.** «Nei, ikke lagre» sletter
     dagens første serie *og* lar være å lagre den du står i. Begge var skutt
     med det samme feiljusterte siktet, så å beholde den ene ville lagret
     akkurat den halvparten av innskytingen som tilfeldigvis kom først.

Det var *logikken* som var riktig og *teksten* som var mangelfull. Runde 12 ga
dialogen et advarselsikon; runde 13 gir den en tekst som sier hva som forsvinner:

> Du svarte at våpenet ikke er innskutt. Da er begge dagens serier skutt med et
> våpen som ikke traff der du siktet.
>
> «Nei, ikke lagre» sletter BEGGE — denne serien og dagens første. De teller da
> ikke med i statistikken din.

Én ting til, verdt å vite: siden runde 12 er sletting **soft-delete**. Serien
får en gravstein i stedet for å forsvinne, så den kan hentes tilbake fra en
sikkerhetskopi hvis det skulle vise seg å være feil. Den er borte fra
statistikken, ikke fra disken.

### «Hva er beste løsning for siste økt?» — nei, ikke framskriv

Framskriving er den ene tingen grafen ikke bør gjøre. Et beregnet punkt kan ikke
skilles fra et målt punkt, og det står akkurat der brukeren ser nøyest etter —
høyre kant, «hvor ligger jeg nå». Skulle framskrivningen bomme, oppdager man det
først når linja *hopper* ved neste serie, og da har grafen løyet én gang og
rettet seg selv uten å si fra. Det er verre enn å vise litt mindre.

Løsningen som er bygget viser i stedet **to ting samtidig**, uten å finne på noe:

- **Linja** er det rullende snittet. Den svarer på *hvilken form er jeg i* og
  skal med vilje bevege seg tregt — det er hele grunnen til et vindu på 20 skudd.
- **En svak prikk per dag** er dagens *eget* snitt, uansett hvor mange skudd det
  var. Den svarer på *hva skjøt jeg i dag*. Ligger prikken godt over linja, ser
  du at det går oppover før snittet rekker å flytte seg.
- **Siste punkt tegnes åpent** (hul ring) så lenge vinduet ikke er fullt, med
  antall skudd i teksten under: «foreløpig, basert på 12 skudd». Da vet du at
  tallet fortsatt beveger seg — og det gjør det uten at vi har gjettet.

Så: ingen framskriving, ingen etterjustering. Det som er målt, står; det som er
usikkert, ser usikkert ut.

### Dagsnitt kontra rullende vindu

Regelen er implementert nøyaktig som du beskrev, og den henger sammen:

- Ett punkt per **dag** det er skutt (ikke per serie — serier er ujevne).
- Verdien er snittet av de **siste 20 skuddene** fram til og med den dagen,
  på tvers av dager.
- **Unntak:** har dagen alene mer enn 20 skudd, er dagen sitt eget vindu, og
  punktet er dagens snitt. Ellers ville en lang treningsdag delt seg over flere
  punkter som alle viste den samme dagen.

Vinduet telles i **skudd**, ikke serier. En serie er 5–10 skudd, så «fem serier»
ville betydd 25 skudd for én bruker og 50 for en annen — det gjorde tallet
uleselig på tvers av brukere (samme lærdom som backend-runde 10 gjorde).

### Y-aksen: «laveste punkt minst 25 % under sentrum»

Regnet ut, med `lo = min − a` og `hi = maks + b` og spenn `R = maks − min`:

```
(R + b − a)/2  ≥  (R + a + b)/4     ⟺     a ≤ (R + b)/3
```

Altså: **luften under kurven er begrenset til en tredjedel av (spenn + luft
over)**. I praksis betyr det at det laveste punktet aldri havner høyere enn en
firedel opp på aksen. Det er den riktige regelen å ha, for det er nettopp mye
tom plass under kurven som får en flat utvikling til å se ut som framgang.

Kantfallet er verdt å nevne: med ett eneste datapunkt er `R = 0`, og regelen
alene ville tvunget punktet ned på gulvet. `Stats.trendAxis` har derfor en
minste aksehøyde, og da lander det ene punktet nøyaktig på 25 %-linja — som er
konsistent med regelen, ikke et unntak fra den.

### Hjelpeteksten om deponering — du hadde misforstått én ting

Du skrev at dataene «ikke er krypterte på serveren». De *er* fortsatt kryptert,
også med valget på. Forskjellen er at **vi holder nøkkelen**, og dermed kan låse
dem opp. Praktisk er konsekvensen din helt riktig — den som kommer seg inn hos
oss, kommer til innholdet — men formuleringen ville gitt en teknisk kyndig
bruker inntrykk av at vi lagrer klartekst, og det gjør vi ikke. Teksten som
ligger inne nå:

> Slår du dette på, kan du hente dataene dine tilbake selv om du mister både
> telefonen og gjenopprettingskoden.
>
> MEN: da oppbevarer vi nøkkelen til kopien din. Kopien er fortsatt kryptert,
> men vi kan låse den opp — og det kan også noen som bryter seg inn hos oss.
>
> Spørsmålet du bør stille deg er: «Hvor ille er det for meg om andre kan lese
> alt jeg har i appen?»

Spørsmålet ditt til slutt er beholdt nesten ordrett. Det er den beste setningen
i hele dialogen, fordi den flytter valget fra teknologi til noe brukeren faktisk
kan svare på.

---

## Hva som ble gjort

### STOP-ikon (`ic_stop.xml`, `Ui.stopDialog`)

En åttekant med tverrstrek, tintet i samme røde som advarselstrekanten.
`Ui.stopDialog()` brukes **ett** sted: «Slett alle data». Skillet mellom trekant
og åttekant er verdiløst i det øyeblikket åttekanten dukker opp to steder, så
den er reservert for det ene valget som fjerner alt på én gang.

De fem dialogene fra runde 12 beholder trekanten.

### Ekte utlogging og serialisert fornyelse (`Auth.kt`, `Secrets.kt`)

Begge punktene du meldte er implementert.

**Utlogging** (`Auth.logout`) gjør tre ting i denne rekkefølgen, og rekkefølgen
er poenget:

1. `POST /v1/devices/unregister` — *mens tokenet fortsatt virker*. Gjøres det
   etterpå, er det for sent.
2. `POST /v1/auth/logout` — tilbakekaller refresh-tokenet.
3. Sletter begge tokenene lokalt — **uansett**, også når nettet er nede. Alt
   annet ville latt «logg ut» feile stille og etterlatt et gyldig access-token
   på telefonen i opptil en time.

**Fornyelse** (`Auth.refresh`) er `@Synchronized` og eneste vei inn. En tråd som
har ventet på låsen, sjekker om tokenet allerede er fornyet og gjør da
ingenting. `Api` prøver på nytt etter 401 **nøyaktig én gang**; auth-kallene selv
går med `authRetry = false`, ellers ville et 401 fra `/refresh` utløst en ny
fornyelse i ring.

**Om `EncryptedSharedPreferences`:** du foreslo den, og jeg gikk en annen vei.
`androidx.security:security-crypto` er avviklet av Google — å ta inn et avviklet
bibliotek for seksti linjer kode er gjeld vi må betale igjen senere. `Secrets.kt`
bruker det biblioteket selv sto på: én AES-256-GCM-nøkkel i Android Keystore,
chiffertekst i base64. Samme beskyttelse, ingen avhengighet.

At det ligger i en **egen prefs-fil** er ikke en detalj: `Store.exportPrefs()` er
generisk over hele `bestefar_ui`, så et token lagret der ville havnet inne i
sikkerhetskopien. `authToken`, `authExpiresAt` og `pushToken` er dessuten
eksplisitt utelatt fra kopien — de hører til telefonen, ikke til dataene — og
`importPrefs` tar vare på dem gjennom `clear()`, så en gjenoppretting ikke logger
deg ut midt i.

Gjenopprettingskoden er flyttet fra `bestefar_ui` til `Secrets` av samme grunn:
en sikkerhetskopi som inneholder sin egen nøkkel er en sirkel vi ikke skal tegne.

### Forhåndsvisning før felling-pushen (`Announce.kt`)

Backendens anbefaling er fulgt til punkt og prikke. Etter en vellykket felling
får du en dialog med den **nøyaktige** setningen vennene dine får:

> «Ola har felt et villsvin i Molde.»

Stedet står i et redigerbart felt (forhåndsutfylt fra jaktloggens stedsnavn) og
kan tømmes — da blir det «Ola har felt et villsvin.» Teksten oppdateres mens du
skriver, så det du ser er alltid det som sendes.

Tre valg som er verdt å begrunne:

- **Bøyningen kommer fra klienten.** Serveren limer sammen «{navn} har felt
  {art}», så «et rådyr» kontra «en elg» må ligge her (`Announce.speciesPhrase`).
- **Kunngjøringen kommer etter lagringen** og kan aldri stoppe den. Avbryter du
  dialogen, er jaktposten fortsatt lagret.
- **Bare vellykkede fellinger tilbys.** Bom og ettersøk kunngjøres ikke — det er
  ikke en gladmelding, og det er ikke vår sak å kringkaste.

Kvitteringen sier hvor mange **enheter** som fikk varselet, ikke hvor mange
venner. Serveren teller enheter, og vi lover ikke brukeren mer enn det som
faktisk skjedde.

### Trendgrafen (`Stats.trendPoints`, `Stats.trendAxis`, `TrendView.kt`)

Ligger øverst på **Serier**-siden. Den viser alltid to jaktår, uavhengig av
«Denne sesongen / Alle»-filteret under — en trend som stopper 1. april er ikke
en trend.

- **X:** dato. Sesongskiftet 1. april tegnes som en loddrett strek med
  sesongetikett, så et fall over sommeren ikke leses som en plutselig
  forverring.
- **Y:** poeng per skudd, med hjelpelinjer på halve poeng — det er enheten
  skytteren tenker i.
- Vindusfiltreringen er på **sesongnøkkel**, ikke «730 dager siden», slik at
  grafen bytter innhold på samme dato som resten av appen bytter jaktår.

### Nøkkelforvaltning (`BackupKeys.kt`)

Tre veier inn til den samme nøkkelen, i prioritert rekkefølge:

1. **Lokalt** (`Secrets`) — har vi den, er vi ferdige.
2. **Block Store** — men bare når `isEndToEndEncryptionAvailable()` er sann.
   Er den ikke det, lagrer vi **ingenting** der: en nøkkel som ligger lesbar hos
   en tredjepart er dårligere enn ingen nøkkel der. `setShouldBackupToCloud(true)`
   er satt, for det er telefonen som *aldri kommer tilbake* som er scenarioet.
3. **Deponering hos serveren** — kun når brukeren har slått på «Gjenopprett uten
   kode».

Finner ingen av dem noe, og bare da, spør vi etter koden.

Praktisk følge: «Sikkerhetskopier nå» spør ikke lenger om noe. Koden lages,
lagres i Block Store og deponeres (hvis valgt) før bloben lastes opp — en kopi på
serveren uten en nøkkel noe sted er det eneste utfallet som er verre enn ingen
kopi. Koden vises nå bare når brukeren ber om den, under Avanserte innstillinger
→ Sikkerhetskopi → «Vis gjenopprettingskode».

**Backend-instansen bygget dette samtidig, uavhengig — og vi landet på samme
design.** Deres endepunkt er `PUT/GET/DELETE /v1/backup/key-escrow` med
`{"key_material": "<base64>"}`, materialet kryptert i ro med en Fly-secret som
ikke ligger i databasen. Klienten er rettet til nettopp den kontrakten (koden
sendes som base64 av ASCII). Begge sider kom uavhengig fram til at UI-teksten må
si rett ut hva deponering betyr, noe jeg tar som et tegn på at det er riktig.

Svarer serveren **503**, er deponering ikke slått på der. Da sier klienten
«Serveren tar ikke imot nøkler nå. Bruk gjenopprettingskoden.» og bryteren går
tilbake til av. Den skal aldri stå på og late som nøkkelen ligger trygt hos oss.

### Opplåsing foran jaktloggen (`Lock.kt`)

Bryteren «Sikkerhet: krev opplåsing for jaktloggen», av som standard.

- **Biometri ELLER skjermlås** (`BIOMETRIC_WEAK or DEVICE_CREDENTIAL`). Vi
  krever ikke fingeravtrykk; PIN er nok.
- Gate på **begge** inngangene til Jakt: «Registrer jaktskudd» og «Se registrerte
  skudd». Låsen kommer før samtykkedialogen — den viser ingen jaktdata, men skal
  heller ikke kunne brukes til å bekrefte at det *finnes* en jaktlogg her.
- **Avvist opplåsing gjør ingenting.** Du blir stående der du sto. Appen lukkes
  ikke, og du kastes ikke ut av skjermen du kom fra.
- **Fem minutters frist**, i prosessminnet. En ny appstart spør på nytt. Uten
  fristen ville en tur ut og inn i loggen under samme jakt bedt om fingeravtrykk
  hver gang — og da slår brukeren av hele funksjonen.
- **Bryteren skjules** når `canAuthenticate()` ikke er `SUCCESS`, med et hint om
  at telefonen må ha skjermlås. En bryter som ikke kan virke, skal ikke stå der
  og se ut som en mulighet.
- Resten av appen er **aldri** gatet. Scan-flyten skal virke med hansker.

Og hjelpeteksten sier hva dette faktisk er:

> Merk: dette er en dør foran skjermen, ikke kryptering. Den beskytter mot en
> ulåst telefon i feil hender.

Jaktloggen ligger like lesbar på disk som før. Å påstå noe annet ville vært å
selge en trygghet vi ikke leverer.

### Skadedata for forskning — sendt videre, ikke bygget

Dette er backend-siden sitt (`research_filter.py`), og jeg har lagt inn kravet i
`backend_spec.md` §7 framfor å røre `backend/`. Din begrunnelse er den faglig
riktige: et materiale som bare inneholder de vellykkede fellingene kan ikke si
noe om skadeskyting, og det er nettopp det ettersøksforskningen finnes for.

Kravet slik det er formulert til backend-instansen:

1. **Egen bryter** (`share_wound_data`), **av som standard**. Skadedata er det
   mest sensitive i loggen — det er der en jeger kan bli hengt ut — så det skal
   aldri følge med på kjøpet av et annet ja.
2. `outcome`, `follow_up` og `ran_m` på tillatelseslista, gatet på den bryteren
   alene.
3. Klient-UI når `/v1/research/sharing` faktisk kan lagre valget. Forskningen er
   fortsatt sperret av `Dialogs.RESEARCH_ENABLED`, og personvernerklæring +
   DPIA-vurdering gjelder like fullt — et *mer* sensitivt felt gjør ikke den
   vurderingen mindre nødvendig.

---

## Nye avhengigheter

| Bibliotek | Hvorfor |
|---|---|
| `androidx.biometric:biometric:1.1.0` | Opplåsing med biometri **eller** skjermlås, med fallback helt ned til minSdk 26 |
| `com.google.android.gms:play-services-auth-blockstore:16.4.0` | Nøkkelen overlever telefonbytte uten at brukeren taster en kode |

`androidx.security:security-crypto` ble **ikke** tatt inn — se begrunnelsen over.

---

## Verifisert

- `compileDebugKotlin` grønt.
- Release-bygg OK, kopiert til `dist\Bestefar-0.16.apk`.

## Ikke verifisert (krever enhet)

- Block Store krever Play-tjenester og skjermlås. `e2eAvailable()` returnerer
  `false` i emulator uten Play, og da faller vi tilbake på koden — det er den
  tiltenkte oppførselen, men *at* den er tiltenkt er ikke det samme som at den
  er sett virke.
- `BiometricPrompt` med `DEVICE_CREDENTIAL` bør prøves på en enhet med **bare**
  PIN (ikke fingeravtrykk), siden det er den stien androidx emulerer for eldre
  API-nivåer.
- Trendgrafen er tegnet mot genererte serier. Med ekte data over to jaktår er
  det verdt å se om sesongstreken og etiketten kolliderer med kurven.

## Fortsatt åpent

- **`bf_version()` i `bestefar_ffi.h`.** Fortsatt ubesvart fra v0.14 og v0.15.
  `core_version` i donasjonene er appens `versionName`, som er feil så snart
  kjernen og appen versjoneres hver for seg. Ren kjerne-endring; begge
  instanser anbefaler den.
- **Innlogging finnes ikke i klienten ennå.** `Auth.kt` er komplett på
  økt-siden, men ingenting *starter* en økt: det er ingen «Logg inn»-knapp, og
  ingen Google/Apple/e-post-flyt. Alt som krever konto — sikkerhetskopi til
  server, felling-kunngjøring, deponering — svarer derfor 401 i dag. Neste
  runde bør bygge inngangen, og da med Credential Manager
  (`androidx.credentials` + `googleid`); den gamle Google Sign-In-SDK-en er
  utfaset. **Merk for backend:** `aud` blir da **web**-klient-ID-en som sendes
  til `setServerClientId(...)`, ikke Android-klient-ID-en, så det er den som må
  inn i `GOOGLE_CLIENT_IDS`.
- **Serie-synk-køen** (`/v1/stats`) er fortsatt ikke bygget, av samme grunn som i
  runde 12: det er ikke avgjort om bloben eller `/v1/stats` eier sannheten.
- **FCM er ikke koblet inn i klienten.** `google-services.json` ligger i
  `android/app/`, men appen registrerer ingen enhet og mottar ingen varsler ennå.
  `Store.pushToken` finnes og brukes av utloggingen, men står tom til den dagen.

---

# Backend — runde 11 (notatene i musings_backend)

Fire spørsmål besvart, tre ting bygget. **182 tester grønne** mot SQLite og
Postgres.

## Svar på spørsmålene

### «Hva gjør `/start`?»

`POST /v1/auth/email/start` er **steg 1 av 2** i e-postinnlogging. Kroppen er
`{"email": "..."}`. Den lager en sekssifret engangskode, lagrer *hashen* av
den, og sender koden på e-post. Den svarer **alltid 202**, også for en adresse
som ikke finnes fra før — skilte den mellom kjent og ukjent, ville endepunktet
vært et oppslagsverk over hvem som bruker appen.

Steg 2 er `POST /v1/auth/email/verify` med `{"email", "code"}`, som gir
tokenparet. Konto opprettes der, ikke i `/start`.

Svaret fra `/start` er nå utvidet:

```json
{"status": "sendt", "resend_after_seconds": 60, "expires_in_minutes": 15}
```

### «Ble timer for *send ny kode* implementert?»

Nei — det fantes bare en kvote på 10 koder per adresse per time. **Nå er den
der.** Sperrefristen er 60 sekunder og ligger på serveren:

- Innen fristen svarer `/start` **429** med `Retry-After`-header, og **ingen
  e-post sendes**.
- `resend_after_seconds` i 202-svaret er tallet klienten skal telle ned på.
  Les det derfra i stedet for å hardkode 60 — da følger nedtellingen med hvis
  verdien endres på serveren.

Grunnen til at den ligger begge steder: en nedtelling i klienten er
bekvemmelighet, ikke et vern. En modifisert klient kan klikke den bort, og
resultatet ville vært at hvem som helst kan sende ti e-poster i slengen til en
fremmed adresse. E-poster kan ikke tas tilbake.

### «Blir forskningsdataene slettet, eller bare flagget som slettet?»

**Ingen av delene, i dag.** Det ærlige svaret er at
`research_deletion_requested: true` betyr at det er lagt inn en *anmodning* i
`research.deletion_requests` — en rad med pseudonym-ID og tidspunkt. Selve
radene i `research.records` står urørt til noen kjører en jobb som sletter dem
og kvitterer i `completed_at`. **Den jobben finnes ikke.**

Slik må det være så lenge §7 gjelder: brukerskjemaet vet hvem kontoen er,
forskningsskjemaet vet bare pseudonymet, og ingenting får koble dem sammen på
tvers. Kontosletting kan derfor ikke selv gå inn og slette forskningsradene —
det ville krevd nettopp den koblingen paragrafen forbyr.

Konsekvensen er en **driftsforpliktelse, ikke en kodeoppgave**: en rutine som
tømmer køen. Til den finnes, er svaret til brukeren om at dataene fjernes ikke
helt sant ennå. Det ligger under «venter på utvikler» nedenfor, og det bør
være på plass før `RESEARCH_ENABLED` slås på.

### Trendgrafen

Ja — **grafen er frontendens jobb.** Tidsakse på to sesonger, 20-skudds
rullende snitt, aksen skalert så avstanden fra laveste punkt til sentrum aldri
er under 25 % — alt det er visning over data klienten allerede har lokalt, og
den bør ikke gå via serveren for å tegne sin egen logg.

To ting fra backendsiden:

1. `trend`-feltet i venneprofilen er noe **annet** og beholdes som det er:
   siste 20 skudd minus de 20 foregående — én tallverdi som sier *retning*,
   ikke nivå. Den brukes i vennelista, ikke i grafen. Skal en venn kunne se
   grafen din, er det et nytt delingsvalg og et nytt endepunkt — si fra.
2. Om «snittet for dagen når det skytes mer enn 20 skudd»: da er det ikke
   lenger et rullende snitt, men dagspunkter. Begge deler er forsvarlige, men
   de gir ulike kurver — det rullende snittet er jevnere og reagerer seinere.
   Forslaget mitt: **behold rullende snitt over 20 skudd, og plott punktet på
   datoen til det siste skuddet i vinduet.** Da trengs ingen spesialregel for
   dager med mange skudd, og ingen framskriving for den siste økta — det siste
   punktet *er* status nå, og det flytter seg av seg selv når neste serie
   kommer. Framskriving av en trend er dessuten en påstand om framtida som
   appen ikke har grunnlag for å gjøre.

## Bygget denne runden

### 1. Nøkkeldeponering — `/v1/backup/key-escrow` (§2.1)

| Endepunkt | Inn | Ut |
|---|---|---|
| `PUT /v1/backup/key-escrow` | `{"key_material": "<base64>"}` | `{escrowed: true, updated_at}` |
| `GET /v1/backup/key-escrow` | — | `{key_material, updated_at}` / 404 |
| `DELETE /v1/backup/key-escrow` | — | 204, idempotent |

`GET /v1/backup/meta` har fått **`escrowed: true/false`**, så en ny telefon kan
si «kopien kan gjenopprettes uten kode» før den laster ned 16 MB.

`key_material` er base64 av ugjennomsiktige byte (≤ 512). Serveren bryr seg
ikke om det er en nøkkel eller gjenopprettingskoden — den lagrer og leverer
tilbake det den fikk.

**Hjelpeteksten din er nesten riktig, men én setning bør endres.** Du skrev
«dataene dine er ikke krypterte på serveren». De *er* krypterte — forskjellen
er at vi da har nøkkelen. Materialet krypteres dessuten i ro med AES-256-GCM
under en hemmelighet som ligger som Fly-secret, altså et annet sted enn
databasen; en Supabase-dump alene gir ingen nøkler. Det gjør det ikke like
trygt som å la være, men presisjonen betyr noe. Forslag:

> Datasikkerhet: Slår du dette på, kan du få dataene tilbake selv om du mister
> både telefonen og gjenopprettingskoden.
>
> Prisen er at vi da har nøkkelen til kopien din. Den ligger kryptert og
> atskilt fra selve kopien, men vi *kunne* i prinsippet lese jaktloggen din —
> og en angriper som kom seg inn overalt hos oss, kunne det også.
>
> Spørsmålet du må stille deg: hvor ille ville det være om noen leste alt du
> har i appen?

Tre praktiske ting til klienten:

- `DELETE /v1/backup` fjerner **bloben, ikke deponeringen**. Deponeringen er en
  innstilling, ikke en følge av at det finnes en kopi akkurat nå. Hadde vi
  slettet den der, ville neste opplasting stilltiende vært udeponert med
  bryteren stående på.
- `DELETE /v1/backup/key-escrow` virker **også når serveren ikke har
  hemmeligheten konfigurert**. Å komme seg *ut* av valget skal aldri kunne
  feile.
- Er `BACKUP_ESCROW_SECRET` ikke satt, svarer `PUT`/`GET` **503**. Skjul eller
  deaktiver bryteren da — ikke vis den som en feil. Vi lagrer heller ingenting
  enn å lagre nøkler i klartekst.

### 2. Skadedata kan nå lagres for forskning (§7)

Du har rett i at det er den sentrale informasjonen. Det var før *umulig* å
dele: §7 sa «private som standard», og uten en bryter betyr det i praksis
«aldri».

Nå finnes bryteren: `share_injury_data` i `GET/PUT /v1/research/sharing`, **av
som standard**. Feltene som slipper gjennom når den er på: `wounded`,
`injury`, `hit_placement`, `shots_fired`, `tracking_distance_m`,
`tracking_time_min`, `dog_used`, `recovered`.

Den står **for seg selv**, ikke sammen med art og sted. Grunnen: «jeg skjøt
stående på 85 meter» og «dyret ble skadeskutt og aldri funnet» er ikke samme
opplysning å dele om seg selv, og den som vil bidra med det siste skal ikke
måtte dele det første. Klienten bør speile det med en egen bryter og en
hjelpetekst som sier hva som faktisk sendes.

### 3. Sperrefrist på «send ny kode»

Se svaret over.

## Om «`deleted`-flagg i backend for slettede jaktskudd»

Sto som `<Krever_backend_verifisering>` i musingsUI. Svaret er at det **ikke
er en backendoppgave i dag**: jaktskudd finnes ikke som rader på serveren.
De ligger inne i den klient-krypterte bloben, der soft-delete allerede er
innført (`HuntRecord.deletedAt`, §13). Gravsteinen *er* flagget, og den er
klientens.

Det blir en backendoppgave den dagen §5-synken kommer og enkeltposter får
egne rader. Da må slettede poster sendes som gravstein, ikke utelates — ellers
gjenoppstår de ved neste synk fra en annen enhet. Det står i §13.

## Nytt som venter på utvikler

| Hva | Konsekvens til det er gjort |
|---|---|
| `BACKUP_ESCROW_SECRET` | `/v1/backup/key-escrow` svarer 503; «gjenopprett uten kode» er utilgjengelig |

```powershell
flyctl secrets set BACKUP_ESCROW_SECRET=<lang tilfeldig streng> -a bestefar-api
```

**Denne roteres aldri uten en plan.** Roterer du den, blir alle deponerte
nøkler uleselige, og brukerne må falle tilbake på gjenopprettingskoden — som
er nettopp den de slo på valget for å slippe.

## Migrasjon

`e5b0c72a94d1` — ny tabell `backup_key_escrow`, ny kolonne
`research_sharing_preferences.share_injury_data` (`server_default false`, så
ingen eksisterende bruker får utvidet deling av en migrasjon).

---

# Backend — runde 12 (deponering verifisert i produksjon)

**188 tester grønne.** Deponeringsflyten er kjørt mot `bestefar-api.fly.dev`
med en ekte konto, ikke bare mot testsuiten.

## Verifisert i produksjon

| Steg | Resultat |
|---|---|
| `GET /v1/backup/key-escrow` før deponering | 404 |
| `PUT /v1/backup` | 200, `escrowed: false` |
| `PUT /v1/backup/key-escrow` | 200, `escrowed: true` |
| `GET /v1/backup/key-escrow` | 200 — **identisk materiale tilbake** |
| `GET /v1/backup/meta` | `escrowed: true` |
| `DELETE /v1/backup` → `GET key-escrow` | 204 → **200**, deponeringen står |
| `DELETE key-escrow` ×2 → `GET` | 204, 204, 404 |
| `/email/start` to ganger innen 60 s | 202, så **429 med `Retry-After: 48`** |

Rundturen gjennom AES-256-GCM under produksjonshemmeligheten fungerer. Kontoen
er ryddet etterpå — ingen testblob og ingen deponering ligger igjen.

## Hemmeligheten er selv et enkeltpunkt

Spørsmålet fra utvikler: `BACKUP_ESCROW_SECRET` lever bare i Fly, og ble
autogenerert uten kopi. Mister Fly den, er de deponerte nøklene borte.

**Det er ikke datatap.** Bloben ligger fortsatt der, fortsatt kryptert med
brukerens egen nøkkel, fortsatt åpningsbar med den 20-tegns
gjenopprettingskoden. Systemet degraderer til slik det var før deponering
fantes.

**Men tapet treffer skjevt.** De som slo på «gjenopprett uten kode», gjorde det
stort sett fordi de ikke ville forholde seg til en kode — og har derfor med
stor sannsynlighet ikke tatt vare på den. Feilen rammer presis den gruppen som
er dårligst rustet til å tåle den.

**Den realistiske trusselen er ikke at Fly går under**, men
`flyctl secrets set BACKUP_ESCROW_SECRET=...` kjørt en gang for mye. Menneskelig
feil, ikke leverandørsvikt.

### Bygget som svar

1. **`BACKUP_ESCROW_SECRET_OLD`** — rader som ikke åpnes av den gjeldende
   hemmeligheten prøves med den forrige, og **krypteres om ved første lesing**.
   En utskiftning migrerer seg selv. Rekkefølge: sett `_OLD` = dagens, sett ny,
   følg `/health` til den sier `ok`, fjern `_OLD`.
2. **Nøkkel-ID på hver rad** (`key_check`) — HMAC over en fast streng under den
   avledede nøkkelen. Et fingeravtrykk som ikke røper hemmeligheten, men lar
   `/health` si `escrow: "av" | "ok" | "N rader paa annen hemmelighet"`. Uten
   det ville en feilsatt hemmelighet først vist seg den dagen en bruker prøvde
   å gjenopprette.

Forkastet: envelope-kryptering med KMS. Flytter holdbarhetsproblemet til noen
med bedre SLA, men koster en avhengighet — og en KMS-nøkkel har samme
egenskap: sletter du den, er den borte.

### Anbefaling: lag en ny hemmelighet nå

Den nåværende ble autogenerert uten at verdien ble sett, så det finnes ingen
kopi noe sted. **Akkurat nå er det gratis å bytte:** funksjonen er timer gammel,
ingen klient bruker den ennå, og det ligger null deponerte nøkler i basen.

Generer en ny slik at du *ser* verdien, legg den i passordhvelvet ditt, og sett
den så:

```powershell
$ny = [Convert]::ToBase64String((1..48 | ForEach-Object { Get-Random -Max 256 }))
$ny                     # les den, lagre den i hvelvet
flyctl secrets set BACKUP_ESCROW_SECRET=$ny -a bestefar-api
```

Regelen om at hemmeligheter ikke skal i chat-loggen gjelder transkriptet, ikke
passordhvelvet ditt. Denne verdien *bør* skrives ned: den gir ingenting alene
(uten databasen er den verdiløs), skal aldri roteres rutinemessig, og utløper
ikke.

## Migrasjon

`f1a7d3c8e206` — `backup_key_escrow.key_check` (`server_default ""`).
Eksisterende rader rapporteres som avvikende og retter seg selv ved
første lesing.
