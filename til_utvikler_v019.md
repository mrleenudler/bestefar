# Til utvikler — v0.19 (meldingskøen leses)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en seksjon nederst — ikke overskriv.

## Oppdraget

> «Bygg henting av /v1/messages ved appstart, visning av ventende meldinger, og
> POST /v1/messages/ack etter at meldingen faktisk er vist.»

Meldt som issue #4 før arbeidet, lukket med denne runden.

---

## Hva som var galt

Backenden har siden fase 8 hatt to leveringsveier for §11-varsler, og bare den
ene virket.

Køen legges **først**: når et lag skifter navn, når du fjernes fra et lag, når
noen tilbyr deg lederskapet, når en avstemning starter. Deretter sendes en push.
Backendens egen invariant sier det rett ut — «push feiler aldri oppover;
meldingskøen er garantien, push er bekvemmeligheten» — og `PUSH_BUDGET_SECONDS`
avbryter til og med utsendingen midt i en runde, nettopp fordi køen skal ta imot
det som ikke rakk å bli sendt.

**Klienten hentet aldri køen.** Push var dermed ikke bekvemmeligheten, den var
alt. Rammet:

- brukere som har svart nei på varseltillatelsen — og på Android 13+ er det
  neiet nesten permanent,
- telefoner som var av da meldingen ble sendt,
- enheter der FCM-tokenet er rotert uten at `onNewToken` rakk å melde fra,
- alle tilfeller der push-budsjettet tok slutt.

Beskjedene det gjelder tåler ikke å forsvinne. «Avstemningen er åpen i 7 dager»
er en frist. Går den ut fordi beskjeden aldri ble vist, kan den ikke rettes opp
etterpå.

---

## Hva som er bygget

`Messages.kt` henter køen, `MainActivity` viser den, og kvitteringen sendes
etterpå.

### Rekkefølgen er hele poenget

**Kvitteringen sendes når beskjeden ER vist**, ikke når den hentes. Serveren
sletter ikke raden når du kvitterer — den setter `delivered_at`. Det er et
bevisst valg fra backendsiden, og det er verdiløst hvis klienten kvitterer for
tidlig: da har vi kastet bort nettopp den toleransen som skulle dekke en app som
lukkes midt i.

Prisen er at en beskjed i sjeldne tilfeller kan vises to ganger — appen lukkes
mellom visning og kvittering. Det er den billige feilen av de to.

### Beskjedene kommer *etter* oppstartsvinduene

Hentingen starter parallelt med oppstartskjeden, men beskjedene holdes til
intro, bildedelingsspørsmål og tutorial er unnagjort. Et nettverkssvar som
lander midt i tutorialen skal ikke legge seg oppå den.

Det er derfor `onStartupOverlaysDone` finnes i `MainActivity`: hver utgang av
oppstartskjeden må kalle den, og en gren som glemmer det, gir en bruker som
aldri ser beskjedene sine. Verdt å vite hvis du legger til et nytt
oppstartsvindu.

### Presentasjonen

Fullskjerm, samme form som oppstartsmeldingen, én beskjed om gangen i serverens
rekkefølge. Overskrift, tekst, og tidspunktet som en dempet linje under —
tidspunktet står for seg selv fordi det ikke er en del av beskjeden, det er når
den kom.

**Teksten vises ordrett slik serveren sendte den.** Klienten omskriver ikke og
reparerer ikke; ser du en skrivefeil i en beskjed, er den backendens.

### Alt som kan feile, feiler stille

Ingen konto: kallet sendes ikke. Offline, 401, ubrukelig svar: tom liste, ingen
feilmelding, køen blir stående på serveren og hentes ved neste oppstart. Dette
er en oppstartsjobb i en offline-først app — den skal aldri hindre at appen
starter, og aldri vise en feil for noe brukeren ikke kan gjøre noe med.

---

## En ekte feil funnet underveis

**`Api.send` med `GET` traff serveren som `POST`.**

`request()` åpnet alltid utstrømmen, og `HttpURLConnection` gjør en GET om til
POST i det øyeblikket utstrømmen åpnes. Det er arvet, dokumentert JDK-oppførsel,
ikke en Android-særhet — men den er stille.

Det rammet to kall som har ligget i appen siden v0.15/v0.16:

| Kall | Hva som faktisk skjedde |
|---|---|
| `GET /v1/backup/meta` | POST → **405** |
| `GET /v1/backup/key-escrow` | POST → **405** |

Bekreftet mot produksjon: `GET /v1/backup/meta` svarer 401 uten token, `POST`
svarer 405. Serveren har altså aldri sett de to kallene slik de var ment.

Fikset ved å bare åpne utstrømmen når det finnes en kropp å skrive.
`Api.download` var alltid riktig; det er `Api.send` som var feil. `PUT` og
`DELETE` ble ikke rørt av dette — bare `GET`.

*Rettelse, lagt til etter gjennomgangen under:* i første versjon av dette
notatet skrev jeg at 405-grenen i `AvansertActivity.confirmEscrow` skjulte
feilen. Det stemmer ikke — den grenen gjelder `escrowPut`, som er en `PUT` med
kropp og aldri var rammet. Hvordan feilen faktisk overlevde, står i seksjonen
under.

**APK-er som allerede er ute i felt gjør dette fortsatt.** Ser dere 405 på de to
rutene i backend-loggen, er det en gammel klient — ikke en rutefeil.

---

## Gjennomgang: hvorfor overlevde GET-feilen tre versjoner?

Spørsmålet er bedre enn feilen. Svaret er to uavhengige grunner, én per kall.

### `Backup.meta()` hadde ingen kaller

Funksjonen var skrevet, dokumentert og ødelagt fra dag én i v0.15. Ingen kalte
den. En funksjon som ikke kalles, blir aldri verifisert — den ser ferdig ut i
diffen og finnes ikke i kjøringen.

Det er verdt å merke seg *hva* som lå ubrukt: `GET /v1/backup/meta` er kallet
som svarer «har jeg noe å gjenopprette?» på en ny telefon, og backenden bygget
det nettopp for at klienten skulle slippe å laste ned 16 MB for å finne det ut.

**Nå brukes den.** «Gjenopprett fra sikkerhetskopi» spør serveren først:

- **404** → «Ingen sikkerhetskopi funnet», og flyten stopper der. Før dette ble
  brukeren bedt om gjenopprettingskoden sin, tastet den inn, og *deretter* fikk
  beskjed om at det ikke fantes noe — svaret på et spørsmål de ikke stilte.
- **401** → «du må være innlogget».
- **Noe annet** (offline, 5xx) → vi går videre til den gamle veien. Et oppslag
  som ikke svarer, skal ikke blokkere en gjenoppretting som ellers kunne gått
  bra.
- **200** → bekreftelsesdialogen sier nå **når kopien ble laget**. «Dette
  erstatter alt» er en annen beslutning når man vet om kopien er fra i går
  eller fra i fjor.

### `BackupKeys.escrowGet()` ble svelget av en fallback

`resolve()` prøver lokalt → Block Store → deponering, og returnerte **tom
streng både når deponeringen var tom og når kallet ikke nådde fram**. Med et
405 hver gang så det ut som «ingen nøkkel deponert».

Konsekvensen for brukeren var den motsatte av hensikten: den som hadde slått på
«Gjenopprett uten kode» ble likevel bedt om koden på den nye telefonen — altså
akkurat det bryteren fantes for å slippe. Og siden det å bli spurt om koden er
en helt normal tilstand, var det ingenting å melde fra om.

**Nå logges alt som ikke er 404** (`BestefarKeys: Deponeringen svarte 405`).
Selve tvetydigheten er ikke fjernet — brukeren får fortsatt kodedialogen, som er
det trygge utfallet — men den er ikke lenger usynlig.

### Lærdommen, ført inn i rot-`CLAUDE.md` §7.3

- En feilkode som får en forklaring før den får en årsak, blir stående.
- En fallback-kjede som behandler en feil som et fravær, skjuler feilen.
- Kode uten kaller blir aldri verifisert.

### Er gjenopprettingsflyten hel nå?

Så langt den kan verifiseres uten to telefoner: ja. Kjeden er
`meta` → `resolve` (lokalt → Block Store → deponering) → `GET /v1/backup` →
dekryptering → `replaceAll`. `GET /v1/backup` gikk gjennom `Api.download`, som
alltid har brukt riktig verb — **selve nedlastingen har aldri vært ødelagt.** Det
som var ødelagt, var de to kallene som skulle gjøre gjenopprettingen *enkel*:
å vite om det finnes noe, og å slippe å taste koden.

Det som fortsatt ikke er verifisert, står under.

---

## Det du må gjøre

Ingenting for at dette skal virke. Men for å **se** at det virker trengs det en
beskjed i køen, og den enkleste veien dit er to kontoer i samme lag der den ene
endrer lagnavnet. Alternativt kan backend legge en rad i `pending_messages`
direkte.

Fra v0.18 står fortsatt: `FCM_SERVICE_ACCOUNT_JSON` og `FCM_PROJECT_ID` må settes
som Fly-secrets for at pushen skal sendes. Merk at **meldingskøen virker uten
dem** — det er hele poenget med at den finnes.

---

## Verifisert

- `compileDebugKotlin` grønt.
- Release-bygg OK, kopiert til `dist\Bestefar-0.19.apk`.
- Metodefølsomheten bekreftet mot produksjon (`GET` → 401, `POST` → 405 på både
  `/v1/backup/meta` og `/v1/messages`).
- Skjemaet lest ut av `routers/messages.py` og `models/social.py`, ikke antatt
  fra speccen.

## Ikke verifisert

- **Ingen beskjed er faktisk hentet og vist.** Hele veien er bygget, men den
  krever en rad i køen, og det krever to kontoer i et lag med server-kobling.
- Dato- og klokkeslettformateringen er ikke sett med et ekte `created_at` fra
  serveren. Tre parsingforsøk dekker offset, `Z` og naken ISO; feiler alle,
  vises ingen dato i stedet for en ISO-streng.
- Rettelsen i `Api.send` er verifisert ved at koden kompilerer og ved at
  serveren skiller metodene — ikke ved et kall fra enhet.
- **Gjenoppretting er ikke kjørt ende-til-ende.** Den krever en kopi på
  serveren og helst en annen telefon. `meta`-gaten, datoen i bekreftelsen og
  deponeringsoppslaget er lest og kompilert, ikke sett i drift. Dette er
  funksjonen man oppdager er ødelagt den dagen man trenger den, så den bør
  prøves med vilje: ta en kopi, slett appdata, gjenopprett.

## Fortsatt åpent

- **Beskjeder som ber om en handling kan ikke besvares.** «Bekreft i appen» og
  «Avstemningen er åpen i 7 dager» vises som tekst. Det finnes ingen knapp til
  laget, fordi lagsidene fortsatt er lokale skjeletter uten server-kobling —
  `team_id` fra serveren peker ikke på noe klienten kjenner. Beskjeden når fram;
  svaret kan ikke gis. Dette er nå den største gjenstående luken i §11.
- **Køen hentes bare ved appstart**, ikke ved `onResume`.
- **Ingen ruting på `kind`** — verken fra push eller fra køen.
- **Serie-synk-køen** (`/v1/stats`) er fortsatt ikke bygget; ÅP-B4 står.
- **ÅP-U13** (backup-metadataene går forbi hverandre) er ført opp, ikke løst.

## Meldt til de andre

- **#5 (backend):** `backend/KONTRAKT.md` har ingen seksjon for meldingskøen.
  Skjemaet måtte leses ut av routeren, og to felttyper bryter med mønsteret
  ellers (`id` er `int`, `team_id` er streng).
- **#6 (backend):** brukerteksten i `leader_demoted` mangler norske tegn —
  «staar uten lagleder». Den går rett inn i `body`, som klienten nå viser
  ordrett.
