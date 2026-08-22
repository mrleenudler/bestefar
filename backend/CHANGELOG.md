# Endringslogg — backend

Hva som ble bygget når. **Teksten er flyttet ordrett** ut av `backend_spec.md`
2026-08-07, der de dateres notatene lå inne i løpende tekst og gjorde det umulig
å lese en paragraf som en beskrivelse av hvordan backenden er i dag.

Arbeidsdelingen mellom de fire dokumentene:

| Spørsmål | Fil |
|---|---|
| Hvordan er backenden i dag? | `../backend_spec.md` |
| Hva kan andre stole på? | `KONTRAKT.md` |
| Hvorfor ble det slik, og hva ble forkastet? | `BESLUTNINGER.md` |
| Når ble det bygget? | denne fila |

Begrunnelsene er i `BESLUTNINGER.md`. Står det en begrunnelse her og ikke der,
er det en feil — meld fra.

Nyeste først. Datoene er de som sto i spec-notatene.

---

## 2026-08-22

**§6 — `capture_trigger` på donasjonen** (B-53, issue #11). Nytt valgfritt
multipart-felt, `auto` | `timeout`, med kolonne (`c4a91e7b2f38`). Eget felt og
ikke en ny `tag`-verdi: `tag` sier hva donasjonen viser, `capture_trigger`
hvordan bildet ble tatt, og en timeout-capture kan ende som hvilken som helst
tag. **NULL betyr «ikke oppgitt», ikke `auto`** — donasjonene fra v0.29-vinduet
er dels timeout-utløste uten å kunne si det, og det er nettopp de radene ÅP-K1
skal måles på. Ingen backfill.

**§4 — klienten kan kjenne igjen seg selv i et lag** (B-54, issue #14).
`my_role` ∈ {`leader`, `member`, `null`} på **alle** lagsvar, også listesvaret
der `members[]` ikke er med, og `public_id` i `members[]`. To felt fordi det er
to spørsmål: hva jeg er, og hvilket element som er meg. `members[].user_id`
står fortsatt — hele flaten tar imot den formen — og konverteringen til
`public_id` hører sammen med venne-modellen (ÅP-U31).

**`mailer.py`: SMTP-grenen sendte til utviklerinnboksen.** `msg["To"]` var
hardkodet til `cfg.feedback_to` uansett mottaker, så innloggingskoder og
lag-invitasjoner ville gått til utvikleren i stedet for brukeren. Resend-grenen
ble rettet da `to` ble innført; denne ble stående. Produksjonen bruker Resend,
så feilen var sovende. Ny `tests/test_mailer.py` kjører **begge** grenene mot
det samme kravet, og testen er verifisert som en ekte port: den feiler på den
gamle koden.

## 2026-08-21

**§6 — donasjonen er frikoblet fra kontoen** (B-52, del 1 av ÅP-B13).
`series_id` tas ikke lenger imot av `POST /v1/failed-analyses`, og kolonnen er
droppet (migrasjon `b8d24a0f5c17`) — også for radene som alt fantes. Feltet var
den ene veien fra et donert bilde til en person: samme ID som serien lagres
under i `/v1/stats`. **Ingenting leste den** — den hadde til og med en indeks
uten et eneste oppslag — og kalibreringsmaterialet (`tag`, `detected_scores`,
`ocr_scores`) er selvstendig.

Kolonnen droppes i stedet for å nulles: en «vi slutter å skrive den»-løsning
ville latt de eksisterende radene stå igjen som personopplysninger, og da kunne
ikke ÅPENT PUNKT 6 i `personvernerklaring.txt` lukkes.

**En eldre klient som fortsatt sender feltet, får 201 som før.** FastAPI
ignorerer skjemafelt ruten ikke erklærer. Et 4xx ville vært ikke-`retryable` og
dermed stille tap av donasjonen hos brukere vi ikke kan oppdatere i samme
øyeblikk — klienten kan slutte å sende det når det passer.

**Talt i produksjon rett før migrasjonen: 9 rader, 6 med `series_id`, 0 med
`user_id`.** Seks koblinger til en konto forsvant. Tallet kan ikke hentes i
ettertid, og det er ikke det samme som «sju objekter» i B-50 (objekter i R2) —
se ÅP-B13 for hvorfor de tallene ikke skal stemme overens.

**`user_id` er droppet i samme migrasjon.** Den var tom og har aldri vært satt
av endepunktet, så den var ingen personopplysning — men en ferdig oppkoblet
fremmednøkkel til `users` på nettopp denne tabellen er en mulighet som nå
krever en ny migrasjon for å gjenåpnes.

**Verifisert i produksjon 2026-08-21**, og det er dét som lukket ÅP-B13 del 1 og
ÅPENT PUNKT 6 — ikke at koden var skrevet: en ekte donasjon kom inn etter
utrullingen, `failed_analyses` gikk fra 9 til 10 rader, mot et skjema uten
begge feltene.

Del 2 av ÅP-B13 — at vellykkede scans ikke skal lagre bilde, bare
treffkoordinater — er **ikke** bygget: det krever at klienten slutter å legge
ved bildet og begynner å sende koordinater, altså en kontraktendring. Meldt som
issue #10 med label `ui`.

## 2026-08-18

**§6 — `/health` skiller «av» fra «i stykker»** (B-51, lukker ÅP-B12). Feltet
`bilder` hadde to verdier, `"r2"` og `"ikke konfigurert (§6)"`, og den første
betydde bare at fire strenger var ikke-tomme. Under bucketbyttet dagen før
svarte den `"r2"` gjennom tre ulike feilkonfigurasjoner mens produksjonen ikke
kunne skrive et eneste bilde.

Nå er det tre svar: `"r2"`, `"ikke konfigurert (§6)"` (ingenting satt — en
funksjon som er av) og `"feilkonfigurert (<hva>)"`. Det siste dekker både
halvveis satte secrets og de feilene som **kan ses uten et nettverkskall**:
mellomrom eller linjeskift rundt en verdi, `R2_ENDPOINT` som ikke er en URL
eller som har sti i seg, `R2_BUCKET` med skråstrek, og `R2_ACCESS_KEY_ID` som
ikke er 32 tegn. `objstore.feilkonfigurasjon()` er sjekken;
`objstore.tilstand()` er teksten.

`POST /v1/failed-analyses` avviser nå det feilkonfigurerte tilfellet på samme
måte som det ukonfigurerte — 503 før kroppen leses — og logger hvilken av dem
det var. Klienten får samme svar som før. `tools/r2_check.py` kjører de samme
sjekkene før den spør Cloudflare, og `bucketflytt.kopier()` nekter å starte på
et oppsett som umulig kan virke.

To ting fulgte med: `objstore.backend_name()` er slettet (ingen kallere, og den
svarte `"database"` — et sted bildene ikke har ligget siden B-49), og
testoppsettene bruker nå en 32 tegns nøkkel-ID.

**Ikke dekket:** at tokenet mangler EU-jurisdiksjon. Det var den tredje feilen
2026-08-18, og den kan bare et faktisk kall svare på — `r2_check.py`.

## 2026-08-16

**§6 — bildene skal til en jurisdiksjonsbundet bucket** (B-50). Den gamle
bucketen har *location hint* `EEUR` uten jurisdiksjonsbinding — et hint er ingen
garanti, og «Eastern Europe» dekker land utenfor EØS. Ny
`app/services/bucketflytt.py` med `tools/copy_bucket.py` kopierer objektene med
**uendret nøkkel** til `bestefar-scan-failures-eur`, og rører ikke databasen:
`object_key` peker på hele stien. Tørrkjøring som standard, kan kjøres om igjen,
og sletter ingenting — den gamle bucketen står til den nye er verifisert med en
ekte donasjon.

To ting i `objstore` fulgte med: **`Bucket`** samler endepunkt, bucketnavn og
nøkler, siden en jurisdiksjonsbundet bucket har sitt eget endepunkt (`.../eu`)
og to buckets derfor ligger på hver sin URL; og **`LagringFeilet.status`**, så
404 kan skilles fra 403 uten å lete etter et tall i feilteksten.

Krever nye Fly-secrets, se `tools/copy_bucket.py`: `R2_ENDPOINT` og `R2_BUCKET`
byttes til den nye bucketen, og kilden oppgis med `R2_KILDE_ENDPOINT` og
`R2_KILDE_BUCKET`.

**Kjørt i produksjon 2026-08-18: sju objekter, 16 096 622 byte.** Ikke fem — de
fem gamle `image_legacy`-radene pluss to ekte donasjoner som kom inn etter at R2
ble koblet på. De ble med fordi jobben leser nøklene fra databasen og ikke fra
en telling; en jobb skrevet mot «de fem» ville latt det mest verdifulle
materialet ligge igjen i bucketen uten jurisdiksjonsbinding. Verifisert ved at
en ny tørrkjøring melder alle sju som «allerede der», altså lest fra målet og
sammenlignet byte for byte. Den gamle bucketen er ikke tømt (ÅP-E11 steg 4).

Fire konfigurasjonsfeil måtte rettes underveis, og `/health` svarte
`"bilder": "r2"` gjennom alle sammen — se ÅP-B12.

**Skrivingen er verifisert i drift samme dag.** En ekte donasjon fra appen (rad
11, 18:53:55Z) ligger i `bestefar-scan-failures-eur` med 3 289 511 byte og
finnes ikke i den gamle bucketen — den kom altså dit ved innsending, ikke ved
kopiering. Det er noe annet enn at `r2_check.py` kan skrive: dette er klientens
egen multipart hele veien gjennom mottaket.

## 2026-08-15

**§6 — `image_legacy` er borte, og uten R2 tar vi ikke imot donasjoner**
(B-49). Migrasjon `a3f7c1e59b24` dropper kolonnen. Fallbacken «uten R2 → legg i
basen» hadde da ikke lenger noe sted å gå, og er erstattet med **503 før
kroppen leses** — et 201 på noe vi kastet ville vært kvittering for datatap.
`GET /health` svarer nå `"bilder": "r2"` eller `"ikke konfigurert (§6)"`;
radtellingen forsvant med kolonnen. `services/legacy_bilder.py`,
`tools/migrate_legacy_images.py` og testene deres er slettet — engangsjobben er
utført (se commit `b29b4b3` og `3f4e634` for koden). Merk for lokal utvikling:
uten R2-nøkler kan endepunktet ikke lenger ta imot noe.

**§6 — de gamle `image_legacy`-bildene flyttes til R2** (B-48, ÅP-B11). Ny
`app/services/legacy_bilder.py` med `tools/migrate_legacy_images.py` som skall.
Per rad: last opp, **les tilbake og sammenlign byte for byte**, og først da tøm
kolonnen — basen er den eneste kopien, så en PUT som svarte 200 uten et lesbart
objekt ville kostet bildet. Commit per rad, tørrkjøring som standard (`--utfoer`
for å skrive), og `MAX_UPLOAD_BYTES` gjelder ikke, siden den er en regel for
mottaket. Bildesniffingen flyttet fra routeren til `objstore.bildetype()`, siden
både mottaket og flyttingen trenger den.

**Kjørt i produksjon samme dag.** Fem rader, 13 byte til 3 844 036 byte,
10 509 298 byte til sammen, alle JPEG og alle med innsendingsdatoen i nøkkelen.
`GET /health` svarer nå `"bilder": "r2"` uten radtelling, og `image_legacy` er
tom. ÅP-B11 lukket. Merk at «11 MB-raden» som sto som begrunnelse i B-48 ikke
fantes — det tallet var totalen; ingen enkeltrad var over 8 MB.

**`tools/r2_check.py` er med i produksjonsimaget** (B-47). Verktøyet krever de
fire R2-verdiene, og de finnes bare som Fly secrets — Fly leser dem ikke ut
igjen. Verifiseringen kunne derfor bare kjøres et sted den ikke virket. Nå:
`flyctl ssh console -a bestefar-api -C "python tools/r2_check.py"`.
`gen_openapi.py` er bevisst utelatt; skillet står som kommentar i `Dockerfile`.

**§6 — feilanalyse-bildene lastes opp til Cloudflare R2** (B-44, B-45, B-46;
ÅP-B5 lukket). Ny `app/services/objstore.py`: SigV4-signert PUT/GET/DELETE mot
S3-API-et, uten boto3. `POST /v1/failed-analyses` skriver bildet dit og setter
`failed_analyses.object_key`; `image_legacy` fylles ikke lenger når R2 er
konfigurert. Nye innstillinger `R2_ENDPOINT`, `R2_BUCKET`, `R2_ACCESS_KEY_ID`,
`R2_SECRET_ACCESS_KEY`, `R2_REGION` (standard `auto`), `R2_TIMEOUT_SECONDS`.

Tre ting fulgte med:

- **`GET /health` fikk feltet `bilder`**: `r2`, `r2 (N gamle rader i basen)`
  eller `database (avvik fra spec §6)`. Uten det ville en glemt secret sett ut
  som normal drift mens bildene la seg i basen §6 forbyr.
- **En feilet opplasting gir 503**, og raden rulles tilbake — aldri 201 med
  bildet lagt i basen i stillhet (B-45).
- **415 på noe som ikke er JPEG/PNG/WebP**, avgjort av de første bytene (B-46).
  Endepunktet er åpent og skriver nå til betalt lagring.

`tools/r2_check.py` gjør en ekte PUT/GET/DELETE med de secretene som står, og er
måten å verifisere at signeringen virker mot ekte R2. Ingen migrasjon:
`object_key` fantes fra `d21e5f8ac782`. Rader fra før i dag er ikke flyttet.
Endepunktet hadde ingen tester før nå; `tests/test_failed_analyses.py` er ny.

## 2026-08-12

**§3 — standard blokkliste for visningsnavn** (B-43). `DISPLAY_NAME_BLOCKLIST`
var tom som standard, og moderasjonen fanget derfor ingen ord i noen
installasjon som ikke satte den. Standardverdien er nå en kurert liste i
`app/services/blocklist.py`; miljøvariabelen **legges til** den og kan ikke slå
den av. Ordene er valgt mot delstrengsmatching på foldet form — de vraket
kandidatene står med begrunnelse i docstringen. Ny `DISPLAY_NAME_ALLOWLIST` for
navn som treffes utilsiktet på tvers av for- og etternavn («Anne Gerd» →
«neger»); `moderation.review` tar unntakslista som tredje argument. Ingen
migrasjon, ingen ruteendring.

## 2026-08-11

**§1 — visningsnavn fra ID-tokenet** (issue #7). `name` leses nå ut av de
verifiserte kravene i stedet for å forkastes, så en Google-konto får et ekte
navn og ikke lokaldelen av adressen. Navnet modereres, og med den ekte ordlista
— den utledede navnet gikk tidligere gjennom moderasjonen med tom blokkliste.

**§1 — `email` i tokenparet** (issue #8). `auth_sessions.identity_id` husker
hvilken identitet økten ble startet med, så svaret sier hva brukeren logget inn
*som* — også etter `/refresh`. Migrasjon `c7e2b91f4d08`.

**Rettelser.** `backend/CLAUDE.md` oppga feil eierskapsgrense i `backend_spec`
(sa §0–§13; riktig er §0–§11, klienten eier §12–§17). §15 og §16 fikk
verifiseringen ved seg i stedet for en påstand om når en secret ble satt — §16
sin dato var usann. ÅP-B8 utvidet med at `advarsel`-feltet i `PUT /v1/profile`
er død kode: `moderation.review` returnerer aldri `pending`.

## 2026-08-10

**§15 — `GOOGLE_CLIENT_IDS` satt i produksjon** (ÅP-E4). `/v1/auth/google`
svarer nå 401 på et ugyldig token i stedet for 503. Ingen kodeendring.

**§16 — `FCM_SERVICE_ACCOUNT_JSON` satt i produksjon** (ÅP-E3). `/health` svarer
`"push":"fcm"`, og §11-kjeden er hel fra utløsende hendelse til varsel på
telefonen. `FCM_PROJECT_ID` ble aldri nødvendig — den leses fra JSON-en. Punktet
påsto tidligere at begge måtte settes; det var feil, og er rettet i samme
omgang. Ingen kodeendring.

**§2.1 — kopi av `BACKUP_ESCROW_SECRET` utenfor Fly er på plass** (ÅP-E6, utført
av eier). Ingen kodeendring; det var det siste av de tre tiltakene mot at
hemmeligheten er et enkeltpunkt, og alle tre er nå på plass.

## 2026-08-09

**§11 — push flyttet ut av forespørselen** (`BackgroundTasks`, issue #3).
Brukeren venter ikke lenger på at andre får varsel. `BackgroundTasks` deler
prosess, så pusher som ikke er sendt når maskinen dør, er tapt — køraden
overlever. Loggingen strammet til to linjer: budsjettavbrudd og FCM-avvisning
som ikke er et dødt token.

**§11 — kvorum på lederavstemningen.** 25 % av medlemstallet ved start, rundet
opp, låst i `member_count_at_open`. Under kvorum ved fristen gir `expired`, uten
sperrefrist før en ny avstemning. Migrasjon `b4c81f0d3a97`.

**§11 — fristen gjort absolutt.** `active_election` returnerte raden også når
den nettopp var avgjort, så det første kallet etter fristen fikk stemme på en
avstemning som allerede var utløpt. `active_challenge` hadde samme feil.

**§11 — overkjørte kømeldinger leveres ikke.** `pending_messages.superseded_at`
settes når et utfall skrives, og `GET /v1/messages` filtrerer dem bort.

## 2026-08-08

**§11 — kø-garantien ble reell.** Klientens v0.19 henter `/v1/messages` ved
appstart og kvitterer etter visning (issue #4). Ingen backend-endring; det er
motparten som kom på plass.

**§3, §4, §1 — fem brukertekster fikk norske tegn.** `leader_demoted`,
moderasjonsadvarselen i profilen, 503-teksten for kort `JWT_SECRET`,
mottakerfeilen i lag-invitasjon og begge `EscrowUnreadable`-meldingene sto med
ASCII-translitterering (issue #6). Konvensjonen gjelder kommentarer og logg, ikke
tekst brukeren ser.

**`backend/KONTRAKT.md` §4.1 — meldingskøens skjema dokumentert** med typene
eksplisitt, etter at klienten måtte lese dem ut av koden (issue #5).

## 2026-08-07

**§0.1 — rammeverket beskrevet riktig.** Flyttet fra `docs/ARCHITECTURE.md`, som
beskrev backenden som «FastAPI + SQLite … tre routere». SQLite er testdialekten,
ikke produksjonsdialekten, og routerne er nå fjorten.

**§14–§16 redusert til pekere.** Klientsiden av §1 og §11 gjentok
`android/ARCHITECTURE.md` nesten setning for setning. En invariant eies av den
som håndhever den; det som står igjen er bare det serveren selv håndhever.

## 2026-08-06

**§1 — sperrefrist på «send ny kode»** (`EMAIL_CODE_RESEND_COOLDOWN_SECONDS`,
60 s). Innen fristen svarer `/email/start` 429 med `Retry-After`, og ingen
e-post sendes. Nedtellingen i klienten er bekvemmelighet — en klient kan endres,
og en gratis e-post til en fremmed adresse er akkurat det man ikke vil kunne
sende i løkke. Verdiene ligger i 202-svaret (`resend_after_seconds`,
`expires_in_minutes`) så klienten slipper å hardkode dem. Kvoten på 10 koder per
adresse per time gjelder i tillegg.

**§2.1 — nøkkeldeponering.** `PUT/GET/DELETE /v1/backup/key-escrow`, av som
standard. Materialet krypteres i ro med AES-256-GCM, nøkkel avledet med
HKDF-SHA256 fra `BACKUP_ESCROW_SECRET`, med bruker-ID-en som AAD.

**§2.1 — `BACKUP_ESCROW_SECRET_OLD` og nøkkel-ID per rad.** Rader som ikke åpnes
av den gjeldende hemmeligheten prøves med den forrige, og krypteres om ved
første lesing. `GET /health` rapporterer `escrow`: `"av"`, `"ok"`, eller
`"N rader paa annen hemmelighet"`.

**§7 — skadedata fikk egen bryter.** Erstattet «skadedata lagres aldri»:

> ~~**Skadedata lagres aldri.**~~ *Endret 2026-08-06:* skadedata har fått
> sin **egen bryter** (`share_injury_data`, av som standard). «Private som
> standard» er oppfylt av standardverdien og av at det kreves et aktivt
> valg — ikke av at det er umulig. Ettersøksdata er den mest verdifulle
> delen av materialet: hvor ofte dyr skadeskytes, hvor langt de går, om de
> blir funnet. Bryteren står **for seg selv**, ikke sammen med art og sted,
> fordi «jeg skjøt stående på 85 meter» og «dyret ble skadeskutt og aldri
> funnet» ikke er samme opplysning å dele. Felt på tillatelseslista:
> `wounded`, `injury`, `hit_placement`, `shots_fired`,
> `tracking_distance_m`, `tracking_time_min`, `dog_used`, `recovered`.

**§8 — `bf_version()` løst.** Kjernen eksponerer nå en semver-streng, og
`core_version` i donasjonene kommer derfra i stedet for fra appens
`versionName`. Ingen backend-endring var nødvendig; kolonnen tok imot den
uendret.

## 2026-08-05

**§11, fase 8 — push.**

> `PUT /v1/devices` (idempotent registrering), `GET /v1/devices` (uten
> `push_token` i svaret), `POST /v1/devices/unregister`.
> Utsending skjer i `teamgov.varsle`, som er eneste stedet §11-varsler oppstår —
> køraden legges inn **først**, push er best effort og kastes aldri oppover.
> FCM HTTP v1 tar én mottaker per kall, så et stort lag blir mange kall; derfor
> et samlet tidsbudsjett (`PUSH_BUDGET_SECONDS`) som avbryter resten. Det er
> ikke datatap — køen bærer meldingen. Døde tokens (`UNREGISTERED`) slettes.
> Uten `FCM_SERVICE_ACCOUNT_JSON` logges push bare, og køen står alene.

*Merk: «det er ikke datatap — køen bærer meldingen» forutsetter at klienten
henter køen. Det gjorde ingen før klientens v0.19 (2026-08-08, issue #4), så
påstanden var udekket i tre døgn. Se `KONTRAKT.md` §9 for hva garantien dekker
nå — oppstart til oppstart, ikke sanntid.*

**§3 — `kills[]` avklart, løst som flyktig kunngjøring.** Feltet kunne ikke
leveres som en liste; `POST /v1/hunts/announce` erstatter det.

**§3 — `trend` definert** som snitt per skudd i de siste ~20 skuddene minus de
~20 foregående, talt i skudd og ikke i serier.

## 2026-08-04

**§1, fase 3 — innlogging.** Endepunkttabellen for `/v1/auth/*`, tokenparet, og
`Authorization: Bearer` på alle andre endepunkter.

**§7, fase 7 — delingsvalg for forskning.**

> `GET/PUT /v1/research/sharing` leser og setter valgene, og **serveren
> filtrerer innkommende jakt-payload etter dem**
> (`services/research_filter.py`). Samtykket sier at resultattypen kan deles;
> delingsvalgene sier hva *av* den. Begge må gjelde.

**§9 — kontosletting.**

> de to lagrene ryddes ulikt, og må gjøre det.
> Brukerskjemaet tømmes med det samme — serier, treff, backup, venner,
> lagmedlemskap, stemmer, enheter og innlogginger er borte når kallet
> returnerer. Forskningsskjemaet kan vi *ikke* røre herfra: radene er
> pseudonymiserte, og §7 forbyr koblingen tilbake. I stedet legges det inn en
> sletteanmodning på pseudonymet, og alle samtykker trekkes tilbake med én
> gang så ingenting nytt kommer inn mens anmodningen behandles.

## 2026-08-03

**§2 — backup-bloben.**

> bloben sendes som rå `application/octet-stream` med
> metadataene som query-parametere (sparer base64-påslaget på vår største
> nyttelast). `GET /v1/backup/meta` gir metadata uten å laste ned bloben, slik at
> «har jeg noe å gjenopprette?» på en ny telefon er et lite kall.
> **Tillegg til konfliktløsningen:** serveren avviser en `PUT` der `client_ts` er
> eldre enn den lagrede (409). Last-write-wins per post-ID kan bare håndheves
> klient-side — serveren ser ikke inn i den krypterte bloben — så uten dette
> vernet kunne en telefon som synker første gang på måneder viske ut alt som er
> logget siden. `?force=true` overstyrer ved et bevisst brukervalg
> («gjenopprett fra denne enheten»). Grense: 16 MB.

**§3 — moderasjon av visningsnavn.**

> regelsettet håndhever tegnsett og lengde (speiler
> `Ui.nameFilters()` — klientfilteret er bekvemmelighet, ikke sikkerhet) pluss en
> ordliste satt med `DISPLAY_NAME_BLOCKLIST`. Ordlista sammenlignes på en foldet
> form (uten aksenter, tegnsetting og store bokstaver), så «S-t-y-g-t» ikke
> slipper unna. Avvist navn **lagres ikke i det hele tatt** — da kan det heller
> ikke lekke. Er navnet ikke godkjent, eksponeres «Ukjent skytter» for andre.
> Den manuelle køen krever en admin-flate som ikke finnes ennå; navn som passerer
> regelsettet godkjennes derfor direkte.

**§4 — invitasjonslenken.**

> `GET /i/{token}` leser User-Agent og svarer 302 til
> Play/App Store. Den svarer likt **uansett om tokenet finnes** — lenken deles i
> åpne kanaler, og et svar som skilte gyldig fra ugyldig ville gjort den til et
> oppslagsverk over hvilke lag som eksisterer. Telefonnumre normaliseres til
> E.164 (norske 8-sifrede får +47). Siden SMS er utsatt til v2, får en
> telefoninvitasjon `delivery_status: failed` **med lenken vedlagt**, slik at
> klienten kan dele den via ACTION_SEND i stedet.

**§11 — meldingskøen.**

> meldingskøen er `GET /v1/messages` +
> `POST /v1/messages/ack`. Kvittering **markerer** raden som levert i stedet for
> å slette den, så en klient som krasjer mellom henting og visning ikke mister
> meldingen. Køen erstatter ikke push — push når brukeren mens appen er lukket,
> køen er garantien for at meldingen når fram til slutt.

## 2026-08-02

**§7 — forskningsskjemaet.**

> tabellene ligger i et eget Postgres-**skjema**
> (`research.consents`, `research.records`, `research.deletion_requests`) uten
> fremmednøkler til brukertabellene. Pseudonymet **avledes** med
> HMAC-SHA256(server-hemmelighet, bruker-UUID) i stedet for å lagres i en
> oppslagstabell — en slik tabell ville vært nettopp den reversible koblingen
> denne paragrafen forbyr. Konsekvens: hemmeligheten kan ikke roteres uten å
> bryte koblingen til allerede innsamlede forskningsdata.

**§3.1 — bruker-ID-en.**

> (`backend/app/services/ids.py`): **8 signifikante
> tegn** — 7 tilfeldige + sjekksiffer — vist som `BF-XXXX-XXXX`, altså akkurat
> eksempelet over. Det gir ~3,4 · 10¹⁰ ID-er; ett tegn kortere enn de «9 tegn»
> teksten nevner, men fortsatt langt over det gjettingsargumentet krever, og
> lettere å lese opp. Sjekksifferet er `sum mod 32` i samme alfabet (ikke
> Crockfords mod-37-variant, som ville trukket inn symboler utenfor alfabetet).
> Innlesing folder I/L→1 og O→0, så vanlige lesefeil godtas.

**§0.1 — infrastrukturen etablert.** App `bestefar-api` på Fly.io,
`https://bestefar-api.fly.dev`. Database: Supabase-prosjekt `Bestefar_base`.
