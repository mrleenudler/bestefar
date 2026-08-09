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

## 2026-08-09

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
