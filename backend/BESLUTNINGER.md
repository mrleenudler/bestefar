# Beslutninger — backend

Én post per designbeslutning: **kontekst**, **valg**, og **hva som ble forkastet
og hvorfor**.

**Alt her er ekstrahert fra `backend_spec.md` og fra koden, 2026-08-07.**
Ingenting er rekonstruert. Der en beslutning er tatt uten at begrunnelsen finnes
skrevet ned, står det **«begrunnelse ikke dokumentert»** — en gjettet begrunnelse
er verre enn ingen, fordi den ikke lenger kan skilles fra eierens egen.

Kilden står på hver post så påstanden kan etterprøves.

---

## B-01 To tokentyper i stedet for én

**Kontekst.** Klienten trenger en økt som varer, men hvert API-kall må kunne
autoriseres billig.

**Valg.** Access-token er et kortlivet JWT (HS256, 60 min) som verifiseres med
signatur alene — ingen databaseoppslag per forespørsel. Refresh-token er 32
tilfeldige byte, ugjennomsiktige, lagret som SHA-256.

**Forkastet: JWT også til refresh.** Et refresh-token må kunne trekkes tilbake
ved utlogging og tyveri, og da trengs uansett en rad i basen. Er raden der, er
det enklere at tokenet ikke sier noe i seg selv.

**Prisen som ble akseptert:** access-tokenet kan ikke tilbakekalles. Derfor er
levetiden kort, og `deleted_at` sjekkes ved hvert kall.

*Kilde: `services/tokens.py`, modul-docstring; `deps.py`.*

## B-02 Gjenbrukt refresh-token tilbakekaller alle økter

**Kontekst.** Rotasjon betyr at et brukt refresh-token aldri skal dukke opp
igjen. Gjør det likevel det, er det enten en kopi på avveie eller et kall som
ble kjørt to ganger.

**Valg.** Alle brukerens økter tilbakekalles.

**Forkastet: å skille de to tilfellene.** Serveren har ingen måte å gjøre det
på. Da velges det trygge.

**Følgen, som er verdt å si høyt:** normal bruk kan utløse dette hvis klienten
kjører to fornyelser parallelt. Serveren har altså en avhengighet til klientens
serialisering — `android/KONTRAKT.md` §6.

*Kilde: `routers/auth.py`, `forny`; `backend_spec.md` §1.*

## B-03 Ingen standardverdi for `JWT_SECRET`

**Kontekst.** Tjenesten må kunne starte uten at alt er konfigurert.

**Valg.** Tom `JWT_SECRET` ⇒ alle `/v1/auth/*` svarer 503. Kortere enn 32 byte
avvises også (RFC 7518 §3.2).

**Forkastet: en standardverdi i repoet.** Den ville gjort hvert eneste token
forfalskbart av hvem som helst som leser koden. 503 er riktig svar — tjenesten
er feilkonfigurert, ikke klienten.

*Kilde: `services/tokens.py`, `krev_hemmelighet`; `config.py`.*

## B-04 `/v1/auth/email/start` svarer alltid 202

**Kontekst.** Endepunktet tar imot en e-postadresse og sender en engangskode.

**Valg.** 202 uansett om adressen finnes fra før, og også når selve
e-postutsendingen feiler (feilen logges).

**Forkastet: å svare forskjellig for kjent og ukjent adresse.** Det ville gjort
endepunktet til et oppslagsverk over hvem som bruker appen. Samme resonnement
gjelder utsendingsfeil: en annen respons der ville også vært et signal.

*Kilde: `routers/auth.py`, `send_kode`, docstring og `except`-grenen.*

## B-05 Kvoter for e-postkoder ligger i databasen

**Kontekst.** Fly kjører to maskiner. En teller per prosess halverer effekten av
enhver grense.

**Valg.** Både timekvoten (`EMAIL_CODE_RATE_PER_HOUR`, 10) og sperrefristen
(`EMAIL_CODE_RESEND_COOLDOWN_SECONDS`, 60 s) telles på rader i
`email_login_codes`.

**Forkastet: `ratelimit.py` (glidende vindu i minnet).** Den er fortsatt i bruk
for `/v1/feedback`, og er derfor kjent unøyaktig med to maskiner — se ÅP-B9.

**Merk hva kvoten teller:** *forespørsler*, ikke leveranser. Derfor er den satt
romslig — en bruker som ber om ny kode fordi e-posten er treg, skal ikke bli
utestengt. Vernet mot gjetting er de fem forsøkene, ikke antall koder.

*Kilde: `routers/auth.py`, `send_kode`; `config.py`; `ratelimit.py`.*

## B-06 Kontosammenslåing kun på verifisert e-post

**Kontekst.** Samme person kan logge inn med Google én gang og e-postkode neste
gang, og skal ha én konto.

**Valg.** Kobling skjer på `email` når leverandøren har markert den som
verifisert.

**Forkastet: å koble på uverifisert adresse.** Da kunne en angriper laget en
konto hos en slapp leverandør med en annen persons adresse og overtatt kontoen
deres.

*Kilde: `routers/auth.py`, modul-docstring og `_finn_eller_lag_bruker`.*

## B-07 409 på eldre `client_ts` i backup

**Kontekst.** Spec §2 sier last-write-wins per post-ID. Serveren kan ikke
håndheve det — den ser ikke inn i den klient-krypterte bloben.

**Valg.** `PUT /v1/backup` avviser en blob hvis `client_ts` er **eldre** enn den
lagrede. `?force=true` overstyrer ved et bevisst brukervalg. Likt `client_ts`
godtas.

**Forkastet: å stole på klientsidens konfliktløsning alene.** Uten vernet ville
en telefon som synker første gang på måneder visket ut alt som er logget siden.

**Hvorfor likt tidsstempel godtas:** en omprøving etter et avbrutt kall skal
ikke feile.

*Kilde: `routers/backup.py`, `upload_backup`, docstring.*

## B-08 `DELETE /v1/backup` beholder deponert nøkkel

**Kontekst.** «Slett sikkerhetskopien» fjerner bloben. Deponeringen er en
separat innstilling brukeren har slått på.

**Valg.** Bare bloben slettes. Nøkkelen blir stående.

**Forkastet: å slette begge.** Da ville neste opplasting stilltiende vært
udeponert med bryteren stående på. En sletting skal ikke i stillhet skru av en
innstilling. `DELETE /v1/account` fjerner begge deler.

*Kilde: `routers/backup.py`, `delete_backup`, docstring.*

## B-09 Deponert nøkkelmateriale krypteres i ro, med hemmelighet utenfor basen

**Kontekst.** Deponering er det eneste tilfellet der serveren kan åpne brukerens
blob.

**Valg.** AES-256-GCM, nøkkel avledet med HKDF-SHA256 fra
`BACKUP_ESCROW_SECRET`, med bruker-ID-en som AAD. Hemmeligheten ligger som
Fly-secret — et annet sted enn databasen.

**Hva det kjøper:** en Supabase-dump alene gir ingen nøkler, og en rad kan ikke
flyttes fra én bruker til en annen (AAD-en binder den).

**Forkastet: envelope-kryptering med en KMS.** Flytter holdbarhetsproblemet til
en leverandør med bedre SLA, men innfører en avhengighet og en kostnad — og en
KMS-nøkkel har samme egenskap: slettes den, er den borte.

**Forkastet: å lagre nøkler i klartekst når hemmeligheten mangler.** Endepunktet
svarer 503 i stedet. Vi lagrer heller ingenting.

*Kilde: `backend_spec.md` §2.1; `services/escrow.py`; `routers/backup.py`.*

## B-10 `BACKUP_ESCROW_SECRET_OLD` og nøkkel-ID per rad

**Kontekst.** Hemmeligheten i B-09 er selv et enkeltpunkt. Mistes den, er ikke
dataene tapt — bloben kan fortsatt åpnes med gjenopprettingskoden — men
funksjonen degraderer til «som om deponering aldri var slått på», og det treffer
skjevt: nettopp de som slo den på, er de som minst sannsynlig tok vare på koden.
Den realistiske trusselen er en `flyctl secrets set` for mye, ikke at Fly går
under.

**Valg, to grep.** (1) Rader som ikke åpnes av den gjeldende hemmeligheten prøves
med den forrige, og **krypteres om ved første lesing** — en utskiftning migrerer
seg selv. (2) En KCV per rad (HMAC over en konstant streng under den avledede
nøkkelen) som `/health` teller avvik på.

**Forkastet: en advarsel i en kommentar.** Den sto der først — «roteres ALDRI
uten en plan» — og er ikke et tiltak. Lager du et enkeltpunkt, bygger du
utskiftningsveien i samme runde.

**Forkastet: en engangsjobb for omkryptering.** Ingen husker å kjøre den.

*Kilde: `backend_spec.md` §2.1; `services/escrow.py`; `routers/health.py`,
`_escrow_status`.*

## B-11 `DELETE /v1/backup/key-escrow` virker uten hemmeligheten

**Kontekst.** PUT og GET svarer 503 når `BACKUP_ESCROW_SECRET` mangler.

**Valg.** DELETE gjør det ikke. Den krever ingen konfigurasjon og er idempotent.

**Forkastet: symmetri.** Et 503 her ville låst brukeren inne i et personvernvalg
hen vil ut av. Veien *ut* skal aldri kunne feile på en driftsinnstilling.

*Kilde: `routers/backup.py`, `slett_noekkel`, docstring.*

## B-12 Forsknings-pseudonymet avledes, det lagres ikke

**Kontekst.** §7 forbyr en reversibel kobling mellom konto og forskningsdata.

**Valg.** HMAC-SHA256(hemmelighet, `user_id`), avkortet til 160 bit, base32 uten
padding.

**Forkastet: en oppslagstabell.** Den *ville vært* nettopp den reversible
koblingen paragrafen forbyr. Med HMAC inneholder forskningsskjemaet kun
pseudonymet, og veien tilbake finnes ikke uten hemmeligheten — som ligger i Fly
secrets, ikke i databasen.

**Prisen som ble akseptert:** `RESEARCH_PSEUDONYM_SECRET` kan ikke roteres uten
at eksisterende forskningsdata mister koblingen til nye innsendinger fra samme
bruker.

*Kilde: `services/pseudonym.py`, modul-docstring; `backend_spec.md` §7.*

## B-13 Tillatelsesliste, ikke forbudsliste, i forskningsfilteret

**Kontekst.** Det konkrete feltinnholdet i forskningsdatasettet er ikke avklart
(`TODO(eier)`, ÅP-K2).

**Valg.** Jakt-payload matches mot en eksplisitt liste kanoniske nøkler. Ukjente
nøkler droppes stille.

**Forkastet: en forbudsliste.** Med den ville hvert *nytt* felt vært delt som
standard, helt til noen kom på å forby det. Når feltlista foreligger, er
`research_filter.py` stedet den utvides.

**Hvorfor filteret er server-side:** klienten filtrerer også, men det er
bekvemmelighet, ikke vern — en modifisert klient kan sende hva som helst.

*Kilde: `services/research_filter.py`, modul-docstring.*

## B-14 Posisjonsgrovhet velger felt, ikke avrunding

**Kontekst.** Brukeren velger hvor grovt stedet deles: `exact`, `kommune`,
`fylke`, `none`.

**Valg.** Grovheten bestemmer **hvilke stedsfelt** som lagres.

**Forkastet: å avrunde koordinatene.** Serveren har ingen kommune- eller
fylkesgrenser å slå opp i, og en avrunding ville uansett ikke vært det brukeren
valgte — «kommune» er et navn, ikke et antall desimaler. Klienten kjenner stedet
og sender navnet; serveren håndhever hva som lagres.

*Kilde: `services/research_filter.py`, `_tillatte_noekler`.*

## B-15 Uten datosamtykke beholdes året

**Kontekst.** `captured_at` er en obligatorisk kolonne.

**Valg.** Uten `share_date` settes datoen til 1. januar samme år.

**Forkastet: å avvise hele innsendingen.** Året alene sier ikke når noen var på
jakt, men lar materialet grupperes per sesong — som er hele poenget med å ta
imot raden.

*Kilde: `services/research_filter.py`, `filtrer_tidspunkt`, docstring.*

## B-16 Skadedata: fra «aldri» til egen bryter

**Kontekst.** §7 sier skadedata er private som standard. Det var implementert
som «ingen bryter, aldri lagret».

**Valg.** Egen bryter `share_injury_data`, av som standard, som står **for seg
selv** — ikke sammen med art og sted.

**Hva som var galt med det forrige:** «privat som standard» uten en bryter er
«aldri», og det gjorde den mest verdifulle delen av materialet umulig å samle
inn. Et datasett uten skadeskytingene kan ikke brukes til å si noe om
skadeskyting. Standardverdien `False` oppfyller kravet; umulighet er noe annet.

**Hvorfor bryteren står alene:** «jeg skjøt stående på 85 meter» og «dyret ble
skadeskutt og aldri funnet» er ikke samme opplysning å dele om seg selv.
Delingsvalg med ulik sosial kostnad slås ikke sammen.

**Uavklart:** eieravklaringen i §7 ber om bryternavnet `share_wound_data` og
feltene `outcome`/`follow_up`/`ran_m`. Implementasjonen har andre navn — ÅP-B2.

*Kilde: `backend_spec.md` §7; `services/research_filter.py`, `SKADEDATA`.*

## B-17 `kills[]` ble en flyktig kunngjøring

**Kontekst.** Venne-modellen i §3 hadde `kills[]`. Jaktloggen ligger inne i den
klient-krypterte backup-bloben, som serveren ikke kan lese.

**Valg.** `POST /v1/hunts/announce` sender «{navn} har felt {art} i {kommune}»
som push til vennene, og så er den borte. Kun `users.hunt_announced_at` lagres —
et tidsstempel, for å bremse gjentakelser.

**Forkastet: å synke jaktposter som egne rader.** Det ville lagt hele
jaktloggen i klartekst hos oss — nettopp det §2 unngår.

**Konsekvensen som er tilsiktet:** når en venn ikke hadde telefonen på, gikk
meldingen tapt. Dette er en gladmelding i øyeblikket, ikke en logg. Derfor
brukes `push.send` direkte og ikke `teamgov.varsle`, som alltid legger inn en
kørad først.

*Kilde: `routers/hunts.py`, modul-docstring; `backend_spec.md` §3.*

## B-18 Push feiler aldri oppover

**Kontekst.** §11-varsler har to kanaler: meldingskøen i basen, og push.

**Valg.** Køraden legges inn **først**. Push er best effort; alle feil logges og
svelges.

**Forkastet: å la push-feil propagere.** Da ville en nede FCM-tjeneste gjort det
umulig å bytte lagnavn.

**Samme resonnement gjelder `PUSH_BUDGET_SECONDS`:** når budsjettet er brukt
opp, droppes resten av mottakerne. Det er ikke datatap — køen er garantien.

**Premisset var ikke innfridd før 2026-08-08.** Klienten hentet ikke
`/v1/messages`, så push var i praksis eneste leveringsvei, og et avbrutt
budsjett *var* tap for de mottakerne som aldri ble forsøkt. Fra v0.19 leses køen
ved appstart (`android/KONTRAKT.md`, issue #4), og begrunnelsen holder igjen.

**Med den presiseringen den alltid burde hatt:** køen hentes ved *oppstart*,
ikke løpende. For en melding som oppstår mens appen står åpen, er push den
eneste raske veien, og §11-meldingene med 7-dagers frist tåler ikke ubegrenset
utsettelse. «Køen bærer meldingen» er sant, men den bærer den til neste
appstart — ikke til nå.

*Kilde: `services/push.py`, modul-docstring og `send`; `backend/KONTRAKT.md`
§4 og §9; `android/app/src/main/java/no/bestefar/app/Messages.kt`.*

## B-19 Ikke `google-auth` for FCM

**Kontekst.** FCM HTTP v1 krever et OAuth-token fra en tjenestekonto.

**Valg.** Signere JWT-en selv med PyJWT, som allerede lå der for §1.

**Forkastet: `google-auth`-biblioteket.** Vi trenger nøyaktig én ting derfra.
Ett avhengighetstre mindre å holde oppdatert.

*Kilde: `services/push.py`, modul-docstring; `_access_token`.*

## B-20 `push_token` er nøkkelen, ikke brukeren

**Kontekst.** En telefon kan bytte konto.

**Valg.** `PUT /v1/devices` slår opp på `push_token`. Finnes raden under en
annen bruker, **flytter den seg**.

**Forkastet: én rad per (bruker, token).** Da ville varsler til den forrige
brukeren havnet på en telefon som nå er logget inn som noen andre.

**Beslektet:** døde tokens (`UNREGISTERED`, `INVALID_ARGUMENT`, `NOT_FOUND`)
slettes — ellers vokser tabellen med adresser vi aldri når, og hver av dem
koster et kall av budsjettet ved neste varsel.

*Kilde: `routers/devices.py`, `register_device`, docstring; `services/push.py`,
`_DODE_KODER`.*

## B-21 `GET /v1/devices` returnerer ikke push-tokenet

**Kontekst.** Endepunktet finnes til «innlogget på disse enhetene» i UI-et.

**Valg.** Tokenet utelates fra svaret.

**Forkastet: å ta det med.** Det gir ingen nytte i klienten, og et svar er et
sted det kan lekke.

*Kilde: `routers/devices.py`, `list_devices`, docstring.*

## B-22 `/health` svarer alltid 200

**Kontekst.** Fly bruker helsesjekken til å avgjøre om en deploy skal rulles
tilbake.

**Valg.** 200 så lenge prosessen lever. Databasens tilstand står i kroppen.

**Forkastet: å speile DB-tilstanden i statuskoden.** Da ville en midlertidig
DB-feil fått Fly til å rulle tilbake en ellers frisk deploy.

**Beslektet:** `database` rapporterer «feilkonfigurert» hvis `ENV=prod` og
`DATABASE_URL` peker på SQLite — ellers ville `SELECT 1` svart «ok» på en
containerfil som forsvinner ved omstart.

*Kilde: `routers/health.py`, modul-docstring og `_database_status`.*

## B-23 Engine lages ved første bruk, ikke ved import

**Kontekst.** Fly starter containeren og helsesjekker den.

**Valg.** `create_engine` kalles lazily i `engine()`.

**Forkastet: å lage den ved import.** Da dør prosessen ved oppstart hvis
databasen er nede eller `DATABASE_URL` mangler, og Fly gir en deploy-loop uten
forklaring.

*Kilde: `app/db.py`, modul-docstring.*

## B-24 Fremmednøkler håndheves også på SQLite

**Kontekst.** SQLite har `PRAGMA foreign_keys` av som standard. Postgres har
ikke noe valg.

**Valg.** En `connect`-lytter slår den på for SQLite-engines.

**Forkastet: å la det stå.** Uten den er en hel feilklasse usynlig lokalt og
dukker først opp i CI mot Postgres — slik den gjorde 2026-08-03, da
`sharing_preferences` ble satt inn før `users` uten at noen testkjøring på
SQLite reagerte.

*Kilde: `app/db.py`, `_enforce_sqlite_foreign_keys`, docstring.*

## B-25 Forskningstabellene i eget Postgres-skjema

**Kontekst.** §0 krever strukturell adskillelse, ikke bare egne tabellnavn.

**Valg.** Skjemaet `research`, uten fremmednøkler til brukertabellene. På SQLite
oversettes skjemaet til `None` (`schema_translate_map`), siden SQLite ikke har
skjemaer.

**Konsekvens som må håndteres:** Alembic-autogenerate blir ubrukelig mot SQLite
— den ser ikke `research` og rapporterer hele skjemaet som slettet. Migrasjonene
skrives derfor for hånd, og `tests/test_migrations.py` kjører bare mot Postgres.

*Kilde: `app/db.py`, `schema_translate_map` og `alembic_compare_opts`;
`backend/README.md`.*

## B-26 Karantenen i basen, rate-limiteren i minnet

**Kontekst.** §3.1 krever karantene etter gjentatte mislykkede søk.

**Valg.** `services/quarantine.py` lagrer i databasen. `ratelimit.py`
(glidende vindu i minnet) beholdes for `/v1/feedback`.

**Hvorfor forskjellen:** karantenen skal overleve omstart og gjelde på tvers av
begge Fly-maskinene. Feedback-kvoten gjør ikke det, og er derfor kjent
unøyaktig — ÅP-B9.

**Bare bom telles.** Treff er normal bruk. En sjekksifferfeil i en bruker-ID gir
422 uten oppslag og teller ikke som bom — det er en tastefeil, ikke et forsøk på
å finne noen.

*Kilde: `routers/friends.py`, modul-docstring og `soek`; `ratelimit.py`,
modul-docstring.*

## B-27 `/i/{token}` svarer likt uansett om tokenet finnes

**Kontekst.** Invitasjonslenken er den samme som QR-koden, og deles i åpne
kanaler.

**Valg.** 302 til riktig butikk basert på User-Agent, uansett.

**Forkastet: 404 på ukjent token.** Det ville gjort lenken til et oppslagsverk
over hvilke lag som eksisterer.

*Kilde: `routers/teams.py`, `invite_redirect`, docstring.*

## B-28 SMS-invitasjon returnerer `failed` med lenken vedlagt

**Kontekst.** Det finnes ingen SMS-leverandør (utsatt til v2, ÅP-E9).

**Valg.** `delivery_status: failed` med `url` i svaret.

**Forkastet: å avvise telefoninvitasjoner.** Med lenken i svaret kan klienten
dele den via ACTION_SEND i stedet, og brukeren merker ikke at vi ikke sendte
noe selv.

*Kilde: `routers/teams.py`, `invite`.*

## B-29 Brukerraden tømmes, den slettes ikke

**Kontekst.** `DELETE /v1/account` (§9).

**Valg.** Alt personidentifiserende nulles, `deleted_at` settes — men raden blir
stående med `public_id` intakt.

**Forkastet: å slette raden.** Da kunne `public_id` gjenbrukes av en ny konto,
og en venn som fortsatt har ID-en lagret ville plutselig sett en fremmed.

**Rekkefølgen er bevisst:** sletteanmodningen i forskningsskjemaet legges inn
**først**, fordi den er det eneste vi ikke kan gjenskape hvis noe feiler
underveis. Samtykkene trekkes samtidig, så ingenting nytt kommer inn mens
anmodningen behandles.

**Merk:** FK-ene har `ondelete=CASCADE`, men siden brukerraden ikke slettes,
fyrer ingen cascade. Alt må slettes eksplisitt.

*Kilde: `routers/account.py`, modul-docstring, `slett_konto` og
`_slett_brukerdata`.*

## B-30 `X-Debug-User-Id` beholdt, men død i produksjon

**Kontekst.** Testene for §2–§11 ble skrevet før innloggingen fantes.

**Valg.** Headeren godtas fortsatt når `ENV != prod`. Sjekken på `is_prod` står
først, og det finnes ingen konfigurasjon som slår den på i produksjon.

**Forkastet: å fjerne den.** Den gjør det mulig å prøvekjøre endepunkter lokalt
uten å sette opp Google-klienter.

*Kilde: `deps.py`, modul-docstring og `current_user`.*

## B-31 Feilanalyse-endepunktet krever ikke innlogging

**Kontekst.** §6 knytter bildedonasjon til bildedelings-samtykket.

**Valg.** `POST /v1/failed-analyses` er åpent.

**Forkastet: å kreve konto.** Donasjonen skal fungere også for brukere uten
konto — samtykket er knyttet til bildedeling, ikke til innlogging.

*Kilde: `routers/failed_analyses.py`, modul-docstring.*

## B-32 Bildet ligger i databasen inntil R2 kobles inn

**Kontekst.** §6 og §0.1 er tydelige på at bilder aldri lagres i databasen. R2
er opprettet, men ikke koblet inn.

**Valg.** Bildet lagres midlertidig i kolonnen `image_legacy`, og endepunktet
avviser filer over `MAX_UPLOAD_BYTES` så basen ikke fylles opp.

**Dette er en kjent avvikelse fra speccen**, ikke en beslutning om å bli
værende. Kolonnenavnet sier det. ÅP-B5.

*Kilde: `routers/failed_analyses.py`, modul-docstring.*

## B-33 Enum-kolonner er VARCHAR + CHECK

**Kontekst.** Modellene bruker `StrEnum` flere steder.

**Valg.** `native_enum=False`.

**Forkastet: Postgres-native ENUM.** Den krever `ALTER TYPE`-migrasjoner ved
hver utvidelse og finnes per skjema. Ikke verdt det.

*Kilde: `backend/README.md`, «Datamodell»; `models/ops.py` m.fl.*

## B-34 Serie-ID er klientens egen UUID

**Kontekst.** Klienten køer usendte serier og kan sende samme serie flere
ganger.

**Valg.** Klientens UUID er primærnøkkel, så opplasting blir idempotent.

*Begrunnelse for å forkaste alternativet (server-generert ID) er ikke
dokumentert, men følger av idempotenskravet.*

*Kilde: `backend/README.md`, «Datamodell».*

## B-35 Bloben sendes som rå oktettstrøm, ikke base64 i JSON

**Kontekst.** Backup-bloben er den største nyttelasten appen har (grense 16 MB).

**Valg.** `PUT /v1/backup` tar `application/octet-stream` med metadataene som
query-parametere.

**Forkastet: base64 i en JSON-kropp.** Det ville lagt ~33 % på nettopp den
forespørselen som er størst, og metadataene er få og enkle nok til å ligge i
URL-en.

*Kilde: `backend_spec.md` §2; `routers/backup.py`, `upload_backup`.*

## B-36 Et avvist visningsnavn lagres ikke i det hele tatt

**Kontekst.** Visningsnavnet er det eneste feltet som alltid deles med venner,
og dermed det eneste stedet en bruker kan skrive fritt til andre.

**Valg.** Moderasjonen kjører før navnet lagres. Blir det avvist, skrives det
ikke til basen — brukeren får en begrunnelse og må velge et annet.

**Forkastet: å lagre navnet med status `rejected`.** Et navn som ikke finnes,
kan heller ikke lekke — verken gjennom en feil i visningskoden, en databasedump
eller en senere endring som glemmer å filtrere på status. Er navnet ikke
godkjent, eksponeres «Ukjent skytter» for andre.

*Kilde: `backend_spec.md` §3; `services/moderation.py`, modul-docstring.*

## B-37 Ordlista sammenliknes på foldet form

**Kontekst.** `DISPLAY_NAME_BLOCKLIST` er en enkel ordliste, tom som standard.

**Valg.** Både navnet og hvert ord i lista foldes før sammenlikning: små
bokstaver, aksenter fjernet, alt som ikke er alfanumerisk strippet bort.

**Forkastet: direkte delstrengsammenlikning.** Uten foldingen slipper
«B-a-n-n-e» unna et treff på «banne», og lista blir et spill om skrivemåter i
stedet for et filter.

**Beslektet valg:** lista er tom som standard. En hardkodet norsk banneordliste
ville vært både ufullstendig og umulig å vedlikeholde fra repoet.

*Kilde: `services/moderation.py`, `_fold` og modul-docstring.*

## B-38 Telefonnumre normaliseres til E.164

**Kontekst.** Invitasjoner og brukersøk tar imot telefonnumre skrevet på
vilkårlig form.

**Valg.** Skilletegn strippes, og et norsk åttesifret nummer får `+47`.
Lengdekontroll 8–15 sifre etter landkode.

**Hvorfor det betyr noe her:** søk gir kun *eksakt* treff (B-26). Uten
normalisering ville «912 34 567» og «+4791234567» vært to forskjellige
brukere, og et legitimt søk ville telt som bom mot karantenen.

*Kilde: `backend_spec.md` §4; `services/contacts.py`, `classify`.*

---

# Beslutninger uten dokumentert begrunnelse

Disse verdiene og valgene står i koden uten at det er skrevet ned *hvorfor*
akkurat den verdien. De er ikke nødvendigvis feil — men de kan ikke forsvares
fra dokumentasjonen, og de bør ikke endres på gjetning heller.

| Beslutning | Hvor | Hva som mangler |
|---|---|---|
| `ACCESS_TOKEN_TTL_MINUTES = 60` | `config.py` | *At* levetiden skal være kort er begrunnet (tokenet kan ikke tilbakekalles). Hvorfor 60 og ikke 15 eller 240, er ikke dokumentert. |
| `REFRESH_TOKEN_TTL_DAYS = 90` | `config.py` | Begrunnelse ikke dokumentert. |
| `EMAIL_CODE_TTL_MINUTES = 15`, `EMAIL_CODE_MAX_ATTEMPTS = 5` | `config.py` | Kommentaren begrunner *sekssifret kode* ut fra at den er kortlivet og har få forsøk — men ikke de to tallene selv. |
| `FEEDBACK_RATE_PER_HOUR = 5` per IP | `config.py` | Begrunnelse ikke dokumentert. Merk at den reelle grensen er 10 med to maskiner (ÅP-B9). |
| `MAX_UPLOAD_BYTES = 8 MB` | `config.py` | Formålet er dokumentert («så databasen ikke fylles opp»), tallet ikke. |
| `PUSH_TIMEOUT_SECONDS = 5`, `PUSH_BUDGET_SECONDS = 6` | `config.py` | *Hvorfor* det finnes et budsjett er godt begrunnet. Tallene er det ikke. **Målt 2026-08-07:** budsjettet tåler nøyaktig ett tregt kall, og veggtiden kan bli 11 s fordi sjekken bare skjer mellom kall. Issue #3. |
| Synkron utsending av push, inne i forespørselen | `services/push.py` | At det er et valg framgår av `PUSH_BUDGET_SECONDS`, men begrunnelsen for å ikke bruke en bakgrunnstråd står ikke i koden eller speccen. |
| Pseudonymet avkortes til 160 bit | `services/pseudonym.py` | «160 bit er rikelig» er en påstand, ikke en begrunnelse for avkortingen framfor full digest. |
| `app_store_url` peker på Play-siden | `config.py` | Markert som midlertidig («iOS er ikke publisert ennå»), men ingen beslutning om hva den skal peke på i mellomtiden for faktiske iOS-brukere. |
