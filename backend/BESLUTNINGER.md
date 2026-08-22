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

## B-06.1 Visningsnavnet hentes fra ID-tokenet, ikke fra klienten

**Kontekst.** En ny konto trenger et visningsnavn. `oidc.py` leste `sub`,
`email` og `email_verified` ut av de verifiserte kravene, men ikke `name` — så
lokaldelen av adressen ble visningsnavn også for Google-kontoer, der et ekte
navn lå rett ved siden av.

**Valg.** `name` leses fra de samme verifiserte kravene og brukes når det finnes.
Faller tilbake på lokaldelen av adressen, så på «Skytter».

**Forkastet: å la klienten sende navnet.** Credential Manager gir klienten
navnet, og det ville vært enklere. Men det bytter en signaturverifisert kilde mot
en klient-oppgitt, i **det ene feltet som alltid deles med venner** (§3). En
modifisert klient kunne satt hva som helst som andres første inntrykk.

**Navnet modereres som ethvert annet.** En leverandør garanterer ikke at
brukeren har valgt et navn vi vil vise. Her ble også en eksisterende svakhet
rettet: den utledede navnet gikk tidligere gjennom `moderation.review` med en
**tom** ordliste, altså uten `DISPLAY_NAME_BLOCKLIST`. Det var lite farlig så
lenge kilden var en e-postadresse brukeren allerede eide; det er noe annet når
kilden er et fritt valgt profilnavn.

*Kilde: issue #7; `services/oidc.py`, `Identitet.navn`; `routers/auth.py`,
`_foerste_navn`.*

## B-06.2 Økten husker identiteten den ble startet med

**Kontekst.** Klienten skal kunne vise «Logget inn med Google som
ola@example.com». Tokenparet inneholdt ingen adresse.

**Valg.** `auth_sessions.identity_id` peker på `AuthIdentity`-raden
innloggingen kom gjennom, og tokenparet svarer med `email` derfra — også ved
fornyelse.

**Forkastet: å la klienten lese adressen ut av ID-tokenet.** Kontosammenslåing
skjer på verifisert e-post (B-06), så kontoen kan være knyttet til en annen
adresse enn den man nettopp logget inn med. Skjermen ville løyet i akkurat det
tilfellet den finnes for.

**Forkastet: å returnere «kontoens e-post».** Etter en sammenslåing har kontoen
flere identiteter og ingen av adressene er mer «kontoens» enn de andre. Det
finnes ikke noe entydig svar å gi, og et vilkårlig valgt ett ville sett entydig
ut.

**Forkastet: å utlede den ved fornyelse.** Ved `/refresh` finnes ikke
ID-tokenet lenger. Uten kolonnen måtte serveren gjettet — og «Logget inn som»
ville tømt seg selv etter en time.

`SET NULL` og ikke `CASCADE`: forsvinner identiteten, skal ikke brukeren logges
ut. Hen mister bare adressen i visningen.

*Kilde: issue #8; migrasjon `c7e2b91f4d08`; `routers/auth.py`, `_start_oekt`.*

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

## B-18.1 Utsendingen ut av forespørselen — og loggen ned til avvik

**Kontekst.** Push ble opprinnelig sendt synkront inne i forespørselen, med
`PUSH_BUDGET_SECONDS` som vern mot at et stort lag skulle henge svaret.
Målingene i issue #3 viste at budsjettet ikke gjorde den jobben: sjekken skjer
bare *mellom* kall, så seks sekunders budsjett kunne bruke elleve.

**Valg.** Utsendingen flyttet til `BackgroundTasks`. Enhetene slås opp mens
sesjonen lever; bare HTTP-kallene og opprydding av døde tokens ligger etter
svaret. Budsjettet beholdes, men er nå en grense på *jobben* og ikke på
svartiden.

**Forkastet: å justere budsjettet.** Et budsjett som skal begrense svartid, må
kunne avbryte et kall som pågår — ikke bare la være å starte det neste. Det
krever timeout-håndtering per kall, og hele problemet forsvinner når ingen
venter.

**Prisen, som er akseptert:** `BackgroundTasks` kjører i samme prosess. Dør
maskinen mellom svar og utsending, går de gjenstående pushene tapt uten spor.
Køraden er committet før svaret, så meldingen kommer ved neste appstart. Det er
holdbart bare fordi push er nedgradert til varsel: fristen avgjør, resultatet er
meldingen som teller, og køen tar resten.

**Loggingen er samtidig strammet til to linjer.** Én når budsjettet avbryter
(med antall mottakere og antall forsøkt), og én ved en FCM-avvisning som *ikke*
er et dødt token — døde tokens er normal opprydding, ikke en feil. «Push er ikke
konfigurert» er flyttet fra INFO til DEBUG; den tilstanden står i `/health`, og
som INFO ville den lagt igjen en linje ved hver eneste varsling.

**Forkastet: en tabell eller teller over utsendinger.** Prosjektet er i alfa
uten reelle brukere. En tom logg er svaret vi trenger nå, og da må stillhet bety
noe — det gjør den bare hvis ingenting skriver til loggen uten grunn.

*Kilde: eieravklaring 2026-08-09 (issue #3); `services/teamgov.py`, `varsle` og
`send_og_rydd`; `services/push.py`, `send`.*

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

**Avsluttet 2026-08-15 av B-44.** Opplastingen er koblet inn. Teksten over blir
stående fordi den forklarer hvorfor `image_legacy` finnes i det hele tatt, og
hvorfor rader fra før den datoen ligger der de ligger.

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

**Omgjort 2026-08-12 (B-43).** Innvendingen over står seg — lista er fortsatt
ufullstendig og vanskelig å vedlikeholde. Men en tom liste fanger *ingenting*,
og «ufullstendig» ble i praksis til «ingen moderasjon i det hele tatt» i hver
eneste installasjon som ikke satte miljøvariabelen. Standardverdien er nå den
kurerte lista; miljøvariabelen utvider den.

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

## B-39 Kvorum på 25 %, målt ved avstemningens start

**Kontekst.** Et lederløst lag velger ny leder ved avstemning med 7-dagers
frist. Uten en nedre grense kan én stemme i et lag på tjue kåre en leder.

**Valg.** Minst 25 % av medlemstallet **da avstemningen startet** må ha stemt,
rundet opp. Tallet låses i `member_count_at_open` når avstemningen opprettes.

**Hvorfor terskelen finnes i det hele tatt:** eierens begrunnelse, og den er
bedre enn den prosedyremessige — *et lag som ikke får 25 % til å stemme, har et
kommunikasjonsproblem en avstemning ikke løser.* Kvorumet er ikke der for å
gjøre valget «gyldigere», men for å hindre at et lag får en leder det ikke vet
om at det har fått.

**Forkastet: å måle medlemstallet ved avgjørelse.** Da er terskelen
manipulerbar — man fjerner medlemmer til de gjenværende stemmene holder. Og den
som starter en avstemning i et lederløst lag er ofte den samme som kan fjerne
folk. En terskel som kan senkes av den som skal klare den, er ingen terskel.

**Forkastet: unntak for små lag.** 25 % av tre er 1, og det er akseptert. Et
unntak ville krevd en grense for «lite», og den grensen ville vært like
vilkårlig som kvorumet uten å ha begrunnelsen bak seg.

**Rader fra før kolonnen fantes har `member_count_at_open = 0`**, og `kvorum()`
gir da 0 — altså ingen krav. En avstemning som allerede løper skal ikke kunne
bli ugyldig av en migrasjon.

*Kilde: eieravklaring 2026-08-09; `services/teamgov.py`, `kvorum()` og
`resolve_election`; `models/social.py`, `TeamElection`.*

## B-40 Under kvorum gir `expired`, og ingen sperrefrist

**Kontekst.** Hva skjer når fristen går ut uten nok stemmer?

**Valg.** `expired` — nøyaktig samme utfall som uavgjort. Laget kan starte en ny
avstemning umiddelbart.

**Forkastet: et eget utfall for «for få stemmer».** Klienten ville måttet
håndtere en tredje tilstand som fører til akkurat samme handling.

**Forkastet: en sperrefrist før ny avstemning.** Et lederløst lag er allerede i
den tilstanden ordningen finnes for å komme ut av. Å låse det ute en periode
straffer laget for at for få stemte — og det er som regel de mest passive lagene
som trenger flest forsøk.

*Kilde: eieravklaring 2026-08-09; `services/teamgov.py`, `resolve_election`.*

## B-41 Fristen er absolutt, selv om avgjørelsen er lat

**Kontekst.** Frister avgjøres lat (B-18-mønsteret): første gang noen spør. Da
oppstår spørsmålet om hva som skjer med den som spør *etter* fristen.

**Valg.** Utfallet skrives, og avstemningen reåpnes ikke. En stemme etter
`closes_at` svarer 404.

**Forkastet: å la den som utløser avgjørelsen få stemme først.** Det er
fristende, fordi koden allerede har raden i hånden — men det ville gjort fristen
avhengig av *hvem som åpnet appen først*, som er ren tilfeldighet.

**Bug funnet av denne avklaringen (2026-08-09):** `active_election` returnerte
raden også når `resolve_election` nettopp hadde avgjort den, så det første
kallet etter fristen fikk stemme på en avstemning som allerede var `expired`.
Fristen var altså absolutt for alle andre enn den som tilfeldigvis utløste den
late avgjørelsen. Rettet i samme runde; `active_challenge` hadde samme feil.

*Kilde: eieravklaring 2026-08-09; `services/teamgov.py`, `active_election` og
`active_challenge`; `tests/test_teams.py`,
`test_fristen_er_absolutt_stemme_etter_frist_avvises`.*

## B-42 Serveren filtrerer overkjørte kømeldinger

**Kontekst.** Klienten henter meldingskøen ved *appstart*, ikke løpende
(B-18). En «avstemningen er åpen i 7 dager» kan derfor hentes ni dager senere og
vises rett over «avstemningen er avsluttet».

**Valg.** Når et utfall skrives, får køede meldinger som bare varslet om at
prosessen pågikk, `superseded_at` satt, og de filtreres bort i
`GET /v1/messages`.

**Forkastet: å la klienten skjule dem.** Klienten måtte da gjenskapt
lagstyre-logikken — hvilke `kind`-verdier som overkjører hvilke, og om utfallet
finnes i det hele tatt. Serveren vet det allerede, og en regel som håndheves ett
sted kan ikke komme i utakt med seg selv. Det følger også eierskapsprinsippet:
invarianten eies av den som håndhever den.

**Forkastet: å slette raden.** Samme grunn som at en kvittering markerer i
stedet for å slette — historikken er verdt mer enn de få bytene.

**Utfallsmeldingen annulleres aldri.** Det er varselet om at noe *pågår* som
er ferskvare.

*Kilde: eieravklaring 2026-08-09; `services/teamgov.py`,
`annuller_overkjoerte`; `routers/messages.py`; `backend/KONTRAKT.md` §4.1.*

## B-43 Standardlista er kurert mot delstrengsmatching, ikke mot fullstendighet

**Kontekst.** B-37 lot ordlista være tom, og resultatet var at moderasjonen
ikke fanget noe som helst med mindre driften hadde satt
`DISPLAY_NAME_BLOCKLIST`. Visningsnavnet er det ene feltet en bruker kan skrive
fritt til andre med (B-36).

**Valg.** En kurert liste i `services/blocklist.py` er standardverdien;
`DISPLAY_NAME_BLOCKLIST` **legges til** den og kan ikke slå den av. Ordene er
valgt etter én regel: sammenligningen er delstreng på foldet form, og foldingen
fjerner mellomrommene mellom fornavn og etternavn. Et ord må derfor være langt
nok og særpreget nok til at det ikke kan oppstå inne i, eller på tvers av, et
ekte navn. Lista er kort med vilje — dekning er underordnet det å ikke stenge
folk ute fra sitt eget navn.

**Forkastet: en så komplett banneordliste som mulig.** De vraket kandidatene
står med begrunnelse i docstringen til `blocklist.py`: «hore» rammer
«Thoresen», «fag» rammer «Fagerli», «nazi» rammer fornavnet «Nazir», «kkk»
rammer «Erik K. Kristiansen». En liste som fanger alt, avviser også ekte navn —
og en bruker som ikke får bruke navnet sitt, har ingen vei rundt.

**Forkastet: at miljøvariabelen erstatter standardlista.** Da ville et miljø som
ville legge til ett ord, i praksis slått av alle de andre uten å ha ment det.

**`DISPLAY_NAME_ALLOWLIST` kom i samme runde.** En hardkodet liste er et
enkeltpunkt, og veien ut måtte finnes fra dag én (§7.3: «lager du et
enkeltpunkt, bygg utskiftningsveien i samme runde»). Unntakene klippes ut av den
foldede formen *før* ordlista sjekkes, slik at «Anne Gerd Hansen» går klar og
ikke bare «Anne Gerd». Det ene kjente sammenstøtet — «Anne Gerd» inneholder
«neger» når mellomrommet er borte — står i standardunntakene.

**Ord som beskriver hvem noen er, står ikke i lista.** «jøde», «muslim»,
«same», «homofil» er nøytrale ord. Det er skjellsordene som blokkeres, ikke
gruppene de rammer.

*Kilde: eierbeslutning 2026-08-12; `services/blocklist.py`, modul-docstring;
`config.py`, `display_name_blocklist_list`; `backend_spec.md` §3.*

## B-44 Feilanalyse-bildene lastes opp til R2 (ÅP-B5 lukket)

**Kontekst.** ÅP-B5 var stilt som et valg: koble inn R2, eller avvikle
oppsettet. Speccen avgjorde det — §6 og §0.1 sier begge at bildene ligger i
objektlagring og at bare metadata ligger relasjonelt. Å avvikle R2 ville krevd
at *speccen* ble endret, og det er ikke en kodebeslutning. Bildene er dessuten
det materialet §6 finnes for: én donasjon er opptil 8 MB, og en Postgres-rad er
et dårlig sted for dem.

**Valg.** `services/objstore.py` laster opp til R2 ved mottak;
`failed_analyses.object_key` er referansen, og bildet skrives ikke til basen.

**Forkastet: å migrere de gamle `image_legacy`-radene i samme runde.** En
flytting leser fra basen og skriver til et sted vi ikke hadde skrevet til før,
og verktøyet som skulle gjort det ville vært kjørt nøyaktig én gang uten å ha
vært prøvd. Radene blir liggende, og `/health` teller dem så det er synlig når
kolonnen kan fjernes.

**Forkastet: boto3.** Vi trenger PUT, GET og DELETE av ett objekt. boto3 drar
med botocore og datafiler for hele AWS-tjenestekatalogen. Samme avveining som
B-19, der google-auth ble droppet til fordel for PyJWT vi allerede hadde. Prisen er at SigV4-signeringen er vår egen — den er derfor både
enhetstestet (`test_failed_analyses.py`) og verifiserbar mot ekte R2 med
`tools/r2_check.py`.

**Ikke konfigurert ⇒ basen, som før.** Testene og lokal kjøring har ingen
R2-nøkler, og et endepunkt som svarer 503 uten dem ville gjort §6 uprøvbar
lokalt.

*Kilde: `services/objstore.py`, modul-docstring; `backend_spec.md` §6, §0.1.*

## B-45 En feilet opplasting blir 503, aldri en stille fallback til basen

**Kontekst.** «Ikke konfigurert» og «konfigurert, men opplastingen feilet» ser
like ut hvis begge ender med at bildet legges i basen. Klienten gikk i nøyaktig
den fella andre veien: `BackupKeys.resolve` behandlet «fikk ikke svar» som
«fant ingenting», og en 405 var usynlig i tre versjoner (rot-`CLAUDE.md` §7.3).

**Valg.** Er R2 konfigurert og opplastingen feiler, rulles raden tilbake og
svaret er 503 med feilen logget. Ingen rad, ingen halv donasjon.

**Hvorfor 503 og ikke 400.** `android/KONTRAKT.md` klassifiserer ≥ 500 som
`retryable`. Et midlertidig R2-problem besvart med 4xx ville fått klienten til å
kaste donasjonen — stille datatap hos brukeren.

**Rekkefølgen er `flush()` før opplasting.** Objektnøkkelen navngis med rad-IDen,
så raden må finnes; men den finnes bare i transaksjonen, og en feilet opplasting
ruller den tilbake. Alternativet — å committe raden først og laste opp etterpå —
ville etterlatt rader som peker på objekter som aldri ble skrevet.

**Følgen for driften:** en feilsatt nøkkel viser seg som 503 og en logglinje
med R2s egen feilkode (`SignatureDoesNotMatch`, `NoSuchBucket`,
`AccessDenied`), ikke som stille normal drift. `/health` sier i tillegg hvilken
lagring som faktisk er i bruk.

*Kilde: `routers/failed_analyses.py`; `services/objstore.py`, modul-docstring.*

## B-46 Innholdet avgjør bildetypen, ikke `Content-Type`

**Kontekst.** `POST /v1/failed-analyses` krever ikke innlogging (B-31), og
skriver fra 2026-08-15 til betalt objektlagring. Det er en åpen skrivevei.

**Valg.** De første bytene må være JPEG, PNG eller WebP; ellers 415. Endelsen og
`Content-Type` på objektet i R2 utledes av det samme, ikke av det klienten
påstår i multiparten.

**Forkastet: å stole på `image.content_type`.** Den kommer fra avsenderen og
sier ingenting. Da ville en `.jpg`-nøkkel i R2 kunne inneholde hva som helst.

**Dette verner ikke mot misbruk**, bare mot det utilsiktede: endepunktet er
fortsatt åpent og uten kvote. Se ÅP-B9 for ratebegrensning som teller riktig.

*Kilde: `routers/failed_analyses.py`, `_BILDETYPER`.*

## B-47 `r2_check.py` ligger i produksjonsimaget

**Kontekst.** B-44 la SigV4-signeringen i vår egen kode, og `tools/r2_check.py`
er verifiseringen som hører til: den er den eneste måten å vite at Cloudflare
godtar signaturen, siden testene bytter ut HTTP-kallet. Men den krever de fire
R2-verdiene, og de finnes bare som Fly secrets — Fly kan ikke lese dem ut igjen.
Verktøyet kunne altså bare kjøres på en maskin der forutsetningen for å kjøre
det ikke er oppfylt.

**Valg.** `COPY tools/r2_check.py ./tools/r2_check.py` i `backend/Dockerfile`,
så den kan kjøres der secretene faktisk står:

```powershell
flyctl ssh console -a bestefar-api -C "python tools/r2_check.py"
```

**Begrunnelsen er §7.3, brukt på seg selv: et tiltak som ikke kan utføres der
problemet oppstår, er ikke et tiltak.** Det er den samme feilen som
`BACKUP_ESCROW_SECRET`-advarselen var — en instruks uten en vei til å følge den.
Alternativet, å legge de fire verdiene i `backend\.env` på utviklerens maskin,
lager en ny kopi av produksjonens nøkler for å teste noe som allerede har dem.

**Forkastet: `COPY tools ./tools`.** `gen_openapi.py` skriver inn i repoet
(`contracts/openapi.json`) og har ingen rolle i drift. Skillet står som kommentar
i Dockerfile, så neste verktøy havner riktig: verktøy som trenger produksjonens
secrets hører hjemme i imaget, repo-verktøy ikke.

**Forkastet: en engangssnutt over `flyctl ssh console -C python -c ...`.** Den
ville verifisert like godt én gang, og etterlatt ingenting som kan kjøres neste
gang nøkkelen roteres.

*Kilde: `backend/Dockerfile`; `tools/r2_check.py`, modul-docstring; ÅP-E10.*

## B-48 De gamle bildene flyttes, og basen tømmes først etter tilbakelesing

**Kontekst.** Fem donasjoner ble tatt imot før opplastingen ble koblet inn og
ligger som blob i `image_legacy` (ÅP-B11). Eieravklaring 2026-08-15: de skal
**flyttes, ikke kastes** — det er materialet ÅP-U14 mangler for dedupliseringen,
og ett av dem er en ekte over-deteksjon fra felt.

**Valg.** `services/legacy_bilder.py` gjør per rad: last opp → **les tilbake og
sammenlign byte for byte** → tøm `image_legacy` → commit. Feiler noe av det,
står raden urørt med bildet i behold.

**Hvorfor tilbakelesingen ikke er overflødig.** Databasen er den *eneste* kopien.
En PUT som svarte 200 uten at objektet er lesbart, ville kostet bildet, og vi
ville sett en vellykket flytting. Det er samme klasse som gjenopprettingen som
skrev over lokale data før noen hadde sett hva kopien inneholdt (§7.3).

**Commit per rad, ikke én til slutt.** Feiler den fjerde, står de tre første
flyttet og resten urørt. Ingen halvveis tilstand å rydde i.

**Tørrkjøring er standard.** `flytt(..., toerrkjoer=True)` er default, og
`tools/migrate_legacy_images.py` krever `--utfoer` for å skrive. Et verktøy som
kjøres én gang mot produksjon, skal kunne kjøres én gang uten å gjøre noe.

**`MAX_UPLOAD_BYTES` gjelder ikke her.** Grensen er en regel for hva *mottaket*
tar imot, ikke for hva vi flytter — å avvise en stor rad her ville vært å kaste
det største bildet vi har fordi en senere regel ikke likte det.

*Rettet 2026-08-15 etter kjøringen:* begrunnelsen over ble skrevet med «én rad er
11 MB» som eksempel. Det stemte ikke. De fem radene var 13 byte til 3 844 036
byte, **10 509 298 byte til sammen** — 11 MB var totalen. Ingen rad traff
grensen, så valget ble aldri prøvd i praksis. Prinsippet står; eksempelet var
feil, og det er skilt ut her i stedet for å rettes bort, siden det er avstanden
mellom antatt og målt som er verdt å huske.

**En blob vi ikke kjenner igjen, blir liggende.** `objstore.bildetype()` avgjør
typen fra de første bytene; er den ukjent, hoppes raden over og telles som
«hoppet over», ikke som feil. Vi flytter ikke noe vi ikke kan navngi.

**Bildesniffingen flyttet fra routeren til `objstore`** i samme runde. Både
mottaket og flyttingen må ta den samme avgjørelsen, og flyttingen har ingen
multipart å spørre.

*Kilde: `services/legacy_bilder.py`, modul-docstring (fjernet med B-49);
migrasjonen `a3f7c1e59b24`; `backend/CHANGELOG.md` 2026-08-15.*

## B-49 `image_legacy` er fjernet, og uten R2 tar vi ikke imot donasjoner

**Kontekst.** Kolonnen var tom etter B-48, og §6 sier at bilder aldri skal ligge
i databasen. Migrasjon `a3f7c1e59b24` dropper den.

**Følgen måtte avgjøres, ikke bare aksepteres.** Fallbacken «uten R2 → legg i
basen» (B-44) hadde ikke lenger noe sted å legge bildet. Valget er **503 før
kroppen leses**: uten objektlagring tar vi ikke imot donasjonen i det hele tatt.

**Forkastet: å ta imot og forkaste bildet.** Et 201 på noe vi kastet er
kvittering for datatap — den klassen feil hele §6-arbeidet har handlet om.

**Forkastet: å beholde kolonnen «i tilfelle».** En tom kolonne er en åpning: den
neste som mangler R2 lokalt, fyller den, og §6 er brutt igjen uten at noen tar en
beslutning. Nå er invarianten strukturell, og `tests/test_failed_analyses.py`
har en test som faller hvis kolonnen kommer tilbake.

**Prisen er lokal utvikling.** Uten R2-nøkler kan man ikke lenger ta imot en
donasjon lokalt; testene stubber `objstore`. Det er samme form som
`JWT_SECRET`-invarianten (uten den utstedes ingenting), og det er en pris verdt å
betale for at §6 ikke kan brytes ved et uhell.

**`/health` sier nå `r2` eller `ikke konfigurert (§6)`.** Radtellingen forsvant
med kolonnen — den svarte på når kolonnen kunne fjernes, og det svaret er gitt.
*De to viste seg å være for få: B-51 la til `feilkonfigurert (…)`.*

**Vinduet under deploy er kjent og akseptert.** `release_command` kjører
migrasjonen før den nye versjonen slippes til, så i noen sekunder kjører gammel
kode mot et skjema uten kolonnen. Bare INSERT-en i `POST /v1/failed-analyses`
treffes; den feiler med 500, som klienten behandler som `retryable`. `/health`
tåler det, fordi tellingen der lå i en try/except.

**`legacy_bilder.py`, verktøyet og testene er slettet, ikke flagget.** Regelen om
å pause funksjonalitet bak et navngitt flagg gjelder funksjoner som skal tilbake.
Dette var en engangsflytting som er utført; koden står i historikken, og
CHANGELOG sier hvilken commit.

*Kilde: migrasjonen `a3f7c1e59b24`, docstring; `routers/failed_analyses.py`,
modul-docstring; `routers/health.py`, `_bilder_status`.*

## B-50 Bildene flyttes til en jurisdiksjonsbundet bucket

**Kontekst.** `bestefar-scan-failures` ble opprettet med *location hint* `EEUR`
og uten jurisdiksjonsbinding. Et hint er en preferanse, ikke en garanti — og
«Eastern Europe» dekker land utenfor EØS. Personvernerklæringen kan da ikke si
hvor bildene ligger. `bestefar-scan-failures-eur` er opprettet med jurisdiksjon
`eu`.

**Valg.** Objektene kopieres med **uendret nøkkel** til den nye bucketen.
`failed_analyses.object_key` peker på hele stien, og jobben rører derfor ikke
databasen i det hele tatt — den leser bare hvilke nøkler som finnes.

**Følgen for koden: en bucket kan ikke adresseres av navnet alene.** En
jurisdiksjonsbundet bucket har sitt *eget endepunkt* (`.../eu`), så to buckets i
samme konto ligger på hver sin URL. `objstore.Bucket` samler derfor endepunkt,
bucketnavn og nøkler i ett objekt, og `fra_settings()` lager den som gjelder i
vanlig drift. Uten det kunne koden bare snakke med «bucketen i konfigurasjonen»,
og en kopiering mellom to var umulig å skrive.

**`LagringFeilet` fikk `status`.** Kopieringen må skille «objektet er ikke der»
(404) fra «du får ikke lov» (403), og det skal ikke gjøres ved å lete etter et
tall i feilteksten. Et 403 som leses som et fravær ville fått jobben til å laste
opp på nytt i det uendelige og melde suksess — samme forveksling som
`BackupKeys.resolve` gjorde hos klienten (§7.3).

**Rekkefølgen: bytt secretene FØRST, kopier etterpå.** Da skriver tjenesten
allerede til den nye bucketen, og kilden kan ikke få nye objekter mens jobben
går. Motsatt rekkefølge etterlater et vindu der donasjoner lander i den gamle
bucketen etter at kopieringen er ferdig.

**Ingenting slettes i denne runden.** Den gamle bucketen står urørt til den nye
er verifisert i produksjon med en ekte donasjon. Opprydding er en egen runde —
en kopiering som viser seg å være ufullstendig, skal ikke ha slettet originalen.

**Jobben kan kjøres om igjen.** Objekter som allerede ligger i målet med
identisk innhold hoppes over; ligger de der med *annet* innhold, er det en feil
og ikke noe vi overskriver i stillhet.

*Kilde: `services/bucketflytt.py`, modul-docstring; `tools/copy_bucket.py`;
`tests/test_bucketflytt.py`.*

---

## B-51 «Satt» er ikke «virker» — `/health` fikk en tredje tilstand

**Kontekst.** Under byttet til den EU-bundne bucketen (B-50) tok det fire
forsøk å få secretene riktige, og `GET /health` svarte `"bilder": "r2"` gjennom
alle sammen — mens hver eneste donasjon fikk 503. `er_konfigurert()` sjekket at
fire strenger var ikke-tomme, og det var alt ordet «konfigurert» betydde. To av
de tre feilene kunne vært sett uten å spørre Cloudflare: `R2_ENDPOINT` med
bucketnavnet i stien (signaturen dekker da en annen sti enn forespørselen), og
en `R2_ACCESS_KEY_ID` satt til plassholderteksten — R2 sier det selv,
«Credential access key has length 9, should be 32».

**Valg: tre tilstander i samme felt**, ikke to og ikke et nytt felt.

| `bilder` | Betyr |
|---|---|
| `r2` | Ingenting vi kan se uten å spørre Cloudflare er galt |
| `ikke konfigurert (§6)` | **Ingen** av de fire verdiene er satt — funksjonen er av |
| `feilkonfigurert (<hva>)` | Verdier er satt, og noe er beviselig galt |

Det er den samme tredelingen `database` allerede hadde (`ok` /
`feilkonfigurert (DATABASE_URL mangler)` / `utilgjengelig`), så feltet leses
uten ny forklaring.

**Hvorfor ikke la feilkonfigurasjon telle som «ikke konfigurert».** Det var det
enkleste, og det ble forkastet: det er nøyaktig fellen `BackupKeys.resolve`
gikk i hos klienten (rot-`CLAUDE.md` §7.3) — en feil som behandles som et
fravær forsvinner. `objstore` sin egen modul-docstring har forbudt det for
opplastingsveien siden B-45; det ville vært rart å innføre det i diagnostikken.

**Hvorfor ikke et eget felt.** Da ville `bilder` fortsatt sagt `"r2"` når
lagringen ikke virker, og det er selve påstanden som var usann.

**Halvveis satt er «feilkonfigurert», ikke «av».** Tre av fire secrets satt er
en jobb noen ikke ble ferdig med, og navnene på de manglende står i teksten.
Bare et helt tomt oppsett er «av» — den tilstanden er normal i utvikling.

**Bare det som er alltid feil.** Sjekkene er: mellomrom eller linjeskift rundt
en verdi (usynlig i panelet, gir `SignatureDoesNotMatch`), `R2_ENDPOINT` som
ikke er en URL eller har sti, `R2_BUCKET` med skråstrek, og `R2_ACCESS_KEY_ID`
som ikke er 32 tegn. Ikke jurisdiksjon, ikke om bucketen finnes, ikke om
signaturen godtas — det krever et kall, og det er `r2_check.py` sitt ærend. En
sjekk med falske positiver ville stengt en fungerende bucket, og det er verre
enn tilstanden dette retter.

**Mottaket spør om samme tilstand som `/health` viser.** `kan_brukes()` er
definert som «`tilstand()` sier `r2`», så de to kan ikke komme i utakt. To
uavhengige sjekker på samme spørsmål var måten uenigheten oppsto på.

**Verdiene skrives aldri ut**, bare navnene: teksten går både i `/health`, som
er åpen, og i loggen, og to av de fire er hemmeligheter.

*Kilde: `services/objstore.py` (`feilkonfigurasjon`, `tilstand`, `kan_brukes`);
ÅP-B12, målt 2026-08-18; `tests/test_failed_analyses.py`.*

---

## B-52 `series_id` fjernes fra donasjonen, i stedet for at bildene får en frist

**Kontekst.** Et donert skivebilde var ikke merket med konto, og endepunktet
krever ikke innlogging — men raden bar `series_id`, og det er *samme* ID som
serien lagres under i `/v1/stats`. Den som har databasen kunne dermed koble
bildet til en person, og bildene var derfor personopplysninger
(`personvernerklaring.txt` 2.7, ÅPENT PUNKT 6).

**Det forkastede alternativet var en slettefrist.** 3 år ble diskutert.
Eieravklaring 2026-08-20 valgte bort tid som virkemiddel: en frist *utsetter*
koblingen i stedet for å fjerne den, og den forutsetter dessuten en ryddejobb
som ikke finnes (ÅP-B14). Frikobling ved innsending gjør spørsmålet om
lagringstid på bildene irrelevant — det er derfor de to punktene henger sammen.

**Feltet hadde ingen funksjon.** En feilet analyse er per definisjon ikke et
resultat, så det finnes ingen serie å knytte den til på en meningsfull måte.
Verifisert før fjerning, ikke antatt: ingenting i backend-treet leser kolonnen.
Den hadde en indeks (`ix_failed_analyses_series_id`) uten et eneste oppslag, som
er så tydelig et tegn på ubrukt felt som man får. Kalibreringsmaterialet
ÅP-U14 trenger, ligger i `tag`, `status_code` og de to poenglistene.

**Kolonnen droppes, den nulles ikke.** Å bare slutte å skrive den ville latt de
eksisterende radene stå igjen som personopplysninger, og da ville tiltaket ikke
lukket noe som helst. Migrasjonen `b8d24a0f5c17` fjerner kolonnen og indeksen;
downgrade gir kolonnen tilbake, men tom — verdiene er borte, og det er hensikten.

**Talt i produksjon rett før migrasjonen, 2026-08-21: 9 rader, 6 med
`series_id`, 0 med `user_id`.** Seks koblinger til en konto forsvant. Tallet
står her og i ÅP-B13 fordi det ikke kan hentes i ettertid — etter migrasjonen
finnes ikke kolonnene å telle. Det er ikke det samme tallet som de «sju
objektene» i B-50 (objekter i R2) eller «rad 11» i ÅP-E11 (en ID, ikke et
antall).

**`user_id` ble droppet i samme migrasjon.** Den var tom — null av ni — og har
aldri vært satt av endepunktet, så den var ingen personopplysning. Men den var
en *ferdig oppkoblet* fremmednøkkel til `users` på en tabell hvis hele poeng nå
er at radene ikke kan knyttes til en konto, og en kobling som allerede ligger
der er noe annet enn en som må lages. Etter dette kreves en ny migrasjon for å
gjøre donasjoner identifiserbare igjen. Det er den terskelen som er poenget:
ikke at det skal være umulig, men at det skal måtte besluttes.

**Et felt vi ikke lenger tar imot, avvises ikke.** Klientene i felten sender
det fortsatt. FastAPI ignorerer skjemafelt ruten ikke erklærer, og det er den
riktige oppførselen her: 4xx er ikke `retryable` i `android/KONTRAKT.md`, så et
422 på et overflødig felt ville vært stille tap av donasjoner hos alle som ikke
hadde oppdatert. Klienten kan fjerne feltet når det passer, uten koordinering.

*Kilde: ÅP-B13 og eieravklaring 2026-08-20; `migrations/versions/b8d24a0f5c17`;
`routers/failed_analyses.py`; `tests/test_failed_analyses.py`.*

---

## B-53 `capture_trigger` som eget felt, ikke som en `tag`-verdi

**Kontekst.** Klienten fikk i v0.29 en tidsgrense på auto-capture: utløser ikke
gatingen innen 7 sekunder, tas gjeldende ramme og analyseres likevel. Lykkes
analysen da, er det ikke en feil — det er måledata om at gatingen var for
streng, og nettopp den observasjonen mangler ÅP-K1. Uten et merke i donasjonen
er den ikke skillbar fra en helt ordinær scan. Klienten kunne ikke velge formen
selv: `tag`-enumet er vårt, og en gjettet verdi gir 422 (issue #11).

**Valg: et eget felt `capture_trigger` ∈ {`auto`, `timeout`}.** Det forkastede
alternativet var nye `tag`-verdier. `tag` svarer på *hva donasjonen viser*,
`capture_trigger` på *hvordan bildet ble tatt*, og de to er **ortogonale** — en
timeout-capture kan ende som hvilken som helst av de tre taggene. En
`timeout`-verdi i `tag` ville derfor overskrevet OCR-utfallet, og skulle begge
deler bevares måtte enumet bli `timeout_ocr_match`, `timeout_ocr_mismatch`,
`timeout_rejected` — og dobles igjen ved neste capture-årsak. Klienten
anbefalte det samme, og eier sa seg enig; her er begrunnelsen skrevet ned så
den ikke må gjenoppdages.

**NULL betyr «ikke oppgitt», ikke `auto`.** Kolonnen er nullable og har ingen
default. Klienten fikk timeout-capture i v0.29, men sendte bevisst ikke feltet
før formen var avtalt — donasjonene fra det vinduet *er* dels timeout-utløste
uten å kunne si det. En default på `auto` ville stemplet dem som gatede, og det
er akkurat disse radene ÅP-K1 skal måles på. Samme skille som `objstore` gjør
mellom «ikke konfigurert» og «feilkonfigurert» (B-51), og som `BackupKeys`
manglet: et fravær er ikke en verdi. Ingen backfill, av samme grunn — det
finnes ikke en riktig verdi å fylle inn.

**Enumet er vårt, og det har en rekkefølge-konsekvens.** En ukjent verdi gir
422, og 422 er ikke `retryable` — donasjonen kastes. Nye `capture_trigger`-
verdier må derfor rulles ut på serveren **før** klienten begynner å sende dem.
Det står i `models/base.py` ved siden av enumet, fordi det er der noen kommer
til å legge til den neste verdien.

*Kilde: issue #11; `migrations/versions/c4a91e7b2f38`; `models/base.py`;
`tests/test_failed_analyses.py`.*

---

## B-54 Kalleren kjenner seg selv i et lag med `my_role` + `members[].public_id`

**Kontekst.** `GET /v1/teams/{id}` ga `members[]` med intern `user_id`, og
klienten kjenner bare `public_id` fra innloggingssvaret. Den kunne dermed ikke
kjenne igjen seg selv i lista (issue #14). Det blokkerte «(Du)»-merkingen og
hele ledergatingen — «Rediger lag» mot «Velg leder», og redigeringsmenyen som
bare skal finnes for en leder. Klienten gatet i mellomtiden på `has_leader`,
som behandler «laget har en leder» som «jeg er lederen».

**Valg: to felt, fordi spørsmålet er to spørsmål.** `my_role` sier *hva jeg er*,
`members[].public_id` sier *hvilket element som er meg*. Ett av dem alene
holder ikke: `my_role` peker ikke ut en rad, og en `public_id` i lista sier
ingenting på listeskjermen, der `members[]` ikke er med. `my_role` ligger
derfor på **alle** lagsvar, også `GET /v1/teams` og `/near`, så listeskjermen
slipper å hente detaljer for å vite hva den skal tilby.

**`public_id` og ikke et ekko av `user_id`.** Klienten foreslo begge deler.
`public_id` er formen klienten allerede har, og det er formen venne-modellen
skal bruke (ÅP-U31, «unik per `public_id`») — bygger klienten gjenkjenning på
den nå, står den seg gjennom steg 4.

**Det forkastede alternativet var å bytte ut `user_id` i `members[]`.** Det
ville sett renere ut, men gjort lista ubrukelig som kilde til de kallene den
finnes for: `DELETE …/members/{id}`, `POST …/leaders/{id}` og `candidate_id` i
avstemningen tar alle imot den *interne* ID-en. Å bytte `members[]` alene ville
brutt dem; å bytte alt sammen er en flatedekkende endring som hører hjemme i
samme runde som venne-modellen, ikke i en runde som skal fjerne en blokkering.
Begge former står derfor side om side inntil da, og hva hver av dem er til,
står i `KONTRAKT.md` §6.

**Å matche visningsnavn ble aldri vurdert som en utvei**, og det er verdt å
notere hvorfor: det ville virket i de fleste lag og feilet stille i akkurat de
lagene der to personer heter det samme — med det utfallet at feil person får
lederknappene.

*Kilde: issue #14; `routers/teams.py` (`_team_ut`); `tests/test_teams.py`.*

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
