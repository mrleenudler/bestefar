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

Kilder: `bestefar_CV-kjerne_spec.md` (kjerne-spec), `backend_spec.md`,
`bestefar_UI_spec-v0-4.md` (UI-spec).

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

### ÅP-E3 — `FCM_SERVICE_ACCOUNT_JSON` og `FCM_PROJECT_ID` er ikke satt
> «**Gjenstår hos eier** … Uten dem sender backenden ingenting, og klienten kan
> ikke skille det fra «ingen venner hadde varsler på».»
> — backend_spec §16, linje 611–613

`/health` viser `"push":"log"` til de er satt. Hele §11-kjeden er bygget på begge
sider og venter bare på dette.

### ÅP-E4 — `GOOGLE_CLIENT_IDS` er ikke satt
Web-klient-ID-en står i backend_spec §15, linje 570–575. Til den er satt som
Fly-secret svarer `/v1/auth/google` 503, og klienten viser «Innlogging er ikke
slått på på serveren ennå».

### ÅP-E5 — Apple: utviklerkonto og verifisert domene
> «Krever Apple Developer Program + verifisert domene, fordi Sign in with Apple på
> Android går via web-flyten. Services ID-en er verdien.»
> — backend_spec §1 og §15, linje 582–586

Endepunktet er bygget og røres ikke. Klientknappen finnes ikke.

### ÅP-E6 — Kopi av `BACKUP_ESCROW_SECRET` utenfor Fly
> «**En kopi utenfor Fly.** … En kopi i utviklerens passordhvelv er det enkleste
> og mest virkningsfulle tiltaket som finnes, og det koster ingenting.»
> — backend_spec §2.1, linje 136–140

Tiltak 2 og 3 i samme avsnitt (`BACKUP_ESCROW_SECRET_OLD` og nøkkel-ID i
`/health`) er bygget. Tiltak 1 er en handling hos eier, og er ikke bekreftet
utført.

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

---

## E. Uavklart teknisk retning

Ikke gjeld — beslutninger som mangler.

### ÅP-B5 — Cloudflare R2 er betalt for og ubrukt · label `backend`
Speccen sier bilder skal ligge i objektlagring, aldri i databasen
(backend_spec §6 linje 253–254, §0.1 linje 410–411). R2 er opprettet med nøkler,
men backenden bruker det ikke. Åpent: om feilanalysebildene skal dit nå eller om
oppsettet skal avvikles.

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

### ÅP-B8 — Manuell moderasjonskø for visningsnavn mangler flate · label `backend`
> «Den manuelle køen krever en admin-flate som ikke finnes ennå; navn som passerer
> regelsettet **godkjennes derfor direkte**.»
> — backend_spec §3, linje 199–201

Speccen forutsetter «regelsett + evt. manuell kø». Bare halvparten finnes.

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
| Utskiftning av `BACKUP_ESCROW_SECRET` | 2026-08-06 | `_OLD`-fallback med omkryptering ved lesing + nøkkel-ID i `/health`. Se ÅP-E6 for det som gjenstår. |
