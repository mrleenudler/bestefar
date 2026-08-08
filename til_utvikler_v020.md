# Til utvikler — v0.20 (klienten mot openapi.json)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en egen seksjon nederst — ikke overskriv.

## Oppdraget

> «Gå gjennom Api.kt og alle kallstedene og sammenlign mot skjemaet.»

## Resultatet i én setning

**Alle femten ruter og verb stemte. Fem avvik lå i felter, parametere og
grenser — og alle fem var klientens feil.** Ingen issue mot backend fra denne
runden.

Rutesjekken ble kjørt maskinelt: hvert `Api.`-kall i
`android/app/src/main/java/no/bestefar/app/` plukket ut med regex, verbet
utledet, og slått opp i `paths` i skjemaet. Femten treff, null bom.

---

## De fem avvikene

### 1. `client_ts` ble sendt som epoke-millisekunder

Skjemaet deklarerer `date-time`. Serveren tar imot begge — det står nå
uttrykkelig i `BackupMeta.client_ts` — men ms-varianten hvilte på en
*tolkning*: parseren behandler store heltall som millisekunder framfor
sekunder.

Blir den tolkningen noen gang strengere, leses `1786132462317` som sekunder.
Tidspunktet havner titusener av år fram i tid, og hver eneste opplasting ser ut
som den nyeste. **Da snur 409-vernet:** det ville sluppet gjennom nøyaktig den
utdaterte enheten det finnes for å stoppe, og resultatet er tapt jaktlogg.

Sendes nå som ISO-8601 med `Z`. **Aldri `+00:00`** — et `+` i en query-streng
leses som mellomrom.

Tidsstempelet tas dessuten nå *før* bloben bygges. Feltet heter «klientens
tidsstempel for øyeblikksbildet», og nøkkelutledningen tar et kvart sekund; da
skal tallet ikke være fra etterpå.

### 2. `app_version` fantes ikke i skjemaet

Klienten sendte den på `PUT /v1/backup`. Serveren tar ikke imot en slik
parameter, og ukjente query-parametere forsvinner stille. Verdien har aldri nådd
fram — dette var halve ÅP-U13.

Droppet. Appversjonen ligger uansett inne i bloben (`app`), der den er til nytte
for den som kan dekryptere den.

### 3. `schema_version` ble aldri sendt

Klienten *har* en `SNAPSHOT_VERSION`, serveren lagrer feltet og gir det tilbake
i `/meta`. Ingen sendte det, så alt lå på serverens default.

Sendes nå. Poenget er konkret: en ny telefon kan se om den kan lese kopien
**før** den laster ned 16 MB.

### 4. Tilbakemeldingsskjemaet håndhevet ingen av serverens grenser

`FeedbackIn` setter emne til 200 tegn, melding til 10 000 og `device_model` til
64. Klienten begrenset ingen av dem.

`device_model` er den som faktisk slo ut: `"$MANUFACTURER $MODEL"` uten
avkorting. På en telefon med langt fabrikant- og modellnavn ga det 422 på hele
tilbakemeldingen — av en grunn som ikke har noe med meldingen å gjøre. Og siden
422 ikke er 429, falt koden ut i mailto-grenen: **«Send melding» åpnet
e-postappen i stedet, uten at noen skjønte hvorfor.** Det er akkurat den klassen
feil som ikke blir rapportert, fordi den ser ut som en funksjon.

Emne og melding har nå lengdefilter i selve feltet — brukeren skal merke
grensen mens de skriver, ikke miste slutten uten beskjed. `device_model`
avkortes til 64, slik `Push.kt` allerede gjorde.

### 5. `client_ts` fra `/meta` ble lest med `optString` på et nullable felt

org.json gir strengen `"null"` for en JSON-null. Det virket — datoparseren
feilet, og dialogen falt tilbake på den udaterte teksten — men det virket ved
uhell. Leses nå med `isNull` først, som `Messages.kt` allerede gjorde for
`team_id`.

---

## Hva skjemaet ikke kunne svare på

`contracts/README.md` er ærlig om dette, og det stemte: 34 av 48 operasjoner har
`additionalProperties: true` som svarkropp (ÅP-B10), det finnes ingen
`securitySchemes`, og blobformatet er en binærstruktur OpenAPI ikke kan
beskrive. **De fjorten operasjonene som *har* svarmodell, er nøyaktig de
klienten kaller** — så for denne gjennomgangen var dekningen full der det
gjaldt.

To ting jeg ellers merket meg, ingen av dem feil:

- **`is_new` i innloggingssvaret leses ikke av klienten.** Det er signalet vi
  trenger for «tilbyd sikkerhetskopi rett etter første innlogging», som har
  stått åpent siden v0.17. Feltet er der; koblingen mangler.
- **`Authorization` står som en valgfri header-parameter** på alle beskyttede
  ruter, fordi den leses med `Header(default=None)`. Et generert klientbibliotek
  ville trodd den var valgfri. Vår klient setter den alltid når vi har et token,
  så det treffer oss ikke — men det er verdt å vite før noen genererer en klient
  fra fila.

---

## Verifisert

- `compileDebugKotlin` og `assembleRelease` grønt, `dist\Bestefar-0.20.apk`.
- Rutesjekken kjørt mot `contracts/openapi.json`, ikke lest for hånd.

## Ikke verifisert

- **Ingen av de fem rettelsene er kjørt mot serveren.** ISO-tidsstempelet er den
  som er verdt en test: ta en sikkerhetskopi, og sjekk at `GET /v1/backup/meta`
  gir riktig `client_ts` og `schema_version: 1`.
- Feedback-grensene er ikke prøvd med en for lang tekst.
