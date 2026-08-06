# Til utvikler — v0.17 (innlogging i klienten)

> **Merk til den andre instansen:** denne fila deles. Legg backend-notater til
> som en egen seksjon nederst — ikke overskriv.

## Oppdraget

> «Bygg innloggingen i klienten med Credential Manager»

Backendens §1 har vært ferdig siden fase 3, og appen har hatt `Auth.kt` siden
v0.16 — men ingenting *startet* en økt. Alt som krever konto svarte 401.
Det er nå borte: appen har en vei inn.

---

## Hva som er bygget

### `Login.kt` — veien inn

To måter å logge inn på, og begge ender samme sted: leverandøren gir oss et
ID-token, vi bytter det inn i **våre egne** tokener hos backenden, og alt videre
går på dem. `Auth` eier øktene; `Login` eier bare inngangen.

**Google — `GetSignInWithGoogleOption`, ikke bunnarken.** Credential Manager har
to varianter, og forskjellen har praktisk betydning:

| | `GetGoogleIdOption` | `GetSignInWithGoogleOption` |
|---|---|---|
| Ser ut som | bunnark som dukker opp | svar på et knappetrykk |
| Uten autoriserte kontoer | feiler (`NoCredentialException`) | virker |

Den andre er valgt. En «Logg inn»-knapp finnes nettopp for brukeren som *aldri*
har logget inn før, og den er den ene brukeren den første varianten ikke
håndterer.

**Callback-API-et, ikke `suspend`.** `getCredentialAsync` brukes framfor den
suspenderende varianten, slik at innloggingen ikke drar inn en
coroutine-avhengighet i en kodebase som ikke har én fra før. Svaret kommer på
vår egen tråd, og selve token-vekslingen legges på `Api.io` slik at den står i
samme kø som resten av nettverkstrafikken.

**Avbrutt er ikke feil.** `Login.Result` skiller `Avbrutt` fra `Feil`. En bruker
som trykker tilbake i kontovelgeren får ingen melding — de vet hva de gjorde.
`NoCredentialException` (ingen Google-konto på telefonen) får derimot en tekst
som sier hva de kan gjøre i stedet: legge til en konto, eller bruke e-post.

**E-postkode** mot `/v1/auth/email/start` og `/email/verify`. Serveren svarer
alltid 202 på `start`, også for ukjent adresse — et svar som skilte kjent fra
ukjent ville gjort endepunktet til et oppslagsverk over hvem som bruker appen.
Sperrefristen leses fra svaret (`resend_after_seconds`) i stedet for å hardkodes;
nedtellingen i knappen er ren bekvemmelighet, og 429 fra serveren er det som
faktisk håndhever den.

### `LoggInnActivity` — skjermen

To tilstander, ingen mellomting. Du kommer hit fra **Min profil → Konto**, som
viser tilstanden i selve knappeteksten («Logg inn» kontra «Konto: Ola»), så du
slipper å åpne siden for å finne ut om du er innlogget.

Tonen er valgt bevisst. Teksten sier hva kontoen *gir* deg, og sier like tydelig
at appen virker uten:

> Alt annet — scan, innsikt, serier og jaktlogg — virker akkurat som før uten
> konto. Du kan logge inn når som helst, og ingenting du allerede har lagret går
> tapt.

En jeger som bare vil scanne skiver skal ikke føle at hen har hoppet over noe.
Ingenting i appen ber om innlogging uoppfordret.

Utlogging bekreftes, men **uten** advarselsikon — ingenting går tapt, og teksten
sier det rett ut, for det er nøyaktig det brukeren lurer på i det øyeblikket.
`Login.logout` kaller `Auth.logout` (avregistrer enhet → tilbakekall
refresh-token → slett lokalt uansett) og lar i tillegg jaktlogg-låsen glemme at
den er åpnet, så neste bruker av telefonen ikke arver en gyldig frist.

### Web-klient-ID-en leses fra `google-services.json`

Byggfila plukker ut `client_type: 3` og legger den i
`BuildConfig.GOOGLE_WEB_CLIENT_ID`. Grunnen til å lese den framfor å skrive den
inn i `gradle.properties`: to kopier av samme ID kommer i utakt, og symptomet på
utakt er et gyldig token som backenden avviser — med en feilmelding som ikke sier
noe om hvorfor.

Mangler fila, blir verdien tom, og Google-knappen **skjules** i stedet for at
appen krasjer. E-postinnlogging virker fortsatt.

---

## Det du må gjøre — tre ting, alle utenfor koden

### 1. `GOOGLE_CLIENT_IDS` på Fly må inneholde **web**-klient-ID-en

Ikke Android-klient-ID-en. Dette er den vanligste feilen i hele oppsettet, fordi
begge ligger i `google-services.json` og bare skiller seg på `client_type`.

Verdien fra din egen fil (`client_type: 3`):

```
977694072067-i8enscnhed5clstll7o92mpmkmpfrbit.apps.googleusercontent.com
```

Kommandoen kjører du selv — hemmeligheter skal ikke i chat-loggen:

```powershell
flyctl secrets set GOOGLE_CLIENT_IDS="977694072067-i8enscnhed5clstll7o92mpmkmpfrbit.apps.googleusercontent.com" -a bestefar-api
```

(Den er strengt tatt ikke hemmelig — den står i APK-en — men den settes samme
sted som resten, så den hører hjemme i samme kommando.)

### 2. Signeringsnøkkelen — release er OK, debug er det ikke

Google Identity kontrollerer pakkenavn **og** signatur mot en registrert
Android-OAuth-klient. Jeg har sammenlignet:

| | SHA-1 | Status |
|---|---|---|
| Registrert i `google-services.json` | `35f0da…b7257` | — |
| Release-APK-en (`dist\Bestefar-0.17.apk`) | `35f0da…b7257` | ✅ stemmer |
| Debug-keystoren din | `9D:D6:79:BA:D6:D3:53:CA:81:F1:75:C8:09:5C:EE:16:DC:B3:78:2B` | ❌ ikke registrert |

**Google-innlogging virker altså i release-bygget, men vil feile i debug** til du
legger til en Android-OAuth-klient med debug-SHA-1-en i Google Cloud Console
(samme prosjekt, pakkenavn `no.bestefar.app`). Det er verdt å gjøre, ellers kan
du ikke teste innloggingen mot en lokal backend.

### 3. E-postinnlogging krever at `RESEND_API_KEY` fortsatt står

Den er allerede satt (backend-runde 8 verifiserte e-postinnlogging ende-til-ende
mot produksjon), så dette er bare en påminnelse om at e-postveien er den som
virker *uansett* signeringsnøkkel — bruk den til å teste at klientflyten er
riktig før du bryner deg på Google-oppsettet.

---

## Apple

Ikke bygget. Det krever en Apple-utviklerkonto, og den står «på is» hos deg.
Backendens `/v1/auth/apple` er ferdig, så det er én knapp og noen linjer den
dagen kontoen finnes. Skjermen sier det til brukeren i stedet for å vise en
knapp som ikke virker.

---

## Verifisert

- `compileDebugKotlin` grønt.
- Release-bygg OK, kopiert til `dist\Bestefar-0.17.apk`.
- Signeringssertifikatet i APK-en sammenlignet mot `google-services.json`
  (se tabellen over) — de stemmer.

## Ikke verifisert (krever enhet og oppsettet over)

- **Selve Google-flyten er ikke kjørt.** Den kan ikke kjøres før
  `GOOGLE_CLIENT_IDS` står på Fly; til da svarer backenden 503, og klienten
  viser «Innlogging er ikke slått på på serveren ennå».
- **E-postflyten er ikke kjørt fra klienten.** Backend har verifisert
  endepunktene mot produksjon, men ikke gjennom dette UI-et.
- Nedtellingen på «Send ny kode» er testet som kode, ikke mot en ekte 429.

## Fortsatt åpent

- **FCM er ikke koblet inn.** `google-services.json` ligger der og
  `Store.pushToken` finnes, men appen registrerer ingen enhet. Konsekvens nå som
  innlogging finnes: `PUT /v1/devices` blir aldri kalt, så ingen varsler kommer
  fram — heller ikke felling-kunngjøringen fra v0.16. Det er den neste naturlige
  runden.
- **Serie-synk-køen** (`/v1/stats`) er fortsatt ikke bygget. Nå som konto finnes,
  er spørsmålet fra runde 12 blitt akutt: eier bloben eller `/v1/stats`
  sannheten? Det bør avgjøres før begge veier finnes.
- **`bf_version()` i `bestefar_ffi.h`** — ubesvart fra v0.14, v0.15 og v0.16.
- Etter første innlogging bør appen tilby å ta en sikkerhetskopi med det samme.
  Den koblingen er ikke laget; nå må brukeren finne veien til Avanserte
  innstillinger selv.
