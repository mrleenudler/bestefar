# Til utvikler — v0.18 (push-varsler kobles inn)

> **Merk til den andre instansen:** denne fila deles. Legg backend-notater til
> som en egen seksjon nederst — ikke overskriv.

## Oppdraget

> «Koble inn FCM så push-varslene når fram»

Kjeden har hatt ett brudd i midten. Backenden har kunnet sende siden fase 8,
klienten har kunnet be om å sende siden v0.16 (felling-kunngjøringen), og siden
v0.17 har det finnes en konto å knytte det til. Men `PUT /v1/devices` ble aldri
kalt, så backenden hadde **ingen adresse å sende til**. Alt virket, og ingenting
kom fram.

Det er nå tettet.

---

## Kjeden, hele veien

1. **Firebase gir telefonen en adresse** — FCM-tokenet.
2. **Appen melder adressen inn** hos backenden (`PUT /v1/devices`).
3. **Backenden sender** til de adressene den kjenner (`push.send`).
4. **Android viser varselet** — enten selv, eller via appen. Se under; det er
   ikke det samme.

### `Push.kt` — steg 1 og 2

`Push.register(ctx)` henter tokenet og melder enheten inn. Den kalles fra
`BestefarApp.onCreate` ved **hver** oppstart, ikke bak et engangsflagg:
backenden bruker `PUT` nettopp fordi det skal være idempotent, og et flagg som
sier «allerede registrert» er et flagg som kan bli feil uten at noen merker det.

**Registrering krever konto**, og det er ikke en begrensning jeg har funnet på:
et varsel er alltid til *noen*. Uten innlogging har vi ingen bruker å knytte
adressen til, så da hentes tokenet ikke engang.

`PushService.onNewToken` fanger opp at Firebase roterer tokenet — ny
installasjon, gjenoppretting, app-data tømt. Skjer det uten at vi melder fra,
sender backenden videre til en adresse som ikke finnes, og brukeren merker bare
at varslene stilner.

### `PushService.kt` — steg 4, og den ene fella

**Tjenesten kalles sjeldnere enn man tror.** Backenden sender en
`notification`-blokk (`push.py:_melding`), og da tegner **Android selv** varselet
når appen ligger i bakgrunnen. `onMessageReceived` kjøres bare når appen er i
forgrunnen.

Det betyr at to ting må stemme samtidig, og de bor på hver sin kant:

| Tilfelle | Hvem tegner | Hva må være riktig |
|---|---|---|
| App i bakgrunnen | Android | `default_notification_*`-meta-data i manifestet |
| App i forgrunnen | `PushService` | koden i `onMessageReceived` |

Glemmer man det andre, forsvinner en melding som kom mens brukeren så på
skjermen, sporløst. Glemmer man det første, får bakgrunnsvarsler standardikon og
havner i en «Diverse»-kanal brukeren ikke kan skru av alene. Begge er på plass.

Kanalen (`bestefar_varsler`) opprettes i `BestefarApp.onCreate` — den **må**
finnes før det første varselet, også i bakgrunnstilfellet der appen ikke har
kjørt kode.

**Trykk på varselet åpner forsiden.** Jeg ruter bevisst ikke videre på `kind`
ennå: en dyplenke som lander på en tom skjerm er verre enn en som lander på
forsiden, og venne- og lagsidene er fortsatt front-end-skjelett. Når de får
innhold, er `data["kind"]` og `data["team_id"]` allerede med i meldingen.

### Varselikonet

`ic_notification.xml`, en skive med to ringer og full blink. Android tegner
varselikoner som **silhuett** — alt som ikke er gjennomsiktig blir hvitt — så
form er det eneste som overlever. Et fargerikt ikon her blir en hvit klump.
Aksentfargen (`notification_accent`, samme brune som resten av appen) er det
Android fargelegger bakgrunnen med.

### Tillatelsen — spurt på riktig tidspunkt

Fra Android 13 er varsler en kjøretidstillatelse. Den spørres **etter vellykket
innlogging**, ikke ved appstart, og med en forklaring før systemdialogen:

> Vi varsler deg når en venn feller et dyr, og når det skjer noe i et lag du er
> med i — for eksempel at det skal velges ny leder.
>
> Ingenting annet. Du kan skru dem av igjen når som helst.

Grunnen til rekkefølgen: et systemvindu uten kontekst får «nei», og det «nei-et»
er nesten permanent — Android viser dialogen bare et par ganger, og etterpå må
brukeren finne fram i systeminnstillingene. Før innlogging finnes det dessuten
ingen varsler å gi, så spørsmålet ville vært løgn.

Svarer brukeren nei, registreres enheten **likevel**. Skrur de på varsler senere
i systeminnstillingene, skal adressen allerede være der.

---

## Det du må gjøre

Én ting, og den er på serveren: **`FCM_SERVICE_ACCOUNT_JSON` og
`FCM_PROJECT_ID`** må stå som Fly-secrets (`backend/app/config.py:90–91`).
Tjenestekonto-JSON-en lastes ned i Firebase Console → Prosjektinnstillinger →
Tjenestekontoer → «Generer ny privat nøkkel». `FCM_PROJECT_ID` er `bestefar`.

Kommandoen kjører du selv — den JSON-en er en ekte hemmelighet og skal ikke inn
i chat-loggen. `flyctl secrets set FCM_PROJECT_ID=bestefar -a bestefar-api` og
tilsvarende for JSON-en (som én linje).

Står de ikke, sender backenden ingenting, og klienten merker ingen forskjell fra
«ingen venner hadde varsler på akkurat nå».

---

## Hvordan du tester at det virker

Rekkefølgen er valgt slik at hvert steg bekrefter det forrige:

1. **Logg inn** i appen (e-postkode er raskest, den er uavhengig av
   signeringsnøkkel). Si ja til varsler.
2. **Sjekk at enheten er registrert:** `GET /v1/devices` med tokenet ditt skal
   returnere én rad med modellnavnet på telefonen din. Kommer den ikke, er det
   steg 2 i kjeden som svikter — se logcat-taggen `BestefarPush`.
3. **Utløs et varsel.** Enkleste vei: logg en vellykket felling i appen og send
   kunngjøringen. Med bare deg selv som bruker har du ingen venner å sende til,
   så `devices_notified` blir 0 — det bekrefter at kallet virker, men ikke at
   varselet kommer fram.
4. **Ekte ende-til-ende krever to kontoer** som er venner. Alternativt kan
   backend sende en testmelding direkte via `push.send` mot tokenet ditt.
5. **Test begge tilstandene:** med appen åpen (da går den gjennom
   `PushService`) og med appen i bakgrunnen (da tegner Android det). De to
   stiene er ulik kode, og bare den ene testes hvis du glemmer å låse skjermen.

---

## Verifisert

- `compileDebugKotlin` grønt.
- Release-bygg OK, kopiert til `dist\Bestefar-0.18.apk`.
- `google-services`-pluginen kjører og finner `app/google-services.json` (den
  ble tatt inn i repoet i v0.17, og pluginen **feiler** byggingen om den
  mangler — som er riktig: et push-bygg uten prosjektkonfigurasjon ville bare
  vært stille ødelagt).

## En feil i verktøykjeden, ikke i koden

Release-bygget stoppet først på at **lint krasjet** i sin egen kode:

```
Unexpected failure during lint analysis of PushService.kt
(this is a bug in lint or one of the libraries it depends on)
... resolveSyntheticJavaPropertyAccessorCall ...
```

Utløseren er at Kotlin leser Java-getterne `getNotification()`/`getData()` som
egenskaper (`msg.notification`), og AGP 9s lint faller på nettopp den
oppløsningen inne i en lokal variabel. Koden kompilerer helt fint — det er
analysen som ryker.

Løsningen er å kalle getterne eksplisitt, med en kommentar som sier hvorfor.
Jeg valgte det framfor å skru av `lintVital` for release: en avslått lint hadde
skjult alle framtidige *ekte* funn for å komme rundt én bug i verktøyet.

## Ikke verifisert (krever enhet + Fly-secrets)

- **Ingen push er faktisk mottatt.** Hele kjeden er bygget, men den kan ikke
  kjøres før `FCM_SERVICE_ACCOUNT_JSON` står på Fly.
- Bakgrunns- og forgrunnsstien er ulik kode; ingen av dem er kjørt på enhet.
- Varselikonet er tegnet, ikke sett. Silhuett-regelen gjør at feil her er
  synlige umiddelbart (en hvit klump), så det er verdt et blikk i statusfeltet.
- `onNewToken` utløses sjelden; den er ikke framprovosert.

## Fortsatt åpent

- **Ruting på `kind`.** Varsler åpner forsiden. Når venne- og lagsidene får
  ekte innhold, bør «Ola har felt et villsvin» åpne vennesiden og et
  lederskapsvarsel åpne laget.
- **Meldingskøen (`/v1/messages`) leses ikke.** §11-varsler legges i en kø hos
  backenden *før* pushen sendes — det er køen som er garantien, ikke pushen.
  Klienten henter den ikke, så et varsel som gikk tapt (telefonen av) er tapt
  for brukeren også. Det bør den gjøre ved oppstart.
- **Serie-synk-køen** (`/v1/stats`) er fortsatt ikke bygget, og spørsmålet fra
  runde 12 står: eier bloben eller `/v1/stats` sannheten om seriene?
- **`bf_version()` i `bestefar_ffi.h`** — ubesvart fra v0.14 og framover.
