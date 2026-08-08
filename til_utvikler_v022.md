# Til utvikler — v0.22 (gjenoppretting ved innlogging)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en egen seksjon nederst — ikke overskriv.

## Oppdraget

> «Speil tilbudet for eksisterende kontoer.»

## Flyten

Etter vellykket innlogging med `is_new = false`, og etter varseldialogen:

1. `GET /v1/backup/meta`.
2. **Ikke 200 → ingenting.** Ingen dialog, ingen toast. 404 (ingen kopi),
   offline, 5xx — alle behandles likt, fordi de betyr det samme for brukeren i
   dette øyeblikket: det er ingenting å tilby.
3. **200 → «Kopien ble laget 7. august kl. 09:14. Dette ERSTATTER alt …»**
4. Ja → nøkkelen skaffes → last ned, dekrypter, erstatt.

**`escrowed` avgjør om koden etterspørres.** Er nøkkelen deponert, hentes den fra
serveren, og brukeren ser aldri en kodedialog.

Ber brukeren *selv* om gjenoppretting fra Avanserte innstillinger, får de
fortsatt beskjed ved 404 og 401. Der er stillhet feil oppførsel: de har stilt et
spørsmål og skal ha svar. Forskjellen er hvem som startet det.

---

## En ekte feil funnet på veien

**`BackupKeys.resolve` spurte den lokale bryteren før den spurte serveren.**

```kotlin
if (st.backupEscrow && Auth.isLoggedIn(ctx)) { … escrowGet(ctx) … }
```

`st.backupEscrow` ligger i app-preferansene. **Preferansene er borte etter en
reinstallasjon.** På en ny telefon sto bryteren derfor av — og deponeringen ble
hoppet over i nøyaktig det scenarioet den finnes for.

Utfallet for brukeren: du slår på «Gjenopprett uten kode», mister telefonen,
kjøper ny, logger inn — og blir bedt om gjenopprettingskoden. Altså akkurat det
bryteren skulle spare deg for, i akkurat det øyeblikket den skulle virke.

Serverens `escrowed` overlever telefonen; den lokale bryteren gjør det ikke.
`resolve` tar nå imot serverens svar og bruker det, og et vellykket oppslag
setter den lokale bryteren tilbake på, så innstillingssiden slutter å påstå noe
annet enn det som er.

Dette er samme klasse som GET-som-ble-POST i v0.19: en feil som bare viser seg
i gjenopprettingssituasjonen, altså den ene gangen det virkelig gjelder, og som
ser ut som normal oppførsel resten av tiden.

## Hele flyten deles nå

Gjenopprettingen — bekreftelse, nøkkeloppslag, kodedialog, nedlasting — er
flyttet fra `AvansertActivity` til `Dialogs`. To innganger som gjør det samme
skal ikke ha hver sin kopi; det er slik de to begynner å oppføre seg ulikt uten
at noen bestemte det.

## Verifisert

- `compileDebugKotlin` og `assembleRelease` grønt, `dist\Bestefar-0.22.apk`.
- `escrowed` og `client_ts` er lest ut av `contracts/openapi.json`
  (`BackupMeta`), ikke antatt. Begge er `required` i svaret.

## Ikke verifisert

- **Flyten er ikke kjørt.** Den krever en konto som *har* en kopi, og helst en
  telefon der appdata er tømt. Den ærlige testen er: ta en kopi, slå på
  «Gjenopprett uten kode», avinstaller appen, installer på nytt, logg inn — og
  se at kopien tilbys **uten** at koden etterspørres.
- 404-stillheten er lett å bekrefte og verdt å ta med: logg inn på en konto uten
  kopi, og se at ingenting skjer.

## Fortsatt åpent

- **Ruting på `kind`** for varsler og beskjeder. Beskjeder som ber om et svar
  («Bekreft i appen») når fram, men kan ikke besvares.
- **`device_id` på backupen** — ÅP-U13.
- **Serie-synk-køen** (`/v1/stats`) er ikke bygget; ÅP-B4 om hvem som eier
  sannheten står.
