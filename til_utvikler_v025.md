# Til utvikler — v0.25 (datatapet ved gjenoppretting)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en egen seksjon nederst — ikke overskriv.

## Hypotesen din holdt ikke, og det er verdt å si først

**`deletedAt` er ikke årsaken.** Feltet skrives med `put("deletedAt", deletedAt)`
og leses med `optLong("deletedAt", 0L)` i både `SeriesRecord` og `HuntRecord`.
Det er symmetrisk, og en post uten feltet får `0L` — altså «i live». Jeg gikk
gjennom hele `toJson`/`fromJson` for begge typene: ingen feltnavn spriker, og
ingen påkrevd nøkkel mangler.

**Postene ble heller ikke skrevet og så filtrert bort.** `replaceAll` skriver
`series.json` og `hunts.json` direkte fra lista den får. Var lista tom, ble
filene tomme.

## Hva som faktisk skjedde, så langt koden kan fortelle

Gjenopprettingen skrev **det den fikk**, og det den fikk var ingenting.

```kotlin
store.replaceAll(
    (0 until series.length()).mapNotNull {
        try { SeriesRecord.fromJson(...) }
        catch (e: Exception) { Log.w(TAG, "Hopper over ugyldig serie", e); null }
    }, …)
```

To ting gjorde tapet mulig, og begge er mine:

1. **Parsefeil ble svelget per post.** Feilet én post, hoppet vi over den.
   Feilet *alle*, ble resultatet en tom liste — og en tom liste er ikke til å
   skille fra «kopien var tom». Begge ble skrevet som en vellykket
   gjenoppretting.
2. **Nedlasting, dekryptering og overskriving skjedde i ett kall.**
   `downloadAndRestore` gjorde alt tre. Det fantes altså ikke et øyeblikk der
   noen — hverken kode eller bruker — kunne se hva kopien inneholdt før det
   lokale var borte.

**Den mest sannsynlige kilden til den tomme kopien er førstegangstilbudet jeg
bygget i v0.21.** Det tok øyeblikksbilde rett etter innlogging uten å sjekke om
det fantes noe å ta vare på. Logget du inn før du hadde skutt en serie, la vi en
tom kopi på serveren. Den lå der og så ut som en sikkerhetskopi — `/meta` svarer
200, størrelsen er ikke null fordi innstillingene er med — helt til du
gjenopprettet fra den.

Det stemmer med det du så: kopien ble funnet, den var 151 kB, og det eneste som
kom tilbake var visningsnavnet. **Visningsnavnet ligger i `prefs`**, og prefs var
det eneste kopien inneholdt.

Jeg kan ikke bevise det uten kopien eller logcat, og sier det derfor som det
mest sannsynlige, ikke som et fastslått faktum. Se «Hvordan vi får visshet».

## Hva som er endret

- **Lesing er skilt fra skriving.** `hentOgLes` laster ned, dekrypterer og leser
  uten å røre noe. `skriv` kalles først når beslutningen er tatt.
- **En tom kopi får ikke erstatte data som finnes.** Da vises tallene og
  spørsmålet, med *avbryt* som standardvalg:

  > Sikkerhetskopien inneholder ingen serier og ingen jaktposter.
  > På denne telefonen ligger det 42 serier og 7 jaktposter. Fortsetter du, blir
  > de slettet og erstattet med ingenting.

- **Uleselige poster telles og meldes.** «5 poster i sikkerhetskopien var i et
  format appen ikke forsto.» Det er en annen beskjed enn «kopien var tom», fordi
  det er en annen situasjon.
- **Resultatet sier hva som kom tilbake:** «Gjenopprettet: 42 serier og 7
  jaktposter.»
- **Ingen kopi av ingenting.** Førstegangstilbudet hoppes over når det ikke
  finnes noe å kopiere.

## Hvordan vi får visshet

`Backup.les` logger nå én linje per gjenoppretting:

```
BestefarBackup: Kopi lest: 42 serier, 7 jaktposter, 0 hoppet over
```

Gjør du en gjenoppretting til med v0.25 og henter logcat på `BestefarBackup`,
sier den linja hvilken av de to det var. Er tallene 0/0/0, var kopien tom — da er
det v0.21-tilbudet som er skyldig. Er «hoppet over» høyt, er det parsing, og da
logges hver enkelt feil med stacktrace.

**Dataene som gikk tapt, er tapt.** Kopien på serveren inneholder ikke serier,
og telefonens filer ble overskrevet. Jeg har ikke funnet noen vei tilbake — hvis
du har en eldre APK-installasjon eller en Android-systembackup fra før, er det
den eneste sjansen.

## De tre mindre funnene

**1. `series_id` var filnavnstammen.** `{uuid}_{tag}` er 49 tegn mot serverens
`VARCHAR(36)`, og 500-svaret er `retryable` — så elementet ble stående i køen og
prøvd i det uendelige mot noe som aldri kunne gå bra. Nå sendes UUID-en alene.
Er kandidaten likevel for lang, sendes feltet tomt: det er valgfritt, og en
donasjon uten serie-ID er verdt mer enn en som aldri kommer fram.

**2. Bildegrensen sjekkes nå før noe køes.** 11 MB mot `MAX_UPLOAD_BYTES` på
8 MB. Utfallet var uansett at bildet ble kastet (413 er ikke `retryable`), men
først etter at 11 MB var kopiert inn i appens filkatalog og lastet opp. Nå
avvises det ved køing, og allerede køede storfiler kastes før opplasting.

Verdt å merke: dette betyr at *noen* donasjoner er blitt borte i felt. Grensa er
serverens, og bildene fra kameraet er tydeligvis større enn den.

**3. Fokus og kontoraden.** Trykk utenfor et tekstfelt tar nå fokus ut av det —
visningsnavnet sendes ved fokustap, så uten dette kom moderasjonssvaret aldri
mens du så på feltet.

Kontoraden viser ingen adresse fordi **økten din er eldre enn v0.24**: klienten
lagrer adresse og leverandør ved innlogging, og din innlogging skjedde før de
feltene fantes. Fornyelse gir oss adressen, men ikke leverandøren — den er
klientens egen kunnskap om sitt eget kall. Logg ut og inn igjen, så står raden
der. Jeg har lagt inn «Logget inn som \<adresse\>» for tilfellet der vi har
adressen men ikke leverandøren.

## Verifisert

- `compileDebugKotlin` og `assembleRelease` grønt, `dist\Bestefar-0.25.apk`.
- `toJson`/`fromJson` gjennomgått felt for felt i begge posttypene.

## Ikke verifisert

- **Ingen gjenoppretting er kjørt med v0.25.** Vakten mot tom kopi er ikke sett
  utløse. Den kan prøves trygt: du har nå en tom kopi på serveren og (forhåpentlig)
  data på telefonen igjen — da skal dialogen komme, og *avbryt* skal la alt stå.
- Rettelsen av `series_id` er ikke sett gi 201 mot serveren.
