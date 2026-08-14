# Til utvikler — v0.26 (deponering på, Androids kopi av)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en egen seksjon nederst — ikke overskriv.

## Rekkefølgen ble holdt

Deponering på som standard er inne og bygger **før** `allowBackup="false"` ble
satt. Nettet under er på plass før det gamle ble kuttet.

## 1. Nøkkeldeponering er på som standard

Standardverdien er snudd. En bruker som *har* slått den av, blir stående av —
den lagrede verdien vinner over standardverdien, så dette overkjører ingen som
har tatt et valg.

**Teksten ved første kopi sier hva som byttes:**

> Kopien krypteres på telefonen din. Vi tar vare på nøkkelen, slik at
> gjenoppretting virker uten at du må huske noe — men det betyr også at vi kan
> låse opp kopien din. Er det ikke greit, kan du slå det av under Avanserte
> innstillinger; da er gjenopprettingskoden din eneste vei tilbake.

**Koden vises fortsatt, også med deponering på.** Den er nødutgangen hvis
brukeren senere slår deponeringen av, eller hvis vi ikke kan levere — og en
nødutgang man først får vite om når man trenger den, er ingen nødutgang.

**Slår man deponeringen av**, kommer koden fram der og da, med avkryssingen som
allerede fantes og en egen tekst:

> Nå er denne koden den ENESTE veien tilbake til sikkerhetskopien din. […]
> Mister du telefonen uten å ha koden, er kopien tapt — også for oss.

To detaljer i den flyten er valgt med vilje:

- **Koden vises før nøkkelen slettes hos oss.** Motsatt rekkefølge kan ende med
  at brukeren står uten begge deler.
- **Avbryter man dialogen, blir bryteren stående på.** Ingenting er endret, og
  skjermen skal ikke påstå noe annet. Uten en vei ut ville en påkrevd
  avkryssing vært en felle — den som ikke vil skrive ned koden, ville stått
  fast.

## 2. `android:allowBackup="false"`

**Verifisert hva som faktisk sto:** `allowBackup="true"`, og **ingen**
`dataExtractionRules` eller `fullBackupContent`. Altså standardomfang, som er
hele `filesDir`, `shared_prefs` og `databases`. Konkret:

| Hva | Innhold |
|---|---|
| `filesDir/hunts.json` | art, sted, koordinater, utfall, ettersøk |
| `filesDir/series.json` | alle serier med treffpunkter |
| `filesDir/dev_uploads/` | køede skivebilder |
| `shared_prefs/bestefar_ui.xml` | alle innstillinger, lag, venner |

Alt hos Google, i klartekst. `bestefar_secrets` var med, men er ufarlig:
innholdet er chiffertekst fra en Keystore-nøkkel som er enhetsbundet og slettes
ved avinstallering.

**Så ja — det forklarer sannsynligvis at data kom tilbake etter avinstallering
og reinstallering av v0.25.** Jeg kan ikke bevise det i etterkant, men
mekanismen var på, omfanget dekket nøyaktig de filene det gjaldt, og appens egen
kopi inneholdt som kjent ingen serier.

**Grunn 1: det motsa hele backup-designet.** Vi krypterer bloben nettopp for at
serveren ikke skal kunne lese jaktloggen, og deponerer nøkkelen bare når
brukeren sier ja. Samtidig lå den samme loggen ukryptert hos en tredjepart, uten
at noen hadde spurt. Den ene halvparten gjorde den andre meningsløs.

**Grunn 2: den gjorde gjenopprettingstestene verdiløse.** Vi kunne ikke skille
«vår gjenoppretting virket» fra «Android la det tilbake» — og gjenoppretting er
den ene funksjonen vi ikke har råd til å tro på uten bevis.

**Begrunnelsen står to steder** — i `android/ARCHITECTURE.md` og som kommentar
rett over `<application>` i manifestet — nettopp fordi den neste som ser en app
uten sikkerhetskopi vil slå den på igjen.

### En rettelse jeg skylder deg

Du hadde rett i at dette treffer v0.22-diagnosen. Der skrev jeg at
`BackupKeys.resolve` hoppet over deponeringen fordi «prefs er borte etter en
reinstallasjon». **Det holder ikke når Androids kopi er på** — da legges prefs
tilbake, og `backupEscrow` overlever.

Rettelsen i v0.22 var likevel riktig, men av en sterkere grunn enn den jeg
skrev: serverens `escrowed` er autoritativ uansett hva som skjer med lokale
preferanser, mens den lokale bryteren avhenger av en mekanisme vi verken styrer
eller kan forutsi. Kommentaren i koden og `ARCHITECTURE.md` er rettet.

### Prisen, sagt rett ut

Bytter du telefon uten å ha tatt en sikkerhetskopi i appen, er dataene borte.
Det er en reell kostnad. Den er akseptert fordi alternativet er å dele
jaktloggen med en tredjepart uten samtykke — og motvekten er at kopi nå tilbys
ved første innlogging og at deponering er på, slik at appens egen kopi er den
som virker.

## 3. Resultatet ved 0 og 0

Vises nå som dialog, ikke toast:

> **Kopien inneholdt ingen data.** Sikkerhetskopien ble hentet og låst opp, men
> den inneholdt 0 serier og 0 jaktposter — bare innstillingene dine.

En bruker uten logcat ser dermed det samme som logglinja sier.

## Verifisert

- `compileDebugKotlin` og `assembleRelease` grønt, `dist\Bestefar-0.26.apk`.
- Manifestets tidligere tilstand lest direkte, ikke antatt: `allowBackup="true"`
  og ingen regelfiler under `res/xml/`.

## Ikke verifisert

- **Ingen av de to endringene er sett i drift.** Særlig verdt å prøve:
  avinstaller og reinstaller v0.26, og se at data **ikke** kommer tilbake av seg
  selv. Det er testen som beviser at Android faktisk er ute av bildet — og
  dermed at den neste gjenopprettingstesten sier noe sant.
- Deponeringen er ikke sett skru seg på for en ny konto, og av-veien med
  kodedialogen er ikke klikket gjennom.
**Én ting jeg sjekket fordi den nye standardverdien gjør den kritisk:**
`BACKUP_ESCROW_SECRET` **står**. `/health` svarer `"escrow":"ok"` i produksjon
(sjekket 2026-08-12). Uten den ville `PUT /v1/backup/key-escrow` svart 503, og
bryteren gått tilbake til av for hver eneste nye bruker — altså ville
standardverdien vært på i teksten og av i praksis.
