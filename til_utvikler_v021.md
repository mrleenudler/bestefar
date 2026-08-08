# Til utvikler — v0.21 (første sikkerhetskopi)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en egen seksjon nederst — ikke overskriv.

## Oppdraget

> «Koble is_new til tilbudet om sikkerhetskopi rett etter første innlogging.»

Punktet har stått åpent siden v0.17. Feltet lå i svaret fra alle tre
innloggingsveiene; ingen leste det.

## Én rettelse i premisset

> «gjenopprettingskoden vises én gang og ikke kan hentes igjen»

**Det stemmer ikke i denne kodebasen.** Koden ligger i `Secrets` (Android
Keystore) og kan hentes fram når som helst under Avanserte innstillinger →
Sikkerhetskopi → «Vis gjenopprettingskoden».

Det var sant i v0.15, da koden ble generert og vist som et hinder man måtte
forbi ved første kopi. Runde 13 degraderte den til nødutgang, nettopp fordi
Block Store dekker det vanlige tilfellet, og siden da har den vært en verdi man
kan slå opp.

Det som *ikke* kan hentes igjen, er koden til en telefon som er borte. Og det er
den situasjonen kravet ditt egentlig beskytter mot — så begge kravene er
oppfylt, bare med en tekst som sier det som faktisk gjelder:

- Dialogen kan **ikke avbrytes** (ikke tilbakeknapp, ikke trykk utenfor).
- Første gang er **knappen deaktivert** til brukeren har krysset av for at koden
  er skrevet ned *et annet sted enn på telefonen*.
- Teksten sier hva som går tapt: uten koden er kopien tapt **også for oss**.
- Teksten sier også at koden finnes igjen i innstillingene så lenge telefonen
  gjør det. Å påstå noe annet ville vært en løgn brukeren kan avsløre på tretti
  sekunder, og da mister resten av advarselen troverdighet.

Ber brukeren *selv* om å se koden senere, er avkryssingen borte. Da er den ren
friksjon foran noe de nettopp ba om.

## Flyten

Etter vellykket innlogging:

1. Velkomsttoast.
2. **Varseldialogen** — forklaring, så systemets tillatelsesdialog.
3. **Så**, og bare hvis serveren sa `is_new`: «Ta vare på loggen din?»
4. Ja → nøkkelen skaffes → **gjenopprettingskoden vises** → `PUT /v1/backup`.

To rekkefølgevalg er verdt å si høyt:

- **Varsler før sikkerhetskopi.** To vinduer som kappes om skjermen gir et «nei»
  til begge. Varseldialogen kommer først fordi et «nei» der er nesten permanent
  på Android 13+, mens sikkerhetskopien kan tas når som helst senere.
- **Koden før opplastingen.** En kopi på serveren som brukeren ikke har nøkkelen
  til, er det eneste utfallet som er verre enn ingen kopi.

**Bare nyopprettede kontoer får tilbudet.** Logger du inn på en konto du
allerede har, er det riktige spørsmålet «vil du gjenopprette?» — og det er en
annen flyt. Se under.

## Et flagg som ble skrevet og aldri lest

`Store.backupCodeShown` ble satt hver gang koden var vist, og ingen leste den
noen gang. Nå styrer den om avkryssingen kreves. Det er samme klasse som de tre
plassholderne i rot-`CLAUDE.md` §7.3 — en tilstand som ser ut som den betyr noe.

## Verifisert

- `compileDebugKotlin` og `assembleRelease` grønt, `dist\Bestefar-0.21.apk`.
- `is_new` er lest ut av `contracts/openapi.json` (`TokenParNyBruker`), ikke
  antatt. Feltet er `required` i alle tre innloggingssvarene.

## Ikke verifisert

- **Flyten er ikke kjørt.** Den krever en *ny* konto, og e-postkoden er den
  eneste veien dit uten en ny Google-konto. Verdt å teste med vilje: logg inn
  med en ubrukt e-postadresse og se at tilbudet kommer, at knappen er død til
  avkryssingen er gjort, og at kopien faktisk lander (`GET /v1/backup/meta`).
- Mangler `is_new` i svaret, antar klienten eksisterende konto og tilbyr
  ingenting. Det er den trygge retningen, men den er ikke prøvd.

## Fortsatt åpent

- **Gjenoppretting tilbys ikke ved innlogging på eksisterende konto.** Dette er
  nå den symmetriske mangelen: en bruker som installerer på nytt og logger inn,
  får ingen beskjed om at det ligger en kopi klar. `GET /v1/backup/meta` svarer
  allerede på spørsmålet, og `escrowed` i samme svar sier til og med om det kan
  gjøres uten kode. Neste runde bør koble det.
- Ruting på `kind` for varsler og beskjeder.
- `device_id` på backupen — ÅP-U13.
