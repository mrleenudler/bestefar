# Til utvikler — v0.24 (kontoraden)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en egen seksjon nederst — ikke overskriv.

## Verifisert: eksisterende konto med lokaldelen som navn

**Ingenting skjer ved neste innlogging. Navnet står.**

`_finn_eller_lag_bruker` returnerer tidlig når identiteten finnes
(`auth.py`, `return rad.user, False, rad`). Den oppdaterer `rad.email` hvis
leverandøren har fått ny adresse, men rører aldri `user.display_name`.
`_foerste_navn` kalles bare fra `_ny_bruker`, altså kun for kontoer som faktisk
opprettes.

**Det er riktig oppførsel**, og verdt å si høyt hvorfor: å sette navnet fra
ID-tokenet ved hver innlogging ville stille tilbakestilt et navn brukeren selv
har valgt. En bruker som har rettet «ola nordmann73» til «Ola» ville fått
Google-navnet sitt tilbake neste gang de logget inn, uten at noe sa fra.

Konsekvensen er at din egen konto — opprettet før rettelsen — fortsatt heter det
den het. Veien ut kom i v0.23: feltet i profilen sender nå faktisk navnet.

## Verifisert: `email` kommer med på `/refresh`

`forny` avslutter med

```python
return _start_oekt(s, cfg, user, user_agent or oekt.user_agent, oekt.identity)
```

og `_start_oekt` legger `identitet.email` i svaret. Identiteten bæres altså
videre gjennom rotasjonen, og adressen er den samme etter en fornyelse som ved
innlogging. Kolonnen gjør jobben den ble lagt inn for.

**Klienten gjør sin del av den avtalen:** en `null` eller manglende `email`
overskriver aldri en adresse vi allerede har. `null` betyr «serveren vet ikke»
— økt startet før kolonnen fantes, eller Apple med skjult e-post — ikke
«adressen er borte». Uten den regelen ville linja tømt seg selv ved første
fornyelse, altså akkurat det kolonnen skulle hindre, bare ett lag lenger ut.

## Kontoraden

To linjer, fordi det er to forskjellige opplysninger:

```
Konto: Ola Nordmann
Logget inn med Google som ola@gmail.com
```

Øverst visningsnavnet — det vennene ser, og det som modereres. Under
kontoidentifikatoren, som **bare vises for deg**. Hele adressen, siden
`ola@gmail.com` og `ola@hotmail.com` er to forskjellige kontoer.

**Adressen kommer fra tokenparet**, som avtalt, ikke fra ID-tokenet. Grunnen er
verdt å gjenta: etter en sammenslåing på verifisert e-post kan kontoen være
knyttet til en annen adresse enn den man nettopp logget inn med, og linja ville
løyet i nøyaktig det tilfellet den finnes for.

### Leverandøren

Tokenparet har ingen `provider`, og trenger ikke ha det. Hvilket endepunkt
klienten kalte er klientens egen kunnskap om sitt eget kall — ikke en gjetning
om servertilstand. Økten er knyttet til én identitet, og fornyelse bytter den
ikke, så verdien holder så lenge økten gjør. Den settes først når økten faktisk
er lagret, og tømmes ved utlogging.

### Når vi ikke vet nok

Mangler adressen, står det bare «Logget inn med Google». Mangler begge, står det
ingenting. En halv setning om hvilken konto du er logget inn med er verre enn
ingen — da er det bedre at spørsmålet ikke ser ut som om det er besvart.

## En ting jeg la merke til, men ikke rørte

`accountEmail` og `accountProvider` er unntatt sikkerhetskopien: kopien kan
gjenopprettes på en telefon som er logget inn som noen andre, og «Logget inn
som» skal ikke komme fra kopien.

**`accountName` og `accountPublicId` er derimot med i kopien**, og har samme
problem — en gjenoppretting fra konto A på en telefon logget inn som konto B
ville overskrive dem. Det er en eldre skjevhet, den er ikke innført nå, og jeg
lot den ligge framfor å blande den inn i denne runden. Verdt en egen vurdering.

## Verifisert

- `compileDebugKotlin` og `assembleRelease` grønt, `dist\Bestefar-0.24.apk`.
- `email` lest ut av `contracts/openapi.json` (`TokenPar`, `TokenParNyBruker`):
  **nullbar og ikke i `required`**, så klienten behandler den som valgfri.
- `/refresh`-veien lest i `routers/auth.py`, ikke antatt.

## Ikke verifisert

- **Ingen innlogging er kjørt.** Linja er ikke sett på skjerm. Testen er billig:
  logg inn på nytt, og se at adressen står under kontonavnet — og at den
  fortsatt står etter en time, når access-tokenet er fornyet.
- Apple-veien (adresse mangler → bare leverandør) kan ikke prøves;
  Apple-innlogging er ikke bygget i klienten.
