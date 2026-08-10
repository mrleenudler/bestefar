# Til utvikler — v0.23 (visningsnavn kontra kontoidentitet)

> **Merk til de andre instansene:** denne fila deles. Legg egne notater til som
> en egen seksjon nederst — ikke overskriv.

## Svaret på de to spørsmålene

**1. Sender klienten `name` videre, og leser serveren det?** Nei og nei — men
klienten trenger ikke å sende det. Serveren har det allerede.

Klienten sender bare `{id_token}`. Serveren verifiserer tokenet og leser `sub`,
`email` og `email_verified` ut av kravene — `name` er der, men forkastes
(`oidc.py:86`, `:98`; `Identitet` har ikke feltet). Startnavnet settes i stedet
av `_navn_fra_epost`, som tar lokaldelen av adressen for **alle** nye kontoer,
også Google (`auth.py:89–96`).

Dette er derfor serverens å rette, og det er meldt som **issue #7**. Jeg har
bevisst *ikke* løst det klientside, selv om Credential Manager gir oss en
`displayName` vi kunne sendt med: da hadde visningsnavnet vært klient-oppgitt og
dermed ikke til å stole på, mens serveren allerede har det samme navnet i et
token den selv har verifisert signaturen på. Å flytte det ville byttet en
verifisert kilde mot en uverifisert.

**2. E-post i tokenparet?** Nei, og det er meldt som **issue #8**.
«Logget inn med Google som ola@gmail.com» er derfor ikke bygget. Klienten kunne
lest adressen ut av ID-tokenet før den sendes inn — men den ville da vist noe
klienten har funnet på. Kontosammenslåing på verifisert adresse betyr at kontoen
kan være knyttet til en *annen* adresse enn den man nettopp logget inn med, og
da ville skjermen løyet i nøyaktig det tilfellet den finnes for.

Kontoraden viser inntil videre navnet alene, uten å påstå at det er en adresse.

---

## Det som viste seg å være verre

**Feltet som heter «Visningsnavn» ble aldri sendt noe sted.**

Det skrev til en lokal verdi (`store.nickname`) som brukes på venne- og
lagskjermene — begge lokale skjeletter — mens serveren hadde sitt eget
`display_name`, det vennene faktisk ville sett, som brukeren ikke hadde noen vei
til å endre.

Så på spørsmålet «hvordan håndterer klienten et avvist visningsnavn?» var svaret
at den ikke kunne få et: den sendte aldri noe å avvise. Moderasjonen var
uoppnåelig fra klienten.

Det er samme mønster som de tre plassholderne i rot-`CLAUDE.md` §7.3 — en flate
som ser ut som den gjør noe, uten en sender bak.

### Hva som er bygget

`PUT /v1/profile` kalles nå, ved fokustap og ved at skjermen forlates — ikke per
tastetrykk, det ville vært ett kall per bokstav.

**Avvisning vises der og da.** Moderasjonen er synkron, så svaret er endelig når
det kommer: 200 = godkjent og lagret, 422 = avvist og **ikke** lagret. Serverens
egen begrunnelse vises ordrett, fordi den er skrevet for å leses av brukeren og
sier presist hva som må endres («Ikke tillatt: @ …»).

**Et avvist navn blir ikke stående i feltet.** Serveren lagret det ikke, så
feltet settes tilbake til navnet som faktisk gjelder. Ellers ville skjermen vist
et navn ingen andre kan se.

**Ingen «venter på moderasjon»-tilstand.** Jeg sjekket: `moderation.review`
returnerer bare `approved` eller `rejected`, aldri `pending`. Serverens
`advarsel`-felt («Navnet vises for andre når moderasjonen har godkjent det»,
`profile.py:60`) er derfor i praksis død kode. Klienten leser det ikke.

Innlogget fylles feltet fra serverens navn, ikke fra den lokale verdien — ellers
er man tilbake til to navn.

Uten konto sendes ingenting; da er navnet bare et lokalt kallenavn.

### Én ting som er verdt å vite

Klientens tegnfilter og serverens regelsett er allerede like: 24 tegn, latinske
bokstaver og tall, mellomrom og `-_.'`. Det betyr at den eneste avvisningen en
bruker realistisk kan treffe fra appen, er **blokklista** — altså nettopp den
som må vises, og den som ikke kan forutses lokalt.

## Verifisert

- `compileDebugKotlin` og `assembleRelease` grønt, `dist\Bestefar-0.23.apk`.
- Skjemaet for `PUT /v1/profile` er lest ut av `contracts/openapi.json`
  (`ProfileIn`), og 422-formen ut av `routers/profile.py:53–56`.

## Ikke verifisert

- **Ingen navneendring er sendt mot serveren.** Det krever konto. Testen er
  billig når du er innlogget: skriv et navn med et tegn utenfor settet, eller et
  ord fra blokklista, og se at begrunnelsen kommer i feltet med det samme og at
  navnet ikke blir stående.
- `onPause`-sendingen (navn endret, skjerm forlatt uten fokustap) er ikke prøvd
  på enhet. En avvisning kan ikke vises på en skjerm som er borte — men den blir
  ikke usynlig: neste gang profilen åpnes, fylles feltet fra serverens navn.

## Fortsatt åpent

- **#7 og #8** — begge blokkerer flater beskrevet i UI-spec §6.
- Ruting på `kind` for varsler og beskjeder.
- `device_id` på backupen — ÅP-U13.
