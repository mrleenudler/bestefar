# backend/ — arbeidsinstruks

**Les rot-`CLAUDE.md` først**, særlig §7 (feller og lærdommer). Det som står der
gjentas ikke her: eierskapsregelen, issue-flyten, PowerShell-syntaks,
ISO-datoer, de delte filene og de tverrgående fellene.

Denne fila dekker bare det som er særegent for backend-området.

## Bygg og kjøring

Backenden har sitt **eget** virtuelle miljø. Prosjektets `.venv` i roten er
CV-pipelinen (`cv2`, `scipy`) — de skal ikke blandes.

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q
backend\.venv\Scripts\python.exe -m uvicorn app.main:app --reload   # fra backend\
backend\.venv\Scripts\python.exe -m alembic upgrade head            # fra backend\
```

Førstegangsoppsett og driftskommandoer (Fly, secrets, deploy-token) står i
`backend/README.md`.

**Migrasjoner skrives for hånd.** `alembic revision --autogenerate` er ikke
brukbart her: mot SQLite rapporterer den hele `research`-skjemaet som slettet,
fordi `schema_translate_map` gjør det usynlig for refleksjon. Kjør den mot
Postgres hvis du vil ha et utkast, og les det som et utkast.

`tests/test_migrations.py` kjører bare mot Postgres, av samme grunn. CI kjører
derfor hele suiten to ganger — SQLite og Postgres.

## Hvor tingene står

| Fil | Hva |
|---|---|
| `../backend_spec.md` | Kravene. Backend eier §0–§13; §14–§16 er pekere til klienten. |
| `backend/KONTRAKT.md` | Det vi garanterer utad — statuskoder, idempotens, grenser |
| `backend/BESLUTNINGER.md` | Hvorfor det ble slik, og hva som ble forkastet |
| `backend/CHANGELOG.md` | Når det ble bygget. **Datostemplede notater hører hit, ikke inn i spec-paragrafene** — der gjorde de det umulig å lese en paragraf som en beskrivelse av nåtilstanden. |
| `backend/README.md` | Oppsett, drift, datamodell, migrasjoner |
| `../contracts/openapi.json` | Generert fra appen. **Regenerer og sjekk inn i samme commit** når du endrer en rute, en Pydantic-modell eller en parameter — CI feiler ellers. `tools/gen_openapi.py`. |
| `../AAPNE_PUNKTER.md` | Det som ikke kan besluttes i kode. Backend-punktene er ÅP-B*, driftspunktene ÅP-E* |
| `../docs/ARCHITECTURE.md` | Bygg/CI for alle tre områder |

Det finnes **ingen** `backend/ARCHITECTURE.md`. Backend-arkitekturen er
`backend_spec.md` pluss `BESLUTNINGER.md`; ikke opprett en tredje.

## Invarianter — omgjøres ikke uten at eier har sagt fra

Disse er kjøpt dyrt eller følger av personvernkravene. Rører du en av dem, skal
det stå i `BESLUTNINGER.md` hvorfor, og den gamle begrunnelsen skal bli stående.

1. **Forsknings-pseudonymet avledes, det lagres aldri.**
   HMAC-SHA256(hemmelighet, `user_id`) i `services/pseudonym.py`. En
   oppslagstabell ville vært den reversible koblingen spec §7 forbyr. Følgen er
   at `RESEARCH_PSEUDONYM_SECRET` ikke kan roteres uten å miste koblingen til
   alt som allerede er samlet inn.
2. **Forskningsskjemaet har ingen fremmednøkler til brukertabellene.** Egen
   Postgres-`schema`, ikke bare egne tabeller.
3. **`research_filter.py` er en tillatelsesliste.** Ukjente nøkler droppes
   stille. Blir den en forbudsliste, deles hvert nytt felt som standard til
   noen kommer på å forby det.
4. **Serveren ser aldri inn i backup-bloben.** Ingen validering av innholdet,
   ingen konfliktløsning per post.
5. **Uten `JWT_SECRET` utstedes ingenting** — 503, aldri en standardverdi.
   Under 32 byte avvises også (RFC 7518 §3.2).
6. **`deleted_at` sjekkes ved hvert kall.** Access-tokenet kan ikke
   tilbakekalles, så uten sjekken har en slettet konto tilgang til tokenet
   utløper.
7. **Ved kontosletting tømmes brukerraden, den slettes ikke.** `public_id` må
   bli stående så den ikke gjenbrukes av en ny konto.
8. **Veien ut av et personvernvalg kan aldri feile på en driftsinnstilling.**
   `DELETE /v1/backup/key-escrow` virker uten `BACKUP_ESCROW_SECRET`.
9. **Push feiler aldri oppover.** Meldingskøen er garantien; push er
   bekvemmeligheten. En nede FCM-tjeneste skal ikke kunne blokkere et lagbytte.
10. **Ratebegrensning som skal telle riktig, ligger i basen.** Fly kjører to
    maskiner; en teller i minnet dobler grensen i praksis. `ratelimit.py` er
    fortsatt i minnet og er derfor kjent unøyaktig — se ÅP-B9.
11. **`X-Debug-User-Id` er død i produksjon.** Sjekken står først, og det finnes
    ingen konfigurasjon som slår den på. Den er også utelatt fra
    `contracts/openapi.json` — en testsnarvei skal ikke stå i en delt kontrakt
    og se ut som en støttet innloggingsmåte.
12. **`contracts/openapi.json` regenereres i samme commit som endringen.** En
    innsjekket kontrakt som stille kommer i utakt med koden er verre enn ingen
    kontrakt: da bygger UI mot noe som ser autoritativt ut uten å være det.

## Hva de andre eier

Du leser gjerne koden deres. Du redigerer den ikke — issue med label
`kjerne`/`ui` (rot-`CLAUDE.md` §2.1, form: issue #1).

| Eies av | Hva det betyr for deg |
|---|---|
| **ui** — `android/KONTRAKT.md` | `retryable`-klassifiseringen: `code 0`, 408, 429, ≥ 500 prøves på nytt, alt annet kastes. **Du må rette deg etter den:** 5xx/429 på alt som kan gå bra senere, 4xx kun på det du aldri vil kunne ta imot. Et midlertidig problem besvart med 400 gir stilltiende datatap hos brukeren. |
| **ui** — `android/KONTRAKT.md` §6 | Klienten garanterer at to `/v1/auth/refresh` aldri går parallelt. Tyverideteksjonen din hviler på det. |
| **ui** — `android/KONTRAKT.md` §2, §3, §5 | Sidecar-formatet, blobens ytre format, at gravsteiner sendes. Du eier **mottaket** — feltnavnene i multipart-skjemaet er dine (`KONTRAKT.md` §3 her). |
| **kjerne** — `core/ARCHITECTURE.md` | `core_version` i donasjonene kommer fra `bf_version()`. Kolonnen tar imot hva som helst; ikke valider formen. |

## Test-etikette

- **Aldri mot produksjonsdatabasen.** `conftest` gjør `alembic downgrade base`.
- **Les grenser fra `settings()`**, ikke hardkod dem. En test som låste kvoten
  til 5 feilet da kvoten ble hevet til 10 — på kvoten, ikke på oppførselen.
- **Innfører du en tidsgrense: sjekk hvem som allerede kaller endepunktet i
  løkke.** Sperrefristen på «send ny kode» brøt ni innloggingstester.
- Lager du en engine i en test eller i `env.py`, må du `dispose()` den.
