# contracts/

Maskinlesbar kontraktflate, delt mellom områdene. Ligger i repo-roten og ikke
under `backend/` fordi UI skal kunne lese den uten å lete i en annen instans'
katalog.

| Fil | Hva |
|---|---|
| `openapi.json` | OpenAPI 3.1, generert fra FastAPI-appen. **Ikke rediger for hånd.** |

## Regenerering

```powershell
cd backend
.\.venv\Scripts\python.exe tools\gen_openapi.py            # skriv
.\.venv\Scripts\python.exe tools\gen_openapi.py --check    # bare sammenlign
```

`--check` kjøres i CI (`ci.yml`, backend-jobben) og **feiler bygget hvis fila er
utdatert mot koden**. Endrer du en rute, en Pydantic-modell eller en parameter,
må du regenerere og sjekke inn resultatet i samme commit.

Utskriften er `sort_keys=True` med fast innrykk. Uten det ville rekkefølgen
fulgt deklarasjonsrekkefølgen i koden, og en flyttet rute ville sett ut som en
kontraktsendring.

## Hva fila IKKE dekker

Dette er det viktigste avsnittet her. **`openapi.json` er ikke uttømmende**, og
et generert klientbibliotek vil derfor mangle ting som er avtalt.

### 1. Ingen svarskjemaer — for noen av endepunktene

Per 2026-08-08: **48 av 48 operasjoner med svarkropp beskriver ikke kroppen.**
Håndtererne er annotert `-> dict` eller `-> list[dict]`, og FastAPI kan bare
utlede `{"type": "object", "additionalProperties": true}` av det.

Det ene unntaket er `GET /v1/messages`, som har `response_model=list[MessageOut]`
fordi klienten måtte lese skjemaet ut av kildekoden vår (issue #5).

Praktisk betydning: fila forteller **hvilke ruter som finnes, hva de tar imot,
og hvilke parametere de har** — ikke hva de svarer med. For svarene er
`backend/KONTRAKT.md` fasit. Se `AAPNE_PUNKTER.md` ÅP-B10.

### 2. Ikke-JSON-nyttelaster

| Flate | Hvorfor OpenAPI ikke rekker |
|---|---|
| Backup-bloben (`PUT`/`GET /v1/backup`) | `application/octet-stream`. Byte-formatet — `"BFBK" \| versjon \| salt \| IV \| AES-256-GCM` — er en binærstruktur, ikke et JSON-skjema. Eies av `android/KONTRAKT.md` §3. |
| Sidecar-multiparten (`POST /v1/failed-analyses`) | Feltnavnene står som `Form(...)`-parametere og kommer med, men *forholdet* mellom sidecar-JSON-en på klientens disk og skjemafeltene gjør det ikke — og det er nettopp der de to spriker (`detected` → `detected_scores`). Eies av `backend/KONTRAKT.md` §3 og `android/KONTRAKT.md` §2. |
| Metadata i svarhoder (`GET /v1/backup`) | `X-Backup-Schema-Version`, `X-Backup-Device-Id`, `X-Backup-Client-Ts`, `X-Backup-Updated-At` er ikke deklarert. |

### 3. Autentisering

Det finnes **ingen `securitySchemes`**. `Authorization` dukker opp som en
valgfri header-parameter på hvert beskyttet endepunkt, fordi den leses med
`Header(default=None)` og ikke gjennom FastAPIs sikkerhetsavhengigheter. Et
generert klientbibliotek vil derfor ikke vite at dette er Bearer-auth, og vil
tro at headeren er valgfri. Den er ikke det — se `backend/KONTRAKT.md` §1.

`X-Debug-User-Id` er **bevisst utelatt** (`include_in_schema=False`). Den er død
i produksjon, og en testsnarvei hører ikke hjemme i en delt kontrakt der den ser
ut som en støttet innloggingsmåte.

### 4. Ingen `servers`-blokk

Fila sier ikke hvor API-et bor. Produksjon er `https://bestefar-api.fly.dev`.

### 5. Semantikk

Idempotens, hva som er trygt å prøve på nytt, hvilke statuskoder som betyr
«prøv senere» kontra «send aldri dette igjen» — ingenting av det kan uttrykkes i
et OpenAPI-dokument. `retryable`-regelen eies av `android/KONTRAKT.md` §1, og
backendens statuskodedisiplin av `backend/KONTRAKT.md` §0.

---

**Regel:** spriker `openapi.json` og et KONTRAKT-dokument, er **koden fasit** —
`openapi.json` er generert fra den. Men et sprik er nesten alltid en feil i
kontrakten som skal rettes, ikke noe man bare noterer seg.
