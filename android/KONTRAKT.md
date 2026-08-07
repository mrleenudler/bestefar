# Klientens kontrakt mot backend

**Eier: UI-instansen.** Klienten håndhever disse reglene, derfor eier den
teksten. `backend_spec.md` og `android/ARCHITECTURE.md` skal **peke hit**, ikke
gjenta — en regel som står tre steder, blir før eller siden tre ulike regler.

**Hva som hører hjemme her:** det andre kan stole på. Ytre formater, statuskoder,
hva som må sendes. **Ikke** hvorfor klienten valgte som den gjorde — det er
beslutninger, og de bor i `android/ARCHITECTURE.md`. Serveren lagrer bytes den
ikke kan lese, og bryr seg ikke om hvordan nøkkelen ble til.

Opprettet 2026-08-07 som del av dokumentsplitten (se `docs/ARCHITECTURE.md`).
Teksten er flyttet ordrett fra `backend_spec.md` §12/§13 og det gamle
`docs/ARCHITECTURE.md`.

---

## 1. Feilklassifisering — hva køen prøver på nytt

`retryable` = `code == 0` (nådde aldri fram), 408, 429, ≥ 500. Alt annet
(400/413/422) er permanent — køen kaster elementet i stedet for å prøve i evig
tid. Serveren bør derfor svare 4xx på data den aldri vil kunne ta imot, og
5xx/429 på alt som kan gå bra senere.

Håndheves i `Api.kt`. Konsekvensen for backend er at et *midlertidig* problem
som svares med 400 fører til stilltiende datatap hos brukeren, og et *permanent*
problem som svares med 500 gir en kø som aldri tømmes.

## 2. Sidecar-format v2 — feilanalyse-innsending

Feltnavnene mappes 1:1 til multipart-feltene i `POST /v1/failed-analyses`.

```json
{"v":2,"series_id":"<uuid>","tag":"ocr_match|ocr_mismatch|rejected",
 "status_code":0,"confidence":0.83,"core_version":"0.14",
 "detected":[10.4,9.8],"ocr":[10.4,9.9]}
```

- **`tag` ∈ {`ocr_match`, `ocr_mismatch`, `rejected`}.** Køfilene heter
  `filesDir/dev_uploads/{seriesId}_{tag}.jpg|json`, ett par per innsending.
- **`detected` er alltid poengene CV-kjernen ga**, også når OCR har overskrevet
  visningen — ellers ville en `ocr_match`-donasjon ikke si noe om hva kjernen så.
- **`confidence = -1.0`** betyr *ukjent*: sendes for kø-filer skrevet før v0.14
  (format v1 hadde bare `detected` + `tag`). Behandle som «ikke målt», ikke som
  lav konfidens.
- **`core_version`** er CV-kjernens egen versjon, hentet med `bf_version()` over
  JNI (`BestefarCore.version`). Fram til 2026-08-06 var den appens
  `versionName` — eldre innsendinger bærer den verdien.

## 3. Sikkerhetskopiens blob — bytene serveren ikke kan lese

**Ytre format** (`Backup.kt`):
`"BFBK" | 1 B versjon | 16 B salt | 12 B IV | AES-256-GCM (tag 128 bit)`.

Alt etter de fire magiske bytene og versjonsbyten er ugjennomsiktig. Ingenting av
innholdet kan valideres server-side. Hvordan nøkkelen utledes, er en
klientbeslutning som ikke angår serveren — `android/ARCHITECTURE.md`.

**Det ene serveren må vite om nøkkelen:** den er brukerens, og **serveren kan
ikke hjelpe** en bruker som har mistet den. Den ene ærlige måten å hjelpe på er
at brukeren uttrykkelig gir oss nøkkelen — se `backend_spec.md` §2.1. Ikke bygg
noen annen vei inn; en «gjenopprett kopien min» uten det samtykket finnes det
ingen implementasjon av som er sann.

**Opplasting:** klienten bruker `client_ts` = tidspunktet snapshotet ble laget,
og setter `?force=true` kun når brukeren har svart ja på «overskriv den nyere
kopien». 409 vises som en egen dialog, ikke som en feil.

**Grense:** 16 MB.

## 4. Nøkkeldeponering — hva `key_material` inneholder

`key_material` er base64 av ugjennomsiktige byte (≤ 512 byte). Serveren bryr seg
ikke om det er en nøkkel eller en gjenopprettingskode, og skal ikke tolke dem.

**503 betyr «ikke slått på på serveren»**, og vises som det — ikke som en feil
brukeren har gjort. Bryteren går tilbake til av hvis `PUT` ikke svarer 2xx.

## 5. Gravsteiner sendes, de utelates ikke

En slettet serie eller jaktpost forsvinner ikke fra det som går over ledningen —
den sendes som gravstein. Uten den finnes ingen forskjell på «har aldri
eksistert» og «brukeren slettet den», og last-write-wins per post-ID kan ikke
håndheves ved gjenoppretting.

I bloben ligger `series`/`hunts` derfor **rå**, altså inkludert de slettede.
**Når §5-synken kommer, gjelder det samme der: slettede poster sendes som
gravstein, ikke utelates.**

Hvordan gravsteinene er representert internt, og hvordan visningskoden skjermes
fra dem, står i `android/ARCHITECTURE.md`.

## 6. Øktoppførsel serveren kan regne med

Serverens tyverideteksjon (`backend_spec.md` §1, §14) hviler på disse tre. Uten
dem ser normal bruk ut som et tokentyveri, og brukeren logges ut overalt.

- **Aldri to `/v1/auth/refresh` parallelt med samme token.** Fornyelse er
  serialisert bak én lås, og en tråd som ventet sjekker om tokenet allerede er
  fornyet før den prøver selv.
- **Én omprøving etter 401, ikke flere.** Kall mot `/v1/auth/*` prøves aldri om
  igjen, så et 401 fra `/refresh` ikke utløser fornyelse i ring.
- **Utlogging skjer i fast rekkefølge:** `POST /v1/devices/unregister` først —
  mens access-tokenet fortsatt virker — så `POST /v1/auth/logout`, og deretter
  slettes begge tokenene lokalt uansett utfall, også offline. En utlogging som
  avbrytes av dårlig dekning skal ikke etterlate en telefon som fortsetter å få
  varsler.

Hvorfor det er løst slik, og hvor tokenene ligger: `android/ARCHITECTURE.md`.

---

## Hvor resten står

| Tema | Eier |
|---|---|
| Endepunktene selv, tokens, kvoter | `backend_spec.md` |
| Klientens interne arkitektur og begrunnelser | `android/ARCHITECTURE.md` |
| CV-kontrakten (`BfResult`, statuskoder) | `core/ARCHITECTURE.md` |
| Skjermflyt | `docs/flytskjema.md` |
