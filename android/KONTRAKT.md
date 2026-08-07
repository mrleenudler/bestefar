# Klientens kontrakt mot backend

**Eier: UI-instansen.** Klienten håndhever disse reglene, derfor eier den
teksten. `backend_spec.md` og `android/ARCHITECTURE.md` skal **peke hit**, ikke
gjenta — en regel som står tre steder, blir før eller siden tre ulike regler.

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

Serveren lagrer denne bloben ugjennomsiktig. Formatet står her bare så
backend-siden vet hva bytene *er*; ingenting av det kan valideres server-side.

**Blob-format** (`Backup.kt`):
`"BFBK" | 1 B versjon | 16 B salt | 12 B IV | AES-256-GCM (tag 128 bit)`.
Klartekst er JSON: `{v, app, ts, prefs, series[], hunts[]}` — `series`/`hunts`
er **rå**, altså inkludert soft-slettede poster.

**Nøkkel:** PBKDF2-HMAC-SHA256, 210 000 runder, over en generert
gjenopprettingskode på 20 tegn (Crockford-base32 minus I/L/O/U ⇒ 100 bit).
Ikke et brukervalgt passord: angriperen har hele bloben og kan gjette offline,
så ingen server kan bremse ham. Konsekvensen står i UI-et — mister brukeren
koden, er kopien tapt, og **serveren kan ikke hjelpe**.

Det gjelder fortsatt så lenge nøkkelen er brukerens. Den ene ærlige måten å
hjelpe på er at brukeren uttrykkelig gir oss nøkkelen — se `backend_spec.md`
§2.1. Ikke bygg noen annen vei inn; en «gjenopprett kopien min» uten det
samtykket finnes det ingen implementasjon av som er sann.

**Opplasting:** klienten bruker `client_ts` = tidspunktet snapshotet ble laget,
og setter `?force=true` kun når brukeren har svart ja på «overskriv den nyere
kopien». 409 vises som en egen dialog, ikke som en feil.

**Grense:** 16 MB.

## 4. Nøkkeldeponering — hva `key_material` inneholder

`key_material` er base64 av ugjennomsiktige byte (≤ 512 byte). Serveren bryr seg
ikke om det er en nøkkel eller en gjenopprettingskode.

Konkret fra klienten: det er **gjenopprettingskoden som ASCII, base64-kodet**.
Bryteren «Gjenopprett uten kode» er av som standard og går tilbake til av hvis
`PUT` ikke svarer 2xx, og **503 vises som «ikke slått på på serveren»**, ikke
som en feil brukeren har gjort.

## 5. Soft-delete — gravsteiner går over ledningen

`SeriesRecord.deletedAt` / `HuntRecord.deletedAt` (0 = lever). Gravsteinene
ligger i bloben, så last-write-wins per post-ID kan faktisk håndheves ved
gjenoppretting. **Når §5-synken kommer, må slettede poster sendes som gravstein
— ikke utelates.**

---

## Hvor resten står

| Tema | Eier |
|---|---|
| Endepunktene selv, tokens, kvoter | `backend_spec.md` |
| Klientens interne arkitektur | `android/ARCHITECTURE.md` |
| CV-kontrakten (`BfResult`, statuskoder) | `core/ARCHITECTURE.md` |
| Skjermflyt | `docs/flytskjema.md` |
