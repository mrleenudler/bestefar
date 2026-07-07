# Bestefar backend

Liten FastAPI-backend med TRE ATSKILTE ansvarsomraader (kravspec §5):

1. **`/v1/stats`** — brukerens egne resultatdata (statistikk-sync).
   Standard: kun treffdata, ingen bilder. Bildelagring er brukerstyrt opsjon.
2. **`/v1/failed-analyses`** — opt-in innsending av feilede/lav-konfidens
   analyser (bilde + metadata) for CV-forbedring.
3. **`/v1/research`** — forskningsdata, STRUKTURELT ADSKILT (egne tabeller,
   pseudonym skytter-ID, eksplisitt samtykke). Kravspec §6.

## Kjoere

```
pip install -r requirements.txt
uvicorn app.main:app --reload
```

SQLite som standard (`bestefar.db`); bytt `DATABASE_URL` for Postgres.

## Datamodell (§6)

- Tidsserie: `Session` -> `Series` -> `Shot`, alle tidsstemplet.
- To resultattyper: `result_type` = training | hunt (ulik personvernprofil).
- Forskning: `ResearchConsent` (type + tidspunkt + tilbaketrekking) og
  `ResearchRecord` med pseudonym `subject_id` — ingen FK til brukertabellene.
- Konkret feltinnhold i forskningsdatasettet er IKKE avklart:
  se `TODO(eier)`-markeringene i `app/models.py`.
