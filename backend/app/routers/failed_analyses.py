"""
Feilede analyser / OCR-donasjon (backend_spec §6), opt-in.

Klienten koer innsendinger i `filesDir/dev_uploads/{seriesId}_{tag}.jpg|json`
naar brukeren har godtatt bildedeling, og sender dem opportunistisk.

Bildet lagres i Cloudflare R2 og er bare REFERERT herfra (`object_key`); §6/§0.1
er tydelige paa at bilder aldri lagres i databasen. Er R2 ikke konfigurert -
lokalt og i testene - legges bildet i `image_legacy` i stedet, og /health sier
«bilder»: «database (avvik fra spec §6)». Se services/objstore.py.

Endepunktet krever ikke innlogging: donasjonen skal fungere ogsaa for brukere
uten konto (§6 er koblet til bildedelings-samtykket, ikke til kontoen). Det er
ogsaa grunnen til at innholdet sjekkes: dette er en aapen skrivevei inn i betalt
objektlagring, saa vi tar imot bilder og ikke hva som helst.
"""
import json
import logging

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..models import FailedAnalysis, FailedTag
from ..services import objstore

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["feilanalyse"])

# Magiske byte -> (content-type, filendelse). Klienten sender JPEG; PNG og WebP
# staar her fordi de er de to andre en telefon kan produsere, ikke fordi noen
# ber om dem.
_BILDETYPER = (
    (b"\xff\xd8\xff", "image/jpeg", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", "image/png", ".png"),
)


class DonasjonMottatt(BaseModel):
    """
    Svarmodell (se routers/auth.py). MERK at multipart-INNGANGEN ikke kan
    beskrives fullt ut i OpenAPI: feltnavnene her er ikke de samme som noeklene
    i klientens sidecar-JSON (`detected` -> `detected_scores`), og den
    kartleggingen finnes bare i backend/KONTRAKT.md §3.
    """
    id: int


def _scores(raw: str) -> list:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"Ugyldig JSON i poengfeltet: {exc}") from exc
    if not isinstance(parsed, list):
        raise HTTPException(422, "Poengfeltet må være en JSON-liste")
    return parsed


def _bildetype(data: bytes) -> tuple[str, str] | None:
    """(content-type, endelse) ut fra de foerste bytene, eller None."""
    for magi, ctype, endelse in _BILDETYPER:
        if data.startswith(magi):
            return ctype, endelse
    # RIFF....WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", ".webp"
    return None


@router.post("/failed-analyses", status_code=201,
             response_model=DonasjonMottatt)
async def submit_failed(status_code: int = Form(...),
                        confidence: float = Form(...),
                        core_version: str = Form(...),
                        tag: FailedTag = Form(FailedTag.rejected),
                        series_id: str | None = Form(None),
                        detected_scores: str = Form(""),
                        ocr_scores: str = Form(""),
                        image: UploadFile = File(...),
                        s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    data = await image.read()
    if len(data) > cfg.max_upload_bytes:
        raise HTTPException(413, f"Bildet er stoerre enn {cfg.max_upload_bytes} byte")
    type_og_endelse = _bildetype(data)
    if type_og_endelse is None:
        raise HTTPException(415, "Filen er ikke et JPEG-, PNG- eller WebP-bilde")
    content_type, endelse = type_og_endelse

    fa = FailedAnalysis(status_code=status_code, confidence=confidence,
                        core_version=core_version, tag=tag, series_id=series_id,
                        detected_scores=_scores(detected_scores),
                        ocr_scores=_scores(ocr_scores))

    if not objstore.er_konfigurert(cfg):
        fa.image_legacy = data
        s.add(fa)
        s.commit()
        return {"id": fa.id}

    # Raden foerst, for id-en noekkelen navngis med - men bare i basens
    # transaksjon. Feiler opplastingen, rulles den tilbake, og det ligger ingen
    # rad igjen som peker paa et objekt som aldri ble skrevet.
    s.add(fa)
    s.flush()
    noekkel = objstore.objektnoekkel(fa.id, tag.value, endelse)
    try:
        objstore.legg(cfg, noekkel, data, content_type)
    except objstore.LagringFeilet:
        s.rollback()
        # 503 og ikke 4xx: dette kan gaa bra senere, og klienten proever igjen
        # paa alt fra 500 og opp (android/KONTRAKT.md). En 400 her ville gjort
        # en midlertidig R2-feil til stille tap av donasjonen.
        log.exception("Fikk ikke lastet opp feilanalyse-bilde til R2")
        raise HTTPException(503, "Bildelagringen er utilgjengelig - proev igjen senere")
    fa.object_key = noekkel
    s.commit()
    return {"id": fa.id}
