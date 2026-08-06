"""
Feilede analyser / OCR-donasjon (backend_spec §6), opt-in.

Klienten koer innsendinger i `filesDir/dev_uploads/{seriesId}_{tag}.jpg|json`
naar brukeren har godtatt bildedeling, og sender dem opportunistisk.

MERK: bildet skal ligge i Cloudflare R2 og bare vaere referert herfra (§6/§0.1
er tydelige paa at bilder aldri lagres i databasen). R2-opplastingen er ikke
koblet inn ennaa; inntil da lagres bildet i `image_legacy`, og endepunktet
avviser filer over `max_upload_bytes` saa databasen ikke fylles opp.

Endepunktet krever ikke innlogging: donasjonen skal fungere ogsaa for brukere
uten konto (§6 er koblet til bildedelings-samtykket, ikke til kontoen).
"""
import json

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..models import FailedAnalysis, FailedTag

router = APIRouter(prefix="/v1", tags=["feilanalyse"])


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


@router.post("/failed-analyses", status_code=201)
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

    fa = FailedAnalysis(status_code=status_code, confidence=confidence,
                        core_version=core_version, tag=tag, series_id=series_id,
                        detected_scores=_scores(detected_scores),
                        ocr_scores=_scores(ocr_scores),
                        image_legacy=data)
    s.add(fa)
    s.commit()
    return {"id": fa.id}
