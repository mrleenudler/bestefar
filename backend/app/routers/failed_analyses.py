"""
Feilede analyser / OCR-donasjon (backend_spec §6), opt-in.

Klienten koer innsendinger i `filesDir/dev_uploads/{seriesId}_{tag}.jpg|json`
naar brukeren har godtatt bildedeling, og sender dem opportunistisk.

Bildet lagres i Cloudflare R2 og er bare REFERERT herfra (`object_key`); §6/§0.1
er tydelige paa at bilder aldri lagres i databasen. Se services/objstore.py.

UTEN R2 KONFIGURERT TAR VI IKKE IMOT NOE - 503, ikke 201. Kolonnen som holdt
bildene i basen er fjernet (migrasjon a3f7c1e59b24), saa det finnes ikke lenger
et annet sted aa gjoere av dem, og aa svare 201 paa noe vi kastet ville vaert
stille datatap. Samme form som JWT_SECRET: uten det vi trenger, utsteder vi
ingenting. 503 er `retryable` hos klienten, saa koen i `dev_uploads/` beholdes.

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
    if not objstore.er_konfigurert(cfg):
        # Foer noe leses: uten lagring har donasjonen ingen steder aa gaa, og et
        # 201 ville vaert en kvittering paa noe vi kastet. Se modul-docstringen.
        raise HTTPException(503, "Bildelagringen er ikke konfigurert")
    data = await image.read()
    if len(data) > cfg.max_upload_bytes:
        raise HTTPException(413, f"Bildet er stoerre enn {cfg.max_upload_bytes} byte")
    type_og_endelse = objstore.bildetype(data)
    if type_og_endelse is None:
        raise HTTPException(415, "Filen er ikke et JPEG-, PNG- eller WebP-bilde")
    content_type, endelse = type_og_endelse

    fa = FailedAnalysis(status_code=status_code, confidence=confidence,
                        core_version=core_version, tag=tag, series_id=series_id,
                        detected_scores=_scores(detected_scores),
                        ocr_scores=_scores(ocr_scores))

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
