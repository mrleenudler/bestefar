"""
Bestefar backend: tre atskilte ansvarsomraader (kravspec §5).
Skjelett - autentisering/deploy kommer senere.
"""
import os
from datetime import datetime

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as OrmSession, sessionmaker

from .models import (Base, FailedAnalysis, ResearchConsent, ResearchRecord,
                     ResultType, Series, Session, Shooter, Shot)

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./bestefar.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}
                       if DATABASE_URL.startswith("sqlite") else {})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(engine)

app = FastAPI(title="Bestefar backend", version="0.1")


def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


# ====================================================================
# 1) Statistikk-sync
# ====================================================================

class ShotIn(BaseModel):
    r_rel: float
    theta: float
    decimal: float
    integer: int


class SeriesIn(BaseModel):
    device_id: str
    started_at: datetime
    captured_at: datetime
    result_type: ResultType = ResultType.training
    sum_decimal: float
    sum_integer: int
    confidence: float
    shots: list[ShotIn] = Field(default_factory=list)


@app.post("/v1/stats/series", status_code=201)
def sync_series(body: SeriesIn, s: OrmSession = Depends(db)):
    shooter = s.scalar(select(Shooter).where(Shooter.device_id == body.device_id))
    if shooter is None:
        shooter = Shooter(device_id=body.device_id)
        s.add(shooter)
        s.flush()
    session = Session(shooter_id=shooter.id, started_at=body.started_at,
                      result_type=body.result_type)
    s.add(session)
    s.flush()
    series = Series(session_id=session.id, captured_at=body.captured_at,
                    sum_decimal=body.sum_decimal, sum_integer=body.sum_integer,
                    confidence=body.confidence)
    s.add(series)
    s.flush()
    for sh in body.shots:
        s.add(Shot(series_id=series.id, **sh.model_dump()))
    s.commit()
    return {"series_id": series.id}


@app.get("/v1/stats/{device_id}/series")
def list_series(device_id: str, s: OrmSession = Depends(db)):
    shooter = s.scalar(select(Shooter).where(Shooter.device_id == device_id))
    if shooter is None:
        raise HTTPException(404)
    out = []
    for sess in shooter.sessions:
        for ser in sess.series:
            out.append({"series_id": ser.id, "captured_at": ser.captured_at,
                        "sum_decimal": ser.sum_decimal,
                        "sum_integer": ser.sum_integer,
                        "result_type": sess.result_type.value})
    return out


# ====================================================================
# 2) Feilede analyser (opt-in)
# ====================================================================

@app.post("/v1/failed-analyses", status_code=201)
async def submit_failed(status_code: int = Form(...), confidence: float = Form(...),
                        core_version: str = Form(...),
                        image: UploadFile = File(...),
                        s: OrmSession = Depends(db)):
    fa = FailedAnalysis(status_code=status_code, confidence=confidence,
                        core_version=core_version, image=await image.read())
    s.add(fa)
    s.commit()
    return {"id": fa.id}


# ====================================================================
# 3) Forskning (strukturelt adskilt; samtykke-gatet)
# ====================================================================

class ConsentIn(BaseModel):
    subject_id: str
    consent_type: ResultType


@app.post("/v1/research/consent", status_code=201)
def grant_consent(body: ConsentIn, s: OrmSession = Depends(db)):
    c = ResearchConsent(subject_id=body.subject_id, consent_type=body.consent_type)
    s.add(c)
    s.commit()
    return {"consent_id": c.id}


class ResearchIn(BaseModel):
    subject_id: str
    session_ref: str
    captured_at: datetime
    result_type: ResultType
    # TODO(eier): konkret feltinnhold ikke avklart (kravspec §6) - payload
    # er midlertidig beholder til feltdefinisjonene foreligger.
    payload: dict = Field(default_factory=dict)


@app.post("/v1/research/records", status_code=201)
def submit_research(body: ResearchIn, s: OrmSession = Depends(db)):
    consent = s.scalar(
        select(ResearchConsent)
        .where(ResearchConsent.subject_id == body.subject_id,
               ResearchConsent.consent_type == body.result_type,
               ResearchConsent.withdrawn_at.is_(None)))
    if consent is None:
        raise HTTPException(403, "Mangler gyldig samtykke for denne resultattypen")
    r = ResearchRecord(subject_id=body.subject_id, session_ref=body.session_ref,
                       captured_at=body.captured_at, result_type=body.result_type,
                       payload_json=body.payload)
    s.add(r)
    s.commit()
    return {"record_id": r.id}
