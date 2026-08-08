"""
§10 Direkte melding til utvikler.

Krever ikke auth (userId er valgfri), og er derfor foerste endepunkt ut:
en ende-til-ende-test av hele kjeden klient -> API -> videresending, foer
datamodellen finnes.

Meldingen lagres foerst, videresendes deretter i bakgrunnen. Feiler
videresendingen, staar meldingen fortsatt i databasen med forward_error satt.
"""
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as OrmSession

from ..config import Settings, settings
from ..db import db
from ..models import Feedback, utcnow
from ..ratelimit import SlidingWindow, client_ip
from ..services import mailer

log = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["feedback"])

_limiter: SlidingWindow | None = None


def limiter() -> SlidingWindow:
    global _limiter
    if _limiter is None:
        _limiter = SlidingWindow(settings().feedback_rate_per_hour, 3600.0)
    return _limiter


class FeedbackIn(BaseModel):
    subject: str = Field(min_length=1, max_length=200)
    body: str = Field(min_length=1, max_length=10_000)
    app_version: str = Field(default="", max_length=32)
    device_model: str = Field(default="", max_length=64)
    user_id: str | None = Field(default=None, max_length=64)


def _forward(cfg: Settings, feedback_id: int, payload: FeedbackIn) -> None:
    from ..db import db as _db

    body = (f"{payload.body}\n\n"
            f"---\n"
            f"appVersion: {payload.app_version or '-'}\n"
            f"deviceModel: {payload.device_model or '-'}\n"
            f"userId: {payload.user_id or '-'}\n"
            f"feedbackId: {feedback_id}\n")
    error: str | None = None
    try:
        mailer.send(cfg, f"[Bestefar] {payload.subject}", body)
    except Exception as exc:                       # noqa: BLE001 - lagres paa raden
        error = str(exc)[:300]
        log.exception("Videresending av feedback %s feilet", feedback_id)

    session = next(_db())
    try:
        row = session.get(Feedback, feedback_id)
        if row is not None:
            if error is None:
                row.forwarded_at = utcnow()
            else:
                row.forward_error = error
            session.commit()
    finally:
        session.close()


class FeedbackMottatt(BaseModel):
    """Svarmodell (se routers/auth.py). 202, ikke 200: meldingen er lagret, men
    videresendingen skjer i bakgrunnen og kan feile uten at brukeren merker
    det - `forward_error` settes paa raden."""
    id: int
    status: str = Field(examples=["mottatt"])


@router.post("/feedback", status_code=202, response_model=FeedbackMottatt)
def submit_feedback(body: FeedbackIn, request: Request, tasks: BackgroundTasks,
                    s: OrmSession = Depends(db)) -> dict:
    if not limiter().allow(client_ip(request)):
        raise HTTPException(429, "For mange meldinger. Prøv igjen senere.")

    row = Feedback(subject=body.subject, body=body.body,
                   app_version=body.app_version, device_model=body.device_model,
                   user_id=body.user_id)
    s.add(row)
    s.commit()
    tasks.add_task(_forward, settings(), row.id, body)
    return {"id": row.id, "status": "mottatt"}
