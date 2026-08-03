"""
Forskning (backend_spec §7) - strukturelt adskilt, samtykke-gatet.

TRE SPERRER, med vilje:
  1. `RESEARCH_ENABLED` er av som standard. §7/§9 gjoer personvernerklaering
     og en avklaring av DPIA-behovet til en FORUTSETNING for at innsamling kan
     starte - ikke en implementasjonsdetalj. Klienten har samme sperre
     (Dialogs.RESEARCH_ENABLED).
  2. Uten `RESEARCH_PSEUDONYM_SECRET` kan vi ikke pseudonymisere, og da skal
     ingenting lagres.
  3. Hver innsending krever et gyldig, ikke-tilbaketrukket samtykke for den
     aktuelle resultattypen.

Kallerens konto-ID forlater aldri dette laget: den byttes umiddelbart mot et
HMAC-avledet pseudonym, og bare pseudonymet naar forskningsskjemaet.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..deps import current_user
from ..models import ResearchConsent, ResearchRecord, ResultType, User, utcnow
from ..services import pseudonym

router = APIRouter(prefix="/v1/research", tags=["forskning"])


def _pseudonym_for(user: User) -> str:
    cfg = settings()
    if not cfg.research_enabled:
        raise HTTPException(
            503, "Forskningsinnsamling er ikke aktivert (krever personvern"
                 "erklaering og avklart DPIA-behov, jf. spec §7).")
    try:
        return pseudonym.for_user(cfg, user.id)
    except pseudonym.PseudonymNotConfigured as exc:
        raise HTTPException(503, str(exc)) from exc


class ConsentIn(BaseModel):
    consent_type: ResultType


@router.post("/consent", status_code=201)
def grant_consent(body: ConsentIn, user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> dict:
    pid = _pseudonym_for(user)
    existing = s.scalar(select(ResearchConsent).where(
        ResearchConsent.pseudonym_id == pid,
        ResearchConsent.consent_type == body.consent_type))
    if existing is not None:
        existing.revoked_at = None          # gjenopptatt samtykke
        s.commit()
        return {"consent_id": existing.id, "status": "fornyet"}

    c = ResearchConsent(pseudonym_id=pid, consent_type=body.consent_type)
    s.add(c)
    s.commit()
    return {"consent_id": c.id, "status": "gitt"}


@router.delete("/consent/{consent_type}", status_code=200)
def revoke_consent(consent_type: ResultType, user: User = Depends(current_user),
                   s: OrmSession = Depends(db)) -> dict:
    """Samtykke skal alltid kunne trekkes tilbake (§0 «samtykke styrer alt»)."""
    pid = _pseudonym_for(user)
    c = s.scalar(select(ResearchConsent).where(
        ResearchConsent.pseudonym_id == pid,
        ResearchConsent.consent_type == consent_type))
    if c is None:
        raise HTTPException(404, "Ingen samtykke aa trekke tilbake")
    c.revoked_at = utcnow()
    s.commit()
    return {"consent_id": c.id, "status": "tilbaketrukket"}


class ResearchIn(BaseModel):
    session_ref: str = Field(max_length=64)
    captured_at: datetime
    result_type: ResultType
    # TODO(eier): konkret feltinnhold ikke avklart (kjerne-spec §6) - payload
    # er midlertidig beholder til feltdefinisjonene foreligger.
    payload: dict = Field(default_factory=dict)


@router.post("/records", status_code=201)
def submit_research(body: ResearchIn, user: User = Depends(current_user),
                    s: OrmSession = Depends(db)) -> dict:
    pid = _pseudonym_for(user)
    consent = s.scalar(select(ResearchConsent).where(
        ResearchConsent.pseudonym_id == pid,
        ResearchConsent.consent_type == body.result_type,
        ResearchConsent.revoked_at.is_(None)))
    if consent is None:
        raise HTTPException(403, "Mangler gyldig samtykke for denne resultattypen")

    r = ResearchRecord(pseudonym_id=pid, session_ref=body.session_ref,
                       captured_at=body.captured_at, result_type=body.result_type,
                       payload_json=body.payload)
    s.add(r)
    s.commit()
    return {"record_id": r.id}
