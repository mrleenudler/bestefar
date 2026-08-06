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
from ..models import (PositionGranularity, ResearchConsent, ResearchRecord,
                      ResearchSharingPreference, ResultType, User, utcnow)
from ..services import pseudonym, research_filter

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


def _sharing(s: OrmSession, user: User) -> ResearchSharingPreference:
    """
    Valgene ligger i BRUKERSKJEMAET, ikke i forskningsskjemaet: selve valget er
    persondata, mens forskningslageret kun skal inneholde pseudonymiserte
    observasjoner (§7). Standardverdiene deler ingenting.
    """
    row = s.get(ResearchSharingPreference, user.id)
    if row is None:
        row = ResearchSharingPreference(user_id=user.id)
        s.add(row)
        s.commit()
    return row


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
        raise HTTPException(404, "Ingen samtykke å trekke tilbake")
    c.revoked_at = utcnow()
    s.commit()
    return {"consent_id": c.id, "status": "tilbaketrukket"}


# --------------------------------------------------------------------
# Delingsvalg for jaktdata (§7)
# --------------------------------------------------------------------

SHARING_FELT = ["share_species", "share_date", "share_shot_situation"]


class ResearchSharingIn(BaseModel):
    share_species: bool | None = None
    share_date: bool | None = None
    share_shot_situation: bool | None = None
    position_granularity: PositionGranularity | None = None


def _sharing_ut(row: ResearchSharingPreference) -> dict:
    ut = {f: getattr(row, f) for f in SHARING_FELT}
    ut["position_granularity"] = row.position_granularity.value
    return ut


@router.get("/sharing")
def get_research_sharing(user: User = Depends(current_user),
                         s: OrmSession = Depends(db)) -> dict:
    return _sharing_ut(_sharing(s, user))


@router.put("/sharing")
def update_research_sharing(body: ResearchSharingIn,
                            user: User = Depends(current_user),
                            s: OrmSession = Depends(db)) -> dict:
    """
    Endringen gjelder framover. Den rydder IKKE i allerede innsendte rader:
    de er pseudonymiserte og kan ikke knyttes til kontoen herfra - det er hele
    poenget med §7. Vil brukeren ha dem bort, er veien `DELETE /v1/account`,
    som legger inn en sletteanmodning paa pseudonymet.
    """
    row = _sharing(s, user)
    for f in SHARING_FELT:
        verdi = getattr(body, f)
        if verdi is not None:
            setattr(row, f, verdi)
    if body.position_granularity is not None:
        row.position_granularity = body.position_granularity
    s.commit()
    return _sharing_ut(row)


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

    # Samtykket sier at resultattypen kan deles; delingsvalgene sier HVA av
    # den som kan deles. Begge maa gjelde, og filtreringen skjer her - ikke i
    # klienten, som kan vaere modifisert.
    pref = _sharing(s, user)
    payload = research_filter.filtrer_payload(body.result_type, body.payload, pref)
    captured_at = research_filter.filtrer_tidspunkt(body.result_type,
                                                    body.captured_at, pref)

    r = ResearchRecord(pseudonym_id=pid, session_ref=body.session_ref,
                       captured_at=captured_at, result_type=body.result_type,
                       payload_json=payload)
    s.add(r)
    s.commit()
    # `stored_fields` gjoer filtreringen synlig for klienten, saa den kan vise
    # brukeren hva som faktisk ble delt i stedet for aa paastaa noe annet.
    return {"record_id": r.id, "stored_fields": sorted(payload)}
