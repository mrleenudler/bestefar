"""
Profil og delingsvalg (backend_spec §1, §3).

Visningsnavnet gaar gjennom moderasjon hver gang det endres (§3). Til det er
godkjent, ser andre en noeytral plassholder - ikke det nye navnet.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..deps import current_user
from ..models import NameStatus, SharingPreference, User, utcnow
from ..services import moderation

router = APIRouter(prefix="/v1/profile", tags=["profil"])


class ProfileIn(BaseModel):
    display_name: str | None = Field(default=None, max_length=64)
    birth_year: int | None = Field(default=None, ge=1900, le=2100)
    home_kommune: str | None = Field(default=None, max_length=64)
    phone: str | None = Field(default=None, max_length=20)
    findable: bool | None = None


def _profile(user: User) -> dict:
    return {
        "id": user.id,
        "public_id": user.public_id,
        "display_name": user.display_name,
        "display_name_status": user.display_name_status.value,
        "birth_year": user.birth_year,
        "home_kommune": user.home_kommune,
        "phone": user.phone,
        "findable": user.findable,
    }


@router.get("")
def get_profile(user: User = Depends(current_user)) -> dict:
    return _profile(user)


@router.put("")
def update_profile(body: ProfileIn, user: User = Depends(current_user),
                   s: OrmSession = Depends(db)) -> dict:
    advarsel = ""
    if body.display_name is not None:
        navn, status, begrunnelse = moderation.review(
            body.display_name, settings().display_name_blocklist_list)
        if status == NameStatus.rejected:
            # §3: avvist navn deles ikke, og brukeren varsles. Vi lagrer det
            # ikke i det hele tatt - da kan det heller ikke lekke.
            raise HTTPException(422, begrunnelse)
        user.display_name = navn
        user.display_name_status = status
        user.display_name_reviewed_at = utcnow() if status == NameStatus.approved else None
        if status != NameStatus.approved:
            advarsel = "Navnet vises for andre naar moderasjonen har godkjent det."

    if body.birth_year is not None:
        user.birth_year = body.birth_year
    if body.home_kommune is not None:
        user.home_kommune = body.home_kommune or None
    if body.phone is not None:
        user.phone = body.phone or None
    if body.findable is not None:
        user.findable = body.findable
    s.commit()

    ut = _profile(user)
    if advarsel:
        ut["advarsel"] = advarsel
    return ut


# --------------------------------------------------------------------
# Delingsvalg (§3)
# --------------------------------------------------------------------

FELT = ["share_phone", "share_home_kommune", "share_shots_total",
        "share_shots_season", "share_avg_score", "share_trend", "share_kills"]


class SharingIn(BaseModel):
    share_phone: bool | None = None
    share_home_kommune: bool | None = None
    share_shots_total: bool | None = None
    share_shots_season: bool | None = None
    share_avg_score: bool | None = None
    share_trend: bool | None = None
    share_kills: bool | None = None


def _prefs(s: OrmSession, user: User) -> SharingPreference:
    row = s.get(SharingPreference, user.id)
    if row is None:
        row = SharingPreference(user_id=user.id)
        s.add(row)
        s.commit()
    return row


@router.get("/sharing")
def get_sharing(user: User = Depends(current_user),
                s: OrmSession = Depends(db)) -> dict:
    row = _prefs(s, user)
    return {f: getattr(row, f) for f in FELT}


@router.put("/sharing")
def update_sharing(body: SharingIn, user: User = Depends(current_user),
                   s: OrmSession = Depends(db)) -> dict:
    """
    Deaktivering nuller delte felt (§3): siden serveren filtrerer utgaaende
    data paa disse valgene, slutter feltet aa foelge med i samme oeyeblikk -
    det ligger ingen kopi hos vennen som maa ryddes.
    """
    row = _prefs(s, user)
    for f in FELT:
        verdi = getattr(body, f)
        if verdi is not None:
            setattr(row, f, verdi)
    s.commit()
    return {f: getattr(row, f) for f in FELT}
