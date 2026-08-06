"""
Venner (backend_spec §3, §3.1).

Vennskap krever AKSEPT hos mottaker; ingen data deles foer det. Naar det er
inngaatt, filtrerer serveren utgaaende felt paa delerens egne valg
(services/sharing.py).

Soek er den foelsomme delen: et endepunkt som svarer «finnes / finnes ikke» paa
telefonnumre er en enumereringsmaskin. Derfor: kun `findable`-brukere, eksakt
treff (ingen delvis soek paa telefon), og karantene ved gjentatte bom
(services/quarantine.py).
"""
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session as OrmSession

from ..db import db
from ..deps import current_user
from ..models import (Friendship, FriendshipStatus, QuarantineScope,
                      SharingPreference, User, utcnow)
from ..ratelimit import client_ip
from ..services import ids, quarantine, sharing

router = APIRouter(prefix="/v1", tags=["venner"])


# --------------------------------------------------------------------
# Soek (§3, §3.1)
# --------------------------------------------------------------------

def _guard(s: OrmSession, user: User, request: Request) -> str:
    """Avviser soek fra konto eller IP i karantene. Returnerer IP-en."""
    ip = client_ip(request)
    for scope, subject in ((QuarantineScope.account, user.id),
                           (QuarantineScope.ip, ip)):
        until = quarantine.blocked_until(s, scope, subject)
        if until is not None:
            raise HTTPException(429, {
                "melding": "For mange mislykkede søk. Prøv igjen senere.",
                "sperret_til": until.isoformat(),
            })
    return ip


@router.get("/users/search")
def search_users(q: str, request: Request, user: User = Depends(current_user),
                 s: OrmSession = Depends(db)) -> dict:
    """
    `q` er enten en bruker-ID (`BF-7Q4K-9F2M`) eller et telefonnummer.
    Ingen fritekstsoek paa navn: det ville gjort hele brukerbasen listbar.
    """
    ip = _guard(s, user, request)

    public_id = ids.normalize(q)
    er_id = public_id is not None
    if er_id and not ids.is_valid(q):
        # Sjekksifferet stemmer ikke - dette er en tastefeil, ikke et forsoek
        # paa aa finne noen. Vi slaar ikke opp, og teller det ikke som bom.
        raise HTTPException(422, "Ugyldig bruker-ID. Sjekk at du har skrevet riktig.")

    treff = None
    if er_id:
        treff = s.scalar(select(User).where(
            User.public_id == ids.format_id(public_id),
            User.findable.is_(True), User.deleted_at.is_(None)))
        grense = quarantine.ID_LIMIT
    else:
        treff = s.scalar(select(User).where(
            User.phone == q.strip(),
            User.findable.is_(True), User.deleted_at.is_(None)))
        grense = quarantine.PHONE_LIMIT

    if treff is None or treff.id == user.id:
        # Bare bom telles (§3.1) - treff er normal bruk.
        quarantine.register_failure(s, QuarantineScope.account, user.id, grense)
        quarantine.register_failure(s, QuarantineScope.ip, ip, grense)
        raise HTTPException(404, "Fant ingen bruker.")

    # Soeketreffet viser bare det som trengs for aa sende en forespoersel.
    # Delte statistikkfelt kommer foerst naar vennskapet er akseptert.
    return {"id": treff.id, "public_id": treff.public_id,
            "display_name": sharing.friend_view(s, treff, None)["display_name"]}


# --------------------------------------------------------------------
# Forespoersel og aksept (§3)
# --------------------------------------------------------------------

class RequestIn(BaseModel):
    user_id: str | None = Field(default=None, max_length=36)
    public_id: str | None = Field(default=None, max_length=20)


def _existing(s: OrmSession, a: str, b: str) -> Friendship | None:
    return s.scalar(select(Friendship).where(or_(
        (Friendship.requester_id == a) & (Friendship.addressee_id == b),
        (Friendship.requester_id == b) & (Friendship.addressee_id == a))))


@router.post("/friends/request", status_code=201)
def request_friend(body: RequestIn, user: User = Depends(current_user),
                   s: OrmSession = Depends(db)) -> dict:
    if body.user_id:
        andre = s.get(User, body.user_id)
    elif body.public_id and ids.is_valid(body.public_id):
        andre = s.scalar(select(User).where(
            User.public_id == ids.format_id(ids.normalize(body.public_id))))
    else:
        raise HTTPException(422, "Oppgi user_id eller en gyldig public_id.")

    if andre is None or andre.deleted_at is not None:
        raise HTTPException(404, "Fant ingen bruker.")
    if andre.id == user.id:
        raise HTTPException(422, "Du kan ikke legge til deg selv.")

    finnes = _existing(s, user.id, andre.id)
    if finnes is not None:
        if finnes.status == FriendshipStatus.accepted:
            raise HTTPException(409, "Dere er allerede venner.")
        if finnes.status == FriendshipStatus.pending:
            # Krysset forespoersel: den andre har allerede spurt deg.
            if finnes.addressee_id == user.id:
                finnes.status = FriendshipStatus.accepted
                finnes.responded_at = utcnow()
                s.commit()
                return {"id": finnes.id, "status": finnes.status.value}
            raise HTTPException(409, "Forespørsel er allerede sendt.")
        # Avslaatt tidligere - la brukeren proeve paa nytt.
        finnes.requester_id, finnes.addressee_id = user.id, andre.id
        finnes.status = FriendshipStatus.pending
        finnes.responded_at = None
        s.commit()
        return {"id": finnes.id, "status": finnes.status.value}

    rad = Friendship(requester_id=user.id, addressee_id=andre.id)
    s.add(rad)
    s.commit()
    return {"id": rad.id, "status": rad.status.value}


class RespondIn(BaseModel):
    request_id: int
    accept: bool


@router.post("/friends/respond")
def respond_friend(body: RespondIn, user: User = Depends(current_user),
                   s: OrmSession = Depends(db)) -> dict:
    rad = s.get(Friendship, body.request_id)
    # Bare mottakeren kan svare - og vi avslorer ikke at forespoerselen finnes
    # for andre enn de to involverte.
    if rad is None or rad.addressee_id != user.id:
        raise HTTPException(404, "Fant ingen forespørsel.")
    if rad.status != FriendshipStatus.pending:
        raise HTTPException(409, "Forespørselen er allerede besvart.")

    rad.status = FriendshipStatus.accepted if body.accept else FriendshipStatus.declined
    rad.responded_at = utcnow()
    s.commit()
    return {"id": rad.id, "status": rad.status.value}


@router.get("/friends/requests")
def list_requests(user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> list[dict]:
    """Innkommende, ubesvarte forespoersler."""
    rader = s.scalars(select(Friendship).where(
        Friendship.addressee_id == user.id,
        Friendship.status == FriendshipStatus.pending))
    ut = []
    for rad in rader:
        fra = s.get(User, rad.requester_id)
        ut.append({"request_id": rad.id, "created_at": rad.created_at,
                   "fra": sharing.friend_view(s, fra, None)})
    return ut


@router.get("/friends")
def list_friends(user: User = Depends(current_user),
                 s: OrmSession = Depends(db)) -> list[dict]:
    rader = s.scalars(select(Friendship).where(
        Friendship.status == FriendshipStatus.accepted,
        or_(Friendship.requester_id == user.id,
            Friendship.addressee_id == user.id)))
    ut = []
    for rad in rader:
        annen_id = (rad.addressee_id if rad.requester_id == user.id
                    else rad.requester_id)
        venn = s.get(User, annen_id)
        if venn is None or venn.deleted_at is not None:
            continue
        ut.append(sharing.friend_view(s, venn, s.get(SharingPreference, venn.id)))
    return ut


@router.delete("/friends/{user_id}", status_code=204)
def remove_friend(user_id: str, user: User = Depends(current_user),
                  s: OrmSession = Depends(db)):
    rad = _existing(s, user.id, user_id)
    if rad is not None:
        s.delete(rad)
        s.commit()
    return Response(status_code=204)
