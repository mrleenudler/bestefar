"""
Meldingskoe (backend_spec §11).

Klienten henter koen ved oppstart og viser meldingene som foerste skjerm -
navneendring paa laget, fjernet fra lag, ny lagleder osv.

Koen er ikke det samme som push, og erstatter den ikke: push er varselet som
naar brukeren mens appen er lukket, mens koen er GARANTIEN for at meldingen
naar fram til slutt - ogsaa naar push feiler, tokenet er utloept eller
brukeren har skrudd av varsler.
"""
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..db import db
from ..deps import current_user
from ..models import PendingMessage, User, utcnow

router = APIRouter(prefix="/v1/messages", tags=["meldinger"])


@router.get("")
def list_messages(user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> list[dict]:
    rader = s.scalars(select(PendingMessage).where(
        PendingMessage.user_id == user.id,
        PendingMessage.delivered_at.is_(None))
        .order_by(PendingMessage.created_at))
    return [{"id": r.id, "kind": r.kind, "title": r.title, "body": r.body,
             "team_id": r.team_id, "created_at": r.created_at} for r in rader]


class AckIn(BaseModel):
    ids: list[int] = Field(default_factory=list)


@router.post("/ack", status_code=204)
def ack_messages(body: AckIn, user: User = Depends(current_user),
                 s: OrmSession = Depends(db)):
    """
    Klienten kvitterer naar meldingen er VIST. Serveren sletter ikke raden -
    den markeres som levert, saa en klient som krasjer mellom henting og
    visning ikke mister meldingen for godt.
    """
    if body.ids:
        for rad in s.scalars(select(PendingMessage).where(
                PendingMessage.user_id == user.id,
                PendingMessage.id.in_(body.ids),
                PendingMessage.delivered_at.is_(None))):
            rad.delivered_at = utcnow()
        s.commit()
    return Response(status_code=204)
