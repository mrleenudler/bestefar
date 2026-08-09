"""
Meldingskoe (backend_spec §11).

Klienten henter koen ved oppstart og viser meldingene som foerste skjerm -
navneendring paa laget, fjernet fra lag, ny lagleder osv.

Koen er ikke det samme som push, og erstatter den ikke: push er varselet som
naar brukeren mens appen er lukket, mens koen er GARANTIEN for at meldingen
naar fram til slutt - ogsaa naar push feiler, tokenet er utloept eller
brukeren har skrudd av varsler.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..db import db
from ..deps import current_user
from ..models import PendingMessage, User, utcnow

router = APIRouter(prefix="/v1/messages", tags=["meldinger"])


class MessageOut(BaseModel):
    """
    Én ventende melding.

    Denne modellen finnes for at contracts/openapi.json skal beskrive SVARET og
    ikke bare forespoerselen. Resten av API-et returnerer `dict` og har derfor
    ingen svarskjema i det hele tatt - se AAPNE_PUNKTER ÅP-B10. Koeen er tatt
    foerst fordi klienten maatte lese skjemaet ut av kildekoden vaar (issue #5).
    """
    id: int = Field(description="Heltall, ikke UUID. Autoinkrement, og bare "
                                "meningsfull innenfor denne installasjonen.")
    kind: str = Field(max_length=32,
                      description="FRI STRENG, ikke et enum. Nye "
                                  "meldingstyper skal kunne legges til uten en "
                                  "ny klientutgivelse, saa en klient MAA "
                                  "behandle en ukjent verdi som gyldig og falle "
                                  "tilbake paa title + body. Verdiene i bruk i "
                                  "dag er listet i backend/KONTRAKT.md §4.1.")
    title: str = Field(max_length=120)
    body: str = Field(description="Ferdig formulert norsk brukertekst, ikke en "
                                  "noekkel klienten skal oversette.")
    team_id: str | None = Field(default=None,
                                description="UUID som streng naar meldingen "
                                            "gjelder et lag, ellers null.")
    created_at: datetime


@router.get("", response_model=list[MessageOut])
def list_messages(user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> list[dict]:
    # `superseded_at`: meldingen er overkjoert av et senere utfall og skal ikke
    # leveres. Koen hentes ved appstart, saa «avstemningen er aapen i 7 dager»
    # kan ellers bli vist ni dager for sent, rett over resultatet.
    rader = s.scalars(select(PendingMessage).where(
        PendingMessage.user_id == user.id,
        PendingMessage.delivered_at.is_(None),
        PendingMessage.superseded_at.is_(None))
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
