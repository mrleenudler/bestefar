"""
Registrering av enheter for push (backend_spec §11, fase 8).

Klienten melder inn FCM-tokenet sitt ved oppstart og hver gang Firebase
roterer det. Tokenet er ikke en hemmelighet i seg selv, men det er en adresse
til en bestemt telefon - derfor krever alle endepunktene innlogging, og en
bruker ser og rører bare sine egne enheter.
"""
import logging

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..db import db
from ..deps import current_user
from ..models import Device, Platform, User, utcnow

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/devices", tags=["enheter"])


class DeviceIn(BaseModel):
    push_token: str = Field(min_length=1, max_length=255)
    platform: Platform = Platform.android
    app_version: str = Field(default="", max_length=32)
    model: str = Field(default="", max_length=64)


@router.put("", status_code=200)
def register_device(body: DeviceIn, user: User = Depends(current_user),
                    s: OrmSession = Depends(db)) -> dict:
    """
    Idempotent med vilje - derfor PUT og ikke POST. Klienten kaller dette ved
    hver oppstart, og skal ikke trenge aa vite om tokenet er nytt.

    `push_token` er unik paa tvers av brukere. Byttes telefonen mellom to
    kontoer, skal raden FOELGE tokenet: ellers ville varsler til den forrige
    brukeren havnet paa en telefon som naa er logget inn som noen andre.
    """
    rad = s.scalar(select(Device).where(Device.push_token == body.push_token))
    if rad is None:
        rad = Device(user_id=user.id, push_token=body.push_token)
        s.add(rad)
    elif rad.user_id != user.id:
        log.info("Push-token flyttet til en annen konto")
        rad.user_id = user.id

    rad.platform = body.platform
    rad.app_version = body.app_version
    rad.model = body.model
    rad.last_seen_at = utcnow()
    s.commit()
    return {"id": rad.id, "platform": rad.platform.value}


@router.get("")
def list_devices(user: User = Depends(current_user),
                 s: OrmSession = Depends(db)) -> list[dict]:
    """
    Til «innlogget paa disse enhetene» i UI-et. Selve push-tokenet returneres
    IKKE - det gir ingen nytte i klienten, og et svar er et sted det kan lekke.
    """
    rader = s.scalars(select(Device).where(Device.user_id == user.id)
                      .order_by(Device.last_seen_at.desc()))
    return [{"id": r.id, "platform": r.platform.value, "model": r.model,
             "app_version": r.app_version, "last_seen_at": r.last_seen_at}
            for r in rader]


class UnregisterIn(BaseModel):
    push_token: str = Field(min_length=1, max_length=255)


@router.post("/unregister", status_code=204)
def unregister_device(body: UnregisterIn, user: User = Depends(current_user),
                      s: OrmSession = Depends(db)):
    """
    Kalles ved utlogging. Idempotent: et ukjent token gir ogsaa 204, av samme
    grunn som /v1/auth/logout - avregistrering skal aldri feile og etterlate
    klienten i tro om at den fortsatt faar varsler.

    Bare egne enheter kan avregistreres. Uten det filteret kunne hvem som helst
    skrudd av varslene til en annen ved aa gjette et token.
    """
    rad = s.scalar(select(Device).where(Device.push_token == body.push_token,
                                        Device.user_id == user.id))
    if rad is not None:
        s.delete(rad)
        s.commit()
    return Response(status_code=204)
