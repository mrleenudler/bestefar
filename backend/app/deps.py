"""
Felles FastAPI-avhengigheter.

`current_user` godtar to ting:

  Authorization: Bearer <access-token>   den ekte veien (fase 3, §1)
  X-Debug-User-Id: <uuid>                KUN utenfor produksjon

Debug-headeren er igjen fordi testene for §2-§11 ble skrevet foer innloggingen
fantes, og fordi den gjoer det mulig aa proevekjoere endepunkter lokalt uten aa
sette opp Google-klienter. Den er DOED i produksjon - sjekken staar foerst, og
det finnes ingen konfigurasjon som slaar den paa.
"""
from datetime import timedelta

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session as OrmSession

from .config import settings
from .db import db
from .models import SharingPreference, User, utcnow
from .services import ids, tokens


def current_user(authorization: str | None = Header(default=None),
                 x_debug_user_id: str | None = Header(default=None),
                 s: OrmSession = Depends(db)) -> User:
    cfg = settings()

    if authorization:
        skjema, _, token = authorization.partition(" ")
        if skjema.lower() != "bearer" or not token:
            raise HTTPException(401, "Forventet «Authorization: Bearer <token>».")
        user = s.get(User, tokens.les_access_token(cfg, token))
        return _sjekk(s, user)

    if cfg.is_prod or not x_debug_user_id:
        raise HTTPException(401, "Innlogging kreves.")

    user = s.get(User, x_debug_user_id)
    if user is None:
        user = User(id=x_debug_user_id, public_id=ids.generate(),
                    display_name=f"Testbruker {x_debug_user_id[:6]}")
        s.add(user)
        # MAA skylles foer raden som peker paa den. Det finnes ingen
        # relationship() mellom User og SharingPreference, saa SQLAlchemy har
        # ingen rekkefoelge aa sortere etter og kan sette inn barnet foerst.
        s.flush()
        s.add(SharingPreference(user_id=user.id))
        s.commit()
    return _sjekk(s, user)


def _sjekk(s: OrmSession, user: User | None) -> User:
    """
    En slettet konto (§9) skal ikke ha tilgang. Access-tokenet kan ikke
    tilbakekalles, saa sjekken MAA skje ved hvert kall - ellers har den
    slettede kontoen tilgang helt til tokenet utloeper.
    """
    if user is None or user.deleted_at is not None:
        raise HTTPException(401, "Kontoen finnes ikke.")
    touch(s, user)
    return user


# §11 bruker «har leder brukt appen siste maaned?» til aa avgjoere om en
# lagleder er inaktiv, saa aktivitet MAA registreres. Vi skriver ikke ved hvert
# kall - én skriving per time er rikelig naar grensen er 30 dager.
TOUCH_INTERVAL = timedelta(hours=1)


def touch(s: OrmSession, user: User) -> None:
    now = utcnow()
    if user.last_seen_at is None or now - user.last_seen_at > TOUCH_INTERVAL:
        user.last_seen_at = now
        s.commit()
