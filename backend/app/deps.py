"""
Felles FastAPI-avhengigheter.

`current_user` er MIDLERTIDIG: ekte innlogging (Google/Apple/e-post) kommer i
fase 3. Til da svarer alle endepunkter som krever bruker 501 i produksjon -
bevisst, saa ingenting kan tas i bruk uautentisert ved et uhell. Lokalt og i
tester kan brukeren angis med headeren `X-Debug-User-Id`, slik at
datamodellen fra fase 2 kan proevekjoeres.
"""
from datetime import timedelta

from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session as OrmSession

from .config import settings
from .db import db
from .models import SharingPreference, User, utcnow
from .services import ids


def current_user(x_debug_user_id: str | None = Header(default=None),
                 s: OrmSession = Depends(db)) -> User:
    cfg = settings()
    if cfg.is_prod or not x_debug_user_id:
        raise HTTPException(501, "Innlogging er ikke implementert ennaa (fase 3).")

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
