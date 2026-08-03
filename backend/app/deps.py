"""
Felles FastAPI-avhengigheter.

`current_user` er MIDLERTIDIG: ekte innlogging (Google/Apple/e-post) kommer i
fase 3. Til da svarer alle endepunkter som krever bruker 501 i produksjon -
bevisst, saa ingenting kan tas i bruk uautentisert ved et uhell. Lokalt og i
tester kan brukeren angis med headeren `X-Debug-User-Id`, slik at
datamodellen fra fase 2 kan proevekjoeres.
"""
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session as OrmSession

from .config import settings
from .db import db
from .models import SharingPreference, User
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
        s.add(SharingPreference(user_id=user.id))
        s.commit()
    return user
