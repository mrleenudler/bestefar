"""Helsesjekk. Skal ALLTID svare 200 saa lenge prosessen lever - databasens
tilstand rapporteres i kroppen, ikke som statuskode. Ellers ville en midlertidig
DB-feil faatt Fly til aa rulle tilbake en ellers frisk deploy."""
from fastapi import APIRouter

from .. import db as database
from ..config import settings
from ..services import mailer, push

router = APIRouter(tags=["drift"])


def _database_status() -> str:
    cfg = settings()
    # SQLite i produksjon = filen ligger i containeren og forsvinner ved
    # omstart. Da svarer SELECT 1 fint, men dataene er tapt - derfor et eget
    # svar i stedet for "ok".
    if cfg.is_prod and cfg.database_url.startswith("sqlite"):
        return "feilkonfigurert (DATABASE_URL mangler)"
    return "ok" if database.ping() else "utilgjengelig"


@router.get("/health")
def health() -> dict:
    cfg = settings()
    return {
        "status": "ok",
        "env": cfg.env,
        "database": _database_status(),
        "mailer": mailer.backend_name(cfg),
        "push": push.backend_name(cfg),
    }
