"""Helsesjekk. Skal ALLTID svare 200 saa lenge prosessen lever - databasens
tilstand rapporteres i kroppen, ikke som statuskode. Ellers ville en midlertidig
DB-feil faatt Fly til aa rulle tilbake en ellers frisk deploy."""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from .. import db as database
from ..config import Settings, settings
from ..db import db
from ..models import BackupKeyEscrow
from ..services import escrow, mailer, push

log = logging.getLogger(__name__)

router = APIRouter(tags=["drift"])


def _database_status() -> str:
    cfg = settings()
    # SQLite i produksjon = filen ligger i containeren og forsvinner ved
    # omstart. Da svarer SELECT 1 fint, men dataene er tapt - derfor et eget
    # svar i stedet for "ok".
    if cfg.is_prod and cfg.database_url.startswith("sqlite"):
        return "feilkonfigurert (DATABASE_URL mangler)"
    return "ok" if database.ping() else "utilgjengelig"


def _escrow_status(cfg: Settings, s: OrmSession) -> str:
    """
    Sier om BACKUP_ESCROW_SECRET fortsatt er den radene er kryptert med.

    Uten dette ville en hemmelighet satt feil - eller skiftet ut uten
    BACKUP_ESCROW_SECRET_OLD - foerst vist seg den dagen en bruker proevde aa
    gjenopprette kopien sin. Det er verst tenkelige tidspunkt aa oppdage det
    paa, og det rammer nettopp de brukerne som slo paa valget for aa slippe aa
    ta vare paa gjenopprettingskoden.

    «avvikende» er ikke i seg selv en feil mens en utskiftning paagaar: radene
    krypteres om etter hvert som de leses, og tallet skal da telle nedover.
    Staar det stille, mangler den gamle hemmeligheten.
    """
    if not escrow.er_konfigurert(cfg):
        return "av"
    naa = escrow.noekkel_id(cfg)
    try:
        avvik = s.scalar(select(func.count()).select_from(BackupKeyEscrow)
                         .where(BackupKeyEscrow.key_check != naa))
    except Exception:                                      # noqa: BLE001
        # /health skal aldri kunne velte av sin egen diagnostikk.
        log.exception("Kunne ikke lese noekkel-ID for deponerte noekler")
        return "ukjent"
    return "ok" if not avvik else f"{avvik} rader paa annen hemmelighet"


@router.get("/health")
def health(s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    return {
        "status": "ok",
        "env": cfg.env,
        "database": _database_status(),
        "mailer": mailer.backend_name(cfg),
        "push": push.backend_name(cfg),
        "escrow": _escrow_status(cfg, s),
    }
