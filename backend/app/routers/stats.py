"""
Treningsresultater (backend_spec §5).

Opplasting er IDEMPOTENT: klienten koer usendte serier (uploaded=false) og kan
sende samme serie flere ganger uten aa lage duplikater - serie-ID-en er
klientens egen UUID og er primaernoekkel.

Krever innlogging, som kommer i fase 3; se deps.current_user.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..db import db
from ..deps import current_user
from ..models import PosModifier, Position, Series, Shot, User

router = APIRouter(prefix="/v1/stats", tags=["statistikk"])


class ShotIn(BaseModel):
    r_rel: float
    theta: float
    decimal: float
    integer: int


class SeriesIn(BaseModel):
    id: str = Field(max_length=36)              # klientens UUID
    ts: datetime
    weapon_id: str | None = None
    ammo_name: str = ""
    distance_m: int
    position: Position
    modifier: PosModifier = PosModifier.UTEN
    corrected: bool = False
    season_key: str = ""
    confidence: float | None = None
    shots: list[ShotIn] = Field(default_factory=list)


def _serialize(series: Series) -> dict:
    return {
        "id": series.id, "ts": series.ts, "weapon_id": series.weapon_id,
        "ammo_name": series.ammo_name, "distance_m": series.distance_m,
        "position": series.position.value, "modifier": series.modifier.value,
        "corrected": series.corrected, "season_key": series.season_key,
        "sum_decimal": series.sum_decimal, "sum_integer": series.sum_integer,
        "shot_count": series.shot_count,
        "shots": [{"r_rel": sh.r_rel, "theta": sh.theta,
                   "decimal": sh.decimal, "integer": sh.integer}
                  for sh in series.shots],
    }


@router.put("/series/{series_id}")
def upsert_series(series_id: str, body: SeriesIn,
                  user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> dict:
    existing = s.get(Series, series_id)
    if existing is not None and existing.user_id != user.id:
        # Ikke avsloer at ID-en finnes hos en annen bruker.
        raise HTTPException(404)

    series = existing or Series(id=series_id, user_id=user.id)
    series.ts = body.ts
    series.weapon_id = body.weapon_id
    series.ammo_name = body.ammo_name
    series.distance_m = body.distance_m
    series.position = body.position
    series.modifier = body.modifier
    series.corrected = body.corrected
    series.season_key = body.season_key
    series.confidence = body.confidence
    # Desimalpoeng har én desimal; uten avrunding blir summen 20.200000000000003.
    series.sum_decimal = round(sum(sh.decimal for sh in body.shots), 1)
    series.sum_integer = sum(sh.integer for sh in body.shots)
    series.shot_count = len(body.shots)

    if existing is None:
        s.add(series)
    else:
        # Treffene erstattes i sin helhet. Slettingen MAA skylles foer de nye
        # settes inn: SQLAlchemy legger ellers INSERT foer DELETE i samme
        # flush, og unikhetskravet (series_id, idx) brytes.
        series.shots.clear()
        s.flush()

    for i, sh in enumerate(body.shots):
        series.shots.append(Shot(idx=i, **sh.model_dump()))
    s.commit()
    return {"id": series.id, "created": existing is None}


@router.get("/series")
def list_series(user: User = Depends(current_user),
                s: OrmSession = Depends(db),
                season_key: str | None = None, limit: int = 200) -> list[dict]:
    q = select(Series).where(Series.user_id == user.id)
    if season_key:
        q = q.where(Series.season_key == season_key)
    q = q.order_by(Series.ts.desc()).limit(min(limit, 500))
    return [_serialize(row) for row in s.scalars(q)]
