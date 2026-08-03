"""
Backup / dataoverfoering (backend_spec §2).

Loeser «mister loggen»: appdata forsvinner ved avinstaller/reinstall uten
konto. Bloben er KLIENT-KRYPTERT - serveren lagrer og leverer ut bytes den
ikke kan lese, og gjoer ingen konfliktloesning inne i den. Den skjer
klient-side, last-write-wins per post-ID (postene har allerede UUID + ts).

Bloben sendes som raa `application/octet-stream`; metadataene foelger som
query-parametere. Det sparer base64-paaslaget (~33 %) paa en nyttelast som er
den stoerste vi haandterer.

Krever innlogging, som kommer i fase 3; se deps.current_user.
"""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..deps import current_user
from ..models import Backup, User, utcnow

router = APIRouter(prefix="/v1/backup", tags=["backup"])


def _meta(row: Backup) -> dict:
    return {
        "bytes": row.payload_bytes,
        "schema_version": row.schema_version,
        "device_id": row.device_id,
        "client_ts": row.client_ts,
        "updated_at": row.updated_at,
    }


@router.put("")
async def upload_backup(request: Request,
                        client_ts: datetime = Query(...),
                        schema_version: int = Query(1),
                        device_id: str = Query("", max_length=64),
                        force: bool = Query(False),
                        user: User = Depends(current_user),
                        s: OrmSession = Depends(db)) -> dict:
    """
    Lagrer bloben. `client_ts` er klientens tidsstempel for oeyeblikksbildet.

    VERN MOT UTDATERT ENHET: er `client_ts` eldre enn den lagrede, svarer vi 409
    i stedet for aa overskrive. Spec §2 sier last-write-wins per post-ID, men
    den regelen kan bare haandheves KLIENT-side - serveren ser ikke inn i den
    krypterte bloben. Uten dette vernet ville en gammel telefon som synker for
    foerste gang paa maaneder kunne viske ut alt som er logget siden.
    Klienten kan overstyre med `force=true` naar brukeren har tatt et
    bevisst valg (f.eks. «gjenopprett fra denne enheten»).
    """
    cfg = settings()
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > cfg.max_backup_bytes:
        raise HTTPException(413, f"Backup er stoerre enn {cfg.max_backup_bytes} byte")

    payload = await request.body()
    if not payload:
        raise HTTPException(422, "Tom backup")
    if len(payload) > cfg.max_backup_bytes:
        raise HTTPException(413, f"Backup er stoerre enn {cfg.max_backup_bytes} byte")

    row = s.get(Backup, user.id)
    if row is not None and not force and row.client_ts is not None \
            and client_ts < row.client_ts:
        raise HTTPException(409, {
            "melding": "Lagret backup er nyere enn den innsendte",
            "lagret_client_ts": row.client_ts.isoformat(),
            "innsendt_client_ts": client_ts.isoformat(),
        })

    if row is None:
        row = Backup(user_id=user.id)
        s.add(row)
    row.payload = payload
    row.payload_bytes = len(payload)
    row.schema_version = schema_version
    row.device_id = device_id
    row.client_ts = client_ts
    row.updated_at = utcnow()
    s.commit()
    return _meta(row)


@router.get("")
def download_backup(user: User = Depends(current_user),
                    s: OrmSession = Depends(db)) -> Response:
    row = s.get(Backup, user.id)
    if row is None:
        raise HTTPException(404, "Ingen backup lagret")
    return Response(
        content=row.payload,
        media_type="application/octet-stream",
        headers={
            "X-Backup-Schema-Version": str(row.schema_version),
            "X-Backup-Device-Id": row.device_id,
            "X-Backup-Client-Ts": row.client_ts.isoformat() if row.client_ts else "",
            "X-Backup-Updated-At": row.updated_at.isoformat(),
        })


@router.get("/meta")
def backup_meta(user: User = Depends(current_user),
                s: OrmSession = Depends(db)) -> dict:
    """
    Metadata uten selve bloben - slik at «har jeg en backup aa gjenopprette?»
    paa en ny telefon ikke krever nedlasting av hele nyttelasten.
    """
    row = s.get(Backup, user.id)
    if row is None:
        raise HTTPException(404, "Ingen backup lagret")
    return _meta(row)


@router.delete("", status_code=204)
def delete_backup(user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> Response:
    row = s.get(Backup, user.id)
    if row is not None:
        s.delete(row)
        s.commit()
    return Response(status_code=204)
