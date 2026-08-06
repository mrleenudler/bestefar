"""
Backup / dataoverfoering (backend_spec §2).

Loeser «mister loggen»: appdata forsvinner ved avinstaller/reinstall uten
konto. Bloben er KLIENT-KRYPTERT - serveren lagrer og leverer ut bytes den
ikke kan lese, og gjoer ingen konfliktloesning inne i den. Den skjer
klient-side, last-write-wins per post-ID (postene har allerede UUID + ts).

Bloben sendes som raa `application/octet-stream`; metadataene foelger som
query-parametere. Det sparer base64-paaslaget (~33 %) paa en nyttelast som er
den stoerste vi haandterer.

NOEKKELFORVALTNINGEN ER TREDELT (§2/§13):
  1. Block Store paa telefonen - standardveien, ett trykk, ingen kode.
  2. Den 20-tegns gjenopprettingskoden - noedutgangen naar telefonen er borte.
  3. FRIVILLIG deponering hos oss (`/v1/backup/key-escrow`) - det ENESTE
     tilfellet der serveren kan lese bloben. Av som standard.

Krever innlogging, som kommer i fase 3; se deps.current_user.
"""
import base64
import binascii
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session as OrmSession

from ..config import Settings, settings
from ..db import db
from ..deps import current_user
from ..models import Backup, BackupKeyEscrow, User, as_utc, utcnow
from ..services import escrow

router = APIRouter(prefix="/v1/backup", tags=["backup"])


def _meta(s: OrmSession, row: Backup) -> dict:
    return {
        "bytes": row.payload_bytes,
        "schema_version": row.schema_version,
        "device_id": row.device_id,
        "client_ts": row.client_ts,
        "updated_at": row.updated_at,
        # §2: lar en ny telefon si «kopien kan gjenopprettes uten kode» FOER
        # den laster ned 16 MB, i stedet for aa be om koden foerst og oppdage
        # etterpaa at den ikke trengtes.
        "escrowed": s.get(BackupKeyEscrow, row.user_id) is not None,
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
        raise HTTPException(413, f"Backup er større enn {cfg.max_backup_bytes} byte")

    # Klienten sender ofte ISO-tid uten offset; tolk den som UTC saa den kan
    # sammenlignes med den lagrede verdien.
    client_ts = as_utc(client_ts)

    payload = await request.body()
    if not payload:
        raise HTTPException(422, "Tom backup")
    if len(payload) > cfg.max_backup_bytes:
        raise HTTPException(413, f"Backup er større enn {cfg.max_backup_bytes} byte")

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
    return _meta(s, row)


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
    return _meta(s, row)


@router.delete("", status_code=204)
def delete_backup(user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> Response:
    """
    Sletter BLOBEN. En eventuell deponert noekkel blir staaende - den aapner
    ingenting naar bloben er borte, og valget «gjenopprett uten kode» er en
    innstilling brukeren har slaatt paa, ikke en foelge av at det finnes en
    kopi akkurat naa. Ville vi slettet noekkelen her, ville neste opplasting
    stilltiende vaert udeponert selv om bryteren stod paa.
    Kontosletting (§9) fjerner begge deler.
    """
    row = s.get(Backup, user.id)
    if row is not None:
        s.delete(row)
        s.commit()
    return Response(status_code=204)


# --------------------------------------------------------------------
# Noekkeldeponering (§2, §13) - FRIVILLIG
# --------------------------------------------------------------------

class EscrowIn(BaseModel):
    # Base64 fordi materialet er byte, ikke tekst. Grensen er romslig nok til
    # en 32-byte noekkel eller en 20-tegns gjenopprettingskode, og for liten
    # til at endepunktet kan misbrukes som lagringsplass.
    key_material: str = Field(min_length=4, max_length=1024)


def _raa(body: EscrowIn, maks: int) -> bytes:
    try:
        raa = base64.b64decode(body.key_material, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(422, "key_material må være base64.") from exc
    if not raa:
        raise HTTPException(422, "key_material er tomt.")
    if len(raa) > maks:
        raise HTTPException(413, f"key_material er større enn {maks} byte.")
    return raa


def _krev_konfigurert(cfg: Settings) -> None:
    if not escrow.er_konfigurert(cfg):
        # 503, ikke 500: dette er en manglende driftsinnstilling, og svaret
        # skal si at funksjonen er av - ikke lagre noekkelen i klartekst som
        # noedloesning.
        raise HTTPException(503, "Nøkkeldeponering er ikke slått på på "
                                 "serveren. Bruk gjenopprettingskoden.")


@router.put("/key-escrow")
def deponer_noekkel(body: EscrowIn, user: User = Depends(current_user),
                    s: OrmSession = Depends(db)) -> dict:
    """
    Lagrer noekkelmaterialet til backup-bloben, kryptert i ro.

    MERK HVA DETTE BETYR: serveren har da baade bloben og noekkelen, og kan
    lese jaktloggen. Det er en bevisst avveining brukeren gjoer mot risikoen
    for aa miste alt sammen med telefonen, og UI-teksten skal si det rett ut.
    Idempotent - en ny PUT erstatter materialet.
    """
    cfg = settings()
    _krev_konfigurert(cfg)
    raa = _raa(body, cfg.max_escrow_bytes)

    row = s.get(BackupKeyEscrow, user.id)
    if row is None:
        row = BackupKeyEscrow(user_id=user.id)
        s.add(row)
    row.material = escrow.krypter(cfg, raa, user.id)
    row.updated_at = utcnow()
    s.commit()
    return {"escrowed": True, "updated_at": row.updated_at}


@router.get("/key-escrow")
def hent_noekkel(user: User = Depends(current_user),
                 s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    _krev_konfigurert(cfg)
    row = s.get(BackupKeyEscrow, user.id)
    if row is None:
        raise HTTPException(404, "Ingen nøkkel er deponert.")
    try:
        raa = escrow.dekrypter(cfg, row.material, user.id)
    except escrow.EscrowUnreadable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"key_material": base64.b64encode(raa).decode(),
            "updated_at": row.updated_at}


@router.delete("/key-escrow", status_code=204)
def slett_noekkel(user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> Response:
    """
    Idempotent, og krever IKKE at hemmeligheten er konfigurert: aa slutte aa
    deponere noekkelen skal alltid gaa gjennom. Den motsatte veien (503) ville
    laast brukeren inne i et valg hen vil ut av.
    """
    row = s.get(BackupKeyEscrow, user.id)
    if row is not None:
        s.delete(row)
        s.commit()
    return Response(status_code=204)
