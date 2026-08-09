"""
Kunngjoering av felt dyr (backend_spec §3).

Bakgrunn: `kills[]` i §3-modellen kunne ikke leveres. Jaktloggen ligger inne i
den klient-krypterte backup-bloben (§2), som serveren ikke kan lese, og aa
synke jaktposter som egne rader ville lagt hele loggen i klartekst hos oss.

Loesningen er en FLYKTIG kunngjoering: brukeren trykker «del», vennene faar en
push - «Arnold har felt et villsvin i Molde» - og saa er det borte. Ingen
PendingMessage-rad, ingenting i basen om hva som ble felt eller hvor. Naar en
venn ikke hadde telefonen paa, gikk meldingen tapt, og det er MENINGEN: dette
er en gladmelding i oeyeblikket, ikke en logg.

Derfor bruker denne modulen push.send direkte og ikke teamgov.varsle - den
siste legger alltid inn en koerad foerst, som er riktig for §11-varsler og
feil her.
"""
import logging
from datetime import timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import or_, select
from sqlalchemy.orm import Session as OrmSession

from ..db import db
from ..deps import current_user
from ..models import (Device, Friendship, FriendshipStatus, NameStatus,
                      SharingPreference, User, utcnow)
from ..services import teamgov

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/hunts", tags=["jakt"])

# Bremser gjentatte kunngjoeringer. Lavt nok til at to dyr paa samme jakt kan
# meldes hver for seg, hoeyt nok til at et dobbelttrykk eller en kaapefull
# klient ikke spammer vennelista.
MIN_INTERVALL = timedelta(minutes=5)


class AnnounceIn(BaseModel):
    species: str = Field(min_length=1, max_length=40)
    # Stedet er valgfritt og kommer fra klienten for DENNE meldingen - det er
    # ikke profilens home_kommune. Brukeren ser teksten foer den sendes.
    kommune: str | None = Field(default=None, max_length=64)


def _venner(s: OrmSession, user_id: str) -> list[str]:
    rader = s.scalars(select(Friendship).where(
        Friendship.status == FriendshipStatus.accepted,
        or_(Friendship.requester_id == user_id,
            Friendship.addressee_id == user_id)))
    return [r.addressee_id if r.requester_id == user_id else r.requester_id
            for r in rader]


class Kunngjort(BaseModel):
    """Svarmodell (se routers/auth.py)."""
    status: str = Field(examples=["sendt"])
    message: str = Field(description="Setningen vennene faktisk fikk. Klienten "
                                     "viste den samme teksten som "
                                     "forhaandsvisning foer sending.")
    devices_notified: int = Field(
        description="Antall ENHETER som fikk varselet, ikke antall venner. "
                    "Klienten skal ikke love brukeren mer enn det som skjedde.")


@router.post("/announce", response_model=Kunngjort)
def announce_kill(body: AnnounceIn, tasks: BackgroundTasks,
                  user: User = Depends(current_user),
                  s: OrmSession = Depends(db)) -> dict:
    """
    Sender «{navn} har felt {art} i {kommune}» til vennene som push.

    Artikkelen ligger i `species`-strengen klienten sender («et villsvin»), ikke
    her - boeyningen er klientens (Announce.speciesPhrase). Serveren limer bare
    sammen, saa forhaandsvisningen brukeren saa er ordrett den vennene faar.

    Krever `share_kills` (§3). Bryteren fantes allerede i delingsvalgene, og
    det er den samme brukeren sa ja til - vi lager ingen ny.
    """
    prefs = s.get(SharingPreference, user.id)
    if prefs is None or not prefs.share_kills:
        raise HTTPException(403, "Du har ikke slått på deling av felte dyr.")

    naa = utcnow()
    if user.hunt_announced_at is not None \
            and naa - user.hunt_announced_at < MIN_INTERVALL:
        raise HTTPException(429, "Du kunngjorde nettopp et dyr. Vent noen "
                                 "minutter før du sender en ny.")

    # Er navnet ikke godkjent, bruker vi den samme noeytrale plassholderen som
    # ellers (§3). Et avvist navn skal ikke kunne snike seg ut i en push.
    navn = (user.display_name if user.display_name_status == NameStatus.approved
            else "En venn")
    sted = f" i {body.kommune}" if body.kommune else ""
    tekst = f"{navn} har felt {body.species}{sted}."

    venn_ider = _venner(s, user.id)
    tokens = []
    if venn_ider:
        tokens = list(s.scalars(
            select(Device.push_token).where(Device.user_id.in_(venn_ider))))

    # Utsendingen ligger UTENFOR forespoerselen: jegeren som nettopp felte et
    # dyr skal ikke staa og vente paa at tjue venner faar varsel. Her betyr det
    # ekstra mye - dette er det ene §11-varselet som er en gladmelding i
    # oeyeblikket, og oeyeblikket er nettopp naa.
    if tokens:
        tasks.add_task(teamgov.send_og_rydd, tokens, "Felt dyr", tekst,
                       {"kind": "hunt_announced"})

    user.hunt_announced_at = naa
    s.commit()
    # `devices_notified` er antall ENHETER varselet ble sendt TIL, ikke antall
    # som fikk det: utsendingen skjer etter at dette svaret er sendt. Klienten
    # skal ikke love brukeren mer enn det - og ingen av delene er en garanti,
    # push er varsel og ikke leveranse.
    return {"status": "sendt", "message": tekst, "devices_notified": len(tokens)}
