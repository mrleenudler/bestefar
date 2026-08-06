"""
Kontosletting (backend_spec §9).

To lagre maa ryddes, og de kan ikke ryddes likt:

  BRUKERSKJEMAET  slettes med det samme. Alt som kan knyttes til personen -
                  serier, backup, venner, lagmedlemskap, innlogginger - er
                  borte naar kallet returnerer.

  FORSKNINGSSKJEMAET  kan vi ikke roere herfra. Radene der er pseudonymiserte,
                  og §7 forbyr en kobling tilbake til kontoen. I stedet legges
                  det inn en SLETTEANMODNING paa pseudonymet
                  (research.deletion_requests), som forskningslageret
                  etterkommer uten aa faa vite hvem kontoen tilhoerte.

Brukerraden SLETTES IKKE, men toemmes: `public_id` maa staa igjen saa den ikke
kan gjenbrukes av en ny konto. En venn som fortsatt har ID-en lagret skal ikke
plutselig se en fremmed.
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..deps import current_user
from ..models import (AuthIdentity, AuthSession, Backup, BackupKeyEscrow,
                      Device, Friendship, PendingMessage, ResearchConsent,
                      ResearchDeletionRequest, ResearchSharingPreference,
                      Series, SharingPreference, TeamMember, TeamVote, User,
                      utcnow)
from ..services import pseudonym

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/account", tags=["konto"])


def _slett_brukerdata(s: OrmSession, user_id: str) -> None:
    # Series -> Shot har cascade paa relationship-nivaa, saa ORM-sletting maa
    # brukes for den; resten er rene rader uten barn.
    for serie in s.scalars(select(Series).where(Series.user_id == user_id)):
        s.delete(serie)

    # BackupKeyEscrow er med her fordi §2 lover at deponert noekkelmateriale
    # slettes sammen med resten av brukerskjemaet. Det er det eneste stedet
    # serveren har hatt noe som kan aapne bloben.
    for modell in (AuthIdentity, AuthSession, Backup, BackupKeyEscrow, Device,
                   PendingMessage, SharingPreference,
                   ResearchSharingPreference, TeamMember):
        s.execute(delete(modell).where(modell.user_id == user_id))

    s.execute(delete(Friendship).where(
        (Friendship.requester_id == user_id) | (Friendship.addressee_id == user_id)))
    s.execute(delete(TeamVote).where(
        (TeamVote.voter_id == user_id) | (TeamVote.candidate_id == user_id)))


@router.delete("", status_code=200)
def slett_konto(user: User = Depends(current_user),
                s: OrmSession = Depends(db)) -> dict:
    cfg = settings()
    naa = utcnow()

    # Sletteanmodningen FOERST: den er det eneste vi ikke kan gjenskape hvis
    # noe feiler underveis. Pseudonymet avledes saa lenge hemmeligheten
    # finnes - ogsaa naar RESEARCH_ENABLED er av, siden brukeren kan ha sendt
    # inn data i en periode da den var paa.
    forskning_varslet = False
    try:
        pid = pseudonym.for_user(cfg, user.id)
    except pseudonym.PseudonymNotConfigured:
        # Uten hemmeligheten finnes det heller ingen forskningsdata fra denne
        # brukeren - de kunne aldri vaert lagret. Ingenting aa be om.
        log.info("Ingen pseudonym-hemmelighet - hopper over sletteanmodning")
    else:
        s.add(ResearchDeletionRequest(pseudonym_id=pid))
        # Samtykkene trekkes tilbake med det samme, saa ingenting nytt kan
        # komme inn mens anmodningen behandles.
        s.execute(update(ResearchConsent)
                  .where(ResearchConsent.pseudonym_id == pid,
                         ResearchConsent.revoked_at.is_(None))
                  .values(revoked_at=naa))
        forskning_varslet = True

    _slett_brukerdata(s, user.id)

    # Raden beholdes, men toemmes for alt som peker paa en person.
    user.display_name = "Slettet bruker"
    user.phone = None
    user.birth_year = None
    user.home_kommune = None
    user.findable = False
    user.deleted_at = naa
    s.commit()

    return {"status": "slettet",
            "research_deletion_requested": forskning_varslet}
