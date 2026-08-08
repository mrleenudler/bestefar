"""
Lagstyre: lederavstemning og utfordring av inaktiv leder (backend_spec §11).

LAT AVGJOERELSE, ikke bakgrunnsjobb: baade avstemningen og utfordringen har en
7-dagers frist, men appen har ingen jobbkjoerer. I stedet avgjoeres de foerste
gang noen ser paa dem (`resolve_*` kalles fra endepunktene). Utfallet blir det
samme - en avstemning som ingen spoer etter, har heller ingen som venter paa
svaret - og vi slipper en scheduler som maa driftes. Naar push (fase 8) er paa
plass, boer et periodisk kall likevel legges inn saa varselet gaar ut paa
fristen og ikke ved neste besoek.
"""
import logging
from collections import Counter
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..models import (ChallengeOutcome, Device, ElectionOutcome,
                      LeaderChallenge, PendingMessage, Team, TeamElection,
                      TeamMember, TeamRole, TeamVote, User, utcnow)
from . import push

log = logging.getLogger(__name__)

FRIST = timedelta(days=7)
# §11: «hvis leder har brukt appen siste maaned -> Lagleder er ikke inaktiv.»
INAKTIV_GRENSE = timedelta(days=30)


def varsle(s: OrmSession, user_ids, kind: str, title: str, body: str,
           team_id: str | None = None) -> None:
    """
    Legger meldinger i koen (§11) og sender push til brukerens enheter.

    Rekkefoelgen er ikke tilfeldig: koeraden legges inn FOERST. Den er
    garantien for at meldingen naar fram - push er bare det som naar brukeren
    med én gang. Feiler push, mister ingen noe (services/push.py kaster aldri).

    Push skjer synkront, inne i forespoerselen. Det koster litt svartid, men
    alternativet - en bakgrunnstraad - er verre her: Fly skalerer til null, og
    en traad som ligger og venter blir drept naar maskinen sovner. Budsjettet i
    push.send holder kostnaden nede for store lag.
    """
    ider = list(user_ids)
    for uid in ider:
        s.add(PendingMessage(user_id=uid, kind=kind, title=title, body=body,
                             team_id=team_id))
    _push(s, ider, kind, title, body, team_id)


def _push(s: OrmSession, user_ids: list[str], kind: str, title: str,
          body: str, team_id: str | None) -> None:
    if not user_ids:
        return
    # Belte og seler: push.send kaster ikke, men et varsel skal uansett aldri
    # kunne velte operasjonen som utloeste det. Koeraden er allerede lagt inn,
    # saa det verste som skjer her er at brukeren maa aapne appen selv.
    try:
        enheter = list(s.scalars(select(Device).where(Device.user_id.in_(user_ids))))
        if not enheter:
            return

        _, doede = push.send(settings(), [e.push_token for e in enheter], title,
                             body, {"kind": kind, "team_id": team_id})

        # FCM sier fra naar et token er avinstallert eller byttet ut. Rydder vi
        # ikke, vokser devices-tabellen med adresser vi aldri naar - og hver av
        # dem koster et kall av push-budsjettet ved neste varsel.
        for e in enheter:
            if e.push_token in doede:
                s.delete(e)
    except Exception:                                      # noqa: BLE001
        log.exception("Push feilet under varsling (%s) - koeen baerer meldingen", kind)


def medlemmer(s: OrmSession, team_id: str) -> list[TeamMember]:
    return list(s.scalars(select(TeamMember).where(TeamMember.team_id == team_id)))


def ledere(s: OrmSession, team_id: str) -> list[TeamMember]:
    return [m for m in medlemmer(s, team_id) if m.role == TeamRole.leader]


# --------------------------------------------------------------------
# Lederavstemning (§11)
# --------------------------------------------------------------------

def resolve_election(s: OrmSession, election: TeamElection) -> TeamElection:
    if election.outcome != ElectionOutcome.pending:
        return election

    stemmer = list(s.scalars(select(TeamVote)
                             .where(TeamVote.election_id == election.id)))
    antall_medlemmer = len(medlemmer(s, election.team_id))
    telling = Counter(v.candidate_id for v in stemmer)
    forfalt = utcnow() >= election.closes_at

    # §11: «enstemmighet avslutter tidlig».
    enstemmig = (len(stemmer) == antall_medlemmer and antall_medlemmer > 0
                 and len(telling) == 1)

    if not (forfalt or enstemmig):
        return election

    if not telling:
        election.outcome = ElectionOutcome.expired
        election.resolved_at = utcnow()
        s.commit()
        return election

    vinner, topp = telling.most_common(1)[0]
    # Uavgjort ved fristen avgjoeres ikke av terningkast - avstemningen
    # utloeper, og laget kan starte en ny.
    if sum(1 for a in telling.values() if a == topp) > 1:
        election.outcome = ElectionOutcome.expired
    else:
        election.outcome = ElectionOutcome.elected
        election.elected_user_id = vinner
        rad = s.scalar(select(TeamMember).where(
            TeamMember.team_id == election.team_id, TeamMember.user_id == vinner))
        if rad is not None:
            rad.role = TeamRole.leader
        team = s.get(Team, election.team_id)
        navn = s.get(User, vinner)
        varsle(s, [m.user_id for m in medlemmer(s, election.team_id)],
               "leader_elected", "Ny lagleder",
               f"{navn.display_name} er valgt som lagleder for "
               f"{team.name if team else 'laget'}.", election.team_id)

    election.resolved_at = utcnow()
    s.commit()
    return election


def active_election(s: OrmSession, team_id: str) -> TeamElection | None:
    rad = s.scalar(select(TeamElection).where(
        TeamElection.team_id == team_id,
        TeamElection.outcome == ElectionOutcome.pending))
    return resolve_election(s, rad) if rad is not None else None


# --------------------------------------------------------------------
# Utfordring av inaktiv leder (§11)
# --------------------------------------------------------------------

def resolve_challenge(s: OrmSession, ch: LeaderChallenge) -> LeaderChallenge:
    if ch.outcome != ChallengeOutcome.pending:
        return ch

    leder = s.get(User, ch.leader_id)
    # Logger lederen paa i loepet av fristen, avbrytes prosessen (§11).
    if leder is not None and leder.last_seen_at is not None \
            and leder.last_seen_at > ch.opened_at:
        ch.outcome = ChallengeOutcome.cancelled_leader_active
        ch.resolved_at = utcnow()
        s.commit()
        return ch

    if utcnow() < ch.deadline_at:
        return ch

    # Lederen mister lederstatus, men forblir MEDLEM (§11).
    rad = s.scalar(select(TeamMember).where(
        TeamMember.team_id == ch.team_id, TeamMember.user_id == ch.leader_id))
    if rad is not None:
        rad.role = TeamRole.member
    ch.outcome = ChallengeOutcome.leader_demoted
    ch.resolved_at = utcnow()
    team = s.get(Team, ch.team_id)
    varsle(s, [m.user_id for m in medlemmer(s, ch.team_id)],
           "leader_demoted", "Laget har ingen lagleder",
           f"{team.name if team else 'Laget'} står uten lagleder. "
           "Dere kan velge en ny.", ch.team_id)
    s.commit()
    return ch


def active_challenge(s: OrmSession, team_id: str) -> LeaderChallenge | None:
    rad = s.scalar(select(LeaderChallenge).where(
        LeaderChallenge.team_id == team_id,
        LeaderChallenge.outcome == ChallengeOutcome.pending))
    return resolve_challenge(s, rad) if rad is not None else None
