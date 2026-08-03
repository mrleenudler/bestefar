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
from collections import Counter
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..models import (ChallengeOutcome, ElectionOutcome, LeaderChallenge,
                      PendingMessage, Team, TeamElection, TeamMember,
                      TeamRole, TeamVote, User, utcnow)

FRIST = timedelta(days=7)
# §11: «hvis leder har brukt appen siste maaned -> Lagleder er ikke inaktiv.»
INAKTIV_GRENSE = timedelta(days=30)


def varsle(s: OrmSession, user_ids, kind: str, title: str, body: str,
           team_id: str | None = None) -> None:
    """Legger meldinger i koen (§11). Push kobles paa i fase 8; koen er
    garantien for at meldingen naar fram ogsaa naar push feiler."""
    for uid in user_ids:
        s.add(PendingMessage(user_id=uid, kind=kind, title=title, body=body,
                             team_id=team_id))


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
           f"{team.name if team else 'Laget'} staar uten lagleder. "
           "Dere kan velge en ny.", ch.team_id)
    s.commit()
    return ch


def active_challenge(s: OrmSession, team_id: str) -> LeaderChallenge | None:
    rad = s.scalar(select(LeaderChallenge).where(
        LeaderChallenge.team_id == team_id,
        LeaderChallenge.outcome == ChallengeOutcome.pending))
    return resolve_challenge(s, rad) if rad is not None else None
