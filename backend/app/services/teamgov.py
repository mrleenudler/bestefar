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

from fastapi import BackgroundTasks
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..models import (ChallengeOutcome, Device, ElectionOutcome,
                      LeaderChallenge, PendingMessage, Team, TeamElection,
                      TeamMember, TeamRole, TeamVote, User, utcnow)
from . import push

log = logging.getLogger(__name__)

FRIST = timedelta(days=7)
# §11: «hvis leder har brukt appen siste maaned -> Lagleder er ikke inaktiv.»
INAKTIV_GRENSE = timedelta(days=30)


def varsle(s: OrmSession, user_ids, kind: str, title: str, body: str,
           team_id: str | None = None,
           bg: BackgroundTasks | None = None) -> None:
    """
    Legger meldinger i koen (§11) og sender push til brukerens enheter.

    Rekkefoelgen er ikke tilfeldig: koeraden legges inn FOERST. Den er
    garantien for at meldingen naar fram - push er bare det som naar brukeren
    med én gang. Feiler push, mister ingen noe (services/push.py kaster aldri).

    UTSENDINGEN LIGGER UTENFOR FORESPOERSELEN naar `bg` er gitt. Brukeren som
    bytter lagnavn skal ikke vente paa at tjue andre faar varsel. Budsjettet i
    push.send begrenset uansett aldri svartiden - sjekken skjer bare MELLOM
    kall, saa seks sekunders budsjett kunne bruke elleve. Naa er det ingen som
    venter, og budsjettet kan settes ut fra hva som er rimelig for jobben.

    Uten `bg` (tester, skript) sendes det inline, saa oppfoerselen er den samme
    bortsett fra hvem som venter.

    Enhetene slaas opp HER, mens sesjonen lever. Bare selve HTTP-kallet og
    oppryddingen av doede tokens gaar i bakgrunnen.
    """
    ider = list(user_ids)
    for uid in ider:
        s.add(PendingMessage(user_id=uid, kind=kind, title=title, body=body,
                             team_id=team_id))
    if not ider:
        return

    tokens = list(s.scalars(select(Device.push_token)
                            .where(Device.user_id.in_(ider))))
    if not tokens:
        return

    data = {"kind": kind, "team_id": team_id}
    if bg is not None:
        bg.add_task(send_og_rydd, tokens, title, body, data)
    else:
        send_og_rydd(tokens, title, body, data)


def send_og_rydd(tokens: list[str], title: str, body: str, data: dict) -> None:
    """
    Sender og rydder bort doede tokens. Kjoerer som BackgroundTask, altsaa
    ETTER at svaret er sendt - og i SAMME PROSESS.

    Doer maskinen mellom svar og utsending, gaar de gjenstaaende pushene tapt.
    Det er akseptert: koeraden er allerede committet, saa brukeren faar
    meldingen ved neste app-aapning. Se backend/KONTRAKT.md §4.

    Kaster aldri. Et varsel skal ikke kunne velte noe - og her finnes det ikke
    engang en forespoersel aa velte.
    """
    try:
        _, doede = push.send(settings(), tokens, title, body, data)
        if not doede:
            return
        # Egen sesjon: den som utloeste varselet er lukket for lengst.
        # FCM sier fra naar et token er avinstallert eller byttet ut. Rydder vi
        # ikke, vokser devices-tabellen med adresser vi aldri naar - og hver av
        # dem koster et kall av push-budsjettet ved neste varsel.
        s = next(db())
        try:
            for e in s.scalars(select(Device).where(Device.push_token.in_(doede))):
                s.delete(e)
            s.commit()
        finally:
            s.close()
    except Exception:                                      # noqa: BLE001
        log.exception("Push feilet etter svar - koeen baerer meldingen")


def medlemmer(s: OrmSession, team_id: str) -> list[TeamMember]:
    return list(s.scalars(select(TeamMember).where(TeamMember.team_id == team_id)))


def ledere(s: OrmSession, team_id: str) -> list[TeamMember]:
    return [m for m in medlemmer(s, team_id) if m.role == TeamRole.leader]


# --------------------------------------------------------------------
# Lederavstemning (§11)
# --------------------------------------------------------------------

# Andel av medlemstallet VED START som maa ha stemt for at avstemningen skal
# kunne kaare en leder. Se backend/BESLUTNINGER.md B-39 for hvorfor den finnes.
KVORUM_ANDEL = 0.25


def kvorum(election: TeamElection) -> int:
    """
    Minste antall stemmer for et gyldig utfall: 25 % av medlemstallet ved
    start, RUNDET OPP. Ingen unntak for smaa lag - 25 % av tre er én.

    0 for rader fra foer `member_count_at_open` fantes: en avstemning som
    allerede loeper skal ikke kunne bli ugyldig av en migrasjon.
    """
    n = election.member_count_at_open or 0
    return -(-n * 1 // 4) if n else 0        # ceil(n/4) uten aa dra inn math


def annuller_overkjoerte(s: OrmSession, team_id: str, kinds: list[str]) -> None:
    """
    Merker koeede meldinger som OVERKJOERT av et utfall som nettopp ble skrevet.

    Klienten henter koen ved appstart, saa en «avstemningen er aapen i 7 dager»
    kan bli hentet ni dager senere og vist rett over «avstemningen er
    avsluttet». Filtreringen ligger her og ikke i klienten fordi det er
    serveren som vet at utfallet finnes - klienten ville maattet gjenskape
    lagstyre-logikken for aa avgjoere det samme. Se backend/KONTRAKT.md §4.1.

    Raden slettes ikke, av samme grunn som en kvittering ikke sletter den.
    """
    naa = utcnow()
    for rad in s.scalars(select(PendingMessage).where(
            PendingMessage.team_id == team_id,
            PendingMessage.kind.in_(kinds),
            PendingMessage.delivered_at.is_(None),
            PendingMessage.superseded_at.is_(None))):
        rad.superseded_at = naa


def resolve_election(s: OrmSession, election: TeamElection,
                     bg: BackgroundTasks | None = None) -> TeamElection:
    # `bg` traades hele veien inn hit fordi avgjoerelsen er LAT: varselet om ny
    # lagleder oppstaar i et hvilket som helst kall som tilfeldigvis er det
    # foerste etter fristen. Den brukeren skal ikke betale svartiden for et
    # varsel til alle andre.
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

    # FRISTEN ER ABSOLUTT. Naar vi kommer hit etter `closes_at`, skrives
    # utfallet - avstemningen reaapnes ikke fordi ingen saa paa den i tide.
    # Lat avgjoerelse er en implementasjonsdetalj, ikke en utsettelse.
    def _avslutt(utfall: ElectionOutcome) -> TeamElection:
        election.outcome = utfall
        election.resolved_at = utcnow()
        annuller_overkjoerte(s, election.team_id, ["election_started"])
        s.commit()
        return election

    if not telling:
        return _avslutt(ElectionOutcome.expired)

    # Kvorum: for faa stemmer gir `expired`, samme utfall som uavgjort. Ingen
    # sperrefrist - et lederloest lag skal kunne proeve igjen med én gang.
    if len(stemmer) < kvorum(election):
        log.info("Avstemning %s naadde ikke kvorum (%d av %d kreves)",
                 election.id, len(stemmer), kvorum(election))
        return _avslutt(ElectionOutcome.expired)

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
               f"{team.name if team else 'laget'}.", election.team_id, bg)

    election.resolved_at = utcnow()
    annuller_overkjoerte(s, election.team_id, ["election_started"])
    s.commit()
    return election


def active_election(s: OrmSession, team_id: str,
                    bg: BackgroundTasks | None = None) -> TeamElection | None:
    """
    Den PAAGAAENDE avstemningen, eller None.

    Merk siste linje: `resolve_election` kan avgjoere raden i dette kallet, og
    da er den ikke lenger paagaaende. Uten sjekken returnerte funksjonen en
    ferdig avstemning som om den var aapen, og det foerste kallet etter fristen
    fikk lov til aa stemme - fristen var absolutt for alle andre enn den som
    tilfeldigvis utloeste den late avgjoerelsen.
    """
    rad = s.scalar(select(TeamElection).where(
        TeamElection.team_id == team_id,
        TeamElection.outcome == ElectionOutcome.pending))
    if rad is None:
        return None
    rad = resolve_election(s, rad, bg)
    return rad if rad.outcome == ElectionOutcome.pending else None


# --------------------------------------------------------------------
# Utfordring av inaktiv leder (§11)
# --------------------------------------------------------------------

def resolve_challenge(s: OrmSession, ch: LeaderChallenge,
                      bg: BackgroundTasks | None = None) -> LeaderChallenge:
    if ch.outcome != ChallengeOutcome.pending:
        return ch

    leder = s.get(User, ch.leader_id)
    # Logger lederen paa i loepet av fristen, avbrytes prosessen (§11).
    if leder is not None and leder.last_seen_at is not None \
            and leder.last_seen_at > ch.opened_at:
        ch.outcome = ChallengeOutcome.cancelled_leader_active
        ch.resolved_at = utcnow()
        annuller_overkjoerte(s, ch.team_id, ["leader_challenged"])
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
    annuller_overkjoerte(s, ch.team_id, ["leader_challenged"])
    team = s.get(Team, ch.team_id)
    varsle(s, [m.user_id for m in medlemmer(s, ch.team_id)],
           "leader_demoted", "Laget har ingen lagleder",
           f"{team.name if team else 'Laget'} står uten lagleder. "
           "Dere kan velge en ny.", ch.team_id, bg)
    s.commit()
    return ch


def active_challenge(s: OrmSession, team_id: str,
                     bg: BackgroundTasks | None = None) -> LeaderChallenge | None:
    """Samme forbehold som active_election: en utfordring som avgjoeres i dette
    kallet, er ikke lenger paagaaende."""
    rad = s.scalar(select(LeaderChallenge).where(
        LeaderChallenge.team_id == team_id,
        LeaderChallenge.outcome == ChallengeOutcome.pending))
    if rad is None:
        return None
    rad = resolve_challenge(s, rad, bg)
    return rad if rad.outcome == ChallengeOutcome.pending else None
