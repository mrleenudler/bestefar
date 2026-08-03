"""
Lag: jaktlag og skytterlag (backend_spec §4, §11).

Rollemodellen er bevisst enkel: rollen ligger paa MEDLEMSKAPET, og et lag kan
ha flere ledere (§4). «Ingen leder» er en gyldig tilstand - da kan laget velge
en ny gjennom en avstemning (§11).

Frister avgjoeres LAT, se services/teamgov.py.
"""
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from ..config import settings
from ..db import db
from ..deps import current_user
from ..models import (DeliveryStatus, ElectionOutcome, InviteTarget,
                      LeaderChallenge, PendingMessage, Team, TeamElection,
                      TeamInvite, TeamKind, TeamMember, TeamRole, TeamVote,
                      User, utcnow)
from ..services import contacts, geo, mailer, teamgov

router = APIRouter(prefix="/v1/teams", tags=["lag"])


# --------------------------------------------------------------------
# Hjelpere
# --------------------------------------------------------------------

def _medlemskap(s: OrmSession, team_id: str, user_id: str) -> TeamMember:
    rad = s.scalar(select(TeamMember).where(
        TeamMember.team_id == team_id, TeamMember.user_id == user_id))
    if rad is None:
        # Vi avslorer ikke at laget finnes for noen som ikke er medlem.
        raise HTTPException(404, "Fant ikke laget.")
    return rad


def _krev_leder(s: OrmSession, team_id: str, user_id: str) -> TeamMember:
    rad = _medlemskap(s, team_id, user_id)
    if rad.role != TeamRole.leader:
        raise HTTPException(403, "Bare lagleder kan gjoere dette.")
    return rad


def _team_ut(s: OrmSession, team: Team, med_medlemmer: bool = False) -> dict:
    medl = teamgov.medlemmer(s, team.id)
    ut = {
        "id": team.id, "name": team.name, "kind": team.kind.value,
        "member_count": len(medl),
        "lat": team.lat, "lon": team.lon,
        "leaders": [m.user_id for m in medl if m.role == TeamRole.leader],
        "has_leader": any(m.role == TeamRole.leader for m in medl),
    }
    if med_medlemmer:
        ut["members"] = [
            {"user_id": m.user_id, "role": m.role.value,
             "display_name": (s.get(User, m.user_id).display_name
                              if s.get(User, m.user_id) else ""),
             "joined_at": m.joined_at}
            for m in medl]
    return ut


# --------------------------------------------------------------------
# Oppretting og oversikt (§4)
# --------------------------------------------------------------------

class TeamIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    kind: TeamKind
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    # §4: «jeg er leder» / «opprett for leder». Er den false, staar laget uten
    # leder til noen tar rollen eller laget velger en.
    i_am_leader: bool = True


@router.post("", status_code=201)
def create_team(body: TeamIn, user: User = Depends(current_user),
                s: OrmSession = Depends(db)) -> dict:
    team = Team(name=body.name.strip(), kind=body.kind, lat=body.lat, lon=body.lon,
                created_by=user.id)
    s.add(team)
    s.flush()
    # Oppretteren er alltid medlem - ogsaa naar laget opprettes PAA VEGNE av en
    # leder som ikke har appen ennaa.
    s.add(TeamMember(team_id=team.id, user_id=user.id,
                     role=TeamRole.leader if body.i_am_leader else TeamRole.member))
    s.commit()
    return _team_ut(s, team)


@router.get("")
def my_teams(user: User = Depends(current_user),
             s: OrmSession = Depends(db)) -> list[dict]:
    ider = list(s.scalars(select(TeamMember.team_id)
                          .where(TeamMember.user_id == user.id)))
    return [_team_ut(s, s.get(Team, tid)) for tid in ider]


@router.get("/near")
def teams_near(lat: float = Query(ge=-90, le=90),
               lon: float = Query(ge=-180, le=180),
               r: float = Query(50_000, gt=0),
               user: User = Depends(current_user),
               s: OrmSession = Depends(db)) -> list[dict]:
    """
    §4: inntil 20 lag innenfor radiusen, sortert etter avstand - og de 3
    naermeste er ALLTID med, uansett hvor langt unna de er. Ellers ville en
    bruker i et tynt befolket omraade faatt en tom liste.

    Alle lag med koordinater leses inn og sorteres i Python. Det holder i
    lang tid (vi snakker hundrevis av lag), men maa byttes til en
    geo-indeks - PostGIS eller geohash-kolonne - naar tabellen vokser.
    """
    kandidater = []
    for team in s.scalars(select(Team).where(Team.lat.is_not(None),
                                             Team.lon.is_not(None))):
        kandidater.append((geo.distance_m(lat, lon, team.lat, team.lon), team))
    kandidater.sort(key=lambda t: t[0])

    innenfor = [k for k in kandidater if k[0] <= r][:20]
    if len(innenfor) < 3:
        innenfor = kandidater[:3]
    return [{**_team_ut(s, team), "distance_m": round(avstand)}
            for avstand, team in innenfor]


@router.get("/{team_id}")
def team_details(team_id: str, user: User = Depends(current_user),
                 s: OrmSession = Depends(db)) -> dict:
    _medlemskap(s, team_id, user.id)      # §11: egen bruker er alltid i lista
    team = s.get(Team, team_id)
    ut = _team_ut(s, team, med_medlemmer=True)
    valg = teamgov.active_election(s, team_id)
    ut["election_open"] = valg is not None and valg.outcome == ElectionOutcome.pending
    return ut


class RenameIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)


@router.put("/{team_id}")
def rename_team(team_id: str, body: RenameIn, user: User = Depends(current_user),
                s: OrmSession = Depends(db)) -> dict:
    _krev_leder(s, team_id, user.id)
    team = s.get(Team, team_id)
    gammelt = team.name
    team.name = body.name.strip()
    if gammelt != team.name:
        # §11: alle medlemmer varsles, og meldingen vises foerst ved neste
        # app-aapning.
        teamgov.varsle(
            s, [m.user_id for m in teamgov.medlemmer(s, team_id) if m.user_id != user.id],
            "team_renamed", "Navneendring",
            f"Lagleder for {gammelt} har endret navnet paa laget til {team.name}.",
            team_id)
    s.commit()
    return _team_ut(s, team)


# --------------------------------------------------------------------
# Invitasjon (§4)
# --------------------------------------------------------------------

class InviteIn(BaseModel):
    email_or_phone: str = Field(min_length=1, max_length=255)


@router.post("/{team_id}/invite", status_code=201)
def invite(team_id: str, body: InviteIn, user: User = Depends(current_user),
           s: OrmSession = Depends(db)) -> dict:
    _medlemskap(s, team_id, user.id)
    cfg = settings()
    team = s.get(Team, team_id)

    try:
        kind, verdi = contacts.classify(body.email_or_phone)
    except contacts.UgyldigMottaker as exc:
        raise HTTPException(422, str(exc)) from exc

    token = secrets.token_urlsafe(16)
    rad = TeamInvite(team_id=team_id, inviter_id=user.id, target=verdi,
                     target_kind=kind, token=token)
    s.add(rad)

    lenke = f"{cfg.invite_base_url}/{token}"
    if kind == InviteTarget.email:
        try:
            mailer.send(cfg, f"Invitasjon til {team.name}",
                        f"{user.display_name} har invitert deg til {team.name} "
                        f"i Bestefar.\n\nAapne lenken for aa bli med:\n{lenke}\n")
            rad.delivery_status = DeliveryStatus.sent
            rad.sent_at = utcnow()
        except Exception as exc:                   # noqa: BLE001 - meldes tilbake
            rad.delivery_status = DeliveryStatus.failed
            rad.delivery_error = str(exc)[:300]
    else:
        # SMS krever betalt leverandoer og er utsatt til v2 (§1). Klienten kan
        # i stedet dele lenken med ACTION_SEND - derfor faar den lenken tilbake
        # selv om vi ikke sendte noe.
        rad.delivery_status = DeliveryStatus.failed
        rad.delivery_error = ("SMS-utsending er ikke tilgjengelig (utsatt til "
                              "v2). Del lenken fra appen i stedet.")
    s.commit()

    # §4: leveringskvittering/-feil tilbake til klienten.
    return {"invite_id": rad.id, "target_kind": kind.value, "target": verdi,
            "url": lenke, "delivery_status": rad.delivery_status.value,
            "delivery_error": rad.delivery_error}


class JoinIn(BaseModel):
    token: str = Field(min_length=1, max_length=64)


@router.post("/join")
def join_team(body: JoinIn, user: User = Depends(current_user),
              s: OrmSession = Depends(db)) -> dict:
    rad = s.scalar(select(TeamInvite).where(TeamInvite.token == body.token))
    if rad is None:
        raise HTTPException(404, "Ugyldig invitasjon.")

    finnes = s.scalar(select(TeamMember).where(
        TeamMember.team_id == rad.team_id, TeamMember.user_id == user.id))
    if finnes is None:
        s.add(TeamMember(team_id=rad.team_id, user_id=user.id))
    # Invitasjonen kan brukes av flere (den deles gjerne i en gruppechat), saa
    # vi merker foerste bruk og lar den fortsatt virke.
    if rad.accepted_at is None:
        rad.accepted_at = utcnow()
        rad.accepted_by = user.id
        rad.delivery_status = DeliveryStatus.accepted
    s.commit()
    return _team_ut(s, s.get(Team, rad.team_id))


# --------------------------------------------------------------------
# Medlemmer og lederskap (§11)
# --------------------------------------------------------------------

@router.delete("/{team_id}/members/{member_id}", status_code=204)
def remove_member(team_id: str, member_id: str, user: User = Depends(current_user),
                  s: OrmSession = Depends(db)):
    _krev_leder(s, team_id, user.id)
    rad = s.scalar(select(TeamMember).where(
        TeamMember.team_id == team_id, TeamMember.user_id == member_id))
    if rad is None:
        raise HTTPException(404, "Fant ikke medlemmet.")
    team = s.get(Team, team_id)
    s.delete(rad)
    if member_id != user.id:
        teamgov.varsle(s, [member_id], "removed_from_team", "Fjernet fra lag",
                       f"Lagleder har fjernet deg fra {team.name}.", team_id)
    s.commit()
    return Response(status_code=204)


@router.post("/{team_id}/leaders/confirm")
def confirm_leadership(team_id: str, user: User = Depends(current_user),
                       s: OrmSession = Depends(db)) -> dict:
    # MAA staa foer /leaders/{member_id}: FastAPI tar foerste rute som
    # matcher, og «confirm» ville ellers blitt lest som en medlems-ID.
    rad = _medlemskap(s, team_id, user.id)
    tilbud = s.scalar(select(PendingMessage).where(
        PendingMessage.user_id == user.id,
        PendingMessage.team_id == team_id,
        PendingMessage.kind == "leadership_offered"))
    if tilbud is None:
        raise HTTPException(404, "Du har ingen tilbud om lederskap i dette laget.")
    rad.role = TeamRole.leader
    s.delete(tilbud)
    s.commit()
    return _team_ut(s, s.get(Team, team_id))


@router.post("/{team_id}/leaders/{member_id}")
def offer_leadership(team_id: str, member_id: str,
                     user: User = Depends(current_user),
                     s: OrmSession = Depends(db)) -> dict:
    """
    §11 «Overfoer lederskap»: velg et eksisterende medlem - men rollen settes
    foerst naar den valgte BEKREFTER. Ingen skal vaakne opp som lagleder uten
    aa ha sagt ja.
    """
    _krev_leder(s, team_id, user.id)
    rad = s.scalar(select(TeamMember).where(
        TeamMember.team_id == team_id, TeamMember.user_id == member_id))
    if rad is None:
        raise HTTPException(404, "Fant ikke medlemmet.")
    if rad.role == TeamRole.leader:
        raise HTTPException(409, "Medlemmet er allerede lagleder.")

    team = s.get(Team, team_id)
    teamgov.varsle(s, [member_id], "leadership_offered", "Tilbud om lederskap",
                   f"{user.display_name} vil gjoere deg til lagleder for "
                   f"{team.name}. Bekreft i appen.", team_id)
    s.commit()
    return {"status": "avventer_bekreftelse", "member_id": member_id}


# --------------------------------------------------------------------
# Lederavstemning (§11)
# --------------------------------------------------------------------

@router.post("/{team_id}/election", status_code=201)
def start_election(team_id: str, user: User = Depends(current_user),
                   s: OrmSession = Depends(db)) -> dict:
    _medlemskap(s, team_id, user.id)
    if teamgov.ledere(s, team_id):
        raise HTTPException(409, "Laget har allerede en lagleder.")
    if teamgov.active_election(s, team_id) is not None:
        raise HTTPException(409, "Det paagaar allerede en avstemning.")

    valg = TeamElection(team_id=team_id, closes_at=utcnow() + teamgov.FRIST)
    s.add(valg)
    team = s.get(Team, team_id)
    teamgov.varsle(s, [m.user_id for m in teamgov.medlemmer(s, team_id)],
                   "election_started", "Velg lagleder",
                   f"{team.name} skal velge lagleder. Avstemningen er aapen i "
                   "7 dager.", team_id)
    s.commit()
    return {"election_id": valg.id, "closes_at": valg.closes_at}


class VoteIn(BaseModel):
    candidate_id: str = Field(max_length=36)


@router.post("/{team_id}/vote")
def cast_vote(team_id: str, body: VoteIn, user: User = Depends(current_user),
              s: OrmSession = Depends(db)) -> dict:
    _medlemskap(s, team_id, user.id)
    valg = teamgov.active_election(s, team_id)
    if valg is None:
        raise HTTPException(404, "Ingen paagaaende avstemning.")
    if s.scalar(select(TeamMember).where(
            TeamMember.team_id == team_id,
            TeamMember.user_id == body.candidate_id)) is None:
        raise HTTPException(422, "Kandidaten er ikke medlem av laget.")

    stemme = s.scalar(select(TeamVote).where(
        TeamVote.election_id == valg.id, TeamVote.voter_id == user.id))
    if stemme is None:
        s.add(TeamVote(election_id=valg.id, voter_id=user.id,
                       candidate_id=body.candidate_id))
    else:
        # §11: stemmer kan endres til fristen.
        stemme.candidate_id = body.candidate_id
    s.commit()

    teamgov.resolve_election(s, valg)
    return vote_status(team_id, user, s)


@router.get("/{team_id}/vote-status")
def vote_status(team_id: str, user: User = Depends(current_user),
                s: OrmSession = Depends(db)) -> dict:
    _medlemskap(s, team_id, user.id)
    valg = s.scalar(select(TeamElection).where(TeamElection.team_id == team_id)
                    .order_by(TeamElection.opened_at.desc()))
    if valg is None:
        raise HTTPException(404, "Ingen avstemning.")
    valg = teamgov.resolve_election(s, valg)

    stemmer = list(s.scalars(select(TeamVote).where(TeamVote.election_id == valg.id)))
    telling: dict[str, int] = {}
    for v in stemmer:
        telling[v.candidate_id] = telling.get(v.candidate_id, 0) + 1
    min_stemme = next((v.candidate_id for v in stemmer if v.voter_id == user.id), None)
    return {
        "election_id": valg.id,
        "outcome": valg.outcome.value,
        "closes_at": valg.closes_at,
        "elected_user_id": valg.elected_user_id,
        # Klienten viser nedtellingen som dager -> timer < 24 -> minutter < 120.
        "seconds_left": max(0, int((valg.closes_at - utcnow()).total_seconds()))
                        if valg.outcome == ElectionOutcome.pending else 0,
        "member_count": len(teamgov.medlemmer(s, team_id)),
        "votes": telling,
        "my_vote": min_stemme,
    }


# --------------------------------------------------------------------
# Fjern inaktiv lagleder (§11)
# --------------------------------------------------------------------

@router.post("/{team_id}/leader-challenge", status_code=201)
def challenge_leader(team_id: str, user: User = Depends(current_user),
                     s: OrmSession = Depends(db)) -> dict:
    _medlemskap(s, team_id, user.id)
    lagledere = teamgov.ledere(s, team_id)
    if not lagledere:
        raise HTTPException(409, "Laget har ingen lagleder.")
    if teamgov.active_challenge(s, team_id) is not None:
        raise HTTPException(409, "Det paagaar allerede en slik prosess.")

    leder = s.get(User, lagledere[0].user_id)
    grense = utcnow() - teamgov.INAKTIV_GRENSE
    if leder.last_seen_at is not None and leder.last_seen_at > grense:
        # §11, ordrett: dette er ikke en feil, men et svar.
        raise HTTPException(409, "Lagleder er ikke inaktiv. Ta kontakt.")

    ch = LeaderChallenge(team_id=team_id, leader_id=leder.id,
                         initiated_by=user.id,
                         deadline_at=utcnow() + teamgov.FRIST)
    s.add(ch)
    team = s.get(Team, team_id)
    teamgov.varsle(s, [leder.id], "leader_challenged", "Er du fortsatt aktiv?",
                   f"Et medlem har meldt at {team.name} staar uten aktiv "
                   "lagleder. Logger du paa innen 7 dager, skjer ingenting.",
                   team_id)
    s.commit()
    return {"challenge_id": ch.id, "deadline_at": ch.deadline_at}


@router.get("/{team_id}/leader-challenge")
def challenge_status(team_id: str, user: User = Depends(current_user),
                     s: OrmSession = Depends(db)) -> dict:
    _medlemskap(s, team_id, user.id)
    ch = s.scalar(select(LeaderChallenge).where(LeaderChallenge.team_id == team_id)
                  .order_by(LeaderChallenge.opened_at.desc()))
    if ch is None:
        raise HTTPException(404, "Ingen paagaaende prosess.")
    ch = teamgov.resolve_challenge(s, ch)
    return {"challenge_id": ch.id, "outcome": ch.outcome.value,
            "leader_id": ch.leader_id, "deadline_at": ch.deadline_at,
            "seconds_left": max(0, int((ch.deadline_at - utcnow()).total_seconds()))}


# --------------------------------------------------------------------
# Redirect-URL for invitasjon og QR (§4)
# --------------------------------------------------------------------

invite_router = APIRouter(tags=["lag"])


@invite_router.get("/i/{token}", include_in_schema=False)
def invite_redirect(token: str, request: Request) -> RedirectResponse:
    """
    §4: samme URL i QR-koden og i e-post/SMS-invitasjonen. Serveren leser
    User-Agent og sender videre til riktig butikk.

    Ingen autentisering, og vi sier ikke om tokenet finnes: lenken deles i
    aapne kanaler, og et svar som skiller gyldig fra ugyldig ville gjort den
    til et oppslagsverk over hvilke lag som eksisterer.
    """
    cfg = settings()
    ua = request.headers.get("user-agent", "").lower()
    ios = any(k in ua for k in ("iphone", "ipad", "ipod", "mac os"))
    maal = cfg.app_store_url if ios else cfg.play_store_url
    skille = "&" if "?" in maal else "?"
    return RedirectResponse(f"{maal}{skille}invite={token}", status_code=302)
