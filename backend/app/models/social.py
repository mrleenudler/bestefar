"""
Venner og lag (backend_spec §3, §4, §11).
"""
from datetime import datetime

from sqlalchemy import (CheckConstraint, Enum, Float, ForeignKey,
                        String, Text, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import (Base, ChallengeOutcome, DeliveryStatus, ElectionOutcome,
                   FriendshipStatus, InviteTarget, TeamKind, TeamRole,
                   new_uuid, utcnow)


class Friendship(Base):
    """
    §3: vennskap krever AKSEPT hos mottaker; ingen data deles foer det.
    Én rad per par - retningen (hvem som spurte) ligger i requester/addressee.
    """
    __tablename__ = "friendships"
    __table_args__ = (
        UniqueConstraint("requester_id", "addressee_id", name="uq_friendship_pair"),
        CheckConstraint("requester_id <> addressee_id", name="ck_friendship_not_self"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    requester_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                              index=True)
    addressee_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                              index=True)
    status: Mapped[FriendshipStatus] = mapped_column(
        Enum(FriendshipStatus, native_enum=False, length=16),
        default=FriendshipStatus.pending, index=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    responded_at: Mapped[datetime | None] = mapped_column(nullable=True)


class Team(Base):
    """§4: jaktlag eller skytterlag. `lat`/`lon` driver /v1/teams/near."""
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(64))
    kind: Mapped[TeamKind] = mapped_column(Enum(TeamKind, native_enum=False, length=16))
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lon: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    members: Mapped[list["TeamMember"]] = relationship(back_populates="team",
                                                       cascade="all, delete-orphan")


class TeamMember(Base):
    """§11: flere ledere er mulig; rollen ligger paa medlemskapet."""
    __tablename__ = "team_members"
    __table_args__ = (UniqueConstraint("team_id", "user_id", name="uq_team_member"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[TeamRole] = mapped_column(Enum(TeamRole, native_enum=False, length=16),
                                           default=TeamRole.member)
    joined_at: Mapped[datetime] = mapped_column(default=utcnow)

    team: Mapped[Team] = relationship(back_populates="members")


class TeamInvite(Base):
    """
    §4: invitasjon via redirect-URL (samme URL som QR). Serveren leser
    User-Agent og sender videre til riktig butikk. `token` er den delen av
    URL-en som identifiserer invitasjonen.
    """
    __tablename__ = "team_invites"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), index=True)
    inviter_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    target: Mapped[str] = mapped_column(String(255))        # e-post eller telefonnummer
    target_kind: Mapped[InviteTarget] = mapped_column(
        Enum(InviteTarget, native_enum=False, length=16))
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    delivery_status: Mapped[DeliveryStatus] = mapped_column(
        Enum(DeliveryStatus, native_enum=False, length=16), default=DeliveryStatus.pending)
    delivery_error: Mapped[str | None] = mapped_column(String(300), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    sent_at: Mapped[datetime | None] = mapped_column(nullable=True)
    accepted_at: Mapped[datetime | None] = mapped_column(nullable=True)
    accepted_by: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class TeamElection(Base):
    """
    §11 «Velg leder»: avstemning med 7-dagers nedtelling. Stemmer kan endres
    til fristen; enstemmighet avslutter tidlig.
    """
    __tablename__ = "team_elections"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), index=True)
    opened_at: Mapped[datetime] = mapped_column(default=utcnow)
    closes_at: Mapped[datetime] = mapped_column(index=True)
    # Medlemstallet DA AVSTEMNINGEN STARTET, laast fordi kvorumet regnes av det.
    # Maalte vi ved avgjoerelse, kunne terskelen senkes ved aa fjerne medlemmer
    # mens avstemningen paagaar. 0 paa rader fra foer kolonnen fantes; de
    # behandles som «uten kvorumskrav», se services/teamgov.kvorum().
    member_count_at_open: Mapped[int] = mapped_column(default=0)
    resolved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    outcome: Mapped[ElectionOutcome] = mapped_column(
        Enum(ElectionOutcome, native_enum=False, length=24), default=ElectionOutcome.pending)
    elected_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True)


class TeamVote(Base):
    """Én stemme per velger per avstemning; kan endres til fristen (§11)."""
    __tablename__ = "team_votes"
    __table_args__ = (UniqueConstraint("election_id", "voter_id", name="uq_vote_election_voter"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    election_id: Mapped[int] = mapped_column(ForeignKey("team_elections.id", ondelete="CASCADE"),
                                             index=True)
    voter_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    candidate_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    cast_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class LeaderChallenge(Base):
    """
    §11 «Fjern inaktiv lagleder». Dette er IKKE en avstemning: lederen faar
    push og 7 dager paa seg. Logger lederen paa, avbrytes prosessen; ellers
    mister lederen lederstatus (men forblir medlem).
    """
    __tablename__ = "leader_challenges"

    id: Mapped[int] = mapped_column(primary_key=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id", ondelete="CASCADE"), index=True)
    leader_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    initiated_by: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    opened_at: Mapped[datetime] = mapped_column(default=utcnow)
    deadline_at: Mapped[datetime] = mapped_column(index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(nullable=True)
    outcome: Mapped[ChallengeOutcome] = mapped_column(
        Enum(ChallengeOutcome, native_enum=False, length=32), default=ChallengeOutcome.pending)


class PendingMessage(Base):
    """
    §11: «pending messages»-koe. Klienten henter den ved oppstart og viser
    meldingene som foerste skjerm (navneendring, fjernet fra lag, ...).
    Push er varselkanalen; denne koen er garantien for at meldingen naar fram
    ogsaa naar push feiler.
    """
    __tablename__ = "pending_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32))       # team_renamed | removed_from_team | ...
    title: Mapped[str] = mapped_column(String(120))
    body: Mapped[str] = mapped_column(Text)
    team_id: Mapped[str | None] = mapped_column(
        ForeignKey("teams.id", ondelete="CASCADE"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    delivered_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # Satt naar meldingen er OVERKJOERT av et senere utfall - f.eks. en
    # «avstemningen er aapen i 7 dager» etter at avstemningen er avgjort.
    # Klienten henter koen ved appstart, saa en slik melding kan bli hentet ni
    # dager for sent og vist rett over resultatet. Overkjoerte meldinger
    # leveres ikke; se backend/KONTRAKT.md §4.1.
    superseded_at: Mapped[datetime | None] = mapped_column(nullable=True)
