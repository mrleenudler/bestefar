"""
Datamodell (backend_spec §1-§11).

  user.py      konto, identitet, delingsvalg, backup, misbruksvern
  training.py  serier og treff (§5)
  social.py    venner og lag (§3, §4, §11)
  ops.py       feilanalyse (§6) og feedback (§10)
  research.py  forskningsdata i EGET skjema (§7) - ingen FK til brukertabellene
"""
from .base import (RESEARCH_SCHEMA, Base, ChallengeOutcome, DeliveryStatus,
                   ElectionOutcome, FailedTag, FriendshipStatus, InviteTarget,
                   NameStatus, Platform, PosModifier, Position,
                   PositionGranularity, Provider, QuarantineScope, ResultType,
                   TeamKind, TeamRole, UtcDateTime, as_utc, new_uuid, utcnow)
from .ops import FailedAnalysis, Feedback
from .research import (ResearchConsent, ResearchDeletionRequest,
                       ResearchRecord)
from .social import (Friendship, LeaderChallenge, PendingMessage, Team,
                     TeamElection, TeamInvite, TeamMember, TeamVote)
from .training import Series, Shot
from .user import (AuthIdentity, AuthSession, Backup, BackupKeyEscrow, Device,
                   EmailLoginCode, ResearchSharingPreference, SearchQuarantine,
                   SharingPreference, User)

__all__ = [
    "RESEARCH_SCHEMA", "Base", "UtcDateTime", "as_utc", "new_uuid", "utcnow",
    # enums
    "ChallengeOutcome", "DeliveryStatus", "ElectionOutcome", "FailedTag",
    "FriendshipStatus", "InviteTarget", "NameStatus", "Platform",
    "PosModifier", "Position", "PositionGranularity", "Provider",
    "QuarantineScope", "ResultType", "TeamKind", "TeamRole",
    # tabeller
    "AuthIdentity", "AuthSession", "Backup", "BackupKeyEscrow", "Device",
    "EmailLoginCode",
    "FailedAnalysis", "Feedback",
    "Friendship", "LeaderChallenge", "PendingMessage",
    "ResearchConsent", "ResearchDeletionRequest", "ResearchRecord",
    "ResearchSharingPreference", "SearchQuarantine", "Series",
    "SharingPreference", "Shot", "Team", "TeamElection", "TeamInvite",
    "TeamMember", "TeamVote", "User",
]
