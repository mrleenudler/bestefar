"""
Felles fundament for datamodellen.

STRUKTURELL ADSKILLELSE (backend_spec §0/§7): forskningstabellene ligger i et
EGET Postgres-skjema (`research`) og har ingen fremmednoekler til
brukertabellene. SQLite (tester/lokalt) har ikke skjemaer; der oversettes
`research` til None via `schema_translate_map` i db.py og migrations/env.py.
"""
import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import DeclarativeBase

RESEARCH_SCHEMA = "research"


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_uuid() -> str:
    return str(uuid.uuid4())


class StrEnum(str, enum.Enum):
    """Enum som lagres som tekst.

    Vi bruker gjennomgaaende `native_enum=False` paa kolonnene: Postgres-native
    ENUM-typer maa endres med egne ALTER TYPE-migrasjoner og finnes per skjema.
    VARCHAR + CHECK er billigere aa utvide, og verdien er lesbar i Supabases
    admin-UI.
    """


class Provider(StrEnum):
    google = "google"
    apple = "apple"
    email = "email"
    # phone (OTP) er utsatt til v2 - backend_spec §1.


class Platform(StrEnum):
    android = "android"
    ios = "ios"


class NameStatus(StrEnum):
    """Sensur av visningsnavn (§3): navnet eksponeres foerst etter godkjenning."""
    pending = "pending"
    approved = "approved"
    rejected = "rejected"


class ResultType(StrEnum):
    training = "training"
    hunt = "hunt"          # mer sensitivt enn treningsdata (kravspec §6)


class Position(StrEnum):
    """Speiler Position i android/.../Model.kt (BENK er fjernet)."""
    LIGGENDE = "LIGGENDE"
    SITTENDE = "SITTENDE"
    KNESTAENDE = "KNESTAENDE"
    STAAENDE = "STAAENDE"


class PosModifier(StrEnum):
    UTEN = "UTEN"
    ANLEGG = "ANLEGG"
    REIM = "REIM"


class FriendshipStatus(StrEnum):
    pending = "pending"
    accepted = "accepted"
    declined = "declined"
    blocked = "blocked"


class TeamKind(StrEnum):
    jakt = "jakt"
    skytter = "skytter"


class TeamRole(StrEnum):
    member = "member"
    leader = "leader"      # flere ledere er mulig (§4)


class InviteTarget(StrEnum):
    email = "email"
    phone = "phone"


class DeliveryStatus(StrEnum):
    """Leveringskvittering/-feil tilbake til klienten (§4)."""
    pending = "pending"
    sent = "sent"
    failed = "failed"
    accepted = "accepted"


class ElectionOutcome(StrEnum):
    pending = "pending"
    elected = "elected"
    cancelled = "cancelled"
    expired = "expired"


class ChallengeOutcome(StrEnum):
    pending = "pending"
    cancelled_leader_active = "cancelled_leader_active"
    leader_demoted = "leader_demoted"


class FailedTag(StrEnum):
    """`tag` i backend_spec §6."""
    ocr_match = "ocr_match"
    ocr_mismatch = "ocr_mismatch"
    rejected = "rejected"


class PositionGranularity(StrEnum):
    """Grovhet paa delt jaktposisjon (§7). Skadedata deles aldri som standard."""
    none = "none"
    kommune = "kommune"
    fylke = "fylke"
    exact = "exact"


class QuarantineScope(StrEnum):
    account = "account"
    ip = "ip"
