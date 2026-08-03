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

from sqlalchemy import DateTime, TypeDecorator
from sqlalchemy.orm import DeclarativeBase

RESEARCH_SCHEMA = "research"


class UtcDateTime(TypeDecorator):
    """
    Tidsstempel som ALLTID er tidssone-bevisst i Python og alltid UTC i basen.

    Uten dette blir det en stille felle: `utcnow()` gir en aware datetime, men
    en naiv `DateTime`-kolonne leverer en naive datetime tilbake. Alt som
    SAMMENLIGNER en ny verdi med en lagret - karantenevinduer, «er denne
    backupen eldre enn den lagrede?» - kaster da
    «can't subtract offset-naive and offset-aware datetimes», og bare paa de
    kodestiene som faktisk sammenligner. SQLite gjoer det verre ved aa levere
    naive verdier selv naar kolonnen er erklaert med timezone=True.

    Naive verdier inn (typisk ISO-tidsstempler fra klienten uten offset)
    tolkes som UTC.
    """
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class Base(DeclarativeBase):
    # Alle `Mapped[datetime]`-kolonner faar UtcDateTime uten aa maatte oppgi
    # typen eksplisitt.
    type_annotation_map = {datetime: UtcDateTime()}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """Normaliserer en innkommende datetime (f.eks. fra en query-parameter) til
    aware UTC, saa den kan sammenlignes med lagrede verdier."""
    return (value.replace(tzinfo=timezone.utc) if value.tzinfo is None
            else value.astimezone(timezone.utc))


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
