"""
Konto, identitet, samtykke og backup (backend_spec §1, §2, §3, §3.1, §7).
"""
from datetime import datetime

from sqlalchemy import (Boolean, DateTime, Enum, ForeignKey, Integer,
                        LargeBinary, String, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import (Base, NameStatus, Platform, PositionGranularity, Provider,
                   QuarantineScope, new_uuid, utcnow)


class User(Base):
    """
    §1: intern UUID som primaernoekkel. `public_id` er den korte, haandskrivbare
    ID-en fra §3.1 (Crockford base32 med sjekksiffer, f.eks. BF-7Q4K-9F2M) som
    brukes til venneforespoersler og QR.

    Forsknings-ID lagres IKKE her: den avledes deterministisk fra `id` med en
    server-hemmelighet (services/pseudonym.py), slik at forskningslageret ikke
    inneholder noen kobling tilbake til kontoen (§1, §7).
    """
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_uuid)
    public_id: Mapped[str] = mapped_column(String(16), unique=True, index=True)

    display_name: Mapped[str] = mapped_column(String(24))
    # Navnet eksponeres for andre foerst naar moderasjonen har godkjent det (§3).
    display_name_status: Mapped[NameStatus] = mapped_column(
        Enum(NameStatus, native_enum=False, length=16), default=NameStatus.pending)
    display_name_reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)  # egenrapportert
    home_kommune: Mapped[str | None] = mapped_column(String(64), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)  # E.164
    findable: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # §9: sletting. Raden beholdes kort for aa hindre gjenbruk av public_id.
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    identities: Mapped[list["AuthIdentity"]] = relationship(back_populates="user",
                                                           cascade="all, delete-orphan")
    devices: Mapped[list["Device"]] = relationship(back_populates="user",
                                                   cascade="all, delete-orphan")


class AuthIdentity(Base):
    """§1: Google / Apple / e-post ved lansering. Telefon-OTP er utsatt til v2."""
    __tablename__ = "auth_identities"
    __table_args__ = (UniqueConstraint("provider", "subject", name="uq_identity_provider_subject"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    provider: Mapped[Provider] = mapped_column(Enum(Provider, native_enum=False, length=16))
    subject: Mapped[str] = mapped_column(String(255))       # `sub` fra leverandoeren
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="identities")


class Device(Base):
    """Push-registrering per enhet (§11). FCM/APNs-token."""
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    platform: Mapped[Platform] = mapped_column(Enum(Platform, native_enum=False, length=16))
    push_token: Mapped[str] = mapped_column(String(255), unique=True)
    app_version: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    user: Mapped[User] = relationship(back_populates="devices")


class SharingPreference(Base):
    """
    §3: hver bruker velger hvilke felt som deles med venner. Visningsnavn deles
    alltid og har derfor ingen bryter. Serveren filtrerer UTGAAENDE data etter
    delerens valg - klienten skal aldri faa felt den ikke har lov til aa se.
    """
    __tablename__ = "sharing_preferences"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                         primary_key=True)
    share_phone: Mapped[bool] = mapped_column(Boolean, default=False)
    share_home_kommune: Mapped[bool] = mapped_column(Boolean, default=False)
    share_shots_total: Mapped[bool] = mapped_column(Boolean, default=False)
    share_shots_season: Mapped[bool] = mapped_column(Boolean, default=False)
    share_avg_score: Mapped[bool] = mapped_column(Boolean, default=False)
    share_trend: Mapped[bool] = mapped_column(Boolean, default=False)
    share_kills: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class ResearchSharingPreference(Base):
    """
    §7: brukerens valg for hva som kan deles til forskning. Ligger BEVISST i
    brukerskjemaet, ikke i forskningsskjemaet - selve valget er persondata,
    mens forskningslageret kun skal inneholde pseudonymiserte observasjoner.
    """
    __tablename__ = "research_sharing_preferences"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                         primary_key=True)
    share_species: Mapped[bool] = mapped_column(Boolean, default=False)
    share_date: Mapped[bool] = mapped_column(Boolean, default=False)
    share_shot_situation: Mapped[bool] = mapped_column(Boolean, default=False)
    position_granularity: Mapped[PositionGranularity] = mapped_column(
        Enum(PositionGranularity, native_enum=False, length=16),
        default=PositionGranularity.none)
    # Skadedata er private som standard (§7) og har ingen bryter her.
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class Backup(Base):
    """
    §2: klient-kryptert blob (serier + jaktlogg + innstillinger). Serveren kan
    ikke lese innholdet og gjoer ingen konfliktloesning inne i bloben - den
    skjer klient-side, last-write-wins per post-ID (postene har UUID + ts).
    """
    __tablename__ = "backups"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                         primary_key=True)
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    payload_bytes: Mapped[int] = mapped_column(Integer, default=0)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    device_id: Mapped[str] = mapped_column(String(64), default="")
    client_ts: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class SearchQuarantine(Base):
    """
    §3.1 misbruksvern: teller mislykkede oppslag per konto og per IP.
    5 mislykkede telefonsoek paa en dag -> karantene 1 doegn, eskalerende til
    7 doegn ved gjentakelse.

    Ligger i databasen (ikke i minnet som ratelimit.py) fordi karantenen skal
    overleve omstart og gjelde paa tvers av Fly-maskiner.
    """
    __tablename__ = "search_quarantines"
    __table_args__ = (UniqueConstraint("scope", "subject", name="uq_quarantine_scope_subject"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    scope: Mapped[QuarantineScope] = mapped_column(
        Enum(QuarantineScope, native_enum=False, length=16))
    subject: Mapped[str] = mapped_column(String(64))        # bruker-ID eller IP/subnett
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    window_started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    quarantined_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    offense_count: Mapped[int] = mapped_column(Integer, default=0)   # styrer eskaleringen
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
