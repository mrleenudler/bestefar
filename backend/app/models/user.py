"""
Konto, identitet, samtykke og backup (backend_spec §1, §2, §3, §3.1, §7).
"""
from datetime import datetime

from sqlalchemy import (Boolean, Enum, ForeignKey, Integer,
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
    display_name_reviewed_at: Mapped[datetime | None] = mapped_column(nullable=True)

    birth_year: Mapped[int | None] = mapped_column(Integer, nullable=True)  # egenrapportert
    home_kommune: Mapped[str | None] = mapped_column(String(64), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)  # E.164
    findable: Mapped[bool] = mapped_column(Boolean, default=True)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    last_seen_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # §3: naar brukeren sist kunngjorde et felt dyr til vennene sine. KUN et
    # tidsstempel - art og sted lagres ALDRI. Jaktloggen ligger med vilje i den
    # klient-krypterte bloben (§2), og en serverkolonne med «hva ble felt hvor»
    # ville gjenskapt nettopp den loggen i klartekst. Kolonnen finnes bare for
    # aa kunne bremse gjentatte kunngjoeringer.
    hunt_announced_at: Mapped[datetime | None] = mapped_column(nullable=True)
    # §9: sletting. Raden beholdes kort for aa hindre gjenbruk av public_id.
    deleted_at: Mapped[datetime | None] = mapped_column(nullable=True, index=True)

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
    created_at: Mapped[datetime] = mapped_column(default=utcnow)

    user: Mapped[User] = relationship(back_populates="identities")


class AuthSession(Base):
    """
    §1: en innlogget enhet. Holder refresh-tokenet, ikke access-tokenet -
    det siste er kortlivet og verifiseres med signatur alene.

    Vi lagrer bare SHA-256 av refresh-tokenet. En lekket databasedump gir da
    ingen brukbare tokens, og siden tokenet er 32 tilfeldige byte trengs ingen
    langsom hash (det er ingen gjettbar passordentropi aa beskytte).

    Rotasjon: hver bruk av refresh-tokenet gir et nytt og ugyldiggjoer det
    gamle. Dukker et allerede brukt token opp igjen, er det enten en kopi paa
    avveie eller et nettverksforsoek som ble kjoert to ganger - vi kan ikke
    skille, saa hele oekten tilbakekalles. Det er det trygge valget.
    """
    __tablename__ = "auth_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    refresh_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    user_agent: Mapped[str] = mapped_column(String(200), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_used_at: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime] = mapped_column()
    revoked_at: Mapped[datetime | None] = mapped_column(nullable=True)


class EmailLoginCode(Base):
    """
    §1: engangskode for e-postinnlogging.

    Koden lagres hashet av samme grunn som refresh-tokenet. Raden beholdes
    etter bruk (`consumed_at`) saa den samme koden ikke kan spilles av paa
    nytt, og `attempts` gjoer at koden brenner opp etter faa gjett - seks
    siffer taaler ikke ubegrenset proeving.
    """
    __tablename__ = "email_login_codes"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    code_hash: Mapped[str] = mapped_column(String(64))
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime] = mapped_column()
    consumed_at: Mapped[datetime | None] = mapped_column(nullable=True)


class Device(Base):
    """Push-registrering per enhet (§11). FCM/APNs-token."""
    __tablename__ = "devices"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    platform: Mapped[Platform] = mapped_column(Enum(Platform, native_enum=False, length=16))
    push_token: Mapped[str] = mapped_column(String(255), unique=True)
    app_version: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(default=utcnow)

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
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


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
    # Skadedata (ettersoek, treffpunkt, om dyret ble funnet). §7 sier «private
    # som standard» - standardverdien er derfor False, men de har naa en bryter.
    # Uten den var «som standard» i praksis «aldri», og det er nettopp disse
    # radene som gjoer materialet verdt aa samle inn.
    share_injury_data: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


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
    client_ts: Mapped[datetime | None] = mapped_column(nullable=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class BackupKeyEscrow(Base):
    """
    §2/§13: FRIVILLIG deponering av noekkelen til backup-bloben.

    Standardveien er fortsatt Block Store paa telefonen pluss den 20-tegns
    gjenopprettingskoden, og serveren kan da ikke lese kopien. Slaar brukeren
    paa «gjenopprett uten kode», legges noekkelmaterialet her - og da KAN
    serveren lese bloben. Det er hele avveiningen, og den skal staa i klartekst
    i UI-teksten, ikke gjemmes bort.

    `material` er kryptert i ro (services/escrow.py) med en server-hemmelighet
    som ikke ligger i basen. En databasedump alene er derfor ikke nok - det
    kreves ogsaa tilgang til Fly-hemmeligheten. Det gjoer ikke deponeringen
    like sterk som ingen deponering, men det er forskjellen paa «lekket base»
    og «lekket base + lekket driftsmiljoe».

    Ingen andre endepunkter leser denne tabellen.
    """
    __tablename__ = "backup_key_escrow"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"),
                                         primary_key=True)
    material: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


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
    window_started_at: Mapped[datetime] = mapped_column(default=utcnow)
    quarantined_until: Mapped[datetime | None] = mapped_column(nullable=True)
    offense_count: Mapped[int] = mapped_column(Integer, default=0)   # styrer eskaleringen
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
