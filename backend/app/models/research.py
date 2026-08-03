"""
Forskningsdata (backend_spec §7) - STRUKTURELT ADSKILT.

Alle tabellene her ligger i skjemaet `research` og har INGEN fremmednoekler til
brukertabellene. Koblingen gaar kun én vei, gjennom en pseudonym-ID som
avledes med en server-hemmelighet (services/pseudonym.py): med tilgang KUN til
forskningsskjemaet er det ikke mulig aa finne tilbake til kontoen.

IKKE AKTIVER I PRODUKSJON foer personvernerklaering foreligger og behovet for
DPIA er avklart (§7/§9). Klienten har samme sperre (Dialogs.RESEARCH_ENABLED).
"""
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import RESEARCH_SCHEMA, Base, ResultType, utcnow


class ResearchConsent(Base):
    """Samtykketabell (§7): { pseudonymId, type, granted_at, revoked_at? }."""
    __tablename__ = "consents"
    __table_args__ = (
        UniqueConstraint("pseudonym_id", "consent_type", name="uq_consent_subject_type"),
        {"schema": RESEARCH_SCHEMA},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    pseudonym_id: Mapped[str] = mapped_column(String(64), index=True)
    consent_type: Mapped[ResultType] = mapped_column(
        Enum(ResultType, native_enum=False, length=16))
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ResearchRecord(Base):
    """
    En forskningsobservasjon. To resultattyper (trening og jakt) med ulik
    personvernprofil; jakt-deling styres av brukerens valg
    (ResearchSharingPreference) og filtreres FOER innsending.

    TODO(eier): konkret feltinnhold er ikke avklart (kjerne-spec §6).
    `payload_json` er en midlertidig beholder - erstatt med eksplisitte
    kolonner naar feltene defineres.
    """
    __tablename__ = "records"
    __table_args__ = ({"schema": RESEARCH_SCHEMA},)

    id: Mapped[int] = mapped_column(primary_key=True)
    pseudonym_id: Mapped[str] = mapped_column(String(64), index=True)
    session_ref: Mapped[str] = mapped_column(String(64), index=True)  # klient-generert
    captured_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    result_type: Mapped[ResultType] = mapped_column(
        Enum(ResultType, native_enum=False, length=16))
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)
    received_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ResearchDeletionRequest(Base):
    """
    §9: DELETE /v1/account sletter kontoen og legger inn en sletteanmodning
    her. Anmodningen bruker pseudonym-ID, saa forskningslageret kan etterkomme
    den uten aa faa vite hvem kontoen tilhoerte.
    """
    __tablename__ = "deletion_requests"
    __table_args__ = ({"schema": RESEARCH_SCHEMA},)

    id: Mapped[int] = mapped_column(primary_key=True)
    pseudonym_id: Mapped[str] = mapped_column(String(64), index=True)
    requested_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
