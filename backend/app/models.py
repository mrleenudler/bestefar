"""
Datamodell. Kravspec §5-§6:
  - Brukerens statistikkdata og forskningsdata er STRUKTURELT adskilt:
    forskningstabellene har pseudonym subject_id og INGEN fremmednoekler til
    brukertabellene. Adskillelsen er dyr aa ettermontere - derfor fra dag en.
  - To resultattyper (trening/jakt) med ulik personvernprofil.
  - Tidsserie: alt er tidsstemplet og koblet til oekt/serie for aggregering
    per skytter over tid.
"""
import enum
from datetime import datetime, timezone

from sqlalchemy import (JSON, Boolean, DateTime, Enum, Float, ForeignKey,
                        Integer, LargeBinary, String)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class ResultType(enum.Enum):
    training = "training"
    hunt = "hunt"          # mer sensitivt enn treningsdata (kravspec §6)


# ====================================================================
# 1) BRUKERENS EGNE DATA (statistikk-sync, /v1/stats)
# ====================================================================

class Shooter(Base):
    __tablename__ = "shooters"
    id: Mapped[int] = mapped_column(primary_key=True)
    device_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    store_images: Mapped[bool] = mapped_column(Boolean, default=False)  # brukerstyrt opsjon

    sessions: Mapped[list["Session"]] = relationship(back_populates="shooter")


class Session(Base):
    """En oekt (paa banen eller paa jakt). Tidsserie-anker (kravspec §6)."""
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    shooter_id: Mapped[int] = mapped_column(ForeignKey("shooters.id"), index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    result_type: Mapped[ResultType] = mapped_column(Enum(ResultType),
                                                    default=ResultType.training)

    shooter: Mapped[Shooter] = relationship(back_populates="sessions")
    series: Mapped[list["Series"]] = relationship(back_populates="session")


class Series(Base):
    """En serie (ett analysert skjermbilde = en serie)."""
    __tablename__ = "series"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), index=True)
    captured_at: Mapped[datetime] = mapped_column(DateTime)   # CV-kontraktens tidsstempel
    sum_decimal: Mapped[float] = mapped_column(Float)
    sum_integer: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float)

    session: Mapped[Session] = relationship(back_populates="series")
    shots: Mapped[list["Shot"]] = relationship(back_populates="series")


class Shot(Base):
    """Ett treff (relativ polar, kravspec §3-output)."""
    __tablename__ = "shots"
    id: Mapped[int] = mapped_column(primary_key=True)
    series_id: Mapped[int] = mapped_column(ForeignKey("series.id"), index=True)
    r_rel: Mapped[float] = mapped_column(Float)     # ringavstander fra senter
    theta: Mapped[float] = mapped_column(Float)     # radianer
    decimal: Mapped[float] = mapped_column(Float)
    integer: Mapped[int] = mapped_column(Integer)

    series: Mapped[Series] = relationship(back_populates="shots")


# ====================================================================
# 2) FEILEDE ANALYSER (/v1/failed-analyses, opt-in)
# ====================================================================

class FailedAnalysis(Base):
    __tablename__ = "failed_analyses"
    id: Mapped[int] = mapped_column(primary_key=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    status_code: Mapped[int] = mapped_column(Integer)          # BF_REJECTED_* / lav konfidens
    confidence: Mapped[float] = mapped_column(Float)
    core_version: Mapped[str] = mapped_column(String(32))
    image: Mapped[bytes] = mapped_column(LargeBinary)          # kun ved eksplisitt opt-in
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


# ====================================================================
# 3) FORSKNINGSDATA (/v1/research) - STRUKTURELT ADSKILT
#    Pseudonym subject_id; INGEN FK til shooters/sessions.
# ====================================================================

class ResearchConsent(Base):
    """Eksplisitt samtykke, adskilt fra ordinaer app-bruk (kravspec §6)."""
    __tablename__ = "research_consents"
    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(64), index=True)  # pseudonym
    consent_type: Mapped[ResultType] = mapped_column(Enum(ResultType))
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    withdrawn_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ResearchRecord(Base):
    """
    En forskningsobservasjon: tidsstemplet, koblet til oekt/serie via
    pseudonyme klient-genererte IDer (aggregering per skytter over tid, §6).

    TODO(eier): Konkret feltinnhold i forskningsdatasettet er IKKE avklart
    (kravspec §6 'AApent punkt'). payload_json er en midlertidig beholder;
    erstatt med eksplisitte kolonner naar feltene defineres.
    """
    __tablename__ = "research_records"
    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[str] = mapped_column(String(64), index=True)   # pseudonym
    session_ref: Mapped[str] = mapped_column(String(64), index=True)  # klient-generert
    captured_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    result_type: Mapped[ResultType] = mapped_column(Enum(ResultType))
    payload_json: Mapped[dict] = mapped_column(JSON, default=dict)    # TODO(eier): felter
