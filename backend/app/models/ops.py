"""
Drift: feilanalyse-innsending (§6) og melding til utvikler (§10).
"""
from datetime import datetime

from sqlalchemy import (JSON, Enum, Float, ForeignKey, Integer,
                        LargeBinary, String, Text)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, FailedTag, utcnow


class FailedAnalysis(Base):
    """
    §6: opt-in innsending av bilder der analysen feilet eller OCR var uenig.

    Bildet ligger i Cloudflare R2 og er bare REFERERT herfra (`object_key`);
    §6/§0.1 er tydelige paa at bilder aldri lagres i databasen.

    `image_legacy` brukes fortsatt naar R2 ikke er konfigurert (lokalt og i
    testene), og holder dessuten bildene som ble tatt imot foer opplastingen ble
    koblet inn 2026-08-15. Kolonnen kan ikke fjernes foer de radene er flyttet
    eller kastet - /health teller dem under «bilder».
    """
    __tablename__ = "failed_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    submitted_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    series_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)

    tag: Mapped[FailedTag] = mapped_column(Enum(FailedTag, native_enum=False, length=16),
                                           default=FailedTag.rejected, index=True)
    status_code: Mapped[int] = mapped_column(Integer)      # BF_REJECTED_* / lav konfidens
    confidence: Mapped[float] = mapped_column(Float)
    core_version: Mapped[str] = mapped_column(String(32))

    object_key: Mapped[str | None] = mapped_column(String(255), nullable=True)  # R2-noekkel
    image_legacy: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)

    detected_scores: Mapped[list] = mapped_column(JSON, default=list)
    ocr_scores: Mapped[list] = mapped_column(JSON, default=list)
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class Feedback(Base):
    """
    §10: melding fra bruker til utvikler. Lagres ALLTID; e-postvideresending er
    et sidespor som kan feile uten at meldingen gaar tapt.
    """
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    received_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)
    subject: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text)
    app_version: Mapped[str] = mapped_column(String(32), default="")
    device_model: Mapped[str] = mapped_column(String(64), default="")
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    forwarded_at: Mapped[datetime | None] = mapped_column(nullable=True)
    forward_error: Mapped[str | None] = mapped_column(String(300), nullable=True)
