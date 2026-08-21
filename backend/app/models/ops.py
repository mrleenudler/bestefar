"""
Drift: feilanalyse-innsending (§6) og melding til utvikler (§10).
"""
from datetime import datetime

from sqlalchemy import JSON, Enum, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, FailedTag, utcnow


class FailedAnalysis(Base):
    """
    §6: opt-in innsending av bilder der analysen feilet eller OCR var uenig.

    Bildet ligger i Cloudflare R2 og er bare REFERERT herfra (`object_key`);
    §6/§0.1 er tydelige paa at bilder aldri lagres i databasen.

    Kolonnen `image_legacy` holdt bildene mens opplastingen ikke fantes. Den ble
    fjernet 2026-08-15 (migrasjon a3f7c1e59b24) etter at de fem radene som laa
    igjen var flyttet til R2. Er ikke R2 konfigurert, tar endepunktet ikke imot
    donasjoner i det hele tatt - det finnes ikke lenger et annet sted aa gjoere
    av dem.

    RADEN KAN IKKE KOBLES TIL EN PERSON, og det er en invariant (B-52). Begge
    veiene dit er fjernet i b8d24a0f5c17:

      `series_id` var den samme ID-en som serien lagres under i `/v1/stats`, og
      dermed den ene faktiske koblingen. Seks av ni rader hadde den da den ble
      fjernet 2026-08-21.

      `user_id` var en fremmednoekkel til `users` som ALDRI ble satt (null av
      ni). Den var ingen kobling, men en ferdig oppkoblet mulighet for aa lage
      én, paa en tabell hvis hele poeng er det motsatte. Skal donasjoner kunne
      knyttes til en konto igjen, krever det en ny migrasjon - og det er den
      terskelen som er poenget.
    """
    __tablename__ = "failed_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    submitted_at: Mapped[datetime] = mapped_column(default=utcnow, index=True)

    tag: Mapped[FailedTag] = mapped_column(Enum(FailedTag, native_enum=False, length=16),
                                           default=FailedTag.rejected, index=True)
    status_code: Mapped[int] = mapped_column(Integer)      # BF_REJECTED_* / lav konfidens
    confidence: Mapped[float] = mapped_column(Float)
    core_version: Mapped[str] = mapped_column(String(32))

    object_key: Mapped[str | None] = mapped_column(String(255), nullable=True)  # R2-noekkel

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
