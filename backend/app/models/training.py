"""
Treningsresultater (backend_spec §5). Speiler `SeriesRecord`/`Shot` i
android/app/src/main/java/no/bestefar/app/Model.kt, slik at synk blir en
direkte felt-for-felt-mapping.

MERK: `weapon_id` er en ren streng uten fremmednoekkel. Vaapen og optikk ligger
i den klient-krypterte backup-bloben (§2) - serveren skal ikke kjenne dem, men
maa kunne gruppere serier per vaapen for venners delte statistikk (§3).
"""
from datetime import datetime

from sqlalchemy import (Boolean, Enum, Float, ForeignKey, Integer,
                        String, UniqueConstraint)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, PosModifier, Position, utcnow


class Series(Base):
    """
    En serie = ett analysert skjermbilde. Primaernoekkelen er klientens egen
    UUID, saa opplasting av en koet serie er idempotent (§5: klienten koer
    usendte med uploaded=false og kan sende samme serie flere ganger).
    """
    __tablename__ = "series"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)   # klient-UUID
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    ts: Mapped[datetime] = mapped_column(index=True)      # klientens tidsstempel
    weapon_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    ammo_name: Mapped[str] = mapped_column(String(64), default="")
    distance_m: Mapped[int] = mapped_column(Integer)
    position: Mapped[Position] = mapped_column(Enum(Position, native_enum=False, length=16))
    modifier: Mapped[PosModifier] = mapped_column(
        Enum(PosModifier, native_enum=False, length=16), default=PosModifier.UTEN)
    corrected: Mapped[bool] = mapped_column(Boolean, default=False)
    season_key: Mapped[str] = mapped_column(String(16), default="", index=True)

    # Denormalisert for statistikk-spoerringer (snitt/utvikling per venn, §3).
    sum_decimal: Mapped[float] = mapped_column(Float, default=0.0)
    sum_integer: Mapped[int] = mapped_column(Integer, default=0)
    shot_count: Mapped[int] = mapped_column(Integer, default=0)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)

    shots: Mapped[list["Shot"]] = relationship(back_populates="series",
                                               cascade="all, delete-orphan",
                                               order_by="Shot.idx")


class Shot(Base):
    """Ett treff (relativ polar, kravspec §3-output)."""
    __tablename__ = "shots"
    __table_args__ = (UniqueConstraint("series_id", "idx", name="uq_shot_series_idx"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    series_id: Mapped[str] = mapped_column(ForeignKey("series.id", ondelete="CASCADE"),
                                           index=True)
    idx: Mapped[int] = mapped_column(Integer)       # rekkefoelge slik klienten viser dem
    r_rel: Mapped[float] = mapped_column(Float)     # ringsteg fra senter (1 ringsteg = 1 poeng)
    theta: Mapped[float] = mapped_column(Float)     # radianer
    decimal: Mapped[float] = mapped_column(Float)
    integer: Mapped[int] = mapped_column(Integer)

    series: Mapped[Series] = relationship(back_populates="shots")
