"""
SQLAlchemy models for the Sentinel Camera Grid (C2 inventory mapping + C3 stream sources).

camera_mappings : grid catalogue id (external_id) -> canonical M3 camera id, plus the probed
                  codec/resolution and the liveness bookkeeping used by grid sync.
stream_sources  : which protocols (RTSP / WHEP / HLS) each canonical camera can be pulled over,
                  their priority, availability and which one is currently active.

NOTE: this file was missing from the working copy because the root .gitignore contains a bare
`models/` pattern that also matches `app/models/`. It was reconstructed column-by-column from
its usages (app/api/grid.py, app/services/grid_sync.py, app/relay/manager.py, app/relay/__main__.py,
tests/test_grid_relay.py).
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, UniqueConstraint

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CameraMappingModel(Base):
    """
    One row per grid camera ever seen: maps the vendor id to the canonical camera id and
    records the last probed stream characteristics.
    Rows are never deleted; sync flips `status` between ACTIVE and INACTIVE.
    """
    __tablename__ = "camera_mappings"
    __table_args__ = (
        UniqueConstraint("system_id", "external_id", name="uq_camera_mappings_system_external"),
    )

    # Composite natural key: which system's catalogue entry this is (e.g. "sentinel-grid", "cam04")
    system_id = Column(String(64), primary_key=True)
    external_id = Column(String(128), primary_key=True)

    camera_id = Column(String(64), index=True, nullable=False)   # canonical M3 id, e.g. "GRID-CAM04"
    status = Column(String(32), default="ACTIVE")                # "ACTIVE" | "INACTIVE"

    # C2 sync bookkeeping
    missed_runs = Column(Integer, default=0)                     # consecutive syncs missing this camera
    last_seen_at = Column(DateTime(timezone=True), nullable=True)

    # Probed stream characteristics (authority for relay config)
    codec = Column(String(16), nullable=True)                    # "h264" | "h265" | "mjpeg"
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Float, nullable=True)

    # Connection test bookkeeping (C7)
    probed_at = Column(DateTime(timezone=True), nullable=True)
    probe_error = Column(String(200), nullable=True)

    def __repr__(self) -> str:
        return f"<CameraMapping {self.system_id}/{self.external_id} -> {self.camera_id} ({self.status})>"


class StreamSourceModel(Base):
    """
    One row per (canonical camera, protocol): can this camera be pulled over that protocol
    right now, with what priority, and which protocol is the active source.
    """
    __tablename__ = "stream_sources"
    __table_args__ = (
        UniqueConstraint("camera_id", "protocol", name="uq_stream_sources_camera_protocol"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), index=True, nullable=False)   # canonical M3 camera id
    protocol = Column(String(16), nullable=False)                # "RTSP" | "WHEP" | "HLS"
    priority = Column(Integer, default=4)                        # lower = tried first
    available = Column(Boolean, nullable=True)                   # last probe result (None = never tried)
    active = Column(Boolean, default=False)                      # currently flowing source
    detail = Column(String(200), nullable=True)                  # credential-free failure detail
    checked_at = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<StreamSource {self.camera_id}/{self.protocol} active={self.active}>"
