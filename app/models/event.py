"""
SQLAlchemy model for Canonical Sentinel Events.
Stores normalized events adhering to the C4 specification and cross-module contract with Module 4.
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CanonicalEventRecord(Base):
    """
    Persisted canonical Sentinel event.
    Provides complete traceability:
    Physical Camera -> Processing Node -> AI Model -> Canonical Event Envelope -> Module 4
    """
    __tablename__ = "canonical_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), unique=True, index=True, nullable=False)  # Canonical event UUID
    source_event_id = Column(String(128), index=True, nullable=False)  # Provider event ID
    provider = Column(String(64), index=True, nullable=False)  # e.g., "KAAVAL_AI", "HIKVISION_NVR"
    adapter = Column(String(64), nullable=False)  # e.g., "KAAVAL_EDGE", "KAAVAL_SERVER"
    camera_id = Column(String(64), index=True, nullable=False)  # e.g., "CAM-001"
    processing_node_id = Column(String(64), index=True, nullable=True)  # e.g., "RDK-X5-001"
    event_type = Column(String(64), index=True, nullable=False)  # e.g., "HELMET_VIOLATION", "PLATE_READ"
    timestamp = Column(DateTime(timezone=True), index=True, nullable=False)
    confidence = Column(Float, nullable=False, default=1.0)

    # Object & violation attributes
    object_type = Column(String(64), nullable=True)  # "VEHICLE", "PERSON"
    plate_number = Column(String(32), index=True, nullable=True)
    bounding_box = Column(JSON, default=list)  # [x, y, w, h]
    attributes = Column(JSON, default=dict)  # {"riders": 3, "vehicle_class": "TWO_WHEELER", ...}

    # AI Provenance metadata
    ai_provenance = Column(JSON, default=dict)  # {"model_name": ..., "model_version": ..., "latency_ms": ...}

    # Evidence references (Module 5 / Mock URLs)
    evidence_ref = Column(JSON, default=dict)  # {"image": "mock://...", "video": "mock://..."}

    # Complete raw vendor payload for audit & forensic traceability
    raw_payload = Column(JSON, default=dict)

    # Downstream delivery state
    delivery_status = Column(String(32), default="DELIVERED")  # "DELIVERED", "BUFFERED", "DROPPED_DUP"
    created_at = Column(DateTime(timezone=True), default=utcnow)
