"""
SQLAlchemy model for Component Health & Connectivity Monitoring (C6).
Separates health across 4 distinct dimensions:
1. Camera health (e.g. ONLINE, OFFLINE)
2. Processing Node health (e.g. ONLINE, OFFLINE)
3. AI Runtime health (e.g. RUNNING, CRASHED, RESTARTING)
4. Event Pipe health (e.g. UP, CUT)
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, JSON

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class HealthRecordModel(Base):
    """
    Stores independent health status records for system components.
    """
    __tablename__ = "health_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(String(64), index=True, nullable=False)
    entity_type = Column(String(32), index=True, nullable=False)  # "CAMERA", "PROCESSING_NODE", "AI_RUNTIME", "EVENT_PIPE"
    status = Column(String(32), nullable=False)  # "ONLINE", "RUNNING", "CRASHED", "UP", "CUT", etc.
    details = Column(JSON, default=dict)
    last_observed = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
