"""
SQLAlchemy models for Systems, Cameras, Processing Nodes, and Camera Assignments.
Implements the core Gujarat CCTV federation topology:
System -> Cameras
Camera -> Processing Node -> AI Provider / Model
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship

from app.db.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SystemModel(Base):
    """
    Upstream connected system (VMS, NVR, DVR, or Kaaval Edge / Server fleet).
    """
    __tablename__ = "systems"

    id = Column(String(64), primary_key=True, index=True)  # e.g., "VMS-HIK-001", "KAAVAL-EDGE-SYSTEM"
    name = Column(String(128), nullable=False)
    vendor = Column(String(64), nullable=False)  # "HIKVISION", "DAHUA", "KAAVAL", "GENERIC_ONVIF"
    protocol = Column(String(32), nullable=False)  # "ONVIF", "RTSP", "KAAVAL_EDGE", "KAAVAL_SERVER"
    connection_params = Column(JSON, default=dict)  # host, port, credential_ref, etc.
    status = Column(String(32), default="ACTIVE")  # "ACTIVE", "INACTIVE", "DEGRADED"
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    # Relationships
    cameras = relationship("CameraModel", back_populates="system", cascade="all, delete-orphan")
    nodes = relationship("ProcessingNodeModel", back_populates="system", cascade="all, delete-orphan")


class ProcessingNodeModel(Base):
    """
    Compute hardware running AI analytics (e.g., RDK-X5 edge box or Central GPU server).
    """
    __tablename__ = "processing_nodes"

    id = Column(String(64), primary_key=True, index=True)  # e.g., "RDK-X5-001", "KAAVAL-SERVER-AHM-001"
    name = Column(String(128), nullable=False)
    node_type = Column(String(32), nullable=False)  # "EDGE_RDK_X5", "SERVER_GPU", "CLOUD"
    system_id = Column(String(64), ForeignKey("systems.id"), nullable=True)
    ip_address = Column(String(64), nullable=True)
    status = Column(String(32), default="ONLINE")  # "ONLINE", "OFFLINE", "DEGRADED", "CRASHED"
    last_heartbeat = Column(DateTime(timezone=True), default=utcnow)
    metrics = Column(JSON, default=dict)  # {"cpu": 45.2, "mem": 60.1, "temp_c": 52.0, "fps": 15.2}
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    system = relationship("SystemModel", back_populates="nodes")
    assignments = relationship("CameraAssignmentModel", back_populates="node", cascade="all, delete-orphan")


class CameraModel(Base):
    """
    Physical or logical camera registered in Sentinel.
    Module 1 is the primary system of record; M3 maintains federation & streaming metadata.
    """
    __tablename__ = "cameras"

    id = Column(String(64), primary_key=True, index=True)  # Canonical ID: e.g., "CAM-001"
    external_id = Column(String(128), nullable=True, index=True)  # Vendor ID: e.g., "CAM_01" or "CH-07"
    name = Column(String(128), nullable=False)
    system_id = Column(String(64), ForeignKey("systems.id"), nullable=False)
    rtsp_url = Column(String(512), nullable=True)
    location_name = Column(String(256), nullable=True)  # e.g., "SG Highway Junction, Ahmedabad"
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    status = Column(String(32), default="ONLINE")  # "ONLINE", "OFFLINE", "STREAM_LOST"
    resolution = Column(String(32), default="1920x1080")
    fps = Column(Float, default=25.0)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    system = relationship("SystemModel", back_populates="cameras")
    assignments = relationship("CameraAssignmentModel", back_populates="camera", cascade="all, delete-orphan")


class CameraAssignmentModel(Base):
    """
    Maps which processing node runs AI analytics for which camera.
    Answers:
    1) Which processing node is responsible for this camera?
    2) If this processing node fails, which cameras are affected?
    """
    __tablename__ = "camera_assignments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), ForeignKey("cameras.id"), nullable=False, index=True)
    node_id = Column(String(64), ForeignKey("processing_nodes.id"), nullable=False, index=True)
    pipeline_profile = Column(String(64), default="TRAFFIC_ANALYTICS")  # e.g., "TRAFFIC_ANALYTICS", "ANPR_SPEED"
    status = Column(String(32), default="ACTIVE")  # "ACTIVE", "SUSPENDED"
    assigned_at = Column(DateTime(timezone=True), default=utcnow)

    # Relationships
    camera = relationship("CameraModel", back_populates="assignments")
    node = relationship("ProcessingNodeModel", back_populates="assignments")
