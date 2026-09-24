"""
Pydantic schemas for System, Camera, Processing Node, and Camera Assignment.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field

from app.schemas.common import SystemVendor, SystemProtocol, NodeType, ComponentStatus


# --- System Schemas ---
class SystemBase(BaseModel):
    name: str = Field(..., example="VMS-AHMEDABAD-HQ")
    vendor: SystemVendor = Field(..., example=SystemVendor.HIKVISION)
    protocol: SystemProtocol = Field(..., example=SystemProtocol.ONVIF)
    connection_params: Dict[str, Any] = Field(default_factory=dict)
    status: str = Field(default="ACTIVE")


class SystemCreate(SystemBase):
    id: str = Field(..., example="VMS-HIK-001")


class SystemRead(SystemBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --- Processing Node Schemas ---
class ProcessingNodeBase(BaseModel):
    name: str = Field(..., example="RDK-X5-SG-HIGHWAY-01")
    node_type: NodeType = Field(..., example=NodeType.EDGE_RDK_X5)
    system_id: Optional[str] = Field(None, example="KAAVAL-EDGE-SYSTEM")
    ip_address: Optional[str] = Field(None, example="192.168.1.101")
    status: str = Field(default="ONLINE")
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ProcessingNodeCreate(ProcessingNodeBase):
    id: str = Field(..., example="RDK-X5-001")


class ProcessingNodeRead(ProcessingNodeBase):
    id: str
    last_heartbeat: datetime
    created_at: datetime

    class Config:
        from_attributes = True


# --- Camera Schemas ---
class CameraBase(BaseModel):
    name: str = Field(..., example="SG Highway Junction 1")
    external_id: Optional[str] = Field(None, example="CAM_01")
    system_id: str = Field(..., example="VMS-HIK-001")
    rtsp_url: Optional[str] = Field(None, example="rtsp://192.168.1.50:554/live/ch0")
    location_name: Optional[str] = Field(None, example="SG Highway Junction, Ahmedabad")
    latitude: Optional[float] = Field(None, example=23.0225)
    longitude: Optional[float] = Field(None, example=72.5714)
    status: str = Field(default="ONLINE")
    resolution: str = Field(default="1920x1080")
    fps: float = Field(default=25.0)


class CameraCreate(CameraBase):
    id: str = Field(..., example="CAM-001")


class CameraRead(CameraBase):
    id: str
    created_at: datetime

    class Config:
        from_attributes = True


# --- Stream Source Schema (C3 Broker) ---
class StreamSourceRead(BaseModel):
    camera_id: str
    relay_url: str
    codec: str = "H264"
    width: int = 1920
    height: int = 1080
    fps: float = 25.0
    status: str = "AVAILABLE"


# --- Camera Assignment Schemas ---
class CameraAssignmentCreate(BaseModel):
    camera_id: str = Field(..., example="CAM-001")
    node_id: str = Field(..., example="RDK-X5-001")
    pipeline_profile: str = Field(default="TRAFFIC_ANALYTICS", example="TRAFFIC_ANALYTICS")
    status: str = Field(default="ACTIVE")


class CameraAssignmentRead(BaseModel):
    id: int
    camera_id: str
    node_id: str
    pipeline_profile: str
    status: str
    assigned_at: datetime

    class Config:
        from_attributes = True


# --- Node with Assigned Cameras Response ---
class NodeWithCamerasRead(BaseModel):
    node: ProcessingNodeRead
    assigned_cameras: List[CameraRead]
