"""
Common enumeration types and response wrappers.
"""
from __future__ import annotations

import enum
from typing import Generic, TypeVar, Optional, Any
from pydantic import BaseModel

T = TypeVar("T")


class SystemVendor(str, enum.Enum):
    HIKVISION = "HIKVISION"
    DAHUA = "DAHUA"
    KAAVAL = "KAAVAL"
    GENERIC_ONVIF = "GENERIC_ONVIF"
    GENERIC_RTSP = "GENERIC_RTSP"


class SystemProtocol(str, enum.Enum):
    ONVIF = "ONVIF"
    RTSP = "RTSP"
    KAAVAL_EDGE = "KAAVAL_EDGE"
    KAAVAL_SERVER = "KAAVAL_SERVER"


class NodeType(str, enum.Enum):
    EDGE_RDK_X5 = "EDGE_RDK_X5"
    SERVER_GPU = "SERVER_GPU"
    CLOUD = "CLOUD"


class ComponentStatus(str, enum.Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


class RuntimeStateEnum(str, enum.Enum):
    RUNNING = "RUNNING"
    CRASHED = "CRASHED"
    RESTARTING = "RESTARTING"


class PipeStateEnum(str, enum.Enum):
    UP = "UP"
    CUT = "CUT"


class ApiResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str = "Success"
    data: Optional[T] = None
