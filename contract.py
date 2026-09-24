"""
Driver contract shared by mock and real hardware drivers.

This is the ONLY module application code (Module 3 adapters, C6 probes, tests)
should import types from. Mock and real drivers both implement the protocols
below, so swapping one for the other is a config change, not a code change.

Mirrors the M3-C1 adapter contract v1.1 (video + AI capabilities) and its
terminal error states.
"""
from __future__ import annotations

import enum
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Capabilities & states
# --------------------------------------------------------------------------- #
class Capability(str, enum.Enum):
    LIVE_STREAM = "LIVE_STREAM"
    SNAPSHOT = "SNAPSHOT"
    PTZ = "PTZ"
    CLOCK = "CLOCK"
    HEALTH = "HEALTH"
    ENCODER_CONFIG = "ENCODER_CONFIG"
    EVENTS = "EVENTS"
    AI_CAPABILITIES = "AI_CAPABILITIES"
    MODEL_INFO = "MODEL_INFO"
    AI_HEALTH = "AI_HEALTH"


class ConnectionState(str, enum.Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"


class RuntimeState(str, enum.Enum):
    RUNNING = "RUNNING"
    CRASHED = "CRASHED"
    RESTARTING = "RESTARTING"


# --------------------------------------------------------------------------- #
# Error taxonomy (codes match the M3 spec terminal states)
# --------------------------------------------------------------------------- #
class ErrorCode(str, enum.Enum):
    CONNECTION_FAILED = "CONNECTION_FAILED"
    NOT_CONNECTED = "NOT_CONNECTED"
    DEVICE_DISCONNECTED = "DEVICE_DISCONNECTED"
    VENDOR_TIMEOUT = "VENDOR_TIMEOUT"
    VENDOR_AUTH_FAILED = "VENDOR_AUTH_FAILED"
    MAPPING_ERROR = "MAPPING_ERROR"
    CAPABILITY_NOT_SUPPORTED = "CAPABILITY_NOT_SUPPORTED"
    INVALID_COMMAND = "INVALID_COMMAND"


RETRYABLE = {ErrorCode.CONNECTION_FAILED, ErrorCode.DEVICE_DISCONNECTED, ErrorCode.VENDOR_TIMEOUT}


class DeviceError(Exception):
    code: ErrorCode = ErrorCode.CONNECTION_FAILED

    def __init__(self, message: str, *, device_id: str | None = None, detail: Any = None):
        super().__init__(message)
        self.device_id = device_id
        self.detail = detail

    @property
    def retryable(self) -> bool:
        return self.code in RETRYABLE

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value}, device={self.device_id}, msg={self})"


class ConnectionFailed(DeviceError):
    code = ErrorCode.CONNECTION_FAILED


class NotConnected(DeviceError):
    code = ErrorCode.NOT_CONNECTED


class DeviceDisconnected(DeviceError):
    code = ErrorCode.DEVICE_DISCONNECTED


class DeviceTimeout(DeviceError):
    code = ErrorCode.VENDOR_TIMEOUT


class AuthFailed(DeviceError):
    code = ErrorCode.VENDOR_AUTH_FAILED


class MappingError(DeviceError):
    code = ErrorCode.MAPPING_ERROR


class CapabilityNotSupported(DeviceError):
    code = ErrorCode.CAPABILITY_NOT_SUPPORTED


class InvalidCommand(DeviceError):
    code = ErrorCode.INVALID_COMMAND


# --------------------------------------------------------------------------- #
# Canonical DTOs (strict: unknown or malformed vendor data must fail mapping)
# --------------------------------------------------------------------------- #
class _DTO(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DeviceInfo(_DTO):
    device_id: str
    kind: Literal["camera", "ai_node"]
    vendor: str
    model: str
    firmware: str
    serial: str
    capabilities: tuple[Capability, ...]


class DeviceStatus(_DTO):
    device_id: str
    state: ConnectionState
    uptime_s: float
    observed_at: datetime
    fps: float | None = None
    bitrate_kbps: float | None = None


class StreamSource(_DTO):
    profile: Literal["MAIN", "SUB"]
    uri: str
    codec: Literal["H264", "H265"]
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    bitrate_kbps: int = Field(gt=0)


class Snapshot(_DTO):
    content_type: str
    data: bytes
    captured_at: datetime


class PtzPosition(_DTO):
    pan: float = Field(ge=-1.0, le=1.0)
    tilt: float = Field(ge=-1.0, le=1.0)
    zoom: float = Field(ge=0.0, le=1.0)
    moving: bool


class ClockInfo(_DTO):
    device_time: datetime
    offset_s: float  # device_time - reference time (positive = device ahead)


class AICapabilities(_DTO):
    node_id: str
    functions: tuple[str, ...]
    assigned_cameras: tuple[str, ...]


class ModelInfo(_DTO):
    analytics_provider: str
    analytics_version: str
    model_version: str
    loaded_at: datetime


class AIHealth(_DTO):
    node_id: str
    runtime: RuntimeState
    inference_fps: float = Field(ge=0)
    latency_ms: float = Field(ge=0)
    cpu_pct: float = Field(ge=0, le=100)
    mem_pct: float = Field(ge=0, le=100)
    temp_c: float
    throttled: bool
    observed_at: datetime


class AIEvent(_DTO):
    event_id: str
    node_id: str
    camera_external_id: str
    event_type: str  # Kaaval-native type; M3-C4 maps it to a Sentinel event_type
    occurred_at: datetime
    confidence: float = Field(ge=0, le=1)
    attributes: dict[str, Any]
    model_version: str


class Heartbeat(_DTO):
    node_id: str
    runtime: RuntimeState
    inference_fps: float
    sent_at: datetime


StreamItem = AIEvent | Heartbeat


# --------------------------------------------------------------------------- #
# Driver protocols
# --------------------------------------------------------------------------- #
@runtime_checkable
class DeviceDriver(Protocol):
    device_id: str

    @property
    def state(self) -> ConnectionState: ...

    def supports(self, capability: Capability) -> bool: ...

    async def connect(self) -> DeviceInfo: ...

    async def disconnect(self) -> None: ...

    async def get_status(self) -> DeviceStatus: ...


@runtime_checkable
class CameraDriver(DeviceDriver, Protocol):
    async def get_stream_uri(self, profile: Literal["MAIN", "SUB"] = "MAIN") -> StreamSource: ...

    async def get_snapshot(self) -> Snapshot: ...

    async def set_encoder(self, profile: Literal["MAIN", "SUB"], *, fps: int, bitrate_kbps: int) -> StreamSource: ...

    async def ptz_move(self, *, pan: float, tilt: float, zoom: float) -> None: ...

    async def ptz_get_position(self) -> PtzPosition: ...

    async def get_clock(self) -> ClockInfo: ...


@runtime_checkable
class AINodeDriver(DeviceDriver, Protocol):
    async def get_ai_capabilities(self) -> AICapabilities: ...

    async def get_model_info(self) -> ModelInfo: ...

    async def get_ai_health(self) -> AIHealth: ...

    async def load_model(self, model_version: str) -> ModelInfo: ...

    def stream(self, *, poll_interval_s: float = 1.0) -> AsyncIterator[StreamItem]: ...
