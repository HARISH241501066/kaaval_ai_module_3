"""
Real hardware driver skeletons.

They implement the same contract as the mocks, so application code never
changes when you flip M3_DRIVER_MODE from "mock" to "real". Each method names
the vendor operation it must call. Fill these in when hardware is on the desk,
then run tests/test_conformance.py against them (set M3_CONFORMANCE_REAL=1).

Rules the real drivers must keep (the mocks already do):
  * credentials come from a CredentialProvider (M3-C7 / Vault), never config
  * every vendor call has a deadline -> DeviceTimeout
  * 401/403 -> AuthFailed; socket reset -> DeviceDisconnected
  * vendor payloads go through strict DTOs; any mismatch -> MappingError
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol

from ..contract import (
    AICapabilities,
    AIHealth,
    Capability,
    ClockInfo,
    ConnectionState,
    DeviceInfo,
    DeviceStatus,
    ModelInfo,
    PtzPosition,
    Snapshot,
    StreamItem,
    StreamSource,
)


class CredentialProvider(Protocol):
    async def resolve(self, credential_ref: str) -> tuple[str, str]:
        """Return (username, secret) for a C7 credential reference. Short-lived; never log it."""
        ...


class _RealBase:
    is_simulated = False

    def __init__(self, device_id: str, endpoint: str, credential_ref: str,
                 credentials: CredentialProvider, *, deadline_s: float = 5.0):
        self.device_id, self.endpoint, self.credential_ref = device_id, endpoint, credential_ref
        self._credentials, self.deadline_s = credentials, deadline_s
        self._state = ConnectionState.DISCONNECTED
        self._caps: frozenset[Capability] = frozenset()

    @property
    def state(self) -> ConnectionState:
        return self._state

    def supports(self, capability: Capability) -> bool:
        return capability in self._caps

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()

    def _todo(self, op: str):
        raise NotImplementedError(f"{type(self).__name__}.{op}: implement against real hardware")


class OnvifCameraDriver(_RealBase):
    """ONVIF Profile S/T camera via onvif-zeep (wrap blocking calls in asyncio.to_thread + wait_for)."""
    kind = "camera"

    async def connect(self) -> DeviceInfo:
        # user, pwd = await self._credentials.resolve(self.credential_ref)
        # cam = ONVIFCamera(host, port, user, pwd); devicemgmt.GetDeviceInformation(); GetCapabilities()
        self._todo("connect")

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def get_status(self) -> DeviceStatus:
        self._todo("get_status")          # GetSystemDateAndTime + RTSP OPTIONS probe for liveness

    async def get_stream_uri(self, profile: Literal["MAIN", "SUB"] = "MAIN") -> StreamSource:
        self._todo("get_stream_uri")      # media.GetProfiles -> GetStreamUri(RTP-Unicast/RTSP)

    async def set_encoder(self, profile, *, fps: int, bitrate_kbps: int) -> StreamSource:
        self._todo("set_encoder")         # media.GetVideoEncoderConfiguration -> SetVideoEncoderConfiguration

    async def get_snapshot(self) -> Snapshot:
        self._todo("get_snapshot")        # media.GetSnapshotUri -> HTTP GET with digest auth

    async def ptz_move(self, *, pan: float, tilt: float, zoom: float) -> None:
        self._todo("ptz_move")            # ptz.AbsoluteMove

    async def ptz_get_position(self) -> PtzPosition:
        self._todo("ptz_get_position")    # ptz.GetStatus

    async def get_clock(self) -> ClockInfo:
        self._todo("get_clock")           # devicemgmt.GetSystemDateAndTime (UTCDateTime)


class KaavalHttpDriver(_RealBase):
    """Kaaval node management API over HTTPS + mTLS (httpx.AsyncClient with cert=..., timeout=deadline)."""
    kind = "ai_node"

    async def connect(self) -> DeviceInfo:
        self._todo("connect")             # GET /v1/node/info

    async def disconnect(self) -> None:
        self._state = ConnectionState.DISCONNECTED

    async def get_status(self) -> DeviceStatus:
        self._todo("get_status")

    async def get_ai_capabilities(self) -> AICapabilities:
        self._todo("get_ai_capabilities")  # GET /v1/node/capabilities

    async def get_model_info(self) -> ModelInfo:
        self._todo("get_model_info")       # GET /v1/node/model

    async def get_ai_health(self) -> AIHealth:
        self._todo("get_ai_health")        # GET /v1/node/health

    async def load_model(self, model_version: str) -> ModelInfo:
        self._todo("load_model")           # PUT /v1/node/model, then poll health until RUNNING

    def stream(self, *, poll_interval_s: float = 1.0) -> AsyncIterator[StreamItem]:
        # Production: Kaaval pushes to M3-C4 (HMAC/mTLS). Pull variant: GET /v1/node/events?cursor=
        self._todo("stream")
