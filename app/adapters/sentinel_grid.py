"""
SENTINEL_GRID adapter (C1): the official Sentinel Camera Grid as a live-video source.

Capabilities: LIVE_STREAM and HEALTH. The grid is a set of live feeds: no PTZ, no playback, no export, no
events of its own (events come from the analytics that watch these feeds), so everything else raises
CAPABILITY_NOT_SUPPORTED.

Every call to the grid goes through one guard: 5 s timeouts, retry x2 with exponential backoff and jitter,
and a per-system circuit breaker, reporting failures with the shared M3 error codes. Nothing raised or
logged here contains the grid credentials.
"""
from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, Optional, TypeVar

from app.adapters.base import BaseAdapter
from app.core.config import Settings, get_settings
from app.grid.breaker import CircuitBreaker
from app.grid.probe import ProbeResult, probe_rtsp, probe_whep
from app.live.grid import GridCamera, GridClient, GridError
from app.live.redact import Redactor
from contract import AuthFailed, CapabilityNotSupported, ConnectionFailed, DeviceError

log = logging.getLogger("sentinel.m3.adapters.grid")
T = TypeVar("T")

CAPABILITIES = frozenset({"LIVE_STREAM", "HEALTH"})
RETRIES = 2


@dataclass(frozen=True)
class StreamCandidate:
    """One way to pull a camera, in priority order. `url` never carries credentials."""
    protocol: str            # "RTSP" | "WHEP" | "HLS"
    url: str
    priority: int
    note: str = ""


class SentinelGridAdapter(BaseAdapter):
    def __init__(self, settings: Settings | None = None, grid: GridClient | None = None,
                 breaker: CircuitBreaker | None = None, sleep: Callable[[float], None] = time.sleep):
        self._settings = settings
        self._grid = grid
        self._breaker = breaker or CircuitBreaker(threshold=5, reset_s=30)
        self._sleep = sleep
        self.hls_server_side: bool | None = None       # measured by connection_test(); None = not measured yet
        self.hls_note = "not checked"

    # -------------------------------------------------------------- plumbing
    @property
    def settings(self) -> Settings:
        return self._settings or get_settings()

    @property
    def grid(self) -> GridClient:
        if self._grid is None:
            self._grid = GridClient(self.settings)
        return self._grid

    @property
    def redactor(self) -> Redactor:
        return Redactor(*self.grid.secrets())

    @property
    def breaker(self) -> CircuitBreaker:
        return self._breaker

    def _guard(self, fn: Callable[[], T]) -> T:
        """5 s timeouts live in the callees; here: breaker, retry x2 with backoff + jitter, error mapping."""
        if not self._breaker.allow():
            raise ConnectionFailed("the Sentinel Grid circuit breaker is open: too many recent failures, retrying shortly")
        last: DeviceError | None = None
        for attempt in range(RETRIES + 1):
            try:
                result = fn()
            except AuthFailed as e:                     # wrong credentials: retrying will not help
                self._breaker.failure()
                raise e
            except GridError as e:
                last = AuthFailed(str(e)) if "rejected the login" in str(e) else ConnectionFailed(str(e))
            except DeviceError as e:
                last = e
            else:
                self._breaker.success()
                return result
            if isinstance(last, AuthFailed):
                self._breaker.failure()
                raise last
            if attempt < RETRIES:
                self._sleep(min(30.0, 0.5 * 2 ** attempt) * random.uniform(0.8, 1.2))
        self._breaker.failure()
        assert last is not None
        raise last

    # ---------------------------------------------------------- BaseAdapter
    @property
    def name(self) -> str:
        return "SENTINEL_GRID"

    @property
    def protocol(self) -> str:
        return "SENTINEL_GRID"

    @property
    def capabilities(self) -> frozenset[str]:
        return CAPABILITIES

    def supports(self, capability: str) -> bool:
        return capability in CAPABILITIES

    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        return False                                    # the grid does not push events

    def normalize(self, raw_payload, camera_id, processing_node_id=None, context=None):
        raise CapabilityNotSupported("SENTINEL_GRID provides live video only; events come from the analytics nodes")

    def check_health(self, target_id: str) -> Dict[str, Any]:
        """HEALTH: can we open this camera's stream right now?"""
        try:
            res = self.probe(target_id)
            return {"adapter": self.name, "target": target_id, "status": "ONLINE", "protocol": "RTSP",
                    "codec": res.codec, "resolution": res.resolution, "fps": res.fps}
        except DeviceError as e:
            return {"adapter": self.name, "target": target_id, "status": "OFFLINE", "protocol": "RTSP",
                    "error_code": e.code.value, "message": str(e)}

    # ------------------------------------------------- unsupported capabilities
    def _unsupported(self, capability: str):
        raise CapabilityNotSupported(f"SENTINEL_GRID does not support {capability}")

    def ptz(self, *_a, **_k):
        self._unsupported("PTZ")

    def playback(self, *_a, **_k):
        self._unsupported("PLAYBACK")

    def export_clip(self, *_a, **_k):
        self._unsupported("EXPORT")

    def snapshot(self, *_a, **_k):
        self._unsupported("SNAPSHOT")

    # ------------------------------------------------------------- inventory
    def inventory(self) -> list[GridCamera]:
        """Canonical camera descriptors, built from the grid's catalogue (never a hard-coded list)."""
        return self._guard(self.grid.fetch_catalogue)

    # ----------------------------------------------------------- live stream
    def get_stream_uri(self, cam_id: str) -> list[StreamCandidate]:
        """
        Source candidates in priority order: RTSP (TCP), then WHEP, then HLS. HLS is offered only when
        the grid actually lets a server-side client fetch it (measured in connection_test).
        """
        out = [StreamCandidate("RTSP", self.grid.rtsp_url_public(cam_id), 1, "tcp only"),
               StreamCandidate("WHEP", self.grid.whep_url_public(cam_id), 2, "basic auth")]
        if self.hls_server_side:
            out.append(StreamCandidate("HLS", self.grid.hls_url_public(cam_id), 3, "session cookie"))
        return out

    def source_url(self, protocol: str, cam_id: str) -> str:
        """
        Credentialed pull URL for the relay. Handed only to the code that writes the git-ignored relay
        config; it must never be logged, stored in the database or returned by an API.
        """
        if protocol == "RTSP":
            return self.grid.rtsp_url(cam_id)
        if protocol == "WHEP":
            return self.grid.whep_source_url(cam_id)
        raise CapabilityNotSupported(f"{protocol} cannot be pulled server-side from the grid")

    def probe(self, cam_id: str) -> ProbeResult:
        """ffprobe the camera over RTSP/TCP: codec, resolution, fps."""
        url = self.grid.rtsp_url(cam_id)
        red = self.redactor
        return self._guard(lambda: probe_rtsp(url, red))

    def probe_whep(self, cam_id: str) -> ProbeResult:
        user, pw = self.grid.basic_auth()
        return self._guard(lambda: probe_whep(self.grid.whep_url_public(cam_id), user, pw))

    # ---------------------------------------------- C7: connection test
    def connection_test(self, sample_camera: Optional[str] = None) -> dict[str, Any]:
        """
        The gate for registering the system as ACTIVE: catalogue fetch + one RTSP probe. Also measures
        whether HLS is usable server-side. Raises a DeviceError if the gate fails.
        """
        cameras = self._guard(self.grid.connection_test)
        cam = next((c for c in cameras if c.id == sample_camera), cameras[0])
        result = self.probe(cam.id)
        ok, note = self.grid.hls_server_side(cam.id)
        self.hls_server_side, self.hls_note = ok, note
        log.info("grid connection test passed: %d cameras, probe %s %s, HLS server-side: %s (%s)",
                 len(cameras), cam.id, result.codec, ok, note)
        return {"cameras": len(cameras), "probe_camera": cam.id, "codec": result.codec,
                "resolution": result.resolution, "hls_server_side": ok, "hls_note": note}
