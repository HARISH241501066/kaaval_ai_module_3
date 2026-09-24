"""Mock camera driver: implements contract.CameraDriver against a simulated CameraModel."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from ..contract import Capability, ClockInfo, DeviceInfo, DeviceStatus, PtzPosition, Snapshot, StreamSource
from ..models.camera import CameraModel
from ..transport import SimTransport
from .base import BaseSimDriver

Profile = Literal["MAIN", "SUB"]
_BASE_CAPS = {Capability.LIVE_STREAM, Capability.SNAPSHOT, Capability.CLOCK,
              Capability.HEALTH, Capability.ENCODER_CONFIG}


class MockCameraDriver(BaseSimDriver):
    kind = "camera"

    def __init__(self, model: CameraModel, transport: SimTransport):
        super().__init__(model.device_id, transport)
        self.sim = model  # sim-only handle for tests/demos; application code must not use it

    async def _fetch_info(self):
        raw = await self._call("device_info", self.sim.device_information)

        def build(r):
            if not isinstance(r["PTZ"], bool):
                raise TypeError("PTZ flag must be boolean")
            caps = set(_BASE_CAPS) | ({Capability.PTZ} if r["PTZ"] else set())
            return DeviceInfo(device_id=self.device_id, kind="camera", vendor=r["Manufacturer"], model=r["Model"],
                              firmware=r["FirmwareVersion"], serial=r["SerialNumber"],
                              capabilities=tuple(sorted(caps, key=lambda c: c.value))), frozenset(caps)
        return self._map("device_info", raw, build)

    async def get_status(self) -> DeviceStatus:
        raw = await self._call("status", self.sim.system_status)
        return self._map("status", raw, lambda r: DeviceStatus(
            device_id=self.device_id, state=self.state, uptime_s=r["Uptime"], observed_at=self._now(),
            fps=r["Stats"]["FrameRate"], bitrate_kbps=r["Stats"]["Bitrate"]))

    def _to_stream(self, profile: Profile, r) -> StreamSource:
        vec = r["VideoEncoderConfiguration"]
        return StreamSource(profile=profile, uri=r["MediaUri"]["Uri"], codec=vec["Encoding"],
                            width=vec["Resolution"]["Width"], height=vec["Resolution"]["Height"],
                            fps=vec["RateControl"]["FrameRateLimit"], bitrate_kbps=vec["RateControl"]["BitrateLimit"])

    async def get_stream_uri(self, profile: Profile = "MAIN") -> StreamSource:
        self._require(Capability.LIVE_STREAM)
        raw = await self._call("get_stream_uri", lambda: self.sim.stream_uri(profile), {"profile": profile})
        return self._map("get_stream_uri", raw, lambda r: self._to_stream(profile, r))

    async def set_encoder(self, profile: Profile, *, fps: int, bitrate_kbps: int) -> StreamSource:
        self._require(Capability.ENCODER_CONFIG)
        params = {"profile": profile, "fps": fps, "bitrate_kbps": bitrate_kbps}
        raw = await self._call("set_encoder", lambda: self.sim.set_encoder(profile, fps, bitrate_kbps), params)
        return self._map("set_encoder", raw, lambda r: self._to_stream(profile, r))

    async def get_snapshot(self) -> Snapshot:
        self._require(Capability.SNAPSHOT)
        raw = await self._call("snapshot", self.sim.snapshot)
        return self._map("snapshot", raw, lambda r: Snapshot(
            content_type=r["ContentType"], data=r["Data"], captured_at=self._now()))

    async def ptz_move(self, *, pan: float, tilt: float, zoom: float) -> None:
        self._require(Capability.PTZ)
        await self._call("ptz_absolute_move", lambda: self.sim.ptz_absolute_move(pan, tilt, zoom),
                         {"pan": pan, "tilt": tilt, "zoom": zoom})

    async def ptz_get_position(self) -> PtzPosition:
        self._require(Capability.PTZ)
        raw = await self._call("ptz_status", self.sim.ptz_status)
        return self._map("ptz_status", raw, lambda r: PtzPosition(
            pan=r["Position"]["PanTilt"]["x"], tilt=r["Position"]["PanTilt"]["y"], zoom=r["Position"]["Zoom"]["x"],
            moving="MOVING" in (r["MoveStatus"]["PanTilt"], r["MoveStatus"]["Zoom"])))

    async def get_clock(self) -> ClockInfo:
        self._require(Capability.CLOCK)
        raw = await self._call("system_date", self.sim.system_date)

        def build(r):
            device_time = datetime.fromisoformat(r["UTCDateTime"])
            return ClockInfo(device_time=device_time, offset_s=round((device_time - self._now()).total_seconds(), 3))
        return self._map("system_date", raw, build)
