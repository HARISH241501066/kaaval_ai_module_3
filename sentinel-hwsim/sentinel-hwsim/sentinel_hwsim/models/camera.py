"""
Simulated IP camera (ONVIF-style). Holds device state and physics and answers
with *vendor-shaped* payloads (ONVIF-like field names), never canonical DTOs.
Mapping vendor -> canonical is the driver's job, exactly as with real hardware.
"""
from __future__ import annotations

import math
import random
import struct
import zlib
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from ..clock import SimClock


@dataclass
class EncoderProfile:
    codec: str
    width: int
    height: int
    fps: int
    bitrate_kbps: int
    path: str


def _default_profiles() -> dict[str, EncoderProfile]:
    return {
        "MAIN": EncoderProfile("H265", 2560, 1440, 25, 4096, "Streaming/Channels/101"),
        "SUB": EncoderProfile("H264", 640, 360, 15, 512, "Streaming/Channels/102"),
    }


def _fault(code: str, reason: str) -> dict[str, Any]:
    """ONVIF-style SOAP fault body."""
    return {"Fault": {"Code": code, "Reason": reason}}


def _png(width: int, height: int, rows: list[bytes]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + r for r in rows)
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit greyscale
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class CameraModel:
    PAN_TILT_SPEED = 0.5   # normalised units per second
    ZOOM_SPEED = 0.25
    TRAFFIC_CYCLE_S = 600  # simulated "rush hour" period for scene activity

    def __init__(self, device_id: str, clock: SimClock, rng: random.Random, *,
                 ip: str = "10.20.0.11", rtsp_base: str | None = None, serial: str | None = None,
                 vendor: str = "SimVision", model: str | None = None, firmware: str = "V5.7.3 build 260812",
                 clock_offset_s: float = 0.0, clock_drift_s_per_hour: float = 1.2, ptz: bool = True,
                 stream_files: dict[str, Any] | None = None, media_files: dict[str, Any] | None = None):
        self.device_id = device_id
        self.clock = clock
        self.rng = rng
        self.ip = ip
        self.rtsp_base = rtsp_base or f"rtsp://{ip}:554"
        self.stream_files = stream_files or media_files or {}
        self.serial = serial or f"SV{rng.randrange(10**9, 10**10)}"
        self.vendor, self.firmware = vendor, firmware
        self.model = model or ("SV-IPC-4MP-PTZ" if ptz else "SV-IPC-4MP-BULLET")
        self.has_ptz = ptz
        self.profiles = _default_profiles()
        self._clock_offset0 = clock_offset_s
        self._drift_per_s = clock_drift_s_per_hour / 3600
        self._boot_age_s = rng.uniform(86400, 30 * 86400)

        self.ptz_pos = [0.0, 0.0, 0.0]
        self.ptz_target = [0.0, 0.0, 0.0]
        self.actual_fps = {k: float(p.fps) for k, p in self.profiles.items()}
        self.actual_kbps = {k: float(p.bitrate_kbps) * 0.8 for k, p in self.profiles.items()}
        self._t = clock.now()

    # ----------------------------------------------------------------- physics
    def _advance(self) -> None:
        now = self.clock.now()
        dt, self._t = max(0.0, now - self._t), now
        if dt == 0:
            return
        # PTZ: move each axis toward its target at a fixed speed
        for i, speed in enumerate((self.PAN_TILT_SPEED, self.PAN_TILT_SPEED, self.ZOOM_SPEED)):
            diff = self.ptz_target[i] - self.ptz_pos[i]
            step = math.copysign(min(abs(diff), speed * dt), diff)
            self.ptz_pos[i] += step
        # Encoder: first-order approach to configured targets, scene-driven VBR bitrate
        alpha = 1 - math.exp(-dt / 2.0)
        activity = 0.6 + 0.4 * math.sin(2 * math.pi * now / self.TRAFFIC_CYCLE_S)
        for name, p in self.profiles.items():
            self.actual_fps[name] += alpha * (p.fps - self.actual_fps[name])
            target_kbps = p.bitrate_kbps * min(1.1, activity + self.rng.gauss(0, 0.05))
            self.actual_kbps[name] += alpha * (target_kbps - self.actual_kbps[name])

    def _noisy_fps(self, profile: str) -> float:
        return round(max(0.5, self.actual_fps[profile] + self.rng.gauss(0, 0.3)), 2)

    def _device_time(self):
        return self.clock.wall() + timedelta(seconds=self._clock_offset0 + self._drift_per_s * self.clock.now())

    # ------------------------------------------------------- vendor endpoints
    def device_information(self) -> dict[str, Any]:
        return {"Manufacturer": self.vendor, "Model": self.model, "FirmwareVersion": self.firmware,
                "SerialNumber": self.serial, "HardwareId": "SV-HW-0042", "PTZ": self.has_ptz}

    def system_status(self) -> dict[str, Any]:
        self._advance()
        return {"Uptime": round(self._boot_age_s + self.clock.now(), 1),
                "Stats": {"FrameRate": self._noisy_fps("MAIN"), "Bitrate": round(self.actual_kbps["MAIN"], 1)}}

    def stream_uri(self, profile: str) -> dict[str, Any]:
        if profile not in self.profiles:
            return _fault("ter:InvalidArgVal/ter:NoProfile", f"profile {profile} not found")
        self._advance()
        p = self.profiles[profile]

        # Use mapped local video file if configured for this profile
        if self.stream_files and profile in self.stream_files:
            target = self.stream_files[profile]
            if hasattr(target, "as_uri"):
                uri = target.as_uri()
            elif isinstance(target, str) and ("://" in target):
                uri = target
            else:
                p_obj = Path(target)
                if not p_obj.exists():
                    media_candidate = Path(__file__).resolve().parent.parent / "media" / target
                    if media_candidate.exists():
                        p_obj = media_candidate
                uri = p_obj.resolve().as_uri() if p_obj.exists() else str(target)
        else:
            uri = f"{self.rtsp_base.rstrip('/')}/{p.path}"

        return {"MediaUri": {"Uri": uri, "InvalidAfterConnect": False},
                "VideoEncoderConfiguration": {
                    "Encoding": p.codec, "Resolution": {"Width": p.width, "Height": p.height},
                    "RateControl": {"FrameRateLimit": p.fps, "BitrateLimit": p.bitrate_kbps}}}

    def set_encoder(self, profile: str, fps: int, bitrate_kbps: int) -> dict[str, Any]:
        if profile not in self.profiles:
            return _fault("ter:InvalidArgVal/ter:NoProfile", f"profile {profile} not found")
        if not (1 <= fps <= 30) or not (64 <= bitrate_kbps <= 16384):
            return _fault("ter:InvalidArgVal/ter:ConfigModify", "fps must be 1..30, bitrate 64..16384 kbps")
        self._advance()
        p = self.profiles[profile]
        p.fps, p.bitrate_kbps = fps, bitrate_kbps  # actual values converge over ~2s
        return self.stream_uri(profile)

    def snapshot(self) -> dict[str, Any]:
        """64x36 greyscale PNG: road gradient + a 'vehicle' whose position depends on time and pan."""
        self._advance()
        w, h = 64, 36
        car_x = int((self.clock.now() * 12 - self.ptz_pos[0] * 40) % w)
        rows = []
        for y in range(h):
            row = bytearray(min(255, 40 + y * 4) for _ in range(w))
            if 20 <= y <= 27:
                for x in range(car_x, min(w, car_x + 10)):
                    row[x] = 230
            rows.append(bytes(row))
        return {"ContentType": "image/png", "Data": _png(w, h, rows)}

    def ptz_absolute_move(self, pan: float, tilt: float, zoom: float) -> dict[str, Any]:
        if not self.has_ptz:
            return _fault("ter:ActionNotSupported/ter:PTZNotSupported", "device has no PTZ")
        if not (-1 <= pan <= 1 and -1 <= tilt <= 1 and 0 <= zoom <= 1):
            return _fault("ter:InvalidArgVal/ter:InvalidPosition", "position out of range")
        self._advance()
        self.ptz_target = [pan, tilt, zoom]
        return {}

    def ptz_status(self) -> dict[str, Any]:
        if not self.has_ptz:
            return _fault("ter:ActionNotSupported/ter:PTZNotSupported", "device has no PTZ")
        self._advance()
        pt_moving = any(abs(self.ptz_target[i] - self.ptz_pos[i]) > 1e-3 for i in (0, 1))
        z_moving = abs(self.ptz_target[2] - self.ptz_pos[2]) > 1e-3
        return {"Position": {"PanTilt": {"x": round(self.ptz_pos[0], 4), "y": round(self.ptz_pos[1], 4)},
                             "Zoom": {"x": round(self.ptz_pos[2], 4)}},
                "MoveStatus": {"PanTilt": "MOVING" if pt_moving else "IDLE", "Zoom": "MOVING" if z_moving else "IDLE"}}

    def system_date(self) -> dict[str, Any]:
        return {"UTCDateTime": self._device_time().isoformat()}
