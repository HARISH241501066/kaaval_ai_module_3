"""
Stream probing for the Sentinel Camera Grid (the logic of tools/grid/check_grid.py as a proper module).

* ffprobe is called with an argument LIST (never a shell string), `-rtsp_transport tcp`, a 5 s connect
  timeout and a hard subprocess timeout on top.
* Everything ffprobe prints goes through the redactor before it can reach a log or an exception: its
  error messages quote the URL, credentials included.
* Failures are classified with the shared M3 error codes.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass

import httpx

from app.grid.urls import norm_codec
from app.live.redact import Redactor
from contract import AuthFailed, ConnectionFailed, DeviceError, DeviceTimeout

log = logging.getLogger("sentinel.m3.grid.probe")

CONNECT_TIMEOUT_S = 5
HARD_TIMEOUT_S = CONNECT_TIMEOUT_S + 10


@dataclass(frozen=True)
class ProbeResult:
    codec: str | None            # normalised: 'h264' | 'h265' | ...
    width: int | None
    height: int | None
    fps: float | None

    @property
    def resolution(self) -> str | None:
        return f"{self.width}x{self.height}" if self.width and self.height else None


def _fps(rate: str | None) -> float | None:
    try:
        num, den = (rate or "").split("/")
        return round(float(num) / float(den), 2) if float(den) else None
    except (ValueError, ZeroDivisionError):
        return None


def probe_rtsp(url: str, redactor: Redactor, *, connect_timeout_s: int = CONNECT_TIMEOUT_S,
               hard_timeout_s: int = HARD_TIMEOUT_S) -> ProbeResult:
    """ffprobe an RTSP URL (also a relay URL). Raises a DeviceError subclass, with credentials scrubbed."""
    exe = shutil.which("ffprobe")
    if not exe:
        raise ConnectionFailed("ffprobe is not installed or not on PATH")
    cmd = [exe, "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=codec_name,width,height,avg_frame_rate", "-of", "json",
           "-rtsp_transport", "tcp", "-timeout", str(int(connect_timeout_s * 1_000_000)), url]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=hard_timeout_s)
    except subprocess.TimeoutExpired:
        raise DeviceTimeout("probe timed out") from None
    if out.returncode != 0:
        err = redactor.scrub(out.stderr.strip().splitlines()[-1] if out.stderr.strip() else "probe failed")
        if "401" in err or "nauthorized" in err:
            raise AuthFailed(f"probe rejected the credentials ({err[:120]})")
        if "timed out" in err.lower() or "timeout" in err.lower():
            raise DeviceTimeout(f"probe timed out ({err[:120]})")
        raise ConnectionFailed(f"probe failed ({err[:120]})")
    try:
        stream = json.loads(out.stdout)["streams"][0]
    except (ValueError, KeyError, IndexError):
        raise ConnectionFailed("probe found no video stream") from None
    return ProbeResult(norm_codec(stream.get("codec_name")), stream.get("width"), stream.get("height"),
                       _fps(stream.get("avg_frame_rate")))


def probe_whep(url: str, user: str, password: str, *, timeout_s: float = CONNECT_TIMEOUT_S) -> ProbeResult:
    """
    WHEP cannot be read by ffprobe. Reachability check instead: an authenticated OPTIONS must answer 2xx.
    (A full offer/answer handshake is exercised by the relay itself when it pulls the stream.)
    """
    try:
        r = httpx.request("OPTIONS", url, auth=(user, password), timeout=timeout_s)
    except httpx.TimeoutException:
        raise DeviceTimeout("WHEP endpoint timed out") from None
    except httpx.HTTPError as e:
        raise ConnectionFailed(f"WHEP endpoint unreachable ({e.__class__.__name__})") from None
    if r.status_code in (401, 403):
        raise AuthFailed("WHEP endpoint rejected the credentials")
    if r.status_code >= 300:
        raise ConnectionFailed(f"WHEP endpoint answered HTTP {r.status_code}")
    return ProbeResult(None, None, None, None)


__all__ = ["ProbeResult", "probe_rtsp", "probe_whep", "DeviceError"]
