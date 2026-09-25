"""
Sentinel M3 hardware console: a local web UI over sentinel-hwsim.

    python ui/server.py                 # then open http://127.0.0.1:8765
    python ui/server.py --speed 10 --open

Stdlib-only HTTP + Server-Sent Events server. It builds a simulated fleet
(4 cameras, 1 Kaaval edge node), monitors it through the driver contract the
way M3-C6 would, and exposes the chaos controls so failures can be triggered
from the browser.

Like examples/, this is demo tooling: the chaos endpoints use SimControls. The
monitoring code (Fleet._camera_loop, Fleet._node_*, classify_node) only uses
the contract, so it would run unchanged against real drivers.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import ipaddress
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from pathlib import Path
from typing import Any

from sentinel_hwsim import (
    AIEvent,
    AINodeDriver,
    Capability,
    ConnectionState,
    DeviceError,
    ErrorCode,
    RuntimeState,
    Simulator,
    create_driver,
)
from sentinel_hwsim.faults import Fault

log = logging.getLogger("sentinel_console")
STATIC = Path(__file__).parent / "static"
STATIC_TYPES = {"index.html": "text/html; charset=utf-8", "styles.css": "text/css; charset=utf-8",
                "app.js": "text/javascript; charset=utf-8", "wall.js": "text/javascript; charset=utf-8", "favicon.svg": "image/svg+xml"}


def build_csp(relay_origins: list[str]) -> str:
    """Strict CSP. The only origins beyond our own are the stream relay's (WebRTC signalling and HLS), never the camera grid."""
    relay = " ".join(relay_origins)
    return (f"default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; "
            f"connect-src 'self' {relay}; media-src 'self' blob: {relay}; base-uri 'none'; form-action 'none'; "
            "frame-ancestors 'none'")


def default_relay_origins() -> list[str]:
    host = os.getenv("RELAY_PUBLIC_HOST", "localhost")
    return [f"http://{host}:{os.getenv('RELAY_WEBRTC_PORT', '8889')}", f"http://{host}:{os.getenv('RELAY_HLS_PORT', '8888')}"]


# Video source. `real` (the default) shows the live Sentinel Camera Grid through Module 3's relay; `mock` plays the
# recorded sample clips instead, fully offline. The simulated devices (health, AI node) are simulated in both modes.
VIDEO_MODES = ("real", "mock")

# Recorded clips that stand in for the cameras' video (the simulator has no pixels of its own).
FEED_FILES = {"MAIN": "camera_main.mp4", "SUB": "camera_sub.mp4"}
FEED_CHUNK = 256 * 1024


def find_feed_dir() -> Path | None:
    """Nearest `camera_feed/` folder in this file's parent directories (the workspace root, normally)."""
    return next((p / "camera_feed" for p in Path(__file__).resolve().parents if (p / "camera_feed").is_dir()), None)


HEARTBEAT_S = 5.0
NODE_ID = "EDGE-NODE-001"
# In M3 this comes from the C7 registry.
FLEET = [
    {"kind": "camera", "device_id": "CAM-001", "sim": {"faults": "healthy", "ip": "10.20.0.11", "clock_offset_s": 37}},
    {"kind": "camera", "device_id": "CAM-002", "sim": {"faults": "healthy", "ip": "10.20.0.12", "ptz": False}},
    {"kind": "camera", "device_id": "CAM-003", "sim": {"faults": "flaky", "ip": "10.20.0.13"}},
    {"kind": "camera", "device_id": "CAM-004", "sim": {"faults": "healthy", "ip": "10.20.0.14", "ptz": False}},
    {"kind": "ai_node", "device_id": NODE_ID,
     "sim": {"heartbeat_interval_s": HEARTBEAT_S, "cameras": ["CAM-001", "CAM-002", "CAM-003", "CAM-004"]}},
]
CRITICAL = {"OFFLINE", "AUTH_FAILED", "RUNTIME_DOWN", "INFERENCE_DEGRADED", "PIPE_DISCONNECTED", "NODE_OFFLINE"}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def err(e: DeviceError) -> dict[str, Any]:
    return {"code": e.code.value, "message": str(e), "retryable": e.retryable, "at": utcnow()}


# --------------------------------------------------------------------------- #
# Monitoring (contract-only application logic)
# --------------------------------------------------------------------------- #
async def classify_node(node: AINodeDriver, last_heartbeat_at, now, connected_for_s: float):
    """Layered check from M3-C6: node -> runtime -> inference -> pipe. Returns (state, health|None, layers)."""
    layers = {"node": False, "runtime": False, "inference": False, "pipe": False}
    try:
        h = await node.get_ai_health()
    except DeviceError as e:
        return "NODE_OFFLINE", None, layers, err(e)
    layers["node"] = True
    if h.runtime is RuntimeState.CRASHED:
        return "RUNTIME_DOWN", h, layers, None
    if h.runtime is RuntimeState.RESTARTING:
        return "RESTARTING", h, layers, None
    layers["runtime"] = True
    if h.inference_fps == 0:
        return "INFERENCE_DEGRADED", h, layers, None
    layers["inference"] = True
    if last_heartbeat_at is None:
        return ("STARTING" if connected_for_s < 3 * HEARTBEAT_S else "PIPE_DISCONNECTED"), h, layers, None
    if now - last_heartbeat_at > timedelta(seconds=3 * HEARTBEAT_S):
        return "PIPE_DISCONNECTED", h, layers, None
    layers["pipe"] = True
    return ("THROTTLED" if h.throttled else "HEALTHY"), h, layers, None


class Fleet:
    def __init__(self, *, speed: float, seed: int, feed_dir: Path | None = None, video_mode: str = "real"):
        self.speed, self.seed, self.video_mode = speed, seed, video_mode
        if video_mode == "real":
            feed_dir = None                      # no looping sample video anywhere in real mode
        self.sim = Simulator(speed=speed, seed=seed)
        self.drivers = {d["device_id"]: create_driver(d, simulator=self.sim) for d in FLEET}
        self.cameras = [d["device_id"] for d in FLEET if d["kind"] == "camera"]
        # feed clips that exist on disk, by profile -> URL
        self.feeds = {p: f"/feed/{p.lower()}.mp4" for p, name in FEED_FILES.items() if feed_dir and (feed_dir / name).is_file()}
        # cameras alternate between the two clips: CAM-001 MAIN, CAM-002 SUB, CAM-003 MAIN, CAM-004 SUB
        self.cam: dict[str, dict[str, Any]] = {c: {"id": c, "health": "CONNECTING", "info": None, "status": None,
                                                    "streams": {}, "ptz": None, "last_error": None, "errors": 0,
                                                    "feed": "MAIN" if i % 2 == 0 else "SUB"}
                                               for i, c in enumerate(self.cameras)}
        self.node: dict[str, Any] = {"id": NODE_ID, "health": "CONNECTING", "info": None, "capabilities": None,
                                     "model": None, "ai_health": None, "layers": {}, "last_error": None,
                                     "last_heartbeat_at": None, "events_received": 0, "heartbeats_received": 0,
                                     "events_last_min": 0, "dead_letters": 0, "watchlist": [], "ambient_c": 34.0}
        self.events: deque[dict[str, Any]] = deque(maxlen=200)
        self.alerts: deque[dict[str, Any]] = deque(maxlen=100)
        self.dlq: deque[dict[str, Any]] = deque(maxlen=50)
        self._arrivals: deque[float] = deque()
        self._alert_ids = 0
        self._node_connected_at: float | None = None
        self.subscribers: set[asyncio.Queue[str]] = set()
        self.started_at = utcnow()

    # ------------------------------------------------------------- pub/sub
    def publish(self, kind: str, data: Any) -> None:
        msg = f"event: {kind}\ndata: {json.dumps(data, default=str)}\n\n"
        for q in list(self.subscribers):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                self.subscribers.discard(q)  # slow client: drop it; EventSource reconnects and resyncs

    def snapshot(self) -> dict[str, Any]:
        return {"meta": {"speed": self.speed, "seed": self.seed, "started_at": self.started_at,
                         "heartbeat_s": HEARTBEAT_S, "feeds": self.feeds, "video_mode": self.video_mode},
                "cameras": [self.cam[c] for c in self.cameras], "node": self.node,
                "events": list(self.events)[-50:], "alerts": list(self.alerts), "dlq": list(self.dlq)}

    def _transition(self, device: str, old: str, new: str, detail: str) -> None:
        """One alert per transition into a bad state; resolve open alerts on recovery."""
        if old == new:
            return
        if new in CRITICAL or new == "THROTTLED":
            self._alert_ids += 1
            alert = {"id": self._alert_ids, "at": utcnow(), "device": device, "state": new,
                     "severity": "critical" if new in CRITICAL else "warning", "message": detail, "resolved_at": None}
            self.alerts.append(alert)
            self.publish("alert", alert)
        elif new in ("HEALTHY", "ONLINE"):
            for a in self.alerts:
                if a["device"] == device and a["resolved_at"] is None:
                    a["resolved_at"] = utcnow()
                    self.publish("alert", a)

    # ------------------------------------------------------------- cameras
    def _set_cam_health(self, cid: str, new: str, detail: str = "") -> None:
        st = self.cam[cid]
        old, st["health"] = st["health"], new
        self._transition(cid, old, new, detail or f"camera {new.lower().replace('_', ' ')}")

    async def _camera_loop(self, cid: str) -> None:
        d, st = self.drivers[cid], self.cam[cid]
        recent: deque[bool] = deque(maxlen=5)
        connect_failures = 0
        while True:
            if d.state is not ConnectionState.CONNECTED:
                try:
                    st["info"] = (await d.connect()).model_dump(mode="json")
                    connect_failures = 0
                except DeviceError as e:
                    connect_failures += 1
                    st["last_error"], st["errors"] = err(e), st["errors"] + 1
                    if e.code is ErrorCode.VENDOR_AUTH_FAILED:
                        self._set_cam_health(cid, "AUTH_FAILED", f"{cid} rejected credentials; no retry until healed")
                    elif connect_failures >= 2:
                        self._set_cam_health(cid, "OFFLINE", f"{cid} unreachable ({e.code.value}); stream unavailable")
                    else:
                        self._set_cam_health(cid, "RECONNECTING")
                    self.publish("camera", st)
                    await asyncio.sleep(min(8.0, 0.5 * 2 ** connect_failures))
                    continue
            try:
                for profile in ("MAIN", "SUB"):
                    if profile not in st["streams"]:
                        st["streams"][profile] = (await d.get_stream_uri(profile)).model_dump(mode="json")
                st["status"] = (await d.get_status()).model_dump(mode="json")
                if d.supports(Capability.PTZ):
                    st["ptz"] = (await d.ptz_get_position()).model_dump(mode="json")
                recent.append(False)
            except DeviceError as e:
                st["last_error"], st["errors"] = err(e), st["errors"] + 1
                recent.append(True)
                # timeouts keep the session "up"; three failed polls in a row means the device is gone
                if e.retryable and len(recent) >= 3 and all(list(recent)[-3:]):
                    await d.disconnect()
                    recent.clear()
            if d.state is ConnectionState.CONNECTED:
                self._set_cam_health(cid, "DEGRADED" if any(recent) else "ONLINE")
            self.publish("camera", st)
            await asyncio.sleep(1.0)

    # ---------------------------------------------------------------- node
    async def _node_stream(self) -> None:
        node: AINodeDriver = self.drivers[NODE_ID]
        while True:
            try:
                if node.state is not ConnectionState.CONNECTED:
                    self.node["info"] = (await node.connect()).model_dump(mode="json")
                    self._node_connected_at = time.monotonic()
                    self.node["capabilities"] = (await node.get_ai_capabilities()).model_dump(mode="json")
                    self.node["model"] = (await node.get_model_info()).model_dump(mode="json")
                async for item in node.stream(poll_interval_s=1.0):
                    if isinstance(item, AIEvent):
                        ev = item.model_dump(mode="json")
                        ev["received_at"] = utcnow()
                        self.events.append(ev)
                        self._arrivals.append(time.monotonic())
                        self.publish("ai_event", ev)
                    self._sync_node_counters(node)
            except DeviceError as e:
                self.node["last_error"] = err(e)
                await asyncio.sleep(1.0)

    def _sync_node_counters(self, node) -> None:
        # receive-side bookkeeping kept by the driver (pipe layer + DLQ)
        self.node["events_received"] = node.events_received
        self.node["heartbeats_received"] = node.heartbeats_received
        self.node["last_heartbeat_at"] = node.last_heartbeat_at.isoformat() if node.last_heartbeat_at else None
        self.node["heartbeat_age_s"] = (round((self.sim.clock.wall() - node.last_heartbeat_at).total_seconds(), 1)
                                        if node.last_heartbeat_at else None)
        if len(node.dead_letters) != self.node["dead_letters"]:
            for dl in list(node.dead_letters)[self.node["dead_letters"]:]:
                entry = {**dl, "at": utcnow()}
                self.dlq.append(entry)
                self.publish("dlq", entry)
            self.node["dead_letters"] = len(node.dead_letters)

    async def _node_health(self) -> None:
        node: AINodeDriver = self.drivers[NODE_ID]
        while True:
            connected_for = time.monotonic() - self._node_connected_at if self._node_connected_at else 0.0
            state, h, layers, e = await classify_node(node, node.last_heartbeat_at, self.sim.clock.wall(),
                                                      connected_for * self.speed)
            if state == "NODE_OFFLINE" and node.state is not ConnectionState.CONNECTED and connected_for == 0.0:
                state = "CONNECTING"
            cams = ", ".join(self.node["capabilities"]["assigned_cameras"]) if self.node["capabilities"] else "?"
            impact = "AI degraded on" if state == "THROTTLED" else "lost AI on"
            old, self.node["health"] = self.node["health"], state
            self._transition(NODE_ID, old, state, f"{NODE_ID} {state}: {impact} {cams}")
            self.node["ai_health"] = h.model_dump(mode="json") if h else None
            self.node["layers"] = layers
            if e:
                self.node["last_error"] = e
            cutoff = time.monotonic() - 60
            while self._arrivals and self._arrivals[0] < cutoff:
                self._arrivals.popleft()
            self.node["events_last_min"] = len(self._arrivals)
            self._sync_node_counters(node)
            self.publish("node", self.node)
            await asyncio.sleep(1.0)

    def start(self) -> list[asyncio.Task]:
        tasks = [asyncio.create_task(self._camera_loop(c)) for c in self.cameras]
        tasks += [asyncio.create_task(self._node_stream()), asyncio.create_task(self._node_health())]
        return tasks

    # ------------------------------------------------------------ commands
    def trace(self, limit: int) -> list[dict[str, Any]]:
        entries = [e for d in self.drivers.values() for e in list(d.trace)[-limit:]]
        entries.sort(key=lambda e: e.request_id)
        return [{"id": e.request_id, "at": e.at.isoformat(), "device": e.device_id, "op": e.op, "params": e.params,
                 "latency_ms": round(e.latency_ms, 1), "outcome": e.outcome} for e in entries[-limit:]]

    async def command(self, device: str, action: str, body: dict[str, Any]) -> dict[str, Any]:
        if device not in self.drivers:
            raise LookupError(f"unknown device {device}")
        d, ctl = self.drivers[device], self.sim.controls(device)
        match action:
            case "ptz":
                await d.ptz_move(pan=float(body["pan"]), tilt=float(body["tilt"]), zoom=float(body["zoom"]))
                return {"ok": True, "message": f"{device}: PTZ move accepted"}
            case "encoder":
                src = await d.set_encoder(body["profile"], fps=int(body["fps"]), bitrate_kbps=int(body["bitrate_kbps"]))
                self.cam[device]["streams"][src.profile] = src.model_dump(mode="json")
                self.publish("camera", self.cam[device])
                return {"ok": True, "message": f"{device}: {src.profile} set to {src.fps:g} fps, {src.bitrate_kbps} kbps"}
            case "fault":
                fault = Fault(body["fault"])
                ctl.fail_next(fault)
                return {"ok": True, "message": f"{device}: next operation will fail with {fault.value}"}
            case "offline":
                ctl.offline(bool(body.get("on", True)))
                return {"ok": True, "message": f"{device}: {'powered off / unreachable' if body.get('on', True) else 'back online'}"}
            case "revoke":
                ctl.revoke_credentials(bool(body.get("on", True)))
                return {"ok": True, "message": f"{device}: credentials {'revoked' if body.get('on', True) else 'restored'}"}
            case "heal":
                ctl.heal()
                return {"ok": True, "message": f"{device}: all faults cleared"}
        if device != NODE_ID:
            raise LookupError(f"unknown camera action {action}")
        match action:
            case "crash":
                ctl.crash_runtime()
            case "restart":
                ctl.restart_runtime()
            case "stall" | "unstall":
                ctl.stall_inference(action == "stall")
            case "cut" | "restore":
                ctl.cut_pipe(action == "cut")
            case "ambient":
                celsius = float(body["celsius"])
                if not 0 <= celsius <= 70:
                    raise ValueError("ambient must be between 0 and 70 °C")
                ctl.set_ambient(celsius)
                self.node["ambient_c"] = celsius
            case "watchlist":
                plates = sorted({p.strip().upper() for p in body.get("plates", []) if p.strip()})
                bad = [p for p in plates if not re.fullmatch(r"[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}", p)]
                if bad:
                    raise ValueError(f"not a valid Indian plate: {', '.join(bad)}")
                ctl.set_watchlist(plates)
                self.node["watchlist"] = plates
            case "model":
                info = await d.load_model(str(body["model_version"]))
                self.node["model"] = info.model_dump(mode="json")
                return {"ok": True, "message": f"model {info.model_version} loaded and confirmed"}
            case _:
                raise LookupError(f"unknown node action {action}")
        return {"ok": True, "message": f"{device}: {action} applied"}


# --------------------------------------------------------------------------- #
# Minimal HTTP/1.1 server (one request per connection; SSE keeps it open)
# --------------------------------------------------------------------------- #
class Server:
    def __init__(self, fleet: Fleet, host: str, port: int, feed_dir: Path | None = None, *,
                 m3_url: str = "http://127.0.0.1:8000", relay_origins: list[str] | None = None):
        self.fleet, self.host, self.port = fleet, host, port
        self.feed_dir = None if fleet.video_mode == "real" else feed_dir
        self.m3_url = m3_url.rstrip("/")
        self.relay_origins = relay_origins if relay_origins is not None else default_relay_origins()
        self.csp = build_csp(self.relay_origins)
        self.loopback = ipaddress.ip_address(host).is_loopback if host != "localhost" else True
        self.allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
        self._gz: dict[str, bytes] = {}

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            lines = head.decode("latin-1").split("\r\n")
            method, target, _ = lines[0].split(" ", 2)
            headers = {k.strip().lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:] if ln)}
            length = int(headers.get("content-length") or 0)
            if length > 64 * 1024:
                return await self.json(writer, 413, {"error": "PAYLOAD_TOO_LARGE", "message": "body too large"})
            body = await reader.readexactly(length) if length else b""
            if self.loopback and headers.get("host") not in self.allowed_hosts:
                return await self.json(writer, 421, {"error": "BAD_HOST", "message": "unexpected Host header"})
            await self.route(method, target, headers, body, writer)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError, ValueError, ConnectionError):
            pass
        except Exception:  # never let one request kill the server
            log.exception("request failed")
            try:
                await self.json(writer, 500, {"error": "INTERNAL", "message": "internal error, see server log"})
            except ConnectionError:
                pass
        finally:
            writer.close()

    @staticmethod
    def head(status: int, ctype: str, length: int, extra: dict[str, str] | None = None) -> bytes:
        hdrs = {"Content-Type": ctype, "Content-Length": str(length), "Connection": "close",
                "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store",
                **(extra or {})}
        text = f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n" + "".join(f"{k}: {v}\r\n" for k, v in hdrs.items())
        return text.encode("latin-1") + b"\r\n"

    async def send(self, writer, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        writer.write(self.head(status, ctype, len(body), extra) + body)
        await writer.drain()

    async def feed(self, writer, headers: dict[str, str], profile: str) -> None:
        """Stream a recorded camera clip. Browsers need HTTP Range support to play (and loop) large video."""
        path = self.feed_dir / FEED_FILES[profile] if self.feed_dir else None
        if path is None or not path.is_file():
            return await self.json(writer, 404, {"error": "NOT_FOUND", "message": f"{FEED_FILES[profile]} not found"})
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        if m := re.fullmatch(r"bytes=(\d*)-(\d*)", headers.get("range", "").strip()):
            first, last = m.groups()
            if first:
                start, end = int(first), min(int(last), size - 1) if last else size - 1
            elif last:  # suffix range: the final N bytes (players read the mp4 index from the tail)
                start = max(0, size - int(last))
            if not (first or last) or start > end or start >= size:
                return await self.send(writer, 416, b"", "text/plain", {"Content-Range": f"bytes */{size}"})
            status = 206
        length = end - start + 1
        extra = {"Accept-Ranges": "bytes", "Cache-Control": "private, max-age=60"}
        if status == 206:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        writer.write(self.head(status, "video/mp4", length, extra))
        with path.open("rb") as f:
            f.seek(start)
            while length > 0:
                chunk = await asyncio.to_thread(f.read, min(FEED_CHUNK, length))
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()    # raises ConnectionError if the player seeked away or closed the tab
                length -= len(chunk)

    async def json(self, writer, status: int, data: Any) -> None:
        await self.send(writer, status, json.dumps(data, default=str).encode(), "application/json")

    async def static(self, writer, name: str, headers: dict[str, str]) -> None:
        body = (STATIC / name).read_bytes()
        extra = {"Cache-Control": "no-cache"}
        if name == "index.html":
            extra["Content-Security-Policy"] = self.csp
        if "gzip" in headers.get("accept-encoding", "") and len(body) > 1024:
            body = gzip.compress(body, 6)
            extra |= {"Content-Encoding": "gzip", "Vary": "Accept-Encoding"}
        await self.send(writer, 200, body, STATIC_TYPES[name], extra)

    async def route(self, method: str, target: str, headers: dict[str, str], body: bytes, writer) -> None:
        path, _, query = target.partition("?")
        if method == "GET":
            if path == "/":
                return await self.static(writer, "index.html", headers)
            if path.lstrip("/") in STATIC_TYPES:
                return await self.static(writer, path.lstrip("/"), headers)
            if m := re.fullmatch(r"/feed/(main|sub)\.mp4", path):
                return await self.feed(writer, headers, m.group(1).upper())
            if path == "/api/state":
                return await self.json(writer, 200, self.fleet.snapshot())
            if path == "/api/wall/cameras":
                return await self.wall(writer, "/api/v1/grid/cameras")
            if m := re.fullmatch(r"/api/wall/cameras/([A-Za-z0-9_-]{1,64})/stream-sources", path):
                profile = "main" if re.search(r"(?:^|&)profile=main(?:&|$)", query) else "web"
                return await self.wall(writer, f"/federation/cameras/{m.group(1)}/stream-sources?profile={profile}", timeout=45)
            if path == "/api/stream":
                return await self.sse(writer)
            if path == "/api/trace":
                m = re.search(r"(?:^|&)limit=(\d+)", query)
                return await self.json(writer, 200, self.fleet.trace(min(500, int(m.group(1)) if m else 150)))
            if m := re.fullmatch(r"/api/devices/([A-Z0-9-]+)/snapshot", path):
                return await self.snapshot(writer, m.group(1))
        elif method == "POST" and (m := re.fullmatch(r"/api/devices/([A-Z0-9-]+)/([a-z]+)", path)):
            # JSON-only POSTs force a CORS preflight, so other sites cannot drive the console
            if not headers.get("content-type", "").startswith("application/json"):
                return await self.json(writer, 415, {"error": "UNSUPPORTED_MEDIA_TYPE", "message": "send JSON"})
            try:
                payload = json.loads(body or b"{}")
                result = await self.fleet.command(m.group(1), m.group(2), payload)
            except DeviceError as e:
                return await self.json(writer, 502, {"error": e.code.value, "message": str(e), "retryable": e.retryable})
            except KeyError as e:
                return await self.json(writer, 400, {"error": "BAD_REQUEST", "message": f"missing field {e}"})
            except LookupError as e:
                return await self.json(writer, 404, {"error": "NOT_FOUND", "message": str(e).strip("'")})
            except (ValueError, TypeError, json.JSONDecodeError) as e:
                return await self.json(writer, 400, {"error": "BAD_REQUEST", "message": str(e)})
            return await self.json(writer, 200, result)
        await self.json(writer, 404, {"error": "NOT_FOUND", "message": f"{method} {path}"})

    async def wall(self, writer, upstream: str, timeout: float = 10) -> None:
        """
        The camera wall's data comes from Module 3, never from the grid: the browser asks this console, the console
        asks M3 (fixed paths only, so this is not an open proxy) and passes the credential-free answer through.
        """
        if self.fleet.video_mode != "real":
            return await self.json(writer, 404, {"error": "NOT_FOUND", "message": "the live wall is off in mock video mode"})

        def fetch() -> tuple[int, bytes]:
            try:
                with urllib.request.urlopen(f"{self.m3_url}{upstream}", timeout=timeout) as r:
                    return r.status, r.read(2_000_000)
            except urllib.error.HTTPError as e:
                return e.code, e.read(200_000)

        try:
            status, body = await asyncio.to_thread(fetch)
        except (urllib.error.URLError, TimeoutError, OSError):
            return await self.json(writer, 502, {"error": "M3_UNREACHABLE", "message":
                                                 f"Module 3 is not reachable at {self.m3_url}. Start it: uvicorn app.main:app --port 8000"})
        if status >= 400:
            try:
                detail = json.loads(body).get("detail", {})
                msg = detail.get("message") if isinstance(detail, dict) else str(detail)
            except ValueError:
                msg = None
            return await self.json(writer, 502 if status >= 500 else status,
                                   {"error": "M3_ERROR", "message": msg or f"Module 3 answered HTTP {status}"})
        await self.send(writer, 200, body, "application/json")

    async def snapshot(self, writer, device: str) -> None:
        d = self.fleet.drivers.get(device)
        if d is None or device not in self.fleet.cam:
            return await self.json(writer, 404, {"error": "NOT_FOUND", "message": device})
        try:
            snap = await d.get_snapshot()
        except DeviceError as e:
            return await self.json(writer, 502, {"error": e.code.value, "message": str(e)})
        await self.send(writer, 200, snap.data, snap.content_type)

    async def sse(self, writer) -> None:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        self.fleet.subscribers.add(q)
        head = ("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n"
                "Connection: keep-alive\r\nX-Accel-Buffering: no\r\n\r\n")
        writer.write(head.encode() + b"retry: 2000\n\n")
        writer.write(f"event: state\ndata: {json.dumps(self.fleet.snapshot(), default=str)}\n\n".encode())
        try:
            await writer.drain()
            while q in self.fleet.subscribers:
                try:
                    msg = await asyncio.wait_for(q.get(), 15)
                except TimeoutError:
                    msg = ": keep-alive\n\n"
                writer.write(msg.encode())
                await writer.drain()
        except ConnectionError:
            pass
        finally:
            self.fleet.subscribers.discard(q)


async def main() -> None:
    ap = argparse.ArgumentParser(description="Sentinel M3 hardware console (web UI over sentinel-hwsim)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=4.0, help="simulation time compression (default 4x)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--feed-dir", type=Path, default=find_feed_dir(),
                    help="folder with camera_main.mp4 and camera_sub.mp4 (default: nearest camera_feed/ above this file)")
    env_mode = os.getenv("M3_DRIVER_MODE", "real")
    ap.add_argument("--video-mode", choices=VIDEO_MODES, default=env_mode if env_mode in VIDEO_MODES else "real",
                    help="real (default): live grid cameras via Module 3's relay. mock: looping sample clips, offline. Env: M3_DRIVER_MODE")
    ap.add_argument("--m3-url", default=os.getenv("M3_API_URL", "http://127.0.0.1:8000"),
                    help="Module 3 API (camera inventory and stream broker)")
    ap.add_argument("--relay-origin", action="append", default=None, metavar="URL",
                    help="origin of the stream relay the browser may connect to (repeatable; default from RELAY_PUBLIC_HOST/RELAY_*_PORT)")
    ap.add_argument("--open", action="store_true", help="open the console in the default browser")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("sentinel_hwsim").setLevel(logging.WARNING)  # wire faults only; the UI shows the full trace

    fleet = Fleet(speed=args.speed, seed=args.seed, feed_dir=args.feed_dir, video_mode=args.video_mode)
    server = Server(fleet, args.host, args.port, args.feed_dir, m3_url=args.m3_url, relay_origins=args.relay_origin)
    if args.video_mode == "real":
        log.info("video: LIVE grid cameras through Module 3 (%s) and the relay (%s); no sample clips are served",
                 args.m3_url, ", ".join(server.relay_origins))
    elif fleet.feeds:
        log.info("camera feeds: %s from %s", ", ".join(f"{p} ({FEED_FILES[p]})" for p in fleet.feeds), args.feed_dir)
    else:
        log.warning("no camera feed clips found (%s); the Cameras page will show a placeholder", args.feed_dir)
    tasks = fleet.start()
    srv = await asyncio.start_server(server.handle, args.host, args.port)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    log.info("Sentinel console on %s  (speed %gx, seed %d)  Ctrl+C to stop", url, args.speed, args.seed)
    if args.open:
        webbrowser.open(url)
    async with srv:
        try:
            await srv.serve_forever()
        finally:
            for t in tasks:
                t.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
