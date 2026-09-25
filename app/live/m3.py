"""
Talks to the M3 API on behalf of the live worker:

* register the grid as an upstream system, the detector as a processing node, each watched
  camera and its camera->node assignment (idempotent: safe to run on every start), and
* send detected events as signed batches to the C4 ingestion endpoint, exactly as a Kaaval node
  does (HMAC-SHA256 over "<timestamp>." + body).

Nothing here ever sees the grid credentials: cameras are registered with credential-free URLs.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import random
import time
from typing import Any

import httpx

from app.live.grid import GridCamera

log = logging.getLogger("sentinel.m3.client")

TIMEOUT_S = 5.0
RETRIES = 2
SYSTEM_ID = "sentinel-grid"


class M3Error(Exception):
    pass


def canonical_camera_id(grid_id: str) -> str:
    return f"GRID-{grid_id.upper()}"


def sign(secret: str, timestamp: str, body: bytes) -> str:
    """Same scheme as sentinel_hwsim.push.sign() and pipeline.verify_kaaval_hmac()."""
    return hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()


class M3Client:
    def __init__(self, base_url: str, secret: str, node_id: str, http: httpx.Client | None = None):
        self.node_id, self._secret = node_id, secret
        self._http = http or httpx.Client(base_url=base_url.rstrip("/"), timeout=TIMEOUT_S)

    # ------------------------------------------------------------ registration
    def register(self, cameras: list[GridCamera], host: str, rtsp_port: int, web_base: str) -> None:
        self._ensure("/api/v1/systems", SYSTEM_ID, {
            "id": SYSTEM_ID, "name": "Sentinel Camera Grid", "vendor": "GENERIC_RTSP", "protocol": "RTSP",
            "status": "ACTIVE",       # only registered after the connection test passed
            "connection_params": {"adapter_key": "SENTINEL_GRID", "zone": "default", "host": host,
                                  "rtsp_port": rtsp_port, "web_base": web_base,
                                  "credential_ref": "env:SENTINEL_GRID_EMAIL,SENTINEL_GRID_PASSWORD"}})
        self._ensure("/api/v1/nodes", self.node_id, {
            "id": self.node_id, "name": "Live video detector", "node_type": "CLOUD", "system_id": SYSTEM_ID,
            "status": "ONLINE"})
        known = {c["id"] for c in self._get("/api/v1/cameras", params={"system_id": SYSTEM_ID})}
        assigned = {(a["camera_id"], a["node_id"]) for a in self._get("/api/v1/assignments")}
        for cam in cameras:
            cid = canonical_camera_id(cam.id)
            if cid not in known:
                self._post("/api/v1/cameras", {
                    "id": cid, "external_id": cam.id, "name": cam.name, "system_id": SYSTEM_ID,
                    "rtsp_url": f"rtsp://{host}:{rtsp_port}/stream/{cam.id}",       # no credentials, ever
                    "status": "ONLINE",
                    **({"resolution": f"{cam.width}x{cam.height}"} if cam.width and cam.height else {}),
                    **({"fps": cam.fps} if cam.fps else {})})
            if (cid, self.node_id) not in assigned:
                self._post("/api/v1/assignments", {"camera_id": cid, "node_id": self.node_id,
                                                   "pipeline_profile": "TRAFFIC_ANALYTICS"})
        log.info("registered %d camera(s) with M3 under system '%s'", len(cameras), SYSTEM_ID)

    # ---------------------------------------------------------------- ingestion
    def send(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        body = json.dumps({"node": self.node_id, "batch": items}, separators=(",", ":")).encode()
        last: str = ""
        for attempt in range(RETRIES + 1):
            ts = str(int(time.time()))
            headers = {"Content-Type": "application/json", "X-Kaaval-Node": self.node_id,
                       "X-Kaaval-Timestamp": ts, "X-Kaaval-Signature": sign(self._secret, ts, body)}
            try:
                r = self._http.post(f"/federation/ingest/{self.node_id}", content=body, headers=headers)
                if r.status_code == 401:
                    raise M3Error("M3 rejected the signature: KAAVAL_HMAC_SECRET differs between .env files or the clocks disagree")
                if r.status_code < 500:
                    if r.status_code >= 400:
                        raise M3Error(f"M3 refused the batch (HTTP {r.status_code})")
                    return r.json()
                last = f"HTTP {r.status_code}"
            except httpx.HTTPError as e:
                last = e.__class__.__name__
            if attempt < RETRIES:
                time.sleep(min(30.0, 0.5 * 2 ** attempt) * random.uniform(0.8, 1.2))
        raise M3Error(f"M3 unreachable ({last})")

    # ------------------------------------------------------------------ helpers
    def _get(self, path: str, **kw: Any) -> list[dict[str, Any]]:
        r = self._call("GET", path, **kw)
        return r.json()

    def _post(self, path: str, body: dict[str, Any]) -> None:
        r = self._call("POST", path, json=body)
        if r.status_code not in (200, 201, 400):     # 400 = already exists (race with another start)
            raise M3Error(f"{path} failed (HTTP {r.status_code})")

    def _ensure(self, path: str, item_id: str, body: dict[str, Any]) -> None:
        if item_id not in {x["id"] for x in self._get(path)}:
            self._post(path, body)

    def _call(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as e:
            raise M3Error(f"cannot reach the M3 API at {self._http.base_url} ({e.__class__.__name__}). "
                          "Start it first: uvicorn app.main:app --port 8000") from None
        if r.status_code >= 500:
            raise M3Error(f"M3 API error on {path} (HTTP {r.status_code})")
        return r
