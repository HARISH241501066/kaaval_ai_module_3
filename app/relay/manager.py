"""
Relay manager (C3): decides which grid protocol each camera is pulled over, and keeps a running MediaMTX in
step with the registry.

Source failover, per camera, tried in priority order:   RTSP (TCP)  ->  WHEP  ->  (HLS, only if the grid
allows it server-side; today it does not: it answers 403 "browser required" to non-browser clients).
MediaMTX cannot fail over between protocols by itself, so `ensure()` does it: point the camera's `main`
path at a candidate, open it through the relay, and keep it if frames arrive. The first success is recorded
in stream_sources as the active protocol. RTSP is always tried first, so a camera returns to RTSP by itself
once RTSP works again.

`ensure()` only acts when the camera is not already flowing, so cameras that are being watched are never
probed: the grid never sees an extra connection for a camera that is working.
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.sentinel_grid import SentinelGridAdapter
from app.core.config import Settings, get_settings
from app.grid import urls
from app.live.m3 import SYSTEM_ID
from app.models.grid import CameraMappingModel, StreamSourceModel
from app.relay import config as cfg
from contract import ConnectionFailed, DeviceError, MappingError

log = logging.getLogger("sentinel.m3.relay")
PULLABLE = ("RTSP", "WHEP")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class RelayManager:
    def __init__(self, settings: Settings | None = None, adapter: SentinelGridAdapter | None = None,
                 http: httpx.Client | None = None):
        self._settings = settings
        self.adapter = adapter or SentinelGridAdapter(settings)
        self._http = http
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    @property
    def s(self) -> Settings:
        return self._settings or get_settings()

    def _lock(self, cam_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(cam_id, threading.Lock())

    # ------------------------------------------------------------- MediaMTX API
    def _client(self) -> httpx.Client:
        if self._http is None:
            self._http = httpx.Client(base_url=self.s.relay_api_url, timeout=5.0)
        return self._http

    def _api(self, method: str, path: str, **kw: Any) -> httpx.Response:
        try:
            return self._client().request(method, path, **kw)
        except httpx.HTTPError as e:
            raise ConnectionFailed(f"the relay API is not reachable ({e.__class__.__name__}); start it with: python -m app.relay run") from None

    def api_up(self) -> bool:
        try:
            return self._client().get("/v3/paths/list", params={"itemsPerPage": 1}).status_code == 200
        except httpx.HTTPError:
            return False

    def path_info(self, name: str) -> dict[str, Any] | None:
        try:
            r = self._client().get(f"/v3/paths/get/{name}")
        except httpx.HTTPError:
            return None
        return r.json() if r.status_code == 200 else None

    def path_ready(self, name: str) -> bool:
        return bool((self.path_info(name) or {}).get("ready"))

    @staticmethod
    def protocol_of(info: dict[str, Any]) -> str | None:
        """MediaMTX's own name for what feeds a path -> our protocol name."""
        return {"rtspSource": "RTSP", "rtspsSource": "RTSP", "webRTCSource": "WHEP", "hlsSource": "HLS"}.get((info.get("source") or {}).get("type"))

    def _put_path(self, name: str, conf: dict[str, Any]) -> None:
        """Create or replace one path on the running relay."""
        r = self._api("POST", f"/v3/config/paths/replace/{name}", json=conf)
        if r.status_code == 404:
            r = self._api("POST", f"/v3/config/paths/add/{name}", json=conf)
        if r.status_code >= 300:
            raise ConnectionFailed(f"the relay refused the path {name} (HTTP {r.status_code})")

    # ------------------------------------------------------ registry -> config
    def _active_protocols(self, db: Session) -> dict[str, str]:
        rows = db.execute(select(StreamSourceModel).where(StreamSourceModel.active.is_(True))).scalars()
        return {r.camera_id: r.protocol for r in rows}

    def _plans(self, db: Session) -> list[tuple[cfg.CameraPlan, str]]:
        active = self._active_protocols(db)
        out = []
        for m in db.execute(select(CameraMappingModel).where(CameraMappingModel.system_id == SYSTEM_ID,
                                                             CameraMappingModel.status == "ACTIVE")
                            .order_by(CameraMappingModel.external_id)).scalars():
            out.append((cfg.CameraPlan(m.external_id, m.codec, active.get(m.camera_id, "RTSP")), m.camera_id))
        return out

    def desired_paths(self, db: Session) -> dict[str, dict[str, Any]]:
        ffmpeg = cfg.find_ffmpeg(self.s)
        paths: dict[str, dict[str, Any]] = {}
        for plan, _ in self._plans(db):
            paths.update(cfg.path_settings(plan, self.s, self.adapter.source_url(plan.protocol, plan.cam_id), ffmpeg))
        return paths

    def write_config(self, db: Session):
        return cfg.write_config(self.s, cfg.render(self.s, self.desired_paths(db)))

    def sync_paths(self, db: Session) -> dict[str, int]:
        """Bring a RUNNING relay in line with the registry (new cameras added, inactive ones removed) and rewrite the file."""
        desired = self.desired_paths(db)
        cfg.write_config(self.s, cfg.render(self.s, desired))
        if not self.api_up():
            return {"added": 0, "removed": 0, "relay": 0}
        r = self._api("GET", "/v3/config/paths/list", params={"itemsPerPage": 1000})
        current = {i["name"] for i in r.json().get("items", []) if i["name"].startswith("cam/")}
        added = removed = 0
        for name in desired.keys() - current:
            self._put_path(name, desired[name])
            added += 1
        for name in current - desired.keys():
            self._api("DELETE", f"/v3/config/paths/delete/{name}")
            removed += 1
        return {"added": added, "removed": removed, "relay": 1}

    # ------------------------------------------------------------ source choice
    def ensure(self, db: Session, cam_id: str) -> dict[str, Any]:
        """
        Make sure the camera can be watched: if it is already flowing, do nothing; otherwise try the protocols
        in priority order until one delivers frames. Returns the credential-free status (see `status`).
        """
        mapping = self._mapping(db, cam_id)
        with self._lock(cam_id):
            info = self.path_info(cfg.path_name(cam_id, cfg.MAIN))
            if info and info.get("ready"):                       # already flowing: leave it alone, but learn what feeds it
                if (proto := self.protocol_of(info)) and self.status(db, cam_id)["active_source"] != proto:
                    self._record(db, cam_id, proto, PULLABLE.index(proto) + 1 if proto in PULLABLE else 3, True, None, active=True)
                    self._record_hls(db, cam_id)
                return self.status(db, cam_id)
            if not self.api_up():
                return {**self.status(db, cam_id), "state": "RELAY_DOWN"}
            candidates = {c.protocol: c for c in self.adapter.get_stream_uri(cam_id)}
            active_ok = False
            for protocol in PULLABLE:
                cand = candidates.get(protocol)
                if cand is None:
                    continue
                ok, detail = self._try(db, mapping, protocol)
                self._record(db, cam_id, protocol, cand.priority, ok, detail, active=ok and not active_ok)
                if ok and not active_ok:
                    active_ok = True
                    break
            self._record_hls(db, cam_id)
            if not active_ok:
                log.warning("%s: no protocol delivered frames", cam_id)
            return self.status(db, cam_id)

    def _try(self, db: Session, mapping: CameraMappingModel, protocol: str) -> tuple[bool, str | None]:
        """
        Is this protocol usable right now? Asked of the GRID directly, with no retries (failover must be quick). The
        relay's own path is deliberately NOT opened here: the grid replays its buffered GOP to whoever connects first,
        and that first frame must go to the viewer, not to a probe. Cameras with sparse keyframes (some send one every
        20+ s) would otherwise leave the viewer waiting for the next natural keyframe.
        """
        cam_id = mapping.external_id
        try:
            res = self.adapter.probe(cam_id, retries=0) if protocol == "RTSP" else self.adapter.probe_whep(cam_id, retries=0)
            plan = cfg.CameraPlan(cam_id, res.codec or mapping.codec, protocol)
            ffmpeg = cfg.find_ffmpeg(self.s)
            for name, conf in cfg.path_settings(plan, self.s, self.adapter.source_url(protocol, cam_id), ffmpeg).items():
                self._put_path(name, conf)
        except DeviceError as e:
            log.info("%s: %s did not work (%s)", cam_id, protocol, e.code.value)
            return False, f"{e.code.value}: {e}"[:200]
        if res.codec and res.codec != mapping.codec:              # the real stream is the authority on the codec
            mapping.codec, mapping.width, mapping.height, mapping.fps = res.codec, res.width, res.height, res.fps
            db.commit()
        log.info("%s: %s works; the relay will pull over it when the first viewer joins", cam_id, protocol)
        return True, None

    # ---------------------------------------------------------------- records
    def _record(self, db: Session, cam_id: str, protocol: str, priority: int, ok: bool, detail: str | None,
                active: bool) -> None:
        camera_id = self._mapping(db, cam_id).camera_id
        row = db.execute(select(StreamSourceModel).where(StreamSourceModel.camera_id == camera_id,
                                                         StreamSourceModel.protocol == protocol)).scalar_one_or_none()
        if row is None:
            row = StreamSourceModel(camera_id=camera_id, protocol=protocol, priority=priority)
            db.add(row)
            db.flush()                       # SessionLocal has autoflush off: make the new row visible to the query below
        row.available, row.detail, row.checked_at = ok, detail, _now()
        if active:
            for other in db.execute(select(StreamSourceModel).where(StreamSourceModel.camera_id == camera_id)).scalars():
                other.active = other.protocol == protocol
        elif not ok:
            row.active = False
        db.commit()

    def _record_hls(self, db: Session, cam_id: str) -> None:
        if self.adapter.hls_server_side:
            return
        camera_id = self._mapping(db, cam_id).camera_id
        row = db.execute(select(StreamSourceModel).where(StreamSourceModel.camera_id == camera_id,
                                                         StreamSourceModel.protocol == "HLS")).scalar_one_or_none()
        if row is None:
            db.add(StreamSourceModel(camera_id=camera_id, protocol="HLS", priority=3, available=False, active=False,
                                     detail=f"not usable server-side: {self.adapter.hls_note}"[:200], checked_at=_now()))
            db.commit()

    def _mapping(self, db: Session, cam_id: str) -> CameraMappingModel:
        m = db.execute(select(CameraMappingModel).where(CameraMappingModel.system_id == SYSTEM_ID,
                                                        CameraMappingModel.external_id == cam_id,
                                                        CameraMappingModel.status == "ACTIVE")).scalar_one_or_none()
        if m is None:
            raise MappingError(f"'{cam_id}' is not an active Sentinel Grid camera", device_id=cam_id)
        return m

    # ------------------------------------------------------------------ output
    def status(self, db: Session, cam_id: str) -> dict[str, Any]:
        """Credential-free view of one camera's sources (what the broker and the wall show)."""
        m = self._mapping(db, cam_id)
        rows = db.execute(select(StreamSourceModel).where(StreamSourceModel.camera_id == m.camera_id)
                          .order_by(StreamSourceModel.priority)).scalars().all()
        active = next((r.protocol for r in rows if r.active), None)
        return {"active_source": active, "state": "AVAILABLE" if active else "UNKNOWN",
                "sources": [{"protocol": r.protocol, "priority": r.priority, "available": bool(r.available),
                             "active": bool(r.active), "detail": r.detail} for r in rows]}

    def relay_urls(self, cam_id: str, profile: str) -> dict[str, str]:
        """Consumer-facing relay URLs: our host, our paths, no credentials."""
        s, name = self.s, cfg.path_name(cam_id, profile)
        return {"whep": f"http://{s.relay_public_host}:{s.relay_webrtc_port}/{name}/whep",
                "hls": f"http://{s.relay_public_host}:{s.relay_hls_port}/{name}/index.m3u8",
                "rtsp": f"rtsp://{s.relay_public_host}:{s.relay_rtsp_port}/{name}"}

    def describe(self, db: Session, cam_id: str, profile: str) -> dict[str, Any]:
        m = self._mapping(db, cam_id)
        codec = "h264" if profile == cfg.WEB else m.codec
        return {"camera_id": cam_id, "canonical_id": m.camera_id, "profile": profile, "codec": codec,
                "source_codec": m.codec, "transcoded": profile == cfg.WEB and urls.needs_web_transcode(m.codec),
                "resolution": f"{m.width}x{m.height}" if m.width and m.height else None,
                "urls": self.relay_urls(cam_id, profile), **self.status(db, cam_id)}


_manager: RelayManager | None = None


def get_relay_manager() -> RelayManager:
    global _manager
    if _manager is None:
        _manager = RelayManager()
    return _manager
