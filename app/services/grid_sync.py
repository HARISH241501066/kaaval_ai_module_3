"""
C7 (integration system) and C2 (inventory sync) for the Sentinel Camera Grid.

C7  register_system(): one system, id "sentinel-grid", adapter_key SENTINEL_GRID, holding the three
    endpoint templates (RTSP, HLS, WHEP) and no credentials. It becomes ACTIVE only after the connection
    test (catalogue fetch + one RTSP probe) passes.
C2  sync_inventory(): every camera in the grid's catalogue becomes an M3 camera (canonical id GRID-CAM04,
    external id cam04) with a camera_mappings row carrying the probed codec and resolution. A camera
    missing from 3 consecutive runs turns mapping-INACTIVE. Nothing is ever deleted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.sentinel_grid import SentinelGridAdapter
from app.grid import urls
from app.live.grid import GridCamera
from app.live.m3 import SYSTEM_ID, canonical_camera_id
from app.models.grid import CameraMappingModel
from app.models.system import CameraModel, SystemModel
from contract import DeviceError

log = logging.getLogger("sentinel.m3.grid.sync")

MISSED_RUNS_BEFORE_INACTIVE = 3
REPROBE_AFTER = timedelta(hours=24)


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SyncReport:
    catalogue: int = 0
    created: int = 0
    updated: int = 0
    reactivated: int = 0
    newly_inactive: list[str] = field(default_factory=list)
    probed: int = 0
    probe_failed: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def endpoint_templates(adapter: SentinelGridAdapter) -> dict[str, str]:
    s = adapter.settings
    return {"rtsp": urls.rtsp_public(s.sentinel_grid_host, s.sentinel_grid_rtsp_port, "{id}"),
            "hls": urls.hls_public(s.sentinel_grid_web_base, "{id}"),
            "whep": urls.whep_public(s.sentinel_grid_host, s.sentinel_grid_whep_port, "{id}")}


# ------------------------------------------------------------------------- C7
def register_system(db: Session, adapter: SentinelGridAdapter) -> dict:
    """Idempotent. Returns {"status": "ACTIVE"|"INACTIVE", ...}; never raises for a failed grid."""
    system = db.get(SystemModel, SYSTEM_ID)
    params = {"adapter_key": adapter.name, "zone": "default", "endpoints": endpoint_templates(adapter),
              "credential_ref": "env:SENTINEL_GRID_EMAIL,SENTINEL_GRID_PASSWORD"}
    if system is None:
        system = SystemModel(id=SYSTEM_ID, name="Sentinel Camera Grid", vendor="GENERIC_RTSP", protocol="RTSP",
                             connection_params=params, status="INACTIVE")
        db.add(system)
        db.commit()
    try:
        test = adapter.connection_test()
    except DeviceError as e:
        system.status = "INACTIVE"
        system.connection_params = {**(system.connection_params or {}), **params, "last_error": f"{e.code.value}: {e}"}
        db.commit()
        log.warning("grid connection test failed, system stays INACTIVE: %s", e)
        return {"status": "INACTIVE", "error_code": e.code.value, "error": str(e)}
    system.connection_params = {**params, "hls_server_side": test["hls_server_side"], "hls_note": test["hls_note"]}
    system.status = "ACTIVE"
    db.commit()
    return {"status": "ACTIVE", **test}


# ------------------------------------------------------------------------- C2
def _upsert_camera(db: Session, cam: GridCamera, adapter: SentinelGridAdapter, report: SyncReport) -> None:
    cid = canonical_camera_id(cam.id)
    camera = db.get(CameraModel, cid)
    rtsp = adapter.grid.rtsp_url_public(cam.id)
    if camera is None:
        camera = CameraModel(id=cid, external_id=cam.id, name=cam.name, system_id=SYSTEM_ID, rtsp_url=rtsp,
                             status="ONLINE")
        db.add(camera)
        report.created += 1
    elif camera.name != cam.name or camera.rtsp_url != rtsp:
        camera.name, camera.rtsp_url = cam.name, rtsp
        report.updated += 1
    mapping = db.execute(select(CameraMappingModel).where(
        CameraMappingModel.system_id == SYSTEM_ID, CameraMappingModel.external_id == cam.id)).scalar_one_or_none()
    if mapping is None:
        db.add(CameraMappingModel(system_id=SYSTEM_ID, external_id=cam.id, camera_id=cid, status="ACTIVE",
                                  missed_runs=0, last_seen_at=_now(), codec=urls.norm_codec(cam.codec),
                                  width=cam.width, height=cam.height, fps=cam.fps))
        return
    if mapping.status == "INACTIVE":
        mapping.status = "ACTIVE"
        camera.status = "ONLINE"
        report.reactivated += 1
    mapping.missed_runs, mapping.last_seen_at = 0, _now()


def probe_pending(db: Session, adapter: SentinelGridAdapter, limit: int, report: SyncReport) -> None:
    """Probe up to `limit` cameras that have no codec yet or a stale probe, one at a time (each probe is a grid connection)."""
    if limit <= 0:
        return
    stale = _now() - REPROBE_AFTER
    todo = [m for m in db.execute(select(CameraMappingModel).where(CameraMappingModel.status == "ACTIVE")
                                  .order_by(CameraMappingModel.external_id)).scalars()
            if m.probed_at is None or m.probed_at.replace(tzinfo=timezone.utc) < stale]
    for m in todo[:limit]:
        camera = db.get(CameraModel, m.camera_id)
        try:
            res = adapter.probe(m.external_id)
        except DeviceError as e:
            m.probe_error, m.probed_at = f"{e.code.value}: {e}"[:200], _now()
            camera.status = "OFFLINE"
            report.probe_failed.append(m.external_id)
        else:
            m.codec, m.width, m.height, m.fps = res.codec, res.width, res.height, res.fps
            m.probe_error, m.probed_at = None, _now()
            camera.status = "ONLINE"
            if res.resolution:
                camera.resolution = res.resolution
            if res.fps:
                camera.fps = res.fps
            report.probed += 1
        db.commit()


def sync_inventory(db: Session, adapter: SentinelGridAdapter, probe_limit: int = 0) -> SyncReport:
    report = SyncReport()
    if db.get(SystemModel, SYSTEM_ID) is None:
        register_system(db, adapter)
    catalogue = adapter.inventory()                       # raises DeviceError if the grid is unreachable: nothing is marked missing
    report.catalogue = len(catalogue)
    for cam in catalogue:
        _upsert_camera(db, cam, adapter, report)
    seen = {c.id for c in catalogue}
    for m in db.execute(select(CameraMappingModel).where(CameraMappingModel.system_id == SYSTEM_ID)).scalars():
        if m.external_id in seen or m.status == "INACTIVE":
            continue
        m.missed_runs += 1
        if m.missed_runs >= MISSED_RUNS_BEFORE_INACTIVE:
            m.status = "INACTIVE"                          # mapping-inactive, the camera row stays
            db.get(CameraModel, m.camera_id).status = "OFFLINE"
            report.newly_inactive.append(m.external_id)
    db.commit()
    probe_pending(db, adapter, probe_limit, report)
    log.info("grid sync: %s", report.as_dict())
    return report
