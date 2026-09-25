"""
Sentinel Camera Grid endpoints (C7 connect, C2 sync, C3 stream broker).

Everything returned here is credential-free: relay URLs, codecs, statuses. The browser and Module 2 talk to
these endpoints and to our relay, and never to the grid itself.
"""
from __future__ import annotations

import logging
from typing import List, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.adapters.registry import get_adapter_registry
from app.adapters.sentinel_grid import SentinelGridAdapter
from app.core.config import get_settings
from app.db.database import get_db
from app.db.repositories import CameraRepository
from app.live.m3 import SYSTEM_ID
from app.models.grid import CameraMappingModel, StreamSourceModel
from app.models.system import CameraModel
from app.relay.manager import RelayManager, get_relay_manager
from app.schemas.grid import GridCameraRead, GridSyncRead, StreamSourcesRead
from app.services import grid_sync
from contract import AuthFailed, CapabilityNotSupported, DeviceError, DeviceTimeout, MappingError

log = logging.getLogger("sentinel.m3.api.grid")

router = APIRouter(prefix="/api/v1/grid", tags=["Sentinel Grid (C1, C2, C7)"])
federation_router = APIRouter(prefix="/federation", tags=["Stream Broker (C3)"])


def _adapter() -> SentinelGridAdapter:
    return get_relay_manager().adapter


def _relay() -> RelayManager:
    return get_relay_manager()


def _http_error(e: DeviceError) -> HTTPException:
    status = 504 if isinstance(e, DeviceTimeout) else 404 if isinstance(e, MappingError) else 422 if isinstance(e, CapabilityNotSupported) else 502
    return HTTPException(status_code=status, detail={"error": e.code.value, "message": str(e)})


@router.post("/connect", summary="C7: register the grid system (ACTIVE only if the connection test passes)")
def connect(db: Session = Depends(get_db)):
    return grid_sync.register_system(db, _adapter())


@router.post("/sync", response_model=GridSyncRead, summary="C2: sync the grid catalogue into the camera registry")
def sync(probe_limit: int = Query(0, ge=0, le=60, description="Probe up to N cameras that have no codec yet (one at a time)."),
         db: Session = Depends(get_db)):
    try:
        report = grid_sync.sync_inventory(db, _adapter(), probe_limit=probe_limit)
    except DeviceError as e:
        raise _http_error(e)
    relay = {}
    if get_settings().relay_enabled:
        try:
            relay = _relay().sync_paths(db)
        except DeviceError as e:
            log.warning("relay path sync skipped: %s", e)
    return GridSyncRead(**report.as_dict(), relay=relay)


@router.get("/cameras", response_model=List[GridCameraRead], summary="The camera wall's inventory (from Module 3, never from the grid)")
def list_grid_cameras(include_inactive: bool = False, db: Session = Depends(get_db)):
    stmt = select(CameraMappingModel).where(CameraMappingModel.system_id == SYSTEM_ID).order_by(CameraMappingModel.external_id)
    if not include_inactive:
        stmt = stmt.where(CameraMappingModel.status == "ACTIVE")
    active = {r.camera_id: r.protocol for r in db.execute(select(StreamSourceModel).where(StreamSourceModel.active.is_(True))).scalars()}
    out = []
    for m in db.execute(stmt).scalars():
        cam = db.get(CameraModel, m.camera_id)
        out.append(GridCameraRead(
            id=m.external_id, canonical_id=m.camera_id, name=cam.name, codec=m.codec,
            resolution=f"{m.width}x{m.height}" if m.width and m.height else None, fps=m.fps, status=cam.status,
            mapping_status=m.status, probed=m.probed_at is not None, active_source=active.get(m.camera_id),
            probe_error=m.probe_error))
    return out


@router.post("/relay/reconcile", summary="Bring a running relay in line with the registry")
def reconcile_relay(db: Session = Depends(get_db)):
    try:
        return _relay().sync_paths(db)
    except DeviceError as e:
        raise _http_error(e)


@federation_router.get("/cameras/{camera_id}/stream-sources", response_model=StreamSourcesRead,
                       summary="C3 broker: credential-free relay URLs for a camera")
def stream_sources(camera_id: str, profile: Literal["main", "web"] = "web", db: Session = Depends(get_db)):
    """
    `web` is always H.264 (browsers); `main` is the camera's own codec (backend consumers). Returns only URLs on
    OUR relay plus the codec, resolution and the grid protocol currently in use: no upstream address, no credentials.
    Asking for a camera that is not flowing makes the relay open it, falling back RTSP -> WHEP if needed.
    """
    cam = CameraRepository(db).resolve_camera(camera_id)
    if cam is None or cam.system_id != SYSTEM_ID:
        raise HTTPException(status_code=404, detail={"error": "MAPPING_ERROR", "message": f"'{camera_id}' is not a Sentinel Grid camera"})
    grid_id = cam.external_id or camera_id
    relay = _relay()
    try:
        if get_settings().relay_enabled:
            relay.ensure(db, grid_id)
        info = relay.describe(db, grid_id, profile)
    except DeviceError as e:
        raise _http_error(e)
    if info["state"] == "UNKNOWN" and any(not s["available"] for s in info["sources"]):
        info["state"] = "UNAVAILABLE"                 # every protocol we could try failed
    return info


__all__ = ["router", "federation_router", "get_adapter_registry"]
