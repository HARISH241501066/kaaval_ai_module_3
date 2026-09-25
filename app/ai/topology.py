"""
Demo topology for locally processed road videos (prototype, idempotent).

Reuses the existing M3 models/repositories (no schema changes):

    SENTINEL-LOCAL-VIDEO (system)  : the simulated CCTV sources (.mp4 feeds)
    RDK-X5-001 (ProcessingNode)    : the Kaaval AI node provenance points at
    CAM-001..N (Camera)            : one per entry in AI_VIDEO_SOURCES
    CameraAssignment               : camera -> RDK-X5-001 (TRAFFIC_ANALYTICS)

This is exactly what makes the event provenance real instead of hard-coded:

    processing.processing_node_id = RDK-X5-001        (resolved from topology)
    processing.analytics_provider = KAAVAL_AI         (adapter)
    processing.model_name/version = <configured>      (payload -> adapter)

Swapping the .mp4 for RTSP later keeps the same camera rows and assignments.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.ai.config import AIConfig, get_ai_config
from app.db.repositories import (
    AssignmentRepository,
    CameraRepository,
    HealthRepository,
    NodeRepository,
    SystemRepository,
)
from app.schemas.system import CameraAssignmentCreate, CameraCreate, ProcessingNodeCreate
from app.schemas.common import NodeType, SystemProtocol, SystemVendor

log = logging.getLogger("sentinel.m3.ai.topology")

LOCAL_VIDEO_SYSTEM_ID = "SENTINEL-LOCAL-VIDEO"


def _probe_video(path: str) -> Optional[Dict[str, object]]:
    """Resolution/fps of the configured video (best effort — the file may not exist yet)."""
    try:
        import cv2

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        info = {
            "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None,
        }
        cap.release()
        return info
    except Exception:  # pragma: no cover - probe must never break startup
        return None


def ensure_ai_video_topology(db: Optional[Session] = None, config: Optional[AIConfig] = None) -> Dict[str, int]:
    """
    Create (once) the demo system/node/cameras/assignments behind AI_VIDEO_SOURCES.
    Safe to call on every startup and every job start.
    """
    cfg = config or get_ai_config()

    def _run(session: Session) -> Dict[str, int]:
        systems = SystemRepository(session)
        nodes = NodeRepository(session)
        cameras = CameraRepository(session)
        assignments = AssignmentRepository(session)
        health = HealthRepository(session)
        created = {"systems": 0, "nodes": 0, "cameras": 0, "assignments": 0}

        if not systems.get(LOCAL_VIDEO_SYSTEM_ID):
            from app.schemas.system import SystemCreate

            systems.create(SystemCreate(
                id=LOCAL_VIDEO_SYSTEM_ID,
                name="Local Road Videos (simulated CCTV)",
                vendor=SystemVendor.GENERIC_RTSP,
                protocol=SystemProtocol.RTSP,
                connection_params={"source": "AI_VIDEO_SOURCES", "note": "local .mp4 files played as camera feeds"},
                status="ACTIVE",
            ))
            created["systems"] += 1

        node_id = cfg.processing_node_id
        if not nodes.get(node_id):
            nodes.create(ProcessingNodeCreate(
                id=node_id,
                name="Kaaval AI local processing node (YOLO helmet detector)",
                node_type=NodeType.EDGE_RDK_X5,
                system_id="KAAVAL-EDGE-SYSTEM",
                ip_address="127.0.0.1",
                status="ONLINE",
                metrics={"device": cfg.device, "model": cfg.model_name},
            ))
            created["nodes"] += 1
        health.upsert(node_id, "PROCESSING_NODE", "ONLINE", {"device": cfg.device, "model": cfg.model_name})
        health.upsert(f"KAAVAL-AI-{node_id}", "AI_RUNTIME", "RUNNING", {"provider": "KAAVAL_AI"})
        health.upsert(f"PIPE-{node_id}", "EVENT_PIPE", "UP", {"status": "CONNECTED"})

        for camera_id, video_path in cfg.video_sources.items():
            if not cameras.get(camera_id):
                probe = _probe_video(video_path) or {}
                cameras.create(CameraCreate(
                    id=camera_id,
                    external_id=camera_id,
                    name=f"{camera_id} — local road video ({Path(video_path).name})",
                    system_id=LOCAL_VIDEO_SYSTEM_ID,
                    rtsp_url=None,
                    location_name="Simulated CCTV feed (local file)",
                    status="ONLINE",
                    resolution=str(probe.get("resolution") or "1920x1080"),
                    fps=float(probe.get("fps") or 25.0),
                ))
                created["cameras"] += 1
            health.upsert(camera_id, "CAMERA", "ONLINE",
                          {"source": video_path, "note": "simulated feed from local video"})
            if (camera_id, node_id) not in {(a.camera_id, a.node_id) for a in assignments.list()}:
                assignments.create(CameraAssignmentCreate(
                    camera_id=camera_id, node_id=node_id, pipeline_profile="TRAFFIC_ANALYTICS",
                ))
                created["assignments"] += 1

        if any(created.values()):
            log.info("AI video topology ensured: %s", created)
        return created

    if db is not None:
        return _run(db)
    from app.db.database import SessionLocal

    with SessionLocal() as session:
        return _run(session)


def camera_video_map(config: Optional[AIConfig] = None) -> Dict[str, str]:
    return dict((config or get_ai_config()).video_sources)
