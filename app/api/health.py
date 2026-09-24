"""
Health & Connectivity Monitoring Router (C6).
Provides independent health monitoring across 4 distinct dimensions:
1. Camera health
2. Processing Node health
3. AI Runtime health
4. Event Pipe health
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text

from app.db.database import get_db
from app.db.repositories import (
    CameraRepository,
    NodeRepository,
    HealthRepository,
)
from app.schemas.health import HealthSummaryResponse, ComponentHealth
from app.core.config import get_settings

router = APIRouter(tags=["Health & Monitoring (C6)"])


@router.get("/health", response_model=HealthSummaryResponse)
def get_health_status(db: Session = Depends(get_db)):
    """
    Module 3 and Integration Health endpoint.
    Reports separate and independent states for:
    - Camera health (ONLINE, OFFLINE)
    - Processing node health (ONLINE, OFFLINE)
    - AI runtime health (RUNNING, CRASHED, RESTARTING)
    - Event pipe health (UP, CUT)
    """
    settings = get_settings()

    # 1. Database check
    db_status = "CONNECTED"
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_status = "ERROR"

    # 2. Query entities
    camera_repo = CameraRepository(db)
    node_repo = NodeRepository(db)
    health_repo = HealthRepository(db)

    cameras = camera_repo.list()
    nodes = node_repo.list()
    health_records = health_repo.list_all()

    # Build component list
    components: list[ComponentHealth] = []

    # Map existing health records
    hr_map = {f"{r.entity_type}:{r.entity_id}": r for r in health_records}

    # Add camera healths
    cams_online = 0
    for cam in cameras:
        rec = hr_map.get(f"CAMERA:{cam.id}")
        st = rec.status if rec else cam.status
        if st == "ONLINE":
            cams_online += 1
        components.append(
            ComponentHealth(
                entity_id=cam.id,
                entity_type="CAMERA",
                status=st,
                last_seen=rec.last_observed if rec else cam.created_at,
                details={"name": cam.name, "system_id": cam.system_id, "resolution": cam.resolution},
            )
        )

    # Add processing node healths & AI runtimes & event pipes
    nodes_online = 0
    runtimes_running = 0
    pipes_up = 0

    for node in nodes:
        # Processing Node
        rec_node = hr_map.get(f"PROCESSING_NODE:{node.id}")
        node_st = rec_node.status if rec_node else node.status
        if node_st == "ONLINE":
            nodes_online += 1
        components.append(
            ComponentHealth(
                entity_id=node.id,
                entity_type="PROCESSING_NODE",
                status=node_st,
                last_seen=node.last_heartbeat,
                details={"type": node.node_type, "ip": node.ip_address, "metrics": node.metrics},
            )
        )

        # AI Runtime
        rec_ai = hr_map.get(f"AI_RUNTIME:KAAVAL-AI-{node.id}")
        ai_st = rec_ai.status if rec_ai else "RUNNING"
        if ai_st == "RUNNING":
            runtimes_running += 1
        components.append(
            ComponentHealth(
                entity_id=f"KAAVAL-AI-{node.id}",
                entity_type="AI_RUNTIME",
                status=ai_st,
                last_seen=rec_ai.last_observed if rec_ai else node.last_heartbeat,
                details=rec_ai.details if rec_ai else {},
            )
        )

        # Event Pipe
        rec_pipe = hr_map.get(f"EVENT_PIPE:PIPE-{node.id}")
        pipe_st = rec_pipe.status if rec_pipe else "UP"
        if pipe_st == "UP":
            pipes_up += 1
        components.append(
            ComponentHealth(
                entity_id=f"PIPE-{node.id}",
                entity_type="EVENT_PIPE",
                status=pipe_st,
                last_seen=rec_pipe.last_observed if rec_pipe else node.last_heartbeat,
                details=rec_pipe.details if rec_pipe else {},
            )
        )

    overall_status = "HEALTHY" if (db_status == "CONNECTED" and nodes_online > 0) else "DEGRADED"

    return HealthSummaryResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc),
        version=settings.version,
        database=db_status,
        summary={
            "cameras_total": len(cameras),
            "cameras_online": cams_online,
            "nodes_total": len(nodes),
            "nodes_online": nodes_online,
            "ai_runtimes_running": runtimes_running,
            "event_pipes_up": pipes_up,
        },
        components=components,
    )
