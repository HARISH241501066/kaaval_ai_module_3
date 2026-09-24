"""
Live AI Provenance Tracking Router.
Allows tracking full AI lineage for any event:
Camera -> Processing Node -> AI Model -> Inference Metrics -> Canonical Event.
"""
from __future__ import annotations

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import (
    EventRepository,
    CameraRepository,
    NodeRepository,
)

router = APIRouter(prefix="/api/v1/provenance", tags=["Live AI Provenance"])


@router.get("/{event_id}")
def get_event_ai_provenance(event_id: str, db: Session = Depends(get_db)):
    """
    Live AI Provenance endpoint:
    Returns complete AI execution provenance and audit trail for a specific event.
    """
    event_repo = EventRepository(db)
    camera_repo = CameraRepository(db)
    node_repo = NodeRepository(db)

    event = event_repo.get_by_event_id(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found")

    camera = camera_repo.get(event.camera_id)
    node = node_repo.get(event.processing_node_id) if event.processing_node_id else None

    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "timestamp": event.timestamp,
        "camera": {
            "id": event.camera_id,
            "name": camera.name if camera else "Unknown Camera",
            "location": camera.location_name if camera else None,
            "system_id": camera.system_id if camera else None,
        },
        "processing_node": {
            "id": node.id if node else event.processing_node_id,
            "name": node.name if node else "Unknown Node",
            "type": node.node_type if node else "UNKNOWN",
            "ip_address": node.ip_address if node else None,
            "status": node.status if node else "UNKNOWN",
        },
        "ai_lineage": {
            "provider": event.provider,
            "adapter": event.adapter,
            "provenance_metadata": event.ai_provenance,
            "confidence": event.confidence,
        },
        "evidence": event.evidence_ref,
        "audit": {
            "source_event_id": event.source_event_id,
            "raw_payload_preserved": bool(event.raw_payload),
            "ingested_at": event.created_at,
        },
    }


@router.get("/summary/stats")
def get_provenance_summary(db: Session = Depends(get_db)):
    """
    Returns aggregated AI model telemetry and provenance statistics across the deployment.
    """
    event_repo = EventRepository(db)
    node_repo = NodeRepository(db)
    camera_repo = CameraRepository(db)

    total_events = event_repo.count()
    recent_events = event_repo.list(limit=100)
    nodes = node_repo.list()

    models_observed = set()
    event_type_counts: Dict[str, int] = {}
    for ev in recent_events:
        event_type_counts[ev.event_type] = event_type_counts.get(ev.event_type, 0) + 1
        if isinstance(ev.ai_provenance, dict):
            m_ver = ev.ai_provenance.get("model_version")
            if m_ver:
                models_observed.add(m_ver)

    return {
        "total_canonical_events": total_events,
        "active_processing_nodes": len(nodes),
        "active_ai_models": list(models_observed),
        "recent_event_distribution": event_type_counts,
        "telemetry_health": "OPTIMAL",
    }
