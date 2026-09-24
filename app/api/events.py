"""
Events Query, Seeding, and Mock Module 4 Consumer Router.
Allows retrieving canonical events and viewing what Module 4 has received.
"""
from __future__ import annotations

import uuid
import random
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.repositories import EventRepository, CameraRepository
from app.schemas.event import CanonicalEventEnvelope, SeedEventsRequest
from app.services.publisher import get_mock_m4_consumer
from app.services.pipeline import get_pipeline

router = APIRouter(prefix="/api/v1", tags=["Events & Module 4 Delivery"])


@router.get("/events", response_model=List[CanonicalEventEnvelope])
def list_canonical_events(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    node_id: Optional[str] = Query(None, description="Filter by processing node ID"),
    event_type: Optional[str] = Query(None, description="Filter by event type (e.g. HELMET_VIOLATION)"),
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """
    Query canonical events stored in the federation layer.
    Normalized according to the Sentinel Event Envelope.
    """
    repo = EventRepository(db)
    records = repo.list(camera_id=camera_id, node_id=node_id, event_type=event_type, skip=skip, limit=limit)
    
    # Transform database records into CanonicalEventEnvelope schemas
    results = []
    for r in records:
        results.append(
            CanonicalEventEnvelope(
                event_id=r.event_id,
                camera_id=r.camera_id,
                event_type=r.event_type,
                timestamp=r.timestamp,
                source={
                    "provider": r.provider,
                    "adapter": r.adapter,
                    "source_event_id": r.source_event_id,
                },
                processing=r.ai_provenance or {},
                confidence=r.confidence,
                object={
                    "type": r.object_type,
                    "plate_number": r.plate_number,
                    "bounding_box": r.bounding_box or [],
                },
                attributes=r.attributes or {},
                evidence_ref=r.evidence_ref or {},
                delivery_status=r.delivery_status,
            )
        )
    return results


@router.get("/events/{event_id}", response_model=CanonicalEventEnvelope)
def get_canonical_event(event_id: str, db: Session = Depends(get_db)):
    """Retrieve a single canonical event by its canonical event ID."""
    repo = EventRepository(db)
    r = repo.get_by_event_id(event_id)
    if not r:
        raise HTTPException(status_code=404, detail=f"Canonical event '{event_id}' not found")

    return CanonicalEventEnvelope(
        event_id=r.event_id,
        camera_id=r.camera_id,
        event_type=r.event_type,
        timestamp=r.timestamp,
        source={
            "provider": r.provider,
            "adapter": r.adapter,
            "source_event_id": r.source_event_id,
        },
        processing=r.ai_provenance or {},
        confidence=r.confidence,
        object={
            "type": r.object_type,
            "plate_number": r.plate_number,
            "bounding_box": r.bounding_box or [],
        },
        attributes=r.attributes or {},
        evidence_ref=r.evidence_ref or {},
        delivery_status=r.delivery_status,
    )


@router.post("/events/seed")
async def seed_demo_events(
    request: SeedEventsRequest,
    db: Session = Depends(get_db),
):
    """
    Seed sample demo events (Helmet violations, ANPR plate reads, Watchlist hits).
    Passes directly through the full C4 pipeline to prove normalization, deduplication, and M4 delivery.
    """
    cam_repo = CameraRepository(db)
    available_cameras = [c.id for c in cam_repo.list()]
    target_cam = request.camera_id or (available_cameras[0] if available_cameras else "CAM-AHM-SG-001")

    sample_plates = ["GJ01AB1234", "GJ27DF9876", "GJ05CD5555", "GJ18ZZ0007"]
    event_templates = [
        {
            "type": "HELMET_VIOLATION",
            "conf": 0.94,
            "data": {"plate": random.choice(sample_plates), "riders": 2, "bbox": [200, 350, 180, 220]},
        },
        {
            "type": "TRIPLE_RIDING",
            "conf": 0.91,
            "data": {"plate": random.choice(sample_plates), "riders": 3, "bbox": [150, 300, 210, 250]},
        },
        {
            "type": "PLATE_READ",
            "conf": 0.98,
            "data": {"plate": random.choice(sample_plates), "vehicle_class": "CAR", "colour": "WHITE", "bbox": [400, 500, 260, 140]},
        },
        {
            "type": "WATCHLIST_HIT",
            "conf": 0.96,
            "data": {"plate": "GJ01AB1234", "list": "STOLEN_VEHICLES_CRIME_BRANCH", "bbox": [300, 420, 280, 160]},
        },
    ]

    pipeline = get_pipeline()
    created = []

    for i in range(request.count):
        tmpl = random.choice(event_templates)
        source_id = f"seed-{uuid.uuid4().hex[:8]}"
        payload = {
            "kind": "event",
            "evt_id": source_id,
            "node": "RDK-X5-AHM-01",
            "cam": target_cam,
            "type": tmpl["type"],
            "ts_ms": int(datetime.now(timezone.utc).timestamp() * 1000),
            "conf": tmpl["conf"],
            "data": tmpl["data"],
            "model_ver": "kaaval-traffic-v2.4",
        }
        res = await pipeline.ingest_single_event(
            db=db,
            raw_payload=payload,
            provider="KAAVAL_AI",
            adapter_name="KAAVAL_EDGE",
        )
        created.append(res)

    return {"message": f"Successfully seeded {len(created)} demo events through C4 pipeline", "results": created}


@router.get("/mock-m4/events")
def get_mock_m4_delivered_events(limit: int = 50):
    """
    Inspect Canonical Sentinel Event Envelopes received by the downstream Module 4 Consumer.
    Confirms end-to-end delivery from edge to Module 4.
    """
    consumer = get_mock_m4_consumer()
    return {
        "consumer": "SENTINEL_MODULE_4_INTELLIGENCE",
        "description": "Mock Module 4 consumer in-memory receive buffer",
        "count": len(consumer.get_events(limit=limit)),
        "delivered_events": consumer.get_events(limit=limit),
    }
