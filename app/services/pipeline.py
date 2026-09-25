"""
C4 Event Ingestion & Translation Pipeline.
Orchestrates the 11-step pipeline from Section 11 of the Architecture Blueprint:
Authenticate -> Buffer/Parse -> Resolve Camera & Node -> Normalize via Adapter ->
Deduplicate -> Persist -> Publish to Module 4 -> Update Health.
"""
from __future__ import annotations

import time
import hmac
import hashlib
import logging
from typing import Dict, Any, Tuple, Optional, List
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.adapters.registry import get_adapter_registry
from app.db.repositories import (
    CameraRepository,
    NodeRepository,
    EventRepository,
    HealthRepository,
)
from app.models.event import CanonicalEventRecord
from app.schemas.event import CanonicalEventEnvelope, IngestResult, BatchIngestResult
from app.services.deduplicator import get_deduplicator
from app.services.publisher import get_publisher

logger = logging.getLogger("sentinel.m3.pipeline")


def verify_kaaval_hmac(
    secret: str,
    timestamp_str: str,
    raw_body: bytes,
    signature: str,
    max_skew_s: int = 300,
) -> Tuple[bool, str]:
    """
    Verifies HMAC-SHA256 signature matching sentinel_hwsim.push.sign():
    sign(secret, ts, body) = hmac.new(secret.encode(), f"{timestamp}.".encode() + body, hashlib.sha256).hexdigest()
    """
    try:
        req_ts = int(timestamp_str)
        now_ts = int(time.time())
        if abs(now_ts - req_ts) > max_skew_s:
            return False, f"Clock skew exceeded: {abs(now_ts - req_ts)}s > {max_skew_s}s"
    except Exception:
        return False, "Invalid timestamp header"

    expected_sig = hmac.new(
        secret.encode(),
        f"{timestamp_str}.".encode() + raw_body,
        hashlib.sha256
    ).hexdigest()

    if not hmac.compare_digest(expected_sig, signature):
        return False, "HMAC signature mismatch"

    return True, "Valid signature"


class C4IngestionPipeline:
    def __init__(self):
        self.settings = get_settings()
        self.registry = get_adapter_registry()
        self.deduplicator = get_deduplicator()
        self.publisher = get_publisher()

    async def ingest_single_event(
        self,
        db: Session,
        raw_payload: Dict[str, Any],
        provider: str = "KAAVAL_AI",
        adapter_name: Optional[str] = None,
    ) -> IngestResult:
        """
        Executes C4 pipeline for a single raw event.
        """
        camera_repo = CameraRepository(db)
        node_repo = NodeRepository(db)
        event_repo = EventRepository(db)
        health_repo = HealthRepository(db)

        # 1. Resolve Adapter
        adapter_key = adapter_name or ("KAAVAL_EDGE" if "node" in raw_payload or "cam" in raw_payload else "KAAVAL_SERVER")
        adapter = self.registry.get(adapter_key)
        if not adapter:
            return IngestResult(
                status="REJECTED",
                duplicate=False,
                message=f"No registered adapter found for key: {adapter_key}",
            )

        # 2. Extract identifiers
        source_event_id = str(raw_payload.get("evt_id") or raw_payload.get("event_id") or "")
        node_identifier = str(raw_payload.get("node") or raw_payload.get("server_id") or "")
        cam_identifier = str(raw_payload.get("cam") or raw_payload.get("camera_id") or "")

        # 3. Resolve Camera
        camera = camera_repo.resolve_camera(cam_identifier)
        if camera:
            resolved_camera_id = camera.id
        else:
            # Prototype resilience: if camera not yet registered, map to default or canonical
            resolved_camera_id = cam_identifier if cam_identifier.startswith("CAM-") else f"CAM-{cam_identifier}"
            logger.info(f"Camera {cam_identifier} not in inventory; resolved as {resolved_camera_id}")

        # 4. Resolve Node
        node = node_repo.get(node_identifier)
        resolved_node_id = node.id if node else node_identifier

        # 5. Check Deduplication before heavy processing
        if source_event_id:
            is_dup, existing_id = self.deduplicator.check_and_record(
                db=db,
                provider=provider,
                source_event_id=source_event_id,
                canonical_event_id="",  # Placeholder until normalized
            )
            if is_dup:
                return IngestResult(
                    status="DUPLICATE",
                    canonical_event_id=existing_id,
                    source_event_id=source_event_id,
                    duplicate=True,
                    message="Duplicate event suppressed by deduplication filter",
                )

        # 6. Normalize via Adapter
        try:
            canonical_envelope: CanonicalEventEnvelope = adapter.normalize(
                raw_payload=raw_payload,
                camera_id=resolved_camera_id,
                processing_node_id=resolved_node_id,
            )
        except Exception as e:
            logger.error(f"Adapter normalization failed: {e}", exc_info=True)
            return IngestResult(
                status="REJECTED",
                duplicate=False,
                source_event_id=source_event_id,
                message=f"Adapter normalization error: {str(e)}",
            )

        # Update deduplication cache with actual canonical event ID
        if source_event_id:
            self.deduplicator._cache[self.deduplicator._make_key(provider, source_event_id)] = (
                time.time(),
                canonical_envelope.event_id,
            )

        # 7. Persist Canonical Event Record
        event_record = CanonicalEventRecord(
            event_id=canonical_envelope.event_id,
            source_event_id=canonical_envelope.source.source_event_id,
            provider=canonical_envelope.source.provider,
            adapter=canonical_envelope.source.adapter,
            camera_id=canonical_envelope.camera_id,
            processing_node_id=canonical_envelope.processing.processing_node_id,
            event_type=canonical_envelope.event_type,
            timestamp=canonical_envelope.timestamp,
            confidence=canonical_envelope.confidence,
            object_type=canonical_envelope.object.type if canonical_envelope.object else None,
            plate_number=canonical_envelope.object.plate_number if canonical_envelope.object else None,
            bounding_box=canonical_envelope.object.bounding_box if canonical_envelope.object else [],
            attributes=canonical_envelope.attributes,
            ai_provenance=canonical_envelope.processing.model_dump(mode="json"),
            evidence_ref=canonical_envelope.evidence_ref,
            raw_payload=raw_payload,
            delivery_status="DELIVERED",
        )
        event_repo.create(event_record)

        # 8. Update Health state (Event Pipe is UP, node observed)
        health_repo.upsert(
            entity_id=f"PIPE-{resolved_node_id}",
            entity_type="EVENT_PIPE",
            status="UP",
            details={"last_event_id": canonical_envelope.event_id, "camera": resolved_camera_id},
        )
        if node:
            node_repo.update_heartbeat(node.id, status="ONLINE")

        # 9. Publish downstream to Module 4
        await self.publisher.publish(canonical_envelope)

        return IngestResult(
            status="ACCEPTED",
            canonical_event_id=canonical_envelope.event_id,
            source_event_id=source_event_id,
            duplicate=False,
            message="Event ingested, normalized, stored, and published to Module 4",
        )

    def process_heartbeat(self, db: Session, item: Dict[str, Any]) -> None:
        """
        Processes Kaaval heartbeat item:
        Updates ProcessingNode, AI Runtime, and Event Pipe health records independently.
        """
        node_id = str(item.get("node", "UNKNOWN_NODE"))
        runtime = str(item.get("runtime", "RUNNING"))
        fps = float(item.get("fps", 0.0))

        node_repo = NodeRepository(db)
        health_repo = HealthRepository(db)

        # 1. Update Processing Node health
        node_repo.update_heartbeat(node_id, metrics={"fps": fps, "runtime": runtime}, status="ONLINE")
        health_repo.upsert(
            entity_id=node_id,
            entity_type="PROCESSING_NODE",
            status="ONLINE",
            details={"fps": fps, "runtime": runtime},
        )

        # 2. Update AI Runtime health (RUNNING / CRASHED / RESTARTING)
        health_repo.upsert(
            entity_id=f"KAAVAL-AI-{node_id}",
            entity_type="AI_RUNTIME",
            status=runtime,
            details={"fps": fps},
        )

        # 3. Update Event Pipe health
        health_repo.upsert(
            entity_id=f"PIPE-{node_id}",
            entity_type="EVENT_PIPE",
            status="UP",
            details={"last_heartbeat_fps": fps},
        )

    async def ingest_kaaval_batch(
        self,
        db: Session,
        node_id: str,
        batch_items: List[Dict[str, Any]],
    ) -> BatchIngestResult:
        """
        Processes a batch of items from Kaaval (as dispatched by sentinel_hwsim.push).
        Splits into events and heartbeats, ensuring no failure in one item crashes the batch.
        """
        processed = len(batch_items)
        accepted = 0
        duplicates = 0
        rejected = 0
        heartbeats = 0
        results: List[IngestResult] = []

        for item in batch_items:
            if not isinstance(item, dict):
                rejected += 1
                results.append(IngestResult(status="REJECTED", duplicate=False, message="Invalid item format"))
                continue

            kind = item.get("kind", "event")
            if kind == "heartbeat":
                heartbeats += 1
                self.process_heartbeat(db, item)
            else:
                # Ensure node ID is populated if missing in item
                if "node" not in item:
                    item["node"] = node_id
                res = await self.ingest_single_event(
                    db=db,
                    raw_payload=item,
                    provider=str(item.get("provider") or "KAAVAL_AI"),
                    adapter_name=str(item.get("adapter") or "KAAVAL_EDGE"),
                )
                if res.status == "ACCEPTED":
                    accepted += 1
                elif res.status == "DUPLICATE":
                    duplicates += 1
                else:
                    rejected += 1
                results.append(res)

        return BatchIngestResult(
            processed=processed,
            accepted=accepted,
            duplicates=duplicates,
            rejected=rejected,
            heartbeats=heartbeats,
            results=results,
        )


# Global pipeline instance
ingestion_pipeline = C4IngestionPipeline()


def get_pipeline() -> C4IngestionPipeline:
    return ingestion_pipeline
