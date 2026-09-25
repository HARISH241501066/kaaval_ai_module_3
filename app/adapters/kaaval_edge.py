"""
Kaaval Edge Adapter (C1).
Normalizes raw telemetry and detection events from Kaaval RDK-X5 edge processing nodes
into Canonical Sentinel Event Envelopes.
Directly compatible with sentinel-hwsim node models.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from app.adapters.base import BaseAdapter
from app.schemas.event import (
    CanonicalEventEnvelope,
    SourceInfo,
    ProcessingProvenance,
    ObjectDetection,
    EvidenceRef,
)


class KaavalEdgeAdapter(BaseAdapter):
    @property
    def name(self) -> str:
        return "KAAVAL_EDGE"

    @property
    def protocol(self) -> str:
        return "KAAVAL_EDGE"

    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        """Validate structure of raw Kaaval edge event."""
        if not isinstance(raw_payload, dict):
            return False
        # Must have either kind='event' with evt_id and type, or kind='heartbeat'
        kind = raw_payload.get("kind", "event")
        if kind == "event":
            return bool(raw_payload.get("evt_id") and raw_payload.get("type"))
        elif kind == "heartbeat":
            return bool(raw_payload.get("node") and raw_payload.get("runtime"))
        return False

    def normalize(
        self,
        raw_payload: Dict[str, Any],
        camera_id: str,
        processing_node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CanonicalEventEnvelope:
        """
        Transforms a Kaaval Edge event payload into the Canonical Sentinel Event Envelope.
        """
        source_evt_id = str(raw_payload.get("evt_id") or uuid.uuid4())
        canonical_evt_id = f"evt-{uuid.uuid4().hex[:12]}"
        
        # Timestamp conversion
        ts_ms = raw_payload.get("ts_ms")
        if ts_ms:
            try:
                event_ts = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc)
            except Exception:
                event_ts = datetime.now(timezone.utc)
        else:
            event_ts = datetime.now(timezone.utc)

        # Confidence
        conf = float(raw_payload.get("conf", 1.0))
        conf = max(0.0, min(1.0, conf))

        # Event type
        raw_type = str(raw_payload.get("type", "UNKNOWN_EVENT")).upper()
        # Normalization mapping
        event_type_map = {
            "HELMET_VIOLATION": "HELMET_VIOLATION",
            "TRIPLE_RIDING": "TRIPLE_RIDING",
            "PLATE_READ": "PLATE_READ",
            "WATCHLIST_HIT": "WATCHLIST_HIT",
            "VEHICLE_DETECTION": "VEHICLE_DETECTION",
            "ANPR": "PLATE_READ",
        }
        event_type = event_type_map.get(raw_type, raw_type)

        data = raw_payload.get("data") or {}
        if not isinstance(data, dict):
            data = {}

        # Object extraction
        plate_number = data.get("plate")
        vehicle_class = data.get("vehicle_class")
        colour = data.get("colour")
        bbox = data.get("bbox") if isinstance(data.get("bbox"), list) else []

        # Local AI events (KAAVAL AI / YOLO) may declare the detected object explicitly
        # (e.g. "MOTORCYCLE_RIDER"); otherwise keep the legacy heuristic.
        obj_type = data.get("object_type") or (
            "VEHICLE" if (plate_number or vehicle_class or "RIDING" in event_type or "HELMET" in event_type)
            else "OBJECT"
        )

        node_id = processing_node_id or raw_payload.get("node", "RDK-X5-UNKNOWN")
        model_version = str(raw_payload.get("model_ver", "kaaval-v2.1"))

        # Build clean attributes without leaking internal raw keys.
        # Optional keys are honored when present (local YOLO events carry real provenance);
        # absent keys keep the legacy defaults, so existing payloads are unchanged.
        attributes = {}
        for key in ("riders", "count", "list", "speed_kmh", "lane",
                    "helmet_status", "track_id", "frame_number", "video_timestamp",
                    "confirmations", "vehicle_bbox", "detection_model"):
            if key in data:
                attributes[key] = data[key]
        if "violation" not in attributes and "VIOLATION" in event_type:
            attributes["violation"] = event_type

        return CanonicalEventEnvelope(
            event_id=canonical_evt_id,
            camera_id=camera_id,
            event_type=event_type,
            timestamp=event_ts,
            source=SourceInfo(
                provider="KAAVAL_AI",
                adapter=self.name,
                source_event_id=source_evt_id,
            ),
            processing=ProcessingProvenance(
                processing_node_id=node_id,
                analytics_version=str(raw_payload.get("agent_ver") or "kaaval-agent-1.4.0"),
                model_name=str(raw_payload.get("model_name") or f"KAAVAL_{event_type}"),
                model_version=model_version,
                inference_fps=float(raw_payload.get("fps") or 15.0),
                latency_ms=float(raw_payload.get("latency_ms") or 38.5),
            ),
            confidence=round(conf, 4),
            object=ObjectDetection(
                type=obj_type,
                plate_number=plate_number,
                vehicle_class=vehicle_class,
                colour=colour,
                bounding_box=bbox,
            ),
            attributes=attributes,
            # Local AI events carry a real evidence_ref ({"type": "local_file", "uri": ...});
            # legacy payloads keep the mock URLs.
            evidence_ref=(
                dict(data["evidence_ref"]) if isinstance(data.get("evidence_ref"), dict)
                else {
                    "image": f"mock://evidence/kaaval/{source_evt_id}.jpg",
                    "video": f"mock://evidence/kaaval/{source_evt_id}.mp4",
                }
            ),
            delivery_status="DELIVERED",
        )

    def check_health(self, target_id: str) -> Dict[str, Any]:
        return {
            "adapter": self.name,
            "target": target_id,
            "status": "ONLINE",
            "protocol": self.protocol,
            "message": "Kaaval edge adapter channel active",
        }
