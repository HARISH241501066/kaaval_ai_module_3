"""
Live Detector Adapter (C1).
Normalizes events that app.live.worker detected in live Sentinel Camera Grid streams into
Canonical Sentinel Event Envelopes, with honest provenance (the real model, real inference rate
and latency) instead of the mock Kaaval labels.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.adapters.base import BaseAdapter
from app.schemas.event import (
    CanonicalEventEnvelope,
    ObjectDetection,
    ProcessingProvenance,
    SourceInfo,
)

_VEHICLE_EVENTS = {"VEHICLE_DETECTION", "TRIPLE_RIDING", "HELMET_VIOLATION", "PLATE_READ"}


class LiveDetectorAdapter(BaseAdapter):
    @property
    def name(self) -> str:
        return "LIVE_DETECTOR"

    @property
    def protocol(self) -> str:
        return "SENTINEL_GRID"

    def validate_payload(self, raw_payload: Dict[str, Any]) -> bool:
        if not isinstance(raw_payload, dict):
            return False
        return bool(raw_payload.get("evt_id") and raw_payload.get("type") and raw_payload.get("cam"))

    def normalize(
        self,
        raw_payload: Dict[str, Any],
        camera_id: str,
        processing_node_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> CanonicalEventEnvelope:
        if not self.validate_payload(raw_payload):
            raise ValueError("live detector event needs evt_id, type and cam")
        evt_id = str(raw_payload["evt_id"])
        event_type = str(raw_payload["type"]).upper()
        try:
            ts = datetime.fromtimestamp(float(raw_payload["ts_ms"]) / 1000.0, tz=timezone.utc)
        except (KeyError, TypeError, ValueError):
            ts = datetime.now(timezone.utc)
        data = raw_payload.get("data") if isinstance(raw_payload.get("data"), dict) else {}
        bbox = data.get("bbox") if isinstance(data.get("bbox"), list) else []

        attributes = {k: data[k] for k in ("vehicle_class", "riders", "track_id", "frame_width", "frame_height") if k in data}
        if raw_payload.get("pts_ms") is not None:
            attributes["stream_pts_ms"] = raw_payload["pts_ms"]      # position in the feed, not wall-clock

        return CanonicalEventEnvelope(
            event_id=f"evt-{uuid.uuid4().hex[:12]}",
            camera_id=camera_id,
            event_type=event_type,
            timestamp=ts,
            source=SourceInfo(provider="SENTINEL_GRID", adapter=self.name, source_event_id=evt_id),
            processing=ProcessingProvenance(
                processing_node_id=processing_node_id or raw_payload.get("node"),
                analytics_version="sentinel-live-1.0",
                model_name=str(raw_payload.get("model_ver", "unknown")),
                model_version=str(raw_payload.get("model_ver", "unknown")),
                inference_fps=_num(raw_payload.get("fps")),
                latency_ms=_num(raw_payload.get("latency_ms")),
            ),
            confidence=round(max(0.0, min(1.0, float(raw_payload.get("conf", 0.0)))), 4),
            object=ObjectDetection(
                type="VEHICLE" if event_type in _VEHICLE_EVENTS else "OBJECT",
                vehicle_class=data.get("vehicle_class"),
                bounding_box=[int(v) for v in bbox],
            ),
            attributes=attributes,
            evidence_ref={},              # no stored clip or image yet: say so instead of faking a URL
            delivery_status="DELIVERED",
        )

    def check_health(self, target_id: str) -> Dict[str, Any]:
        return {"adapter": self.name, "target": target_id, "status": "ONLINE", "protocol": self.protocol,
                "message": "Live detector adapter channel active"}


def _num(v: Any) -> Optional[float]:
    try:
        return round(float(v), 2) if v is not None else None
    except (TypeError, ValueError):
        return None
