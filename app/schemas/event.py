"""
Pydantic schemas for the Canonical Sentinel Event Envelope (C4 specification).
Adheres directly to Section 10 & 11 of the Sentinel Model 3 Architecture & Workflow Blueprint.
Contract interface with Module 4 (Intelligence / Events).
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class SourceInfo(BaseModel):
    provider: str = Field(..., example="KAAVAL_AI")
    adapter: str = Field(..., example="KAAVAL_EDGE")
    source_event_id: str = Field(..., example="3a1c6a21-9d22-48f5-b3e1-88f62c079db4")


class ProcessingProvenance(BaseModel):
    processing_node_id: Optional[str] = Field(None, example="RDK-X5-001")
    analytics_version: Optional[str] = Field(None, example="2.0")
    model_name: Optional[str] = Field(None, example="HELMET_DETECTION")
    model_version: Optional[str] = Field(None, example="HELMET-2.4")
    inference_fps: Optional[float] = Field(None, example=15.2)
    latency_ms: Optional[float] = Field(None, example=42.5)
    # External-provider provenance (additive and optional: payloads that do not carry these
    # keys keep their previous behaviour exactly). When a video is analyzed by a cloud service,
    # the event says which service produced the detections/tracking — the model_* fields above
    # stay the helmet model that decided the violation.
    analytics_provider: Optional[str] = Field(None, example="GOOGLE_VIDEO_INTELLIGENCE")
    analytics_model_name: Optional[str] = Field(None, example="google-cloud-video-intelligence")
    analytics_model_version: Optional[str] = Field(None, example="v1")
    helmet_classifier: Optional[str] = Field(None, example="LOCAL_YOLO_HELMET")
    anpr_provider: Optional[str] = Field(None, example="GOOGLE_CLOUD_VISION")


class ObjectDetection(BaseModel):
    type: Optional[str] = Field(None, example="VEHICLE")  # "VEHICLE", "PERSON"
    plate_number: Optional[str] = Field(None, example="GJ01AB1234")
    vehicle_class: Optional[str] = Field(None, example="TWO_WHEELER")
    colour: Optional[str] = Field(None, example="BLACK")
    bounding_box: Optional[List[int]] = Field(default_factory=list, example=[120, 340, 200, 150])


class EvidenceRef(BaseModel):
    image: Optional[str] = Field(None, example="mock://evidence/evt-001.jpg")
    video: Optional[str] = Field(None, example="mock://evidence/evt-001.mp4")


class CanonicalEventEnvelope(BaseModel):
    """
    The Canonical Sentinel Event Envelope.
    This is the formal cross-module contract between M3 and Module 4.
    """
    event_id: str = Field(..., example="evt-001")
    camera_id: str = Field(..., example="CAM-001")
    event_type: str = Field(..., example="HELMET_VIOLATION")
    timestamp: datetime
    source: SourceInfo
    processing: ProcessingProvenance
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, example=0.94)
    object: Optional[ObjectDetection] = Field(default_factory=ObjectDetection)
    attributes: Dict[str, Any] = Field(default_factory=dict, example={"violation": "NO_HELMET", "riders": 2})
    evidence_ref: Dict[str, Any] = Field(default_factory=dict, example={"image": "mock://evidence/evt-001.jpg"})
    delivery_status: str = Field(default="DELIVERED")

    class Config:
        from_attributes = True


# Ingestion request / response schemas
class IngestResult(BaseModel):
    status: str = Field(..., example="ACCEPTED")  # "ACCEPTED", "DUPLICATE", "REJECTED"
    canonical_event_id: Optional[str] = None
    source_event_id: Optional[str] = None
    duplicate: bool = False
    message: Optional[str] = None


class BatchIngestResult(BaseModel):
    processed: int
    accepted: int
    duplicates: int
    rejected: int
    heartbeats: int
    results: List[IngestResult]


class GenericEventIngestRequest(BaseModel):
    provider: str = Field(default="KAAVAL_AI", example="KAAVAL_AI")
    adapter: Optional[str] = Field(None, example="KAAVAL_EDGE")
    payload: Dict[str, Any] = Field(..., example={
        "kind": "event",
        "evt_id": "test-123",
        "node": "RDK-X5-001",
        "cam": "CAM_01",
        "type": "HELMET_VIOLATION",
        "ts_ms": 1727170000000,
        "conf": 0.95,
        "data": {"plate": "GJ01AB1234", "bbox": [100, 200, 300, 400]}
    })


class SeedEventsRequest(BaseModel):
    count: int = Field(default=5, ge=1, le=50, example=5)
    camera_id: Optional[str] = Field(None, example="CAM-001")
