"""
Typed internal structures for the KAAVAL AI layer.

Detection is the normalized contract between the YOLO implementation and everything
downstream: kaaval_logic, evidence rendering and the video processor never see an
Ultralytics Results object, so the model can later be swapped (custom Kaaval model,
ONNX, TensorRT, edge accelerator) without touching M3.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Canonical class names produced by the detector
WITH_HELMET = "WITH_HELMET"
WITHOUT_HELMET = "WITHOUT_HELMET"
PERSON = "PERSON"
MOTORCYCLE = "MOTORCYCLE"

HELMET_CLASSES = (WITH_HELMET, WITHOUT_HELMET)


@dataclass(frozen=True)
class Detection:
    """One normalized detection on one frame."""

    class_name: str                       # WITH_HELMET / WITHOUT_HELMET / PERSON / MOTORCYCLE / raw upper-case
    confidence: float                     # 0..1
    bbox: Tuple[int, int, int, int]       # x1, y1, x2, y2 in frame pixels
    frame_number: int = 0
    timestamp: float = 0.0                # video timestamp (seconds)
    track_id: Optional[int] = None        # stable id when tracking is enabled

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 3),
            "track_id": self.track_id,
        }


@dataclass
class ConfirmedViolation:
    """
    A confirmed HELMET_VIOLATION produced by the Kaaval AI decision logic.

    One per UNIQUE tracked rider (never one per frame): the same rider observed without a
    helmet across many frames contributes exactly one instance, carrying the frame-level
    counts (observations / violation_detections) and the best (highest-confidence) frame
    as its evidence.
    """

    event_id: str                          # Kaaval-side id, e.g. "EVT-8F2C1A9B04" (used for evidence file name)
    camera_id: str
    detection: Detection                   # the best WITHOUT_HELMET head detection for this rider
    confidence: float
    frame_number: int
    timestamp: float                       # video timestamp (s) of the best frame
    track_id: Optional[int] = None
    confirmations: int = 1                 # consecutive observations that confirmed the violation
    vehicle_bbox: Optional[Tuple[int, int, int, int]] = None   # associated motorcycle, if the model provides one
    vehicle_class: Optional[str] = None
    evidence_uri: Optional[str] = None      # set after the evidence frame was written
    # --- whole-video aggregation for this unique rider ---
    first_seen: float = 0.0                # video seconds of the first WITHOUT_HELMET observation
    last_seen: float = 0.0                 # video seconds of the last WITHOUT_HELMET observation
    observations: int = 0                  # frames this rider was seen WITHOUT_HELMET (>= helmet threshold)
    violation_detections: int = 0          # frame-level violation detections attributed to this rider

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "camera_id": self.camera_id,
            "class_name": self.detection.class_name,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.detection.bbox),
            "frame_number": self.frame_number,
            "timestamp": round(self.timestamp, 3),
            "track_id": self.track_id,
            "confirmations": self.confirmations,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "observations": self.observations,
            "violation_detections": self.violation_detections,
            "vehicle_bbox": list(self.vehicle_bbox) if self.vehicle_bbox else None,
            "evidence_ref": (
                {"type": "local_file", "uri": self.evidence_uri} if self.evidence_uri else None
            ),
        }


class JobStatus(str, enum.Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class Job:
    """A background video-processing job (in-memory, hackathon prototype)."""

    job_id: str
    camera_id: str
    video_path: str
    # "helmet" = local YOLO frame-loop job (/ai/helmet/process)
    # "video"  = provider-based analysis job (/ai/video/process: Google Cloud or local fallback)
    mode: str = "helmet"
    status: JobStatus = JobStatus.QUEUED
    created_at: float = 0.0
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    violations: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.status in (JobStatus.QUEUED, JobStatus.RUNNING)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "camera_id": self.camera_id,
            "video_path": self.video_path,
            "mode": self.mode,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "metrics": self.metrics,
            "result": self.metrics.get("result"),      # the compact success summary (or None)
            "violations": self.violations,
        }


def filter_detections(detections: List[Detection], min_confidence: float) -> List[Detection]:
    """Drop detections below the configured confidence (defense in depth: the model is
    also called with conf=min_confidence, but stub/alternate detectors may not filter)."""
    return [d for d in detections if d.confidence >= min_confidence]
