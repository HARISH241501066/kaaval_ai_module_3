"""
KAAVAL AI — local inference layer (YOLO helmet detection) for Sentinel M3.

Flow (never bypasses M3):

    LOCAL VIDEO -> VideoSource -> YOLO HelmetDetector -> Kaaval AI decision logic
        -> HELMET_VIOLATION Kaaval event -> existing M3 Kaaval adapter / C4 ingestion
        -> validation / identity resolution / deduplication -> Canonical Sentinel Event Envelope
        -> SQLite -> Mock M4

Heavy ML imports (ultralytics / torch) are deliberately lazy so the M3 app and its
tests stay importable without the ML stack installed.
"""
from app.ai.schemas import Detection, ConfirmedViolation, JobStatus  # noqa: F401

__all__ = ["Detection", "ConfirmedViolation", "JobStatus"]
