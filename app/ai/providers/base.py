"""
Video-analytics provider abstraction for the Kaaval AI layer.

One internal contract, several backends:

    LocalYOLOProvider              local Ultralytics helmet model  (offline fallback)
    GoogleVideoIntelligenceProvider Cloud Video Intelligence person/object tracking
    (future)                       Kaaval custom model, Gemini, ONNX, ...

Nothing outside `app/ai/providers/` ever sees a Google protobuf, an Ultralytics
`Results`, or an SDK type: providers normalize their raw output into the dataclasses
below, and the M3 ingestion pipeline only ever receives canonical Sentinel events.

Capabilities are declared explicitly instead of assumed, because the differences matter:

* Cloud Video Intelligence answers "where is a person / object over time" — it does NOT
  answer "is this rider wearing a helmet". Its `capabilities()["helmet_status"]` is False,
  so the pipeline runs a second-stage HelmetClassifier on the tracked rider's frame.
* The local YOLO helmet model answers "helmet / no helmet" for head regions and tracks them
  with ByteTrack, so its `capabilities()["helmet_status"]` is True and no second stage is
  needed.

Both report `unique_identity: True` — the property the unique-violator count depends on.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from app.ai.schemas import Detection

# ANPR outcomes (honest states, never a silent empty string)
ANPR_FOUND = "FOUND"
ANPR_UNCERTAIN = "UNCERTAIN"
ANPR_NOT_FOUND = "NOT_FOUND"
ANPR_NOT_CONFIGURED = "NOT_CONFIGURED"
ANPR_ERROR = "ERROR"

Box = Tuple[int, int, int, int]        # pixels: x1, y1, x2, y2


class ProviderError(RuntimeError):
    """A provider could not complete the request (unavailable, quota, timeout, bad input)."""


class ProviderNotConfigured(ProviderError):
    """The provider is selected but its configuration/credentials are missing."""


@dataclass(frozen=True)
class NormalizedBox:
    """Google returns normalized [0,1] coordinates; M3 works in frame pixels."""

    left: float
    top: float
    right: float
    bottom: float

    def to_pixels(self, width: int, height: int) -> Box:
        x1 = int(round(max(0.0, min(1.0, self.left)) * width))
        y1 = int(round(max(0.0, min(1.0, self.top)) * height))
        x2 = int(round(max(0.0, min(1.0, self.right)) * width))
        y2 = int(round(max(0.0, min(1.0, self.bottom)) * height))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2


@dataclass
class TrackObservation:
    """One time-stamped sighting of one tracked rider/object.

    `bbox` is pixels for local providers and normalized [0,1] for cloud providers; the
    `bbox_normalized` flag says which, and `pixels()` is the only way callers should convert —
    the frame size is not known at normalization time, so guessing would silently shift boxes.
    """

    timestamp: float                       # video seconds
    bbox: Tuple[float, float, float, float]
    confidence: float = 1.0
    frame_number: int = -1
    bbox_normalized: bool = False          # True => bbox is in [0,1] (Google), False => pixels
    # Filled in only when the provider itself knows the helmet status (local YOLO).
    helmet_status: Optional[str] = None    # WITHOUT_HELMET | WITH_HELMET
    helmet_confidence: float = 0.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def pixels(self, width: int, height: int) -> Box:
        """Frame-pixel box for a known frame size (converting only when needed)."""
        x1, y1, x2, y2 = self.bbox
        if not self.bbox_normalized:
            return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        return NormalizedBox(x1, y1, x2, y2).to_pixels(width, height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_number": self.frame_number,
            "bbox": [round(float(v), 4) for v in self.bbox],
            "bbox_normalized": self.bbox_normalized,
            "confidence": round(self.confidence, 4),
            "helmet_status": self.helmet_status,
        }


@dataclass
class TrackedObject:
    """One tracked person/object instance for the whole video — the unit of uniqueness."""

    key: str                               # provider-stable identity ('person:0', 'obj:Car:3', 'trk:17')
    object_type: str                       # PERSON | VEHICLE | ...
    label: str = ""
    confidence: float = 0.0
    observations: List[TrackObservation] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def first_seen(self) -> float:
        return min((o.timestamp for o in self.observations), default=0.0)

    @property
    def last_seen(self) -> float:
        return max((o.timestamp for o in self.observations), default=0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "object_type": self.object_type,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "observations": len(self.observations),
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
        }


@dataclass
class VideoAnalysis:
    """Normalized result of analyzing one whole video with any provider."""

    provider: str                          # GOOGLE_VIDEO_INTELLIGENCE | LOCAL_YOLO
    model_name: str
    model_version: str
    capabilities: Dict[str, Any] = field(default_factory=dict)
    video_path: str = ""
    duration_s: float = 0.0
    video_fps: float = 0.0
    total_frames: int = 0
    frames_analyzed: int = 0
    tracks: List[TrackedObject] = field(default_factory=list)
    segments_analyzed: int = 1             # the request covers the WHOLE video
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def helmet_status_known(self) -> bool:
        return bool(self.capabilities.get("helmet_status"))

    @property
    def rider_tracks(self) -> List[TrackedObject]:
        """Tracks that can plausibly be a rider (persons, and bikes when labelled)."""
        wanted = {"PERSON", "MOTORCYCLE", "BICYCLE", "VEHICLE", "TWO_WHEELER", "RIDER"}
        return [t for t in self.tracks if (t.object_type or "").upper() in wanted or not t.object_type]

    def summary(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": f"{self.model_name} v{self.model_version}",
            "duration_s": round(self.duration_s, 2),
            "video_fps": round(self.video_fps, 3),
            "frames_analyzed": self.frames_analyzed,
            "segments_analyzed": self.segments_analyzed,
            "tracks": len(self.tracks),
            "rider_tracks": len(self.rider_tracks),
            "observations": sum(len(t.observations) for t in self.tracks),
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Provider interfaces
# ---------------------------------------------------------------------------
class VideoAnalyticsProvider(ABC):
    """Turns a video into tracked objects (never into helmet verdicts, unless capable)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier used in provenance (e.g. 'GOOGLE_VIDEO_INTELLIGENCE')."""

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """What this backend can and cannot do (person/object tracking, helmet status, ...)."""

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """READY / AVAILABLE / ERROR + reason. Must never raise."""

    @abstractmethod
    def analyze_video(
        self,
        video_path: str,
        camera_id: str = "CAM-001",
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_observations_per_track: Optional[int] = None,
    ) -> VideoAnalysis:
        """Analyze the COMPLETE video and return normalized tracks."""

    # Optional: providers that need their tracker state cleared between videos
    def reset(self) -> None:  # pragma: no cover - default no-op
        return None


class HelmetClassifier(ABC):
    """Second stage: helmet status for one tracked rider region."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Classifier identifier (e.g. 'LOCAL_YOLO_HELMET')."""

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Classes it can produce, thresholds, model card."""

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """READY / FAILED + reason. Must never raise."""

    @abstractmethod
    def classify(
        self,
        frame: Any,
        bbox: Optional[Box] = None,
        frame_number: int = 0,
        timestamp: float = 0.0,
        track_id: Optional[int] = None,
    ) -> List[Detection]:
        """Classify the rider region (or the whole frame when bbox is None).

        Returns normalized Detection objects with class names from the canonical
        vocabulary (WITH_HELMET / WITHOUT_HELMET / ...).
        """


class AnprProvider(ABC):
    """Reads a plate from ONE selected evidence frame (never from every frame)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier used in provenance (e.g. 'GOOGLE_CLOUD_VISION')."""

    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """What it does (OCR) and, explicitly, what it does not (a plate detector)."""

    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """AVAILABLE / ERROR / NOT_CONFIGURED + reason. Must never raise."""

    @abstractmethod
    def recognize(self, image: Any, context: Optional[Dict[str, Any]] = None) -> "AnprResult":
        """OCR one image (path or BGR ndarray) and extract a plate candidate."""


@dataclass
class AnprResult:
    """Outcome of one OCR attempt, with an explicit state instead of a silent empty plate."""

    status: str                                  # FOUND | UNCERTAIN | NOT_FOUND | NOT_CONFIGURED | ERROR
    plate_number: Optional[str] = None           # normalized, e.g. GJ01AB1234
    raw_text: Optional[str] = None               # what OCR actually returned
    provider: str = ""
    confidence: Optional[float] = None
    attempts: int = 1
    notes: Optional[str] = None

    @property
    def found(self) -> bool:
        return self.status == ANPR_FOUND

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "plate_number": self.plate_number,
            "raw_text": self.raw_text,
            "provider": self.provider,
            "confidence": self.confidence,
            "attempts": self.attempts,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Plate normalization (generic OCR -> candidate Indian plate)
# ---------------------------------------------------------------------------
# Typical formats: GJ01AB1234, GJ1AB1234, GJ01A1234, DL8CAF5030, MH12DE1433.
# OCR output is noisy, so this only ever produces a *candidate* with a status.
_PLATE_PATTERNS: Sequence[str] = (
    r"\b([A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4})\b",       # strict Indian plate shape
    r"\b([A-Z]{2}[0-9]{1,2}[A-Z]{2}[0-9]{4})\b",           # BH-series style
    r"\b([A-Z]{2}[- ]?[0-9]{1,2}[- ]?[A-Z]{1,3}[- ]?[0-9]{3,4})\b",  # spaced/punctuated
)


def normalize_plate(text: Optional[str]) -> Optional[str]:
    """Normalize a raw OCR string to an alphanumeric plate candidate (never inventing characters)."""
    if not text:
        return None
    cleaned = "".join(ch for ch in str(text).upper() if ch.isalnum())
    return cleaned or None


def extract_plate_candidate(
    raw_text: Optional[str],
    min_length: int = 6,
    max_length: int = 12,
) -> Tuple[Optional[str], str]:
    """
    Find the most plate-like token in raw OCR text.

    Returns (plate_candidate, reason). `reason` is empty on a confident match and explains
    the downgrade otherwise — the caller maps it onto ANPR_FOUND / ANPR_UNCERTAIN.
    """
    import re

    if not raw_text:
        return None, "no text detected"
    upper = str(raw_text).upper()
    # 1. strict shape first (highest precision)
    for pattern in _PLATE_PATTERNS:
        for match in re.finditer(pattern, upper):
            candidate = normalize_plate(match.group(1))
            if candidate and min_length <= len(candidate) <= max_length:
                return candidate, ""
    # 2. token that *looks* like a plate but does not match the state/format rules
    best = None
    for token in re.split(r"[^A-Z0-9]+", upper):
        if min_length <= len(token) <= max_length and any(ch.isdigit() for ch in token):
            if best is None or len(token) > len(best):
                best = token
    if best:
        return best, "OCR text does not match an Indian plate pattern"
    return None, "no plate-like token in OCR text"
