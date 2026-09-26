"""
ANPR through Google Cloud Vision OCR — one call per CONFIRMED violation, never per frame.

Honesty rules baked into this module (see the task's sections 17 and 34):

* Cloud Vision `TEXT_DETECTION` is a **generic OCR** engine. It is not a dedicated Indian
  ANPR/plate detector, and this code never claims otherwise.
* There is no plate detector here, so the "plate region" is a heuristic: when the pipeline
  knows the rider's box it is passed as `context["roi"]` and OCR words outside it are
  ignored; otherwise the whole evidence frame is searched.
* Outcomes are explicit states — `FOUND`, `UNCERTAIN`, `NOT_FOUND`, `NOT_CONFIGURED`,
  `ERROR` — so a missing plate is never silently reported as an empty string.
* The raw OCR text is kept next to the normalized candidate, because normalization can only
  clean characters, never invent them.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.ai.config import AIConfig, get_ai_config
from app.ai.providers.base import (
    ANPR_ERROR,
    ANPR_FOUND,
    ANPR_NOT_CONFIGURED,
    ANPR_NOT_FOUND,
    ANPR_UNCERTAIN,
    AnprProvider,
    AnprResult,
    extract_plate_candidate,
)
from app.ai.providers.google_video import map_google_error, sdk_available

log = logging.getLogger("sentinel.m3.ai.providers.vision")

PROVIDER_NAME = "GOOGLE_CLOUD_VISION"
OCR_CONFIDENCE_NOTE = (
    "Cloud Vision TEXT_DETECTION does not return a per-string confidence for sparse text; "
    "the plate status comes from pattern matching, not from a plate detector."
)


def vision_sdk_available() -> bool:
    try:
        import google.cloud.vision  # noqa: F401

        return True
    except Exception:
        return False


class NoAnprProvider(AnprProvider):
    """Explicit 'we did not run ANPR' provider — a missing plate is never faked."""

    def __init__(self, reason: str = "ANPR disabled or no OCR provider selected"):
        self.reason = reason

    @property
    def name(self) -> str:
        return "ANPR_DISABLED"

    def capabilities(self) -> Dict[str, Any]:
        return {"text_detection": False, "plate_detector": False, "cloud_api_calls": 0,
                "note": self.reason}

    def health(self) -> Dict[str, Any]:
        return {"status": "NOT_CONFIGURED", "provider": self.name, "error": None, "note": self.reason}

    def recognize(self, image: Any, context: Optional[Dict[str, Any]] = None) -> AnprResult:
        return AnprResult(status=ANPR_NOT_CONFIGURED, provider=self.name, notes=self.reason)


class GoogleVisionAnprProvider(AnprProvider):
    """OCR one selected evidence frame and extract a plate candidate."""

    def __init__(self, config: Optional[AIConfig] = None, client: Any = None):
        self.config = config or get_ai_config()
        self._client = client                       # injectable for tests
        self._client_error: Optional[str] = None
        self.calls = 0                              # bounded per job by the processor

    # ------------------------------------------------------------------ identity
    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def capabilities(self) -> Dict[str, Any]:
        return {
            "text_detection": True,
            "plate_detector": False,
            "unique_identity": False,
            "cloud_api_calls": "1 per confirmed violation (bounded by ANPR_MAX_PER_JOB)",
            "min_plate_length": self.config.anpr_min_plate_length,
            "note": "Generic Cloud Vision OCR + Indian plate pattern matching — not a dedicated ANPR engine.",
            "confidence_note": OCR_CONFIDENCE_NOTE,
        }

    def health(self) -> Dict[str, Any]:
        base = {
            "provider": PROVIDER_NAME,
            "sdk_installed": vision_sdk_available(),
            "project_id_configured": bool(self.config.google_project_id),
            "credentials_source": self.config.google_credentials_source,
            "calls_this_process": self.calls,
        }
        if not vision_sdk_available():
            return {**base, "status": "ERROR", "error":
                    "google-cloud-vision is not installed. Install it with "
                    "'pip install -r requirements.txt' (google-cloud-vision)"}
        if not self.config.google_project_id:
            return {**base, "status": "NOT_CONFIGURED", "error":
                    "GOOGLE_CLOUD_PROJECT_ID is not set — Cloud Vision OCR cannot be used."}
        if self.config.google_credentials_source == "none":
            return {**base, "status": "NOT_CONFIGURED", "error":
                    "Google credentials not found: set GOOGLE_APPLICATION_CREDENTIALS (git-ignored) "
                    "or run 'gcloud auth application-default login'."}
        if self._client_error:
            return {**base, "status": "ERROR", "error": self._client_error}
        return {**base, "status": "AVAILABLE", "error": None}

    # ------------------------------------------------------------------ client
    def _get_client(self):
        if self._client is not None:
            return self._client
        if not vision_sdk_available():
            raise _not_configured("google-cloud-vision is not installed — run 'pip install -r requirements.txt'")
        if not self.config.google_project_id:
            raise _not_configured("GOOGLE_CLOUD_PROJECT_ID is not set — Cloud Vision needs a project.")
        creds = self.config.google_vision_credentials or self.config.google_credentials_path
        try:
            from google.cloud import vision

            if creds:
                from google.oauth2 import service_account

                credentials = service_account.Credentials.from_service_account_file(creds)
                self._client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                self._client = vision.ImageAnnotatorClient()
        except Exception as e:
            err = map_google_error(e, "Cloud Vision client setup")
            self._client_error = str(err)
            raise err
        return self._client

    # ------------------------------------------------------------------ OCR
    def recognize(self, image: Any, context: Optional[Dict[str, Any]] = None) -> AnprResult:
        context = context or {}
        try:
            content, size = self._image_bytes(image)
        except Exception as e:
            return AnprResult(status=ANPR_ERROR, provider=self.name, notes=f"cannot read image: {e}")

        try:
            client = self._get_client()
            response = client.text_detection(image=self._image_message(content))
        except Exception as e:
            err = map_google_error(e, "Cloud Vision text_detection")
            log.warning("ANPR OCR failed: %s", err)
            return AnprResult(status=ANPR_ERROR, provider=self.name, notes=str(err))
        self.calls += 1

        annotations = list(getattr(response, "text_annotations", None) or [])
        raw_full = (getattr(annotations[0], "description", "") or "").strip() if annotations else ""
        roi = context.get("roi")
        raw_roi = self._words_inside(annotations[1:], roi, size) if roi and len(annotations) > 1 else ""

        # Prefer text inside the rider/vehicle region, fall back to the whole frame.
        for raw, scope in ((raw_roi, "roi"), (raw_full, "frame")):
            candidate, reason = extract_plate_candidate(raw, min_length=self.config.anpr_min_plate_length)
            if candidate and not reason:
                log.info("ANPR FOUND plate %s (%s) on %s", candidate, scope, context.get("event_id"))
                return AnprResult(status=ANPR_FOUND, plate_number=candidate, raw_text=raw_full,
                                  provider=self.name, confidence=None,
                                  notes=f"pattern match in {scope} text; {OCR_CONFIDENCE_NOTE}")
            if candidate:
                return AnprResult(status=ANPR_UNCERTAIN, plate_number=candidate, raw_text=raw_full,
                                  provider=self.name, confidence=None,
                                  notes=f"{reason} (scope={scope}); {OCR_CONFIDENCE_NOTE}")
        return AnprResult(status=ANPR_NOT_FOUND, raw_text=raw_full or None, provider=self.name,
                          confidence=None, notes="no plate-like text detected")

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _image_message(content: bytes) -> Any:
        """`vision.Image(content=...)` when the SDK is present; a plain shim otherwise.

        The shim keeps an injected client (tests, alternate OCR backends) working without the
        SDK installed — the real client is the only one that requires the real message.
        """
        try:
            from google.cloud import vision

            return vision.Image(content=content)
        except Exception:                                        # pragma: no cover - test path
            import types

            return types.SimpleNamespace(content=content)

    @staticmethod
    def _image_bytes(image: Any) -> Tuple[bytes, Tuple[int, int]]:
        """Accept a path or a BGR ndarray; return (jpeg bytes, (width, height))."""
        if isinstance(image, (str, Path)):
            path = Path(str(image))
            if not path.is_file():
                raise FileNotFoundError(f"evidence image not found: {path}")
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"cannot decode evidence image: {path}")
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                raise ValueError("could not encode the evidence image for OCR")
            return buf.tobytes(), (frame.shape[1], frame.shape[0])
        frame = np.asarray(image)
        if frame.size == 0:
            raise ValueError("empty image passed to OCR")
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("could not encode the frame for OCR")
        return buf.tobytes(), (frame.shape[1], frame.shape[0])

    @staticmethod
    def _words_inside(annotations: List[Any], roi: Tuple[int, int, int, int], size: Tuple[int, int]) -> str:
        """Join the OCR words whose bounding box lies inside the region of interest."""
        width, height = size
        x1, y1, x2, y2 = [float(v) for v in roi]
        words: List[str] = []
        for annotation in annotations:
            poly = getattr(annotation, "bounding_poly", None)
            vertices = list(getattr(poly, "vertices", None) or []) if poly is not None else []
            if not vertices:
                continue
            xs = [float(getattr(v, "x", 0) or 0) for v in vertices]
            ys = [float(getattr(v, "y", 0) or 0) for v in vertices]
            cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                text = (getattr(annotation, "description", "") or "").strip()
                if text:
                    words.append(text)
        return " ".join(words)


def _not_configured(message: str):
    from app.ai.providers.base import ProviderNotConfigured

    return ProviderNotConfigured(message)
