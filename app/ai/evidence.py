"""
Evidence frame generation for confirmed helmet violations.

One annotated JPEG per EVENT (never the whole video), stored locally as:

    evidence/<yyyy>/<mm>/<dd>/<event_id>.jpg

The frame carries: boxes + class labels + confidences, the event id, camera id,
frame number, video timestamp and the model name. Storage is deliberately a thin,
replaceable function: swapping the local directory for object storage later means
changing save_evidence_frame only — the canonical evidence_ref dict stays the same.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from app.ai.config import AIConfig
from app.ai.schemas import WITH_HELMET, WITHOUT_HELMET, ConfirmedViolation, Detection

log = logging.getLogger("sentinel.m3.ai.evidence")

_COLOR = {
    WITHOUT_HELMET: (0, 0, 255),     # red head = violation
    WITH_HELMET: (0, 200, 0),        # green head = compliant
}
_DEFAULT_COLOR = (255, 170, 0)


class EvidenceError(RuntimeError):
    """Evidence could not be written (permissions, IO, invalid frame)."""


def _put_label(img: np.ndarray, text: str, x: int, y: int, color) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    y0 = max(0, y - th - 6)
    cv2.rectangle(img, (x, y0), (x + tw + 4, y0 + th + 6), color, -1)
    cv2.putText(img, text, (x + 2, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: List[Detection],
                    violation: Optional[ConfirmedViolation] = None) -> np.ndarray:
    """Annotate a copy of the frame: all detections + the confirmed violation highlighted."""
    img = frame.copy()
    for d in detections:
        color = _COLOR.get(d.class_name, _DEFAULT_COLOR)
        x1, y1, x2, y2 = d.bbox
        is_hit = violation is not None and d.bbox == violation.detection.bbox
        thickness = 4 if is_hit else 2
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        label = f"{d.class_name} {d.confidence:.2f}"
        if is_hit:
            label = f"HELMET_VIOLATION {d.confidence:.2f}"
        _put_label(img, label, x1, max(0, y1 - 4), color)
    return img


def save_evidence_frame(
    frame: np.ndarray,
    detections: List[Detection],
    violation: ConfirmedViolation,
    config: AIConfig,
    model_name: Optional[str] = None,
) -> Optional[str]:
    """
    Write the annotated evidence frame for a confirmed violation.

    Returns the relative uri ('evidence/2026/09/27/EVT-....jpg') on success and None on
    failure — a missing evidence file must never kill the event that references it.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        log.error("evidence skipped for %s: empty frame", violation.event_id)
        return None
    try:
        now = datetime.now(timezone.utc)
        rel_dir = Path(config.evidence_dir) / f"{now.year:04d}" / f"{now.month:02d}" / f"{now.day:02d}"
        rel_dir.mkdir(parents=True, exist_ok=True)
        rel_path = rel_dir / f"{violation.event_id}.jpg"

        img = draw_detections(frame, detections, violation)
        h, w = img.shape[:2]
        banner = (
            f"{violation.camera_id}  {violation.event_id}  frame {violation.frame_number}  "
            f"t={violation.timestamp:.2f}s  conf={violation.confidence:.2f}  {model_name or ''}"
        ).strip()
        cv2.rectangle(img, (0, 0), (w, 34), (0, 0, 0), -1)
        cv2.putText(img, banner, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        ok = cv2.imwrite(str(rel_path), img)
        if not ok:
            log.error("cv2.imwrite failed for evidence frame %s", rel_path)
            return None
        uri = rel_path.as_posix()
        log.info("evidence frame saved: %s", uri)
        return uri
    except PermissionError as e:
        log.error("no permission to write evidence dir '%s': %s", config.evidence_dir, e)
        return None
    except OSError as e:
        log.error("could not write evidence frame for %s: %s", violation.event_id, e)
        return None
