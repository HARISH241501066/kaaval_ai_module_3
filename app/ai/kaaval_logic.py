"""
Kaaval AI decision logic: raw helmet detections -> confirmed HELMET_VIOLATION events.

This is the layer that stops a continuous violation from becoming four events:

    frame 100 -> WITHOUT_HELMET   \
    frame 102 -> WITHOUT_HELMET    >  ONE violation episode  ->  ONE Kaaval event
    frame 104 -> WITHOUT_HELMET    /
    frame 106 -> WITHOUT_HELMET  /

Rules (all configurable via AI_* settings):
  * a rider must be observed as WITHOUT_HELMET for AI_HELMET_CONFIRMATION_FRAMES
    nearby observations (matched by track id when tracking is on, otherwise
    spatially by IoU / centre distance) before an event is created;
  * WITH_HELMET observations on the same rider reset the episode;
  * after emitting, the episode never re-emits; a new episode on the same camera
    within AI_HELMET_EVENT_COOLDOWN_S video-seconds is suppressed (cooldown);
  * only detections >= AI_HELMET_MIN_CONFIDENCE feed the rule engine.

Motorcycle / rider association (prototype, isolated here):
  * if the selected model also yields MOTORCYCLE / PERSON boxes, the WITHOUT_HELMET
    head is associated to the best overlapping vehicle box — never asserted blindly;
  * the default two-class helmet model does NOT output motorcycle boxes, so
    vehicle_bbox stays None and the canonical object is a rider (documented
    limitation in the README: no helmet box is ever guaranteed to belong to a
    specific motorcycle).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.ai.config import AIConfig, get_ai_config
from app.ai.schemas import (
    MOTORCYCLE,
    PERSON,
    WITH_HELMET,
    WITHOUT_HELMET,
    ConfirmedViolation,
    Detection,
)

log = logging.getLogger("sentinel.m3.ai.kaaval")

_MAX_STATES = 64  # bound the per-camera state table

# --- association gate -------------------------------------------------------
# A rider observed at AI_INFERENCE_FPS moves a long way in video pixels between two samples,
# and that distance depends on the frame size, not on the size of the rider. On the 4K demo
# feeds a motorcyclist covers 130-160 px per sample while its head box is only ~60 px tall,
# so the original gate ("1.5 x the head box") could never link two consecutive observations:
# every sample opened a NEW episode, hits never reached AI_HELMET_CONFIRMATION_FRAMES and the
# whole camera reported zero violations. The gate is now the larger of:
#   * a multiple of the rider's own box  (small or cropped sources), and
#   * a fraction of the frame's long side (4K: ~230 px), because pixel motion is meaningless
#     without knowing the frame scale.
MOTION_GATE_BOX_FACTOR = 1.5
MOTION_GATE_FRAME_FRACTION = 0.06


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def _centre(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _same_region(a: Detection, b: Detection, gate_px: float) -> bool:
    """Are these two head boxes plausibly the same rider across inference samples?"""
    ax, ay = _centre(a.bbox)
    bx, by = _centre(b.bbox)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= gate_px


def _gate_px(det: Detection, frame_size: Optional[Tuple[int, int]] = None) -> float:
    """How far (pixels) a rider may travel between two samples and still be the same rider."""
    x1, y1, x2, y2 = det.bbox
    scale = max(x2 - x1, y2 - y1, 1)
    gate = MOTION_GATE_BOX_FACTOR * scale
    if frame_size:
        width, height = frame_size
        gate = max(gate, MOTION_GATE_FRAME_FRACTION * max(int(width), int(height)))
    return gate


@dataclass
class _RiderState:
    """One tracked violation episode (a rider region on this camera)."""
    key: str
    hits: int = 0
    misses: int = 0
    confirmed: bool = False
    emitted: bool = False
    last: Optional[Detection] = None
    last_ts: float = 0.0
    last_round: int = 0
    round_claimed: bool = False

    def to_dict(self) -> Dict:
        return {"key": self.key, "hits": self.hits, "confirmed": self.confirmed, "emitted": self.emitted}


class HelmetViolationLogic:
    """Stateful, per-camera (per-video) decision engine. Not thread-safe by design:
    one instance belongs to one processing job."""

    def __init__(self, camera_id: str, config: Optional[AIConfig] = None):
        self.camera_id = camera_id
        self.config = config or get_ai_config()
        self._states: Dict[str, _RiderState] = {}
        self._round = -1
        self.last_event_ts: Optional[float] = None      # video time of last emitted event on this camera
        self.suppressed = 0                              # episodes dropped by cooldown

    # ------------------------------------------------------------------ utils
    @property
    def miss_limit(self) -> int:
        return max(2, self.config.confirmation_frames)

    def active_states(self) -> List[Dict]:
        return [s.to_dict() for s in self._states.values()]

    def _match_state(self, det: Detection, frame_size: Optional[Tuple[int, int]] = None) -> Optional[_RiderState]:
        """Find the (still unclaimed) episode this detection belongs to: the same track id
        when the tracker provides one, otherwise the NEAREST previous head inside the gate."""
        candidates = [st for st in self._states.values() if not st.round_claimed and st.last is not None]
        if det.track_id is not None:
            for st in candidates:
                if st.last.track_id == det.track_id:
                    return st
        gate = _gate_px(det, frame_size)
        cx, cy = _centre(det.bbox)
        best, best_dist = None, None
        for st in candidates:
            sx, sy = _centre(st.last.bbox)
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            if dist > gate:
                continue
            if best_dist is None or dist < best_dist:
                best, best_dist = st, dist
        return best

    def _new_state(self, det: Detection) -> _RiderState:
        key = f"trk:{det.track_id}" if det.track_id is not None else f"loc:{self._round}:{det.center[0]:.0f},{det.center[1]:.0f}"
        st = _RiderState(key=key, last=det)
        self._states[key] = st
        return st

    def _age_states(self, timestamp: float) -> None:
        expire = [k for k, st in self._states.items()
                  if not st.round_claimed and (
                      st.misses >= self.miss_limit
                      or (timestamp - st.last_ts) > 10.0
                  )]
        for k in expire:
            del self._states[k]
        if len(self._states) > _MAX_STATES:  # hard bound for pathological scenes
            for k in sorted(self._states, key=lambda k: self._states[k].last_round)[: len(self._states) - _MAX_STATES]:
                del self._states[k]

    def _associate_vehicle(self, head: Detection, detections: List[Detection]) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
        """Prototype spatial association: only asserted when the model actually emitted
        MOTORCYCLE/PERSON boxes; otherwise (None, None) — never guess."""
        best_bbox, best_cls, best_score = None, None, 0.0
        for d in detections:
            if d.class_name not in (MOTORCYCLE, PERSON):
                continue
            # a rider box should contain or overlap the head box
            contains = (d.bbox[0] <= head.bbox[0] and d.bbox[1] <= head.bbox[1]
                        and d.bbox[2] >= head.bbox[2] and d.bbox[3] >= head.bbox[3])
            score = _iou(d.bbox, head.bbox) + (1.0 if contains else 0.0)
            if (contains or _iou(d.bbox, head.bbox) > 0.05) and score > best_score:
                best_bbox, best_cls, best_score = d.bbox, d.class_name, score
        return best_bbox, best_cls

    # ---------------------------------------------------------------- observe
    def observe(self, detections: List[Detection], frame_number: int, timestamp: float,
                frame_size: Optional[Tuple[int, int]] = None) -> List[ConfirmedViolation]:
        """
        Feed one inference observation; returns ZERO or MORE confirmed violations.
        Typical run: [] [] [] [violation] [] [] ... — one event per episode, not per frame.

        frame_size is the source frame as (width, height); it scales the association gate so
        riders stay linked across samples at any resolution. Optional so callers that only
        have detections still work (the gate then falls back to the box size alone).
        """
        self._round += 1
        for st in self._states.values():
            st.round_claimed = False

        without = [d for d in detections
                   if d.class_name == WITHOUT_HELMET and d.confidence >= self.config.helmet_min_confidence]
        with_helmet = [d for d in detections if d.class_name == WITH_HELMET]

        # 1. WITH_HELMET on a known rider closes (resets) its episode
        for d in with_helmet:
            st = self._match_state(d, frame_size)
            if st is not None:
                st.round_claimed = True
                log.debug("rider %s now WITH_HELMET at t=%.2fs — episode reset", st.key, timestamp)
                del self._states[st.key]

        # 2. WITHOUT_HELMET observations advance episodes
        for d in without:
            st = self._match_state(d, frame_size) or self._new_state(d)
            st.round_claimed = True
            st.hits += 1
            st.misses = 0
            st.last = d
            st.last_ts = timestamp
            st.last_round = self._round

        # 3. age unmatched states
        for st in self._states.values():
            if not st.round_claimed:
                st.misses += 1
        self._age_states(timestamp)

        # 4. confirm & emit
        violations: List[ConfirmedViolation] = []
        for st in self._states.values():
            if st.emitted or st.hits < self.config.confirmation_frames:
                continue
            st.confirmed = True
            st.emitted = True  # one emission per episode, always
            if self.last_event_ts is not None and (timestamp - self.last_event_ts) < self.config.cooldown_s:
                self.suppressed += 1
                log.info(
                    "confirmed violation on %s at t=%.2fs suppressed by cooldown "
                    "(%.2fs < %.1fs since last event)",
                    self.camera_id, timestamp, timestamp - self.last_event_ts, self.config.cooldown_s,
                )
                continue
            head = st.last
            assert head is not None
            vehicle_bbox, vehicle_cls = self._associate_vehicle(head, detections)
            violation = ConfirmedViolation(
                event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
                camera_id=self.camera_id,
                detection=head,
                confidence=head.confidence,
                frame_number=head.frame_number,
                timestamp=head.timestamp,
                track_id=head.track_id,
                confirmations=st.hits,
                vehicle_bbox=vehicle_bbox,
                vehicle_class=vehicle_cls,
            )
            self.last_event_ts = timestamp
            violations.append(violation)
            log.info(
                "HELMET_VIOLATION confirmed on %s: %s conf=%.2f frame=%d t=%.2fs (%d observations)",
                self.camera_id, violation.event_id, violation.confidence,
                violation.frame_number, violation.timestamp, st.hits,
            )
        return violations
