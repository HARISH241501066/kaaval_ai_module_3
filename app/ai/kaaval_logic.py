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

# --- unique violator registry -------------------------------------------------
# A rider seen WITHOUT_HELMET in 30 consecutive frames is ONE unique violator, not 30.
# The registry below keeps one record per tracked rider for the whole video, so the job can
# report `unique_violators` (people) separately from `violation_detections` (frames).
# Spatial matching (for detectors/trackers without ids) only links observations that are
# both nearby (the same motion gate as episodes) and recent, so two different riders that
# pass the same spot minutes apart are never merged.
VIOLATOR_RECENCY_S = 2.0          # video seconds within which a nearby observation is the same rider
ID_SWITCH_ABSORB_FACTOR = 0.5     # a NEW track id this close to a recent record is an id switch, not a 2nd rider
# One evidence image per unique rider, from that rider's best frame. Re-rendering only a
# materially better observation (and never more than EVIDENCE_MAX_RENDERS times) keeps a 4K
# frame from being re-encoded for every 0.001 confidence wobble while still tracking the peak.
EVIDENCE_IMPROVEMENT_MARGIN = 0.05
EVIDENCE_MAX_RENDERS = 6

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


@dataclass
class ViolatorRecord:
    """
    One UNIQUE rider observed WITHOUT_HELMET anywhere in the video.

    This is the structure that makes "4 riders" different from "137 detections":
        first_seen / last_seen      video seconds of the first / last violation observation
        observations                frames this rider was seen without a helmet
        violation_detections        frame-level detections attributed to this rider
        streak / confirmations      consecutive observations at confirmation time
        confirmed                   temporal confirmation rule satisfied
        max_confidence / best_detection   the representative (highest-confidence) frame
        evidence_uri                the single evidence JPEG written for this rider
        emitted                     one Kaaval event already produced for this rider
    """

    key: str
    camera_id: str
    event_id: str
    track_id: Optional[int] = None
    first_seen: float = 0.0
    last_seen: float = 0.0
    last_frame: int = 0
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    observations: int = 0
    violation_detections: int = 0
    streak: int = 0
    confirmed: bool = False
    confirmed_at: float = 0.0
    confirmations: int = 0
    max_confidence: float = 0.0
    best_detection: Optional[Detection] = None
    best_frame_number: int = -1
    evidence_uri: Optional[str] = None
    evidence_confidence: float = -1.0
    evidence_renders: int = 0
    emitted: bool = False

    @property
    def needs_evidence(self) -> bool:
        """A confirmed rider whose representative frame should be rendered now.

        Rendered once at confirmation, then only for a materially better observation, and at
        most EVIDENCE_MAX_RENDERS times per rider — always ONE file per rider (overwritten),
        never one file per frame.
        """
        if not self.confirmed or self.best_detection is None:
            return False
        if self.evidence_uri is None:
            return True
        return (self.max_confidence >= self.evidence_confidence + EVIDENCE_IMPROVEMENT_MARGIN
                and self.evidence_renders < EVIDENCE_MAX_RENDERS)

    def to_dict(self) -> Dict:
        return {
            "key": self.key,
            "camera_id": self.camera_id,
            "event_id": self.event_id,
            "track_id": self.track_id,
            "first_seen": round(self.first_seen, 3),
            "last_seen": round(self.last_seen, 3),
            "observations": self.observations,
            "violation_detections": self.violation_detections,
            "confirmed": self.confirmed,
            "confirmations": self.confirmations,
            "max_confidence": round(self.max_confidence, 4),
            "best_frame_number": self.best_frame_number,
            "evidence_uri": self.evidence_uri,
            "evidence_renders": self.evidence_renders,
            "emitted": self.emitted,
        }


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
        # Whole-video registry: one record per unique rider that violated the helmet rule.
        self.violators: Dict[str, ViolatorRecord] = {}
        self.violation_detections = 0                    # frame-level WITHOUT_HELMET detections
        self._violator_seq = 0

    # ------------------------------------------------------------------ utils
    @property
    def miss_limit(self) -> int:
        return max(2, self.config.confirmation_frames)

    @property
    def unique_violators(self) -> int:
        """Distinct riders confirmed WITHOUT_HELMET during this video (the primary metric)."""
        return sum(1 for rec in self.violators.values() if rec.confirmed)

    def active_states(self) -> List[Dict]:
        return [s.to_dict() for s in self._states.values()]

    def violators_summary(self) -> List[Dict]:
        return [rec.to_dict() for rec in sorted(self.violators.values(), key=lambda r: r.first_seen)]

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

    # ------------------------------------------------- unique violator registry
    def _new_violator(self, det: Detection) -> ViolatorRecord:
        self._violator_seq += 1
        key = (f"trk:{det.track_id}" if det.track_id is not None
               else f"loc:{det.center[0]:.0f},{det.center[1]:.0f}#{self._violator_seq}")
        rec = ViolatorRecord(
            key=key,
            camera_id=self.camera_id,
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
            track_id=det.track_id,
            first_seen=det.timestamp,
            last_seen=det.timestamp,
        )
        self.violators[key] = rec
        return rec

    def _match_violator(self, det: Detection, timestamp: float,
                        frame_size: Optional[Tuple[int, int]] = None) -> Optional[ViolatorRecord]:
        """Find the unique-rider record this observation belongs to.

        * same track id                       -> the same rider (the common path)
        * no id on either side, nearby+recent -> same rider (bbox trackers/fallback)
        * a NEW id appearing very close to a recent record -> a tracker id switch on the
          same rider (probable), absorbed instead of counted as a second violator.
        """
        if det.track_id is not None:
            for rec in self.violators.values():
                if rec.track_id == det.track_id:
                    return rec
        gate = _gate_px(det, frame_size)
        cx, cy = _centre(det.bbox)
        best, best_dist = None, None
        for rec in self.violators.values():
            if rec.last_bbox is None or (timestamp - rec.last_seen) > VIOLATOR_RECENCY_S:
                continue
            sx, sy = _centre(rec.last_bbox)
            dist = ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
            limit = gate if (rec.track_id is None or det.track_id is None) \
                else ID_SWITCH_ABSORB_FACTOR * gate
            if dist > limit:
                continue
            if best_dist is None or dist < best_dist:
                best, best_dist = rec, dist
        return best

    def _update_violators(self, without: List[Detection], with_helmet_observations: List[Detection],
                          timestamp: float, frame_size: Optional[Tuple[int, int]] = None) -> None:
        """Maintain the whole-video unique-violator registry (counting, not event emission)."""
        # A helmet observation closes the pending streak but never un-counts a confirmed rider:
        # "frame 100 without, frame 101-102 with helmet" must NOT become a violation.
        for d in with_helmet_observations:
            rec = self._match_violator(d, timestamp, frame_size)
            if rec is not None and not rec.confirmed:
                rec.streak = 0

        for d in without:
            rec = self._match_violator(d, timestamp, frame_size) or self._new_violator(d)
            if rec.track_id is None and d.track_id is not None:
                rec.track_id = d.track_id          # absorbed an id switch: keep the rider
            rec.observations += 1
            rec.violation_detections += 1
            rec.streak += 1
            rec.last_seen = d.timestamp
            rec.last_frame = d.frame_number
            rec.last_bbox = d.bbox
            self.violation_detections += 1
            # Representative frame: the highest-confidence observation; the confirming frame
            # seeds it when the earlier (pre-confirmation) observations are already gone.
            if d.confidence > rec.max_confidence or (rec.best_detection is None and rec.confirmed):
                rec.max_confidence = max(rec.max_confidence, float(d.confidence))
                rec.best_detection = d
                rec.best_frame_number = d.frame_number
            if not rec.confirmed and rec.streak >= self.config.confirmation_frames:
                rec.confirmed = True
                rec.confirmed_at = d.timestamp
                rec.confirmations = rec.streak
                if rec.best_detection is None or rec.best_detection.frame_number != d.frame_number:
                    # the best pre-confirmation frame is no longer in memory: use this one
                    rec.max_confidence = max(rec.max_confidence, float(d.confidence))
                    rec.best_detection = d
                    rec.best_frame_number = d.frame_number
                log.info(
                    "unique violator confirmed on %s: %s (track=%s) after %d observations at t=%.2fs",
                    self.camera_id, rec.event_id, rec.track_id, rec.confirmations, d.timestamp,
                )

    def evidence_candidates(self) -> List[Tuple[ViolatorRecord, Detection]]:
        """Confirmed riders whose best frame still needs to be rendered as evidence.

        Always called right after `observe()` on the SAME frame, so the returned detection is
        guaranteed to belong to the frame currently held by the caller — evidence is written
        from the frame being processed, never from a stored one (no frame buffering).
        """
        return [(rec, rec.best_detection) for rec in sorted(self.violators.values(), key=lambda r: r.first_seen)
                if rec.needs_evidence and rec.best_detection is not None
                and rec.best_detection.frame_number == rec.last_frame]

    def mark_evidence(self, record: ViolatorRecord, uri: Optional[str]) -> None:
        """Remember the rendered evidence for a rider (keeps the best-confidence frame)."""
        if uri:
            record.evidence_uri = uri
        record.evidence_confidence = record.max_confidence
        record.evidence_renders += 1

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
                "violation episode confirmed on %s (streaming tracker): %s conf=%.2f frame=%d "
                "t=%.2fs (%d observations) — counted once per unique rider at finalization",
                self.camera_id, violation.event_id, violation.confidence,
                violation.frame_number, violation.timestamp, st.hits,
            )

        # 5. whole-video unique-violator registry (one record per rider, counted once)
        self._update_violators(without, with_helmet, timestamp, frame_size)
        return violations

    # ------------------------------------------------------------- finalization
    def finalize_violations(self) -> List[ConfirmedViolation]:
        """
        End-of-video aggregation: ONE ConfirmedViolation per UNIQUE confirmed violator.

        This is what turns

            track 7   frames 100..200   ->  one violator
            track 12  frames 180..260   ->  one violator

        into "2 unique helmet violators" and exactly two Kaaval events, however many frames
        each rider was visible for. Called by the processor after the last frame; emits each
        rider at most once for the whole video.
        """
        out: List[ConfirmedViolation] = []
        for rec in sorted(self.violators.values(), key=lambda r: r.first_seen):
            if not rec.confirmed or rec.emitted:
                continue
            rec.emitted = True
            det = rec.best_detection
            if det is None:                                    # defensive: no frame kept
                det = Detection(class_name=WITHOUT_HELMET, confidence=rec.max_confidence,
                                bbox=rec.last_bbox or (0, 0, 0, 0),
                                frame_number=rec.last_frame, timestamp=rec.last_seen,
                                track_id=rec.track_id)
            out.append(ConfirmedViolation(
                event_id=rec.event_id,
                camera_id=self.camera_id,
                detection=det,
                confidence=rec.max_confidence or det.confidence,
                frame_number=det.frame_number,
                timestamp=det.timestamp,
                track_id=rec.track_id,
                confirmations=rec.confirmations or rec.observations,
                evidence_uri=rec.evidence_uri,
                first_seen=rec.first_seen,
                last_seen=rec.last_seen,
                observations=rec.observations,
                violation_detections=rec.violation_detections,
            ))
        return out
