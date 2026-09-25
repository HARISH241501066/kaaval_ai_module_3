"""
Turn per-frame detections into events, once per real-world object.

Without tracking, one parked car would raise an event on every sampled frame. The tracker follows
each object across frames and emits:

* VEHICLE_DETECTION  once per vehicle, after it has been seen for `min_hits` sampled frames.
* TRIPLE_RIDING      once per motorbike that carries three or more people for `min_hits` frames.

Helmet and number-plate events need dedicated models (a helmet classifier, a plate detector and
OCR); this general-purpose COCO model cannot produce them, so this tracker does not invent them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.live.detector import VEHICLES, Detection


@dataclass
class Event:
    type: str
    conf: float
    box: tuple[int, int, int, int]
    track_id: int
    data: dict = field(default_factory=dict)


@dataclass
class _Track:
    id: int
    label: str
    box: tuple[int, int, int, int]
    confs: list[float]
    hits: int = 1
    missed: int = 0
    vehicle_sent: bool = False
    triple_hits: int = 0
    triple_sent: bool = False
    max_riders: int = 0


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw, ih = min(ax2, bx2) - max(a[0], b[0]), min(ay2, by2) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def _near(a, b) -> bool:
    """Same-size box whose centre moved less than ~0.8 of its own size (fast vehicles between samples)."""
    acx, acy, bcx, bcy = a[0] + a[2] / 2, a[1] + a[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    size = max(a[2], a[3])
    ratio = (b[2] * b[3]) / max(1, a[2] * a[3])
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 < 0.8 * size and 0.5 <= ratio <= 2.0


class Tracker:
    def __init__(self, min_hits: int = 3, max_missed: int = 4):
        self.min_hits, self.max_missed = min_hits, max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1                 # never reset, so event ids stay unique for the whole run

    def reset(self) -> None:
        """Feed looped or restarted: old objects are gone. Emits nothing."""
        self._tracks.clear()

    def update(self, detections: list[Detection]) -> list[Event]:
        vehicles = [d for d in detections if d.label in VEHICLES]
        people = [d for d in detections if d.label == "person"]
        unmatched = list(vehicles)
        for t in self._tracks:
            best, best_score = None, 0.0
            for d in unmatched:
                if d.label != t.label:
                    continue
                score = _iou(t.box, d.box) or (0.01 if _near(t.box, d.box) else 0.0)
                if score > best_score:
                    best, best_score = d, score
            if best is not None and best_score >= 0.01:
                unmatched.remove(best)
                t.box, t.hits, t.missed = best.box, t.hits + 1, 0
                t.confs.append(best.conf)
            else:
                t.missed += 1
        for d in unmatched:
            self._tracks.append(_Track(self._next_id, d.label, d.box, [d.conf]))
            self._next_id += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]

        events: list[Event] = []
        for t in self._tracks:
            if t.missed:
                continue
            conf = round(sum(t.confs) / len(t.confs), 3)
            if t.hits >= self.min_hits and not t.vehicle_sent:
                t.vehicle_sent = True
                events.append(Event("VEHICLE_DETECTION", conf, t.box, t.id, {"vehicle_class": VEHICLES[t.label]}))
            if t.label == "motorbike" and not t.triple_sent:
                riders = self._riders(t.box, people)
                t.max_riders = max(t.max_riders, riders)
                t.triple_hits = t.triple_hits + 1 if riders >= 3 else 0
                if t.triple_hits >= self.min_hits:
                    t.triple_sent = True
                    events.append(Event("TRIPLE_RIDING", conf, t.box, t.id,
                                        {"vehicle_class": "TWO_WHEELER", "riders": t.max_riders}))
        return events

    @staticmethod
    def _riders(bike: tuple[int, int, int, int], people: list[Detection]) -> int:
        """People whose centre sits on the motorbike, allowing for riders sitting above it."""
        x, y, w, h = bike
        count = 0
        for p in people:
            cx, cy = p.box[0] + p.box[2] / 2, p.box[1] + p.box[3] / 2
            if x - 0.15 * w <= cx <= x + 1.15 * w and y - 1.2 * h <= cy <= y + 0.9 * h:
                count += 1
        return count
