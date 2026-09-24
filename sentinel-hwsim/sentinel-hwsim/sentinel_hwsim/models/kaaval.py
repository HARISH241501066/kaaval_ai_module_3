"""
Simulated Kaaval AI processing node (RDK X5 edge box or analytics server).

Behaviour modelled:
  * node agent (always on while powered) vs Kaaval runtime (can crash / restart)
  * CPU, memory and temperature that respond to load, with thermal throttling
  * inference fps / latency derived from load, throttling and runtime state
  * Poisson event arrivals per camera for each enabled analytics function
  * store-and-forward outbox: if the event pipe is cut, events and heartbeats
    buffer on the node and flush (late, out of order) when the pipe returns

Chaos controls (crash_runtime, stall_inference, cut_pipe ...) are sim-only
methods and deliberately NOT part of the driver contract.
"""
from __future__ import annotations

import math
import random
import re
import uuid
from collections import deque
from datetime import timedelta
from typing import Any

from ..clock import SimClock

FUNCTION_EVENTS = {
    "ANPR": "PLATE_READ",
    "HELMET_DETECTION": "HELMET_VIOLATION",
    "TRIPLE_RIDING": "TRIPLE_RIDING",
    "VEHICLE_DETECTION": "VEHICLE_DETECTION",
    "WATCHLIST_MATCH": "WATCHLIST_HIT",
}
# relative frequency of each event type per camera (events per simulated minute)
EVENT_RATE_PER_MIN = {"PLATE_READ": 6.0, "HELMET_VIOLATION": 1.5, "TRIPLE_RIDING": 0.4,
                      "VEHICLE_DETECTION": 3.0, "WATCHLIST_HIT": 0.0}  # watchlist is derived from plate reads
GJ_RTO = ["01", "05", "06", "18", "03", "27"]  # Ahmedabad, Surat, Vadodara, Gandhinagar, Rajkot, Ahmedabad-East
MODEL_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


class KaavalNodeModel:
    RESTART_S = 3.0

    def __init__(self, node_id: str, clock: SimClock, rng: random.Random, *,
                 node_type: str = "KAAVAL_EDGE", hardware: str = "RDK X5",
                 cameras: list[str] | None = None,
                 functions: list[str] | None = None,
                 model_version: str = "kaaval-anpr-yolov8n-gj-2026.08",
                 analytics_version: str = "2.3.1",
                 ambient_c: float = 34.0, heartbeat_interval_s: float = 10.0,
                 event_rate_multiplier: float = 1.0, clock_offset_s: float = 0.0,
                 outbox_limit: int = 5000, malformed_event_rate: float = 0.0):
        self.node_id = node_id
        self.clock, self.rng = clock, rng
        self.node_type, self.hardware = node_type, hardware
        self.cameras = list(cameras or ["CAM-001", "CAM-002", "CAM-003", "CAM-004"])
        self.functions = list(functions or ["ANPR", "HELMET_DETECTION", "TRIPLE_RIDING", "WATCHLIST_MATCH"])
        unknown = set(self.functions) - FUNCTION_EVENTS.keys()
        if unknown:
            raise ValueError(f"unknown analytics functions: {unknown}")
        self.analytics_version = analytics_version
        self.model_version = model_version
        self.ambient_c = ambient_c
        self.heartbeat_interval_s = heartbeat_interval_s
        self.event_rate_multiplier = event_rate_multiplier
        self.clock_offset_s = clock_offset_s
        self.watchlist: set[str] = set()
        self.malformed_event_rate = malformed_event_rate  # firmware bugs: garbled individual events

        self.runtime = "RUNNING"
        self.inference_stalled = False
        self.pipe_connected = True
        self.powered = True
        self._restart_done_at: float | None = None
        self._pending_model: str | None = None
        self._model_loaded_at = clock.wall()

        self.cpu, self.mem, self.temp = 10.0, 30.0, ambient_c + 8
        self.outbox: deque[dict[str, Any]] = deque(maxlen=outbox_limit)
        self.outbox_dropped = 0
        self._next_heartbeat = clock.now()
        self._next_event = {(cam, et): clock.now() + self._interarrival(et)
                            for cam in self.cameras for et in self._event_types()}
        self._t = clock.now()

    # ----------------------------------------------------------- helpers
    def _event_types(self) -> list[str]:
        return [FUNCTION_EVENTS[f] for f in self.functions if EVENT_RATE_PER_MIN[FUNCTION_EVENTS[f]] > 0]

    def _interarrival(self, event_type: str) -> float:
        rate = EVENT_RATE_PER_MIN[event_type] * self.event_rate_multiplier / 60.0
        return self.rng.expovariate(rate) if rate > 0 else math.inf

    def _device_ts_ms(self, sim_t: float) -> int:
        wall = self.clock.wall() - timedelta(seconds=self.clock.now() - sim_t)
        return int((wall.timestamp() + self.clock_offset_s) * 1000)

    def _plate(self) -> str:
        r = self.rng
        letters = "".join(r.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(2))
        return f"GJ{r.choice(GJ_RTO)}{letters}{r.randrange(1, 10000):04d}"

    @property
    def inferencing(self) -> bool:
        return self.runtime == "RUNNING" and not self.inference_stalled

    @property
    def throttled(self) -> bool:
        return self.temp >= 85.0

    def _enqueue(self, item: dict[str, Any]) -> None:
        if len(self.outbox) == self.outbox.maxlen:
            self.outbox_dropped += 1  # oldest item falls off; real nodes lose data too
        self.outbox.append(item)

    # ----------------------------------------------------------- physics
    def _advance(self) -> None:
        now = self.clock.now()
        dt, t0 = max(0.0, now - self._t), self._t
        self._t = now
        if not self.powered:
            self.cpu, self.mem = 0.0, 0.0
            self.temp += (1 - math.exp(-dt / 60.0)) * (self.ambient_c - self.temp)
            self._next_heartbeat = max(self._next_heartbeat, now)
            for key, t_next in self._next_event.items():
                self._next_event[key] = max(t_next, now)
            return

        if self.runtime == "RESTARTING" and self._restart_done_at is not None and now >= self._restart_done_at:
            self.runtime = "RUNNING"
            if self._pending_model:
                self.model_version, self._pending_model = self._pending_model, None
                self._model_loaded_at = self.clock.wall()
            self._restart_done_at = None

        per_cam = 6 + 3 * len(self.functions)
        cpu_target = 8 + (per_cam * len(self.cameras) if self.inferencing else 0)
        cpu_target += 25 if self.runtime == "RESTARTING" else 0   # model load spike
        cpu_target = min(99.0, cpu_target)
        mem_target = 22 + (6 * len(self.cameras) if self.runtime != "CRASHED" else 0)
        a_fast, a_slow = 1 - math.exp(-dt / 3.0), 1 - math.exp(-dt / 60.0)
        self.cpu += a_fast * (cpu_target - self.cpu)
        self.mem += a_fast * (mem_target - self.mem)
        self.temp += a_slow * (self.ambient_c + 0.45 * self.cpu - self.temp)

        # events generated in the window (t0, now] while inference is healthy
        if self.inferencing:
            for (cam, et), t_next in list(self._next_event.items()):
                while t_next <= now:
                    if t_next > t0:
                        self._enqueue(self._make_event(cam, et, t_next))
                    t_next += self._interarrival(et)
                self._next_event[(cam, et)] = t_next
        else:
            # hold schedules in the future so no burst of back-dated events on recovery
            for key, t_next in self._next_event.items():
                if t_next <= now:
                    self._next_event[key] = now + self._interarrival(key[1])

        # the node agent heartbeats even when the runtime is down (reports its state)
        while self._next_heartbeat <= now:
            self._enqueue({"kind": "heartbeat", "node": self.node_id, "runtime": self.runtime,
                           "fps": self._inference_fps(), "ts_ms": self._device_ts_ms(self._next_heartbeat)})
            self._next_heartbeat += self.heartbeat_interval_s

    def _inference_fps(self) -> float:
        if not self.inferencing:
            return 0.0
        base = 15.0 * (0.6 if self.throttled else 1.0)
        return round(max(0.0, base + self.rng.gauss(0, 0.4)), 2)

    def _make_event(self, cam: str, event_type: str, sim_t: float) -> dict[str, Any]:
        r = self.rng
        conf = min(0.99, max(0.35, r.gauss(0.88 if not self.throttled else 0.74, 0.07)))
        data: dict[str, Any] = {"bbox": [r.randrange(0, 1800), r.randrange(0, 900), r.randrange(80, 300), r.randrange(40, 160)]}
        if event_type == "PLATE_READ":
            plate = self._plate()
            if self.watchlist and "WATCHLIST_MATCH" in self.functions and r.random() < 0.05:
                plate = r.choice(sorted(self.watchlist))
            data |= {"plate": plate, "vehicle_class": r.choice(["CAR", "TWO_WHEELER", "AUTO", "TRUCK", "BUS"]),
                     "colour": r.choice(["WHITE", "BLACK", "SILVER", "RED", "BLUE"])}
            if plate in self.watchlist:
                # a watchlist hit is emitted alongside the plate read, same timestamp
                self._enqueue({"kind": "event", "evt_id": str(uuid.UUID(int=r.getrandbits(128))), "node": self.node_id,
                               "cam": cam, "type": "WATCHLIST_HIT", "ts_ms": self._device_ts_ms(sim_t),
                               "conf": round(conf, 3), "data": {"plate": plate, "list": "STOLEN_VEHICLES"},
                               "model_ver": self.model_version})
        elif event_type == "HELMET_VIOLATION":
            data |= {"plate": self._plate() if r.random() < 0.7 else None, "riders": r.choice([1, 2])}
        elif event_type == "TRIPLE_RIDING":
            data |= {"plate": self._plate() if r.random() < 0.6 else None, "riders": 3}
        elif event_type == "VEHICLE_DETECTION":
            data |= {"vehicle_class": r.choice(["CAR", "TWO_WHEELER", "AUTO", "TRUCK"]), "count": r.randrange(1, 6)}
        event = {"kind": "event", "evt_id": str(uuid.UUID(int=r.getrandbits(128))), "node": self.node_id, "cam": cam,
                 "type": event_type, "ts_ms": self._device_ts_ms(sim_t), "conf": round(conf, 3), "data": data,
                 "model_ver": self.model_version}
        if r.random() < self.malformed_event_rate:
            victim = r.choice(["cam", "ts_ms", "conf", "evt_id"])
            event[victim] = str(event[victim]) if victim in ("ts_ms", "conf") else None
        return event

    # ------------------------------------------------------- vendor endpoints
    def info(self) -> dict[str, Any]:
        return {"node": self.node_id, "type": self.node_type, "hw": self.hardware, "vendor": "Kaaval",
                "sdk": self.analytics_version, "serial": f"KV-{self.node_id}", "agent": "kaaval-agent 1.4.0"}

    def capabilities(self) -> dict[str, Any]:
        return {"node": self.node_id, "functions": list(self.functions), "cameras": list(self.cameras)}

    def model_info(self) -> dict[str, Any]:
        self._advance()
        return {"provider": "Kaaval", "analytics_version": self.analytics_version,
                "model_version": self.model_version, "loaded_at_ms": int(self._model_loaded_at.timestamp() * 1000)}

    def health(self) -> dict[str, Any]:
        self._advance()
        fps = self._inference_fps()
        latency = 0.0 if fps == 0 else 38 * (1 + self.cpu / 200) * (1.5 if self.throttled else 1) + self.rng.gauss(0, 3)
        return {"node": self.node_id, "runtime": self.runtime,
                "inference": {"fps": fps, "latency_ms": round(max(0.0, latency), 1)},
                "system": {"cpu": round(min(100, max(0, self.cpu + self.rng.gauss(0, 1.5))), 1),
                           "mem": round(min(100, self.mem), 1), "temp_c": round(self.temp, 1),
                           "throttled": self.throttled},
                "ts_ms": self._device_ts_ms(self.clock.now())}

    def load_model(self, model_version: str) -> dict[str, Any]:
        if not MODEL_VERSION_RE.match(model_version):
            return {"error": {"code": "BAD_MODEL_VERSION", "message": f"invalid model version {model_version!r}"}}
        if self.runtime == "CRASHED":
            return {"error": {"code": "RUNTIME_DOWN", "message": "runtime crashed; restart first"}}
        self._advance()
        self._pending_model = model_version
        self.runtime, self._restart_done_at = "RESTARTING", self.clock.now() + self.RESTART_S
        return {"accepted": True, "restart_eta_s": self.RESTART_S}

    def drain(self, max_items: int = 500) -> dict[str, Any]:
        """Event pipe delivery. If the pipe is cut, nothing arrives; items stay buffered."""
        self._advance()
        if not (self.pipe_connected and self.powered):
            return {"items": []}
        items = [self.outbox.popleft() for _ in range(min(max_items, len(self.outbox)))]
        return {"items": items}

    # ------------------------------------------------ sim-only chaos controls
    def crash_runtime(self) -> None:
        self._advance()
        self.runtime, self._restart_done_at = "CRASHED", None

    def restart_runtime(self) -> None:
        self._advance()
        self.runtime, self._restart_done_at = "RESTARTING", self.clock.now() + self.RESTART_S

    def stall_inference(self, stalled: bool = True) -> None:
        self._advance()
        self.inference_stalled = stalled

    def cut_pipe(self, cut: bool = True) -> None:
        self._advance()
        self.pipe_connected = not cut

    def power(self, on: bool) -> None:
        """Power-cycle the box. Powering on reboots the agent and the runtime."""
        self._advance()
        if on and not self.powered:
            self.powered = True
            self.outbox.clear()  # RAM buffer is lost on power loss
            self.runtime, self._restart_done_at = "RESTARTING", self.clock.now() + self.RESTART_S
        elif not on:
            self.powered = False

    def set_watchlist(self, plates: set[str] | list[str]) -> None:
        self.watchlist = set(plates)
