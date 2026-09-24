"""
Simulated transport ("the wire") between a driver and a device model.

Every request goes through SimTransport.call(), which applies realistic latency,
enforces the driver's deadline, applies injected faults, and records a trace
entry. Device models never know about faults; drivers never know about models.
"""
from __future__ import annotations

import copy
import itertools
import logging
import random
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .clock import SimClock
from .contract import AuthFailed, ConnectionFailed, DeviceDisconnected, DeviceTimeout
from .faults import Fault, FaultInjector

log = logging.getLogger("sentinel_hwsim.wire")
_req_ids = itertools.count(1)


@dataclass(frozen=True)
class TraceEntry:
    request_id: int
    at: datetime
    device_id: str
    op: str
    params: dict[str, Any] | None
    latency_ms: float
    outcome: str  # "OK" or a Fault value

    def __str__(self) -> str:
        p = f" {self.params}" if self.params else ""
        return (f"#{self.request_id:<5} {self.at:%H:%M:%S.%f}"[:-3]
                + f" {self.device_id:<16} {self.op:<18}{p} {self.latency_ms:7.1f}ms {self.outcome}")


def _stringify(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _stringify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_stringify(v) for v in value]
    return value if isinstance(value, str) else str(value)


def corrupt(payload: Any, rng: random.Random) -> Any:
    """
    Damage a vendor payload the way real firmware and middleboxes do:
      * garbage_body   - proxy/NVR returns an HTML error page instead of data
      * empty_object   - firmware returns {} under load
      * stringly_typed - every scalar arrives as a string (classic XML-to-dict bug)
                         AND one top-level section is missing (truncated response)
    Each strategy is guaranteed to break a strict mapping that reads the payload.
    """
    strategy = rng.choice(["garbage_body", "empty_object", "stringly_typed"])
    if strategy == "garbage_body" or not isinstance(payload, dict) or not payload:
        return "<html><body><h1>502 Bad Gateway</h1></body></html>"
    if strategy == "empty_object":
        return {}
    data = _stringify(copy.deepcopy(payload))
    del data[rng.choice(sorted(data))]
    return data


class SimTransport:
    def __init__(self, device_id: str, clock: SimClock, injector: FaultInjector,
                 rng: random.Random, *, deadline_s: float = 5.0, trace_size: int = 1000):
        self.device_id = device_id
        self.clock = clock
        self.faults = injector
        self.deadline_s = deadline_s
        self._rng = rng
        self.trace: deque[TraceEntry] = deque(maxlen=trace_size)

    def _record(self, rid: int, op: str, params: dict | None, latency: float, outcome: str) -> None:
        entry = TraceEntry(rid, self.clock.wall(), self.device_id, op, params, latency * 1000, outcome)
        self.trace.append(entry)
        (log.debug if outcome == "OK" else log.warning)("%s", entry)

    async def handshake(self) -> None:
        rid = next(_req_ids)
        fault = self.faults.roll(connecting=True)
        latency = self.faults.profile.connect_latency_s + self.faults.latency()
        if fault in (Fault.CONNECT_FAILURE, Fault.TIMEOUT):
            wait = min(self.deadline_s, latency * 3)
            await self.clock.sleep(wait)
            self._record(rid, "connect", None, wait, Fault.CONNECT_FAILURE.value)
            raise ConnectionFailed("device unreachable or refused connection", device_id=self.device_id)
        await self.clock.sleep(latency)
        if fault is Fault.AUTH_FAILURE:
            self._record(rid, "connect", None, latency, fault.value)
            raise AuthFailed("401 Unauthorized during session setup", device_id=self.device_id)
        self._record(rid, "connect", None, latency, "OK")

    async def call(self, op: str, handler: Callable[[], Any], params: dict | None = None) -> Any:
        rid = next(_req_ids)
        fault = self.faults.roll(connecting=False)
        latency = self.faults.latency()

        if fault is Fault.TIMEOUT or latency > self.deadline_s:
            await self.clock.sleep(self.deadline_s)
            self._record(rid, op, params, self.deadline_s, Fault.TIMEOUT.value)
            raise DeviceTimeout(f"{op}: no response within {self.deadline_s}s", device_id=self.device_id)

        await self.clock.sleep(latency)

        if fault is Fault.DROP_CONNECTION:
            self._record(rid, op, params, latency, fault.value)
            raise DeviceDisconnected(f"{op}: connection reset by peer", device_id=self.device_id)
        if fault is Fault.AUTH_FAILURE:
            self._record(rid, op, params, latency, fault.value)
            raise AuthFailed(f"{op}: 401 Unauthorized", device_id=self.device_id)

        payload = handler()  # device-side logic runs only if the request "arrived"
        if fault is Fault.INVALID_DATA:
            payload = corrupt(payload, self._rng)
        vendor_fault = isinstance(payload, dict) and ("Fault" in payload or "error" in payload)
        self._record(rid, op, params, latency, fault.value if fault else ("VENDOR_FAULT" if vendor_fault else "OK"))
        return payload
