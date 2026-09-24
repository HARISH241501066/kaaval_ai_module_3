"""
Configurable latency and fault injection.

Three ways to cause failures, from least to most deterministic:
  1. Probabilistic rates in FaultProfile (soak / resilience testing)
  2. One-shot faults queued with FaultInjector.inject_next() (unit tests)
  3. Sticky conditions set with FaultInjector.set_condition() (demo chaos:
     "device offline", "credentials revoked") that hold until cleared
"""
from __future__ import annotations

import enum
import random
from collections import deque
from dataclasses import dataclass, field, replace


class Fault(str, enum.Enum):
    TIMEOUT = "timeout"                # no response within the driver deadline
    CONNECT_FAILURE = "connect_failure"  # refused / unreachable on connect
    AUTH_FAILURE = "auth_failure"      # 401/403
    INVALID_DATA = "invalid_data"      # malformed / wrong-typed payload
    DROP_CONNECTION = "drop_connection"  # session dies mid-call


_CONNECT_FAULTS = {Fault.CONNECT_FAILURE, Fault.AUTH_FAILURE, Fault.TIMEOUT}
_CALL_FAULTS = {Fault.TIMEOUT, Fault.DROP_CONNECTION, Fault.AUTH_FAILURE, Fault.INVALID_DATA}


class Condition(str, enum.Enum):
    OFFLINE = "offline"                # device unreachable: connects fail, calls time out
    AUTH_REVOKED = "auth_revoked"      # every call returns auth failure


@dataclass(frozen=True)
class LatencyProfile:
    base_ms: float = 80.0
    jitter_ms: float = 40.0
    spike_probability: float = 0.02   # occasional slow responses (network congestion)
    spike_ms: float = 1500.0

    def sample(self, rng: random.Random) -> float:
        ms = max(0.0, rng.gauss(self.base_ms, self.jitter_ms / 2))
        if rng.random() < self.spike_probability:
            ms += rng.uniform(0.5, 1.0) * self.spike_ms
        return ms / 1000.0  # simulated seconds


@dataclass(frozen=True)
class FaultProfile:
    latency: LatencyProfile = field(default_factory=LatencyProfile)
    timeout_rate: float = 0.0
    connect_failure_rate: float = 0.0
    auth_failure_rate: float = 0.0
    invalid_data_rate: float = 0.0
    drop_connection_rate: float = 0.0
    connect_latency_s: float = 0.4   # handshake (TCP + ONVIF/TLS) cost


PRESETS: dict[str, FaultProfile] = {
    "ideal": FaultProfile(latency=LatencyProfile(0, 0, 0, 0), connect_latency_s=0),
    "healthy": FaultProfile(),
    "flaky": FaultProfile(
        latency=LatencyProfile(base_ms=150, jitter_ms=120, spike_probability=0.08, spike_ms=3000),
        timeout_rate=0.05, invalid_data_rate=0.02, drop_connection_rate=0.02, connect_failure_rate=0.1,
    ),
    "hostile": FaultProfile(
        latency=LatencyProfile(base_ms=400, jitter_ms=300, spike_probability=0.2, spike_ms=6000),
        timeout_rate=0.2, invalid_data_rate=0.1, drop_connection_rate=0.1,
        connect_failure_rate=0.4, auth_failure_rate=0.05,
    ),
}


def preset(name: str, **overrides) -> FaultProfile:
    try:
        base = PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None
    return replace(base, **overrides) if overrides else base


class FaultInjector:
    def __init__(self, profile: FaultProfile, rng: random.Random):
        self.profile = profile
        self._rng = rng
        self._queued: deque[Fault] = deque()
        self._conditions: set[Condition] = set()

    # -- control surface ---------------------------------------------------- #
    def inject_next(self, fault: Fault | str, count: int = 1) -> None:
        self._queued.extend([Fault(fault)] * count)

    def set_condition(self, condition: Condition | str, active: bool = True) -> None:
        c = Condition(condition)
        (self._conditions.add if active else self._conditions.discard)(c)

    def has(self, condition: Condition) -> bool:
        return condition in self._conditions

    def clear(self) -> None:
        self._queued.clear()
        self._conditions.clear()

    # -- used by the transport ---------------------------------------------- #
    def latency(self) -> float:
        return self.profile.latency.sample(self._rng)

    def roll(self, *, connecting: bool) -> Fault | None:
        if self._queued and self._queued[0] in (_CONNECT_FAULTS if connecting else _CALL_FAULTS):
            return self._queued.popleft()
        if self.has(Condition.OFFLINE):
            return Fault.CONNECT_FAILURE if connecting else Fault.TIMEOUT
        if self.has(Condition.AUTH_REVOKED):
            return Fault.AUTH_FAILURE
        p, r = self.profile, self._rng.random
        if connecting:
            if r() < p.connect_failure_rate:
                return Fault.CONNECT_FAILURE
            if r() < p.auth_failure_rate:
                return Fault.AUTH_FAILURE
            return None
        for fault, rate in (
            (Fault.TIMEOUT, p.timeout_rate),
            (Fault.DROP_CONNECTION, p.drop_connection_rate),
            (Fault.AUTH_FAILURE, p.auth_failure_rate),
            (Fault.INVALID_DATA, p.invalid_data_rate),
        ):
            if r() < rate:
                return fault
        return None
