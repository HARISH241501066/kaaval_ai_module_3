"""
The swap point between simulated and real hardware.

Application code calls create_driver(config). Whether it receives a mock or a
real driver is decided by config.mode or the M3_DRIVER_MODE env var:

    M3_DRIVER_MODE=mock   -> simulated devices (default)
    M3_DRIVER_MODE=real   -> OnvifCameraDriver / KaavalHttpDriver
"""
from __future__ import annotations

import os
import random
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .clock import SimClock
from .contract import AINodeDriver, CameraDriver
from .drivers.mock_camera import MockCameraDriver
from .drivers.mock_kaaval import MockKaavalDriver
from .drivers.real import CredentialProvider, KaavalHttpDriver, OnvifCameraDriver
from .faults import Condition, Fault, FaultInjector, FaultProfile, preset
from .models.camera import CameraModel
from .models.kaaval import KaavalNodeModel
from .transport import SimTransport


@dataclass
class DriverConfig:
    kind: str                               # "camera" | "ai_node"
    device_id: str
    mode: str | None = None                 # "mock" | "real"; None -> env M3_DRIVER_MODE
    endpoint: str | None = None             # real only: host / base URL
    credential_ref: str | None = None       # real only: C7 reference, e.g. "vault:secret/m3/camera/CAM-001"
    deadline_s: float = 5.0
    sim: dict[str, Any] = field(default_factory=dict)  # mock only: fault preset, model params

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DriverConfig:
        return cls(**d)


class Simulator:
    """
    Owns the shared clock and deterministic RNG streams for a fleet of simulated
    devices. One Simulator per test / demo session.
    """

    def __init__(self, *, speed: float = 1.0, seed: int = 42):
        self.clock = SimClock(speed=speed)
        self.seed = seed
        self.devices: dict[str, MockCameraDriver | MockKaavalDriver] = {}

    def _rng(self, device_id: str, stream: str) -> random.Random:
        # stable per-device streams: adding a device never changes another device's behaviour
        return random.Random(zlib.crc32(f"{self.seed}:{device_id}:{stream}".encode()))

    def _transport(self, device_id: str, faults: FaultProfile | str, deadline_s: float) -> SimTransport:
        profile = preset(faults) if isinstance(faults, str) else faults
        injector = FaultInjector(profile, self._rng(device_id, "faults"))
        return SimTransport(device_id, self.clock, injector, self._rng(device_id, "wire"), deadline_s=deadline_s)

    def camera(self, device_id: str, *, faults: FaultProfile | str = "healthy",
               deadline_s: float = 5.0, **model_kwargs) -> MockCameraDriver:
        model = CameraModel(device_id, self.clock, self._rng(device_id, "model"), **model_kwargs)
        drv = MockCameraDriver(model, self._transport(device_id, faults, deadline_s))
        self.devices[device_id] = drv
        return drv

    def kaaval_node(self, node_id: str, *, faults: FaultProfile | str = "healthy",
                    deadline_s: float = 5.0, **model_kwargs) -> MockKaavalDriver:
        model = KaavalNodeModel(node_id, self.clock, self._rng(node_id, "model"), **model_kwargs)
        drv = MockKaavalDriver(model, self._transport(node_id, faults, deadline_s))
        self.devices[node_id] = drv
        return drv

    def controls(self, device_id: str) -> SimControls:
        return SimControls(self.devices[device_id])


class SimControls:
    """
    Chaos / scenario controls for one simulated device. Deliberately separate
    from the driver contract: if app code needs this, it is test code.
    """

    def __init__(self, driver: MockCameraDriver | MockKaavalDriver):
        if not getattr(driver, "is_simulated", False):
            raise TypeError("SimControls only works with simulated drivers")
        self.driver = driver
        self.faults: FaultInjector = driver.faults

    # generic, any device
    def fail_next(self, fault: Fault | str, count: int = 1) -> None:
        self.faults.inject_next(fault, count)

    def offline(self, on: bool = True) -> None:
        self.faults.set_condition(Condition.OFFLINE, on)
        if isinstance(self.driver, MockKaavalDriver):
            self.driver.sim.power(not on)

    def revoke_credentials(self, on: bool = True) -> None:
        self.faults.set_condition(Condition.AUTH_REVOKED, on)

    def heal(self) -> None:
        self.offline(False)
        self.faults.clear()

    # AI-node specific (the C6 failure layers)
    def _node(self) -> KaavalNodeModel:
        if not isinstance(self.driver, MockKaavalDriver):
            raise TypeError(f"{self.driver.device_id} is not an AI node")
        return self.driver.sim

    def crash_runtime(self) -> None: self._node().crash_runtime()
    def restart_runtime(self) -> None: self._node().restart_runtime()
    def stall_inference(self, on: bool = True) -> None: self._node().stall_inference(on)
    def cut_pipe(self, on: bool = True) -> None: self._node().cut_pipe(on)
    def set_watchlist(self, plates: list[str]) -> None: self._node().set_watchlist(plates)

    def set_ambient(self, celsius: float) -> None:
        self._node().ambient_c = celsius


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
Builder = Callable[..., Any]
_REGISTRY: dict[tuple[str, str], Builder] = {}
_default_sim: Simulator | None = None


def register_driver(kind: str, mode: str, builder: Builder) -> None:
    """Plug in another implementation (e.g. a Hikvision ISAPI driver) without touching callers."""
    _REGISTRY[(kind, mode)] = builder


def default_simulator() -> Simulator:
    global _default_sim
    if _default_sim is None:
        _default_sim = Simulator(speed=float(os.getenv("M3_SIM_SPEED", "1")), seed=int(os.getenv("M3_SIM_SEED", "42")))
    return _default_sim


def _mock_camera(cfg: DriverConfig, *, simulator: Simulator | None = None, **_) -> MockCameraDriver:
    sim = simulator or default_simulator()
    opts = dict(cfg.sim)
    return sim.camera(cfg.device_id, faults=opts.pop("faults", "healthy"), deadline_s=cfg.deadline_s, **opts)


def _mock_node(cfg: DriverConfig, *, simulator: Simulator | None = None, **_) -> MockKaavalDriver:
    sim = simulator or default_simulator()
    opts = dict(cfg.sim)
    return sim.kaaval_node(cfg.device_id, faults=opts.pop("faults", "healthy"), deadline_s=cfg.deadline_s, **opts)


def _real(cls):
    def build(cfg: DriverConfig, *, credentials: CredentialProvider | None = None, **_):
        if not (cfg.endpoint and cfg.credential_ref and credentials):
            raise ValueError(f"real {cfg.kind} driver needs endpoint, credential_ref and a CredentialProvider")
        return cls(cfg.device_id, cfg.endpoint, cfg.credential_ref, credentials, deadline_s=cfg.deadline_s)
    return build


register_driver("camera", "mock", _mock_camera)
register_driver("ai_node", "mock", _mock_node)
register_driver("camera", "real", _real(OnvifCameraDriver))
register_driver("ai_node", "real", _real(KaavalHttpDriver))


def create_driver(cfg: DriverConfig | dict, **deps) -> CameraDriver | AINodeDriver:
    cfg = DriverConfig.from_dict(cfg) if isinstance(cfg, dict) else cfg
    mode = cfg.mode or os.getenv("M3_DRIVER_MODE", "mock")
    try:
        builder = _REGISTRY[(cfg.kind, mode)]
    except KeyError:
        raise ValueError(f"no driver registered for kind={cfg.kind!r} mode={mode!r}; "
                         f"known: {sorted(_REGISTRY)}") from None
    return builder(cfg, **deps)
