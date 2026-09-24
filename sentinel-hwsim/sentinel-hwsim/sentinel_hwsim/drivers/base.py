"""
Shared machinery for mock drivers: connection state machine, the guarded call
path (NotConnected / disconnect handling), and vendor -> canonical mapping
that turns any malformed payload into a MappingError.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from pydantic import ValidationError

from ..contract import (
    AuthFailed,
    Capability,
    CapabilityNotSupported,
    ConnectionState,
    DeviceDisconnected,
    DeviceError,
    DeviceInfo,
    InvalidCommand,
    MappingError,
    NotConnected,
)
from ..transport import SimTransport

T = TypeVar("T")
log = logging.getLogger("sentinel_hwsim.driver")

_MAPPING_EXC = (ValidationError, KeyError, TypeError, ValueError, AttributeError, IndexError)


def as_list(value: Any) -> list:
    """Strict list accessor: a string is NOT a list (prevents tuple('abc') surprises)."""
    if not isinstance(value, list):
        raise TypeError(f"expected list, got {type(value).__name__}")
    return value


class BaseSimDriver:
    kind: str = "device"
    is_simulated = True

    def __init__(self, device_id: str, transport: SimTransport):
        self.device_id = device_id
        self._transport = transport
        self._state = ConnectionState.DISCONNECTED
        self._caps: frozenset[Capability] = frozenset()
        self._info: DeviceInfo | None = None
        self._connected_at: float | None = None

    # ------------------------------------------------------------ properties
    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def trace(self):
        """Recorded wire interactions (sim-only; real drivers expose logs instead)."""
        return self._transport.trace

    @property
    def faults(self):
        """Fault injector for this device (sim-only)."""
        return self._transport.faults

    def _now(self) -> datetime:
        return self._transport.clock.wall()

    def supports(self, capability: Capability) -> bool:
        return capability in self._caps

    def _require(self, capability: Capability) -> None:
        if not self.supports(capability):
            raise CapabilityNotSupported(f"{capability.value} not supported", device_id=self.device_id)

    # ------------------------------------------------------------ lifecycle
    async def _fetch_info(self) -> tuple[DeviceInfo, frozenset[Capability]]:
        raise NotImplementedError

    async def connect(self) -> DeviceInfo:
        if self._state is ConnectionState.CONNECTED and self._info:
            return self._info
        self._state = ConnectionState.CONNECTING
        log.info("connect  device=%s", self.device_id)
        try:
            await self._transport.handshake()
            self._state = ConnectionState.CONNECTED  # session is up; now query identity
            self._info, self._caps = await self._fetch_info()
        except DeviceError:
            self._state = ConnectionState.FAILED
            raise
        self._connected_at = self._transport.clock.now()
        log.info("connected device=%s caps=%s", self.device_id, sorted(c.value for c in self._caps))
        return self._info

    async def disconnect(self) -> None:
        if self._state is ConnectionState.DISCONNECTED:
            return
        await self._transport.clock.sleep(0.02)  # session teardown
        self._state = ConnectionState.DISCONNECTED
        self._connected_at = None
        log.info("disconnected device=%s", self.device_id)

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.disconnect()

    # ------------------------------------------------------------ call path
    async def _call(self, op: str, handler: Callable[[], Any], params: dict | None = None) -> Any:
        if self._state is not ConnectionState.CONNECTED:
            raise NotConnected(f"{op}: driver is {self._state.value}; call connect() first", device_id=self.device_id)
        try:
            raw = await self._transport.call(op, handler, params)
        except DeviceDisconnected:
            self._state = ConnectionState.DISCONNECTED
            log.warning("session lost device=%s op=%s", self.device_id, op)
            raise
        except AuthFailed:
            self._state = ConnectionState.FAILED
            raise
        self._raise_vendor_fault(op, raw)
        return raw

    def _raise_vendor_fault(self, op: str, raw: Any) -> None:
        if not isinstance(raw, dict):
            return
        if "Fault" in raw and isinstance(raw["Fault"], dict):  # ONVIF SOAP fault
            code = str(raw["Fault"].get("Code", ""))
            exc = CapabilityNotSupported if "ActionNotSupported" in code else InvalidCommand
            raise exc(f"{op}: {raw['Fault'].get('Reason')}", device_id=self.device_id, detail=raw["Fault"])
        if "error" in raw and isinstance(raw["error"], dict):  # Kaaval REST error
            raise InvalidCommand(f"{op}: {raw['error'].get('message')}", device_id=self.device_id, detail=raw["error"])

    def _map(self, op: str, raw: Any, build: Callable[[Any], T]) -> T:
        try:
            return build(raw)
        except _MAPPING_EXC as e:
            log.warning("mapping error device=%s op=%s err=%s", self.device_id, op, e.__class__.__name__)
            raise MappingError(f"{op}: vendor payload did not match contract ({e.__class__.__name__})",
                               device_id=self.device_id, detail={"sample": repr(raw)[:300]}) from e
