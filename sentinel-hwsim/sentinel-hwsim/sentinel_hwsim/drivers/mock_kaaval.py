"""Mock Kaaval AI node driver: implements contract.AINodeDriver against a simulated KaavalNodeModel."""
from __future__ import annotations

import logging
from collections import deque
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from typing import Any

from ..contract import (
    AICapabilities,
    AIEvent,
    AIHealth,
    Capability,
    DeviceError,
    DeviceInfo,
    DeviceStatus,
    DeviceTimeout,
    Heartbeat,
    InvalidCommand,
    MappingError,
    ModelInfo,
    RuntimeState,
    StreamItem,
)
from ..models.kaaval import KaavalNodeModel
from ..transport import SimTransport
from .base import BaseSimDriver, as_list

log = logging.getLogger("sentinel_hwsim.driver")
_CAPS = frozenset({Capability.HEALTH, Capability.EVENTS, Capability.AI_CAPABILITIES,
                   Capability.MODEL_INFO, Capability.AI_HEALTH})


def _ms(value: Any) -> datetime:
    if not isinstance(value, int):
        raise TypeError("timestamp must be integer epoch ms")
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


class MockKaavalDriver(BaseSimDriver):
    kind = "ai_node"

    def __init__(self, model: KaavalNodeModel, transport: SimTransport):
        super().__init__(model.node_id, transport)
        self.sim = model
        # receive-side bookkeeping used by health monitoring (C6 "pipe" layer)
        self.last_heartbeat_at: datetime | None = None
        self.last_event_at: datetime | None = None
        self.events_received = 0
        self.heartbeats_received = 0
        self.dead_letters: deque[dict[str, Any]] = deque(maxlen=1000)

    async def _fetch_info(self):
        raw = await self._call("node_info", self.sim.info)
        return self._map("node_info", raw, lambda r: (DeviceInfo(
            device_id=self.device_id, kind="ai_node", vendor=r["vendor"], model=f"{r['type']} / {r['hw']}",
            firmware=f"sdk {r['sdk']}, {r['agent']}", serial=r["serial"],
            capabilities=tuple(sorted(_CAPS, key=lambda c: c.value))), _CAPS))

    async def get_status(self) -> DeviceStatus:
        h = await self.get_ai_health()
        return DeviceStatus(device_id=self.device_id, state=self.state,
                            uptime_s=round(self._transport.clock.now() - (self._connected_at or 0), 1),
                            observed_at=h.observed_at, fps=h.inference_fps)

    async def get_ai_capabilities(self) -> AICapabilities:
        self._require(Capability.AI_CAPABILITIES)
        raw = await self._call("ai_capabilities", self.sim.capabilities)
        return self._map("ai_capabilities", raw, lambda r: AICapabilities(
            node_id=r["node"], functions=tuple(as_list(r["functions"])), assigned_cameras=tuple(as_list(r["cameras"]))))

    async def get_model_info(self) -> ModelInfo:
        self._require(Capability.MODEL_INFO)
        raw = await self._call("model_info", self.sim.model_info)
        return self._map("model_info", raw, lambda r: ModelInfo(
            analytics_provider=r["provider"], analytics_version=r["analytics_version"],
            model_version=r["model_version"], loaded_at=_ms(r["loaded_at_ms"])))

    async def get_ai_health(self) -> AIHealth:
        self._require(Capability.AI_HEALTH)
        raw = await self._call("ai_health", self.sim.health)
        return self._map("ai_health", raw, lambda r: AIHealth(
            node_id=r["node"], runtime=RuntimeState(r["runtime"]), inference_fps=r["inference"]["fps"],
            latency_ms=r["inference"]["latency_ms"], cpu_pct=r["system"]["cpu"], mem_pct=r["system"]["mem"],
            temp_c=r["system"]["temp_c"], throttled=r["system"]["throttled"], observed_at=_ms(r["ts_ms"])))

    async def load_model(self, model_version: str, *, timeout_s: float = 30.0) -> ModelInfo:
        """Write operation: deploy a model. Blocks (in sim time) until the runtime is back up."""
        self._require(Capability.MODEL_INFO)
        await self._call("load_model", lambda: self.sim.load_model(model_version), {"model_version": model_version})
        clock, deadline = self._transport.clock, self._transport.clock.now() + timeout_s
        while clock.now() < deadline:
            await clock.sleep(1.0)
            health = await self.get_ai_health()
            if health.runtime is RuntimeState.RUNNING:
                info = await self.get_model_info()
                if info.model_version != model_version:
                    raise InvalidCommand("runtime restarted but model version did not change",
                                         device_id=self.device_id, detail=info.model_dump())
                return info
            if health.runtime is RuntimeState.CRASHED:
                raise InvalidCommand("runtime crashed while loading model", device_id=self.device_id)
        raise DeviceTimeout(f"model load not confirmed within {timeout_s}s", device_id=self.device_id)

    # ----------------------------------------------------------- event pipe
    def _map_item(self, item: Any) -> StreamItem:
        if not isinstance(item, dict):
            raise TypeError("item is not an object")
        if item.get("kind") == "heartbeat":
            return Heartbeat(node_id=item["node"], runtime=RuntimeState(item["runtime"]),
                             inference_fps=item["fps"], sent_at=_ms(item["ts_ms"]))
        return AIEvent(event_id=item["evt_id"], node_id=item["node"], camera_external_id=item["cam"],
                       event_type=item["type"], occurred_at=_ms(item["ts_ms"]), confidence=item["conf"],
                       attributes=item["data"], model_version=item["model_ver"])

    async def stream(self, *, poll_interval_s: float = 1.0) -> AsyncIterator[StreamItem]:
        """
        Yields AI events and heartbeats until the session drops.
        Transport errors (timeout, disconnect, auth) propagate so the caller decides
        whether to reconnect. Malformed items never kill the stream: they go to
        `dead_letters` (the M3-C4 DLQ equivalent) and are logged.
        """
        self._require(Capability.EVENTS)
        clock = self._transport.clock
        while True:
            raw = await self._call("events.poll", self.sim.drain)
            try:
                items = self._map("events.poll", raw, lambda r: as_list(r["items"]))
            except MappingError as e:
                self.dead_letters.append({"reason": "BATCH_UNPARSEABLE", "sample": e.detail["sample"]})
                items = []
            now = self._now()
            for item in items:
                try:
                    mapped = self._map("events.item", item, self._map_item)
                except DeviceError:
                    self.dead_letters.append({"reason": "PARSE_ERROR", "sample": repr(item)[:300]})
                    continue
                if isinstance(mapped, Heartbeat):
                    self.last_heartbeat_at, self.heartbeats_received = now, self.heartbeats_received + 1
                else:
                    self.last_event_at, self.events_received = now, self.events_received + 1
                yield mapped
            await clock.sleep(poll_interval_s)
