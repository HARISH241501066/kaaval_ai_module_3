"""
Example 2: consume AI events, then break the node layer by layer and watch a
C6-style classifier tell the failures apart and emit ONE alert per node.

    python examples/02_ai_node_health.py
"""
import asyncio
import logging
from datetime import timedelta

from sentinel_hwsim import AINodeDriver, DeviceError, RuntimeState, Simulator, enable_trace_logging

HEARTBEAT_S = 5.0


# ----------------------------------------------------------------- app code
async def classify(node: AINodeDriver, last_heartbeat_at, now) -> str:
    """Layered check from M3-C6: node -> runtime -> inference -> pipe."""
    try:
        h = await node.get_ai_health()
    except DeviceError as e:
        return f"NODE_OFFLINE ({e.code.value})"
    if h.runtime is RuntimeState.CRASHED:
        return "RUNTIME_DOWN"
    if h.runtime is RuntimeState.RESTARTING:
        return "RESTARTING"
    if h.inference_fps == 0:
        return "INFERENCE_DEGRADED"
    if last_heartbeat_at is None or now - last_heartbeat_at > timedelta(seconds=3 * HEARTBEAT_S):
        return "PIPE_DISCONNECTED"
    return "THROTTLED" if h.throttled else "HEALTHY"


class NodeMonitor:
    def __init__(self, node: AINodeDriver, clock, assigned_cameras: tuple[str, ...]):
        self.node, self.clock, self.cameras = node, clock, assigned_cameras
        self.state = "HEALTHY"
        self.alerts: list[str] = []
        self.events_seen = 0

    async def consume(self) -> None:
        try:
            async for item in self.node.stream(poll_interval_s=1.0):
                if hasattr(item, "event_type"):
                    self.events_seen += 1
        except DeviceError:
            pass  # health loop reports it; a real service would reconnect with backoff

    async def check(self) -> str:
        new = await classify(self.node, self.node.last_heartbeat_at, self.clock.wall())
        if new != self.state:
            if new != "HEALTHY":
                cams = self.cameras  # in M3 this comes from C7 camera->node assignments
                impact = "AI degraded on" if new == "THROTTLED" else "lost AI on"
                self.alerts.append(f"{self.node.device_id} {new}: {impact} {len(cams)} cameras ({', '.join(cams)})")
            self.state = new
        return new


# ----------------------------------------------------------------- scenario
async def main() -> None:
    enable_trace_logging(logging.WARNING)  # only log faults
    sim = Simulator(speed=20, seed=3)
    node = sim.kaaval_node("EDGE-NODE-001", heartbeat_interval_s=HEARTBEAT_S,
                           cameras=["CAM-001", "CAM-002", "CAM-003", "CAM-004"])
    chaos = sim.controls("EDGE-NODE-001")
    await node.connect()
    print("model:", (await node.get_model_info()).model_version)
    caps = await node.get_ai_capabilities()
    print("functions:", caps.functions)

    mon = NodeMonitor(node, sim.clock, caps.assigned_cameras)
    consumer = asyncio.create_task(mon.consume())

    async def phase(label: str, sim_seconds: float, action=None) -> None:
        if action:
            action()
        await sim.clock.sleep(sim_seconds)
        state = await mon.check()
        print(f"{label:<34} -> {state:<20} events so far={mon.events_seen}")

    await phase("normal operation", 30)
    await phase("runtime crash", 5, chaos.crash_runtime)
    await phase("runtime restart", 10, chaos.restart_runtime)
    await phase("inference stall", 5, chaos.stall_inference)
    await phase("inference recovered", 5, lambda: chaos.stall_inference(False))
    await phase("event pipe cut", 20, chaos.cut_pipe)
    await phase("event pipe restored (late flush)", 5, lambda: chaos.cut_pipe(False))
    await phase("heatwave, 58C ambient", 240, lambda: chaos.set_ambient(58))
    await phase("node power loss", 5, chaos.offline)

    consumer.cancel()
    print("\nalerts raised (one per transition, per node):")
    for a in mon.alerts:
        print("  ", a)
    print(f"dead letters: {len(node.dead_letters)}")


if __name__ == "__main__":
    asyncio.run(main())
