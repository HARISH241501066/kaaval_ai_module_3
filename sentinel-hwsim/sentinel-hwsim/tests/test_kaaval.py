import asyncio

import pytest
from conftest import run

from sentinel_hwsim import AIEvent, DeviceTimeout, Heartbeat, InvalidCommand, RuntimeState, Simulator


async def collect(node, sim_seconds, clock, poll=1.0):
    """Consume the node stream for a window of simulated time."""
    items = []

    async def consume():
        async for item in node.stream(poll_interval_s=poll):
            items.append(item)

    task = asyncio.create_task(consume())
    await clock.sleep(sim_seconds)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return items


def make(sim, **kw):
    kw.setdefault("heartbeat_interval_s", 5.0)
    kw.setdefault("event_rate_multiplier", 3.0)
    return sim.kaaval_node("EDGE-NODE-001", faults="ideal", **kw)


def test_events_and_heartbeats_flow_with_provenance(sim):
    node = make(sim, functions=["ANPR", "HELMET_DETECTION"])

    async def go():
        await node.connect()
        items = await collect(node, 60, sim.clock)
        events = [i for i in items if isinstance(i, AIEvent)]
        beats = [i for i in items if isinstance(i, Heartbeat)]
        assert len(events) > 10 and len(beats) >= 10
        assert {e.event_type for e in events} <= {"PLATE_READ", "HELMET_VIOLATION"}
        assert all(e.model_version and e.node_id == "EDGE-NODE-001" for e in events)
        plates = [e.attributes["plate"] for e in events if e.event_type == "PLATE_READ"]
        assert all(p.startswith("GJ") for p in plates)
    run(go())


def test_crash_stops_events_but_agent_keeps_heartbeating(sim):
    node = make(sim)

    async def go():
        await node.connect()
        sim.controls("EDGE-NODE-001").crash_runtime()
        await collect(node, 2, sim.clock)  # flush anything generated before the crash
        items = await collect(node, 30, sim.clock)
        assert not [i for i in items if isinstance(i, AIEvent)]
        beats = [i for i in items if isinstance(i, Heartbeat)]
        assert beats and all(b.runtime is RuntimeState.CRASHED for b in beats)
        assert (await node.get_ai_health()).runtime is RuntimeState.CRASHED
    run(go())


def test_stall_is_distinct_from_crash(sim):
    node = make(sim)

    async def go():
        await node.connect()
        sim.controls("EDGE-NODE-001").stall_inference()
        h = await node.get_ai_health()
        assert h.runtime is RuntimeState.RUNNING and h.inference_fps == 0
    run(go())


def test_pipe_cut_buffers_and_flushes_without_loss(sim):
    node = make(sim)
    ctl = sim.controls("EDGE-NODE-001")

    async def go():
        await node.connect()
        ctl.cut_pipe()
        during = await collect(node, 30, sim.clock)
        assert during == []
        assert len(node.sim.outbox) > 0
        buffered = len(node.sim.outbox)
        ctl.cut_pipe(False)
        after = await collect(node, 3, sim.clock)
        assert len(after) >= buffered  # everything buffered arrives, late
    run(go())


def test_load_model_write_changes_version(sim):
    node = make(sim)

    async def go():
        await node.connect()
        info = await node.load_model("kaaval-anpr-yolov8s-gj-2026.09")
        assert info.model_version == "kaaval-anpr-yolov8s-gj-2026.09"
        with pytest.raises(InvalidCommand):
            await node.load_model("../../etc/passwd")
    run(go())


def test_thermal_throttling_under_heat():
    sim = Simulator(speed=400, seed=5)
    node = sim.kaaval_node("EDGE-NODE-001", faults="ideal", ambient_c=60.0)

    async def go():
        await node.connect()
        await sim.clock.sleep(400)
        h = await node.get_ai_health()
        assert h.throttled and h.temp_c >= 85 and h.inference_fps < 12
    run(go())


def test_watchlist_hits(sim):
    node = make(sim, event_rate_multiplier=20.0)
    sim.controls("EDGE-NODE-001").set_watchlist(["GJ01AB1234"])

    async def go():
        await node.connect()
        items = await collect(node, 60, sim.clock)
        hits = [i for i in items if isinstance(i, AIEvent) and i.event_type == "WATCHLIST_HIT"]
        assert hits and all(h.attributes["plate"] == "GJ01AB1234" for h in hits)
    run(go())


def test_malformed_events_go_to_dead_letters_stream_survives(sim):
    node = make(sim, malformed_event_rate=0.3)

    async def go():
        await node.connect()
        items = await collect(node, 60, sim.clock)
        assert node.dead_letters and any(isinstance(i, AIEvent) for i in items)
        assert all(d["reason"] == "PARSE_ERROR" for d in node.dead_letters)
    run(go())


def test_power_loss_times_out_then_node_reboots(sim):
    node = make(sim)
    ctl = sim.controls("EDGE-NODE-001")

    async def go():
        await node.connect()
        ctl.offline()
        with pytest.raises(DeviceTimeout):
            await node.get_ai_health()
        ctl.offline(False)
        assert (await node.get_ai_health()).runtime is RuntimeState.RESTARTING
        await sim.clock.sleep(5)
        assert (await node.get_ai_health()).runtime is RuntimeState.RUNNING
    run(go())
