import pytest
from conftest import run

from sentinel_hwsim import (
    AuthFailed,
    ConnectionFailed,
    ConnectionState,
    DeviceDisconnected,
    DeviceTimeout,
    MappingError,
    Simulator,
)

CAMERA_READS = ["get_status", "get_stream_uri", "get_snapshot", "ptz_get_position", "get_clock"]
NODE_READS = ["get_ai_capabilities", "get_model_info", "get_ai_health"]


@pytest.mark.parametrize("fault, exc, retryable", [
    ("timeout", DeviceTimeout, True),
    ("drop_connection", DeviceDisconnected, True),
    ("auth_failure", AuthFailed, False),
    ("invalid_data", MappingError, False),
])
def test_injected_call_faults(sim, fault, exc, retryable):
    cam = sim.camera("CAM-001", faults="ideal")

    async def go():
        await cam.connect()
        sim.controls("CAM-001").fail_next(fault)
        with pytest.raises(exc) as ei:
            await cam.get_stream_uri()
        assert ei.value.retryable is retryable
        assert cam.trace[-1].outcome == fault
    run(go())


def test_drop_marks_driver_disconnected_and_reconnect_works(sim):
    cam = sim.camera("CAM-001", faults="ideal")

    async def go():
        await cam.connect()
        sim.controls("CAM-001").fail_next("drop_connection")
        with pytest.raises(DeviceDisconnected):
            await cam.get_status()
        assert cam.state is ConnectionState.DISCONNECTED
        await cam.connect()
        assert (await cam.get_status()).state is ConnectionState.CONNECTED
    run(go())


def test_timeout_takes_the_full_deadline(sim):
    cam = sim.camera("CAM-001", faults="ideal", deadline_s=5.0)

    async def go():
        await cam.connect()
        sim.controls("CAM-001").fail_next("timeout")
        t0 = sim.clock.now()
        with pytest.raises(DeviceTimeout):
            await cam.get_status()
        assert sim.clock.now() - t0 >= 5.0
    run(go())


def test_offline_and_revoked_conditions(sim):
    cam = sim.camera("CAM-001", faults="ideal")
    ctl = sim.controls("CAM-001")

    async def go():
        ctl.offline()
        with pytest.raises(ConnectionFailed):
            await cam.connect()
        assert cam.state is ConnectionState.FAILED
        ctl.offline(False)
        ctl.revoke_credentials()
        with pytest.raises(AuthFailed):
            await cam.connect()
        ctl.heal()
        await cam.connect()
    run(go())


@pytest.mark.parametrize("op", CAMERA_READS)
def test_invalid_data_always_surfaces_as_mapping_error_camera(op):
    sim = Simulator(speed=100, seed=99)
    cam = sim.camera("CAM-001", faults="ideal")

    async def go():
        await cam.connect()
        for _ in range(25):  # cover every corruption strategy
            sim.controls("CAM-001").fail_next("invalid_data")
            with pytest.raises(MappingError):
                await getattr(cam, op)()
    run(go())


@pytest.mark.parametrize("op", NODE_READS)
def test_invalid_data_always_surfaces_as_mapping_error_node(op):
    sim = Simulator(speed=100, seed=99)
    node = sim.kaaval_node("EDGE-NODE-001", faults="ideal")

    async def go():
        await node.connect()
        for _ in range(25):
            sim.controls("EDGE-NODE-001").fail_next("invalid_data")
            with pytest.raises(MappingError):
                await getattr(node, op)()
    run(go())


def test_same_seed_same_fault_sequence():
    async def outcomes(seed):
        sim = Simulator(speed=500, seed=seed)
        cam = sim.camera("CAM-001", faults="hostile")
        result = []
        for _ in range(40):
            try:
                if cam.state is not ConnectionState.CONNECTED:
                    await cam.connect()
                await cam.get_status()
                result.append("ok")
            except Exception as e:  # noqa: BLE001
                result.append(type(e).__name__)
        return result

    a, b, c = run(outcomes(1)), run(outcomes(1)), run(outcomes(2))
    assert a == b
    assert a != c
    assert set(a) - {"ok"}, "hostile preset should produce failures"


def test_latency_is_realistic():
    sim = Simulator(speed=200, seed=3)
    cam = sim.camera("CAM-001", faults="healthy")

    async def go():
        await cam.connect()
        for _ in range(100):
            await cam.get_status()
        lat = sorted(e.latency_ms for e in cam.trace if e.op == "status")
        assert 40 < lat[len(lat) // 2] < 150      # median around the 80 ms base
        assert lat[-1] < 5000                      # spikes stay under the deadline
    run(go())
