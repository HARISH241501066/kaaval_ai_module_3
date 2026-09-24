import pytest
from conftest import run

from sentinel_hwsim import Capability, CapabilityNotSupported, ConnectionState, InvalidCommand, NotConnected


def test_connect_disconnect_lifecycle(sim):
    cam = sim.camera("CAM-001", faults="ideal")

    async def go():
        assert cam.state is ConnectionState.DISCONNECTED
        info = await cam.connect()
        assert cam.state is ConnectionState.CONNECTED
        assert info.kind == "camera" and Capability.LIVE_STREAM in info.capabilities
        assert await cam.connect() is info  # idempotent
        await cam.disconnect()
        assert cam.state is ConnectionState.DISCONNECTED
    run(go())


def test_calls_require_connection(sim):
    cam = sim.camera("CAM-001", faults="ideal")
    with pytest.raises(NotConnected):
        run(cam.get_status())


def test_stream_profiles(sim):
    cam = sim.camera("CAM-001", faults="ideal", rtsp_base="rtsp://mediamtx:8554/sim/cam001")

    async def go():
        async with cam:
            main, sub = await cam.get_stream_uri("MAIN"), await cam.get_stream_uri("SUB")
            assert (main.width, main.codec) == (2560, "H265")
            assert (sub.width, sub.codec) == (640, "H264")
            assert main.uri.startswith("rtsp://mediamtx:8554/sim/cam001/")
            assert "@" not in main.uri  # no credentials in URIs
    run(go())


def test_ptz_moves_over_time_and_converges(sim):
    cam = sim.camera("CAM-001", faults="ideal")

    async def go():
        async with cam:
            await cam.ptz_move(pan=1.0, tilt=0.0, zoom=0.0)
            early = await cam.ptz_get_position()
            assert early.moving and early.pan < 1.0
            await sim.clock.sleep(3)  # 0.5 units/s -> 2 s to travel 1.0
            final = await cam.ptz_get_position()
            assert not final.moving and final.pan == pytest.approx(1.0)
    run(go())


def test_encoder_write_changes_measured_fps(sim):
    cam = sim.camera("CAM-001", faults="ideal")

    async def go():
        async with cam:
            assert (await cam.get_status()).fps == pytest.approx(25, abs=1.5)
            src = await cam.set_encoder("MAIN", fps=10, bitrate_kbps=1000)
            assert src.fps == 10
            await sim.clock.sleep(15)
            assert (await cam.get_status()).fps == pytest.approx(10, abs=1.5)
            with pytest.raises(InvalidCommand):
                await cam.set_encoder("MAIN", fps=120, bitrate_kbps=1000)
    run(go())


def test_bullet_camera_has_no_ptz(sim):
    cam = sim.camera("CAM-002", faults="ideal", ptz=False)

    async def go():
        async with cam:
            assert not cam.supports(Capability.PTZ)
            with pytest.raises(CapabilityNotSupported):
                await cam.ptz_move(pan=0, tilt=0, zoom=0)
    run(go())


def test_clock_offset_reported(sim):
    cam = sim.camera("CAM-001", faults="ideal", clock_offset_s=-420.0)

    async def go():
        async with cam:
            assert (await cam.get_clock()).offset_s == pytest.approx(-420, abs=1)
    run(go())


def test_snapshot_is_png(sim):
    cam = sim.camera("CAM-001", faults="ideal")

    async def go():
        async with cam:
            snap = await cam.get_snapshot()
            assert snap.content_type == "image/png" and snap.data.startswith(b"\x89PNG")
    run(go())
