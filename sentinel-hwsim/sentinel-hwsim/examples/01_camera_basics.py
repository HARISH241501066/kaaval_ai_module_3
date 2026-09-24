"""
Example 1: initialise a simulated camera, send commands, read data, poll status.

Note that the application function `inspect_camera` only uses the contract.
It would run unchanged against OnvifCameraDriver.

    python examples/01_camera_basics.py
"""
import asyncio
import logging
from pathlib import Path

from sentinel_hwsim import (
    CameraDriver,
    Capability,
    DeviceError,
    DriverConfig,
    Simulator,
    create_driver,
    enable_trace_logging,
    MEDIA_DIR,
)


async def inspect_camera(cam: CameraDriver) -> None:
    """Application code: contract only, no simulator imports."""
    info = await cam.connect()
    print(f"connected: {info.vendor} {info.model} fw={info.firmware} serial={info.serial}")
    print(f"capabilities: {[c.value for c in info.capabilities]}")

    main = await cam.get_stream_uri("MAIN")
    sub = await cam.get_stream_uri("SUB")
    print(f"MAIN {main.codec} {main.width}x{main.height}@{main.fps} -> {main.uri}")
    print(f"SUB  {sub.codec} {sub.width}x{sub.height}@{sub.fps} -> {sub.uri}")

    clock = await cam.get_clock()
    print(f"device clock offset: {clock.offset_s:+.2f}s")

    # WRITE: drop the main stream to 10 fps / 1500 kbps, then poll until the encoder settles
    await cam.set_encoder("MAIN", fps=10, bitrate_kbps=1500)
    for _ in range(6):
        st = await cam.get_status()
        print(f"  status: fps={st.fps:5.2f} bitrate={st.bitrate_kbps:7.1f} kbps")
        await asyncio.sleep(0.2)

    # WRITE + POLL: PTZ absolute move, poll position until the head stops
    if cam.supports(Capability.PTZ):
        await cam.ptz_move(pan=0.75, tilt=-0.3, zoom=0.4)
        while True:
            pos = await cam.ptz_get_position()
            print(f"  ptz: pan={pos.pan:+.3f} tilt={pos.tilt:+.3f} zoom={pos.zoom:.3f} moving={pos.moving}")
            if not pos.moving:
                break
            await asyncio.sleep(0.1)

    snap = await cam.get_snapshot()
    out = Path("snapshot_CAM-001.png")
    out.write_bytes(snap.data)
    print(f"snapshot: {snap.content_type} {len(snap.data)} bytes -> {out}")

    # invalid command: the device rejects it, the driver raises a typed error
    try:
        await cam.set_encoder("MAIN", fps=90, bitrate_kbps=1500)
    except DeviceError as e:
        print(f"rejected as expected: {e.code.value}: {e}")

    await cam.disconnect()


async def main() -> None:
    enable_trace_logging(logging.INFO)  # use logging.DEBUG to see every wire call

    # Local video files from the project media folder mapped to MAIN and SUB streams
    main_video = MEDIA_DIR / "camera_main.mp4"
    sub_video = MEDIA_DIR / "camera_sub.mp4"

    sim = Simulator(speed=10, seed=7)   # 10x time compression
    cam = create_driver(DriverConfig(kind="camera", device_id="CAM-001", mode="mock",
                                     sim={"faults": "healthy", "clock_offset_s": 37.0,
                                          "stream_files": {
                                              "MAIN": main_video,
                                              "SUB": sub_video,
                                          }}),
                        simulator=sim)
    await inspect_camera(cam)

    print("\nlast 5 wire interactions:")
    for entry in list(cam.trace)[-5:]:
        print(" ", entry)


if __name__ == "__main__":
    asyncio.run(main())
