"""
Contract conformance kit. Every driver, mock or real, must pass this file.

Mock drivers run by default. To run against real hardware once real drivers
exist, set M3_CONFORMANCE_REAL=1 plus M3_REAL_CAMERA / M3_REAL_NODE endpoints
and provide a CredentialProvider in _real_deps().
"""
import os

import pytest
from conftest import run

from sentinel_hwsim import (
    AICapabilities,
    AIHealth,
    AINodeDriver,
    CameraDriver,
    Capability,
    ClockInfo,
    ConnectionState,
    DeviceInfo,
    DeviceStatus,
    DriverConfig,
    ModelInfo,
    NotConnected,
    PtzPosition,
    Simulator,
    Snapshot,
    StreamSource,
    create_driver,
)
from sentinel_hwsim.drivers.real import KaavalHttpDriver, OnvifCameraDriver


def _mock_camera():
    return create_driver(DriverConfig(kind="camera", device_id="CAM-C1", mode="mock", sim={"faults": "ideal"}),
                         simulator=Simulator(speed=100))


def _mock_node():
    return create_driver(DriverConfig(kind="ai_node", device_id="NODE-C1", mode="mock", sim={"faults": "ideal"}),
                         simulator=Simulator(speed=100))


CAMERAS = [pytest.param(_mock_camera, id="mock-camera")]
NODES = [pytest.param(_mock_node, id="mock-node")]
if os.getenv("M3_CONFORMANCE_REAL") == "1":  # pragma: no cover - hardware only
    pytest.skip("wire _real_deps() before enabling real conformance", allow_module_level=True)


@pytest.mark.parametrize("factory", CAMERAS)
def test_camera_contract(factory):
    cam = factory()
    assert isinstance(cam, CameraDriver)

    async def go():
        with pytest.raises(NotConnected):
            await cam.get_status()
        assert isinstance(await cam.connect(), DeviceInfo)
        assert cam.state is ConnectionState.CONNECTED
        assert isinstance(await cam.get_status(), DeviceStatus)
        for profile in ("MAIN", "SUB"):
            assert isinstance(await cam.get_stream_uri(profile), StreamSource)
        assert isinstance(await cam.get_snapshot(), Snapshot)
        assert isinstance(await cam.get_clock(), ClockInfo)
        if cam.supports(Capability.PTZ):
            await cam.ptz_move(pan=0.1, tilt=0.1, zoom=0.1)
            assert isinstance(await cam.ptz_get_position(), PtzPosition)
        await cam.disconnect()
        assert cam.state is ConnectionState.DISCONNECTED
    run(go())


@pytest.mark.parametrize("factory", NODES)
def test_ai_node_contract(factory):
    node = factory()
    assert isinstance(node, AINodeDriver)

    async def go():
        assert isinstance(await node.connect(), DeviceInfo)
        assert isinstance(await node.get_ai_capabilities(), AICapabilities)
        assert isinstance(await node.get_model_info(), ModelInfo)
        assert isinstance(await node.get_ai_health(), AIHealth)
        assert isinstance(await node.get_status(), DeviceStatus)
        await node.disconnect()
    run(go())


def test_real_skeletons_share_the_contract_shape():
    for cls, proto in ((OnvifCameraDriver, CameraDriver), (KaavalHttpDriver, AINodeDriver)):
        drv = cls("X", "endpoint", "vault:ref", credentials=None)
        assert isinstance(drv, proto), f"{cls.__name__} drifted from {proto.__name__}"


def test_factory_honours_env_mode(monkeypatch):
    monkeypatch.setenv("M3_DRIVER_MODE", "real")
    with pytest.raises(ValueError, match="needs endpoint"):
        create_driver({"kind": "camera", "device_id": "CAM-X"})
    with pytest.raises(ValueError, match="no driver registered"):
        create_driver({"kind": "camera", "device_id": "CAM-X", "mode": "hikvision"})
