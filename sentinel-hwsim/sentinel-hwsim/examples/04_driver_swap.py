"""
Example 4: the swap. Application code builds drivers from config and depends
only on the contract. Flip one env var to move from simulation to hardware.

    python examples/04_driver_swap.py                   # everything simulated
    M3_DRIVER_MODE=real python examples/04_driver_swap.py  # real drivers (skeletons until implemented)
"""
import asyncio
import os

from sentinel_hwsim import AINodeDriver, CameraDriver, DeviceError, Simulator, create_driver

# In M3 this list comes from the C7 registry (integration_systems / processing_nodes).
FLEET = [
    {"kind": "camera", "device_id": "CAM-001", "endpoint": "10.20.0.11",
     "credential_ref": "vault:secret/m3/camera/CAM-001", "sim": {"faults": "healthy"}},
    {"kind": "camera", "device_id": "CAM-002", "endpoint": "10.20.0.12",
     "credential_ref": "vault:secret/m3/camera/CAM-002", "sim": {"faults": "flaky", "ptz": False}},
    {"kind": "ai_node", "device_id": "EDGE-NODE-001", "endpoint": "https://10.20.1.5:8443",
     "credential_ref": "vault:secret/m3/node/EDGE-NODE-001",
     "sim": {"cameras": ["CAM-001", "CAM-002"], "functions": ["ANPR", "HELMET_DETECTION"]}},
]


class VaultCredentials:
    """Stand-in for the C7 CredentialProvider used by real drivers."""
    async def resolve(self, credential_ref: str) -> tuple[str, str]:
        raise NotImplementedError("wire to hvac / Vault in M3-C7")


async def onboard(driver) -> str:
    """Contract-only application logic: connection test before a device goes ACTIVE (M3-C7)."""
    try:
        info = await driver.connect()
        if isinstance(driver, AINodeDriver):
            caps = await driver.get_ai_capabilities()
            detail = f"functions={list(caps.functions)} cameras={list(caps.assigned_cameras)}"
        elif isinstance(driver, CameraDriver):
            src = await driver.get_stream_uri("SUB")
            detail = f"sub={src.codec} {src.width}x{src.height}"
        await driver.disconnect()
        return f"ACTIVE   {info.device_id:<14} {info.vendor}/{info.model}  {detail}"
    except DeviceError as e:
        return f"REJECTED {driver.device_id:<14} {e.code.value}: {e}"
    except NotImplementedError as e:
        return f"TODO     {driver.device_id:<14} {e}"


async def main() -> None:
    mode = os.getenv("M3_DRIVER_MODE", "mock")
    sim = Simulator(speed=20, seed=5) if mode == "mock" else None
    print(f"driver mode: {mode}")
    for entry in FLEET:
        driver = create_driver(entry, simulator=sim, credentials=VaultCredentials())
        print(f"  {type(driver).__name__:<18} {await onboard(driver)}")


if __name__ == "__main__":
    asyncio.run(main())
