"""
sentinel_hwsim: simulated camera and Kaaval AI node hardware for Sentinel
Module 3 prototyping. Import contract types from `sentinel_hwsim.contract`,
build drivers with `create_driver`, and keep `Simulator`/`SimControls` in tests
and demo scripts only.
"""
import logging
import sys

from .contract import *
from .factory import DriverConfig, SimControls, Simulator, create_driver, register_driver  # noqa: F401
from .faults import Condition, Fault, FaultProfile, LatencyProfile, preset  # noqa: F401

__version__ = "0.1.0"


def enable_trace_logging(level: int = logging.DEBUG) -> None:
    """Print every simulated wire call. DEBUG = all calls, WARNING = faults only."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)-22s %(message)s", "%H:%M:%S"))
    root = logging.getLogger("sentinel_hwsim")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
