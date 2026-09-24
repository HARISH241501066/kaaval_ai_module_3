import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sentinel_hwsim import Simulator


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def sim():
    # 100x time compression, zero latency: functional tests stay fast and deterministic
    return Simulator(speed=100, seed=1234)
