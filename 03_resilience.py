"""
Example 3: application-side resilience against every simulated failure mode.
Deterministic injected faults first, then a soak test with the "flaky" preset.

The retry / circuit-breaker code here is the shape M3-C1 needs: retry x2 with
exponential backoff + jitter for retryable errors, never retry mapping or
auth errors, open the breaker after repeated failures.

    python examples/03_resilience.py
"""
import asyncio
import random
import statistics
import time
from collections import Counter

from sentinel_hwsim import (
    AuthFailed,
    CapabilityNotSupported,
    ConnectionState,
    DeviceError,
    InvalidCommand,
    MappingError,
    Simulator,
)


class SystemUnavailable(Exception):
    """Raised by the app when the breaker is open (M3 code: SYSTEM_UNAVAILABLE)."""


class CircuitBreaker:
    def __init__(self, now, *, threshold: int = 3, cooldown_s: float = 30.0):
        self.now, self.threshold, self.cooldown_s = now, threshold, cooldown_s
        self.failures, self.opened_at = 0, None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "CLOSED"
        return "HALF_OPEN" if self.now() - self.opened_at >= self.cooldown_s else "OPEN"

    def before_call(self) -> None:
        if self.state == "OPEN":
            raise SystemUnavailable("circuit open; failing fast")

    def success(self) -> None:
        self.failures, self.opened_at = 0, None

    def failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold or self.state == "HALF_OPEN":
            self.opened_at = self.now()


dead_letter_queue: list[dict] = []


async def resilient(driver, op: str, fn, breaker: CircuitBreaker, *, retries: int = 2, base_delay: float = 0.05):
    breaker.before_call()
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if driver.state is not ConnectionState.CONNECTED:
                await driver.connect()           # transparent reconnect after a dropped session
            result = await fn()
            breaker.success()
            return result
        except MappingError as e:             # data problem: retrying returns the same garbage
            dead_letter_queue.append({"device": driver.device_id, "op": op, "sample": e.detail})
            raise
        except (AuthFailed, InvalidCommand, CapabilityNotSupported):
            raise                                # needs a human / config change, not a retry
        except DeviceError as e:
            if not e.retryable:
                raise
            last = e
            breaker.failure()
            if breaker.state == "OPEN" or attempt == retries:
                break
            delay = base_delay * (2 ** attempt) * random.uniform(0.5, 1.5)  # backoff + jitter
            print(f"    retry {attempt + 1}/{retries} after {e.code.value} (sleep {delay * 1000:.0f} ms)")
            await asyncio.sleep(delay)
    raise last  # type: ignore[misc]


async def attempt(label, driver, breaker, op="get_stream_uri"):
    fn = lambda: driver.get_stream_uri("SUB")
    try:
        src = await resilient(driver, op, fn, breaker)
        print(f"{label:<44} OK   {src.uri}   breaker={breaker.state}")
    except SystemUnavailable as e:
        print(f"{label:<44} FAST-FAIL {e}   breaker={breaker.state}")
    except DeviceError as e:
        print(f"{label:<44} FAIL {e.code.value} retryable={e.retryable}   breaker={breaker.state}")


async def main() -> None:
    sim = Simulator(speed=50, seed=11)
    cam = sim.camera("CAM-017", faults="healthy")
    chaos = sim.controls("CAM-017")
    breaker = CircuitBreaker(sim.clock.now, threshold=3, cooldown_s=30)
    await cam.connect()

    print("== deterministic fault injection ==")
    await attempt("baseline", cam, breaker)
    chaos.fail_next("timeout")
    await attempt("one timeout (recovers on retry)", cam, breaker)
    chaos.fail_next("drop_connection")
    await attempt("connection dropped (auto-reconnect)", cam, breaker)
    chaos.fail_next("invalid_data")
    await attempt("garbled payload (MappingError, to DLQ)", cam, breaker)
    chaos.revoke_credentials()
    await attempt("credentials revoked (no retry)", cam, breaker)
    chaos.revoke_credentials(False)
    chaos.offline()
    await attempt("device offline (retries exhausted)", cam, breaker)
    await attempt("device offline again (breaker opens)", cam, breaker)
    await attempt("breaker open (no device call made)", cam, breaker)
    chaos.heal()
    await sim.clock.sleep(31)
    await attempt("healed + cooldown (half-open probe)", cam, breaker)
    print(f"DLQ entries: {len(dead_letter_queue)}")

    print("\n== soak test: 200 calls against the 'flaky' preset ==")
    flaky = sim.camera("CAM-099", faults="flaky")
    br2 = CircuitBreaker(sim.clock.now, threshold=5, cooldown_s=10)
    outcomes, latencies = Counter(), []
    for _ in range(200):
        t0 = sim.clock.now()
        try:
            await resilient(flaky, "status", flaky.get_status, br2, base_delay=0.001)
            outcomes["ok"] += 1
            latencies.append((sim.clock.now() - t0) * 1000)
        except SystemUnavailable:
            outcomes["fast_fail"] += 1
            await sim.clock.sleep(2)
        except DeviceError as e:
            outcomes[e.code.value] += 1
    print(f"outcomes: {dict(outcomes)}")
    print(f"success latency (simulated): p50={statistics.median(latencies):.0f} ms "
          f"p95={statistics.quantiles(latencies, n=20)[18]:.0f} ms")
    print(f"wire faults seen: {Counter(e.outcome for e in flaky.trace if e.outcome != 'OK')}")


if __name__ == "__main__":
    t = time.perf_counter()
    asyncio.run(main())
    print(f"(ran in {time.perf_counter() - t:.1f}s real time)")
