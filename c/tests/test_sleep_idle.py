"""Idle-sleep unit tests for the OpenAI gateway (2026-09-13, --sleep-idle-seconds).

Fast and deterministic: a FakeProcess stands in for the engine child (same technique
as DispatcherTest in test_openai_server.py), so these run in CI without a real model
or a real subprocess. The real-model measurement protocol (RSS, cold/warm wake
seconds against an actual GGUF) is a manual run, not this file -- see
SLEEP_IDLE_2026-09-13.md and c/tests/sleep_idle_bridge_engine.py.

Covers the requirements from the task: default-off (0 = never sleep), sleep after N
idle seconds, health's is_sleeping, exactly one respawn for concurrent wakers, and
the race where a request arrives as the timer fires.
"""
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai_server import Engine, GenerationScheduler, READY  # noqa: E402


class BlockingStream:
    """Minimal readline()/read() stream a dispatcher thread can block on,
    fed from the test thread. Mirrors test_openai_server.py's BlockingStream."""

    def __init__(self, initial=b""):
        self.buffer = bytearray(initial)
        self.closed = False
        self.condition = threading.Condition()

    def feed(self, data):
        with self.condition:
            self.buffer.extend(data)
            self.condition.notify_all()

    def read(self, size=1):
        with self.condition:
            while len(self.buffer) < size and not self.closed:
                self.condition.wait()
            if not self.buffer and self.closed:
                return b""
            size = min(size, len(self.buffer))
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    def readline(self):
        with self.condition:
            while b"\n" not in self.buffer and not self.closed:
                self.condition.wait()
            if not self.buffer and self.closed:
                return b""
            end = self.buffer.find(b"\n")
            size = len(self.buffer) if end < 0 else end + 1
            data = bytes(self.buffer[:size])
            del self.buffer[:size]
            return data

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class FakeProcess:
    """Each construction gets a fresh stdout/writes list -- important here because
    wake() constructs a brand-new FakeProcess per _spawn(), just like a real Popen."""

    def __init__(self, on_write=None):
        self.stdout = BlockingStream(READY + b"STAT 0 0 0 0\n")
        self.stdin = self
        self.on_write = on_write or (lambda _p, _f: None)
        self.writes = []
        self.returncode = None
        self.spawned_at = time.monotonic()

    def write(self, data):
        self.writes.append(data)
        self.on_write(self, data)
        return len(data)

    def flush(self):
        pass

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self.stdout.close()

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.terminate()


def respond_echo(process, frame):
    """Auto-reply DONE to every SUBMIT so generate() returns quickly."""
    fields = frame.split(b"\n", 1)[0].split()
    if fields and fields[0] == b"SUBMIT":
        request_id = fields[1]
        process.stdout.feed(b"DATA " + request_id + b" 2\nok\n")
        process.stdout.feed(b"DONE " + request_id + b" STAT 1 1.0 0 1.0 1 0\n")


class SpawnCountingPopen:
    """Records every construction so tests can assert "exactly one spawn"."""

    def __init__(self, on_write=respond_echo):
        self.count = 0
        self.on_write = on_write
        self.processes = []

    def __call__(self, *args, **kwargs):
        self.count += 1
        process = FakeProcess(self.on_write)
        self.processes.append(process)
        return process


class EngineSleepWakeTest(unittest.TestCase):
    def _engine(self, popen):
        # Patch stays active for the whole test, not just construction: wake()
        # calls subprocess.Popen again later, on the next request after sleep.
        patcher = patch("openai_server.subprocess.Popen", popen)
        patcher.start()
        self.addCleanup(patcher.stop)
        return Engine("engine", "model")

    def test_sleep_if_idle_tears_down_and_wake_respawns(self):
        popen = SpawnCountingPopen()
        engine = self._engine(popen)
        self.assertEqual(popen.count, 1)
        self.assertFalse(engine.sleeping)

        self.assertTrue(engine.sleep_if_idle())
        self.assertTrue(engine.sleeping)
        self.assertIsNone(engine.process)
        self.assertEqual(popen.processes[0].returncode, 0)  # terminated

        # sleep_if_idle() again is a no-op (already asleep) -- idempotent.
        self.assertFalse(engine.sleep_if_idle())

        stats = engine.generate("hello", 4, 0.0, 1.0, lambda _t: None)
        self.assertEqual(popen.count, 2)  # exactly one respawn
        self.assertFalse(engine.sleeping)
        self.assertIsNotNone(engine.last_wake_seconds)
        self.assertEqual(stats["completion_tokens"], 1)
        engine.close()

    def test_busy_generate_blocks_sleep(self):
        """A generate() in flight must never be killed out from under it: the
        DONE frame is withheld until after sleep_if_idle() has had its chance,
        proving the sleep call actually backed off rather than winning a race."""
        release = threading.Event()

        def respond_hold(process, frame):
            fields = frame.split(b"\n", 1)[0].split()
            if fields and fields[0] == b"SUBMIT":
                request_id = fields[1]

                def deferred():
                    release.wait(timeout=2)
                    process.stdout.feed(b"DATA " + request_id + b" 2\nok\n")
                    process.stdout.feed(b"DONE " + request_id + b" STAT 1 1.0 0 1.0 1 0\n")

                threading.Thread(target=deferred, daemon=True).start()

        popen = SpawnCountingPopen(on_write=respond_hold)
        engine = self._engine(popen)

        result = {}

        def do_generate():
            result["stats"] = engine.generate("hello", 4, 0.0, 1.0, lambda _t: None)

        thread = threading.Thread(target=do_generate)
        thread.start()
        # Let generate() reach _enter_busy() and increment busy before we probe.
        deadline = time.monotonic() + 2
        while engine.busy == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(engine.busy, 1)

        # The idle timer firing "at the wrong time" must not tear anything down.
        self.assertFalse(engine.sleep_if_idle())
        self.assertIsNotNone(engine.process)
        self.assertEqual(popen.count, 1)

        release.set()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result["stats"]["completion_tokens"], 1)
        self.assertEqual(engine.busy, 0)
        engine.close()

    def test_concurrent_wakers_spawn_exactly_once(self):
        popen = SpawnCountingPopen()
        engine = self._engine(popen)
        self.assertTrue(engine.sleep_if_idle())
        self.assertEqual(popen.count, 1)

        results = []
        errors = []
        barrier = threading.Barrier(4)

        def worker():
            try:
                barrier.wait(timeout=2)
                stats = engine.generate("hi", 4, 0.0, 1.0, lambda _t: None)
                results.append(stats)
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 4)
        # Exactly one respawn serves all four concurrent post-sleep requests.
        self.assertEqual(popen.count, 2)
        engine.close()

    def test_close_from_sleeping_is_a_noop(self):
        popen = SpawnCountingPopen()
        engine = self._engine(popen)
        self.assertTrue(engine.sleep_if_idle())
        engine.close()  # must not raise (e.g. NoneType.poll())
        self.assertTrue(engine.closed)

    def test_enter_busy_after_close_raises(self):
        popen = SpawnCountingPopen()
        engine = self._engine(popen)
        engine.sleep_if_idle()
        engine.close()
        with self.assertRaisesRegex(RuntimeError, "shutting down"):
            engine.generate("hi", 4, 0.0, 1.0, lambda _t: None)


class SchedulerIdleTimerTest(unittest.TestCase):
    def test_default_zero_never_arms_timer(self):
        scheduler = GenerationScheduler(sleep_idle_seconds=0)
        scheduler.engine = object()  # any non-None sentinel; must never be touched
        with scheduler.admit():
            pass
        self.assertIsNone(scheduler._idle_timer)
        scheduler.close()

    def test_timer_fires_after_idle_window_and_calls_sleep(self):
        calls = []

        class StubEngine:
            def sleep_if_idle(self):
                calls.append(time.monotonic())
                return True

        scheduler = GenerationScheduler(sleep_idle_seconds=0.05)
        scheduler.engine = StubEngine()
        with scheduler.admit():
            pass
        deadline = time.monotonic() + 2
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(calls), 1)
        scheduler.close()

    def test_new_admission_cancels_pending_timer(self):
        calls = []

        class StubEngine:
            def sleep_if_idle(self):
                calls.append(time.monotonic())
                return True

        scheduler = GenerationScheduler(sleep_idle_seconds=0.05)
        scheduler.engine = StubEngine()
        with scheduler.admit():
            pass
        # A new request arrives just under the idle window: the timer must be
        # cancelled, not allowed to fire while this one is admitted/in flight.
        time.sleep(0.03)
        with scheduler.admit():
            time.sleep(0.08)  # longer than sleep_idle_seconds, but we're active
            self.assertEqual(calls, [])
        # Now idle again: the timer re-arms and fires.
        deadline = time.monotonic() + 2
        while not calls and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(calls), 1)
        scheduler.close()


if __name__ == "__main__":
    unittest.main()
