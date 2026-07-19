"""Deterministic lifecycle tests for the optional ML bridge.

These tests inject tiny fake engines.  They never import the real ML package,
open a microphone, read model artifacts, or touch the network.
"""

import asyncio
import sys
import threading
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml_bridge import MLBridge  # noqa: E402


async def _unused_dispatch(event, source_default="pretrained"):
    return event, source_default


def _wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_injected_engine_constructs_without_importing_real_ml():
    constructed = []

    class FakeEngine:
        def __init__(self, **kwargs):
            constructed.append(kwargs)

        def run(self, stop_event):
            stop_event.wait(timeout=1.0)

    bridge = MLBridge(engine_factory=FakeEngine)
    loop = asyncio.new_event_loop()
    try:
        bridge.start(loop, _unused_dispatch)
        assert bridge.available is True
        assert bridge.alive is True
        assert bridge.last_error is None
        assert bridge.engine is not None
        assert constructed and callable(constructed[0]["on_event"])
    finally:
        bridge.stop()
        loop.close()


def test_engine_construction_failure_degrades_health_state():
    def corrupt_alarm_head(**_kwargs):
        raise RuntimeError("corrupt alarm head")

    bridge = MLBridge(engine_factory=corrupt_alarm_head)
    loop = asyncio.new_event_loop()
    try:
        bridge.start(loop, _unused_dispatch)
    finally:
        loop.close()

    assert bridge.available is False
    assert bridge.alive is False
    assert bridge.engine is None
    assert bridge.last_error == "engine init failed: corrupt alarm head"


def test_listener_failure_is_reported_and_not_alive():
    attempted = threading.Event()

    class FailingEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, stop_event):
            attempted.set()
            raise RuntimeError("input stream disconnected")

    bridge = MLBridge(engine_factory=FailingEngine)
    loop = asyncio.new_event_loop()
    try:
        bridge.start(loop, _unused_dispatch)
        assert attempted.wait(timeout=1.0)
        assert _wait_until(lambda: not bridge.alive)
    finally:
        bridge.stop()
        loop.close()

    assert bridge.available is True
    assert bridge.alive is False
    assert bridge.last_error == "listener died: input stream disconnected"


def test_stop_signals_listener_and_joins_thread():
    running = threading.Event()
    stopped = threading.Event()

    class BlockingEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, stop_event):
            running.set()
            stop_event.wait(timeout=1.0)
            stopped.set()

    bridge = MLBridge(engine_factory=BlockingEngine)
    loop = asyncio.new_event_loop()
    try:
        bridge.start(loop, _unused_dispatch)
        assert running.wait(timeout=1.0)
        bridge.stop(timeout=1.0)
    finally:
        loop.close()

    assert stopped.is_set()
    assert bridge.alive is False


def test_trained_source_passes_through_listener_callback():
    event = {
        "label": "fire_smoke_alarm",
        "urgency": "high",
        "confidence": 0.93,
        "source": "trained",
    }

    class EmittingEngine:
        def __init__(self, on_event):
            self.on_event = on_event

        def run(self, stop_event):
            self.on_event(event)

    async def exercise():
        received = []
        delivered = asyncio.Event()

        async def dispatch(raw, source_default="pretrained"):
            received.append((raw, source_default))
            delivered.set()

        bridge = MLBridge(engine_factory=EmittingEngine)
        bridge.start(asyncio.get_running_loop(), dispatch)
        await asyncio.wait_for(delivered.wait(), timeout=1.0)
        bridge.stop()
        return received

    assert asyncio.run(exercise()) == [(event, "trained")]


def test_dispatch_failure_is_reported_without_stopping_listener(capsys):
    emitted = threading.Event()

    class EmittingBlockingEngine:
        def __init__(self, on_event):
            self.on_event = on_event

        def run(self, stop_event):
            self.on_event({
                "label": "fire_smoke_alarm",
                "urgency": "high",
                "source": "trained",
            })
            emitted.set()
            stop_event.wait(timeout=1.0)

    async def exercise():
        release = asyncio.Event()
        dispatch_started = asyncio.Event()

        async def failing_dispatch(_event, source_default="pretrained"):
            dispatch_started.set()
            await release.wait()
            raise RuntimeError("sink exploded")

        bridge = MLBridge(engine_factory=EmittingBlockingEngine)
        bridge.start(asyncio.get_running_loop(), failing_dispatch)
        assert emitted.wait(timeout=1.0)
        await asyncio.wait_for(dispatch_started.wait(), timeout=1.0)
        with bridge._dispatch_lock:
            assert len(bridge._dispatch_futures) == 1
        release.set()
        await asyncio.sleep(0.01)
        with bridge._dispatch_lock:
            assert bridge._dispatch_futures == set()
        state = (bridge.alive, bridge.last_error)
        bridge.stop()
        return state

    alive, last_error = asyncio.run(exercise())

    assert alive is True
    assert last_error == "dispatch failed: sink exploded"
    assert "[ml] dispatch failed: sink exploded" in capsys.readouterr().err


def test_stop_timeout_is_reported_while_listener_is_still_alive(capsys):
    running = threading.Event()
    release = threading.Event()

    class StuckEngine:
        def __init__(self, **_kwargs):
            pass

        def run(self, stop_event):
            running.set()
            release.wait(timeout=1.0)

    bridge = MLBridge(engine_factory=StuckEngine)
    loop = asyncio.new_event_loop()
    try:
        bridge.start(loop, _unused_dispatch)
        assert running.wait(timeout=1.0)
        bridge.stop(timeout=0.01)
        state = (bridge.alive, bridge.last_error)
    finally:
        release.set()
        bridge.stop(timeout=1.0)
        loop.close()

    assert state == (True, "listener stop timed out after 0.01s")
    assert "[ml] listener stop timed out after 0.01s" in capsys.readouterr().err
