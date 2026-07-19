"""Backend tests. Three groups:

  - pure logic (rules, recent buffer, normalize, dispatch fan-out with fakes)
  - the alert controller (priority latch, preemption, timeline building)
  - HTTP/WebSocket integration via FastAPI TestClient (debug event -> sinks,
    teach bounds, rule statuses)

Run: python -m pytest tests/ -q
GPIO auto-mocks off-Pi; ntfy is monkeypatched; no network or hardware needed.
"""

import asyncio
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, core          # noqa: E402
from app.core import Dispatcher, RecentEvents, Rules, normalize_event  # noqa: E402
from app.ml_bridge import MLBridge  # noqa: E402
from app.sinks import Alerts, build_timeline  # noqa: E402

# ======================================================================
# Pure logic
# ======================================================================


def test_normalize_fills_defaults_and_ids():
    a = normalize_event({"label": "doorbell"})
    b = normalize_event({"label": "doorbell"})
    assert a["urgency"] == config.DEFAULT_URGENCY
    assert a["source"] == "pretrained"
    assert a["id"] != b["id"]                      # unique ids
    assert 0.0 <= a["confidence"] <= 1.0


def test_normalize_rejects_bad_urgency():
    assert normalize_event({"label": "x", "urgency": "nuclear"})["urgency"] \
        == config.DEFAULT_URGENCY


def test_normalize_rejects_non_finite_values():
    nan_event = normalize_event({"label": "x", "confidence": float("nan"),
                                 "timestamp": float("inf")})
    assert nan_event["confidence"] == 1.0
    assert nan_event["timestamp"] > 0            # replaced with now, finite
    import math
    assert math.isfinite(nan_event["timestamp"])


def test_normalize_clamps_confidence_range():
    assert normalize_event({"label": "x", "confidence": 7.3})["confidence"] == 1.0
    assert normalize_event({"label": "x", "confidence": -2})["confidence"] == 0.0


def test_recent_is_newest_first_and_bounded():
    r = RecentEvents(maxlen=3)
    for i in range(5):
        r.add({"id": i})
    assert [e["id"] for e in r.list()] == [4, 3, 2]  # newest first, capped


def test_rules_mute_and_override():
    r = Rules(path=None)
    ev = normalize_event({"label": "kettle", "urgency": "low"})
    assert r.apply(ev) is not None                 # no rule -> passes
    r.set("kettle", enabled=False)
    assert r.apply(ev) is None                      # muted -> dropped
    r.set("doorbell", enabled=True, urgency="high")
    out = r.apply(normalize_event({"label": "doorbell", "urgency": "low"}))
    assert out["urgency"] == "high"                 # overridden


def test_rules_persist_roundtrip(tmp_path):
    path = tmp_path / "rules.json"
    Rules(path=path).set("kettle", enabled=False, urgency=None)
    assert Rules(path=path).all() == {
        "kettle": {"enabled": False, "urgency": None}}


def test_rules_set_rolls_back_memory_on_save_failure(tmp_path):
    class FailingSave(Rules):
        def _save(self, rules):
            raise OSError("disk full")

    r = FailingSave(path=tmp_path / "rules.json")
    with pytest.raises(OSError):
        r.set("kettle", enabled=False)
    # Memory must not diverge from disk: the failed rule is not live.
    assert r.all() == {}
    assert r.apply(normalize_event({"label": "kettle"})) is not None


def test_dispatch_fans_out_and_reports_delivery():
    calls = {"broadcast": [], "alert": [], "push": []}

    async def broadcast(ev):
        calls["broadcast"].append(ev)
        return 2                     # two clients received it

    def alert(urgency):
        calls["alert"].append(urgency)
        return "queued"

    async def push(ev, prof):
        calls["push"].append((ev, prof))
        return True

    recent = RecentEvents()
    d = Dispatcher(recent, Rules(path=None), broadcast, alert, push)
    event, delivery = asyncio.run(
        d.dispatch({"label": "smoke_alarm", "urgency": "high"}))
    assert event["urgency"] == "high"
    assert calls["alert"] == ["high"]
    assert delivery == {
        "websocket": {"ok": True, "clients": 2},
        "gpio": {"ok": True, "detail": "queued"},
        "ntfy": {"ok": True},
    }
    assert recent.list()[0]["label"] == "smoke_alarm"


def test_trained_alarm_preserves_source_and_uses_highest_gpio_priority():
    calls = {"broadcast": [], "alert": [], "push": []}

    async def broadcast(event):
        calls["broadcast"].append(event)
        return 1

    def alert(urgency):
        calls["alert"].append(urgency)
        return "queued"

    async def push(event, profile):
        calls["push"].append((event, profile))
        return None

    dispatcher = Dispatcher(
        RecentEvents(), Rules(path=None), broadcast, alert, push
    )
    event, _delivery = asyncio.run(dispatcher.dispatch({
        "label": "fire_smoke_alarm",
        "urgency": "high",
        "confidence": 0.93,
        "source": "trained",
    }))

    assert event["label"] == "fire_smoke_alarm"
    assert event["source"] == "trained"
    assert event["urgency"] == "high"
    assert calls["alert"] == ["high"]
    assert config.URGENCY_RANK[calls["alert"][0]] == max(
        config.URGENCY_RANK.values()
    )
    assert calls["push"] == [(event, config.ALERT_PROFILES["high"])]


def test_dispatch_reports_all_sink_failures():
    async def broadcast(ev):
        return 0                     # nobody connected

    def alert(urgency):
        raise RuntimeError("gpio wedged")

    async def push(ev, prof):
        return None                  # ntfy not configured

    d = Dispatcher(RecentEvents(), Rules(path=None), broadcast, alert, push)
    event, delivery = asyncio.run(
        d.dispatch({"label": "smoke_alarm", "urgency": "high"}))
    assert event is not None                        # accepted...
    assert delivery["websocket"]["ok"] is False     # ...but nothing delivered
    assert delivery["gpio"]["ok"] is False
    assert "gpio wedged" in delivery["gpio"]["error"]
    assert delivery["ntfy"] == {"configured": False}


def test_dispatch_muted_hits_no_sink():
    calls = []

    async def broadcast(ev):
        calls.append(ev)
        return 1

    rules = Rules(path=None)
    rules.set("kettle", enabled=False)
    d = Dispatcher(RecentEvents(), rules, broadcast,
                   lambda u: calls.append(u), lambda e, p: _noop())
    event, delivery = asyncio.run(
        d.dispatch({"label": "kettle", "urgency": "low"}))
    assert event is None and delivery is None and calls == []


async def _noop():
    return None


# ======================================================================
# Alert controller — priority latch, preemption, timelines
# ======================================================================


def test_admit_policy_latches_higher_urgency():
    admit = Alerts._admit
    assert admit(3, 0, 0)          # high starts when idle
    assert not admit(1, 3, 0)      # low cannot preempt active high
    assert not admit(2, 3, 0)      # medium cannot preempt active high
    assert admit(3, 3, 0)          # a new high replaces an active high
    assert admit(3, 1, 0)          # high preempts active low
    assert not admit(1, 0, 3)      # low cannot jump a pending high either


def test_build_timeline_covers_motor_duration_and_strobes():
    steps = build_timeline(config.ALERT_PROFILES["high"])
    total = sum(duration for duration, _, _ in steps)
    assert total == pytest.approx(sum(config.MOTOR_PATTERNS["long"]))
    led_states = [led for _, led, _ in steps]
    assert True in led_states and False in led_states   # actually strobes
    assert steps[0][2] is True                          # motor starts ON


def test_low_cannot_preempt_active_high_and_owner_clears():
    started = threading.Event()
    release = threading.Event()

    def gated_sleep(duration):
        started.set()
        release.wait(timeout=2)

    alerts = Alerts(sleep=gated_sleep)      # mock mode off-Pi
    try:
        assert alerts.alert("high") == "queued"
        assert started.wait(timeout=2)      # worker is mid-pattern
        assert alerts.alert("low") == "dropped:latched"
        assert alerts.alert("medium") == "dropped:latched"
        assert alerts.alert("high") == "queued"    # equal rank may replace
    finally:
        release.set()
        alerts.close()
    # The single worker was the only writer, and it ended with outputs off.
    assert alerts.trace[-1] == (None, False)


# ======================================================================
# HTTP + WebSocket integration (FastAPI TestClient)
# ======================================================================


class StubMLBridge:
    """HTTP-test bridge that never imports or starts the real ML runtime."""

    def __init__(self):
        self.available = False
        self.last_error = "test stub: ML disabled"
        self.alive = False
        self.started = False
        self.stopped = False

    def start(self, loop, dispatch):
        self.started = True

    def stop(self, timeout=3.0):
        self.stopped = True

    def teach(self, name, blobs):
        if name.strip().casefold() == "fire_smoke_alarm":
            raise ValueError(
                "teach name 'fire_smoke_alarm' conflicts with a trained label"
            )
        raise RuntimeError("ML not available")

    def learned_sounds(self):
        return []


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("EARSHOT_NTFY_TOPIC", "")
    from fastapi.testclient import TestClient
    from app import main

    monkeypatch.setattr(main, "MLBridge", StubMLBridge)
    app = main.app
    with TestClient(app) as test_client:
        assert isinstance(app.state.bridge, StubMLBridge)
        assert app.state.bridge.started is True
        bridge = app.state.bridge
        yield test_client
        assert bridge.stopped is False
    assert bridge.stopped is True


def test_debug_event_drives_ws_recent_and_sinks(client):
    with client.websocket_connect("/ws") as ws:
        resp = client.post("/debug/event",
                           json={"label": "doorbell", "urgency": "high"})
        body = resp.json()
        assert body["accepted"] is True
        assert body["delivery"]["websocket"]["ok"] is True
        assert body["delivery"]["gpio"]["ok"] is True
        assert body["delivery"]["ntfy"] == {"configured": False}
        event = ws.receive_json()               # dashboard row arrived
        assert event["label"] == "doorbell" and event["urgency"] == "high"

    recent = client.get("/events/recent").json()
    assert recent[0]["label"] == "doorbell"
    health = client.get("/healthz").json()
    assert health["status"] == "ok"
    assert set(health["ml"]) == {"available", "alive", "last_error"}


def test_debug_event_can_simulate_taught_source(client):
    resp = client.post("/debug/event",
                       json={"label": "kettle", "urgency": "medium",
                             "source": "taught"})
    assert resp.json()["event"]["source"] == "taught"


def test_health_reports_corrupt_alarm_head_startup_failure(client):
    def corrupt_alarm_head(**_kwargs):
        raise RuntimeError("corrupt alarm head")

    bridge = MLBridge(engine_factory=corrupt_alarm_head)
    loop = asyncio.new_event_loop()
    try:
        bridge.start(loop, _noop)
    finally:
        loop.close()

    original_bridge = client.app.state.bridge
    client.app.state.bridge = bridge
    try:
        health = client.get("/healthz").json()["ml"]
    finally:
        client.app.state.bridge = original_bridge

    assert health == {
        "available": False,
        "alive": False,
        "last_error": "engine init failed: corrupt alarm head",
    }


def test_teach_requires_exactly_three_clips(client):
    wav = (b"RIFF\x24\x00\x00\x00WAVE", "clip.wav")
    for count in (1, 2, 4):
        files = [("clips", (f"c{i}.wav", wav[0], "audio/wav"))
                 for i in range(count)]
        resp = client.post("/teach", data={"name": "kettle"}, files=files)
        assert resp.status_code == 422, f"{count} clips should be rejected"
        assert "exactly 3" in resp.json()["detail"]


def test_teach_reports_ml_unavailable_as_503(client):
    files = [("clips", (f"c{i}.wav", b"RIFF", "audio/wav")) for i in range(3)]
    resp = client.post("/teach", data={"name": "kettle"}, files=files)
    # No model in the test environment: bridge is degraded, not crashed.
    assert resp.status_code == 503


def test_teach_rejects_trained_alarm_label_as_422(client):
    files = [("clips", (f"c{i}.wav", b"RIFF", "audio/wav")) for i in range(3)]
    resp = client.post(
        "/teach", data={"name": "fire_smoke_alarm"}, files=files
    )

    assert resp.status_code == 422
    assert "trained label" in resp.json()["detail"]


def test_invalid_rule_urgency_is_422(client):
    resp = client.put("/rules/doorbell",
                      json={"enabled": True, "urgency": "nuclear"})
    assert resp.status_code == 422
    assert "invalid urgency" in resp.json()["detail"]
