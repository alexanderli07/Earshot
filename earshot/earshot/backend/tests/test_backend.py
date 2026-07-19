"""Backend tests. Two groups:

  - pure logic (rules, recent buffer, normalize, dispatch fan-out with fakes)
  - HTTP/WebSocket integration via FastAPI TestClient (debug event -> all sinks)

Run: python -m pytest tests/ -q      (or: python tests/test_backend.py)
GPIO auto-mocks off-Pi; ntfy is monkeypatched; no network or hardware needed.
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, core          # noqa: E402
from app.core import Dispatcher, RecentEvents, Rules, normalize_event  # noqa: E402

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


def test_normalize_maps_legacy_alarm_labels():
    assert normalize_event({"label": "fire_alarm"})["label"] == "smoke_alarm"
    assert (
        normalize_event({"label": "fire_smoke_alarm"})["label"]
        == "smoke_alarm"
    )


def test_normalize_rejects_bad_urgency():
    assert normalize_event({"label": "x", "urgency": "nuclear"})["urgency"] \
        == config.DEFAULT_URGENCY


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


def test_rules_persist_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rules.json"
        Rules(path=path).set("kettle", enabled=False, urgency=None)
        assert Rules(path=path).all() == {
            "kettle": {"enabled": False, "urgency": None}}


def test_rules_canonicalize_legacy_alarm_labels():
    rules = Rules(path=None)
    rules.set("fire_alarm", enabled=False, urgency="low")
    assert rules.all() == {
        "smoke_alarm": {"enabled": False, "urgency": "low"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rules.json"
        path.write_text(json.dumps({
            "fire_alarm": {"enabled": True, "urgency": "low"},
            "fire_smoke_alarm": {"enabled": False, "urgency": "medium"},
        }))
        assert Rules(path=path).all() == {
            "smoke_alarm": {"enabled": False, "urgency": "medium"},
        }


def test_dispatch_fans_out_to_all_sinks():
    calls = {"broadcast": [], "alert": [], "push": []}

    async def broadcast(ev):
        calls["broadcast"].append(ev)

    def alert(urgency):
        calls["alert"].append(urgency)

    async def push(ev, prof):
        calls["push"].append((ev, prof))

    recent = RecentEvents()
    d = Dispatcher(recent, Rules(path=None), broadcast, alert, push)
    ev = asyncio.run(d.dispatch({"label": "smoke_alarm", "urgency": "high"}))
    assert ev["urgency"] == "high"
    assert len(calls["broadcast"]) == 1
    assert calls["alert"] == ["high"]
    assert len(calls["push"]) == 1
    assert recent.list()[0]["label"] == "smoke_alarm"


def test_dispatch_muted_hits_no_sink():
    calls = []

    async def broadcast(ev):
        calls.append(ev)

    rules = Rules(path=None)
    rules.set("kettle", enabled=False)
    d = Dispatcher(RecentEvents(), rules, broadcast,
                   lambda u: calls.append(u), lambda e, p: _noop())
    out = asyncio.run(d.dispatch({"label": "kettle", "urgency": "low"}))
    assert out is None and calls == []


async def _noop():
    return None


# ======================================================================
# HTTP + WebSocket integration (FastAPI TestClient)
# ======================================================================

def test_debug_event_drives_ws_recent_and_sinks(monkeypatch):
    monkeypatch.setenv("EARSHOT_NTFY_TOPIC", "")   # keep push disabled
    from fastapi.testclient import TestClient
    from app.main import app

    pushes = []
    monkeypatch.setattr("app.main.push",
                        lambda client, ev, prof: pushes.append(ev) or _noop())

    with TestClient(app) as client:
        with client.websocket_connect("/ws") as ws:
            resp = client.post("/debug/event",
                               json={"label": "doorbell", "urgency": "high"})
            assert resp.json()["delivered"] is True
            event = ws.receive_json()               # dashboard row arrived
            assert event["label"] == "doorbell" and event["urgency"] == "high"

        recent = client.get("/events/recent").json()
        assert recent[0]["label"] == "doorbell"
        assert client.get("/healthz").json()["status"] == "ok"


if __name__ == "__main__":
    import traceback
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                # crude monkeypatch shim for direct runs (skip the TestClient one)
                if "monkeypatch" in fn.__code__.co_varnames:
                    print(f"skip {name} (needs pytest)")
                    continue
                fn()
                print(f"ok   {name}")
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
