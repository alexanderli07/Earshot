"""FastAPI app: the switchboard between ML, GPIO, and every screen.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
    # or: python -m app.main

Endpoints:
    WS   /ws                 broadcast every event as JSON
    GET  /healthz            server + ML + GPIO status
    GET  /events/recent      recent events (in-memory)
    POST /debug/event        fire a fake event (build without waiting on ML)
    POST /teach              name + 3 audio clips -> ML teach
    GET  /sounds             taught sounds
    GET  /rules              per-sound rules
    PUT  /rules/{label}      set on/off + urgency override
"""

import contextlib
from pathlib import Path

from fastapi import (FastAPI, File, Form, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .core import Dispatcher, RecentEvents, Rules
from .ml_bridge import MLBridge
from .sinks import Alerts, EventHub, push


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    import httpx

    hub = EventHub()
    recent = RecentEvents()
    rules = Rules()
    alerts = Alerts()
    client = httpx.AsyncClient(timeout=3.0)

    async def push_sink(event, profile):
        await push(client, event, profile)

    dispatcher = Dispatcher(recent, rules, hub.broadcast, alerts.alert, push_sink)
    bridge = MLBridge()
    bridge.start(asyncio.get_running_loop(), dispatcher.dispatch)

    app.state.hub = hub
    app.state.recent = recent
    app.state.rules = rules
    app.state.alerts = alerts
    app.state.dispatcher = dispatcher
    app.state.bridge = bridge
    try:
        yield
    finally:
        alerts.close()
        await client.aclose()


app = FastAPI(title="Earshot backend", lifespan=lifespan)

# Allow the dashboard/wearable pages to reach the REST endpoints when opened
# standalone (file:// or served from another host) against the Pi.
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

# Optionally serve the sibling frontend/ pages same-origin, so on the Pi you can
# just open http://<pi>:8000/ui/dashboard.html and http://<pi>:8000/ui/wearable.html
_FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")


# ---- WebSocket: every event as JSON ----

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    hub = websocket.app.state.hub
    await hub.connect(websocket)
    # Replay recent events (oldest first) so a fresh dashboard has history.
    for event in reversed(websocket.app.state.recent.list()):
        await websocket.send_json(event)
    try:
        while True:
            await websocket.receive_text()   # ignore input; just detect close
    except WebSocketDisconnect:
        await hub.disconnect(websocket)


# ---- REST ----

@app.get("/healthz")
async def healthz():
    return {
        "status": "ok",
        "ml": app.state.bridge.available,
        "gpio_mock": app.state.alerts.mock,
        "clients": app.state.hub.count,
        "ntfy": bool(config.NTFY_TOPIC),
    }


@app.get("/events/recent")
async def recent_events(limit: int = 50):
    return app.state.recent.list(limit=limit)


class DebugEvent(BaseModel):
    label: str | None = None
    urgency: str | None = None
    confidence: float = 1.0


@app.post("/debug/event")
async def debug_event(body: DebugEvent | None = None):
    """Fire a fake event through the full fan-out — the 'done when' driver."""
    body = body or DebugEvent()
    event = await app.state.dispatcher.dispatch({
        "label": body.label or config.DEBUG_DEFAULT["label"],
        "urgency": body.urgency or config.DEBUG_DEFAULT["urgency"],
        "confidence": body.confidence,
        "source": "debug",
    })
    return {"delivered": event is not None, "event": event}


@app.post("/teach")
async def teach(name: str = Form(...), clips: list[UploadFile] = File(...)):
    blobs = [(c.filename, await c.read()) for c in clips]
    try:
        learned = app.state.bridge.teach(name, blobs)
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "learned": learned}


@app.get("/sounds")
async def sounds():
    return app.state.bridge.learned_sounds()


@app.get("/rules")
async def get_rules():
    return app.state.rules.all()


class RuleUpdate(BaseModel):
    enabled: bool = True
    urgency: str | None = None


@app.put("/rules/{label}")
async def put_rule(label: str, body: RuleUpdate):
    try:
        return app.state.rules.set(label, enabled=body.enabled,
                                   urgency=body.urgency)
    except ValueError as exc:
        return {"error": str(exc)}


def main():
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
