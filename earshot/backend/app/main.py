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
    GET  /rules              per-sound rules (device-global)
    PUT  /rules/{label}      set on/off + urgency override

Accounts (MongoDB; 503 when EARSHOT_MONGO_URI is unset):
    POST /auth/register      create account + log in
    POST /auth/login         log in (sets httpOnly session cookie)
    POST /auth/logout        log out (records logout time)
    GET  /auth/me            current user
    GET  /auth/sessions      this user's login/logout history
    GET  /me/rules           per-user rules
    PUT  /me/rules/{label}   set a per-user rule
    GET  /me/prefs           per-user preferences
    PUT  /me/prefs           set preferences (own ntfy topic, shown categories)
"""

import asyncio
import contextlib
from pathlib import Path

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request,
                     Response, UploadFile, WebSocket, WebSocketDisconnect)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Teach uploads are raw audio held in memory while processed: bound them.
REQUIRED_CLIPS = 3
MAX_CLIP_BYTES = 5 * 1024 * 1024          # ~2.5 min of 16 kHz 16-bit mono
TEACH_TIMEOUT_S = 30.0

from . import auth, config
from .authstore import UserStore
from .core import Dispatcher, RecentEvents, Rules, _VALID_URGENCY
from .ml_bridge import MLBridge
from .sinks import Alerts, EventHub, pi_alert, push


async def _build_user_store():
    """Create the Mongo-backed UserStore, or (None, None) when auth is off.

    Split out so tests can monkeypatch it to inject an in-memory mongomock
    database and run the real store code without a mongod.
    """
    if not config.MONGO_URI:
        return None, None
    from motor.motor_asyncio import AsyncIOMotorClient
    client = AsyncIOMotorClient(config.MONGO_URI)
    store = UserStore(client[config.MONGO_DB])
    await store.ensure_indexes()
    return store, client


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    import httpx

    hub = EventHub()
    recent = RecentEvents()
    rules = Rules()
    alerts = Alerts()
    client = httpx.AsyncClient(timeout=3.0)

    async def push_sink(event, profile):
        # Phone push and the Pi wearable ride the same sink slot; gather so
        # a dead Pi can't delay the phone (and vice versa).
        await asyncio.gather(push(client, event, profile),
                             pi_alert(client, event))

    dispatcher = Dispatcher(recent, rules, hub.broadcast, alerts.alert, push_sink)
    bridge = MLBridge()
    bridge.start(asyncio.get_running_loop(), dispatcher.dispatch)

    # Per-user accounts (optional; only when EARSHOT_MONGO_URI is set).
    try:
        users, mongo_client = await _build_user_store()
    except Exception as exc:   # bad URI / Mongo down must not brick the server
        print(f"[auth] user store unavailable ({exc}); auth disabled",
              file=__import__("sys").stderr)
        users, mongo_client = None, None

    app.state.hub = hub
    app.state.recent = recent
    app.state.rules = rules
    app.state.alerts = alerts
    app.state.dispatcher = dispatcher
    app.state.bridge = bridge
    app.state.users = users
    try:
        yield
    finally:
        # Shutdown is coordinated: signal + join the ML listener (which owns
        # the microphone), then release hardware and the HTTP client.
        bridge.stop()
        alerts.close()
        await client.aclose()
        if mongo_client is not None:
            mongo_client.close()


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
    bridge = app.state.bridge
    return {
        "status": "ok",
        # available = the ML package imported and constructed;
        # alive = the listener thread is running RIGHT NOW. A dead mic or
        # crashed worker shows alive=false instead of a false green.
        "ml": {
            "available": bridge.available,
            "alive": bridge.alive,
            "last_error": bridge.last_error,
        },
        "gpio_mock": app.state.alerts.mock,
        "clients": app.state.hub.count,
        "ntfy": bool(config.NTFY_TOPIC),
        "auth": app.state.users is not None,
    }


@app.get("/events/recent")
async def recent_events(limit: int = 50):
    return app.state.recent.list(limit=limit)


class DebugEvent(BaseModel):
    label: str | None = None
    urgency: str | None = None
    confidence: float = 1.0
    source: str | None = None   # e.g. "taught" to demo a learned-sound row


@app.post("/debug/event")
async def debug_event(body: DebugEvent | None = None):
    """Fire a fake event through the full fan-out — the 'done when' driver.

    `accepted` means the event passed rules and was recorded; `delivery`
    reports what each sink actually did. They are deliberately not the
    same thing.
    """
    body = body or DebugEvent()
    event, delivery = await app.state.dispatcher.dispatch({
        "label": body.label or config.DEBUG_DEFAULT["label"],
        "urgency": body.urgency or config.DEBUG_DEFAULT["urgency"],
        "confidence": body.confidence,
        "source": body.source or "debug",
    }, source_default="debug")
    return {"accepted": event is not None, "event": event,
            "delivery": delivery}


@app.post("/teach")
async def teach(name: str = Form(...), clips: list[UploadFile] = File(...)):
    if len(clips) != REQUIRED_CLIPS:
        raise HTTPException(
            status_code=422,
            detail=f"teach requires exactly {REQUIRED_CLIPS} clips; "
                   f"got {len(clips)}")
    blobs = []
    for clip in clips:
        data = await clip.read()
        if len(data) > MAX_CLIP_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"clip {clip.filename!r} exceeds "
                       f"{MAX_CLIP_BYTES // (1024 * 1024)} MB")
        blobs.append((clip.filename, data))
    try:
        # Decode + inference are CPU-bound: run off the event loop with a
        # deadline so a wedged teach can't stall the API.
        learned = await asyncio.wait_for(
            asyncio.to_thread(app.state.bridge.teach, name, blobs),
            timeout=TEACH_TIMEOUT_S)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="teach timed out")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
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
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=503,
                            detail=f"could not persist rule: {exc}")


# ======================================================================
# Accounts + per-user data (MongoDB). These endpoints 503 when auth is off.
# The live alert path (/ws, /debug/event, device /rules) is NOT gated — a
# login must never stand between a user and a smoke alarm.
# ======================================================================

def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=config.SESSION_COOKIE, value=token, httponly=True,
        samesite="lax", secure=config.SESSION_COOKIE_SECURE,
        max_age=config.SESSION_TTL_DAYS * 24 * 3600, path="/")


class Credentials(BaseModel):
    username: str
    password: str
    display_name: str | None = None


@app.post("/auth/register")
async def register(body: Credentials, request: Request, response: Response):
    store = auth.get_store(request)
    if not body.username.strip():
        raise HTTPException(status_code=422, detail="username is required")
    try:
        auth.validate_password(body.password)
        user = await store.create_user(
            body.username, auth.hash_password(body.password),
            body.display_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    # Auto-login on register.
    token = auth.new_session_token()
    await store.create_session(user["_id"], auth.hash_token(token),
                               user_agent="register")
    _set_session_cookie(response, token)
    return auth.public_user(user)


@app.post("/auth/login")
async def login(body: Credentials, request: Request, response: Response):
    store = auth.get_store(request)
    user = await store.get_user_by_username(body.username)
    if user is None or not auth.verify_password(body.password,
                                                user["password_hash"]):
        # Same message either way: don't reveal which accounts exist.
        raise HTTPException(status_code=401,
                            detail="invalid username or password")
    token = auth.new_session_token()
    await store.create_session(
        user["_id"], auth.hash_token(token),
        user_agent=request.headers.get("user-agent"))
    _set_session_cookie(response, token)
    return auth.public_user(user)


@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    store = auth.get_store(request)
    token = request.cookies.get(config.SESSION_COOKIE)
    if token:
        await store.end_session(auth.hash_token(token))
    response.delete_cookie(config.SESSION_COOKIE, path="/")
    return {"ok": True}


@app.get("/auth/me")
async def me(user: dict = Depends(auth.current_user)):
    return auth.public_user(user)


@app.get("/auth/sessions")
async def my_sessions(request: Request,
                      user: dict = Depends(auth.current_user)):
    """This user's login/logout history."""
    store = auth.get_store(request)
    return await store.list_sessions(user["_id"])


@app.get("/me/rules")
async def get_my_rules(request: Request,
                       user: dict = Depends(auth.current_user)):
    store = auth.get_store(request)
    return await store.get_user_rules(user["_id"])


@app.put("/me/rules/{label}")
async def put_my_rule(label: str, body: RuleUpdate, request: Request,
                      user: dict = Depends(auth.current_user)):
    if body.urgency is not None and body.urgency not in _VALID_URGENCY:
        raise HTTPException(status_code=422,
                            detail=f"invalid urgency {body.urgency!r}")
    store = auth.get_store(request)
    return await store.set_user_rule(user["_id"], label, body.enabled,
                                     body.urgency)


class Prefs(BaseModel):
    ntfy_topic: str | None = None
    shown_categories: list[str] | None = None


@app.get("/me/prefs")
async def get_my_prefs(request: Request,
                       user: dict = Depends(auth.current_user)):
    store = auth.get_store(request)
    return await store.get_prefs(user["_id"])


@app.put("/me/prefs")
async def put_my_prefs(body: Prefs, request: Request,
                       user: dict = Depends(auth.current_user)):
    store = auth.get_store(request)
    return await store.set_prefs(user["_id"],
                                 body.model_dump(exclude_none=True))


def main():
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
