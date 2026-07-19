# Earshot backend — one event in, four alerts out

FastAPI switchboard on the Pi. Any event — real (from ML) or fake (debug
endpoint) — becomes **light, buzz, phone push, and a dashboard row** within
1 second. Broadcasts over WebSocket, pushes to phones via ntfy, drives the
RGB LED + vibration motor over GPIO.

This is demo software, not a certified alarm or life-safety service. Do not
expose its unauthenticated debug, teach, or rule endpoints to an untrusted
network, and never use it to replace approved alarms or emergency procedures.

Files: [config.py](app/config.py) (pins, ntfy, alert profiles — the tuning
knobs), [core.py](app/core.py) (events | recent | rules | dispatch),
[sinks.py](app/sinks.py) (WebSocket | GPIO | ntfy), [ml_bridge.py](app/ml_bridge.py)
(optional ML integration), [main.py](app/main.py) (FastAPI app).

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ../ml

export EARSHOT_MODEL_DIR="$HOME/.local/share/earshot/models"
mkdir -p "$EARSHOT_MODEL_DIR"
earshot download
python -m pip check

export EARSHOT_NTFY_TOPIC=earshot-<pick-something-unguessable>   # optional; push off if unset
python -m app.main            # serves on 0.0.0.0:8000
```

Use one environment for the backend and ML runtime. Installing only
`requirements.txt` leaves NumPy, sounddevice, and LiteRT unavailable and makes
`/healthz` correctly report ML as unavailable. The Pi does not need the
Windows-only scikit-learn training extra.

Off a Pi, GPIO auto-falls back to a logging mock, so it runs on a laptop for
development. On the Pi it drives the real pins (R=17, G=27, B=22, motor=18).

## Prove it works (the "done when")

```bash
curl -X POST localhost:8000/debug/event \
     -H 'content-type: application/json' \
     -d '{"label":"smoke_alarm","urgency":"high"}'
```
→ LED turns red + strobes, motor does a long buzz, the phone push arrives (if
ntfy configured), and every connected dashboard gets a new row — all at once.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| WS   | `/ws`            | broadcasts every event as JSON (dashboard + wearable) |
| GET  | `/healthz`       | server + ML + GPIO + ntfy status |
| GET  | `/events/recent?limit=` | recent events (in-memory) |
| POST | `/debug/event`   | fire a fake event — build without waiting on ML |
| POST | `/teach`         | `name` + 3 audio files → ML teach |
| GET  | `/sounds`        | taught sounds |
| GET  | `/rules`         | per-sound rules |
| PUT  | `/rules/{label}` | `{enabled, urgency}` — mute or override, persisted to JSON |

## Urgency → alert (edit in [config.py](app/config.py))

| urgency | LED | motor | ntfy |
|---------|-----|-------|------|
| high    | red, strobe | strobe + long buzz | urgent |
| medium  | yellow, pulse | one pulse | high |
| low     | blue, blink | short tick | default |

Rules can mute a sound or override its urgency at runtime.

## Networking (read this before the demo)

Venue Wi-Fi almost always isolates clients, so phones can't reach the Pi on
it. **Run everything on a phone hotspot from minute one, including the demo.**
ntfy still works on any network because it only needs outbound internet.

## ML integration

A compatible trained head emits the same public event used by the YAMNet
fallback:

```json
{
  "label": "smoke_alarm",
  "urgency": "high",
  "source": "trained"
}
```

The source is preserved through dispatch, and high urgency receives the top
GPIO/push priority. Incoming `fire_alarm` and `fire_smoke_alarm` events are
canonicalized to `smoke_alarm`. Saved rules are exposed under that same key;
an existing `smoke_alarm` rule wins, otherwise one legacy alarm rule is
migrated deterministically.

The backend runs standalone (debug events) with no ML present. When the
sibling `../ml` package is importable and its model is downloaded, live mic
events flow in automatically and `/teach` works — including while detection
is live (the ML serializes interpreter access internally).

`/teach` requires exactly 3 clips (max 5 MB each), runs decode + inference
off the event loop with a 30 s deadline, and deletes the temporary audio
files whether teaching succeeds or fails. `/healthz` reports `ml.alive`
(listener thread actually running) separately from `ml.available` (package
imported), plus engine, listener, asynchronous-dispatch, or stop-timeout
errors. `smoke_alarm` and every configured event label are reserved from
both the CLI and direct/backend teach API.

## Tests (no hardware or network)

```bash
python -m pip install pytest
python -m pytest tests/ -q
```

The tests inject fake ML engines and sinks. They do not open a microphone,
load a model, drive GPIO, or send network notifications.
