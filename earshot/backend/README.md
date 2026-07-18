# Earshot backend — one event in, four alerts out

FastAPI switchboard on the Pi. Any event — real (from ML) or fake (debug
endpoint) — becomes **light, buzz, phone push, and a dashboard row** within
1 second. Broadcasts over WebSocket, pushes to phones via ntfy, drives the
RGB LED + vibration motor over GPIO.

Files: [config.py](app/config.py) (pins, ntfy, alert profiles — the tuning
knobs), [core.py](app/core.py) (events | recent | rules | dispatch),
[sinks.py](app/sinks.py) (WebSocket | GPIO | ntfy), [ml_bridge.py](app/ml_bridge.py)
(optional ML integration), [main.py](app/main.py) (FastAPI app).

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export EARSHOT_NTFY_TOPIC=earshot-<pick-something-unguessable>   # optional; push off if unset
python -m app.main            # serves on 0.0.0.0:8000
```

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

The backend runs standalone (debug events) with no ML present. When the
sibling `../ml` package is importable and its model is downloaded, live mic
events flow in automatically and `/teach` works — including while detection
is live (the ML serializes interpreter access internally).

`/teach` requires exactly 3 clips (max 5 MB each), runs decode + inference
off the event loop with a 30 s deadline, and deletes the temporary audio
files whether teaching succeeds or fails. `/healthz` reports `ml.alive`
(listener thread actually running) separately from `ml.available` (package
imported), plus the last listener error if it died.

## Tests (no hardware or network)

```bash
pip install pytest
python -m pytest tests/ -q
```
