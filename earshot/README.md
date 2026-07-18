# Earshot

**Sound awareness through sight and touch.** A Raspberry Pi listens for the
sounds that matter — smoke alarms, doorbells, a baby crying — and turns each
one into light, vibration, a phone push, and a dashboard alert within about a
second. For anyone who can't hear the sound in the moment: deaf and
hard-of-hearing users, headphones on, or just in another room.

```
                                          ┌─> RGB LED + vibration motor (GPIO)
 sound ──> USB mic ──> ML (YAMNet) ──> backend ──> phone push (ntfy)
              on the Raspberry Pi         └─> WebSocket ──> dashboard + wearable
```

## Components

| Directory | What it is | Stack |
|-----------|------------|-------|
| [`ml/`](ml/) | Offline audio-event detector: YAMNet TFLite classifies live 16 kHz mic audio into debounced events, and can be taught new sounds from a few examples. CLI + in-process Python API. | Python, LiteRT/TFLite, NumPy, sounddevice |
| [`backend/`](backend/) | The switchboard: one event in, four alerts out. Drives the LED and motor over GPIO, pushes to phones via ntfy, broadcasts every event over WebSocket. | FastAPI |
| [`frontend/`](frontend/) | Two static pages served by the backend: `dashboard.html` (laptop — live feed, teach flow, per-sound rules) and `wearable.html` (Android wrist phone — full-screen flash + vibration). | TypeScript → plain JS, no bundler |

Each directory has its own README with full setup, tuning, and
troubleshooting details.

## What it hears

Out of the box: **smoke alarm, fire alarm, doorbell, knock, baby cry, glass
break, microwave**. Beyond that, teach mode learns any new sound (a kettle, a
dryer buzzer) from ~3 recorded examples — no retraining, no cloud. Detection
is fully local; internet is needed only for the one-time model download and
optional phone push.

Every event carries an urgency (`high` / `medium` / `low`) that picks the
alert profile — red strobe + long buzz for a smoke alarm, a gentle blue blink
for low-priority sounds. Per-sound rules can mute or re-prioritize at runtime.

## Quick start (no hardware needed)

The backend runs on a laptop — GPIO falls back to a logging mock, and debug
events stand in for the mic — so you can see the whole loop without a Pi.

```bash
# 1. Start the backend
cd earshot/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m app.main                      # serves on 0.0.0.0:8000

# 2. Open the dashboard
#    http://localhost:8000/ui/dashboard.html

# 3. Fire a fake event
curl -X POST localhost:8000/debug/event \
     -H 'content-type: application/json' \
     -d '{"label":"smoke_alarm","urgency":"high"}'
```

A new row appears on the dashboard instantly (and the mock "LED" and "motor"
log what they would have done).

### Live detection

For real mic events, install the ML package and download the pinned model:

```bash
cd earshot/ml
pip install ".[test]"
earshot download        # one-time, checksum-verified
earshot top5            # sanity-check the mic, then:
earshot run             # print live events
```

When `ml/` is importable and its model is present, the backend picks up live
mic events automatically. See [`ml/README.md`](ml/README.md) for Raspberry Pi
and Windows setup, device selection, and threshold tuning.

### The wearable

Open `http://<pi>:8000/ui/wearable.html` on an Android phone (iOS ignores the
vibration API) and tap **ARM** once — browsers only allow vibration and wake
lock after a user gesture.

## Demo-day notes

- **Use a phone hotspot from minute one.** Venue Wi-Fi almost always isolates
  clients, so phones can't reach the Pi on it. ntfy push works on any network
  since it only needs outbound internet.
- The dashboard and wearable auto-discover the backend when served from it;
  opened any other way, set the Pi's `host:port` on the page (or `?host=`).

## Tests

CI runs the ML fast suite on Python 3.11 and 3.14
([workflow](.github/workflows/tests.yml)). None of these need hardware, a
network, or the downloaded model:

```bash
cd ml       && python -m pytest -m "not integration" -q
cd backend  && python -m pytest tests/ -q
cd frontend && node tests/test_frontend.mjs
```

## Safety

Earshot is a prototype, not a certified smoke alarm, fire alarm, or
life-safety device. It must not replace approved alarms, emergency
procedures, or other required safeguards — classifiers miss events and fire
false positives, and microphones can be muted, obstructed, or disconnected.
