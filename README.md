# Earshot

**Sound awareness through sight and touch.** A smoke alarm is useless to
someone who can't hear it. Earshot is a Raspberry Pi that listens for the
sounds that matter — smoke alarms, doorbells, knocking, a baby crying,
breaking glass — and turns each one into light, vibration, a phone push, and
a dashboard alert within about a second. Built for Deaf and hard-of-hearing
users, and for anyone whose ears are busy: headphones on, asleep, in the
shower, two rooms away.

Built at **Hack the 6ix 2026** by
[Alexander Li](https://github.com/alexanderli07),
[Kairav Tupil](https://github.com/KairavT), and
[Nishant Shah](https://github.com/nishantshah0) —
**[story and demo on Devpost](https://devpost.com/software/earshot-boihqx)**.

> **Safety:** Earshot is a prototype, not a certified smoke alarm, fire
> alarm, accessibility device, or life-safety system. Classifiers miss events
> and fire false alerts; microphones can be muted, obstructed, or unplugged.
> Never use it to replace approved alarms, emergency procedures, or required
> safeguards.

```
                                          ┌─> wearable alert unit — RGB LED + motor + buzzer
 sound ──> USB mic ──> ML (YAMNet) ──> backend ──> phone push (ntfy)
              on the Raspberry Pi         └─> WebSocket ──> live dashboard + wearable page
```

## What it hears

Out of the box: **smoke/fire alarm, doorbell, knock, baby cry, and glass
break**. The smoke-alarm event listens to YAMNet's smoke-detector and
fire-alarm classes, reinforced by a purpose-trained detector head
([below](#the-trained-smoke-alarm-head)).

Beyond the built-ins, **teach mode** learns any new sound — your kettle, a
dryer buzzer, an unusual doorbell chime — from ~3 recorded examples, in under
a minute, with no retraining and no cloud. Signed in, taught sounds follow
your account across devices ([accounts](#accounts-optional)).

Every event carries an urgency (`high` / `medium` / `low`) that picks the
alert profile — red strobe and a long buzz for a smoke alarm, a gentle blue
blink for low-priority sounds — and per-sound rules can mute or re-prioritize
anything at runtime.

Detection runs entirely on-device: audio never leaves the room. Internet is
needed only for the one-time model download and the optional phone push.

## What's in the repo

| Directory | What it is | Stack |
|-----------|------------|-------|
| [`earshot/ml/`](earshot/ml/) | Offline audio-event detector: YAMNet TFLite scores live 16 kHz mic audio, three recognition paths (built-in classes, trained alarm head, taught sounds) feed a streak + debounce gate. CLI and in-process Python API. | Python, LiteRT/TFLite, NumPy, sounddevice |
| [`earshot/backend/`](earshot/backend/) | The switchboard: one event in, four alerts out — WebSocket broadcast, GPIO or the Pi alert unit, ntfy push, dashboard feed. Optional per-user accounts. | FastAPI, MongoDB (optional) |
| [`earshot/frontend/`](earshot/frontend/) | Static pages served by the backend: `dashboard.html` (live feed, teach flow, rules, login) and `wearable.html` (a phone-based wearable page — full-screen flash + vibration on Android). | TypeScript → plain JS, no bundler |
| [`earshot/pi/`](earshot/pi/) | The physical wearable alert unit: a systemd service on the Pi driving the RGB LED, motor, and buzzer. Power it and it listens. | Python + GPIO |

Each component README has full setup, tuning, and troubleshooting.
[`WAVs.zip`](WAVs.zip) holds the original alarm recordings behind the
training corpus in [`earshot/ml/data/`](earshot/ml/data/); design docs live
in [`earshot/docs/`](earshot/docs/).

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

A new row appears on the dashboard instantly, and the mock "LED" and "motor"
log what they would have done.

### Live detection

For real microphone events, install the ML package and download the pinned,
checksum-verified model:

```bash
cd earshot/ml
pip install ".[test]"
earshot download        # one-time
earshot top5            # sanity-check the mic, then:
earshot run             # print live events
```

When `earshot/ml` is importable and its model is present, the backend picks
up live mic events automatically. See [`earshot/ml/README.md`](earshot/ml/README.md)
for Raspberry Pi and Windows setup, device selection, and threshold tuning.

### The wearable

The physical wearable is the [Pi alert unit](earshot/pi/) — LED, motor, and
buzzer in a 3D-printed case ([hardware](#the-hardware)). The color identifies
the sound; the urgency sets the intensity.

There's also a no-hardware browser version: open
`http://<pi>:8000/ui/wearable.html` on an Android phone (iOS ignores the
vibration API) and tap **ARM** once — browsers only allow vibration and wake
lock after a user gesture. Full-screen color flash, vibration patterns
mirroring the motor.

### Accounts (optional)

Auth is **off by default** — the demo needs no database, and the live alert
path is never behind a login. Set `EARSHOT_MONGO_URI` to enable per-user
rules, preferences, and taught sounds that roam across devices — either a
local MongoDB (one command: `earshot/backend/setup-mongo.sh`, fully offline)
or MongoDB Atlas. Details and the security posture are in
[`earshot/backend/README.md`](earshot/backend/README.md).

## The trained smoke-alarm head

YAMNet stays frozen and emits a 1,024-value embedding per 0.975 s window; a
deterministic logistic-regression head trained on those embeddings
(`earshot train-alarm` / `earshot evaluate-alarm`) is exported as a small NPZ
with embedded checksums, so the Pi runs it with NumPy alone — no
scikit-learn, and a mismatched artifact fails loudly instead of silently
changing predictions.

On the checked-in demo corpus (7 alarm / 10 negative recordings), the
out-of-fold evaluation caught 7/7 alarm groups with one false-positive group
(≈0.21 false events/minute). That is demo evidence, not certification — the
full methodology and caveats are in the
[ML README](earshot/ml/README.md#collect-and-train-an-alarm-corpus).

## Running on the Pi

One environment serves the backend and the ML runtime (no training extras
needed on-device): install `earshot/backend/requirements.txt` plus
`earshot/ml`, run `earshot download`, copy the trained head's NPZ into the
model directory, and start `python -m app.main`. Alerts can drive the Pi's
own GPIO directly, or be forwarded (`EARSHOT_PI_URL`) to the dedicated
[alert unit](earshot/pi/) running as a systemd appliance. Step-by-step:
[ML Pi setup](earshot/ml/README.md#raspberry-pi-setup) ·
[backend](earshot/backend/README.md) · [alert unit](earshot/pi/README.md).

**Demo-day notes:** venue Wi-Fi almost always isolates clients, so phones
can't reach the Pi on it — run everything on a phone hotspot from minute one
(ntfy still works anywhere; it only needs outbound internet). The dashboard
and wearable auto-discover the backend when served from it; opened any other
way, point them at the Pi with `?host=<pi-ip>:8000`.

## The hardware

The wearable alert unit is deliberately simple — hobby parts on a breadboard,
in a 3D-printed enclosure, cordless on a battery pack:

- **Raspberry Pi** running the alert server as a systemd appliance
- **DC motor on a TB6612 motor driver** — the vibration source
- **RGB LED** for the color-coded strobe and a **buzzer** for audible backup
- **Breadboard and jumper wires** tying it together, powered by a **USB
  battery pack**
- **Two 3D-printed plates** (base with standoff pins + slotted top frame)
  that mount the Pi and breadboard rig
- A **laptop** with the microphone, running ML + backend (see below)

## Hackathon realities

Parts of the demo rig are honest workarounds, and we'd rather say so:

- **No USB microphone.** The Pi was supposed to do its own listening; we
  couldn't get a mic. So the laptop listened — microphone, YAMNet, and
  backend all ran there, forwarding every event to the Pi alert unit over
  the hotspot (`EARSHOT_PI_URL`). Because the backend fans out to sinks,
  the relocation was a config change, not a rewrite — and the on-Pi path
  still works when a mic is present.
- **No vibration motor.** None to be found, so a plain DC motor on the
  TB6612 driver does the shaking. It shakes convincingly.
- **Not much training data.** The trained alarm head learned from just 7
  alarm and 10 non-alarm clips — enough to prove the pipeline end to end,
  nowhere near enough to trust (the numbers and caveats
  [above](#the-trained-smoke-alarm-head) are honest about this).
- **Browsers fight wearables.** The phone-based wearable page needs a user
  gesture before vibration and wake lock are allowed — hence its one-tap
  ARM — and iOS ignores the vibration API entirely, so it's Android-only.
  The real wearable is our own hardware, which asks no permission to shake.

## Tests

Every push runs four CI jobs
([workflow](.github/workflows/tests.yml)): the ML fast suite on Python 3.11
and 3.14, a runtime-without-scikit-learn import check, the backend suite, and
the frontend build + tests. None need hardware, a network, or the downloaded
model:

```bash
cd earshot/ml       && python -m pytest -m "not integration" -q
cd earshot/backend  && python -m pytest -q     # needs: pip install pytest mongomock-motor
cd earshot/frontend && npm ci && npm run build && node tests/test_frontend.mjs
```

## What's next

- Dedicated wearable hardware (ESP32 + haptics) to replace the wrist phone.
- Per-room calibration profiles and taught-sound sharing between devices.
- Alert patterns shaped with the Deaf and hard-of-hearing community — the
  people this is actually for.
