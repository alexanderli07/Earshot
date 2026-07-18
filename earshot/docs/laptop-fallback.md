# Laptop fallback plan — Earshot with no hardware at all

If the Pi dies, the mic module never arrives, the breadboard shorts, or the
hardware desk runs dry: **the entire Earshot demo runs on one laptop.**
Every component already degrades gracefully; this document is the runbook.

## What replaces what

| Hardware | Laptop replacement | How |
|---|---|---|
| Raspberry Pi | the laptop itself | ML + backend are plain Python; `ai-edge-litert` replaces `tflite-runtime` automatically off-ARM |
| USB / I2S microphone | built-in laptop mic | `sounddevice` uses the default input; capture falls back to the mic's native rate and resamples if 16 kHz is refused |
| RGB LED + vibration motor (GPIO) | **virtual puck page** (`/ui/virtual-puck.html`) | browser rendering of the same urgency colors, strobe/pulse/blink patterns, motor timelines, and priority rules; backend GPIO auto-mocks off-Pi |
| Wearable wristband phone | second browser window (`/ui/wearable.html`) | full-screen flash works everywhere; vibration is phone-only, the flash carries the demo |
| Phone push (ntfy app) | ntfy web app in a browser tab | https://ntfy.sh/&lt;your-topic&gt; subscribes in the browser — push demo with zero phones |
| Sound sources (doorbell, smoke alarm) | phone/second laptop playing YouTube clips at the laptop mic | same as the planned demo, just aimed at the laptop |

Nothing else changes: dashboard, teach mode, rules, WebSocket feed, and the
debug endpoint are already hardware-free.

## Runbook

One command (macOS/Linux, from the `earshot/` folder):

```bash
./laptop-demo.sh
```

It creates a venv, installs the ML package + backend deps, downloads the
model on first run, starts the backend, and opens the dashboard and virtual
puck. Ctrl-C stops everything.

Manual equivalent (also the Windows path, in PowerShell-adapted form):

```bash
# 1. ML — install and fetch the model (once, needs internet)
python3 -m venv .venv && source .venv/bin/activate
pip install ./ml
earshot download

# 2. Backend — serves the API and all frontend pages
pip install -r backend/requirements.txt
cd backend && python -m app.main          # http://localhost:8000
```

Open three browser windows:

- `http://localhost:8000/ui/dashboard.html` — feed + teach + rules
- `http://localhost:8000/ui/virtual-puck.html` — the "hardware"
- `http://localhost:8000/ui/wearable.html` — tap ARM (optional fourth: the
  ntfy web app if you exported `EARSHOT_NTFY_TOPIC`)

Smoke-test the full fan-out without ML:

```bash
curl -X POST localhost:8000/debug/event \
     -H 'content-type: application/json' \
     -d '{"label":"smoke_alarm","urgency":"high"}'
```

Expected within a second: dashboard row appears, virtual puck strobes red
and buzzes the motor element, wearable floods red.

**macOS:** grant the terminal Microphone permission
(System Settings → Privacy & Security → Microphone) or capture is silent
zeros. **Windows:** Settings → Privacy → Microphone → allow desktop apps.

## The 30-second demo, laptop edition

1. Play a doorbell clip from a phone at the laptop — the virtual puck
   flashes blue and the dashboard logs DOORBELL before the clip ends.
2. Play a smoke-alarm clip — red strobe + long buzz on the puck, urgent row
   on the dashboard (and an ntfy web notification if configured).
3. Teach mode on the dashboard: record three claps on the laptop mic, then
   clap once — Earshot names a sound that didn't exist a minute ago.

Optional flourish: fire a low-urgency debug event *during* the smoke-alarm
strobe to show the priority latch — the microwave ding cannot interrupt the
alarm, on the virtual puck exactly as on real GPIO.

## Honest limitations

- The virtual puck re-implements the pattern timings client-side; it mirrors
  the GPIO controller's behavior but is a rendering, not the same code path.
  The backend's GPIO layer still runs (as the log mock), so the real code is
  exercised — the puck is presentation.
- Laptop-mic acoustics differ from the Pi mic; thresholds tuned on one may
  need a nudge on the other (`ml/earshot_ml/config.py`).
- The under-two-second latency target measured on a laptop says nothing
  about the Pi. Say so if asked; don't claim Pi numbers from laptop runs.
- Browser vibration only works on Android Chrome, so on a laptop the motor
  is visual (the shaking element). That's the point of the virtual puck.

## When hardware partially exists

Mix and match: a real Pi with no LED/motor can still drive the virtual puck
(open it against the Pi's IP); a laptop with an Arduino vibro can keep the
physical buzz. Each alert sink is independent — use whatever survived the
weekend.
