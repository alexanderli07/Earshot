# Earshot

Earshot is an offline sound-awareness demo for Raspberry Pi. A microphone feeds
a local YAMNet model, Earshot turns selected sounds into debounced events, and a
FastAPI service fans those events out to a dashboard, wearable page, GPIO alert,
and optional phone notification.

> **Safety:** Earshot is not a certified smoke alarm, fire alarm, accessibility
> device, or life-safety system. It can miss alarms and produce false alerts.
> Never use it to replace approved alarms, emergency procedures, supervision,
> or required safeguards.

## How it fits together

```text
microphone
   -> 16 kHz overlapping windows
   -> frozen YAMNet TFLite model
      -> configured AudioSet event rules
      -> optional trained smoke_alarm logistic head
      -> optional user-taught cosine-similarity sounds
   -> streak/evidence gate + 10 second debounce
   -> FastAPI dispatcher
   -> WebSocket UI + GPIO + optional ntfy push
```

The repository components are:

- `earshot/ml`: offline inference, collection, training, evaluation, and CLI.
- `earshot/backend`: FastAPI event dispatcher and hardware/network sinks.
- `earshot/frontend`: static dashboard and wearable TypeScript applications.
- `.github/workflows/tests.yml`: fast ML, runtime-only, backend, and frontend CI.

The trained alarm detector is deliberately small. YAMNet remains frozen and
produces a 1,024-value embedding for each 0.975-second window. Windows training
uses those embeddings to fit a deterministic logistic-regression head. The Pi
loads only the exported NPZ head; it does not need scikit-learn.

## Windows setup

Run from the repository root in PowerShell:

```powershell
Set-Location .\earshot\ml
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test,train]"
.\.venv\Scripts\python.exe -m pip install -r ..\backend\requirements.txt
.\.venv\Scripts\earshot.exe download
```

This single environment contains the ML runtime, Windows-only training tools,
backend dependencies, and tests. The model download is checksum pinned. Normal
detection is offline after it completes.

Build the frontend once:

```powershell
Set-Location ..\frontend
npm.cmd ci
npm.cmd run build
```

Train and evaluate the demo alarm head from `earshot\ml`:

```powershell
Set-Location ..\ml
.\.venv\Scripts\earshot.exe train-alarm
.\.venv\Scripts\earshot.exe evaluate-alarm
```

Run the backend from `earshot\backend` with the same environment:

```powershell
Set-Location ..\backend
..\ml\.venv\Scripts\python.exe -m app.main
```

Then open `http://localhost:8000/ui/dashboard.html`. Use `localhost` for
browser microphone teaching on a laptop; browsers generally block microphone
capture from an insecure `http://<pi-ip>` origin. The dashboard can still point
at the Pi with `?host=<PI-IP>:8000`.

## Test everything locally

From `earshot\ml`:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not integration" -q
.\.venv\Scripts\python.exe -m pytest tests\test_real_model.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_corpus.py -q -s
.\.venv\Scripts\python.exe -m pip check
```

From `earshot\backend`:

```powershell
..\ml\.venv\Scripts\python.exe -m pytest -q
```

From `earshot\frontend`:

```powershell
npm.cmd run build
$env:EARSHOT_TEST_PYTHON = (Resolve-Path '..\ml\.venv\Scripts\python.exe')
node .\tests\test_frontend.mjs
```

The default CI deliberately excludes the downloaded-model and real-corpus
integration tests. Those remain explicit local acceptance gates because they
need the pinned artifacts and real audio.

## Current corpus evidence

The checked-in demo corpus currently contains 7 alarm WAVs and 10 negative
WAVs. The seed-0 run on the full contiguous timelines measured:

- grouped out-of-fold: 7/7 positive groups, 1/10 negative groups, 0.207 false
  events/minute;
- fitted-head in-sample evaluation: 7/7 positive groups, 0/10 negative groups,
  0 false events/minute;
- deployment threshold: approximately 0.99134.

The sole OOF-triggered negative was
`not_alarm/Clapping_sound_effect.wav`, with two emitted events. OOF predictions
also participate in threshold selection, so this is internal validation rather
than an untouched test set. The in-sample result measures fit, not
generalization. The corpus has no manifest, source provenance, or held-out
recording set and lacks several difficult tonal negatives. Treat these numbers
as demo evidence only.

Source WAVs are preserved exactly. The corpus integration test streams hashes
before and after training/evaluation and fails if any WAV or manifest changes.
New or private recordings should live outside Git through
`EARSHOT_ALARM_DATA_DIR`.

## Raspberry Pi deployment

Use one environment for the backend and ML runtime, without the Windows
training extra:

```bash
cd ~/Earshot/earshot/backend
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ../ml

export EARSHOT_MODEL_DIR="$HOME/.local/share/earshot/models"
mkdir -p "$EARSHOT_MODEL_DIR"
earshot download
python -m pip check
```

Copy `fire_smoke_alarm_head.npz` from Windows into that model directory. The
head embeds checksums for its YAMNet model and class map, so a mismatched Pi
artifact fails clearly instead of silently changing predictions.

```bash
python -m sounddevice
earshot top5 --device <INPUT_INDEX>
earshot run --device <INPUT_INDEX>

cd ~/Earshot/earshot/backend
python -m app.main
```

Training on Windows does not inherently reduce Pi accuracy; both platforms use
the same head and pinned feature extractor. Microphone model, gain, placement,
speaker, distance, and room acoustics do affect accuracy. A held-out Windows
playback pass and a separate target-Pi microphone pass are therefore required.

See the component READMEs for collection manifests, CLI details, API behavior,
frontend hosting, and hardware configuration.
