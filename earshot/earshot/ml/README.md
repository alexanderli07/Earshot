# Earshot ML

Earshot ML is a standalone, offline audio-event detector intended for a
Raspberry Pi with a microphone. It converts live 16 kHz mono audio into
debounced event dictionaries, recognizes a configured set of YAMNet classes,
and can learn additional sounds from a few recorded or WAV-file examples.

This directory is the complete Python component. It exposes a command-line
interface and an in-process `EarshotML` API; it does not provide a web server,
database, network protocol, or remote inference service. Internet access is
needed only to download the pinned model and class map. Detection is local
after that.

## Architecture

The main modules are:

- [`config.py`](earshot_ml/config.py): audio geometry, event mappings,
  thresholds, debounce settings, teach-mode settings, and artifact paths.
- [`pipeline.py`](earshot_ml/pipeline.py): WAV helpers, microphone capture,
  bounded buffering, and the YAMNet TFLite interpreter contract.
- [`core.py`](earshot_ml/core.py): event payloads, streak/debounce logic,
  the thread-safe taught-sound store, and the public `EarshotML` engine.
- [`artifacts.py`](earshot_ml/artifacts.py): checksum-verified, atomic model
  downloads.
- [`earshot_ml/cli.py`](earshot_ml/cli.py): the installed CLI.
- [`cli.py`](cli.py): a source-checkout compatibility wrapper for the same
  CLI.

### Pretrained recognition flow

1. `MicStream` captures float32 mono audio at 16,000 Hz.
2. Audio is sliced into overlapping 15,600-sample windows (0.975 seconds) with
   an 8,000-sample hop (about 0.5 seconds).
3. Full YAMNet produces 521 AudioSet class scores and a 1,024-value embedding
   for each window.
4. Each entry in `config.EVENT_MAP` takes the maximum score across its mapped
   YAMNet classes and compares it with that event's threshold.
5. A pretrained event normally needs two consecutive qualifying windows.
   Repeat events with the same source and label are then suppressed for 10
   seconds.
6. Fired events go synchronously to the configured callback and/or event
   queue.

The supplied map includes smoke alarm, fire alarm, doorbell/ding-dong, knock,
baby cry, and glass break events. The thresholds in
[`config.py`](earshot_ml/config.py) are starting points, not universal
calibration.

### Teach-mode flow

1. `teach` accepts uncompressed 8-, 16-, or 32-bit PCM WAV files, or
   one-dimensional float32-compatible 16 kHz arrays through the Python API.
   WAV input is downmixed to mono and linearly resampled to 16 kHz.
2. Each clip is windowed and embedded. Its window embeddings are averaged and
   L2-normalized.
3. One vector per clip is persisted in `taught_sounds.npz`.
4. During live recognition, each YAMNet embedding is compared with the stored
   vectors using cosine similarity. The best match must meet the default 0.80
   cutoff.
5. Taught sounds need one qualifying window by default, then use the same
   source-aware debounce detector as pretrained events.

The taught store uses a re-entrant lock for consistent reads and writes, and
replaces its NPZ file atomically. YAMNet also serializes each interpreter
transaction, so `run` and `teach` can safely share one `EarshotML` instance
across threads. These locks are instance-local, not cross-process: use one
writer process per `EARSHOT_MODEL_DIR` to avoid last-writer-wins updates from
simultaneous `teach` or `forget` commands.

## Requirements

- CPython 3.10 through 3.14 (`>=3.10,<3.15`) on non-`armv7l`
  platforms; Python 3.10 or 3.11 on 32-bit `armv7l`.
- A supported LiteRT/TFLite runtime installed by the package.
- PortAudio and an input device for live or recorded-microphone commands.
- Network access for the one-time `download` command only.

The fast tests, storage-only `sounds` and `forget` commands, and most of the
Python logic do not require a microphone. The real-model integration test
requires downloaded artifacts but still does not use a microphone.

## Windows PowerShell setup

Run these commands from this `ml` directory. Use `python`, not `py` or
`python3`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install ".[test]"
earshot --help
```

If PowerShell policy prevents activation, leave the environment unactivated
and use the executables directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install ".[test]"
.\.venv\Scripts\earshot.exe --help
```

For a persistent writable model and taught-state directory, set
`EARSHOT_MODEL_DIR` before running any Earshot command:

```powershell
$env:EARSHOT_MODEL_DIR = Join-Path $env:LOCALAPPDATA "Earshot\models"
New-Item -ItemType Directory -Force $env:EARSHOT_MODEL_DIR | Out-Null
earshot download
```

Without the override, a source checkout uses this directory's `models`
folder.

## Raspberry Pi setup

Use a supported Raspberry Pi OS release and run from the `ml` directory:

```bash
sudo apt update
sudo apt install -y python3 python3-venv libportaudio2 portaudio19-dev alsa-utils

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ".[test]"

export EARSHOT_MODEL_DIR="$HOME/.local/share/earshot/models"
mkdir -p "$EARSHOT_MODEL_DIR"
earshot --help
```

On 32-bit `armv7l` Raspberry Pi OS, use Python 3.10 or 3.11. The package always
selects the pinned `tflite-runtime==2.14.0` backend on `armv7l`; because that
backend has no compatible pinned wheel for Python 3.12 or newer, installation
on those combinations is intentionally unsupported and dependency resolution
fails instead of producing an installation with no inference backend. On other
architectures, including 64-bit Pi OS, the package selects `ai-edge-litert`.

Add the `EARSHOT_MODEL_DIR` export to the service or shell environment that
will run Earshot. The download, live commands, and storage-only commands must
see the same setting.

## Download and verify the model

With the environment active, run the installed command:

```text
earshot download
```

From a source checkout, the compatibility wrapper is equivalent:

```text
python cli.py download
```

The command downloads or reuses exactly two pinned artifacts:

| Artifact | Required SHA-256 |
| --- | --- |
| `yamnet.tflite` | `141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3` |
| `yamnet_class_map.csv` | `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2` |

Each transfer uses a 30-second blocking-I/O timeout (not a total wall-clock
deadline) and permits at most 128 MiB per artifact. It streams to a
process-unique sibling name ending in `.part`, checks
the SHA-256 before atomic replacement, and attempts to remove its own partial
file after a failure. Concurrent processes therefore never truncate or unlink
one another's temporary download; an already-valid destination is reused
without opening the source. Within one process, destination checksum reads and
final replacements share a short lock so concurrent downloader threads cannot
trigger Windows file-sharing violations; the network transfers themselves may
still overlap. The command then loads the production interpreter
and verifies one float32 waveform input, a 521-value score output, a 1,024-value
embedding output, and 521 class names. A successful run ends with:

```text
validated model and class map
done
```

It may report each file as either `downloaded` or `cached`; `cached`
still means its checksum matched. Do not substitute YAMNet's
classification-only TFLite artifact because it lacks the embedding required
by teach mode.

## Automated tests

Run the fast suite for normal development and CI:

```text
python -m pytest -m "not integration" -q
```

The fast suite uses fakes and temporary files. It does not download the model,
open a microphone, or require hardware. GitHub Actions runs this command on
Ubuntu with Python 3.11 and 3.14.

After `earshot download`, verify the actual production model contract:

```text
python -m pytest tests/test_real_model.py -q
```

This integration test infers on a zero-valued 15,600-sample waveform and
checks finite `(521,)` scores, a finite `(1024,)` embedding, and 521 class
names. If either artifact is absent, it skips with
`run earshot download first`.

Neither automated suite validates a real microphone, room acoustics, device
gain, end-to-end event latency, or false-positive rate. Use the manual
acceptance procedure below on the target hardware.

## Command-line operation

The examples below use the installed `earshot` command. From this source
directory, replace `earshot` with `python cli.py` to use the wrapper.

| Command | Purpose |
| --- | --- |
| `earshot download` | Download, checksum, and validate the model and class map. |
| `earshot top5 [--device INDEX]` | Print live peak level and the top five YAMNet classes. |
| `earshot run [--device INDEX]` | Run configured pretrained and taught event detection. |
| `earshot teach NAME [WAV ...] [--record N] [--seconds S] [--device INDEX]` | Teach a sound from WAV files, microphone recordings, or both. |
| `earshot sounds` | List taught names and stored clip counts. |
| `earshot forget NAME` | Remove every stored clip for one taught name. |

`sounds` and `forget` access only the NPZ store. They intentionally do not
load YAMNet, a LiteRT backend, the class map, or a microphone.

### Select a device and set gain

List the indices reported by `sounddevice`:

```text
python -m sounddevice
```

Pass the intended input index to each microphone command:

```text
earshot top5 --device 3
earshot run --device 3
earshot teach kettle --record 3 --seconds 2 --device 3
```

On Windows, set the selected microphone's input volume under **Settings >
System > Sound > Input**. On Raspberry Pi, run `arecord -l` to inspect ALSA
cards, then run `alsamixer`, press **F6** to select the USB sound card and
**F4** for capture controls, and adjust capture gain.

`top5` prints a `peak` value for each window. As an initial check, normal
sounds around 0.1 to 0.7 are usually usable. Values repeatedly near 1.0 suggest
clipping; values close to zero suggest the wrong device or too little gain.
Treat this range as a diagnostic starting point, not a universal calibration.

### Inspect classes and run events

```text
earshot top5 --device 3
earshot run --device 3
```

`top5` is useful for device selection, gain checks, and discovering which
YAMNet class responds to a sound. `run` prints fired event dictionaries as
JSON. Both block until **Ctrl+C**.

### Teach from files or the microphone

```text
earshot teach kettle samples/kettle-1.wav samples/kettle-2.wav samples/kettle-3.wav
earshot teach dryer_buzzer --record 3 --seconds 2 --device 3
earshot sounds
earshot forget kettle
```

Provide at least one WAV path or a positive `--record` count. Names are
trimmed, must not be empty, and cannot collide case-insensitively with a
configured pretrained event label such as `smoke_alarm` or `doorbell`.
Record examples at the expected distance and against representative room
noise.

## Model and taught-state location

`EARSHOT_MODEL_DIR` controls all three runtime files:

- `yamnet.tflite`
- `yamnet_class_map.csv`
- `taught_sounds.npz`

If the variable is unset, the default is `ml/models` in a source checkout.
For an installed service, set an absolute, writable, persistent directory.
Set the variable before Python starts because the paths are resolved when
`earshot_ml.config` is imported. Moving to a different directory also moves
the apparent taught-sound state; point all commands and the backend at the
same directory.

## Backend integration

Callbacks and queues are in-process; Earshot does not start a network service.
The engine blocks in `run`, so a backend normally owns a worker thread and a
`threading.Event`:

```python
import queue
import threading

from earshot_ml import EarshotML


def handle_event(payload):
    # Keep this callback quick; exceptions intentionally propagate.
    print(payload)


event_queue = queue.Queue()
stop_event = threading.Event()
engine = EarshotML(
    on_event=handle_event,
    event_queue=event_queue,
    device=3,
)
worker = threading.Thread(
    target=engine.run,
    kwargs={"stop_event": stop_event},
    name="earshot-listener",
)
worker.start()

# Elsewhere, a backend consumer may call event_queue.get().
# To stop during shutdown:
stop_event.set()
worker.join(timeout=2.0)
if worker.is_alive():
    raise RuntimeError("Earshot listener did not stop promptly")
```

Either `on_event`, `event_queue`, or both may be supplied. For each fired
event, Earshot invokes the callback first and then calls
`event_queue.put(payload)`. The callback is synchronous: slow work delays
inference, and a callback exception propagates out of `run`. A bounded
application queue can also block when full, so drain it continuously or use
an unbounded queue and enforce limits in the consumer.

This application event queue is separate from microphone buffering.
`MicStream` keeps at most two capture blocks by default. When inference falls
behind, it atomically discards the oldest unprocessed block before adding the
newest. That bounds memory and favors current audio over stale results, but an
overloaded system can miss audio; monitor performance rather than assuming
every block was processed.

The stop event is polled while waiting for microphone blocks. Setting it ends
window generation and leaves the `sounddevice.InputStream` context, which
closes the input stream. A blocked callback or full application queue must
return before the worker can observe the stop event.

The same API supports file/array teaching:

```python
stored = engine.teach(
    "kettle",
    ["clip1.wav", "clip2.wav", "clip3.wav"],
)
known = engine.learned_sounds()
removed = engine.forget("kettle")
```

## Event payload and confidence semantics

Each fired event is a dictionary with the existing public shape:

```json
{
  "label": "smoke_alarm",
  "urgency": "high",
  "confidence": 0.62,
  "source": "pretrained",
  "timestamp": 1752969600.0
}
```

- `label`: configured pretrained label or operator-supplied taught name.
- `urgency`: configured `high`, `medium`, or `low` value.
- `confidence`: rounded to three decimals for compatibility.
- `source`: `pretrained` or `taught`.
- `timestamp`: Unix seconds at the detector firing time.

`confidence` is not a calibrated probability. For pretrained events it is
the maximum raw YAMNet score across the classes mapped to that event. For
taught events it is cosine similarity between the live normalized embedding
and the nearest stored clip vector. Those values have different meanings and
should not be compared directly across sources.

## Tuning and calibration

Tune [`earshot_ml/config.py`](earshot_ml/config.py) on the target Pi,
microphone, placement, and room:

- Raise an `EVENT_MAP` threshold when that event fires on unrelated audio;
  lower it only after confirming gain and device selection.
- Raise `TAUGHT_SIMILARITY_CUTOFF` to make taught matches stricter; lower it
  cautiously if representative positives are missed.
- Increase `CONSECUTIVE_WINDOWS` for more pretrained stability at the cost
  of latency and possible missed transients.
- Change `DEBOUNCE_SECONDS` to control repeated notifications.
- Keep `SAMPLE_RATE`, `WINDOW_SAMPLES`, and `HOP_SAMPLES` aligned with the
  model contract.

A practical calibration pass uses labeled positive examples at the real
distance plus difficult negative/ambient recordings. Record peak levels,
event source, confidence value, latency from sound onset, misses, and false
events. Change one setting at a time and repeat the same material. YAMNet
scores and taught cosine similarities require separate thresholds.

## Manual hardware acceptance

Run this procedure on the actual Pi, microphone, placement, and room:

1. Run `python -m sounddevice`, select the intended input, and pass its index
   explicitly.
2. Run `earshot top5 --device INDEX`; confirm normal sounds produce a useful,
   unclipped peak level.
3. From two metres away, play representative doorbell and smoke/fire-alarm
   clips through the intended playback source.
4. Run `earshot run --device INDEX` and measure sound-onset-to-event latency
   for repeated trials.
5. Record three examples of a new sound, then verify a separate later example
   is recognized.
6. Run at least five minutes of representative ambient room audio and record
   every false event.

The project target is for configured events to be observed in under two
seconds, but this is an acceptance measurement, not a runtime guarantee.
Report the measured results from the actual setup, including misses and false
positives.

## Safety and performance limits

Earshot is not a certified smoke alarm, fire alarm, accessibility device, or
life-safety system. Do not use it to replace approved alarms, emergency
procedures, supervision, or other required safeguards. Machine-learning
classifiers can miss events and produce false positives; microphones can be
muted, disconnected, clipped, obstructed, or delayed; and overloaded capture
intentionally drops old blocks.

The under-two-second target depends on device speed, event thresholds,
consecutive-window settings, microphone position, room acoustics, and callback
or queue load. Treat it as something to measure and document on each
deployment, never as a guaranteed bound.

## Troubleshooting

### `No module named pytest` or an import is missing

Activate the intended `.venv` and reinstall the package with its test extra:

```text
python -m pip install ".[test]"
```

On Windows, `Get-Command python` should point into
`.venv\Scripts\python.exe` after activation. Otherwise use that executable
directly.

### Model or class map not found

Run `earshot download` in the same environment and with the same
`EARSHOT_MODEL_DIR` used by `run`, `top5`, `teach`, or the backend. The
integration test deliberately skips when either artifact is absent.

### Download, checksum, or model-contract error

Check the network connection and retry `earshot download`. The downloader
attempts to remove its own incomplete `.part` file and will not install an
oversized or checksum-mismatched artifact. Errors mentioning 521 scores, 1,024
embeddings, float32 input, or 521 class-map rows indicate an
incompatible/corrupt artifact; use the pinned download command rather than a
classification-only model.

### No LiteRT/TFLite interpreter

Reinstall the project in a supported Python environment. Windows and
non-`armv7l` platforms select `ai-edge-litert`; 32-bit `armv7l` selects
`tflite-runtime==2.14.0` and supports Python 3.10 or 3.11. On `armv7l` with
Python 3.12 or newer, normal installation intentionally fails because no
compatible pinned backend wheel exists. Manually bypassing dependencies leaves
Earshot unable to infer and produces the runtime backend diagnostic.

### PortAudio, invalid device, or no microphone input

Install the OS PortAudio packages, run `python -m sounddevice`, and pass a
valid input index with `--device`. On Pi, confirm the user/service can access
the audio device and that the intended ALSA card is selected. Preserve the
original `sounddevice` diagnostic when reporting a failure.

### Peaks are near zero or repeatedly near 1.0

Confirm the device index first. Increase capture gain for near-zero input and
reduce it for clipping near 1.0. Re-run `top5` after every change before
tuning model thresholds.

### Teach rejects a name or asks for clips

Use a non-empty name that does not collide with a configured pretrained label.
Provide one or more WAV paths, a positive `--record N`, or both.

### Taught store is corrupt or incompatible

`sounds` and `forget` report a concise store error without loading the
model. Restore a known-good `taught_sounds.npz`, or back up and move the
invalid file aside before teaching the sounds again.

### Live output lags or shutdown does not finish

Keep callbacks short and continuously drain bounded application queues. Slow
consumers can block event delivery even though microphone capture itself
drops old blocks. During shutdown, unblock the callback/consumer path, set the
stop event, and join the listener thread.
