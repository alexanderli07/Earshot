# Earshot Reliability and Teach-Mode Design

## Context

Earshot is a standalone Python audio-event detection component intended to run offline on a Raspberry Pi. It captures 16 kHz mono audio, classifies overlapping windows with YAMNet, maps selected AudioSet classes into Earshot events, and lets an operator teach additional sounds from a few examples.

The current download command retrieves the fixed-length `yamnet/classification` TFLite model. That artifact only exposes 521 classification scores, while the runtime requires both those scores and a 1,024-dimensional embedding. The mismatch prevents every model-backed command from starting and makes teach mode unusable.

The existing source is also missing full-stack tests, reproducible package metadata, safe long-running lifecycle controls, bounded live-audio buffering, and synchronized taught-sound persistence.

## Goals

- Preserve the existing `download`, `top5`, `run`, `teach`, `sounds`, and `forget` workflows.
- Preserve `EarshotML`, callbacks, event queues, and teach mode as public capabilities.
- Use an embedding-capable full YAMNet TFLite artifact that produces 521 scores and 1,024-dimensional embeddings.
- Fail early with actionable diagnostics when dependencies, artifacts, audio devices, or model tensor contracts are invalid.
- Keep live listening current instead of accumulating stale audio when inference is slower than capture.
- Make taught-sound access safe when listening and teaching happen in different threads.
- Make downloads and taught-store writes atomic and resistant to partial files.
- Add repeatable unit, integration, CLI, and real-model smoke tests.
- Document exact Windows and Raspberry Pi setup and validation commands.

## Non-goals

- Building a web backend, mobile client, database, or network protocol.
- Training or fine-tuning a new neural network.
- Claiming safety certification or replacing certified smoke and fire alarms.
- Automatically calibrating venue-specific thresholds without labeled evaluation audio.
- Automating microphone acoustic-quality validation without access to the target hardware and room.

## Considered Approaches

### 1. Full embedding-capable YAMNet TFLite model — selected

Use the official full YAMNet TFLite variant rather than the fixed classification-only variant. One interpreter provides both classification scores and embeddings on every window. This retains offline operation, teach mode, and the existing `EarshotML` data flow without running two neural networks.

### 2. Full TensorFlow SavedModel — rejected

The TensorFlow Hub SavedModel exposes the necessary outputs and is straightforward on a development computer, but the TensorFlow runtime is much larger than LiteRT/TFLite and is a poor default for Raspberry Pi deployment.

### 3. Separate classifier and embedding models — rejected

Keeping the small classification model and adding a second embedder would preserve teach mode, but live recognition would require two inference passes for every window. That increases latency, memory use, deployment complexity, and the likelihood of input-preprocessing drift.

## Architecture

### Model acquisition and validation

`cli.py download` will retrieve the versioned full YAMNet TFLite artifact from `https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite` and the class map from TensorFlow Models revision `dfffd623b6be8d1d9744b8e261fbac370d17c46d`. The verified SHA-256 digests are `141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3` for the model and `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2` for the class map. Downloads will stream into a sibling `.part` file, verify the digest, and atomically replace the destination only after validation.

After acquisition, the command will instantiate `YamNet` and validate:

- one float32 waveform input that can accept 16 kHz mono audio;
- an output whose final dimension is 521;
- an output whose final dimension is 1,024;
- exactly 521 class-map entries.

An incompatible or corrupt artifact will be removed only if it is the newly downloaded temporary file; an existing valid model will never be overwritten by a failed download.

### Inference contract

`YamNet` will expose a stable `infer(waveform) -> (scores, embedding)` interface. It will normalize model-specific singleton and frame dimensions into `scores.shape == (521,)` and `embedding.shape == (1024,)`. Input validation will reject incorrect sample counts, non-finite values, and impossible tensor layouts with explicit errors.

The full model may expose additional tensors such as a spectrogram. They will be ignored unless their dimensions match one of the two required outputs.

### Live capture and lifecycle

`MicStream` will retain the current 16 kHz, 15,600-sample window and 8,000-sample hop geometry. Its callback-to-consumer queue will hold at most two microphone blocks by default. If inference falls behind, the oldest unprocessed block will be dropped so output reflects recent sound rather than delayed sound.

`MicStream.windows()` and `EarshotML.run()` will accept an optional `threading.Event`. Setting it will end the blocking generator promptly and close the sounddevice stream through its context manager. Existing calls without a stop event will continue to run until interrupted.

Callback behavior will remain synchronous and explicit: an exception from the application callback will propagate rather than being silently swallowed. Applications that need isolation can use the existing event-queue interface.

### Event detection

The existing threshold, consecutive-window, debounce, urgency, and event-dictionary behavior will remain compatible. Pretrained observations will continue to use the maximum score across mapped YAMNet classes.

Detector identity will use both source and label internally, preventing a taught name from corrupting the streak state of a pretrained label. The CLI will additionally reject taught names that collide with configured pretrained labels, producing an actionable message.

The `confidence` field remains for API compatibility, but documentation will clarify that pretrained values are model scores and taught values are cosine similarities, not calibrated probabilities.

### Teach store

`TeachStore` will protect in-memory names and vectors with a re-entrant lock. Matching will operate on a consistent snapshot, while add, forget, load, and save operations will be synchronized.

Persistence will write a complete NPZ file to a temporary file in the same directory and atomically replace `taught_sounds.npz`. Loading will validate required keys, matching name/vector counts, a vector width of 1,024, finite values, and supported dtypes. Invalid stores will raise a clear error instead of failing later during matrix multiplication.

The `sounds` and `forget` commands will use `TeachStore` directly and will not load YAMNet, the class map, the microphone library, or a TFLite interpreter.

### Audio helpers

WAV loading will keep supporting uncompressed 8-, 16-, and 32-bit PCM, mono conversion, and 16 kHz resampling. Tests will define behavior for short clips, overlapping windows, stereo input, unsupported sample widths, empty audio, and resampling endpoints. The current linear resampler remains acceptable for this lightweight component; replacing it with a larger DSP dependency is outside this scope.

### Packaging and configuration

The `ml` directory will gain a `pyproject.toml` declaring CPython `>=3.10,<3.15`, runtime dependencies, a test extra, and an `earshot` console entry point backed by `earshot_ml.cli:main`. The existing top-level `cli.py` will remain as a compatibility wrapper.

Model paths will continue to default to `ml/models` for source-checkout compatibility. `EARSHOT_MODEL_DIR` will allow deployments to select a writable persistent directory without editing source.

A `.gitignore` will exclude virtual environments, caches, downloaded models, partial downloads, taught stores, and common editor/OS artifacts. Generated `__pycache__`, `.pyc`, and `__MACOSX` artifacts in the supplied archive will be removed after their exact paths are verified.

## Error Handling

- Network failures retain the original exception context and leave no destination file that appears complete.
- Checksum mismatches state the affected filename and expected/actual digest.
- Missing model or class-map errors point to the exact `download` command.
- Missing LiteRT backends list the supported packages for the current platform.
- Model input/output mismatches report all observed tensor names, shapes, and dtypes.
- Microphone/device errors retain sounddevice diagnostics and recommend listing devices.
- Invalid teach names, empty clip lists, empty/non-finite arrays, corrupt stores, and unsupported WAVs fail before persistence is changed.

## Testing Strategy

### Fast automated tests

Pytest will run the existing detector and store behavior plus new coverage for:

- WAV decoding, resampling, and window geometry;
- model tensor discovery with small fake interpreters;
- inference input/output shape validation;
- event-map resolution and `process_window()` with a deterministic fake model;
- callback and event-queue emission;
- source/label collision isolation;
- bounded microphone-block behavior without requiring a physical microphone;
- stop-event lifecycle behavior;
- taught-store validation, locking-visible invariants, forgetting, and atomic replacement;
- download checksum success/failure and cleanup through local test files;
- CLI parsing and storage-only `sounds`/`forget` behavior.

### Real-model smoke test

After `download`, a separate integration test will load the actual model, infer on a zero waveform of 15,600 float32 samples, and assert score and embedding shapes, finite outputs, and a 521-entry class map. This test proves the downloaded artifact matches the production interpreter contract.

### Manual hardware acceptance

On the target microphone, the operator will:

1. list devices and select the intended input;
2. run `top5` and verify normal sounds produce an unclipped peak level;
3. play representative doorbell and alarm clips from two metres away;
4. run `run` and measure event latency;
5. record three examples of a new sound and verify a later example matches;
6. run ambient room audio for five minutes and record false events.

The README acceptance target remains under two seconds for configured events, but results must be reported from the actual Pi, microphone, and room.

## Documentation

The README will explain the architecture, Windows PowerShell setup, Raspberry Pi setup, model download, automated tests, integration smoke test, microphone selection, live commands, teach workflow, backend threading/stopping, state-directory override, threshold calibration, and safety limitations.

## Acceptance Criteria

- A clean supported Python environment installs the project and test extra successfully.
- All fast automated tests pass without a microphone or downloaded model.
- `download` retrieves only checksum-valid, versioned artifacts and validates their tensor contract.
- The real-model smoke test returns `(521,)` scores and `(1024,)` embeddings.
- `sounds` and `forget` work when no model or interpreter is installed.
- A stop event ends a running engine without leaving the input stream open.
- Slow consumers cannot create an unbounded microphone backlog.
- Concurrent matching and teach-store mutation cannot observe mismatched names/vectors or a partial NPZ file.
- Existing CLI command names and event payload fields remain compatible.
- Setup and validation commands are documented for both Windows and Raspberry Pi.
