# Release Pipeline and CLI Report

Date: 2026-07-18

## Completed scope

- Added actionable, cause-preserving `InterpreterBackendError` handling after
  lazy fallback through `tflite_runtime`, `ai_edge_litert`, and TensorFlow Lite.
  The CLI reports this expected failure without a traceback.
- Normalized YAMNet model and class-map arguments with `Path`. Production
  construction checks both artifacts and points to `earshot download` before
  attempting to import an interpreter backend.
- Added `AudioDeviceError` for sounddevice import, `InputStream` construction,
  stream entry/normal close, `rec`, and `wait` failures. Messages retain the
  original diagnostic and recommend `python -m sounddevice`; the CLI handles
  them concisely.
- Preserved `KeyboardInterrupt`, generator shutdown, and application errors.
  A close failure is wrapped only on normal stream exit, so it cannot mask an
  already-active application exception or `BaseException`.
- Serialized each interpreter `set_tensor` / `invoke` / two-`get_tensor`
  transaction with a per-YamNet `threading.RLock`.
- Tightened model output discovery to exactly one float32 final-width-521
  tensor and exactly one float32 final-width-1024 tensor. Quantized and
  ambiguous duplicate candidates report all observed tensor diagnostics.
- Validated actual inference outputs for float32 dtype, nonempty shape, exact
  final width, and finite values before frame averaging.
- Completed PCM WAV, stereo downmix, empty clip, resampling, padding, overlap,
  and validation coverage. Audio helpers now require one-dimensional arrays,
  positive finite sample rates, and positive integer window/hop geometry.
- Added private monotonic microphone block sequence numbers while preserving
  `LatestBlockQueue.get() -> block`. `MicStream` clears its overlap buffer when
  a dropped-block sequence gap is observed.
- Added CLI preflight rejection for negative `--record` and non-positive or
  non-finite `--seconds`, before engine construction or recording.

## Test-first evidence

All commands below used `ml\.venv\Scripts\python.exe`, with the working
directory set to `earshot\ml`. The pre-change scoped baseline was:

```text
31 passed in 0.90s
```

### Audio helpers

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_pipeline.py `
  -k 'wav or resample or clip_windows' -q
```

RED:

```text
17 failed, 17 passed, 15 deselected in 0.66s
```

The intended failures were missing one-dimensional, sample-rate, and
window/hop validation. Fixture-generated PCM decoding, downmix, 24-bit
rejection, empty audio, endpoint, padding, and overlap characterizations
already passed and therefore protected existing valid behavior.

GREEN:

```text
34 passed, 15 deselected in 0.42s
```

### Interpreter backend and artifact preflight

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest `
  tests/test_pipeline.py tests/test_cli.py `
  -k 'backend or missing_artifact or string_artifact' -q
```

RED:

```text
5 failed, 2 passed, 56 deselected in 0.37s
```

The failures showed the absent backend exception, string paths reaching
`.exists()`, missing class-map diagnostics, and an uncaught CLI backend error.

GREEN:

```text
7 passed, 56 deselected in 0.22s
```

### Output contract

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_pipeline.py `
  -k 'required_model_outputs or invalid_actual' -q
```

RED:

```text
8 failed, 53 deselected, 2 warnings in 0.53s
```

Quantized metadata and duplicate required-width tensors were accepted;
float64, empty, and NaN actual outputs escaped validation; malformed width
leaked a NumPy reshape error. The empty output also produced mean warnings.

GREEN:

```text
8 passed, 53 deselected in 0.30s
```

### Interpreter concurrency

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_pipeline.py `
  -k 'serializes_interpreter' -q
```

RED:

```text
1 failed, 61 deselected in 0.29s
```

The coordinated first thread paused inside `invoke`; the second thread then
entered `set_tensor`, proving two active interpreter transactions.

GREEN:

```text
1 passed, 61 deselected in 0.37s
```

### Audio device errors

Initial command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest `
  tests/test_pipeline.py tests/test_cli.py `
  -k 'input_stream_failures or record_wraps or application_errors or keyboard_interrupt or recording_device_error' -q
```

Initial RED:

```text
5 failed, 2 passed, 72 deselected in 0.34s
```

Raw stream-construction/entry and rec/wait failures escaped, including the
CLI recording path. The two passing characterization tests confirmed that
application errors and `KeyboardInterrupt` were already propagated.

Import and normal-close error paths were then isolated in a second genuine
RED before their implementation:

```text
2 failed, 68 deselected in 0.27s
```

The consolidated device focus after implementation reported:

```text
9 passed, 72 deselected in 0.16s
```

### Dropped microphone block continuity

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_pipeline.py `
  -k 'clears_overlap_after' -q
```

RED:

```text
1 failed, 70 deselected in 0.48s
```

The deterministic producer/consumer test observed synthetic
`[1, 2, 5, 6]` after `[3, 4]` was dropped instead of a clean post-gap window.

GREEN, including public latest-queue and stop-event regressions:

```text
4 passed, 67 deselected in 0.17s
```

### CLI record preflight

Command:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_cli.py `
  -k 'invalid_record_options' -q
```

RED:

```text
5 failed, 11 deselected in 0.25s
```

All invalid forms reached forbidden engine construction.

GREEN:

```text
5 passed, 11 deselected in 0.13s
```

## Verification

Complete delegated scope:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest `
  tests/test_pipeline.py tests/test_cli.py -q
```

```text
87 passed in 1.37s
```

Stable integrated fast suite after the concurrent agents completed their
edits:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -m 'not integration' -q
```

```text
165 passed, 1 deselected in 1.75s
```

## Self-review and residual concerns

- The interpreter lock encloses only the mutable interpreter transaction;
  waveform validation occurs before it and output-array validation occurs
  after both tensors have been copied out.
- Required-output candidate selection rejects a quantized tensor even if a
  float32 tensor of the same required width is also present, because that width
  is ambiguous rather than silently choosing one.
- Block sequence metadata is private. Existing max-size, drop-oldest,
  `get()->block`, timeout, and stop behavior remains covered.
- If stream close itself fails while another exception or generator shutdown
  is active, the active exception intentionally wins; masking application
  failures with cleanup diagnostics would violate the error contract.
- The resampler remains the approved simple linear interpolator. No claim is
  made about higher-quality DSP behavior.
- No network request, real model, microphone, Git operation, deletion, or edit
  outside the four delegated code/test files and this report was performed.
