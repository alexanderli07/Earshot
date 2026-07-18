# Task 5 report: event identity, engine validation, and stoppable runtime

## RED evidence

- Command: `ml\.venv\Scripts\python.exe -m pytest tests\test_engine.py -q`
  (run from `ml` before changing `core.py`)
- Result: **22 failed in 0.35s**.
- Focused cause: every engine test reached the intended missing dependency-
  injection seam and failed with `EarshotML.__init__() got an unexpected
  keyword argument 'yamnet'`. This confirmed the new tests were exercising
  behavior absent from the untouched implementation.

## Implementation

- Keyed detector streak and debounce state by `(source, label)` while leaving
  emitted event labels and payload fields unchanged.
- Added optional YamNet injection; omitting it still constructs the production
  `YamNet` from the configured model and class-map paths.
- Calculated each configured pretrained spec's mapped maximum once per window.
- Added teach validation for trimmed names, case-insensitive configured-label
  collisions, non-empty clip iterables, and one-dimensional, non-empty, finite
  float32-compatible audio from both arrays and loaded paths.
- Materialized and validated every clip before any model inference, store add,
  or persistence, so a later invalid clip cannot partially teach earlier clips.
- Forwarded `run(stop_event)` exactly to
  `MicStream(...).windows(stop_event=stop_event)` and allowed natural return.

## GREEN evidence

- Focused engine suite:
  `ml\.venv\Scripts\python.exe -m pytest tests\test_engine.py -q`
  -> **22 passed in 0.27s**.
- Focused plus legacy ML suite:
  `ml\.venv\Scripts\python.exe -m pytest tests\test_engine.py tests\test_ml.py -q`
  -> **36 passed in 0.19s**.
- Full repository suite:
  `ml\.venv\Scripts\python.exe -m pytest -q`
  -> **81 passed in 0.85s**.

## Self-review

- Tests cover deterministic pretrained two-window and taught one-window firing,
  callback plus queue delivery, same-label cross-source independence, maximum-
  score call count, validation and validation atomicity, valid ndarray/path
  teaching with the stored-count return, and exact stop-event forwarding.
- Existing callback ordering, queue behavior, return values, `Event` fields,
  payload rounding, and configured production model loading remain intact.
- Scope review found changes only in `ml/earshot_ml/core.py`, the new
  `ml/tests/test_engine.py`, and this report.

## Concerns

- No known code-level concerns. Per task constraints, verification used fake
  YAMNet and microphone implementations; no real model, microphone, network, or
  hardware integration run was performed.
