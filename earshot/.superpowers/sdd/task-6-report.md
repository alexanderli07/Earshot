# Task 6 report: bounded microphone buffering and stop behavior

## RED evidence

- Command: `ml\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k
  "latest or stop" -q` (run from `ml` before changing `pipeline.py`).
- Result: **3 failed, 19 deselected in 0.22s**.
- Expected causes: `LatestBlockQueue` did not exist,
  `MicStream.windows()` rejected the `stop_event` keyword, and its signature
  had no `stop_event` parameter or default.

## Implementation

- Added `LatestBlockQueue(maxsize=2)`, backed by `queue.Queue`, with atomic
  oldest-block replacement while holding the queue mutex. Positive-size
  validation prevents an accidentally unbounded queue.
- Added the optional `queue_size=2` `MicStream` constructor argument without
  changing existing call behavior, and routed callback blocks through
  `put_latest()`.
- Added `MicStream.windows(stop_event=None)` with 0.1-second timed polling and
  `queue.Empty` handling so a set event exits the input-stream context promptly.
- Kept the existing callback status diagnostic, lazy `sounddevice` import,
  float32 buffering, window size, hop size, and overlapping hop slice.

## GREEN evidence

- Focused queue/stop slice:
  `ml\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -k
  "latest or stop" -q`
  -> **3 passed, 19 deselected in 0.19s**.
- Complete pipeline suite:
  `ml\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q`
  -> **22 passed in 0.60s**.
- Full ML suite:
  `ml\.venv\Scripts\python.exe -m pytest -q`
  -> **84 passed in 1.63s**.

## Self-review

- The full-check, one-item discard, and newest-item enqueue occur under the same
  mutex used by `queue.Queue.get()`, so consumers cannot race between those
  operations. A full size-two queue containing `1, 2` therefore becomes `2, 3`.
- The fake `sounddevice.InputStream` test drives real callback blocks through
  the generator and verifies context entry, yielded float32 data, stop, and
  context exit. A separate signature regression proves the default is `None`,
  resolving the existing `EarshotML.run(stop_event=...)` integration mismatch.
- Stop checks occur before polling, after a block arrives, and before yielding
  buffered windows. The no-event path remains an indefinitely blocking stream.
- Scope review found changes only in `ml/earshot_ml/pipeline.py`,
  `ml/tests/test_pipeline.py`, and this report.

## Concerns

- No known code-level concerns. Per task constraints, all lifecycle coverage
  used a fake sounddevice module; no microphone, network, model, Git, or deletion
  operation was performed.
