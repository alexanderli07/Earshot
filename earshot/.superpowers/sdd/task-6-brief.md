### Task 6: Bounded microphone buffering and stop behavior

**Files:**
- Modify: `ml/earshot_ml/pipeline.py`
- Extend: `ml/tests/test_pipeline.py`

**Interfaces:**
- Produces: `LatestBlockQueue(maxsize=2)` with `put_latest(block)` and `get(timeout)`.
- Produces: `MicStream.windows(stop_event=None)`.

- [ ] **Step 1: Write failing queue/lifecycle tests**

Test that adding three numbered blocks to a two-block queue yields blocks two and three, not one and two. Install a fake `sounddevice` module whose input-stream context records entry/exit and whose callback can be driven by the test; assert a stop event ends the generator and closes the context.

- [ ] **Step 2: Run focused tests to verify RED**

Run: `python -m pytest tests/test_pipeline.py -k "latest or stop" -q`

Expected: failures because the bounded queue and stop parameter do not exist.

- [ ] **Step 3: Implement bounded latest-block behavior**

On `queue.Full`, discard exactly one oldest block and enqueue the newest. Poll `get` with a 0.1-second timeout so a stop event is observed promptly. Retain the current buffering/window overlap and sounddevice context manager.

- [ ] **Step 4: Verify all pipeline tests**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: all tests pass without a microphone.

---

