### Task 5: Event identity, engine validation, and stoppable runtime

**Files:**
- Modify: `ml/earshot_ml/core.py`
- Create: `ml/tests/test_engine.py`

**Interfaces:**
- Produces: `EarshotML.run(stop_event=None)` and teach-name/input validation.
- Preserves: callbacks, event queues, return values, and payload fields.

- [ ] **Step 1: Write failing engine tests**

Use a deterministic fake YAMNet instance and construct `EarshotML` without loading a real model. Test pretrained two-window firing, taught one-window firing, callback and queue emission, independent detector state for identical labels from different sources, rejection of reserved teach names, empty clip lists, empty arrays, and non-finite arrays.

Add a fake `MicStream.windows(stop_event)` implementation and assert `run(stop_event)` forwards the event and exits when it is set.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_engine.py -q`

Expected: failures for source/label identity, validation, dependency injection, or stop-event forwarding.

- [ ] **Step 3: Implement minimal engine changes**

Key detector streak/debounce state by `(source, label)`. Permit constructor injection of a `yamnet` object for tests and integrations. Validate teach inputs before inference or persistence. Compute each pretrained max score once. Forward the optional stop event to `MicStream.windows()`.

- [ ] **Step 4: Verify engine and regression tests**

Run: `python -m pytest tests/test_engine.py tests/test_ml.py -q`

Expected: all tests pass and existing payload assertions remain unchanged.

---

