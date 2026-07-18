### Task 4: Safe taught-sound persistence and storage-only commands

**Files:**
- Modify: `ml/earshot_ml/core.py`
- Create: `ml/tests/test_teach_store_safety.py`

**Interfaces:**
- Produces: thread-safe `TeachStore`, validated NPZ loading, and atomic `save()`.
- Produces: `TeachStoreError` for corrupt or incompatible stores.
- Preserves: `add`, `match`, `learned`, `forget`, and `save` behavior.

- [ ] **Step 1: Write failing corruption and atomicity tests**

Test missing NPZ keys, mismatched name/vector counts, vectors with width other than 1,024, NaN vectors, and preservation of an existing store when `np.savez` raises. Add a thread test that repeatedly matches while adding vectors and asserts no shape/index exception occurs.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_teach_store_safety.py -q`

Expected: current loader accepts invalid state or raises incidental NumPy errors.

- [ ] **Step 3: Implement locking, validation, and atomic save**

Use `threading.RLock`, validate loaded arrays before assigning state, return copies/snapshots for matrix operations, and write NPZ through an opened sibling `.part` file before `os.replace`. Ensure `.part` cleanup after failure and preserve the previous destination.

- [ ] **Step 4: Verify store tests**

Run: `python -m pytest tests/test_teach_store_safety.py tests/test_ml.py -q`

Expected: all tests pass.

---

