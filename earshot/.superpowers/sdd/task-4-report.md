# Task 4 Report: Safe Teach-Store Persistence

## Outcome

Implemented a validated, thread-safe `TeachStore` with atomic NPZ replacement.
Existing `add`, `match`, `learned`, `forget`, and `save` behavior remains intact.

## Files Changed

- Modified `ml/earshot_ml/core.py`
- Created `ml/tests/test_teach_store_safety.py`
- Created this report

## TDD Evidence

The safety tests were added before production code changed.

RED command, from `ml`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_teach_store_safety.py -q
```

Result: exit 1 during collection because the required public error type did not
exist:

```text
ImportError: cannot import name 'TeachStoreError' from 'earshot_ml.core'
1 error in 0.26s
```

Focused GREEN after implementation:

```text
15 passed in 0.35s
```

Focused plus original regression GREEN before review:

```text
29 passed in 0.27s
```

Fresh full-suite GREEN before review:

```text
58 passed in 0.91s
```

## Implementation Notes

- Added `TeachStoreError` for corrupt or incompatible persisted state.
- Loads NPZ files with `allow_pickle=False` and closes the archive before state
  assignment or future replacement.
- Requires `names` and `vectors`, one-dimensional string names,
  two-dimensional real numeric vectors of width 1,024, equal counts, and finite
  values. Validation completes before live state changes.
- Validates added and matched embeddings as finite, float-convertible vectors
  containing exactly 1,024 values.
- Uses a re-entrant lock around state mutation and snapshots names/vectors for
  matching and reporting.
- Saves through an opened sibling `.part` file, then calls `os.replace` only
  after `np.savez` finishes and the handle closes. Failed writes/replacements
  preserve the existing destination and attempt temporary-file cleanup.
- Tests cover missing keys, malformed dimensions, count drift, wrong width,
  non-finite values, object-array rejection, invalid additions, round trips,
  write/replace failure preservation, cleanup, and a deterministic add/match
  race.

## Self-Review and Residual Risks

- No network, microphone, model, Git, or deletion action was used.
- Cleanup after a failed save is best effort so a secondary Windows unlink
  error cannot hide the primary write/replace exception; a locked `.part` file
  could therefore remain for a later save to overwrite.
- Atomic replacement protects readers from partial files but is not a
  cross-process locking or disk-`fsync` guarantee.

## Review Fix

The independent review noted that `add()` appended a name before `np.vstack`
completed. A new test simulated an allocation failure and observed the genuine
RED (`learned()` incorrectly contained the new name). `add()` now computes the
replacement matrix before assigning either names or vectors. Fresh verification
after the fix produced `30 passed` for Task 4 plus legacy tests and `59 passed`
for the full suite.
