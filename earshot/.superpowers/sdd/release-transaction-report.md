# Release Transaction Report

Date: 2026-07-18

## Completed scope

- Added `TeachStore.transaction()`, a context-managed transaction that takes
  copies of names and vectors while holding the store's existing `RLock`,
  keeps that lock across re-entrant update and save calls, restores both
snapshots on any `BaseException`, and re-raises the exception produced by the
transaction body. Storage `OSError`s are now wrapped once as `TeachStoreError`
with the original exception retained as their cause.
- Changed `EarshotML.teach()` to infer, average, and normalize every clip
  embedding before entering the transaction or mutating the store. The
  transaction then adds every prepared embedding and persists them as one
  commit.
- Changed `EarshotML.forget()` to forget and persist inside one transaction.
  A failed save therefore restores the long-lived engine's live store.
- Preserved successful teach and forget return values, persisted results, and
  all existing store/engine APIs.

## Test-first evidence

The pre-change focused baseline was:

```powershell
& 'earshot\ml\.venv\Scripts\python.exe' -m pytest `
  'earshot\ml\tests\test_engine.py' `
  'earshot\ml\tests\test_teach_store_safety.py' -q
```

```text
38 passed in 0.33s
```

After adding only the new regression tests, the same command produced the
required genuine RED:

```text
5 failed, 38 passed in 2.45s
```

The five failures demonstrated that:

1. a first clip's vector remained live when the second inference raised;
2. a first clip's vector remained live when the second embedding was invalid;
3. a failed teach NPZ write left the new name/vector in memory;
4. a failed forget replace left the removed name/vector absent from memory;
5. `TeachStore` had no transaction primitive to hide and roll back transient
   state.

After the minimal implementation, the same focused command reported:

```text
43 passed in 0.36s
```

## Focused compatibility verification

Command:

```powershell
& 'earshot\ml\.venv\Scripts\python.exe' -m pytest `
  'earshot\ml\tests\test_engine.py' `
  'earshot\ml\tests\test_teach_store_safety.py' `
  'earshot\ml\tests\test_ml.py' -q
```

Fresh result before final handoff:

```text
59 passed in 0.45s
```

The new tests use real, previously saved NPZ destinations. Injected
`np.savez` and `os.replace` failures verify exact destination bytes,
`.part` cleanup, restored `learned()` and `match()` results, and byte-for-byte
restoration of live vector arrays. A deterministic second-inference failure
also verifies that save is never called and no vector changes before all clip
embeddings are ready.

A synchronized thread test pauses a failing save after a transient add, then
starts `learned()` and `match()` observers. Both remain blocked until rollback
finishes and then see only the prior state. It also verifies that the exact
`TeachStoreError` produced by `save()` is propagated and that its cause is the
original storage exception, exercising `RLock` re-entrancy through `add()` and
`save()`. Engine rollback tests exercise re-entrant `forget()` and `save()` as
well.

## Self-review

- Transaction snapshots are independent copies: a new names list and a copied
  NumPy vector array.
- Rollback completes before the outer lock is released, so readers cannot see
  transient or partially restored state.
- Storage failures retain their original `OSError` as the chained cause of a
  concise `TeachStoreError`; the transaction does not replace that domain error.
- Successful transaction exit keeps the new state; characterization tests
  verify teach and forget counts plus reloadable persisted results.
- Existing atomic save behavior remains responsible for preserving destination
  bytes and cleaning the sibling `.part` file.
- No pipeline, CLI, artifact, metadata, documentation, model, microphone,
  network, Git, or deletion work was performed.

The parent agent owns the final integrated fast-suite run after concurrent
release agents finish updating the shared tree.
