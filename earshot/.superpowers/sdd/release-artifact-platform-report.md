# Release Artifact and Platform Report

Date: 2026-07-18

## Completed scope

- Hardened `ml/earshot_ml/artifacts.py` with a 30-second `urlopen` timeout and
  a 128 MiB (`134217728` byte) per-artifact maximum.
- Rejects an oversized `Content-Length` before creating a temporary file and
  rejects a headerless or understated response as soon as streamed bytes cross
  the same limit. Neither path replaces an existing destination.
- Downloads into a `tempfile`-allocated, process-unique sibling whose name ends
  in `.part`. The temporary file is closed and SHA-256 validated before
  `os.replace`; cleanup is limited to that invocation's path.
- Preserved `ChecksumError` detail and wrapped transfer/setup/size failures in
  actionable `ArtifactError` messages with the original exception as
  `__cause__`. A cleanup `OSError` cannot mask that primary exception.
- Removed the Python-version clause from the `armv7l` dependency marker.
  `armv7l` now always selects only `tflite-runtime==2.14.0`; other machines
  select only `ai-edge-litert`.
- Added marker-matrix metadata coverage for `armv7l` on Python 3.11 and 3.12,
  plus AMD64 on Python 3.12.
- Added `ml/build/` and `ml/*.egg-info/` to `.gitignore`, changed the legacy
  Windows-facing standalone test command from `python3` to `python`, and
  updated `ml/README.md` with the artifact bounds, concurrency behavior, and
  the intentional 32-bit Pi Python support boundary.

## Test-first evidence

All commands used the required existing interpreter from the repository root.

### Artifact RED

Command:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml\tests\test_artifacts.py -q
```

Before the implementation change:

```text
5 failed, 9 passed in 0.44s
```

The five expected failures showed that no timeout was supplied, declared and
streamed oversize responses reached checksum validation, cleanup
`PermissionError` masked the transfer error, and two simultaneous downloads
contended on the shared `.part` path with Windows error 32.

### Artifact GREEN

The same command after the implementation change reported:

```text
14 passed in 0.27s
```

The concurrent test uses a two-party barrier. While both response reads are
blocked, it observes two distinct sibling `.part` paths; both verified
downloads complete, the destination has the expected payload, and no `.part`
file remains. The oversize and ordinary failure tests also preserve the
pre-existing destination and leave no `.part` files.

### Packaging and hygiene RED

Command:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml\tests\test_packaging.py -q
```

Before the metadata changes:

```text
2 failed, 3 passed in 0.12s
```

The `armv7l`/Python 3.12 environment selected no runtime backend, and both
required ignore entries were absent.

### Packaging and hygiene GREEN

The same command after the metadata changes reported:

```text
5 passed in 0.15s
```

## Scoped verification

Command:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml\tests\test_artifacts.py ml\tests\test_packaging.py -q
```

Fresh result:

```text
19 passed in 0.36s
```

## Concurrent-suite status

A full fast-suite coordination run was attempted while the pipeline/CLI agent
was still implementing its own RED tests:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml -m "not integration" -q
```

Result at that point:

```text
5 failed, 136 passed, 1 deselected in 1.39s
```

All five failures were confined to concurrently owned pipeline/CLI work:
`InterpreterBackendError` had not yet been implemented and `YamNet` had not
yet normalized string paths. No pipeline, core, CLI, engine, or transaction
file was changed in this scope. A stable full-suite rerun remains pending.

## Constraints and concerns

- The timeout is the finite timeout passed to `urlopen`; it bounds applicable
  blocking open/read operations, not total wall-clock duration for an
  arbitrarily slow transfer.
- Normal error paths remove their unique temporary file. If the operating
  system itself refuses cleanup, the primary error remains visible and that
  invocation's uniquely named `.part` may require later operator cleanup.
- Python 3.12+ remains within the project's overall metadata range for other
  platforms. On 32-bit `armv7l`, dependency resolution is intentionally
  expected to fail because the always-selected pinned `tflite-runtime` has no
  compatible wheel.
- No network request, model download, microphone access, Git operation,
  generated-directory deletion, or edit outside the delegated files was
  performed.
