# Task 2 Report: Verified Atomic Artifact Downloads

## Outcome

Implemented the Task 2 artifact contract with test-first development. Earshot now has immutable artifact metadata, streaming SHA-256 calculation, verified sibling-`.part` downloads, atomic installation, failure cleanup, existing-destination preservation, and an environment-configurable model directory. The configured model and class-map artifacts use the exact official URLs and SHA-256 values from the approved design.

All tests in this task used local `Path.as_uri()` sources; no network access was used.

## Files Changed

- Created `ml/earshot_ml/artifacts.py`
  - Added frozen `Artifact(url: str, path: Path, sha256: str)`.
  - Added `ArtifactError` and its `ChecksumError` subclass.
  - Added one-megabyte streaming `sha256_file(path) -> str`.
  - Added `download_artifact(artifact) -> bool` with checksum verification, sibling `.part` staging, `os.replace()` only after successful verification, and `finally` cleanup.
  - Preserves an existing destination until a replacement is fully downloaded and verified.
  - Returns `False` before opening the source when an existing destination digest matches.
- Modified `ml/earshot_ml/config.py`
  - Preserved the source-checkout `ml/models` default.
  - Added `EARSHOT_MODEL_DIR` override support for model, class-map, and taught-store paths.
  - Added `MODEL_ARTIFACT` and `CLASS_MAP_ARTIFACT` with the approved official URLs and digests.
- Created `ml/tests/test_artifacts.py`
  - Added nine tests covering multi-chunk hashing, frozen metadata, successful verified installation, checksum diagnostics, failure cleanup, destination preservation, no-source-access cache hits, default paths, environment override behavior, and exact official artifact metadata.
- Created `.superpowers/sdd/task-2-report.md`
  - Recorded the TDD and verification evidence for this task.

No files outside Task 2 scope were modified by this worker.

## RED Evidence

Production code had not been changed when this command was run from the repository root:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml/tests/test_artifacts.py -q
```

Pytest produced the expected collection failure:

```text
ImportError while importing test module ...\ml\tests\test_artifacts.py
ml\tests\test_artifacts.py:9: in <module>
    from earshot_ml.artifacts import (
E   ModuleNotFoundError: No module named 'earshot_ml.artifacts'
1 error in 0.31s
```

This was a genuine RED caused by the missing requested module, not a test typo or environment failure.

## GREEN Evidence

Interpreter used:

```text
ml\.venv\Scripts\python.exe --version
Python 3.14.0
```

Focused Task 2 suite:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml/tests/test_artifacts.py -q
```

```text
.........                                                                [100%]
9 passed in 0.28s
```

Required Task 2 plus original regression suite:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml/tests/test_artifacts.py ml/tests/test_ml.py -q
```

```text
.......................                                                  [100%]
23 passed in 0.26s
```

All tests currently present under `ml/tests`:

```powershell
ml\.venv\Scripts\python.exe -m pytest ml/tests -q
```

```text
........................                                                 [100%]
24 passed in 0.23s
```

## Requirement Review

- `Artifact` is a frozen dataclass with the required fields and types.
- `ChecksumError` derives from `ArtifactError`.
- File and response bodies are consumed in 1 MiB chunks.
- The download target is exactly the sibling `<destination>.part` path.
- `os.replace()` occurs only after the `.part` file's SHA-256 matches the expected digest.
- A `finally` block removes `.part` after success, checksum mismatch, transfer failure, or installation failure.
- Checksum and transfer failures do not change an existing destination.
- Checksum diagnostics include the destination filename plus expected and actual digests.
- Wrapped transfer/install failures retain the original exception through explicit exception chaining.
- A matching existing destination returns `False` before `urlopen`, proven with a deliberately missing local source URI.
- `EARSHOT_MODEL_DIR` changes all runtime artifact/store paths while the unset default remains `ml/models`.
- Official model URL and SHA-256 exactly match the approved plan.
- Official revision-pinned class-map URL and SHA-256 exactly match the approved plan.

## Self-Review

- Followed strict RED-GREEN TDD: the new tests were written and observed failing for the missing module before either production file was changed.
- Every new public function and requested behavior has focused coverage.
- Tests exercise real filesystem and `urllib` behavior through local file URIs; no network mocks or external services are involved.
- The implementation is limited to the requested module and configuration change and does not integrate the downloader into the CLI early; that remains Task 7.
- No Git initialization, commits, deletion, or network access occurred.

## Concerns and Follow-Up

- No Task 2 blocker remains.
- The fixed sibling `.part` name is intentionally the design-specified behavior and does not serialize simultaneous downloads to the same destination. Cross-process download locking was not part of this task.
- Production HTTPS retrieval and real-artifact digest validation remain the explicitly separate Task 8 integration step; this task verifies the same workflow deterministically with local URIs.
