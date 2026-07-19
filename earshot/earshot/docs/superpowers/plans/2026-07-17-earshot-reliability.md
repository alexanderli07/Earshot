# Earshot Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Earshot installable, preserve teach mode with a compatible full YAMNet model, harden long-running operation and persistence, and prove the complete model contract with automated tests.

**Architecture:** Keep the existing config/pipeline/core separation, add a focused artifact-download module, move CLI implementation into the package while retaining the top-level wrapper, and isolate hardware-independent behaviors behind testable units. The full YAMNet TFLite interpreter remains the single source of both scores and embeddings.

**Tech Stack:** CPython 3.10-3.14, NumPy, sounddevice, LiteRT/TFLite, pytest, setuptools.

## Global Constraints

- Preserve `download`, `top5`, `run`, `teach`, `sounds`, and `forget` command names.
- Preserve `EarshotML`, callback/queue integration, and existing event payload fields.
- Preserve teach mode and require model outputs shaped to 521 scores and 1,024 embeddings.
- Use model SHA-256 `141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3`.
- Use TensorFlow Models revision `dfffd623b6be8d1d9744b8e261fbac370d17c46d` and class-map SHA-256 `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`.
- Keep fast tests independent of a microphone, network, and downloaded model.
- Keep the top-level `python cli.py` workflow compatible.
- The supplied archive has no Git metadata, so commit steps are replaced with verified checkpoints.

---

### Task 1: Package and test harness

**Files:**
- Create: `ml/pyproject.toml`
- Create: `.gitignore`
- Create: `ml/tests/test_packaging.py`
- Modify: `ml/requirements.txt`

**Interfaces:**
- Produces: an editable package install, pytest test discovery, and `earshot` console-script metadata.
- Consumes: existing `earshot_ml` package and top-level `cli.py`.

- [ ] **Step 1: Write the failing packaging test**

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_declares_package_and_cli():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["requires-python"] == ">=3.10,<3.15"
    assert data["project"]["scripts"]["earshot"] == "earshot_ml.cli:main"
    assert "test" in data["project"]["optional-dependencies"]
```

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m pytest tests/test_packaging.py -q`

Expected: failure because `pyproject.toml` does not exist.

- [ ] **Step 3: Add package metadata and ignore rules**

Create a setuptools `pyproject.toml` with project name `earshot-ml`, version `0.1.0`, `requires-python = ">=3.10,<3.15"`, NumPy and sounddevice dependencies, LiteRT for every architecture except 32-bit ARM, `tflite-runtime==2.14.0` for `armv7l` on Python below 3.12, pytest in the `test` extra, package discovery for `earshot_ml*`, and script `earshot = "earshot_ml.cli:main"`.

Update `requirements.txt` to contain `-e .` so the legacy setup command uses package metadata. Add repository-root `.gitignore` entries for `ml/.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `ml/models/*.tflite`, `ml/models/*.csv`, `ml/models/*.npz`, `ml/models/*.part`, `.DS_Store`, and `__MACOSX/`.

- [ ] **Step 4: Verify GREEN and installation metadata**

Run: `python -m pytest tests/test_packaging.py -q`

Expected: one passing test.

Run: `python -m pip install -e ".[test]"`

Expected: editable install succeeds and installs the declared runtime/test dependencies.

---

### Task 2: Verified atomic artifact downloads

**Files:**
- Create: `ml/earshot_ml/artifacts.py`
- Create: `ml/tests/test_artifacts.py`
- Modify: `ml/earshot_ml/config.py`

**Interfaces:**
- Produces: `Artifact(url: str, path: Path, sha256: str)`, `sha256_file(path) -> str`, and `download_artifact(artifact) -> bool` where the boolean reports whether a download occurred.
- Produces: `config.MODEL_ARTIFACT` and `config.CLASS_MAP_ARTIFACT`.

- [ ] **Step 1: Write failing artifact tests**

```python
import hashlib
from pathlib import Path

import pytest

from earshot_ml.artifacts import Artifact, ChecksumError, download_artifact


def artifact_for(source: Path, dest: Path, expected: bytes):
    return Artifact(source.as_uri(), dest, hashlib.sha256(expected).hexdigest())


def test_download_verifies_then_atomically_installs(tmp_path):
    payload = b"verified model"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    dest = tmp_path / "models" / "model.tflite"
    assert download_artifact(artifact_for(source, dest, payload)) is True
    assert dest.read_bytes() == payload
    assert not dest.with_name(dest.name + ".part").exists()


def test_checksum_failure_preserves_existing_destination(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"corrupt")
    dest = tmp_path / "model.tflite"
    dest.write_bytes(b"known good")
    artifact = Artifact(source.as_uri(), dest, "0" * 64)
    with pytest.raises(ChecksumError):
        download_artifact(artifact)
    assert dest.read_bytes() == b"known good"
    assert not dest.with_name(dest.name + ".part").exists()
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_artifacts.py -q`

Expected: import failure because `earshot_ml.artifacts` does not exist.

- [ ] **Step 3: Implement minimal artifact downloader**

Implement a frozen `Artifact` dataclass, `ArtifactError`, `ChecksumError`, one-megabyte streaming via `urllib.request.urlopen`, SHA-256 validation, cleanup of `.part` in `finally`, and `os.replace()` only after successful verification. Return `False` without network access when an existing destination already has the expected digest.

In `config.py`, read `EARSHOT_MODEL_DIR` with the current `ml/models` path as default, and define the exact model/class-map artifact URLs and digests from the global constraints.

- [ ] **Step 4: Run artifact and original tests**

Run: `python -m pytest tests/test_artifacts.py tests/test_ml.py -q`

Expected: all tests pass.

---

### Task 3: Model tensor contract and inference validation

**Files:**
- Modify: `ml/earshot_ml/pipeline.py`
- Create: `ml/tests/test_pipeline.py`

**Interfaces:**
- Produces: `YamNet(..., interpreter=None)` for production loading or deterministic fake-interpreter tests.
- Preserves: `infer(waveform) -> tuple[np.ndarray, np.ndarray]` and `top(scores, k=5)`.

- [ ] **Step 1: Write failing fake-interpreter tests**

Create a fake interpreter exposing configurable input/output details and tensor values. Test that a valid model yields `(521,)` scores and `(1024,)` embeddings, while a score-only model raises `ModelContractError` containing observed output shapes. Test class maps with other than 521 rows, non-float inputs, wrong waveform length, and non-finite waveform values.

Representative assertion:

```python
def test_score_only_model_reports_contract_error(tmp_path):
    class_map = write_class_map(tmp_path, 521)
    fake = FakeInterpreter(outputs=[tensor_detail(7, [1, 521])])
    with pytest.raises(ModelContractError, match="1024"):
        YamNet(Path("unused"), class_map, interpreter=fake)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: failures because injection and contract errors are not implemented.

- [ ] **Step 3: Implement model validation**

Add `ModelContractError`. Validate class-map length, input dtype and shape, required outputs by last dimension, and diagnostic messages listing tensor names/shapes/dtypes. Resize a dynamic or incompatible input to `[WINDOW_SAMPLES]`, then allocate tensors and refresh details. Make `infer` reject anything except a finite float-convertible one-dimensional 15,600-sample waveform and normalize outputs by averaging frame dimensions.

- [ ] **Step 4: Verify model unit tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_ml.py -q`

Expected: all tests pass.

---

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

### Task 7: Package CLI, verified download, and lazy storage commands

**Files:**
- Create: `ml/earshot_ml/cli.py`
- Replace: `ml/cli.py` with a compatibility wrapper
- Create: `ml/tests/test_cli.py`

**Interfaces:**
- Produces: `earshot_ml.cli.main(argv=None)` and package console entry point.
- Preserves: all existing command names and options.

- [ ] **Step 1: Write failing CLI tests**

Test parser help, storage-only `sounds` and `forget` with no model files and no interpreter, `teach` reserved-name errors, checksum error reporting, and `download` calling `YamNet` after both artifacts are installed. Use temporary `EARSHOT_MODEL_DIR` paths and local artifact URLs; do not use network access.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_cli.py -q`

Expected: import failure because `earshot_ml.cli` does not exist.

- [ ] **Step 3: Move and harden CLI implementation**

Move command handlers and parser creation into `earshot_ml/cli.py`, accept optional argv, use artifact helpers for `download`, validate the actual model after download, and instantiate `TeachStore` directly for `sounds` and `forget`. Keep top-level `cli.py` as:

```python
from earshot_ml.cli import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify CLI compatibility**

Run: `python -m pytest tests/test_cli.py -q`

Expected: all tests pass.

Run: `python cli.py --help`

Expected: help lists all six commands.

Run: `earshot --help`

Expected: the installed console entry point lists the same commands.

---

### Task 8: Real-model integration test, documentation, CI, and hygiene

**Files:**
- Create: `ml/tests/test_real_model.py`
- Modify: `ml/README.md`
- Create: `.github/workflows/tests.yml`
- Delete after exact-path verification: generated `__pycache__`, `*.pyc`, and outer `__MACOSX` artifacts

**Interfaces:**
- Produces: opt-in pytest marker `integration` for the actual downloaded model.
- Produces: complete Windows/Pi operator documentation.

- [ ] **Step 1: Write the integration test before downloading into the project**

```python
from pathlib import Path

import numpy as np
import pytest

from earshot_ml import config
from earshot_ml.pipeline import YamNet


pytestmark = pytest.mark.integration


def test_downloaded_model_contract():
    if not config.MODEL_PATH.exists() or not config.CLASS_MAP_PATH.exists():
        pytest.skip("run `earshot download` first")
    model = YamNet()
    scores, embedding = model.infer(np.zeros(15_600, np.float32))
    assert scores.shape == (521,)
    assert embedding.shape == (1024,)
    assert np.isfinite(scores).all()
    assert np.isfinite(embedding).all()
    assert len(model.class_names) == 521
```

- [ ] **Step 2: Confirm the integration test skips before download**

Run: `python -m pytest tests/test_real_model.py -q`

Expected: one skipped test with the documented reason.

- [ ] **Step 3: Update docs and CI**

Rewrite README setup and validation sections for Windows and Raspberry Pi, explain both data flows, document `EARSHOT_MODEL_DIR`, automation versus hardware tests, stop-event backend usage, score semantics, calibration, and safety limitations. Add a repository-root Ubuntu GitHub Actions workflow with `ml` as its working directory; it installs `.[test]` and runs all tests except the integration marker on Python 3.11 and 3.14.

- [ ] **Step 4: Run full fast verification**

Run: `python -m pytest -m "not integration" -q`

Expected: all fast tests pass with zero failures or warnings.

- [ ] **Step 5: Download and verify the real model**

Run: `python -m earshot_ml.cli download`

Expected: both artifacts pass their fixed SHA-256 checks and the CLI reports a valid 521/1,024 tensor contract.

Run: `python -m pytest tests/test_real_model.py -q`

Expected: one passing integration test.

- [ ] **Step 6: Verify packaging and commands from a clean environment**

Create a fresh temporary virtual environment, install `.[test]`, run all fast tests, run both help entry points, and run the real-model test using the verified model directory.

Expected: installation and every non-hardware check succeed.

- [ ] **Step 7: Remove regenerable archive clutter safely**

Resolve and print every targeted cache and `__MACOSX` path, confirm all targets remain under `C:\Users\alexa\Downloads\earshot`, then remove only those verified paths. Do not remove source, models, tests, documentation, or virtual environments.

- [ ] **Step 8: Final review checkpoint**

Inspect the complete file set, rerun fast and real-model suites, and request a code-quality review. Record the exact test counts, skipped hardware-only checks, model digest, and any remaining limitations for the handoff.
