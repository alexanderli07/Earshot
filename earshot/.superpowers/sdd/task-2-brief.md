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

