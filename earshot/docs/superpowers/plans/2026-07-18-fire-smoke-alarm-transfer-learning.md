# Fire and Smoke Alarm Transfer Learning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a small, high-recall `fire_smoke_alarm` logistic head on frozen YAMNet embeddings, evaluate it without recording leakage, and run it through Earshot's existing offline event pipeline on Windows and Raspberry Pi.

**Architecture:** Keep the pinned YAMNet TFLite model as the only audio feature extractor. Add a training-only scikit-learn pipeline that exports a strictly validated NumPy artifact, then feed that artifact from the existing YAMNet embedding into a two-of-eight rolling gate and the existing debounce/event-delivery path. Keep corpus/collection, training, runtime artifact, and CLI responsibilities in separate modules.

**Tech Stack:** Python 3.10-3.14, NumPy, standard-library WAV/JSON/file APIs, existing LiteRT/TFLite YAMNet runtime, scikit-learn 1.x as an optional Windows training dependency, pytest.

## Global Constraints

- Preserve the existing YAMNet model, class map, non-alarm pretrained events, teach store, callback/queue behavior, and top-level compatibility wrapper.
- The runtime on Raspberry Pi must not import or install scikit-learn or TensorFlow training packages.
- Use mono 16 kHz float32 audio, 15,600-sample windows, and an 8,000-sample hop through the existing audio pipeline.
- Combine smoke-detector and building fire-alarm sounds into `label="fire_smoke_alarm"`, `urgency="high"`, and `source="trained"`.
- Require two qualifying windows among the latest eight windows and retain the existing 10-second event debounce.
- Cross-validation must be grouped by `source_group`, use five folds, and keep all derivatives of one source in one fold.
- Require 100% positive source-group recall, no more than 20% triggered negative source groups, and no more than 0.5 debounced false events per negative audio minute.
- Never overwrite a known-good model or report with a partial or failed write.
- Never modify, move, delete, or commit the user's WAV files; `ml/data/` remains local and ignored.
- A missing trained artifact preserves current generic `fire_alarm`/`smoke_alarm` behavior; an existing invalid artifact is a startup error.
- The output is a demo classifier, not a certified fire, smoke, accessibility, or life-safety system.
- Use PowerShell commands from `C:\Users\alexa\Desktop\Projects\Earshot\earshot\ml` and the project-local `.venv`.

---

## File Structure

**Create:**

- `ml/earshot_ml/alarm_data.py` — corpus metadata, manifest parsing, WAV validation, safe imports, and microphone-recording persistence.
- `ml/earshot_ml/alarm_model.py` — NumPy-only trained-head artifact, scoring, validation, atomic persistence, and rolling evidence gate.
- `ml/earshot_ml/alarm_training.py` — training-only preparation, augmentation, grouped cross-validation, fitting, calibration, reporting, and evaluation.
- `ml/tests/test_alarm_data.py` — dataset and collection contract tests.
- `ml/tests/test_alarm_model.py` — artifact, scoring, rollback, checksum, and gate tests.
- `ml/tests/test_alarm_training.py` — preprocessing, augmentation, leakage, fitting, threshold, metrics, and rollback tests.
- `ml/tests/test_alarm_corpus.py` — integration test over the local 17-WAV corpus and production YAMNet.

**Modify:**

- `.gitignore` — ignore local corpus and generated JSON reports.
- `ml/pyproject.toml` — add the training-only scikit-learn extra.
- `ml/earshot_ml/config.py` — add trained-alarm paths, labels, and gate settings.
- `ml/earshot_ml/core.py` — optionally load/use the head while retaining existing events and delivery.
- `ml/earshot_ml/cli.py` — add collect/train/evaluate commands and trained score display.
- `ml/tests/test_packaging.py` — enforce dependency and ignore boundaries.
- `ml/tests/test_artifacts.py` — enforce new config path behavior.
- `ml/tests/test_engine.py` — verify fallback, suppression, trained gating, and event payload.
- `ml/tests/test_cli.py` — verify parser, lazy imports, handlers, output, and concise errors.
- `ml/README.md` — document the full Windows-to-Pi workflow and limitations.

---

### Task 1: Corpus Inventory, Manifest, Configuration, and Git Hygiene

**Files:**

- Create: `ml/earshot_ml/alarm_data.py`
- Create: `ml/tests/test_alarm_data.py`
- Modify: `ml/earshot_ml/config.py:26-31`
- Modify: `ml/tests/test_artifacts.py:339-368`
- Modify: `ml/tests/test_packaging.py:60-69`
- Modify: `.gitignore`

**Interfaces:**

- Consumes: `pipeline.load_wav_16k_mono(path) -> np.ndarray`, `config.WINDOW_SAMPLES`, and `config.SAMPLE_RATE`.
- Produces: `AlarmDataError`, `CorpusEntry`, `CorpusInventory`, `inventory_corpus(data_dir)`, `load_manifest(data_dir)`, and trained-alarm configuration constants used by every later task.

- [ ] **Step 1: Write failing configuration and corpus tests**

Add these representative tests; include adjacent cases for malformed manifests,
invalid labels, invalid segment ranges, clips shorter than 15,600 decoded
samples, non-WAV files, missing class directories, and exact duplicate content.
The five-source-group minimum belongs to Task 6 training validation, not corpus
inventory.

```python
# tests/test_alarm_data.py
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from earshot_ml.alarm_data import AlarmDataError, inventory_corpus


def write_pcm16(path: Path, samples, rate=16_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(np.clip(samples, -1, 1) * 32767, dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def test_inventory_uses_manifest_source_groups_and_segments(tmp_path):
    audio = np.ones(16_000, dtype=np.float32) * 0.2
    write_pcm16(tmp_path / "alarm" / "a.wav", audio)
    write_pcm16(tmp_path / "not_alarm" / "n.wav", audio * 0.5)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": 1,
        "entries": [
            {"path": "alarm/a.wav", "label": "alarm",
             "source_group": "alarm-source", "segments": [[0.1, 0.8]]},
            {"path": "not_alarm/n.wav", "label": "not_alarm",
             "source_group": "negative-source", "segments": []},
        ],
    }), encoding="utf-8")

    inventory = inventory_corpus(tmp_path)

    assert [(entry.label, entry.source_group, entry.segments)
            for entry in inventory.entries] == [
        ("alarm", "alarm-source", ((0.1, 0.8),)),
        ("not_alarm", "negative-source", ()),
    ]


def test_manual_files_receive_stable_in_memory_groups(tmp_path):
    samples = np.linspace(-0.2, 0.2, 16_000, dtype=np.float32)
    write_pcm16(tmp_path / "alarm" / "a.wav", samples)
    write_pcm16(tmp_path / "not_alarm" / "n.wav", -samples)

    first = inventory_corpus(tmp_path)
    second = inventory_corpus(tmp_path)

    assert [entry.source_group for entry in first.entries] == [
        entry.source_group for entry in second.entries
    ]
    assert first.warnings == second.warnings
    assert "manifest" in " ".join(first.warnings).lower()


def test_exact_decoded_duplicates_are_counted_once(tmp_path):
    samples = np.ones(16_000, dtype=np.float32) * 0.25
    write_pcm16(tmp_path / "alarm" / "one.wav", samples)
    write_pcm16(tmp_path / "alarm" / "two.wav", samples)
    write_pcm16(tmp_path / "not_alarm" / "n.wav", -samples)

    inventory = inventory_corpus(tmp_path)

    assert len([e for e in inventory.entries if e.label == "alarm"]) == 1
    assert "duplicate" in " ".join(inventory.warnings).lower()


def test_inventory_rejects_cross_label_duplicate_content(tmp_path):
    samples = np.ones(16_000, dtype=np.float32) * 0.25
    write_pcm16(tmp_path / "alarm" / "a.wav", samples)
    write_pcm16(tmp_path / "not_alarm" / "n.wav", samples)

    with pytest.raises(AlarmDataError, match="both labels"):
        inventory_corpus(tmp_path)
```

Extend configuration and packaging tests:

```python
def test_config_defines_alarm_paths_under_expected_roots(monkeypatch, tmp_path):
    monkeypatch.setenv("EARSHOT_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("EARSHOT_ALARM_MODEL_PATH", str(tmp_path / "head.npz"))
    reloaded = importlib.reload(config)
    assert reloaded.ALARM_MODEL_PATH == tmp_path / "head.npz"
    assert reloaded.ALARM_REPORT_PATH == tmp_path / "models" / "fire_smoke_alarm_report.json"
    assert reloaded.ALARM_DATA_DIR.name == "alarm_demo"


def test_gitignore_excludes_local_alarm_data_and_reports():
    entries = {
        line.strip() for line in (ROOT.parent / ".gitignore").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {"ml/data/", "ml/models/*_report.json"} <= entries
```

- [ ] **Step 2: Run the focused tests to verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_data.py tests\test_artifacts.py tests\test_packaging.py -q
```

Expected: collection fails because `earshot_ml.alarm_data` and the new config constants do not exist; packaging assertions also fail.

- [ ] **Step 3: Add exact configuration and corpus contracts**

Add to `config.py`:

```python
_DEFAULT_ALARM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "alarm_demo"
ALARM_DATA_DIR = Path(os.environ.get("EARSHOT_ALARM_DATA_DIR") or _DEFAULT_ALARM_DATA_DIR)
ALARM_MODEL_PATH = Path(
    os.environ.get("EARSHOT_ALARM_MODEL_PATH")
    or MODEL_DIR / "fire_smoke_alarm_head.npz"
)
ALARM_REPORT_PATH = MODEL_DIR / "fire_smoke_alarm_report.json"
ALARM_EVENT_LABEL = "fire_smoke_alarm"
ALARM_EVENT_URGENCY = "high"
ALARM_REPLACED_LABELS = frozenset({"fire_alarm", "smoke_alarm"})
ALARM_GATE_COUNT = 2
ALARM_GATE_WINDOW = 8
```

Create `alarm_data.py` around these exact public types:

```python
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .pipeline import AudioFileError, load_wav_16k_mono

ALARM = "alarm"
NOT_ALARM = "not_alarm"
VALID_LABELS = frozenset({ALARM, NOT_ALARM})
MANIFEST_VERSION = 1


class AlarmDataError(ValueError):
    """The local supervised alarm corpus is missing or invalid."""


@dataclass(frozen=True)
class CorpusEntry:
    path: Path
    relative_path: str
    label: str
    source_group: str
    segments: tuple[tuple[float, float], ...]
    decoded_sha256: str
    duration_seconds: float


@dataclass(frozen=True)
class CorpusInventory:
    root: Path
    entries: tuple[CorpusEntry, ...]
    warnings: tuple[str, ...]

    def entries_for(self, label: str) -> tuple[CorpusEntry, ...]:
        return tuple(entry for entry in self.entries if entry.label == label)


def _decoded_digest(audio: np.ndarray) -> str:
    canonical = np.asarray(audio, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _validated_segments(value, *, duration: float, relative_path: str):
    if value is None:
        return ()
    if not isinstance(value, list):
        raise AlarmDataError(f"{relative_path}: segments must be a list")
    result = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise AlarmDataError(f"{relative_path}: each segment must be [start, end]")
        start, end = (float(pair[0]), float(pair[1]))
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or end > duration:
            raise AlarmDataError(f"{relative_path}: invalid segment [{start}, {end}]")
        result.append((start, end))
    return tuple(result)
```

Implement `load_manifest(data_dir)` to require a JSON object with `version == 1` and a list of unique relative-path entries. Implement `inventory_corpus(data_dir)` in this order:

1. Resolve `alarm/` and `not_alarm/`; reject missing/empty directories and non-`.wav` files.
2. Read manifest metadata when present; reject label/path disagreements and manifest paths outside `data_dir`.
3. Decode every WAV with `load_wav_16k_mono`; reject fewer than `WINDOW_SAMPLES` samples or any non-finite value.
4. Compute canonical decoded SHA-256 and duration.
5. Use manifest `source_group`, otherwise `manual-<first 16 digest characters>` and emit a warning.
6. Deduplicate same-label identical audio; reject a decoded digest present under both labels.
7. Sort entries by normalized relative path for deterministic folds and reports.

Update `.gitignore` with exactly:

```gitignore
ml/data/
ml/models/*_report.json
```

- [ ] **Step 4: Run focused tests to verify GREEN**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_data.py tests\test_artifacts.py tests\test_packaging.py -q
```

Expected: all focused tests pass, and `git status --short` no longer lists any WAV under `ml/data/`.

- [ ] **Step 5: Commit the inventory boundary**

```powershell
git add ..\.gitignore earshot_ml\config.py earshot_ml\alarm_data.py tests\test_alarm_data.py tests\test_artifacts.py tests\test_packaging.py
git commit -m "feat: add alarm corpus inventory"
```

---

### Task 2: Safe WAV Import and Microphone Collection

**Files:**

- Modify: `ml/earshot_ml/alarm_data.py`
- Modify: `ml/tests/test_alarm_data.py`

**Interfaces:**

- Consumes: Task 1's `AlarmDataError`, manifest schema, label constants, and corpus root layout; `pipeline.record(seconds, device=...)` is injected by the CLI later.
- Produces: `collect_files(label, paths, data_dir, source_group=None) -> tuple[Path, ...]`, `collect_recordings(label, count, seconds, data_dir, source_group=None, device=None, recorder=record, before_capture=None) -> tuple[Path, ...]`, and `write_pcm16_wav(path, audio) -> None`.

- [ ] **Step 1: Write failing collection tests**

```python
from earshot_ml.alarm_data import collect_files, collect_recordings


def test_collect_files_validates_then_copies_without_moving_source(tmp_path):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    samples = np.linspace(-0.25, 0.25, 16_000, dtype=np.float32)
    write_pcm16(source, samples)
    before = source.read_bytes()

    stored = collect_files("alarm", [source], data_dir, source_group="smoke-a")

    assert source.read_bytes() == before
    assert len(stored) == 1
    assert stored[0].parent == data_dir / "alarm"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert manifest["entries"][0]["source_group"] == "smoke-a"


def test_collect_collision_uses_hash_suffix_and_never_overwrites(tmp_path):
    first = tmp_path / "first" / "same.wav"
    second = tmp_path / "second" / "same.wav"
    write_pcm16(first, np.ones(16_000) * 0.1)
    write_pcm16(second, np.ones(16_000) * 0.2)

    stored = collect_files("alarm", [first, second], tmp_path / "data")

    assert stored[0].name == "same.wav"
    assert stored[1].stem.startswith("same-")
    assert stored[0].read_bytes() != stored[1].read_bytes()


def test_collect_recording_writes_atomic_mono_16k_pcm(tmp_path):
    audio = np.linspace(-1, 1, 16_000, dtype=np.float32)
    calls = []

    stored = collect_recordings(
        "not_alarm", 1, 1.0, tmp_path / "data", device=7,
        recorder=lambda seconds, device=None: calls.append(
            (seconds, device)) or audio,
    )

    assert calls == [(1.0, 7)]
    with wave.open(str(stored[0]), "rb") as saved:
        assert (saved.getnchannels(), saved.getframerate(), saved.getsampwidth()) == (1, 16_000, 2)
    assert not list(stored[0].parent.glob("*.part"))


def test_failed_manifest_replace_preserves_previous_manifest(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    source = tmp_path / "source.wav"
    write_pcm16(source, np.ones(16_000) * 0.2)
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest.write_text('{"version": 1, "entries": []}', encoding="utf-8")
    before = manifest.read_bytes()
    monkeypatch.setattr(alarm_data.os, "replace", lambda *_: (_ for _ in ()).throw(PermissionError("read-only")))

    with pytest.raises(AlarmDataError, match="read-only"):
        collect_files("alarm", [source], data_dir)

    assert manifest.read_bytes() == before
```

- [ ] **Step 2: Run collection tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_data.py -q
```

Expected: new imports fail because the collection functions do not exist.

- [ ] **Step 3: Implement atomic collection**

Add label validation, preflight decoding of every input, SHA-based collision names, per-file atomic copy, PCM16 serialization, and one atomic manifest update. Use these exact persistence primitives:

```python
import os
import shutil
import wave


def _atomic_replace_bytes(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    try:
        with part.open("wb") as output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(part, path)
    except OSError as exc:
        raise AlarmDataError(f"could not install {path}: {exc}") from exc
    finally:
        try:
            part.unlink()
        except OSError:
            pass


def write_pcm16_wav(path: Path, audio) -> None:
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size < config.WINDOW_SAMPLES or not np.isfinite(samples).all():
        raise AlarmDataError("recording must be finite mono audio at least one model window long")
    pcm = np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")

    def write(output):
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(config.SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())

    _atomic_replace_bytes(Path(path), write)
```

For imported files, preflight all sources before any destination write, then
copy with a helper that opens the source in a `with source.open("rb")` block and
passes the already-open destination to `shutil.copyfileobj`. Write manifest
JSON with `sort_keys=True`, `indent=2`, and a final newline through the same
atomic helper. If manifest installation fails, remove only newly created
destination files from this call; never remove a pre-existing corpus file.

`collect_recordings` validates label/count/seconds before microphone access,
invokes `before_capture(index, count, seconds)` when supplied, captures the
whole batch through the injected recorder, validates every array, and only then
persists it. Use `alarm-YYYYMMDD-HHMMSS-001.wav` and
`not_alarm-YYYYMMDD-HHMMSS-001.wav` for microphone recordings. Inject a clock
function in tests instead of patching global time. When `source_group` is not
supplied, store `source-<first 16 decoded-digest characters>` in the manifest;
an explicit group associates related edits or repeated microphone captures.
Return only newly added paths so exact duplicates are not counted as stored
clips.

- [ ] **Step 4: Verify collection and inventory together**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_data.py -q
```

Expected: all collection and inventory tests pass.

- [ ] **Step 5: Commit safe collection**

```powershell
git add earshot_ml\alarm_data.py tests\test_alarm_data.py
git commit -m "feat: collect alarm training audio safely"
```

---

### Task 3: NumPy Alarm Head Artifact and Rolling Evidence Gate

**Files:**

- Create: `ml/earshot_ml/alarm_model.py`
- Create: `ml/tests/test_alarm_model.py`

**Interfaces:**

- Consumes: `artifacts.sha256_file(path)`, Task 1's config constants, and a 1,024-value YAMNet embedding.
- Produces: `AlarmModelError`, immutable `AlarmHead`, `RollingEvidenceGate`, `save_alarm_head`, `load_alarm_head`, and `load_optional_alarm_head`.

- [ ] **Step 1: Write failing artifact, score, and gate tests**

```python
import numpy as np
import pytest

from earshot_ml.artifacts import sha256_file
from earshot_ml.alarm_model import (
    AlarmHead, AlarmModelError, RollingEvidenceGate,
    load_alarm_head, load_optional_alarm_head, save_alarm_head,
)


def make_head(model_digest="a" * 64, map_digest="b" * 64):
    weights = np.zeros(1024, dtype=np.float32)
    weights[0] = 2.0
    return AlarmHead(
        label="fire_smoke_alarm", urgency="high", feature_dim=1024,
        mean=np.zeros(1024, np.float32), scale=np.ones(1024, np.float32),
        weights=weights, bias=-1.0, threshold=0.7,
        gate_count=2, gate_window=8,
        yamnet_model_sha256=model_digest, class_map_sha256=map_digest,
    )


def test_head_uses_branch_stable_sigmoid():
    head = make_head()
    positive = np.zeros(1024, np.float32)
    positive[0] = 1_000
    negative = -positive
    assert head.score(positive) == pytest.approx(1.0)
    assert head.score(negative) == pytest.approx(0.0)


def test_artifact_round_trip_and_digest_validation(tmp_path):
    model = tmp_path / "yamnet.tflite"
    class_map = tmp_path / "map.csv"
    model.write_bytes(b"model")
    class_map.write_bytes(b"map")
    head = make_head(sha256_file(model), sha256_file(class_map))
    path = tmp_path / "head.npz"

    save_alarm_head(path, head)
    loaded = load_alarm_head(path, yamnet_model_path=model, class_map_path=class_map)

    assert loaded.label == "fire_smoke_alarm"
    np.testing.assert_array_equal(loaded.weights, head.weights)
    assert not path.with_name(path.name + ".part").exists()


def test_optional_loader_only_ignores_absence(tmp_path):
    assert load_optional_alarm_head(
        tmp_path / "missing.npz",
        yamnet_model_path=tmp_path / "model",
        class_map_path=tmp_path / "map",
    ) is None
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"bad")
    with pytest.raises(AlarmModelError):
        load_optional_alarm_head(
            corrupt, yamnet_model_path=tmp_path / "model",
            class_map_path=tmp_path / "map",
        )


def test_gate_accepts_nonconsecutive_two_of_eight_and_expires_old_evidence():
    gate = RollingEvidenceGate(required_count=2, window_size=8)
    assert gate.update(True) is False
    for _ in range(6):
        assert gate.update(False) is False
    assert gate.update(True) is True
    assert gate.update(False) is False
```

Also parameterize artifact failures for missing keys, object/string vectors, wrong shapes, non-finite arrays, zero/negative scales, invalid thresholds, invalid gate values, unsupported schema, checksum mismatch, and `os.replace` rollback preserving prior bytes.

- [ ] **Step 2: Run artifact tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_model.py -q
```

Expected: import failure because `alarm_model.py` does not exist.

- [ ] **Step 3: Implement the runtime-only artifact contract**

Use this exact public shape and keys:

```python
SCHEMA = "earshot.fire_smoke_alarm_head"
SCHEMA_VERSION = 1
FEATURE_DIM = 1024
ARTIFACT_KEYS = frozenset({
    "schema", "schema_version", "label", "urgency", "feature_dim",
    "mean", "scale", "weights", "bias", "threshold", "gate_count",
    "gate_window", "yamnet_model_sha256", "class_map_sha256",
})


class AlarmModelError(RuntimeError):
    """A trained alarm artifact is missing, corrupt, or incompatible."""


@dataclass(frozen=True)
class AlarmHead:
    label: str
    urgency: str
    feature_dim: int
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float
    gate_count: int
    gate_window: int
    yamnet_model_sha256: str
    class_map_sha256: str

    def score(self, embedding) -> float:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape != (self.feature_dim,) or not np.isfinite(vector).all():
            raise AlarmModelError("embedding must contain 1024 finite values")
        logit = float(np.dot((vector - self.mean) / self.scale, self.weights) + self.bias)
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exponential = np.exp(logit)
        return float(exponential / (1.0 + exponential))


class RollingEvidenceGate:
    def __init__(self, required_count: int, window_size: int):
        if isinstance(required_count, bool) or isinstance(window_size, bool):
            raise ValueError("gate sizes must be integers")
        if required_count <= 0 or window_size <= 0 or required_count > window_size:
            raise ValueError("gate requires 0 < required_count <= window_size")
        self.required_count = int(required_count)
        self.window_size = int(window_size)
        self._values = deque(maxlen=self.window_size)

    def update(self, above: bool) -> bool:
        self._values.append(bool(above))
        return sum(self._values) >= self.required_count

    def reset(self) -> None:
        self._values.clear()
```

`save_alarm_head` must validate the full dataclass first, serialize strings as
Unicode scalar arrays and all vectors as float32, open
`path.name + ".part"` as a file handle so `np.savez` does not append another
suffix, flush, `os.fsync`, and `os.replace`. `load_alarm_head` must use
`np.load(..., allow_pickle=False)`, copy every value before closing, reject
extra or missing keys, reconstruct/validate the dataclass, make the mean,
scale, and weight copies read-only with `setflags(write=False)`, then compare
both current file digests. `load_optional_alarm_head` returns `None` only for
`path is None` or a nonexistent path.

- [ ] **Step 4: Run artifact tests to verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_model.py -q
```

Expected: all model and gate tests pass.

- [ ] **Step 5: Commit the runtime artifact**

```powershell
git add earshot_ml\alarm_model.py tests\test_alarm_model.py
git commit -m "feat: add trained alarm artifact"
```

---

### Task 4: Optional Trained-Alarm Runtime Integration

**Files:**

- Modify: `ml/earshot_ml/core.py:28-101,289-354`
- Modify: `ml/tests/test_engine.py:17-134`

**Interfaces:**

- Consumes: Task 3's `AlarmHead`, `RollingEvidenceGate`, and `load_optional_alarm_head`.
- Produces: optional trained-head loading in `EarshotML`, duplicate generic alarm suppression, and trained events delivered through unchanged callback/queue APIs.

- [ ] **Step 1: Isolate existing engine tests and add failing trained-head tests**

First make local-artifact presence irrelevant to existing tests:

```python
def make_engine(yamnet, **kwargs):
    return EarshotML(
        yamnet=yamnet,
        taught_store_path=None,
        alarm_model_path=None,
        **kwargs,
    )
```

Add a fake head and integration cases:

```python
class FakeAlarmHead:
    label = "fire_smoke_alarm"
    urgency = "high"
    threshold = 0.70
    gate_count = 2
    gate_window = 8

    def score(self, value):
        return float(np.asarray(value)[0])


def test_trained_head_uses_same_embedding_and_two_of_eight_gate():
    positive = embedding(0)
    negative = embedding(1)
    fake = FakeYamNet(
        [(np.array([0.99], np.float32), positive),
         (np.array([0.99], np.float32), negative),
         (np.array([0.99], np.float32), positive)],
        class_names=["Fire alarm"],
    )
    engine = EarshotML(
        yamnet=fake, taught_store_path=None, alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
    )

    assert engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.0) == []
    assert engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.5) == []
    events = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=2.0)

    assert [(event.label, event.source) for event in events] == [
        ("fire_smoke_alarm", "trained")
    ]
    assert len(fake.infer_calls) == 3


def test_trained_head_suppresses_only_generic_alarm_specs():
    fake = FakeYamNet(
        class_names=["Fire alarm", "Smoke detector, smoke alarm", "Doorbell"]
    )
    engine = EarshotML(
        yamnet=fake, taught_store_path=None, alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
    )
    assert [spec["label"] for spec in engine._specs] == ["doorbell"]
```

Add cases proving missing artifact fallback, invalid existing artifact propagation, non-alarm and taught events unchanged, debounce unchanged, and callback/queue payload `source="trained"`.

- [ ] **Step 2: Run engine tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_engine.py tests\test_ml.py -q
```

Expected: constructor rejects `alarm_model_path`/`alarm_head` and trained tests fail.

- [ ] **Step 3: Integrate the optional head without changing EventDetector**

Extend the constructor:

```python
def __init__(self, on_event=None, event_queue=None, device=None,
             model_path=config.MODEL_PATH,
             class_map_path=config.CLASS_MAP_PATH,
             taught_store_path=config.TAUGHT_STORE_PATH,
             alarm_model_path=config.ALARM_MODEL_PATH,
             alarm_head=None,
             yamnet=None):
```

After YAMNet construction:

```python
self.alarm_head = (
    alarm_head if alarm_head is not None else
    load_optional_alarm_head(
        alarm_model_path,
        yamnet_model_path=model_path,
        class_map_path=class_map_path,
    )
)
self.alarm_gate = (
    RollingEvidenceGate(self.alarm_head.gate_count, self.alarm_head.gate_window)
    if self.alarm_head is not None else None
)
```

In `_resolve_event_map`, skip an entry only when a head exists and `entry["label"] in config.ALARM_REPLACED_LABELS`. In `process_window`, after pretrained observations and before taught matching, add:

```python
if self.alarm_head is not None:
    alarm_score = self.alarm_head.score(embedding)
    observations.append(Observation(
        label=self.alarm_head.label,
        confidence=alarm_score,
        above=self.alarm_gate.update(alarm_score >= self.alarm_head.threshold),
        urgency=self.alarm_head.urgency,
        source="trained",
        consecutive=1,
    ))
```

Update `Event.source` documentation to include `trained`; do not change public payload keys or `EventDetector`.

- [ ] **Step 4: Run engine regression tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_engine.py tests\test_ml.py -q
```

Expected: existing and trained engine tests pass.

- [ ] **Step 5: Commit runtime integration**

```powershell
git add earshot_ml\core.py tests\test_engine.py
git commit -m "feat: run trained alarm detections"
```

---

### Task 5: Recording-Level Preparation and Leakage-Safe Augmentation

**Files:**

- Create: `ml/earshot_ml/alarm_training.py`
- Create: `ml/tests/test_alarm_training.py`

**Interfaces:**

- Consumes: Task 1's `CorpusInventory`/`CorpusEntry`, Task 3's `AlarmHead`, and existing `clip_windows`/YAMNet inference.
- Produces: `TrainingError`, `PreparedRecording`, `WeightedWindow`, `_prepare_recordings`, `_evenly_spaced`, `_augment_training_windows`, and `_weighted_standardization` for Task 6.

- [ ] **Step 1: Write failing preprocessing and augmentation tests**

```python
from earshot_ml.alarm_training import (
    TrainingError, _augment_training_windows, _evenly_spaced,
    _prepare_recordings, _weighted_standardization, mix_at_snr, rms,
)


def corpus_entry(tmp_path, *, label="alarm", segments=(), group="group-a"):
    path = tmp_path / label / "clip.wav"
    return CorpusEntry(
        path=path,
        relative_path=f"{label}/clip.wav",
        label=label,
        source_group=group,
        segments=tuple(segments),
        decoded_sha256="a" * 64,
        duration_seconds=2.0,
    )


def inventory(entry):
    return CorpusInventory(
        root=entry.path.parents[1], entries=(entry,), warnings=()
    )


def weighted_window(value, group, weight, *, label=1):
    return WeightedWindow(
        waveform=np.full(config.WINDOW_SAMPLES, value, np.float32),
        label=label,
        source_group=group,
        source_path=f"{group}.wav",
        start_sample=0,
        weight=weight,
        augmentation_noise_group=None,
    )


def test_evenly_spaced_caps_long_recordings_deterministically():
    selected = _evenly_spaced(np.arange(100), limit=40)
    assert len(selected) == 40
    assert selected[0] == 0
    assert selected[-1] == 99
    np.testing.assert_array_equal(selected, _evenly_spaced(np.arange(100), 40))


def test_positive_segments_require_half_window_overlap_and_activity(tmp_path):
    entry = corpus_entry(tmp_path, label="alarm", segments=((0.5, 1.5),))
    quiet = np.zeros(config.WINDOW_SAMPLES, np.float32)
    active = np.ones(config.WINDOW_SAMPLES, np.float32) * 0.2
    audio = np.concatenate([quiet[:8_000], active, quiet[:8_000]])
    prepared = _prepare_recordings(
        inventory(entry), audio_loader=lambda _path: audio
    )
    assert len(prepared[0].windows) == 1
    assert prepared[0].windows[0].start_sample == 8_000


def test_positive_augmentation_preserves_parent_weight_and_training_groups():
    positive = weighted_window(value=0.5, group="positive-a", weight=0.25)
    noise = weighted_window(value=0.1, group="negative-a", weight=1.0)
    augmented = _augment_training_windows(
        [positive], [noise], rng=np.random.default_rng(7)
    )
    assert len(augmented) == 3
    assert sum(item.weight for item in augmented) == pytest.approx(0.25)
    assert {item.source_group for item in augmented} == {"positive-a"}
    assert all(item.augmentation_noise_group in {None, "negative-a"}
               for item in augmented)


def test_noise_scaling_reaches_requested_snr():
    signal = np.ones(config.WINDOW_SAMPLES, np.float32) * 0.4
    noise = np.ones(config.WINDOW_SAMPLES, np.float32) * 0.2
    mixed = mix_at_snr(signal, noise, snr_db=10.0)
    added = mixed - signal
    ratio_db = 20 * np.log10(rms(signal) / rms(added))
    assert ratio_db == pytest.approx(10.0, abs=0.05)


def test_weighted_standardization_uses_weights_and_protects_zero_scale():
    values = np.array([[0.0, 3.0], [10.0, 3.0]], np.float32)
    mean, scale = _weighted_standardization(values, np.array([0.9, 0.1]))
    np.testing.assert_allclose(mean, [1.0, 3.0])
    assert scale[1] == 1.0
```

Also test: positive files with no eligible windows fail by path; negatives retain silence; at most 40 windows per file; manual segments default to the full file; non-silent noise is required; validation groups never become noise; seeded augmentation is deterministic; final all-data augmentation can use all negative groups.

- [ ] **Step 2: Run preprocessing tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_training.py -q
```

Expected: import failure because `alarm_training.py` does not exist.

- [ ] **Step 3: Implement preparation and augmentation seams**

Use immutable internal records:

```python
class TrainingError(AlarmDataError):
    """Alarm training or evaluation could not produce a valid result."""


@dataclass(frozen=True)
class PreparedWindow:
    waveform: np.ndarray
    start_sample: int
    weight: float


@dataclass(frozen=True)
class PreparedRecording:
    entry: CorpusEntry
    windows: tuple[PreparedWindow, ...]
    duration_seconds: float


@dataclass(frozen=True)
class WeightedWindow:
    waveform: np.ndarray
    label: int
    source_group: str
    source_path: str
    start_sample: int
    weight: float
    augmentation_noise_group: str | None
```

Implement `_prepare_recordings(inventory, *, audio_loader=load_wav_16k_mono)`
using window starts
`range(0, len(audio) - WINDOW_SAMPLES + 1, HOP_SAMPLES)`. Positive
eligibility is at least 50% overlap with a configured segment (or the full
file) and RMS at least `max(1e-4, 0.05 * p95_rms)`. Negatives retain all
windows. Pass eligible indices through `_evenly_spaced(..., 40)`, then assign
each original window `1 / retained_count` recording weight.

Implement:

```python
def rms(values):
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(array))))


def mix_at_snr(signal, noise, snr_db):
    signal_rms = rms(signal)
    noise_rms = rms(noise)
    if noise_rms < 1e-6:
        raise TrainingError("augmentation noise must be non-silent")
    target_noise_rms = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    scaled_noise = np.asarray(noise, np.float32) * (target_noise_rms / noise_rms)
    return np.clip(np.asarray(signal, np.float32) + scaled_noise, -1, 1)
```

For each positive original, output original, seeded gain-only, and seeded gain-plus-noise copies at one-third parent weight each. Gain is uniform in `[0.35, 1.0]`; SNR is uniform in `[8.0, 20.0]`. Select noise only from provided non-silent negative training windows. `_weighted_standardization` computes weighted mean/variance in float64 and returns float32 arrays, replacing scale below `1e-6` with `1.0`.

- [ ] **Step 4: Run preprocessing tests to verify GREEN**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_training.py -q
```

Expected: preprocessing and augmentation tests pass without importing scikit-learn.

- [ ] **Step 5: Commit preprocessing**

```powershell
git add earshot_ml\alarm_training.py tests\test_alarm_training.py
git commit -m "feat: prepare alarm training windows"
```

---

### Task 6: Grouped Logistic Training, Threshold Calibration, Evaluation, and Reports

**Files:**

- Modify: `ml/earshot_ml/alarm_training.py`
- Modify: `ml/tests/test_alarm_training.py`
- Modify: `ml/pyproject.toml:18-19`
- Modify: `ml/tests/test_packaging.py:35-44`

**Interfaces:**

- Consumes: Tasks 1, 3, and 5 contracts plus injected `YamNet.infer(waveform) -> (scores, embedding)`.
- Produces: `EvaluationMetrics`, `TrainingReport`, `EvaluationReport`, `train_alarm`, and `evaluate_alarm` for the CLI and integration test.

- [ ] **Step 1: Write failing packaging, fold, calibration, and rollback tests**

Add to packaging tests:

```python
def test_training_extra_is_optional_and_runtime_stays_lightweight():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert any(item.startswith("scikit-learn>=1.7")
               for item in data["project"]["optional-dependencies"]["train"])
    assert all(not item.startswith("scikit-learn")
               for item in data["project"]["dependencies"])
```

Add training tests using a fake embedder whose first feature separates classes and a synthetic inventory with at least five groups per label:

```python
@dataclass(frozen=True)
class ScoredRecording:
    source_group: str
    source_path: str
    label: int
    duration_seconds: float
    scores: tuple[float, ...]


def scored_corpus(*, positive, negative):
    result = []
    for label, groups in ((1, positive), (0, negative)):
        for group, scores in groups.items():
            result.append(ScoredRecording(
                source_group=group,
                source_path=f"{group}.wav",
                label=label,
                duration_seconds=max(1.0, len(scores) * 0.5),
                scores=tuple(float(score) for score in scores),
            ))
    return tuple(result)


def prepared_recording(tmp_path, group, label):
    text_label = "alarm" if label else "not_alarm"
    entry = corpus_entry(
        tmp_path, label=text_label, group=group,
    )
    window = PreparedWindow(
        waveform=np.full(config.WINDOW_SAMPLES, float(label), np.float32),
        start_sample=0,
        weight=1.0,
    )
    return PreparedRecording(entry=entry, windows=(window,), duration_seconds=1.0)


@pytest.fixture
def synthetic_recordings(tmp_path):
    return tuple(
        prepared_recording(tmp_path, f"p-{index}", 1)
        for index in range(5)
    ) + tuple(
        prepared_recording(tmp_path, f"n-{index}", 0)
        for index in range(5)
    )


def test_grouped_folds_never_leak_source_groups(synthetic_recordings):
    folds = _make_grouped_folds(synthetic_recordings, seed=0, folds=5)
    for fold in folds:
        assert set(fold.train_groups).isdisjoint(fold.validation_groups)
        assert {item.label for item in fold.train_recordings} == {0, 1}


def test_threshold_is_highest_candidate_meeting_all_ceilings():
    scored = scored_corpus(
        positive={"p1": [0.9, 0.1, 0.8], "p2": [0.75, 0.8]},
        negative={"n1": [0.7, 0.1], "n2": [0.2, 0.1]},
    )
    threshold = _select_threshold(scored)
    assert threshold == pytest.approx(0.75)
    metrics = _evaluate_threshold(scored, threshold)
    assert metrics.positive_groups_triggered == metrics.positive_groups_total
    assert metrics.negative_groups_triggered <= 0.2 * metrics.negative_groups_total


def test_always_positive_scores_are_rejected():
    scored = scored_corpus(
        positive={f"p{i}": [0.5, 0.5] for i in range(5)},
        negative={f"n{i}": [0.5, 0.5] for i in range(5)},
    )
    with pytest.raises(TrainingError, match="false-alert"):
        _select_threshold(scored)


def test_failed_training_preserves_known_good_artifact(
        tmp_path, monkeypatch):
    output = tmp_path / "head.npz"
    output.write_bytes(b"known-good")
    before = output.read_bytes()
    monkeypatch.setattr(
        alarm_training,
        "inventory_corpus",
        lambda _path: (_ for _ in ()).throw(TrainingError("bad corpus")),
    )
    with pytest.raises(TrainingError):
        train_alarm(tmp_path / "data", output, tmp_path / "report.json")
    assert output.read_bytes() == before
```

Also test deterministic folds/scores, per-file gate/debounce resets, source-group aggregation, exact false-events-per-minute calculation, OOF versus in-sample report labels, final threshold `min(oof, final)`, final-head ceiling enforcement, report-first/artifact-last ordering, JSON rollback, and no model replacement on any failure.

- [ ] **Step 2: Run training tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_training.py tests\test_packaging.py -q
```

Expected: missing training APIs and missing `train` extra cause failures.

- [ ] **Step 3: Add the optional dependency and exact report contracts**

Add:

```toml
[project.optional-dependencies]
test = ["pytest", "tomli; python_version < '3.11'"]
train = ["scikit-learn>=1.7,<2"]
```

Add public immutable report types:

```python
@dataclass(frozen=True)
class Fold:
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    train_recordings: tuple[PreparedRecording, ...]
    validation_recordings: tuple[PreparedRecording, ...]


@dataclass(frozen=True)
class ScoredRecording:
    source_group: str
    source_path: str
    label: int
    duration_seconds: float
    scores: tuple[float, ...]


@dataclass(frozen=True)
class EvaluationMetrics:
    positive_groups_total: int
    positive_groups_triggered: int
    negative_groups_total: int
    negative_groups_triggered: int
    false_events: int
    negative_audio_minutes: float
    false_triggers_per_minute: float
    files: tuple[dict, ...]


@dataclass(frozen=True)
class TrainingReport:
    artifact_path: Path
    report_path: Path
    deployment_threshold: float
    oof_metrics: EvaluationMetrics
    in_sample_metrics: EvaluationMetrics
    payload: dict


@dataclass(frozen=True)
class EvaluationReport:
    metrics: EvaluationMetrics
    evaluation_scope: str
    payload: dict
```

- [ ] **Step 4: Implement grouped fitting and evaluation**

Import scikit-learn only inside helpers:

```python
def _load_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise TrainingError(
            'alarm training requires `python -m pip install -e ".[train]"`'
        ) from exc
    return LogisticRegression, StratifiedGroupKFold
```

Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` over one row per prepared recording. Reject fewer than five source groups in either class, mixed labels within a source group, overlapping train/validation groups, or a training fold missing a label.

For each fold, build training windows and augment positives using only negative training groups. Infer embeddings through the supplied YAMNet instance, compute weighted normalization, then fit:

```python
classifier = LogisticRegression(
    C=1.0,
    solver="liblinear",
    class_weight="balanced",
    max_iter=2000,
    random_state=seed,
)
classifier.fit(normalized_embeddings, labels, sample_weight=weights)
```

Convert each fitted classifier to an `AlarmHead` with its fold normalization, weights, bias, temporary threshold 0.5, two-of-eight gate metadata, and current YAMNet/class-map hashes. Score untouched validation recordings and pool the OOF scores.

`_evaluate_threshold` must instantiate fresh `RollingEvidenceGate(2, 8)` and `EventDetector(10.0)` at each file boundary. Use `now = window_index * HOP_SAMPLES / SAMPLE_RATE`, one `Observation(source="trained", consecutive=1)`, and group-level any-file trigger aggregation. Candidate thresholds are `sorted({0.0, 1.0, *finite_scores}, reverse=True)`. Select the first candidate satisfying all three global ceilings.

Fit the final all-data model using all groups and the same augmentation rules. Find its highest all-positive-recall threshold, deploy `min(oof_threshold, final_threshold)`, and re-evaluate both OOF and final predictions at the deployed value. Both must meet every ceiling.

Implement exact entry points:

```python
def train_alarm(data_dir, output_path, report_path, *, seed=0, yamnet=None,
                yamnet_model_path=config.MODEL_PATH,
                class_map_path=config.CLASS_MAP_PATH) -> TrainingReport:
    """Cross-validate, fit, report, and atomically install one alarm head."""


def evaluate_alarm(data_dir, alarm_model_path, *, yamnet=None,
                   yamnet_model_path=config.MODEL_PATH,
                   class_map_path=config.CLASS_MAP_PATH) -> EvaluationReport:
    """Evaluate one installed head; label matching corpus hashes in-sample."""
```

While embedding original windows, also aggregate each recording's strongest raw
YAMNet class names/scores into a `content_audit` report section. Never use those
raw class scores to include or exclude labeled windows. Write JSON with sorted
keys, indentation, a final newline, flush, `os.fsync`, and atomic replacement.
On success, install the report first and the model artifact last. On failure,
write a diagnostic report to
`report_path.with_name(report_path.stem + "_failed_report.json")`; preserve
existing report and model bytes.

- [ ] **Step 5: Run training and regression tests**

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,train]"
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_training.py tests\test_alarm_model.py tests\test_packaging.py -q
```

Expected: scikit-learn installs only in the development environment, all
focused tests pass, ordinary imports do not require scikit-learn, and failure
tests preserve previous artifacts.

- [ ] **Step 6: Commit supervised training**

```powershell
git add pyproject.toml earshot_ml\alarm_training.py tests\test_alarm_training.py tests\test_packaging.py
git commit -m "feat: train alarm classifier on YAMNet"
```

---

### Task 7: CLI Collection, Training, Evaluation, and Trained Score Display

**Files:**

- Modify: `ml/earshot_ml/cli.py:1-184`
- Modify: `ml/tests/test_cli.py:1-505`

**Interfaces:**

- Consumes: Tasks 2, 3, 4, and 6 public APIs.
- Produces: `earshot collect`, `earshot train-alarm`, `earshot evaluate-alarm`, optional trained score in `top5`, and automatic head use in `run`.

- [ ] **Step 1: Write failing parser, handler, lazy-import, output, and error tests**

Update:

```python
COMMANDS = (
    "download", "top5", "run", "teach", "sounds", "forget",
    "collect", "train-alarm", "evaluate-alarm",
)
```

Add representative handler tests:

```python
def test_collect_combines_files_and_recordings(monkeypatch, tmp_path, capsys):
    imported = []
    captured = []
    monkeypatch.setattr(cli, "collect_files",
                        lambda label, paths, data_dir, source_group=None:
                        imported.extend(paths) or (tmp_path / "a.wav",))
    monkeypatch.setattr(cli, "collect_recordings",
                        lambda label, count, seconds, data_dir, **kwargs:
                        captured.append((label, count, seconds, kwargs))
                        or (tmp_path / "b.wav",))
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli.main(["collect", "alarm", "source.wav", "--record", "1",
              "--seconds", "1", "--device", "1", "--data-dir", str(tmp_path)])

    assert imported == [Path("source.wav")]
    assert len(captured) == 1
    assert captured[0][1:3] == (1, 1.0)
    assert captured[0][3]["device"] == 1
    assert "stored 2" in capsys.readouterr().out


def test_train_alarm_passes_paths_and_seed(monkeypatch, tmp_path, capsys):
    received = []
    fake_report = SimpleNamespace(
        deployment_threshold=0.73,
        oof_metrics=SimpleNamespace(positive_groups_triggered=7,
                                    positive_groups_total=7,
                                    negative_groups_triggered=0,
                                    negative_groups_total=10,
                                    false_triggers_per_minute=0.0),
    )
    monkeypatch.setattr(cli, "_train_alarm",
                        lambda data, output, report, seed=0: received.append(
                            (data, output, report, seed)) or fake_report)

    cli.main(["train-alarm", "--data-dir", str(tmp_path / "data"),
              "--output", str(tmp_path / "head.npz"), "--seed", "9"])

    assert received[0][3] == 9
    assert "threshold 0.730" in capsys.readouterr().out


def test_help_does_not_import_sklearn(monkeypatch):
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith("sklearn"):
            pytest.fail("help imported training dependency")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])
    assert exc_info.value.code == 0
```

Add cases for file-only/recording-only collection, invalid labels/count/duration/no input before mutation, concise audio/model/data/training errors, evaluation scope and per-file output, top5 with/without a head, malformed existing head, and run's configured artifact path.

- [ ] **Step 2: Run CLI tests to verify RED**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli.py -q
```

Expected: commands and handler imports do not exist.

- [ ] **Step 3: Implement lazy handler imports and parser definitions**

Keep training imports out of module import time:

```python
def _train_alarm(*args, **kwargs):
    from .alarm_training import train_alarm
    return train_alarm(*args, **kwargs)


def _evaluate_alarm(*args, **kwargs):
    from .alarm_training import evaluate_alarm
    return evaluate_alarm(*args, **kwargs)


def cmd_train_alarm(args):
    output_path = Path(args.output)
    report_path = (
        config.ALARM_REPORT_PATH
        if output_path == config.ALARM_MODEL_PATH
        else output_path.with_name("fire_smoke_alarm_report.json")
    )
    result = _train_alarm(args.data_dir, output_path, report_path, seed=args.seed)
    metrics = result.oof_metrics
    print(
        f"trained fire_smoke_alarm threshold {result.deployment_threshold:.3f}; "
        f"recall {metrics.positive_groups_triggered}/{metrics.positive_groups_total}; "
        f"negative groups {metrics.negative_groups_triggered}/{metrics.negative_groups_total}; "
        f"false triggers/min {metrics.false_triggers_per_minute:.3f}"
    )


def cmd_evaluate_alarm(args):
    result = _evaluate_alarm(args.data_dir, args.model)
    for item in result.metrics.files:
        print(f"  {item['label']:<9} {item['triggered']!s:<5} {item['path']}")
    print(json.dumps(result.payload, sort_keys=True))
```

`cmd_collect` validates arguments before calling `collect_files` or
`collect_recordings`, passes the existing `record` function as the injected
recorder, supplies a `before_capture` callback that prompts and waits 0.2
seconds, combines stored paths, and prints each plus `stored N clips`. Dataset
functions may be imported at module scope because they do not import
scikit-learn.

Add parser definitions exactly matching the approved CLI:

```text
collect {alarm,not_alarm} [WAV ...] --record N --seconds S --device INDEX --data-dir PATH --source-group NAME
train-alarm --data-dir PATH --output PATH --seed INTEGER
evaluate-alarm --data-dir PATH --model PATH
```

Defaults are `record=0`, `seconds=5.0`, `data_dir=config.ALARM_DATA_DIR`, `output/model=config.ALARM_MODEL_PATH`, and `seed=0`.

In `cmd_top5`, use `scores, embedding = yamnet.infer(waveform)`, load the
optional head once before listening, and append
`  |  fire_smoke_alarm {head.score(embedding):.2f}` only when present. In
`cmd_run`, pass `alarm_model_path=config.ALARM_MODEL_PATH`. Catch
`AlarmDataError` (which is also the base of `TrainingError`) and
`AlarmModelError` in the expected domain-error tuple; never catch generic
`OSError`.

- [ ] **Step 4: Run CLI and engine regression tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_cli.py tests\test_engine.py tests\test_ml.py -q
```

Expected: all CLI, engine, and detector tests pass; help and non-training commands do not import scikit-learn.

- [ ] **Step 5: Commit CLI integration**

```powershell
git add earshot_ml\cli.py tests\test_cli.py
git commit -m "feat: expose alarm training workflow"
```

---

### Task 8: Real Corpus Evaluation, Documentation, and Full Verification

**Files:**

- Create: `ml/tests/test_alarm_corpus.py`
- Modify: `ml/README.md`

**Interfaces:**

- Consumes: the complete public CLI and Python APIs plus the local 17-WAV corpus.
- Produces: repeatable real-data acceptance evidence, a locally generated trained artifact/report, Windows/Pi operating instructions, and final regression evidence.

- [ ] **Step 1: Write the corpus integration test**

```python
import hashlib
from pathlib import Path

import numpy as np
import pytest

from earshot_ml import config
from earshot_ml.alarm_model import load_alarm_head
from earshot_ml.alarm_training import evaluate_alarm, train_alarm

pytestmark = pytest.mark.integration


def file_hashes(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.wav"))
    }


def test_local_alarm_corpus_trains_without_modifying_audio(tmp_path):
    data_dir = config.ALARM_DATA_DIR
    if not config.MODEL_PATH.exists() or not config.CLASS_MAP_PATH.exists():
        pytest.skip("run `earshot download` before the alarm corpus integration test")
    if not (data_dir / "alarm").exists() or not (data_dir / "not_alarm").exists():
        pytest.skip("populate data/alarm_demo/alarm and not_alarm first")
    before = file_hashes(data_dir)
    output = tmp_path / "fire_smoke_alarm_head.npz"
    report_path = tmp_path / "fire_smoke_alarm_report.json"

    trained = train_alarm(data_dir, output, report_path, seed=0)
    head = load_alarm_head(
        output, yamnet_model_path=config.MODEL_PATH,
        class_map_path=config.CLASS_MAP_PATH,
    )
    evaluated = evaluate_alarm(data_dir, output)

    assert trained.oof_metrics.positive_groups_triggered == trained.oof_metrics.positive_groups_total
    assert trained.oof_metrics.negative_groups_triggered <= 0.2 * trained.oof_metrics.negative_groups_total
    assert trained.oof_metrics.false_triggers_per_minute <= 0.5
    assert evaluated.metrics.positive_groups_triggered == evaluated.metrics.positive_groups_total
    assert np.isfinite(head.score(np.zeros(1024, np.float32)))
    assert file_hashes(data_dir) == before
```

- [ ] **Step 2: Install the training extra and run the integration test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[test,train]"
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_corpus.py -q -s
```

Expected: scikit-learn installs only in the Windows development environment; the real test passes, prints any negative triggers, and preserves all WAV hashes. If the metric ceilings fail, do not weaken assertions: inspect mislabeled/mixed files, source grouping, or add hard negatives, then rerun.

- [ ] **Step 3: Generate the local production artifact and verify it**

```powershell
.\.venv\Scripts\earshot.exe train-alarm
.\.venv\Scripts\earshot.exe evaluate-alarm
.\.venv\Scripts\earshot.exe --help
```

Expected: training creates ignored `models\fire_smoke_alarm_head.npz` and `models\fire_smoke_alarm_report.json`; evaluation reports 7/7 positive source groups triggered and both false-alert ceilings satisfied; help lists all nine commands.

- [ ] **Step 4: Update README with exact operating instructions**

Document:

1. `pip install -e ".[test,train]"` on Windows and ordinary `pip install -e .` on Pi.
2. Canonical `data/alarm_demo/{alarm,not_alarm}` layout and uncommitted-data policy.
3. File import and device-1 microphone collection commands, including `--source-group` for related captures.
4. Alarm-only positive clips, optional manifest time segments, hard-negative examples, and current corpus limitations.
5. Training, OOF versus in-sample evaluation, the exact recall/false-alert ceilings, and failure behavior.
6. Artifact/report locations, checksum compatibility,
   `EARSHOT_ALARM_DATA_DIR`, and `EARSHOT_ALARM_MODEL_PATH`.
7. Copying `fire_smoke_alarm_head.npz` to the Pi model directory while retaining matching YAMNet artifacts.
8. `top5` trained-score output, `run` event shape, and two-of-eight plus debounce behavior.
9. A Windows device-1 playback acceptance pass and a later target-Pi microphone pass.
10. The explicit statement that this demo must not replace certified alarms or emergency procedures.

- [ ] **Step 5: Run complete automated verification**

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not integration" -q
.\.venv\Scripts\python.exe -m pytest tests\test_real_model.py -q
.\.venv\Scripts\python.exe -m pytest tests\test_alarm_corpus.py -q -s
.\.venv\Scripts\python.exe -m pip check
git diff --check
git status --short --untracked-files=all
```

Expected:

- Every non-integration test passes.
- The existing production-model contract test passes.
- The local corpus integration test passes all three metric ceilings.
- `pip check` reports no broken requirements.
- `git diff --check` reports no whitespace errors.
- No WAV is listed because `ml/data/` is ignored.
- Only intended source/tests/docs changes remain; generated model/report files stay ignored.

- [ ] **Step 6: Perform the manual Windows microphone acceptance pass**

Run in separate terminals as appropriate:

```powershell
.\.venv\Scripts\earshot.exe top5 --device 1
.\.venv\Scripts\earshot.exe run --device 1
```

Play at least three alarm WAVs through a separate phone/speaker at moderate volume. Expected: `top5` shows the appended `fire_smoke_alarm` score without repeated `peak 1.00` clipping; `run` emits a high-urgency trained event after two qualifying windows within eight. Record any miss, false event, peak level, and latency. This manual evidence is required for microphone acceptance but does not block automated code verification when audio hardware is unavailable to the agent.

- [ ] **Step 7: Commit integration tests and documentation**

```powershell
git add README.md tests\test_alarm_corpus.py
git commit -m "docs: explain trained alarm workflow"
```

---

## Final Review Checklist

- [ ] Confirm no implementation commit contains files under `ml/data/` or generated `ml/models/*.npz`/`*_report.json` artifacts.
- [ ] Confirm runtime imports succeed in an environment without scikit-learn.
- [ ] Confirm an absent head preserves old fire/smoke labels and an invalid present head fails clearly.
- [ ] Confirm trained mode emits only one shared high-urgency event and preserves all unrelated events/teach behavior.
- [ ] Confirm OOF and in-sample metrics are labeled separately and both satisfy the mandatory ceilings.
- [ ] Confirm corpus WAV hashes match their pre-training values.
- [ ] Confirm README and CLI help use the exact implemented names and paths.
- [ ] Confirm the life-safety disclaimer remains prominent.
