import hashlib
import importlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from earshot_ml import config
from earshot_ml import artifacts as artifacts_module
from earshot_ml.artifacts import (
    Artifact,
    ArtifactError,
    ChecksumError,
    download_artifact,
    sha256_file,
)


MODEL_URL = "https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite"
MODEL_SHA256 = "141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3"
CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "dfffd623b6be8d1d9744b8e261fbac370d17c46d/research/audioset/yamnet/"
    "yamnet_class_map.csv"
)
CLASS_MAP_SHA256 = "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"


def artifact_for(source: Path, dest: Path, expected: bytes) -> Artifact:
    return Artifact(source.as_uri(), dest, hashlib.sha256(expected).hexdigest())


def assert_no_part_files(directory: Path) -> None:
    assert list(directory.glob("*.part")) == []


class ChunkedResponse:
    def __init__(self, chunks, *, content_length=None, before_first_read=None):
        self.headers = {}
        if content_length is not None:
            self.headers["Content-Length"] = str(content_length)
        self._chunks = iter(chunks)
        self._before_first_read = before_first_read
        self.read_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, size):
        del size
        self.read_calls += 1
        if self._before_first_read is not None:
            callback = self._before_first_read
            self._before_first_read = None
            callback()
        return next(self._chunks, b"")


def test_sha256_file_streams_complete_file(tmp_path):
    payload = b"a" * (1024 * 1024) + b"second chunk"
    path = tmp_path / "payload.bin"
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_artifact_is_immutable(tmp_path):
    artifact = Artifact("file:///source", tmp_path / "dest", "0" * 64)

    with pytest.raises(FrozenInstanceError):
        artifact.url = "file:///other"


def test_download_verifies_then_atomically_installs(tmp_path):
    payload = b"verified model"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    dest = tmp_path / "models" / "model.tflite"

    assert download_artifact(artifact_for(source, dest, payload)) is True

    assert dest.read_bytes() == payload
    assert_no_part_files(dest.parent)


def test_checksum_failure_preserves_existing_destination(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"corrupt")
    dest = tmp_path / "model.tflite"
    dest.write_bytes(b"known good")
    artifact = Artifact(source.as_uri(), dest, "0" * 64)

    with pytest.raises(ChecksumError) as exc_info:
        download_artifact(artifact)

    actual = hashlib.sha256(b"corrupt").hexdigest()
    assert dest.name in str(exc_info.value)
    assert artifact.sha256 in str(exc_info.value)
    assert actual in str(exc_info.value)
    assert dest.read_bytes() == b"known good"
    assert_no_part_files(dest.parent)


def test_transfer_failure_cleans_part_and_preserves_destination(tmp_path):
    missing_source = tmp_path / "missing.bin"
    dest = tmp_path / "model.tflite"
    dest.write_bytes(b"known good")
    artifact = artifact_for(missing_source, dest, b"expected replacement")

    with pytest.raises(ArtifactError) as exc_info:
        download_artifact(artifact)

    assert not isinstance(exc_info.value, ChecksumError)
    assert exc_info.value.__cause__ is not None
    assert dest.read_bytes() == b"known good"
    assert_no_part_files(dest.parent)


def test_matching_existing_destination_skips_source_access(tmp_path):
    payload = b"already verified"
    missing_source = tmp_path / "missing.bin"
    dest = tmp_path / "model.tflite"
    dest.write_bytes(payload)

    assert download_artifact(artifact_for(missing_source, dest, payload)) is False

    assert dest.read_bytes() == payload
    assert_no_part_files(dest.parent)


def test_existing_destination_checks_are_serialized_with_installation(
    tmp_path, monkeypatch
):
    payload = b"verified replacement"
    source = tmp_path / "source.bin"
    source.write_bytes(payload)
    dest = tmp_path / "model.tflite"
    dest.write_bytes(b"stale destination")

    class TrackingLock:
        def __init__(self):
            self.held = False

        def __enter__(self):
            assert not self.held
            self.held = True

        def __exit__(self, exc_type, exc, traceback):
            self.held = False

    install_lock = TrackingLock()
    destination_checks = []
    real_sha256_file = artifacts_module.sha256_file

    def checked_sha256_file(path):
        if Path(path) == dest:
            destination_checks.append(install_lock.held)
        return real_sha256_file(path)

    monkeypatch.setattr(artifacts_module, "_INSTALL_LOCK", install_lock)
    monkeypatch.setattr(artifacts_module, "sha256_file", checked_sha256_file)

    assert download_artifact(artifact_for(source, dest, payload)) is True

    assert destination_checks == [True, True]
    assert dest.read_bytes() == payload
    assert_no_part_files(tmp_path)


def test_download_passes_a_finite_positive_timeout_to_urlopen(monkeypatch, tmp_path):
    payload = b"bounded connection"
    dest = tmp_path / "model.tflite"
    observed_timeouts = []

    def fake_urlopen(url, timeout=None):
        del url
        observed_timeouts.append(timeout)
        return ChunkedResponse([payload], content_length=len(payload))

    monkeypatch.setattr(artifacts_module, "urlopen", fake_urlopen)

    artifact = Artifact(
        "https://example.invalid/model.tflite",
        dest,
        hashlib.sha256(payload).hexdigest(),
    )
    assert download_artifact(artifact) is True

    assert len(observed_timeouts) == 1
    assert observed_timeouts[0] is not None
    assert math.isfinite(observed_timeouts[0])
    assert observed_timeouts[0] > 0
    assert_no_part_files(tmp_path)


def test_declared_oversize_is_rejected_before_streaming_and_preserves_destination(
    monkeypatch, tmp_path
):
    dest = tmp_path / "model.tflite"
    dest.write_bytes(b"known good")
    response = ChunkedResponse([], content_length=128 * 1024 * 1024 + 1)
    monkeypatch.setattr(artifacts_module, "urlopen", lambda *args, **kwargs: response)
    artifact = Artifact("https://example.invalid/large", dest, "0" * 64)

    with pytest.raises(ArtifactError) as exc_info:
        download_artifact(artifact)

    assert not isinstance(exc_info.value, ChecksumError)
    assert exc_info.value.__cause__ is not None
    assert "maximum" in str(exc_info.value).lower()
    assert response.read_calls == 0
    assert dest.read_bytes() == b"known good"
    assert_no_part_files(tmp_path)


def test_stream_overflow_is_rejected_and_preserves_destination(monkeypatch, tmp_path):
    dest = tmp_path / "model.tflite"
    dest.write_bytes(b"known good")
    response = ChunkedResponse([b"1234", b"56"])
    monkeypatch.setattr(artifacts_module, "MAX_ARTIFACT_BYTES", 5, raising=False)
    monkeypatch.setattr(artifacts_module, "urlopen", lambda *args, **kwargs: response)
    artifact = Artifact("https://example.invalid/stream", dest, "0" * 64)

    with pytest.raises(ArtifactError) as exc_info:
        download_artifact(artifact)

    assert not isinstance(exc_info.value, ChecksumError)
    assert exc_info.value.__cause__ is not None
    assert "maximum" in str(exc_info.value).lower()
    assert dest.read_bytes() == b"known good"
    assert_no_part_files(tmp_path)


def test_cleanup_failure_does_not_mask_primary_transfer_error(monkeypatch, tmp_path):
    dest = tmp_path / "model.tflite"
    dest.write_bytes(b"known good")

    class FailingResponse(ChunkedResponse):
        def read(self, size):
            del size
            raise RuntimeError("primary transfer failure")

    monkeypatch.setattr(
        artifacts_module,
        "urlopen",
        lambda *args, **kwargs: FailingResponse([]),
    )
    original_unlink = Path.unlink

    def fail_part_cleanup(path, *args, **kwargs):
        if path.suffix == ".part":
            raise PermissionError("cleanup failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_part_cleanup)
    artifact = Artifact("https://example.invalid/fails", dest, "0" * 64)

    with pytest.raises(ArtifactError) as exc_info:
        download_artifact(artifact)

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "primary transfer failure"
    assert dest.read_bytes() == b"known good"


def test_concurrent_downloads_use_distinct_part_files(monkeypatch, tmp_path):
    payload = b"same verified artifact"
    dest = tmp_path / "model.tflite"
    observed_parts = []
    replace_entries = []
    replace_entry_lock = threading.Lock()
    first_replace_started = threading.Event()
    second_replace_started = threading.Event()
    release_first_replace = threading.Event()

    def capture_parts():
        observed_parts.extend(tmp_path.glob("*.part"))

    first_reads = threading.Barrier(2, action=capture_parts)

    def fake_urlopen(url, timeout=None):
        del url, timeout
        return ChunkedResponse(
            [payload],
            content_length=len(payload),
            before_first_read=first_reads.wait,
        )

    monkeypatch.setattr(artifacts_module, "urlopen", fake_urlopen)
    real_replace = artifacts_module.os.replace

    def coordinated_replace(source, destination):
        with replace_entry_lock:
            replace_entries.append((source, destination))
            entry_number = len(replace_entries)
        if entry_number == 1:
            first_replace_started.set()
            assert release_first_replace.wait(timeout=1)
        else:
            second_replace_started.set()
        return real_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", coordinated_replace)
    artifact = Artifact(
        "https://example.invalid/model.tflite",
        dest,
        hashlib.sha256(payload).hexdigest(),
    )

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(download_artifact, artifact)
                for _ in range(2)
            ]
            assert first_replace_started.wait(timeout=1)
            replacements_overlapped = second_replace_started.wait(timeout=0.2)
            release_first_replace.set()
            results = [future.result(timeout=2) for future in futures]
    finally:
        release_first_replace.set()

    assert results == [True, True]
    assert not replacements_overlapped
    assert len(replace_entries) == 1
    assert len(observed_parts) == 2
    assert len(set(observed_parts)) == 2
    assert all(path.name.endswith(".part") for path in observed_parts)
    assert dest.read_bytes() == payload
    assert_no_part_files(tmp_path)


def test_config_uses_current_models_directory_by_default(monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.delenv("EARSHOT_MODEL_DIR", raising=False)
        reloaded = importlib.reload(config)
        expected = Path(reloaded.__file__).resolve().parent.parent / "models"

        assert reloaded.MODEL_DIR == expected
        assert reloaded.MODEL_PATH == expected / "yamnet.tflite"
        assert reloaded.CLASS_MAP_PATH == expected / "yamnet_class_map.csv"
        assert reloaded.TAUGHT_STORE_PATH == expected / "taught_sounds.npz"

    importlib.reload(config)


def test_config_model_directory_override_updates_all_paths(monkeypatch, tmp_path):
    override = tmp_path / "persistent models"

    with monkeypatch.context() as scoped:
        scoped.setenv("EARSHOT_MODEL_DIR", str(override))
        reloaded = importlib.reload(config)

        assert reloaded.MODEL_DIR == override
        assert reloaded.MODEL_PATH == override / "yamnet.tflite"
        assert reloaded.CLASS_MAP_PATH == override / "yamnet_class_map.csv"
        assert reloaded.TAUGHT_STORE_PATH == override / "taught_sounds.npz"
        assert reloaded.MODEL_ARTIFACT.path == reloaded.MODEL_PATH
        assert reloaded.CLASS_MAP_ARTIFACT.path == reloaded.CLASS_MAP_PATH

    importlib.reload(config)


def test_config_defines_alarm_paths_under_expected_roots(monkeypatch, tmp_path):
    with monkeypatch.context() as scoped:
        scoped.setenv("EARSHOT_MODEL_DIR", str(tmp_path / "models"))
        scoped.setenv("EARSHOT_ALARM_MODEL_PATH", str(tmp_path / "head.npz"))
        reloaded = importlib.reload(config)

        assert reloaded.ALARM_MODEL_PATH == tmp_path / "head.npz"
        assert (
            reloaded.ALARM_REPORT_PATH
            == tmp_path / "models" / "fire_smoke_alarm_report.json"
        )
        assert reloaded.ALARM_DATA_DIR.name == "alarm_demo"

    importlib.reload(config)


def test_config_defines_trained_alarm_runtime_contract():
    assert config.ALARM_EVENT_LABEL == "fire_smoke_alarm"
    assert config.ALARM_EVENT_URGENCY == "high"
    assert config.ALARM_REPLACED_LABELS == frozenset(
        {"fire_alarm", "smoke_alarm"}
    )
    assert config.ALARM_GATE_COUNT == 2
    assert config.ALARM_GATE_WINDOW == 8
    assert config.RESERVED_EVENT_LABELS == frozenset(
        {
            *(entry["label"].strip().casefold() for entry in config.EVENT_MAP),
            config.ALARM_EVENT_LABEL.strip().casefold(),
        }
    )


def test_config_defines_verified_official_artifacts():
    assert config.MODEL_ARTIFACT == Artifact(
        MODEL_URL,
        config.MODEL_PATH,
        MODEL_SHA256,
    )
    assert config.CLASS_MAP_ARTIFACT == Artifact(
        CLASS_MAP_URL,
        config.CLASS_MAP_PATH,
        CLASS_MAP_SHA256,
    )
