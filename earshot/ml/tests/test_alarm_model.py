from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from earshot_ml import alarm_model
from earshot_ml.alarm_model import (
    ARTIFACT_KEYS,
    FEATURE_DIM,
    SCHEMA,
    SCHEMA_VERSION,
    AlarmHead,
    AlarmModelError,
    RollingEvidenceGate,
    load_alarm_head,
    load_optional_alarm_head,
    save_alarm_head,
)
from earshot_ml.artifacts import sha256_file


def make_head(model_digest="a" * 64, map_digest="b" * 64, **overrides):
    weights = np.zeros(FEATURE_DIM, dtype=np.float32)
    weights[0] = 2.0
    values = {
        "label": "smoke_alarm",
        "urgency": "high",
        "feature_dim": FEATURE_DIM,
        "mean": np.zeros(FEATURE_DIM, dtype=np.float32),
        "scale": np.ones(FEATURE_DIM, dtype=np.float32),
        "weights": weights,
        "bias": -1.0,
        "threshold": 0.7,
        "gate_count": 2,
        "gate_window": 8,
        "yamnet_model_sha256": model_digest,
        "class_map_sha256": map_digest,
    }
    values.update(overrides)
    return AlarmHead(**values)


def artifact_payload(head=None):
    head = head or make_head()
    return {
        "schema": np.array(SCHEMA, dtype=np.str_),
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int64),
        "label": np.array(head.label, dtype=np.str_),
        "urgency": np.array(head.urgency, dtype=np.str_),
        "feature_dim": np.array(head.feature_dim, dtype=np.int64),
        "mean": np.asarray(head.mean),
        "scale": np.asarray(head.scale),
        "weights": np.asarray(head.weights),
        "bias": np.array(head.bias, dtype=np.float64),
        "threshold": np.array(head.threshold, dtype=np.float64),
        "gate_count": np.array(head.gate_count, dtype=np.int64),
        "gate_window": np.array(head.gate_window, dtype=np.int64),
        "yamnet_model_sha256": np.array(
            head.yamnet_model_sha256, dtype=np.str_
        ),
        "class_map_sha256": np.array(head.class_map_sha256, dtype=np.str_),
    }


def write_artifact(path: Path, payload):
    with path.open("wb") as output:
        np.savez(output, **payload)


def compatible_files(tmp_path):
    model = tmp_path / "yamnet.tflite"
    class_map = tmp_path / "map.csv"
    model.write_bytes(b"model")
    class_map.write_bytes(b"map")
    head = make_head(sha256_file(model), sha256_file(class_map))
    return model, class_map, head


def load_with_files(path, model, class_map):
    return load_alarm_head(
        path,
        yamnet_model_path=model,
        class_map_path=class_map,
    )


def test_head_uses_branch_stable_sigmoid():
    head = make_head()
    positive = np.zeros(FEATURE_DIM, dtype=np.float32)
    positive[0] = 1_000
    negative = -positive

    assert head.score(positive) == pytest.approx(1.0)
    assert head.score(negative) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "embedding",
    [
        np.zeros(FEATURE_DIM - 1, dtype=np.float32),
        np.concatenate(
            [np.zeros(FEATURE_DIM - 1, dtype=np.float32), [np.nan]]
        ),
        ["not-a-number"] * FEATURE_DIM,
    ],
    ids=["wrong-size", "non-finite", "non-numeric"],
)
def test_head_rejects_invalid_embeddings(embedding):
    with pytest.raises(AlarmModelError, match="embedding"):
        make_head().score(embedding)


def test_gate_accepts_nonconsecutive_two_of_eight_and_expires_old_evidence():
    gate = RollingEvidenceGate(required_count=2, window_size=8)

    assert gate.update(True) is False
    for _ in range(6):
        assert gate.update(False) is False
    assert gate.update(True) is True
    assert gate.update(False) is False


def test_gate_reset_discards_all_prior_evidence():
    gate = RollingEvidenceGate(required_count=2, window_size=8)
    assert gate.update(True) is False
    gate.reset()

    assert gate.update(True) is False
    assert gate.update(True) is True


@pytest.mark.parametrize(
    ("required_count", "window_size"),
    [
        (0, 8),
        (-1, 8),
        (2, 0),
        (3, 2),
        (True, 8),
        (2, False),
        (2.5, 8),
        (2, 8.5),
        ("2", 8),
    ],
)
def test_gate_rejects_invalid_settings(required_count, window_size):
    with pytest.raises(ValueError, match="gate"):
        RollingEvidenceGate(required_count, window_size)


def test_artifact_uses_exact_keys_shapes_and_dtypes(tmp_path):
    path = tmp_path / "models" / "head.npz"
    save_alarm_head(path, make_head())

    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == ARTIFACT_KEYS
        for key in (
            "schema",
            "label",
            "urgency",
            "yamnet_model_sha256",
            "class_map_sha256",
        ):
            assert archive[key].shape == ()
            assert archive[key].dtype.kind == "U"
        for key in (
            "schema_version",
            "feature_dim",
            "gate_count",
            "gate_window",
        ):
            assert archive[key].shape == ()
            assert archive[key].dtype == np.dtype(np.int64)
        for key in ("bias", "threshold"):
            assert archive[key].shape == ()
            assert archive[key].dtype == np.dtype(np.float64)
        for key in ("mean", "scale", "weights"):
            assert archive[key].shape == (FEATURE_DIM,)
            assert archive[key].dtype == np.dtype(np.float32)

        assert archive["schema"].item() == SCHEMA
        assert archive["schema_version"].item() == SCHEMA_VERSION
        assert archive["label"].item() == "smoke_alarm"


def test_artifact_round_trip_digest_validation_and_immutable_vectors(tmp_path):
    model, class_map, head = compatible_files(tmp_path)
    path = tmp_path / "head.npz"

    save_alarm_head(path, head)
    loaded = load_with_files(path, model, class_map)

    assert loaded.label == "smoke_alarm"
    assert loaded.urgency == "high"
    np.testing.assert_array_equal(loaded.mean, head.mean)
    np.testing.assert_array_equal(loaded.scale, head.scale)
    np.testing.assert_array_equal(loaded.weights, head.weights)
    for vector in (loaded.mean, loaded.scale, loaded.weights):
        assert vector.flags.writeable is False
        with pytest.raises(ValueError):
            vector[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        loaded.bias = 0.0


def test_loader_canonicalizes_legacy_alarm_label(tmp_path):
    model, class_map, head = compatible_files(tmp_path)
    payload = artifact_payload(head)
    payload["label"] = np.array("fire_smoke_alarm", dtype=np.str_)
    path = tmp_path / "legacy-head.npz"
    write_artifact(path, payload)

    loaded = load_with_files(path, model, class_map)

    assert loaded.label == "smoke_alarm"


@pytest.mark.parametrize("missing_key", sorted(ARTIFACT_KEYS))
def test_loader_rejects_every_missing_key(tmp_path, missing_key):
    model, class_map, head = compatible_files(tmp_path)
    payload = artifact_payload(head)
    del payload[missing_key]
    path = tmp_path / "head.npz"
    write_artifact(path, payload)

    with pytest.raises(AlarmModelError, match="keys"):
        load_with_files(path, model, class_map)


def test_loader_rejects_extra_keys(tmp_path):
    model, class_map, head = compatible_files(tmp_path)
    payload = artifact_payload(head)
    payload["unexpected"] = np.array(1, dtype=np.int64)
    path = tmp_path / "head.npz"
    write_artifact(path, payload)

    with pytest.raises(AlarmModelError, match="keys"):
        load_with_files(path, model, class_map)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", np.array("wrong.schema", dtype=np.str_)),
        ("schema_version", np.array(2, dtype=np.int64)),
        ("label", np.array("other_alarm", dtype=np.str_)),
        ("urgency", np.array("medium", dtype=np.str_)),
        ("feature_dim", np.array(512, dtype=np.int64)),
        ("threshold", np.array(0.0, dtype=np.float64)),
        ("threshold", np.array(1.0, dtype=np.float64)),
        ("threshold", np.array(np.nan, dtype=np.float64)),
        ("gate_count", np.array(1, dtype=np.int64)),
        ("gate_window", np.array(7, dtype=np.int64)),
        ("yamnet_model_sha256", np.array("short", dtype=np.str_)),
        ("class_map_sha256", np.array("g" * 64, dtype=np.str_)),
    ],
    ids=[
        "schema",
        "schema-version",
        "label",
        "urgency",
        "feature-dim",
        "threshold-zero",
        "threshold-one",
        "threshold-nan",
        "gate-count",
        "gate-window",
        "model-digest",
        "map-digest",
    ],
)
def test_loader_rejects_incompatible_scalar_contracts(tmp_path, field, value):
    model, class_map, head = compatible_files(tmp_path)
    payload = artifact_payload(head)
    payload[field] = value
    path = tmp_path / "head.npz"
    write_artifact(path, payload)

    with pytest.raises(AlarmModelError):
        load_with_files(path, model, class_map)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mean", np.zeros(FEATURE_DIM, dtype=np.float64)),
        ("mean", np.zeros((FEATURE_DIM, 1), dtype=np.float32)),
        ("mean", np.full(FEATURE_DIM, np.nan, dtype=np.float32)),
        ("scale", np.zeros(FEATURE_DIM, dtype=np.float32)),
        ("scale", -np.ones(FEATURE_DIM, dtype=np.float32)),
        ("scale", np.full(FEATURE_DIM, np.inf, dtype=np.float32)),
        ("weights", np.full(FEATURE_DIM, np.nan, dtype=np.float32)),
        ("weights", np.full(FEATURE_DIM, "1.0", dtype=np.str_)),
    ],
    ids=[
        "wrong-dtype",
        "wrong-shape",
        "nonfinite-mean",
        "zero-scale",
        "negative-scale",
        "nonfinite-scale",
        "nonfinite-weights",
        "string-vector",
    ],
)
def test_loader_rejects_invalid_vectors(tmp_path, field, value):
    model, class_map, head = compatible_files(tmp_path)
    payload = artifact_payload(head)
    payload[field] = value
    path = tmp_path / "head.npz"
    write_artifact(path, payload)

    with pytest.raises(AlarmModelError):
        load_with_files(path, model, class_map)


def test_loader_rejects_object_arrays_without_pickle(tmp_path):
    model, class_map, head = compatible_files(tmp_path)
    payload = artifact_payload(head)
    payload["weights"] = np.array([object()] * FEATURE_DIM, dtype=object)
    path = tmp_path / "head.npz"
    write_artifact(path, payload)

    with pytest.raises(AlarmModelError):
        load_with_files(path, model, class_map)


@pytest.mark.parametrize("changed_file", ["model", "class-map"])
def test_loader_rejects_digest_mismatch(tmp_path, changed_file):
    model, class_map, head = compatible_files(tmp_path)
    path = tmp_path / "head.npz"
    save_alarm_head(path, head)
    target = model if changed_file == "model" else class_map
    target.write_bytes(b"changed")

    with pytest.raises(AlarmModelError, match="digest"):
        load_with_files(path, model, class_map)


def test_loader_wraps_missing_compatibility_files(tmp_path):
    path = tmp_path / "head.npz"
    save_alarm_head(path, make_head())

    with pytest.raises(AlarmModelError):
        load_with_files(path, tmp_path / "missing-model", tmp_path / "missing-map")


def test_optional_loader_only_ignores_none_or_absence(tmp_path):
    missing_model = tmp_path / "model"
    missing_map = tmp_path / "map"
    assert load_optional_alarm_head(
        None,
        yamnet_model_path=missing_model,
        class_map_path=missing_map,
    ) is None
    assert load_optional_alarm_head(
        tmp_path / "missing.npz",
        yamnet_model_path=missing_model,
        class_map_path=missing_map,
    ) is None

    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"bad")
    with pytest.raises(AlarmModelError):
        load_optional_alarm_head(
            corrupt,
            yamnet_model_path=missing_model,
            class_map_path=missing_map,
        )


def test_save_flushes_and_fsyncs_before_install(tmp_path, monkeypatch):
    path = tmp_path / "nested" / "head.npz"
    fsynced = []
    real_fsync = alarm_model.os.fsync

    def tracking_fsync(file_descriptor):
        fsynced.append(file_descriptor)
        return real_fsync(file_descriptor)

    monkeypatch.setattr(alarm_model.os, "fsync", tracking_fsync)
    save_alarm_head(path, make_head())

    assert len(fsynced) == 1
    assert path.is_file()


def test_save_preserves_predictable_part_sentinel(tmp_path):
    path = tmp_path / "head.npz"
    sentinel = path.with_name(path.name + ".part")
    sentinel.write_bytes(b"do not touch")

    save_alarm_head(path, make_head())

    assert sentinel.read_bytes() == b"do not touch"
    assert list(tmp_path.glob(path.name + ".*.part")) == []


def test_save_validation_preserves_previous_artifact_without_temp(tmp_path):
    path = tmp_path / "head.npz"
    path.write_bytes(b"previous artifact")
    invalid_scale = np.ones(FEATURE_DIM, dtype=np.float32)
    invalid_scale[0] = 0.0

    with pytest.raises(AlarmModelError):
        save_alarm_head(path, make_head(scale=invalid_scale))

    assert path.read_bytes() == b"previous artifact"
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("label", "fire_smoke_alarm"),
        (
            "label",
            np.array(["fire_smoke_alarm"], dtype=np.str_),
        ),
        (
            "label",
            np.array("fire_smoke_alarm", dtype=np.str_),
        ),
        ("label", ["fire_smoke_alarm"]),
        ("label", 7),
        (
            "urgency",
            np.array(["high"], dtype=np.str_),
        ),
        (
            "urgency",
            np.array("high", dtype=np.str_),
        ),
        ("urgency", ["high"]),
        ("urgency", 7),
    ],
    ids=[
        "legacy-label",
        "label-vector",
        "label-zero-dimensional-array",
        "label-list",
        "label-integer",
        "urgency-vector",
        "urgency-zero-dimensional-array",
        "urgency-list",
        "urgency-integer",
    ],
)
def test_save_rejects_malformed_string_fields_before_touching_files(
    tmp_path,
    field,
    invalid_value,
):
    path = tmp_path / "head.npz"
    path.write_bytes(b"previous artifact")
    sentinel = path.with_name(path.name + ".part")
    sentinel.write_bytes(b"existing sentinel")

    with pytest.raises(AlarmModelError, match=field):
        save_alarm_head(path, make_head(**{field: invalid_value}))

    assert path.read_bytes() == b"previous artifact"
    assert sentinel.read_bytes() == b"existing sentinel"
    assert list(tmp_path.glob(path.name + ".*.part")) == []


def test_replace_failure_rolls_back_and_cleans_only_unique_temp(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "head.npz"
    path.write_bytes(b"previous artifact")
    sentinel = path.with_name(path.name + ".part")
    sentinel.write_bytes(b"sentinel")

    def fail_replace(source, destination):
        assert Path(source) != sentinel
        assert Path(destination) == path
        raise OSError("replace failed")

    monkeypatch.setattr(alarm_model.os, "replace", fail_replace)

    with pytest.raises(AlarmModelError, match="save"):
        save_alarm_head(path, make_head())

    assert path.read_bytes() == b"previous artifact"
    assert sentinel.read_bytes() == b"sentinel"
    assert list(tmp_path.glob(path.name + ".*.part")) == []


def test_serialization_failure_preserves_previous_artifact_and_cleans_temp(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "head.npz"
    path.write_bytes(b"previous artifact")

    def fail_savez(*args, **kwargs):
        del args, kwargs
        raise OSError("serialization failed")

    monkeypatch.setattr(alarm_model.np, "savez", fail_savez)

    with pytest.raises(AlarmModelError, match="save"):
        save_alarm_head(path, make_head())

    assert path.read_bytes() == b"previous artifact"
    assert list(tmp_path.glob("*.part")) == []
