import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from earshot_ml import config
from earshot_ml.alarm_data import NOT_ALARM
from earshot_ml.alarm_model import load_alarm_head
from earshot_ml.alarm_training import evaluate_alarm, train_alarm


pytestmark = pytest.mark.integration

_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_NEGATIVE_GROUP_FRACTION = 0.20
_MAX_FALSE_TRIGGERS_PER_MINUTE = 0.5


def _streamed_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _wav_snapshot(root: Path) -> dict[str, str]:
    paths = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() == ".wav"
    ]
    paths.sort(
        key=lambda path: (
            path.relative_to(root).as_posix().casefold(),
            path.relative_to(root).as_posix(),
        )
    )
    return {
        path.relative_to(root).as_posix(): _streamed_sha256(path)
        for path in paths
    }


def _optional_digest(path: Path) -> str | None:
    return _streamed_sha256(path) if path.is_file() else None


def _assert_corpus_window_audit(payload: dict) -> dict[str, int]:
    corpus = payload["corpus"]
    recordings = corpus["recordings"]
    expected_windows = {}
    for item in recordings:
        path = item["path"]
        sample_count = item["sample_count"]
        assert path not in expected_windows
        assert type(sample_count) is int
        assert sample_count >= config.WINDOW_SAMPLES
        expected_windows[path] = 1 + (
            sample_count - config.WINDOW_SAMPLES
        ) // config.HOP_SAMPLES

    paths = set(expected_windows)
    retained_windows = [item["retained_windows"] for item in recordings]

    assert len(paths) == len(recordings)
    assert retained_windows
    assert all(type(value) is int and value > 0 for value in retained_windows)
    assert corpus["counts"]["recordings"] == len(recordings)
    assert corpus["counts"]["retained_windows"] == sum(retained_windows)
    assert {item["path"] for item in payload["content_audit"]} == paths
    return expected_windows


def _assert_evaluated_window_audit(
    metrics: dict,
    expected_windows: dict[str, int],
) -> None:
    files = metrics["files"]
    files_by_path = {}
    for item in files:
        path = item["path"]
        assert path not in files_by_path
        files_by_path[path] = item

    assert set(files_by_path) == set(expected_windows)
    assert len(files) == len(expected_windows)
    for path, expected_count in expected_windows.items():
        assert files_by_path[path]["evaluated_windows"] == expected_count
    assert metrics["evaluated_windows"] == sum(expected_windows.values())


def _assert_serialized_metrics(payload: dict, metrics, scope: str) -> None:
    assert payload["evaluation_scope"] == scope
    assert payload["files"] == [dict(item) for item in metrics.files]
    for field in (
        "positive_groups_total",
        "positive_groups_triggered",
        "negative_groups_total",
        "negative_groups_triggered",
        "false_events",
        "evaluated_windows",
    ):
        assert payload[field] == getattr(metrics, field)
    assert payload["negative_audio_minutes"] == pytest.approx(
        metrics.negative_audio_minutes
    )
    assert payload["false_triggers_per_minute"] == pytest.approx(
        metrics.false_triggers_per_minute
    )


def _assert_acceptance(scope: str, metrics) -> None:
    for item in metrics.files:
        if item["label"] == NOT_ALARM and item["triggered"]:
            print(
                f"{scope} triggered negative: "
                f"path={item['path']} events={item['events']}"
            )

    print(
        f"{scope} summary: "
        f"positive_groups={metrics.positive_groups_triggered}/"
        f"{metrics.positive_groups_total} "
        f"negative_groups={metrics.negative_groups_triggered}/"
        f"{metrics.negative_groups_total} "
        f"false_events={metrics.false_events} "
        f"false_triggers_per_minute={metrics.false_triggers_per_minute:.6f} "
        f"evaluated_windows={metrics.evaluated_windows}"
    )

    assert metrics.positive_groups_total >= 5
    assert metrics.negative_groups_total >= 5
    assert metrics.positive_groups_triggered == metrics.positive_groups_total
    assert (
        metrics.negative_groups_triggered / metrics.negative_groups_total
        <= _MAX_NEGATIVE_GROUP_FRACTION
    )
    assert metrics.false_triggers_per_minute <= _MAX_FALSE_TRIGGERS_PER_MINUTE


def test_local_alarm_corpus_meets_demo_acceptance_criteria(tmp_path):
    data_dir = Path(config.ALARM_DATA_DIR)
    model_path = Path(config.MODEL_PATH)
    class_map_path = Path(config.CLASS_MAP_PATH)

    required_artifacts = {
        "pinned YAMNet model": model_path,
        "pinned YAMNet class map": class_map_path,
    }
    missing_artifacts = [
        f"{description} ({path})"
        for description, path in required_artifacts.items()
        if not path.is_file()
    ]
    if missing_artifacts:
        pytest.skip(
            "alarm corpus integration requires downloaded artifacts; missing: "
            + ", ".join(missing_artifacts)
            + "; run `earshot download` first"
        )

    required_corpus_dirs = (data_dir / "alarm", data_dir / "not_alarm")
    missing_corpus_dirs = [str(path) for path in required_corpus_dirs if not path.is_dir()]
    if missing_corpus_dirs:
        pytest.skip(
            "alarm corpus integration requires local alarm and not_alarm "
            "directories; missing: "
            + ", ".join(missing_corpus_dirs)
        )

    artifact_path = tmp_path / "fire_smoke_alarm_head.npz"
    report_path = tmp_path / "fire_smoke_alarm_report.json"
    manifest_path = data_dir / "manifest.json"

    wav_hashes_before = _wav_snapshot(data_dir)
    manifest_hash_before = _optional_digest(manifest_path)

    try:
        assert wav_hashes_before, f"no WAV files found under {data_dir}"
        assert _streamed_sha256(model_path) == config.MODEL_ARTIFACT.sha256, (
            "YAMNet model does not match its pinned SHA-256"
        )
        assert _streamed_sha256(class_map_path) == config.CLASS_MAP_ARTIFACT.sha256, (
            "YAMNet class map does not match its pinned SHA-256"
        )

        trained = train_alarm(
            data_dir,
            artifact_path,
            report_path,
            seed=0,
            yamnet_model_path=model_path,
            class_map_path=class_map_path,
        )

        assert trained.artifact_path == artifact_path
        assert trained.report_path == report_path
        assert artifact_path.is_file()
        assert report_path.is_file()
        assert trained.payload["status"] == "ok"

        on_disk_report = json.loads(report_path.read_text(encoding="utf-8"))
        assert on_disk_report == trained.payload
        assert on_disk_report["status"] == "ok"

        training_windows = _assert_corpus_window_audit(on_disk_report)
        for name, scope, metrics in (
            ("out_of_fold", "out_of_fold", trained.oof_metrics),
            ("final_model", "in_sample", trained.in_sample_metrics),
        ):
            serialized = on_disk_report["metrics"][name]
            _assert_evaluated_window_audit(
                serialized,
                training_windows,
            )
            _assert_serialized_metrics(serialized, metrics, scope)

        head = load_alarm_head(
            artifact_path,
            yamnet_model_path=model_path,
            class_map_path=class_map_path,
        )
        assert head.label == config.ALARM_EVENT_LABEL
        assert head.urgency == config.ALARM_EVENT_URGENCY
        assert head.threshold == pytest.approx(trained.deployment_threshold)
        assert head.threshold == pytest.approx(
            on_disk_report["thresholds"]["deployment"]
        )
        assert math.isfinite(
            head.score(np.zeros(head.feature_dim, dtype=np.float32))
        )

        evaluated = evaluate_alarm(
            data_dir,
            artifact_path,
            yamnet_model_path=model_path,
            class_map_path=class_map_path,
        )
        assert evaluated.evaluation_scope == "in_sample"
        assert evaluated.payload["evaluation_scope"] == "in_sample"
        assert evaluated.payload["metrics"]["evaluation_scope"] == "in_sample"

        evaluation_windows = _assert_corpus_window_audit(evaluated.payload)
        assert evaluation_windows == training_windows
        _assert_evaluated_window_audit(
            evaluated.payload["metrics"],
            evaluation_windows,
        )
        _assert_serialized_metrics(
            evaluated.payload["metrics"],
            evaluated.metrics,
            "in_sample",
        )

        _assert_acceptance("out_of_fold", trained.oof_metrics)
        _assert_acceptance("final_model", trained.in_sample_metrics)
        _assert_acceptance("evaluated_in_sample", evaluated.metrics)
    finally:
        assert _wav_snapshot(data_dir) == wav_hashes_before, (
            "alarm corpus WAV files changed during training or evaluation"
        )
        assert _optional_digest(manifest_path) == manifest_hash_before, (
            "alarm corpus manifest changed during training or evaluation"
        )
