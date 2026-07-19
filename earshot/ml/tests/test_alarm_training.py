import hashlib
import json
import os
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from earshot_ml import alarm_training
from earshot_ml import config
from earshot_ml.alarm_data import CorpusEntry, CorpusInventory
from earshot_ml.alarm_model import AlarmModelError, load_alarm_head
from earshot_ml.alarm_training import (
    PreparedWindow,
    TrainingError,
    WeightedWindow,
    _augment_training_windows,
    _evenly_spaced,
    _prepare_recordings,
    _weighted_standardization,
    mix_at_snr,
    rms,
)


def corpus_entry(
    tmp_path,
    *,
    label="alarm",
    segments=(),
    group="group-a",
    filename="clip.wav",
):
    path = tmp_path / label / filename
    return CorpusEntry(
        path=path,
        relative_path=f"{label}/{filename}",
        label=label,
        source_group=group,
        segments=tuple(segments),
        decoded_sha256="a" * 64,
        duration_seconds=2.0,
    )


def inventory(*entries):
    return CorpusInventory(
        root=entries[0].path.parents[1],
        entries=tuple(entries),
        warnings=(),
    )


def decoded_digest(audio):
    canonical = np.asarray(audio, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def corpus_entry_for_audio(
    tmp_path,
    audio,
    *,
    label="not_alarm",
    segments=(),
    group="group-a",
    filename="clip.wav",
):
    return CorpusEntry(
        path=tmp_path / label / filename,
        relative_path=f"{label}/{filename}",
        label=label,
        source_group=group,
        segments=tuple(segments),
        decoded_sha256=decoded_digest(audio),
        duration_seconds=len(audio) / config.SAMPLE_RATE,
    )


class FirstSampleYamNet:
    @staticmethod
    def infer(waveform):
        embedding = np.zeros(1024, dtype=np.float32)
        embedding[0] = waveform[0]
        return np.zeros(521, dtype=np.float32), embedding


class FirstValueHead:
    @staticmethod
    def score(embedding):
        return float(embedding[0])


def weighted_window(value, group, weight, *, label=1, start_sample=0):
    return WeightedWindow(
        waveform=np.full(config.WINDOW_SAMPLES, value, np.float32),
        label=label,
        source_group=group,
        source_path=f"{group}.wav",
        start_sample=start_sample,
        weight=weight,
        augmentation_noise_group=None,
    )


def scored_corpus(*, positive, negative):
    result = []
    for label, groups in ((1, positive), (0, negative)):
        for group, scores in groups.items():
            result.append(
                alarm_training.ScoredRecording(
                    source_group=group,
                    source_path=f"{group}.wav",
                    label=label,
                    duration_seconds=max(1.0, len(scores) * 0.5),
                    scores=tuple(float(score) for score in scores),
                )
            )
    return tuple(result)


def prepared_recording(tmp_path, group, label):
    text_label = "alarm" if label else "not_alarm"
    entry = corpus_entry(tmp_path, label=text_label, group=group)
    window = PreparedWindow(
        waveform=np.full(config.WINDOW_SAMPLES, float(label), np.float32),
        start_sample=0,
        weight=1.0,
    )
    return alarm_training.PreparedRecording(
        entry=entry,
        windows=(window,),
        duration_seconds=1.0,
    )


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
    folds = alarm_training._make_grouped_folds(
        synthetic_recordings,
        seed=0,
        folds=5,
    )
    for fold in folds:
        assert set(fold.train_groups).isdisjoint(fold.validation_groups)
        assert {
            1 if item.entry.label == "alarm" else 0
            for item in fold.train_recordings
        } == {0, 1}


def test_grouped_folds_are_deterministic(synthetic_recordings):
    first = alarm_training._make_grouped_folds(
        synthetic_recordings, seed=11, folds=5,
    )
    second = alarm_training._make_grouped_folds(
        synthetic_recordings, seed=11, folds=5,
    )
    assert [fold.validation_groups for fold in first] == [
        fold.validation_groups for fold in second
    ]


def test_grouped_folds_reject_mixed_labels_within_source_group(tmp_path):
    recordings = (
        prepared_recording(tmp_path, "shared", 1),
        prepared_recording(tmp_path, "shared", 0),
    )

    with pytest.raises(TrainingError, match="mixed corpus labels"):
        alarm_training._make_grouped_folds(recordings, seed=0, folds=2)


def test_grouped_folds_require_five_source_groups_per_label(tmp_path):
    recordings = tuple(
        prepared_recording(tmp_path, f"positive-{index}", 1)
        for index in range(4)
    ) + tuple(
        prepared_recording(tmp_path, f"negative-{index}", 0)
        for index in range(5)
    )

    with pytest.raises(TrainingError, match="at least 5 source groups"):
        alarm_training._make_grouped_folds(recordings, seed=0, folds=5)


def test_grouped_folds_reject_splitter_group_overlap(
        synthetic_recordings, tmp_path, monkeypatch):
    duplicate = prepared_recording(tmp_path, "p-0", 1)
    recordings = synthetic_recordings + (duplicate,)

    class OverlappingSplitter:
        def __init__(self, **_kwargs):
            pass

        def split(self, _features, _labels, _groups):
            yield np.arange(10), np.array([10])

    monkeypatch.setattr(
        alarm_training,
        "_load_sklearn",
        lambda: (object, OverlappingSplitter),
    )

    with pytest.raises(TrainingError, match="leaked"):
        alarm_training._make_grouped_folds(recordings, seed=0, folds=5)


def test_grouped_folds_reject_training_fold_missing_label(
        synthetic_recordings, monkeypatch):
    class OneLabelSplitter:
        def __init__(self, **_kwargs):
            pass

        def split(self, _features, _labels, _groups):
            yield np.arange(5), np.arange(5, 10)

    monkeypatch.setattr(
        alarm_training,
        "_load_sklearn",
        lambda: (object, OneLabelSplitter),
    )

    with pytest.raises(TrainingError, match="both labels"):
        alarm_training._make_grouped_folds(
            synthetic_recordings,
            seed=0,
            folds=5,
        )


def test_threshold_is_highest_candidate_meeting_all_ceilings():
    scored = scored_corpus(
        positive={"p1": [0.9, 0.1, 0.8], "p2": [0.75, 0.8]},
        negative={"n1": [0.7, 0.1], "n2": [0.2, 0.1]},
    )
    threshold = alarm_training._select_threshold(scored)
    assert threshold == pytest.approx(0.75)
    metrics = alarm_training._evaluate_threshold(scored, threshold)
    assert metrics.positive_groups_triggered == metrics.positive_groups_total
    assert metrics.negative_groups_triggered <= 0.2 * metrics.negative_groups_total


def test_always_positive_scores_are_rejected():
    scored = scored_corpus(
        positive={f"p{i}": [0.5, 0.5] for i in range(5)},
        negative={f"n{i}": [0.5, 0.5] for i in range(5)},
    )
    with pytest.raises(TrainingError, match="false-alert"):
        alarm_training._select_threshold(scored)


def test_file_event_count_resets_gate_and_debounce_per_call():
    assert alarm_training._count_file_events((0.9,), 0.5) == 0
    assert alarm_training._count_file_events((0.9,), 0.5) == 0
    assert alarm_training._count_file_events((0.9, 0.9), 0.5) == 1
    assert alarm_training._count_file_events((0.9, 0.9), 0.5) == 1


def test_metrics_aggregate_any_file_by_group_and_exact_false_event_rate():
    scored = (
        alarm_training.ScoredRecording(
            "positive", "p-a.wav", 1, 1.0, (0.9,),
        ),
        alarm_training.ScoredRecording(
            "positive", "p-b.wav", 1, 1.0, (0.9, 0.9),
        ),
        alarm_training.ScoredRecording(
            "negative-a", "n-a.wav", 0, 60.0, (0.9,) * 22,
        ),
        alarm_training.ScoredRecording(
            "negative-a", "n-b.wav", 0, 60.0, (0.1, 0.1),
        ),
        alarm_training.ScoredRecording(
            "negative-b", "n-c.wav", 0, 120.0, (0.1, 0.1),
        ),
    )

    metrics = alarm_training._evaluate_threshold(scored, 0.5)

    assert metrics.positive_groups_total == 1
    assert metrics.positive_groups_triggered == 1
    assert metrics.negative_groups_total == 2
    assert metrics.negative_groups_triggered == 1
    assert metrics.false_events == 2
    assert metrics.negative_audio_minutes == pytest.approx(4.0)
    assert metrics.false_triggers_per_minute == pytest.approx(0.5)
    assert metrics.evaluated_windows == 29
    assert [item["path"] for item in metrics.files] == [
        "p-a.wav", "p-b.wav", "n-a.wav", "n-b.wav", "n-c.wav",
    ]
    assert [item["evaluated_windows"] for item in metrics.files] == [
        1, 2, 22, 2, 2,
    ]
    assert alarm_training._metrics_payload(
        metrics, "out_of_fold",
    )["evaluated_windows"] == 29


def test_final_threshold_optimizes_positive_recall_without_negative_ceilings():
    scored = scored_corpus(
        positive={f"p{i}": [0.9, 0.8] for i in range(5)},
        negative={f"n{i}": [0.95, 0.95] for i in range(5)},
    )

    threshold = alarm_training._select_positive_recall_threshold(scored)

    assert threshold == pytest.approx(0.8)


class FakeYamNet:
    class_names = (
        "Smoke detector, smoke alarm",
        "Speech",
        *(f"class-{index}" for index in range(2, 521)),
    )

    def infer(self, waveform):
        level = float(np.mean(np.asarray(waveform, dtype=np.float32)))
        scores = np.zeros(521, dtype=np.float32)
        scores[0 if level < 0 else 1] = 0.99
        embedding = np.zeros(1024, dtype=np.float32)
        embedding[0] = level
        return scores, embedding


def fit_recording(tmp_path, group, label, digest_value):
    text_label = "alarm" if label else "not_alarm"
    value = 0.8 if label else -0.2
    audio = np.full(24_000, value, dtype=np.float32)
    audio[-1] += digest_value * 1e-6
    entry = CorpusEntry(
        path=tmp_path / text_label / f"{group}.wav",
        relative_path=f"{text_label}/{group}.wav",
        label=text_label,
        source_group=group,
        segments=(),
        decoded_sha256=decoded_digest(audio),
        duration_seconds=1.5,
    )
    windows = tuple(
        PreparedWindow(
            waveform=audio[
                index * config.HOP_SAMPLES:
                index * config.HOP_SAMPLES + config.WINDOW_SAMPLES
            ],
            start_sample=index * config.HOP_SAMPLES,
            weight=0.5,
        )
        for index in range(2)
    )
    return (
        alarm_training.PreparedRecording(
            entry=entry,
            windows=windows,
            duration_seconds=1.5,
        ),
        audio,
    )


@pytest.fixture
def train_case(tmp_path, monkeypatch):
    prepared = tuple(
        fit_recording(tmp_path, f"positive-{index}", 1, index + 1)
        for index in range(5)
    ) + tuple(
        fit_recording(tmp_path, f"negative-{index}", 0, index + 101)
        for index in range(5)
    )
    recordings = tuple(item[0] for item in prepared)
    audio_by_path = {
        recording.entry.path: audio
        for recording, audio in prepared
    }
    corpus = CorpusInventory(
        root=tmp_path / "data",
        entries=tuple(item.entry for item in recordings),
        warnings=("synthetic warning",),
    )
    monkeypatch.setattr(
        alarm_training,
        "inventory_corpus",
        lambda _path: corpus,
        raising=False,
    )
    monkeypatch.setattr(
        alarm_training,
        "_prepare_recordings",
        lambda _inventory: recordings,
    )
    monkeypatch.setattr(
        alarm_training,
        "load_wav_16k_mono",
        lambda path: audio_by_path[path].copy(),
    )
    model_path = tmp_path / "yamnet.tflite"
    class_map_path = tmp_path / "yamnet_class_map.csv"
    model_path.write_bytes(b"synthetic-yamnet")
    class_map_path.write_bytes(b"synthetic-class-map")
    return {
        "recordings": recordings,
        "corpus": corpus,
        "yamnet": FakeYamNet(),
        "model_path": model_path,
        "class_map_path": class_map_path,
        "output": tmp_path / "fire_smoke_alarm_head.npz",
        "report": tmp_path / "fire_smoke_alarm_report.json",
    }


def test_fitted_head_is_deterministic_for_seeded_training(train_case):
    arguments = {
        "yamnet": train_case["yamnet"],
        "seed": 13,
        "yamnet_model_sha256": "1" * 64,
        "class_map_sha256": "2" * 64,
    }

    first = alarm_training._fit_alarm_head(
        train_case["recordings"], **arguments,
    )
    second = alarm_training._fit_alarm_head(
        train_case["recordings"], **arguments,
    )

    np.testing.assert_array_equal(first.mean, second.mean)
    np.testing.assert_array_equal(first.scale, second.scale)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.bias == pytest.approx(second.bias)


def test_fitting_uses_exact_logistic_configuration(train_case, monkeypatch):
    captured = {}

    class FakeClassifier:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def fit(self, values, labels, *, sample_weight):
            captured["shape"] = values.shape
            captured["labels"] = tuple(labels.tolist())
            captured["sample_weight"] = tuple(sample_weight.tolist())
            self.coef_ = np.zeros((1, 1024), dtype=np.float64)
            self.intercept_ = np.zeros(1, dtype=np.float64)

    monkeypatch.setattr(
        alarm_training,
        "_load_sklearn",
        lambda: (FakeClassifier, object),
    )

    alarm_training._fit_alarm_head(
        train_case["recordings"],
        yamnet=train_case["yamnet"],
        seed=23,
        yamnet_model_sha256="1" * 64,
        class_map_sha256="2" * 64,
    )

    assert captured["kwargs"] == {
        "C": 1.0,
        "solver": "liblinear",
        "class_weight": "balanced",
        "max_iter": 2000,
        "random_state": 23,
    }
    assert captured["shape"] == (40, 1024)
    assert set(captured["labels"]) == {0, 1}
    assert sum(captured["sample_weight"]) == pytest.approx(10.0)


def test_scoring_is_deterministic_and_preserves_recording_boundaries(train_case):
    head = alarm_training._fit_alarm_head(
        train_case["recordings"],
        yamnet=train_case["yamnet"],
        seed=17,
        yamnet_model_sha256="1" * 64,
        class_map_sha256="2" * 64,
    )

    first = alarm_training._score_recordings(
        train_case["recordings"], head, train_case["yamnet"],
    )
    second = alarm_training._score_recordings(
        train_case["recordings"], head, train_case["yamnet"],
    )

    assert first == second
    assert [item.source_path for item in first] == [
        item.entry.relative_path for item in train_case["recordings"]
    ]
    assert [len(item.scores) for item in first] == [2] * 10


def test_scoring_reloads_every_contiguous_window_beyond_fit_cap(tmp_path):
    window_count = 80
    audio = np.zeros(
        config.WINDOW_SAMPLES + config.HOP_SAMPLES * (window_count - 1),
        dtype=np.float32,
    )
    starts = tuple(range(0, len(audio) - config.WINDOW_SAMPLES + 1,
                         config.HOP_SAMPLES))
    for index, start in enumerate(starts):
        audio[start] = index / 100.0
    entry = corpus_entry_for_audio(tmp_path, audio)
    recording = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )[0]
    loader_calls = []

    def loader(path):
        loader_calls.append(path)
        return audio

    scored = alarm_training._score_recordings(
        (recording,), FirstValueHead(), FirstSampleYamNet(), audio_loader=loader,
    )

    assert len(recording.windows) == alarm_training.MAX_WINDOWS_PER_RECORDING
    assert loader_calls == [entry.path]
    assert len(scored[0].scores) == window_count
    assert scored[0].scores == pytest.approx(
        tuple(float(audio[start]) for start in starts)
    )


def test_full_timeline_does_not_bridge_sparse_fit_evidence(tmp_path):
    window_count = config.ALARM_GATE_WINDOW + 2
    audio = np.full(
        config.WINDOW_SAMPLES + config.HOP_SAMPLES * (window_count - 1),
        0.1,
        dtype=np.float32,
    )
    starts = tuple(range(0, len(audio) - config.WINDOW_SAMPLES + 1,
                         config.HOP_SAMPLES))
    audio[starts[0]] = 0.9
    audio[starts[-1]] = 0.9
    entry = corpus_entry_for_audio(tmp_path, audio)
    fit_windows = tuple(
        PreparedWindow(
            waveform=audio[start:start + config.WINDOW_SAMPLES],
            start_sample=start,
            weight=0.5,
        )
        for start in (starts[0], starts[-1])
    )
    recording = alarm_training.PreparedRecording(
        entry=entry,
        windows=fit_windows,
        duration_seconds=len(audio) / config.SAMPLE_RATE,
    )

    scored = alarm_training._score_recordings(
        (recording,),
        FirstValueHead(),
        FirstSampleYamNet(),
        audio_loader=lambda _path: audio,
    )
    metrics = alarm_training._evaluate_threshold(scored, 0.5)

    assert scored[0].scores == pytest.approx(
        (0.9, *(0.1 for _ in range(window_count - 2)), 0.9)
    )
    assert metrics.evaluated_windows == window_count
    assert metrics.negative_groups_triggered == 0
    assert metrics.false_events == 0


def test_scoring_rejects_changed_sample_count_with_source_path(tmp_path):
    audio = np.zeros(
        config.WINDOW_SAMPLES + config.HOP_SAMPLES,
        dtype=np.float32,
    )
    entry = corpus_entry_for_audio(
        tmp_path, audio, filename="changing.wav",
    )
    recording = _prepare_recordings(
        inventory(entry), audio_loader=lambda _path: audio,
    )[0]
    changed = np.concatenate([audio, np.zeros(1, dtype=np.float32)])

    with pytest.raises(
        TrainingError,
        match=r"not_alarm/changing\.wav.*sample count.*changed",
    ):
        alarm_training._score_recordings(
            (recording,),
            object(),
            object(),
            audio_loader=lambda _path: changed,
        )


def test_scoring_rejects_changed_decoded_hash_with_source_path(tmp_path):
    audio = np.zeros(
        config.WINDOW_SAMPLES + config.HOP_SAMPLES,
        dtype=np.float32,
    )
    entry = corpus_entry_for_audio(
        tmp_path, audio, filename="changed-content.wav",
    )
    recording = _prepare_recordings(
        inventory(entry), audio_loader=lambda _path: audio,
    )[0]
    changed = audio.copy()
    changed[0] = 0.25

    with pytest.raises(
        TrainingError,
        match=r"not_alarm/changed-content\.wav.*SHA-256.*changed",
    ):
        alarm_training._score_recordings(
            (recording,),
            object(),
            object(),
            audio_loader=lambda _path: changed,
        )


def test_scoring_wraps_loader_failure_with_source_path(tmp_path):
    audio = np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
    entry = corpus_entry_for_audio(
        tmp_path, audio, filename="unreadable.wav",
    )
    recording = _prepare_recordings(
        inventory(entry), audio_loader=lambda _path: audio,
    )[0]

    def fail_loader(_path):
        raise OSError("source disappeared")

    with pytest.raises(
        TrainingError,
        match=r"not_alarm/unreadable\.wav.*could not load.*source disappeared",
    ):
        alarm_training._score_recordings(
            (recording,),
            object(),
            object(),
            audio_loader=fail_loader,
        )


def test_positive_segments_filter_fit_only_not_evaluation_timeline(tmp_path):
    quiet = np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
    active = np.full(config.WINDOW_SAMPLES, 0.2, dtype=np.float32)
    audio = np.concatenate([quiet[:8_000], active, quiet[:8_000]])
    entry = corpus_entry_for_audio(
        tmp_path,
        audio,
        label="alarm",
        segments=((0.5, 1.475),),
    )
    recording = _prepare_recordings(
        inventory(entry), audio_loader=lambda _path: audio,
    )[0]

    scored = alarm_training._score_recordings(
        (recording,),
        FirstValueHead(),
        FirstSampleYamNet(),
        audio_loader=lambda _path: audio,
    )

    assert [window.start_sample for window in recording.windows] == [8_000]
    assert len(scored[0].scores) == 3


def test_false_rate_scores_full_negative_exposure_not_fit_sample(tmp_path):
    window_count = 121
    audio = np.full(
        config.WINDOW_SAMPLES + config.HOP_SAMPLES * (window_count - 1),
        0.9,
        dtype=np.float32,
    )
    entry = corpus_entry_for_audio(
        tmp_path, audio, filename="long-negative.wav",
    )
    recording = _prepare_recordings(
        inventory(entry), audio_loader=lambda _path: audio,
    )[0]

    scored = alarm_training._score_recordings(
        (recording,),
        FirstValueHead(),
        FirstSampleYamNet(),
        audio_loader=lambda _path: audio,
    )
    metrics = alarm_training._evaluate_threshold(scored, 0.5)
    expected_events = alarm_training._count_file_events(
        (0.9,) * window_count, 0.5,
    )
    expected_minutes = (len(audio) / config.SAMPLE_RATE) / 60.0

    assert len(recording.windows) == alarm_training.MAX_WINDOWS_PER_RECORDING
    assert metrics.evaluated_windows == window_count
    assert metrics.files[0]["evaluated_windows"] == window_count
    assert metrics.false_events == expected_events
    assert metrics.negative_audio_minutes == pytest.approx(expected_minutes)
    assert metrics.false_triggers_per_minute == pytest.approx(
        expected_events / expected_minutes
    )


def test_full_scoring_preserves_file_order_and_resets_gate_between_files(
    tmp_path,
):
    first_audio = np.full(config.WINDOW_SAMPLES, 0.8, dtype=np.float32)
    second_audio = np.full(config.WINDOW_SAMPLES, 0.9, dtype=np.float32)
    first_entry = corpus_entry_for_audio(
        tmp_path,
        first_audio,
        group="first",
        filename="first.wav",
    )
    second_entry = corpus_entry_for_audio(
        tmp_path,
        second_audio,
        group="second",
        filename="second.wav",
    )
    recordings = _prepare_recordings(
        inventory(first_entry, second_entry),
        audio_loader=lambda path: (
            first_audio if path == first_entry.path else second_audio
        ),
    )
    audio_by_path = {
        first_entry.path: first_audio,
        second_entry.path: second_audio,
    }
    loader_calls = []

    def loader(path):
        loader_calls.append(path)
        return audio_by_path[path]

    scored = alarm_training._score_recordings(
        recordings,
        FirstValueHead(),
        FirstSampleYamNet(),
        audio_loader=loader,
    )
    metrics = alarm_training._evaluate_threshold(scored, 0.5)

    assert loader_calls == [first_entry.path, second_entry.path]
    assert [item.source_path for item in scored] == [
        first_entry.relative_path,
        second_entry.relative_path,
    ]
    assert [item.scores[0] for item in scored] == pytest.approx([0.8, 0.9])
    assert metrics.evaluated_windows == 2
    assert metrics.negative_groups_triggered == 0
    assert metrics.false_events == 0


def test_content_audit_reports_raw_classes_without_filtering_labels(train_case):
    audit = alarm_training._build_content_audit(
        train_case["recordings"], train_case["yamnet"],
    )

    assert len(audit) == len(train_case["recordings"])
    assert audit[0]["label"] == "alarm"
    assert audit[0]["top_classes"][0] == {
        "name": "Speech",
        "score": pytest.approx(0.99),
    }
    assert audit[-1]["label"] == "not_alarm"
    assert audit[-1]["top_classes"][0]["name"] \
        == "Smoke detector, smoke alarm"


def test_json_writer_sorts_indents_newline_and_fsyncs(tmp_path, monkeypatch):
    fsynced = []
    monkeypatch.setattr(os, "fsync", lambda descriptor: fsynced.append(descriptor))
    destination = tmp_path / "report.json"

    alarm_training._write_json_atomic(
        destination,
        {"z": 1, "a": {"b": 2}},
    )

    assert destination.read_text() == (
        '{\n  "a": {\n    "b": 2\n  },\n  "z": 1\n}\n'
    )
    assert len(fsynced) == 1


def test_public_report_contracts_are_immutable(tmp_path):
    metrics = alarm_training.EvaluationMetrics(
        positive_groups_total=1,
        positive_groups_triggered=1,
        negative_groups_total=1,
        negative_groups_triggered=0,
        false_events=0,
        negative_audio_minutes=1.0,
        false_triggers_per_minute=0.0,
        evaluated_windows=0,
        files=(),
    )
    trained = alarm_training.TrainingReport(
        artifact_path=tmp_path / "head.npz",
        report_path=tmp_path / "report.json",
        deployment_threshold=0.5,
        oof_metrics=metrics,
        in_sample_metrics=metrics,
        payload={},
    )
    evaluated = alarm_training.EvaluationReport(
        metrics=metrics,
        evaluation_scope="in_sample",
        payload={},
    )

    with pytest.raises(FrozenInstanceError):
        trained.deployment_threshold = 0.4
    with pytest.raises(FrozenInstanceError):
        evaluated.evaluation_scope = "external_corpus"


def test_train_and_evaluate_report_oof_and_in_sample_honestly(train_case):
    result = alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        seed=7,
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    assert isinstance(result, alarm_training.TrainingReport)
    assert result.artifact_path == train_case["output"]
    assert result.report_path == train_case["report"]
    assert result.deployment_threshold == pytest.approx(min(
        result.payload["thresholds"]["out_of_fold"],
        result.payload["thresholds"]["final_positive_recall"],
    ))
    assert result.payload["metrics"]["out_of_fold"]["evaluation_scope"] \
        == "out_of_fold"
    assert result.payload["metrics"]["final_model"]["evaluation_scope"] \
        == "in_sample"
    assert len(result.payload["content_audit"]) == len(train_case["recordings"])
    assert {
        item["path"] for item in result.payload["content_audit"]
    } == {
        item.entry.relative_path for item in train_case["recordings"]
    }
    assert train_case["report"].read_text().endswith("\n")
    assert json.loads(train_case["report"].read_text()) == result.payload

    head = load_alarm_head(
        train_case["output"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    assert head.threshold == pytest.approx(result.deployment_threshold)
    evaluated = alarm_training.evaluate_alarm(
        train_case["corpus"].root,
        train_case["output"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    assert evaluated.evaluation_scope == "in_sample"
    assert evaluated.payload["evaluation_scope"] == "in_sample"
    assert evaluated.metrics.positive_groups_triggered \
        == evaluated.metrics.positive_groups_total


def test_training_report_records_seed_folds_hashes_counts_and_warnings(
        train_case):
    result = alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        seed=19,
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    assert result.payload["seed"] == 19
    assert len(result.payload["folds"]) == 5
    for fold in result.payload["folds"]:
        assert set(fold["train_groups"]).isdisjoint(fold["validation_groups"])
    assert result.payload["corpus"]["recordings"] == [
        {
            "path": recording.entry.relative_path,
            "decoded_sha256": recording.entry.decoded_sha256,
            "label": recording.entry.label,
            "source_group": recording.entry.source_group,
            "retained_windows": len(recording.windows),
            "sample_count": 24_000,
        }
        for recording in train_case["recordings"]
    ]
    assert result.payload["corpus"]["counts"] == {
        "recordings": 10,
        "retained_windows": 20,
        "samples": 240_000,
    }
    assert result.payload["corpus"]["warnings"] == ["synthetic warning"]


def test_deployment_threshold_is_minimum_of_oof_and_final_positive_recall(
        train_case, monkeypatch):
    monkeypatch.setattr(
        alarm_training,
        "_select_threshold",
        lambda _items: 0.8,
    )
    monkeypatch.setattr(
        alarm_training,
        "_select_positive_recall_threshold",
        lambda _items: 0.6,
    )
    monkeypatch.setattr(
        alarm_training,
        "_metrics_meet_ceilings",
        lambda _metrics: True,
    )

    result = alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    assert result.payload["thresholds"] == {
        "out_of_fold": 0.8,
        "final_positive_recall": 0.6,
        "deployment": 0.6,
    }
    head = load_alarm_head(
        train_case["output"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    assert head.threshold == pytest.approx(0.6)


def test_training_is_deterministic_for_same_seed(train_case):
    arguments = {
        "seed": 29,
        "yamnet": train_case["yamnet"],
        "yamnet_model_path": train_case["model_path"],
        "class_map_path": train_case["class_map_path"],
    }
    first = alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        **arguments,
    )
    first_head = load_alarm_head(
        train_case["output"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    second = alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        **arguments,
    )
    second_head = load_alarm_head(
        train_case["output"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    assert first.payload == second.payload
    np.testing.assert_array_equal(first_head.mean, second_head.mean)
    np.testing.assert_array_equal(first_head.scale, second_head.scale)
    np.testing.assert_array_equal(first_head.weights, second_head.weights)
    assert first_head.bias == pytest.approx(second_head.bias)
    assert first_head.threshold == pytest.approx(second_head.threshold)


def test_training_fits_five_grouped_heads_then_one_all_data_head(
        train_case, monkeypatch):
    real_fit = alarm_training._fit_alarm_head
    fit_groups = []

    def capture_groups(recordings, **kwargs):
        fit_groups.append({item.entry.source_group for item in recordings})
        return real_fit(recordings, **kwargs)

    monkeypatch.setattr(alarm_training, "_fit_alarm_head", capture_groups)

    alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    all_groups = {
        item.entry.source_group for item in train_case["recordings"]
    }
    assert len(fit_groups) == 6
    assert all(len(groups) == 8 for groups in fit_groups[:5])
    assert fit_groups[-1] == all_groups


@pytest.mark.parametrize("seed", [True, -1, 2 ** 32])
def test_training_rejects_invalid_seed_with_diagnostic(
        train_case, seed):
    with pytest.raises(TrainingError, match="seed"):
        alarm_training.train_alarm(
            train_case["corpus"].root,
            train_case["output"],
            train_case["report"],
            seed=seed,
            yamnet=train_case["yamnet"],
            yamnet_model_path=train_case["model_path"],
            class_map_path=train_case["class_map_path"],
        )

    assert not train_case["output"].exists()
    assert not train_case["report"].exists()
    assert train_case["report"].with_name(
        "fire_smoke_alarm_report_failed_report.json"
    ).is_file()


def test_corpus_metadata_requires_integer_decoded_sample_count(train_case):
    first = train_case["recordings"][0]
    bad_entry = replace(first.entry, duration_seconds=1.50001)
    bad_recording = replace(first, entry=bad_entry)
    bad_recordings = (bad_recording,) + train_case["recordings"][1:]
    bad_corpus = replace(
        train_case["corpus"],
        entries=(bad_entry,) + train_case["corpus"].entries[1:],
    )

    with pytest.raises(TrainingError, match="sample count"):
        alarm_training._corpus_metadata(bad_corpus, bad_recordings)


def test_evaluate_scope_uses_configured_report_and_exact_corpus_hashes(
        train_case, monkeypatch):
    configured_report = train_case["report"].parent / "configured" / "report.json"
    monkeypatch.setattr(config, "ALARM_MODEL_PATH", train_case["output"])
    monkeypatch.setattr(config, "ALARM_REPORT_PATH", configured_report)
    alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        configured_report,
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    matching = alarm_training.evaluate_alarm(
        train_case["corpus"].root,
        train_case["output"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    assert matching.evaluation_scope == "in_sample"

    payload = json.loads(configured_report.read_text())
    recordings = payload["corpus"]["recordings"]
    first_digest = recordings[0]["decoded_sha256"]
    second_digest = recordings[1]["decoded_sha256"]
    recordings[0]["decoded_sha256"] = second_digest
    recordings[1]["decoded_sha256"] = first_digest
    configured_report.write_text(json.dumps(payload), encoding="utf-8")
    path_mismatched = alarm_training.evaluate_alarm(
        train_case["corpus"].root,
        train_case["output"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    assert path_mismatched.evaluation_scope == "external_corpus"

    recordings[0]["decoded_sha256"] = "f" * 64
    recordings[1]["decoded_sha256"] = second_digest
    configured_report.write_text(json.dumps(payload), encoding="utf-8")
    mismatched = alarm_training.evaluate_alarm(
        train_case["corpus"].root,
        train_case["output"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    assert mismatched.evaluation_scope == "external_corpus"

    configured_report.unlink()
    missing = alarm_training.evaluate_alarm(
        train_case["corpus"].root,
        train_case["output"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    assert missing.evaluation_scope == "external_corpus"


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema", "not-an-alarm-training-report"),
        ("schema_version", 2),
        ("status", "failed"),
    ],
)
def test_evaluate_scope_rejects_invalid_training_report_identity(
        train_case, field, invalid_value):
    alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )
    payload = json.loads(train_case["report"].read_text())
    payload[field] = invalid_value
    train_case["report"].write_text(json.dumps(payload), encoding="utf-8")

    evaluated = alarm_training.evaluate_alarm(
        train_case["corpus"].root,
        train_case["output"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    assert evaluated.evaluation_scope == "external_corpus"


def test_evaluation_payload_includes_current_corpus_and_content_audit(
        train_case):
    trained = alarm_training.train_alarm(
        train_case["corpus"].root,
        train_case["output"],
        train_case["report"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    evaluated = alarm_training.evaluate_alarm(
        train_case["corpus"].root,
        train_case["output"],
        yamnet=train_case["yamnet"],
        yamnet_model_path=train_case["model_path"],
        class_map_path=train_case["class_map_path"],
    )

    assert evaluated.payload["corpus"] == trained.payload["corpus"]
    assert evaluated.payload["content_audit"] == trained.payload["content_audit"]
    assert evaluated.payload["metrics"]["files"] == [
        dict(item) for item in evaluated.metrics.files
    ]


def test_failed_training_preserves_known_good_outputs(tmp_path, monkeypatch):
    output = tmp_path / "head.npz"
    report = tmp_path / "report.json"
    output.write_bytes(b"known-good-model")
    report.write_bytes(b"known-good-report")
    monkeypatch.setattr(
        alarm_training,
        "inventory_corpus",
        lambda _path: (_ for _ in ()).throw(TrainingError("bad corpus")),
        raising=False,
    )

    with pytest.raises(TrainingError, match="bad corpus"):
        alarm_training.train_alarm(tmp_path / "data", output, report)

    assert output.read_bytes() == b"known-good-model"
    assert report.read_bytes() == b"known-good-report"
    failed_report = tmp_path / "report_failed_report.json"
    assert json.loads(failed_report.read_text())["status"] == "failed"


def test_output_failed_report_collision_changes_nothing_before_rejection(
        tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    output = tmp_path / "report_failed_report.json"
    report.write_bytes(b"known-good-report")
    output.write_bytes(b"known-good-model")
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    snapshots = []
    writes = []
    inventory_calls = []
    real_snapshot = alarm_training._snapshot_file
    real_write = alarm_training._write_json_atomic

    def track_snapshot(path):
        snapshots.append(path)
        return real_snapshot(path)

    def track_write(path, payload):
        writes.append(path)
        return real_write(path, payload)

    def fail_if_inventoried(_path):
        inventory_calls.append(True)
        raise TrainingError("inventory should not run")

    monkeypatch.setattr(alarm_training, "_snapshot_file", track_snapshot)
    monkeypatch.setattr(alarm_training, "_write_json_atomic", track_write)
    monkeypatch.setattr(alarm_training, "inventory_corpus", fail_if_inventoried)

    with pytest.raises(TrainingError) as captured:
        alarm_training.train_alarm(tmp_path / "data", output, report)

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert "distinct" in str(captured.value)
    assert snapshots == []
    assert writes == []
    assert inventory_calls == []


@pytest.mark.parametrize("collision", ["output_report", "output_failed_report"])
def test_canonical_path_alias_collisions_reject_before_transaction(
        tmp_path, monkeypatch, collision):
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    report = tmp_path / "report.json"
    failed_report = tmp_path / "report_failed_report.json"
    report.write_bytes(b"known-good-report")
    if collision == "output_report":
        output = subdir / ".." / report.name
    else:
        failed_report.write_bytes(b"known-good-model")
        output = subdir / ".." / failed_report.name
    before = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    snapshots = []
    writes = []
    inventory_calls = []
    real_snapshot = alarm_training._snapshot_file
    real_write = alarm_training._write_json_atomic

    def track_snapshot(path):
        snapshots.append(path)
        return real_snapshot(path)

    def track_write(path, payload):
        writes.append(path)
        return real_write(path, payload)

    def fail_if_inventoried(_path):
        inventory_calls.append(True)
        raise TrainingError("inventory should not run")

    monkeypatch.setattr(alarm_training, "_snapshot_file", track_snapshot)
    monkeypatch.setattr(alarm_training, "_write_json_atomic", track_write)
    monkeypatch.setattr(alarm_training, "inventory_corpus", fail_if_inventoried)

    with pytest.raises(TrainingError) as captured:
        alarm_training.train_alarm(tmp_path / "data", output, report)

    after = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert "distinct" in str(captured.value)
    assert snapshots == []
    assert writes == []
    assert inventory_calls == []


def test_model_install_failure_rolls_back_report_after_report_first(
        train_case, monkeypatch):
    output = train_case["output"]
    report = train_case["report"]
    output.write_bytes(b"known-good-model")
    report.write_bytes(b"known-good-report")

    def fail_model_install(path, head):
        assert path == output
        assert report.read_bytes() != b"known-good-report"
        path.write_bytes(b"corrupt-partial-model")
        raise AlarmModelError("model install failed")

    monkeypatch.setattr(
        alarm_training,
        "save_alarm_head",
        fail_model_install,
        raising=False,
    )

    with pytest.raises(TrainingError, match="model install failed"):
        alarm_training.train_alarm(
            train_case["corpus"].root,
            output,
            report,
            yamnet=train_case["yamnet"],
            yamnet_model_path=train_case["model_path"],
            class_map_path=train_case["class_map_path"],
        )

    assert output.read_bytes() == b"known-good-model"
    assert report.read_bytes() == b"known-good-report"
    assert (train_case["report"].with_name(
        "fire_smoke_alarm_report_failed_report.json"
    )).is_file()


def test_report_write_failure_rolls_back_before_model_install(
        train_case, monkeypatch):
    output = train_case["output"]
    report = train_case["report"]
    output.write_bytes(b"known-good-model")
    report.write_bytes(b"known-good-report")
    real_write = alarm_training._write_json_atomic
    model_called = []

    def fail_primary_report(path, payload):
        if path == report:
            path.write_bytes(b"corrupt-partial-report")
            raise OSError("report install failed")
        return real_write(path, payload)

    monkeypatch.setattr(alarm_training, "_write_json_atomic", fail_primary_report)
    monkeypatch.setattr(
        alarm_training,
        "save_alarm_head",
        lambda *_args: model_called.append(True),
    )

    with pytest.raises(TrainingError, match="report install failed"):
        alarm_training.train_alarm(
            train_case["corpus"].root,
            output,
            report,
            yamnet=train_case["yamnet"],
            yamnet_model_path=train_case["model_path"],
            class_map_path=train_case["class_map_path"],
        )

    assert not model_called
    assert output.read_bytes() == b"known-good-model"
    assert report.read_bytes() == b"known-good-report"
    assert report.with_name(
        "fire_smoke_alarm_report_failed_report.json"
    ).is_file()


def test_failed_first_install_removes_new_report_and_partial_model(
        train_case, monkeypatch):
    output = train_case["output"]
    report = train_case["report"]

    def corrupt_then_fail(path, _head):
        path.write_bytes(b"partial-model")
        raise AlarmModelError("first model install failed")

    monkeypatch.setattr(alarm_training, "save_alarm_head", corrupt_then_fail)

    with pytest.raises(TrainingError, match="first model install failed"):
        alarm_training.train_alarm(
            train_case["corpus"].root,
            output,
            report,
            yamnet=train_case["yamnet"],
            yamnet_model_path=train_case["model_path"],
            class_map_path=train_case["class_map_path"],
        )

    assert not output.exists()
    assert not report.exists()
    assert report.with_name(
        "fire_smoke_alarm_report_failed_report.json"
    ).is_file()


def test_final_head_ceiling_failure_never_replaces_outputs(
        train_case, monkeypatch):
    output = train_case["output"]
    report = train_case["report"]
    output.write_bytes(b"known-good-model")
    report.write_bytes(b"known-good-report")
    ceiling_results = iter((True, False))
    model_called = []
    monkeypatch.setattr(alarm_training, "_select_threshold", lambda _items: 0.5)
    monkeypatch.setattr(
        alarm_training,
        "_select_positive_recall_threshold",
        lambda _items: 0.5,
    )
    monkeypatch.setattr(
        alarm_training,
        "_metrics_meet_ceilings",
        lambda _metrics: next(ceiling_results),
    )
    monkeypatch.setattr(
        alarm_training,
        "save_alarm_head",
        lambda *_args: model_called.append(True),
    )

    with pytest.raises(TrainingError, match="final-head evaluation ceilings"):
        alarm_training.train_alarm(
            train_case["corpus"].root,
            output,
            report,
            yamnet=train_case["yamnet"],
            yamnet_model_path=train_case["model_path"],
            class_map_path=train_case["class_map_path"],
        )

    assert not model_called
    assert output.read_bytes() == b"known-good-model"
    assert report.read_bytes() == b"known-good-report"
    assert report.with_name(
        "fire_smoke_alarm_report_failed_report.json"
    ).is_file()


def test_lower_final_threshold_must_still_meet_oof_ceilings(
        train_case, monkeypatch):
    output = train_case["output"]
    report = train_case["report"]
    output.write_bytes(b"known-good-model")
    report.write_bytes(b"known-good-report")
    model_called = []
    monkeypatch.setattr(alarm_training, "_select_threshold", lambda _items: 0.8)
    monkeypatch.setattr(
        alarm_training,
        "_select_positive_recall_threshold",
        lambda _items: 0.6,
    )
    monkeypatch.setattr(
        alarm_training,
        "_metrics_meet_ceilings",
        lambda _metrics: False,
    )
    monkeypatch.setattr(
        alarm_training,
        "save_alarm_head",
        lambda *_args: model_called.append(True),
    )

    with pytest.raises(TrainingError, match="out-of-fold evaluation ceilings"):
        alarm_training.train_alarm(
            train_case["corpus"].root,
            output,
            report,
            yamnet=train_case["yamnet"],
            yamnet_model_path=train_case["model_path"],
            class_map_path=train_case["class_map_path"],
        )

    assert not model_called
    assert output.read_bytes() == b"known-good-model"
    assert report.read_bytes() == b"known-good-report"


def test_evenly_spaced_caps_long_recordings_deterministically():
    values = np.arange(100)

    selected = _evenly_spaced(values, limit=40)

    assert len(selected) == 40
    assert selected[0] == 0
    assert selected[-1] == 99
    np.testing.assert_array_equal(selected, _evenly_spaced(values, 40))
    np.testing.assert_array_equal(_evenly_spaced(np.arange(5), 40), np.arange(5))


@pytest.mark.parametrize("limit", [0, -1, True, 2.5])
def test_evenly_spaced_rejects_invalid_limits(limit):
    with pytest.raises(TrainingError, match="limit"):
        _evenly_spaced(np.arange(10), limit)


def test_prepare_uses_exact_window_starts_hop_and_recording_weights(tmp_path):
    entry = corpus_entry(tmp_path, label="not_alarm")
    audio = np.linspace(-0.2, 0.2, 31_600, dtype=np.float32)

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )

    windows = prepared[0].windows
    assert [window.start_sample for window in windows] == [0, 8_000, 16_000]
    assert sum(window.weight for window in windows) == pytest.approx(1.0)
    assert all(window.weight == pytest.approx(1 / 3) for window in windows)
    np.testing.assert_array_equal(
        windows[-1].waveform,
        audio[16_000:31_600],
    )


def test_positive_segments_require_half_window_overlap_and_activity(tmp_path):
    entry = corpus_entry(
        tmp_path,
        label="alarm",
        segments=((0.5, 1.475),),
    )
    quiet = np.zeros(config.WINDOW_SAMPLES, np.float32)
    active = np.ones(config.WINDOW_SAMPLES, np.float32) * 0.2
    audio = np.concatenate([quiet[:8_000], active, quiet[:8_000]])

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )

    assert len(prepared[0].windows) == 1
    assert prepared[0].windows[0].start_sample == 8_000


def test_segment_exactly_half_a_window_is_eligible(tmp_path):
    half_window_seconds = (config.WINDOW_SAMPLES / 2) / config.SAMPLE_RATE
    entry = corpus_entry(
        tmp_path,
        segments=((0.0, half_window_seconds),),
    )
    audio = np.full(config.WINDOW_SAMPLES, 0.2, dtype=np.float32)

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )

    assert [window.start_sample for window in prepared[0].windows] == [0]


def test_segment_below_half_a_window_is_ineligible_by_path(tmp_path):
    below_half_seconds = (
        (config.WINDOW_SAMPLES / 2) - 1
    ) / config.SAMPLE_RATE
    entry = corpus_entry(
        tmp_path,
        segments=((0.0, below_half_seconds),),
    )
    audio = np.full(config.WINDOW_SAMPLES, 0.2, dtype=np.float32)

    with pytest.raises(TrainingError, match=r"alarm/clip\.wav.*eligible"):
        _prepare_recordings(
            inventory(entry),
            audio_loader=lambda _path: audio,
        )


def test_empty_positive_segments_default_to_full_file(tmp_path):
    entry = corpus_entry(tmp_path, label="alarm", segments=())
    audio = np.full(config.WINDOW_SAMPLES, 0.2, dtype=np.float32)

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )

    assert [window.start_sample for window in prepared[0].windows] == [0]


def test_positive_activity_uses_recording_p95_threshold(tmp_path):
    entry = corpus_entry(tmp_path, label="alarm", segments=())
    audio = np.zeros(23_600, dtype=np.float32)
    audio[15_600:] = 0.2

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )

    assert [window.start_sample for window in prepared[0].windows] == [8_000]


def test_silent_positive_has_path_specific_no_window_error(tmp_path):
    entry = corpus_entry(tmp_path, label="alarm")
    audio = np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)

    with pytest.raises(TrainingError, match=r"alarm/clip\.wav.*eligible"):
        _prepare_recordings(
            inventory(entry),
            audio_loader=lambda _path: audio,
        )


def test_negative_silence_is_retained(tmp_path):
    entry = corpus_entry(tmp_path, label="not_alarm")
    audio = np.zeros(31_600, dtype=np.float32)

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )

    assert [window.start_sample for window in prepared[0].windows] == [
        0,
        8_000,
        16_000,
    ]
    assert all(rms(window.waveform) == 0.0 for window in prepared[0].windows)


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((config.WINDOW_SAMPLES, 1), dtype=np.float32),
        np.full(config.WINDOW_SAMPLES, np.nan, dtype=np.float32),
        np.zeros(config.WINDOW_SAMPLES - 1, dtype=np.float32),
        ["not-a-number"] * config.WINDOW_SAMPLES,
    ],
    ids=["not-mono", "non-finite", "too-short", "non-numeric"],
)
def test_prepare_rejects_invalid_decoded_audio_by_path(tmp_path, audio):
    entry = corpus_entry(tmp_path, label="not_alarm")

    with pytest.raises(TrainingError, match=r"not_alarm/clip\.wav"):
        _prepare_recordings(
            inventory(entry),
            audio_loader=lambda _path: audio,
        )


def test_prepare_caps_each_recording_at_40_evenly_spaced_windows(tmp_path):
    entry = corpus_entry(tmp_path, label="not_alarm")
    window_count = 100
    sample_count = config.WINDOW_SAMPLES + config.HOP_SAMPLES * (window_count - 1)
    audio = np.zeros(sample_count, dtype=np.float32)

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )

    windows = prepared[0].windows
    assert len(windows) == 40
    assert windows[0].start_sample == 0
    assert windows[-1].start_sample == config.HOP_SAMPLES * (window_count - 1)
    assert sum(window.weight for window in windows) == pytest.approx(1.0)


def test_prepared_waveforms_are_owned_read_only_copies(tmp_path):
    entry = corpus_entry(tmp_path, label="not_alarm")
    audio = np.full(config.WINDOW_SAMPLES, 0.1, dtype=np.float32)

    prepared = _prepare_recordings(
        inventory(entry),
        audio_loader=lambda _path: audio,
    )
    window = prepared[0].windows[0]
    audio[0] = 0.9

    assert window.waveform[0] == pytest.approx(0.1)
    assert window.waveform.flags.writeable is False
    with pytest.raises(ValueError):
        window.waveform[0] = 0.3
    with pytest.raises(FrozenInstanceError):
        window.weight = 0.5


def test_weighted_window_owns_and_freezes_waveform():
    source = np.full(config.WINDOW_SAMPLES, 0.25, dtype=np.float32)
    window = WeightedWindow(
        waveform=source,
        label=1,
        source_group="positive-a",
        source_path="positive.wav",
        start_sample=0,
        weight=1.0,
        augmentation_noise_group=None,
    )
    source[0] = 0.9

    assert window.waveform[0] == pytest.approx(0.25)
    assert window.waveform.flags.writeable is False
    with pytest.raises(ValueError):
        window.waveform[0] = 0.3


def test_prepared_window_rejects_invalid_waveform_shape():
    with pytest.raises(TrainingError, match="waveform"):
        PreparedWindow(
            waveform=np.zeros((config.WINDOW_SAMPLES, 1), dtype=np.float32),
            start_sample=0,
            weight=1.0,
        )


def test_positive_augmentation_preserves_parent_weight_and_training_groups():
    positive = weighted_window(value=0.5, group="positive-a", weight=0.25)
    noise = weighted_window(
        value=0.1,
        group="negative-a",
        weight=1.0,
        label=0,
    )

    augmented = _augment_training_windows(
        [positive],
        [noise],
        rng=np.random.default_rng(7),
    )

    assert len(augmented) == 3
    assert sum(item.weight for item in augmented) == pytest.approx(0.25)
    assert {item.source_group for item in augmented} == {"positive-a"}
    assert all(
        item.augmentation_noise_group in {None, "negative-a"}
        for item in augmented
    )
    assert augmented[0].augmentation_noise_group is None
    assert augmented[1].augmentation_noise_group is None
    assert augmented[2].augmentation_noise_group == "negative-a"
    assert 0.35 <= float(augmented[1].waveform[0] / 0.5) <= 1.0


def test_augmentation_uses_only_provided_non_silent_negative_windows():
    positive = weighted_window(0.4, "positive-a", 1.0)
    provided = weighted_window(0.1, "training-negative", 1.0, label=0)
    silent = weighted_window(0.0, "silent-negative", 1.0, label=0)
    wrong_label = weighted_window(0.2, "validation-positive", 1.0, label=1)

    augmented = _augment_training_windows(
        [positive],
        [silent, wrong_label, provided],
        rng=np.random.default_rng(3),
    )

    assert augmented[-1].augmentation_noise_group == "training-negative"


def test_final_all_data_augmentation_can_use_any_provided_negative_group():
    positive = weighted_window(0.4, "positive-a", 1.0)
    final_negative = weighted_window(
        0.1,
        "final-all-data-negative",
        1.0,
        label=0,
    )

    augmented = _augment_training_windows(
        [positive],
        [final_negative],
        rng=np.random.default_rng(11),
    )

    assert augmented[-1].augmentation_noise_group == "final-all-data-negative"


def test_augmentation_requires_non_silent_negative_noise():
    positive = weighted_window(0.4, "positive-a", 1.0)
    silence = weighted_window(0.0, "negative-a", 1.0, label=0)

    with pytest.raises(TrainingError, match="non-silent"):
        _augment_training_windows(
            [positive],
            [silence],
            rng=np.random.default_rng(1),
        )


def test_seeded_augmentation_is_deterministic():
    positive = weighted_window(0.4, "positive-a", 0.5, start_sample=8_000)
    noises = [
        weighted_window(0.05, "negative-a", 1.0, label=0),
        weighted_window(-0.1, "negative-b", 1.0, label=0),
    ]

    first = _augment_training_windows(
        [positive], noises, rng=np.random.default_rng(42)
    )
    second = _augment_training_windows(
        [positive], noises, rng=np.random.default_rng(42)
    )

    assert [item.augmentation_noise_group for item in first] == [
        item.augmentation_noise_group for item in second
    ]
    assert [item.weight for item in first] == [item.weight for item in second]
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left.waveform, right.waveform)


def test_noise_scaling_reaches_requested_snr():
    signal = np.ones(config.WINDOW_SAMPLES, np.float32) * 0.4
    noise = np.ones(config.WINDOW_SAMPLES, np.float32) * 0.2

    mixed = mix_at_snr(signal, noise, snr_db=10.0)

    added = mixed - signal
    ratio_db = 20 * np.log10(rms(signal) / rms(added))
    assert ratio_db == pytest.approx(10.0, abs=0.05)


def test_mix_at_snr_clips_and_returns_finite_float32():
    signal = np.full(config.WINDOW_SAMPLES, 0.95, dtype=np.float32)
    noise = np.ones(config.WINDOW_SAMPLES, dtype=np.float32)

    mixed = mix_at_snr(signal, noise, snr_db=-20.0)

    assert mixed.dtype == np.dtype(np.float32)
    assert np.isfinite(mixed).all()
    assert np.max(mixed) <= 1.0
    assert np.min(mixed) >= -1.0


@pytest.mark.parametrize(
    ("signal", "noise", "snr_db"),
    [
        (
            np.ones(config.WINDOW_SAMPLES, dtype=np.float32),
            np.zeros(config.WINDOW_SAMPLES, dtype=np.float32),
            10.0,
        ),
        (
            np.full(config.WINDOW_SAMPLES, np.nan, dtype=np.float32),
            np.ones(config.WINDOW_SAMPLES, dtype=np.float32),
            10.0,
        ),
        (
            np.ones(config.WINDOW_SAMPLES, dtype=np.float32),
            np.ones(config.WINDOW_SAMPLES - 1, dtype=np.float32),
            10.0,
        ),
        (
            np.ones(config.WINDOW_SAMPLES, dtype=np.float32),
            np.ones(config.WINDOW_SAMPLES, dtype=np.float32),
            np.nan,
        ),
    ],
    ids=["silent-noise", "non-finite", "shape-mismatch", "non-finite-snr"],
)
def test_mix_at_snr_rejects_invalid_inputs(signal, noise, snr_db):
    with pytest.raises(TrainingError):
        mix_at_snr(signal, noise, snr_db)


def test_weighted_standardization_uses_weights_and_protects_zero_scale():
    values = np.array([[0.0, 3.0], [10.0, 3.0]], np.float32)

    mean, scale = _weighted_standardization(
        values,
        np.array([0.9, 0.1]),
    )

    np.testing.assert_allclose(mean, [1.0, 3.0])
    np.testing.assert_allclose(scale[0], 3.0)
    assert scale[1] == 1.0
    assert mean.dtype == np.dtype(np.float32)
    assert scale.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    ("values", "weights"),
    [
        (np.array([1.0, 2.0]), np.array([1.0, 1.0])),
        (np.empty((0, 2)), np.empty(0)),
        (np.ones((2, 2)), np.ones(3)),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), np.ones(2)),
        (np.ones((2, 2)), np.array([1.0, np.nan])),
        (np.ones((2, 2)), np.array([1.0, -1.0])),
        (np.ones((2, 2)), np.zeros(2)),
    ],
    ids=[
        "values-not-matrix",
        "empty",
        "weight-length",
        "non-finite-values",
        "non-finite-weights",
        "negative-weight",
        "zero-total-weight",
    ],
)
def test_weighted_standardization_rejects_invalid_inputs(values, weights):
    with pytest.raises(TrainingError, match="standardization"):
        _weighted_standardization(values, weights)
