"""Leakage-safe alarm recording preparation, training, and evaluation."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import config
from .alarm_data import (
    ALARM,
    NOT_ALARM,
    AlarmDataError,
    CorpusEntry,
    CorpusInventory,
    _decoded_digest,
    inventory_corpus,
)
from .alarm_model import (
    AlarmHead,
    RollingEvidenceGate,
    load_alarm_head,
    save_alarm_head,
)
from .artifacts import sha256_file
from .core import EventDetector, Observation
from .pipeline import YamNet, load_wav_16k_mono


MAX_WINDOWS_PER_RECORDING = 40
MIN_ACTIVITY_RMS = 1e-4
ACTIVITY_P95_FRACTION = 0.05
MIN_NOISE_RMS = 1e-6
MIN_STANDARDIZATION_SCALE = 1e-6
MAX_NEGATIVE_GROUP_FRACTION = 0.20
MAX_FALSE_TRIGGERS_PER_MINUTE = 0.5


class TrainingError(AlarmDataError):
    """Alarm training or evaluation could not produce a valid result."""


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
    evaluated_windows: int
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


def _write_json_atomic(path, payload) -> None:
    destination = Path(path)
    part: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".part",
            delete=False,
        ) as output:
            part = Path(output.name)
            json.dump(payload, output, sort_keys=True, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(part, destination)
        part = None
    finally:
        if part is not None:
            try:
                part.unlink()
            except OSError:
                pass


def _owned_waveform(value) -> np.ndarray:
    try:
        waveform = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError(
            f"waveform must contain {config.WINDOW_SAMPLES} finite mono samples"
        ) from exc
    if (
        waveform.shape != (config.WINDOW_SAMPLES,)
        or not np.isfinite(waveform).all()
    ):
        raise TrainingError(
            f"waveform must contain {config.WINDOW_SAMPLES} finite mono samples"
        )
    owned = np.array(waveform, dtype=np.float32, copy=True)
    owned.setflags(write=False)
    return owned


def _validated_nonnegative_integer(value, *, field: str) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 0
    ):
        raise TrainingError(f"{field} must be a non-negative integer")
    return int(value)


def _validated_positive_weight(value) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError("window weight must be positive and finite") from exc
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(weight) or weight <= 0:
        raise TrainingError("window weight must be positive and finite")
    return weight


@dataclass(frozen=True)
class PreparedWindow:
    waveform: np.ndarray
    start_sample: int
    weight: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "waveform", _owned_waveform(self.waveform))
        object.__setattr__(
            self,
            "start_sample",
            _validated_nonnegative_integer(
                self.start_sample,
                field="start_sample",
            ),
        )
        object.__setattr__(self, "weight", _validated_positive_weight(self.weight))


@dataclass(frozen=True)
class PreparedRecording:
    entry: CorpusEntry
    windows: tuple[PreparedWindow, ...]
    duration_seconds: float

    def __post_init__(self) -> None:
        if not isinstance(self.entry, CorpusEntry):
            raise TrainingError("prepared recording entry must be a CorpusEntry")
        windows = tuple(self.windows)
        if not windows or not all(isinstance(item, PreparedWindow) for item in windows):
            raise TrainingError("prepared recording must contain prepared windows")
        try:
            duration = float(self.duration_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrainingError("recording duration must be positive and finite") from exc
        if not np.isfinite(duration) or duration <= 0:
            raise TrainingError("recording duration must be positive and finite")
        object.__setattr__(self, "windows", windows)
        object.__setattr__(self, "duration_seconds", duration)


@dataclass(frozen=True)
class WeightedWindow:
    waveform: np.ndarray
    label: int
    source_group: str
    source_path: str
    start_sample: int
    weight: float
    augmentation_noise_group: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "waveform", _owned_waveform(self.waveform))
        if (
            isinstance(self.label, (bool, np.bool_))
            or not isinstance(self.label, (int, np.integer))
            or int(self.label) not in (0, 1)
        ):
            raise TrainingError("window label must be 0 or 1")
        if not isinstance(self.source_group, str) or not self.source_group:
            raise TrainingError("window source_group must be a non-empty string")
        if not isinstance(self.source_path, str) or not self.source_path:
            raise TrainingError("window source_path must be a non-empty string")
        if (
            self.augmentation_noise_group is not None
            and (
                not isinstance(self.augmentation_noise_group, str)
                or not self.augmentation_noise_group
            )
        ):
            raise TrainingError(
                "augmentation_noise_group must be None or a non-empty string"
            )
        object.__setattr__(self, "label", int(self.label))
        object.__setattr__(
            self,
            "start_sample",
            _validated_nonnegative_integer(
                self.start_sample,
                field="start_sample",
            ),
        )
        object.__setattr__(self, "weight", _validated_positive_weight(self.weight))


def rms(values) -> float:
    """Return root-mean-square amplitude using float64 accumulation."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError("RMS values must be non-empty and finite") from exc
    if array.size == 0 or not np.isfinite(array).all():
        raise TrainingError("RMS values must be non-empty and finite")
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sqrt(np.mean(np.square(array))))
    if not np.isfinite(result):
        raise TrainingError("RMS values must produce a finite result")
    return result


def _evenly_spaced(values, limit=MAX_WINDOWS_PER_RECORDING):
    """Return at most *limit* deterministic values, retaining both endpoints."""

    if (
        isinstance(limit, (bool, np.bool_))
        or not isinstance(limit, (int, np.integer))
        or limit <= 0
    ):
        raise TrainingError("evenly spaced limit must be a positive integer")
    array = np.asarray(values)
    if array.ndim != 1:
        raise TrainingError("evenly spaced values must be one-dimensional")
    limit = int(limit)
    if len(array) <= limit:
        return array.copy()
    positions = np.rint(
        np.linspace(0, len(array) - 1, num=limit, dtype=np.float64)
    ).astype(np.intp)
    return array[positions].copy()


def _decoded_audio(entry: CorpusEntry, audio_loader) -> np.ndarray:
    try:
        loaded = audio_loader(entry.path)
    except Exception as exc:
        raise TrainingError(
            f"{entry.relative_path}: could not load decoded audio: {exc}"
        ) from exc
    try:
        audio = np.asarray(loaded, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError(
            f"{entry.relative_path}: decoded audio must be finite mono audio"
        ) from exc
    if (
        audio.ndim != 1
        or len(audio) < config.WINDOW_SAMPLES
        or not np.isfinite(audio).all()
    ):
        raise TrainingError(
            f"{entry.relative_path}: decoded audio must be finite mono audio "
            f"with at least {config.WINDOW_SAMPLES} samples"
        )
    return audio


def _overlaps_half_window(start_sample: int, segments) -> bool:
    window_end = start_sample + config.WINDOW_SAMPLES
    required_overlap = config.WINDOW_SAMPLES / 2
    for segment_start_seconds, segment_end_seconds in segments:
        segment_start = float(segment_start_seconds) * config.SAMPLE_RATE
        segment_end = float(segment_end_seconds) * config.SAMPLE_RATE
        overlap = max(
            0.0,
            min(float(window_end), segment_end)
            - max(float(start_sample), segment_start),
        )
        if overlap >= required_overlap:
            return True
    return False


def _prepare_recordings(
    inventory: CorpusInventory,
    *,
    audio_loader=load_wav_16k_mono,
) -> tuple[PreparedRecording, ...]:
    """Decode and deterministically retain weighted windows per recording."""

    if not isinstance(inventory, CorpusInventory):
        raise TrainingError("inventory must be a CorpusInventory")
    prepared = []
    for entry in inventory.entries:
        audio = _decoded_audio(entry, audio_loader)
        starts = np.arange(
            0,
            len(audio) - config.WINDOW_SAMPLES + 1,
            config.HOP_SAMPLES,
            dtype=np.int64,
        )
        if entry.label == NOT_ALARM:
            eligible_starts = starts
        elif entry.label == ALARM:
            window_rms = np.array(
                [
                    rms(audio[start:start + config.WINDOW_SAMPLES])
                    for start in starts
                ],
                dtype=np.float64,
            )
            p95_rms = float(np.percentile(window_rms, 95))
            activity_threshold = max(
                MIN_ACTIVITY_RMS,
                ACTIVITY_P95_FRACTION * p95_rms,
            )
            segments = entry.segments or (
                (0.0, len(audio) / config.SAMPLE_RATE),
            )
            eligible_starts = np.array(
                [
                    start
                    for start, activity in zip(starts, window_rms)
                    if activity >= activity_threshold
                    and _overlaps_half_window(int(start), segments)
                ],
                dtype=np.int64,
            )
        else:
            raise TrainingError(
                f"{entry.relative_path}: unsupported corpus label {entry.label!r}"
            )

        retained_starts = _evenly_spaced(
            eligible_starts,
            MAX_WINDOWS_PER_RECORDING,
        )
        if not len(retained_starts):
            raise TrainingError(
                f"{entry.relative_path}: no eligible training windows"
            )
        window_weight = 1.0 / len(retained_starts)
        windows = tuple(
            PreparedWindow(
                waveform=audio[
                    int(start):int(start) + config.WINDOW_SAMPLES
                ],
                start_sample=int(start),
                weight=window_weight,
            )
            for start in retained_starts
        )
        prepared.append(
            PreparedRecording(
                entry=entry,
                windows=windows,
                duration_seconds=len(audio) / config.SAMPLE_RATE,
            )
        )
    return tuple(prepared)


def _mix_array(value, *, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError(f"{field} must be finite mono audio") from exc
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise TrainingError(f"{field} must be finite mono audio")
    return array


def mix_at_snr(signal, noise, snr_db):
    """Mix finite mono arrays at the requested signal-to-noise ratio."""

    signal_array = _mix_array(signal, field="signal")
    noise_array = _mix_array(noise, field="noise")
    if signal_array.shape != noise_array.shape:
        raise TrainingError("signal and noise must have matching shapes")
    try:
        snr = float(snr_db)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError("augmentation SNR must be finite") from exc
    if isinstance(snr_db, (bool, np.bool_)) or not np.isfinite(snr):
        raise TrainingError("augmentation SNR must be finite")

    signal_rms = rms(signal_array)
    noise_rms = rms(noise_array)
    if noise_rms < MIN_NOISE_RMS:
        raise TrainingError("augmentation noise must be non-silent")
    target_noise_rms = signal_rms / (10.0 ** (snr / 20.0))
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_noise = noise_array.astype(np.float64) * (
            target_noise_rms / noise_rms
        )
        mixed = signal_array.astype(np.float64) + scaled_noise
    if not np.isfinite(mixed).all():
        raise TrainingError("augmentation mix must contain only finite values")
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _derived_window(
    parent: WeightedWindow,
    *,
    waveform,
    weight: float,
    noise_group: str | None,
) -> WeightedWindow:
    return WeightedWindow(
        waveform=waveform,
        label=parent.label,
        source_group=parent.source_group,
        source_path=parent.source_path,
        start_sample=parent.start_sample,
        weight=weight,
        augmentation_noise_group=noise_group,
    )


def _augment_training_windows(
    positive_windows,
    negative_windows,
    *,
    rng,
) -> tuple[WeightedWindow, ...]:
    """Create gain and training-negative noise variants for positive windows."""

    positives = tuple(positive_windows)
    if not positives:
        return ()
    if not all(isinstance(item, WeightedWindow) and item.label == 1 for item in positives):
        raise TrainingError("positive augmentation inputs must have label 1")
    noise_candidates = tuple(
        item
        for item in negative_windows
        if isinstance(item, WeightedWindow)
        and item.label == 0
        and rms(item.waveform) >= MIN_NOISE_RMS
    )
    if not noise_candidates:
        raise TrainingError("augmentation requires non-silent negative noise")

    augmented = []
    for parent in positives:
        child_weight = parent.weight / 3.0
        augmented.append(
            _derived_window(
                parent,
                waveform=parent.waveform,
                weight=child_weight,
                noise_group=None,
            )
        )

        gain = float(rng.uniform(0.35, 1.0))
        gain_only = np.clip(
            np.asarray(parent.waveform, dtype=np.float32) * gain,
            -1.0,
            1.0,
        ).astype(np.float32)
        augmented.append(
            _derived_window(
                parent,
                waveform=gain_only,
                weight=child_weight,
                noise_group=None,
            )
        )

        noise_parent = noise_candidates[
            int(rng.integers(0, len(noise_candidates)))
        ]
        noisy_gain = float(rng.uniform(0.35, 1.0))
        snr_db = float(rng.uniform(8.0, 20.0))
        gained = np.clip(
            np.asarray(parent.waveform, dtype=np.float32) * noisy_gain,
            -1.0,
            1.0,
        ).astype(np.float32)
        augmented.append(
            _derived_window(
                parent,
                waveform=mix_at_snr(gained, noise_parent.waveform, snr_db),
                weight=child_weight,
                noise_group=noise_parent.source_group,
            )
        )
    return tuple(augmented)


def _weighted_standardization(values, weights):
    """Return weighted float32 mean and protected scale vectors."""

    try:
        matrix = np.asarray(values, dtype=np.float64)
        sample_weights = np.asarray(weights, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError("weighted standardization inputs are invalid") from exc
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or sample_weights.ndim != 1
        or len(sample_weights) != len(matrix)
        or not np.isfinite(matrix).all()
        or not np.isfinite(sample_weights).all()
        or np.any(sample_weights < 0)
    ):
        raise TrainingError("weighted standardization inputs are invalid")
    total_weight = float(np.sum(sample_weights, dtype=np.float64))
    if not np.isfinite(total_weight) or total_weight <= 0:
        raise TrainingError("weighted standardization weights must sum above zero")

    with np.errstate(over="ignore", invalid="ignore"):
        mean = np.sum(
            matrix * sample_weights[:, None],
            axis=0,
            dtype=np.float64,
        ) / total_weight
        centered = matrix - mean
        variance = np.sum(
            np.square(centered) * sample_weights[:, None],
            axis=0,
            dtype=np.float64,
        ) / total_weight
        scale = np.sqrt(np.maximum(variance, 0.0))
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise TrainingError("weighted standardization result must be finite")
    scale = np.where(
        scale < MIN_STANDARDIZATION_SCALE,
        1.0,
        scale,
    )
    return mean.astype(np.float32), scale.astype(np.float32)


def _load_sklearn():
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedGroupKFold
    except ImportError as exc:
        raise TrainingError(
            'alarm training requires `python -m pip install -e ".[train]"`'
        ) from exc
    return LogisticRegression, StratifiedGroupKFold


def _recording_label(recording: PreparedRecording) -> int:
    if recording.entry.label == ALARM:
        return 1
    if recording.entry.label == NOT_ALARM:
        return 0
    raise TrainingError(
        f"{recording.entry.relative_path}: unsupported corpus label "
        f"{recording.entry.label!r}"
    )


def _snapshot_sample_count(recording: PreparedRecording) -> int:
    counts = []
    for duration in (
        recording.duration_seconds,
        recording.entry.duration_seconds,
    ):
        sample_value = float(duration) * config.SAMPLE_RATE
        sample_count = int(round(sample_value))
        if (
            not np.isfinite(sample_value)
            or sample_count < config.WINDOW_SAMPLES
            or not np.isclose(
                sample_value,
                sample_count,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise TrainingError(
                f"{recording.entry.relative_path}: prepared audio duration "
                "does not describe a valid sample count"
            )
        counts.append(sample_count)
    if counts[0] != counts[1]:
        raise TrainingError(
            f"{recording.entry.relative_path}: prepared and inventoried "
            "audio durations disagree"
        )
    return counts[0]


def _fit_alarm_head(
    recordings,
    *,
    yamnet,
    seed,
    yamnet_model_sha256,
    class_map_sha256,
) -> AlarmHead:
    prepared = tuple(recordings)
    original_windows = tuple(
        WeightedWindow(
            waveform=window.waveform,
            label=_recording_label(recording),
            source_group=recording.entry.source_group,
            source_path=recording.entry.relative_path,
            start_sample=window.start_sample,
            weight=window.weight,
            augmentation_noise_group=None,
        )
        for recording in prepared
        for window in recording.windows
    )
    positives = tuple(item for item in original_windows if item.label == 1)
    negatives = tuple(item for item in original_windows if item.label == 0)
    augmented_positives = _augment_training_windows(
        positives,
        negatives,
        rng=np.random.default_rng(seed),
    )
    training_windows = augmented_positives + negatives

    embeddings = []
    labels = []
    weights = []
    for window in training_windows:
        try:
            _, embedding = yamnet.infer(window.waveform)
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except Exception as exc:
            raise TrainingError(
                f"{window.source_path}: YAMNet inference failed: {exc}"
            ) from exc
        if vector.shape != (1024,) or not np.isfinite(vector).all():
            raise TrainingError(
                f"{window.source_path}: YAMNet embedding must contain "
                "1024 finite values"
            )
        embeddings.append(vector)
        labels.append(window.label)
        weights.append(window.weight)

    mean, scale = _weighted_standardization(embeddings, weights)
    normalized = (
        np.asarray(embeddings, dtype=np.float32) - mean
    ) / scale
    LogisticRegression, _ = _load_sklearn()
    classifier = LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
    )
    try:
        classifier.fit(
            normalized,
            np.asarray(labels, dtype=np.int64),
            sample_weight=np.asarray(weights, dtype=np.float64),
        )
    except Exception as exc:
        raise TrainingError(f"could not fit alarm classifier: {exc}") from exc

    fitted_weights = np.asarray(classifier.coef_, dtype=np.float32)
    fitted_bias = np.asarray(classifier.intercept_, dtype=np.float64)
    if (
        fitted_weights.shape != (1, 1024)
        or fitted_bias.shape != (1,)
        or not np.isfinite(fitted_weights).all()
        or not np.isfinite(fitted_bias).all()
    ):
        raise TrainingError("alarm classifier produced invalid fitted parameters")
    return AlarmHead(
        label=config.ALARM_EVENT_LABEL,
        urgency=config.ALARM_EVENT_URGENCY,
        feature_dim=1024,
        mean=mean,
        scale=scale,
        weights=fitted_weights[0],
        bias=float(fitted_bias[0]),
        threshold=0.5,
        gate_count=config.ALARM_GATE_COUNT,
        gate_window=config.ALARM_GATE_WINDOW,
        yamnet_model_sha256=yamnet_model_sha256,
        class_map_sha256=class_map_sha256,
    )


def _score_recordings(
    recordings,
    head: AlarmHead,
    yamnet,
    *,
    audio_loader=None,
) -> tuple[ScoredRecording, ...]:
    """Score full runtime timelines; ``recording.windows`` stay fit-only."""

    loader = load_wav_16k_mono if audio_loader is None else audio_loader
    scored = []
    for recording in tuple(recordings):
        audio = _decoded_audio(recording.entry, loader)
        expected_sample_count = _snapshot_sample_count(recording)
        if len(audio) != expected_sample_count:
            raise TrainingError(
                f"{recording.entry.relative_path}: decoded audio sample count "
                "changed since corpus preparation"
            )
        if _decoded_digest(audio) != recording.entry.decoded_sha256:
            raise TrainingError(
                f"{recording.entry.relative_path}: decoded audio SHA-256 "
                "changed since corpus inventory"
            )
        starts = range(
            0,
            len(audio) - config.WINDOW_SAMPLES + 1,
            config.HOP_SAMPLES,
        )
        values = []
        for start in starts:
            try:
                _, embedding = yamnet.infer(
                    audio[start:start + config.WINDOW_SAMPLES]
                )
                values.append(head.score(embedding))
            except Exception as exc:
                raise TrainingError(
                    f"{recording.entry.relative_path}: "
                    f"could not score YAMNet embedding: {exc}"
                ) from exc
        scored.append(
            ScoredRecording(
                source_group=recording.entry.source_group,
                source_path=recording.entry.relative_path,
                label=_recording_label(recording),
                duration_seconds=recording.duration_seconds,
                scores=tuple(values),
            )
        )
    return tuple(scored)


def _build_content_audit(recordings, yamnet) -> tuple[dict, ...]:
    class_names = tuple(getattr(yamnet, "class_names", ()))
    if not class_names or not all(
        isinstance(name, str) and name for name in class_names
    ):
        raise TrainingError("YAMNet class names must be non-empty strings")

    audit = []
    for recording in tuple(recordings):
        strongest = np.full(len(class_names), -np.inf, dtype=np.float64)
        for window in recording.windows:
            try:
                scores, _ = yamnet.infer(window.waveform)
                raw_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
            except Exception as exc:
                raise TrainingError(
                    f"{recording.entry.relative_path}: "
                    f"could not audit YAMNet scores: {exc}"
                ) from exc
            if (
                raw_scores.shape != (len(class_names),)
                or not np.isfinite(raw_scores).all()
            ):
                raise TrainingError(
                    f"{recording.entry.relative_path}: YAMNet scores must "
                    f"contain {len(class_names)} finite values"
                )
            strongest = np.maximum(strongest, raw_scores)

        indices = sorted(
            range(len(class_names)),
            key=lambda index: (
                -float(strongest[index]),
                class_names[index],
                index,
            ),
        )[:5]
        audit.append({
            "path": recording.entry.relative_path,
            "source_group": recording.entry.source_group,
            "label": recording.entry.label,
            "top_classes": [
                {
                    "name": class_names[index],
                    "score": float(strongest[index]),
                }
                for index in indices
            ],
        })
    return tuple(audit)


def _make_grouped_folds(recordings, *, seed=0, folds=5) -> tuple[Fold, ...]:
    prepared = tuple(recordings)
    if not prepared or not all(
        isinstance(item, PreparedRecording) for item in prepared
    ):
        raise TrainingError("grouped folds require prepared recordings")
    if isinstance(folds, (bool, np.bool_)) or not isinstance(
        folds, (int, np.integer)
    ) or int(folds) < 2:
        raise TrainingError("folds must be an integer of at least two")
    folds = int(folds)

    labels = np.asarray([_recording_label(item) for item in prepared], dtype=np.int64)
    groups = np.asarray(
        [item.entry.source_group for item in prepared],
        dtype=np.str_,
    )
    group_labels = {}
    for group, label in zip(groups.tolist(), labels.tolist()):
        prior = group_labels.setdefault(group, label)
        if prior != label:
            raise TrainingError(
                f"source group {group!r} contains mixed corpus labels"
            )
    for label, description in ((1, ALARM), (0, NOT_ALARM)):
        group_count = sum(value == label for value in group_labels.values())
        if group_count < folds:
            raise TrainingError(
                f"{description} requires at least {folds} source groups; "
                f"found {group_count}"
            )

    _, StratifiedGroupKFold = _load_sklearn()
    splitter = StratifiedGroupKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    result = []
    features = np.zeros((len(prepared), 1), dtype=np.float32)
    for train_indices, validation_indices in splitter.split(
        features,
        labels,
        groups,
    ):
        train_recordings = tuple(prepared[int(index)] for index in train_indices)
        validation_recordings = tuple(
            prepared[int(index)] for index in validation_indices
        )
        train_groups = tuple(sorted({
            item.entry.source_group for item in train_recordings
        }))
        validation_groups = tuple(sorted({
            item.entry.source_group for item in validation_recordings
        }))
        if set(train_groups) & set(validation_groups):
            raise TrainingError("grouped fold leaked a validation source group")
        if {_recording_label(item) for item in train_recordings} != {0, 1}:
            raise TrainingError("grouped training fold must contain both labels")
        result.append(
            Fold(
                train_groups=train_groups,
                validation_groups=validation_groups,
                train_recordings=train_recordings,
                validation_recordings=validation_recordings,
            )
        )
    return tuple(result)


def _count_file_events(scores, threshold) -> int:
    try:
        decision_threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError("threshold must be finite") from exc
    if not np.isfinite(decision_threshold):
        raise TrainingError("threshold must be finite")

    gate = RollingEvidenceGate(
        config.ALARM_GATE_COUNT,
        config.ALARM_GATE_WINDOW,
    )
    detector = EventDetector(config.DEBOUNCE_SECONDS)
    emitted = 0
    for window_index, score in enumerate(scores):
        score = float(score)
        gated = gate.update(
            np.isfinite(score) and score >= decision_threshold
        )
        events = detector.update(
            [Observation(
                label=config.ALARM_EVENT_LABEL,
                confidence=score,
                above=gated,
                urgency=config.ALARM_EVENT_URGENCY,
                source="trained",
                consecutive=1,
            )],
            now=(
                window_index
                * config.HOP_SAMPLES
                / config.SAMPLE_RATE
            ),
        )
        emitted += len(events)
    return emitted


def _evaluate_threshold(scored_recordings, threshold) -> EvaluationMetrics:
    recordings = tuple(scored_recordings)
    try:
        decision_threshold = float(threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrainingError("threshold must be finite") from exc
    if not np.isfinite(decision_threshold):
        raise TrainingError("threshold must be finite")

    group_labels = {}
    group_triggered = {}
    files = []
    false_events = 0
    negative_seconds = 0.0
    evaluated_windows = 0
    for recording in recordings:
        if not isinstance(recording, ScoredRecording):
            raise TrainingError("threshold evaluation requires scored recordings")
        if recording.label not in (0, 1):
            raise TrainingError("scored recording label must be 0 or 1")
        prior = group_labels.setdefault(recording.source_group, recording.label)
        if prior != recording.label:
            raise TrainingError(
                f"source group {recording.source_group!r} contains mixed labels"
            )

        emitted = _count_file_events(recording.scores, decision_threshold)
        file_evaluated_windows = len(recording.scores)
        evaluated_windows += file_evaluated_windows
        triggered = emitted > 0
        group_triggered[recording.source_group] = (
            group_triggered.get(recording.source_group, False) or triggered
        )
        if recording.label == 0:
            false_events += emitted
            negative_seconds += float(recording.duration_seconds)
        files.append({
            "path": recording.source_path,
            "source_group": recording.source_group,
            "label": ALARM if recording.label == 1 else NOT_ALARM,
            "triggered": triggered,
            "events": emitted,
            "duration_seconds": float(recording.duration_seconds),
            "evaluated_windows": file_evaluated_windows,
        })

    positive_groups = {
        group for group, label in group_labels.items() if label == 1
    }
    negative_groups = {
        group for group, label in group_labels.items() if label == 0
    }
    negative_minutes = negative_seconds / 60.0
    false_rate = (
        false_events / negative_minutes if negative_minutes > 0 else 0.0
    )
    return EvaluationMetrics(
        positive_groups_total=len(positive_groups),
        positive_groups_triggered=sum(
            bool(group_triggered.get(group)) for group in positive_groups
        ),
        negative_groups_total=len(negative_groups),
        negative_groups_triggered=sum(
            bool(group_triggered.get(group)) for group in negative_groups
        ),
        false_events=false_events,
        negative_audio_minutes=negative_minutes,
        false_triggers_per_minute=false_rate,
        evaluated_windows=evaluated_windows,
        files=tuple(files),
    )


def _metrics_meet_ceilings(metrics: EvaluationMetrics) -> bool:
    return (
        metrics.positive_groups_triggered == metrics.positive_groups_total
        and metrics.negative_groups_triggered
        <= MAX_NEGATIVE_GROUP_FRACTION * metrics.negative_groups_total
        and metrics.false_triggers_per_minute <= MAX_FALSE_TRIGGERS_PER_MINUTE
    )


def _select_threshold(scored_recordings) -> float:
    recordings = tuple(scored_recordings)
    finite_scores = {
        float(score)
        for recording in recordings
        for score in recording.scores
        if np.isfinite(float(score))
    }
    candidates = sorted({0.0, 1.0, *finite_scores}, reverse=True)
    for candidate in candidates:
        if _metrics_meet_ceilings(
            _evaluate_threshold(recordings, candidate)
        ):
            return candidate
    raise TrainingError(
        "no decision threshold satisfies positive recall and false-alert ceilings"
    )


def _select_positive_recall_threshold(scored_recordings) -> float:
    recordings = tuple(scored_recordings)
    finite_scores = {
        float(score)
        for recording in recordings
        for score in recording.scores
        if np.isfinite(float(score))
    }
    candidates = sorted({0.0, 1.0, *finite_scores}, reverse=True)
    for candidate in candidates:
        metrics = _evaluate_threshold(recordings, candidate)
        if (
            metrics.positive_groups_total > 0
            and metrics.positive_groups_triggered
            == metrics.positive_groups_total
        ):
            return candidate
    raise TrainingError(
        "no decision threshold achieves complete positive-group recall"
    )


def _metrics_payload(metrics: EvaluationMetrics, scope: str) -> dict:
    return {
        "evaluation_scope": scope,
        "positive_groups_total": metrics.positive_groups_total,
        "positive_groups_triggered": metrics.positive_groups_triggered,
        "negative_groups_total": metrics.negative_groups_total,
        "negative_groups_triggered": metrics.negative_groups_triggered,
        "false_events": metrics.false_events,
        "negative_audio_minutes": metrics.negative_audio_minutes,
        "false_triggers_per_minute": metrics.false_triggers_per_minute,
        "evaluated_windows": metrics.evaluated_windows,
        "files": [dict(item) for item in metrics.files],
    }


def _validated_seed(seed) -> int:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
        or int(seed) > np.iinfo(np.uint32).max
    ):
        raise TrainingError("seed must be an integer from 0 through 4294967295")
    return int(seed)


_ABSENT = object()


def _snapshot_file(path: Path):
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return _ABSENT


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    part: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".part",
            delete=False,
        ) as output:
            part = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(part, path)
        part = None
    finally:
        if part is not None:
            try:
                part.unlink()
            except OSError:
                pass


def _restore_snapshot(path: Path, snapshot) -> None:
    if snapshot is _ABSENT:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    _write_bytes_atomic(path, snapshot)


def _failed_report_path(report_path: Path) -> Path:
    return report_path.with_name(report_path.stem + "_failed_report.json")


def _validated_training_paths(output_path, report_path) -> tuple[Path, Path, Path]:
    output = Path(output_path)
    report = Path(report_path)
    failed_report = _failed_report_path(report)
    try:
        canonical_paths = {
            path.resolve(strict=False)
            for path in (output, report, failed_report)
        }
    except (OSError, RuntimeError) as exc:
        raise TrainingError(f"could not resolve alarm output paths: {exc}") from exc
    if len(canonical_paths) != 3:
        raise TrainingError(
            "alarm model, report, and failed report paths must be distinct"
        )
    return output, report, failed_report


def _corpus_metadata(
    inventory: CorpusInventory,
    recordings,
) -> dict:
    prepared = tuple(recordings)
    if tuple(item.entry for item in prepared) != tuple(inventory.entries):
        raise TrainingError(
            "prepared recordings must preserve deterministic corpus order"
        )
    records = []
    for recording in prepared:
        sample_value = (
            float(recording.entry.duration_seconds) * config.SAMPLE_RATE
        )
        sample_count = int(round(sample_value))
        if (
            not np.isfinite(sample_value)
            or sample_count <= 0
            or not np.isclose(
                sample_value,
                sample_count,
                rtol=0.0,
                atol=1e-6,
            )
        ):
            raise TrainingError(
                f"{recording.entry.relative_path}: decoded sample count "
                "must be a positive integer"
            )
        records.append({
            "path": recording.entry.relative_path,
            "decoded_sha256": recording.entry.decoded_sha256,
            "label": recording.entry.label,
            "source_group": recording.entry.source_group,
            "retained_windows": len(recording.windows),
            "sample_count": sample_count,
        })
    return {
        "recordings": records,
        "counts": {
            "recordings": len(records),
            "retained_windows": sum(
                item["retained_windows"] for item in records
            ),
            "samples": sum(item["sample_count"] for item in records),
        },
        "warnings": list(inventory.warnings),
    }


def _path_hash_projection(corpus_metadata: dict) -> list[dict]:
    return [
        {
            "path": item["path"],
            "decoded_sha256": item["decoded_sha256"],
        }
        for item in corpus_metadata["recordings"]
    ]


def train_alarm(
    data_dir,
    output_path,
    report_path,
    *,
    seed=0,
    yamnet=None,
    yamnet_model_path=config.MODEL_PATH,
    class_map_path=config.CLASS_MAP_PATH,
) -> TrainingReport:
    """Cross-validate, fit, report, and atomically install one alarm head."""

    output, report, failed_report = _validated_training_paths(
        output_path,
        report_path,
    )
    output_snapshot = _ABSENT
    report_snapshot = _ABSENT
    snapshots_ready = False
    report_install_attempted = False
    model_install_attempted = False
    try:
        training_seed = _validated_seed(seed)
        output_snapshot = _snapshot_file(output)
        report_snapshot = _snapshot_file(report)
        snapshots_ready = True

        corpus = inventory_corpus(data_dir)
        recordings = _prepare_recordings(corpus)
        model = yamnet
        if model is None:
            model = YamNet(
                model_path=yamnet_model_path,
                class_map_path=class_map_path,
            )
        yamnet_digest = sha256_file(Path(yamnet_model_path))
        class_map_digest = sha256_file(Path(class_map_path))
        content_audit = _build_content_audit(recordings, model)

        folds = _make_grouped_folds(recordings, seed=training_seed, folds=5)
        oof_scored = []
        for fold in folds:
            fold_head = _fit_alarm_head(
                fold.train_recordings,
                yamnet=model,
                seed=training_seed,
                yamnet_model_sha256=yamnet_digest,
                class_map_sha256=class_map_digest,
            )
            oof_scored.extend(
                _score_recordings(
                    fold.validation_recordings,
                    fold_head,
                    model,
                )
            )
        oof_scored = tuple(oof_scored)
        if len(oof_scored) != len(recordings):
            raise TrainingError(
                "grouped out-of-fold scoring must score every recording once"
            )
        oof_threshold = _select_threshold(oof_scored)

        final_head = _fit_alarm_head(
            recordings,
            yamnet=model,
            seed=training_seed,
            yamnet_model_sha256=yamnet_digest,
            class_map_sha256=class_map_digest,
        )
        final_scored = _score_recordings(recordings, final_head, model)
        final_threshold = _select_positive_recall_threshold(final_scored)
        deployment_threshold = min(oof_threshold, final_threshold)
        if not 0.0 < deployment_threshold < 1.0:
            raise TrainingError(
                "deployment threshold must be strictly between zero and one"
            )

        oof_metrics = _evaluate_threshold(oof_scored, deployment_threshold)
        in_sample_metrics = _evaluate_threshold(
            final_scored,
            deployment_threshold,
        )
        if not _metrics_meet_ceilings(oof_metrics):
            raise TrainingError(
                "deployed threshold violates out-of-fold evaluation ceilings"
            )
        if not _metrics_meet_ceilings(in_sample_metrics):
            raise TrainingError(
                "deployed threshold violates final-head evaluation ceilings"
            )
        final_head = replace(final_head, threshold=deployment_threshold)

        payload = {
            "schema": "earshot.fire_smoke_alarm_training_report",
            "schema_version": 1,
            "status": "ok",
            "seed": training_seed,
            "folds": [
                {
                    "train_groups": list(fold.train_groups),
                    "validation_groups": list(fold.validation_groups),
                }
                for fold in folds
            ],
            "corpus": _corpus_metadata(corpus, recordings),
            "thresholds": {
                "out_of_fold": oof_threshold,
                "final_positive_recall": final_threshold,
                "deployment": deployment_threshold,
            },
            "metrics": {
                "out_of_fold": _metrics_payload(
                    oof_metrics,
                    "out_of_fold",
                ),
                "final_model": _metrics_payload(
                    in_sample_metrics,
                    "in_sample",
                ),
            },
            "content_audit": list(content_audit),
        }

        report_install_attempted = True
        _write_json_atomic(report, payload)
        model_install_attempted = True
        save_alarm_head(output, final_head)
    except Exception as exc:
        rollback_errors = []
        if snapshots_ready and model_install_attempted:
            try:
                _restore_snapshot(output, output_snapshot)
            except Exception as rollback_exc:
                rollback_errors.append(f"model rollback failed: {rollback_exc}")
        if snapshots_ready and report_install_attempted:
            try:
                _restore_snapshot(report, report_snapshot)
            except Exception as rollback_exc:
                rollback_errors.append(f"report rollback failed: {rollback_exc}")
        for message in rollback_errors:
            if hasattr(exc, "add_note"):
                exc.add_note(message)
        diagnostic = {
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            _write_json_atomic(failed_report, diagnostic)
        except Exception as diagnostic_exc:
            if hasattr(exc, "add_note"):
                exc.add_note(
                    f"failed to write diagnostic report: {diagnostic_exc}"
                )
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(str(exc)) from exc

    return TrainingReport(
        artifact_path=output,
        report_path=report,
        deployment_threshold=deployment_threshold,
        oof_metrics=oof_metrics,
        in_sample_metrics=in_sample_metrics,
        payload=payload,
    )


def _companion_report_path(alarm_model_path) -> Path:
    model_path = Path(alarm_model_path)
    if model_path == Path(config.ALARM_MODEL_PATH):
        return Path(config.ALARM_REPORT_PATH)
    return model_path.with_name("fire_smoke_alarm_report.json")


def _reported_path_hash_projection(report_path: Path):
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        recordings = payload["corpus"]["recordings"]
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError):
        return None
    if (
        payload.get("schema")
        != "earshot.fire_smoke_alarm_training_report"
        or type(payload.get("schema_version")) is not int
        or payload["schema_version"] != 1
        or payload.get("status") != "ok"
    ):
        return None
    if not isinstance(recordings, list):
        return None
    projection = []
    for item in recordings:
        if not isinstance(item, dict):
            return None
        path = item.get("path")
        digest = item.get("decoded_sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        projection.append({"path": path, "decoded_sha256": digest})
    return projection


def evaluate_alarm(
    data_dir,
    alarm_model_path,
    *,
    yamnet=None,
    yamnet_model_path=config.MODEL_PATH,
    class_map_path=config.CLASS_MAP_PATH,
) -> EvaluationReport:
    """Evaluate one installed head; label matching corpus hashes in-sample."""

    try:
        corpus = inventory_corpus(data_dir)
        recordings = _prepare_recordings(corpus)
        head = load_alarm_head(
            alarm_model_path,
            yamnet_model_path=yamnet_model_path,
            class_map_path=class_map_path,
        )
        model = yamnet
        if model is None:
            model = YamNet(
                model_path=yamnet_model_path,
                class_map_path=class_map_path,
            )
        scored = _score_recordings(recordings, head, model)
        metrics = _evaluate_threshold(scored, head.threshold)
        content_audit = _build_content_audit(recordings, model)
        corpus_metadata = _corpus_metadata(corpus, recordings)
    except Exception as exc:
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(str(exc)) from exc

    recorded_projection = _reported_path_hash_projection(
        _companion_report_path(alarm_model_path)
    )
    evaluation_scope = (
        "in_sample"
        if recorded_projection == _path_hash_projection(corpus_metadata)
        else "external_corpus"
    )
    payload = {
        "schema": "earshot.fire_smoke_alarm_evaluation_report",
        "schema_version": 1,
        "evaluation_scope": evaluation_scope,
        "metrics": _metrics_payload(metrics, evaluation_scope),
        "corpus": corpus_metadata,
        "content_audit": list(content_audit),
    }
    return EvaluationReport(
        metrics=metrics,
        evaluation_scope=evaluation_scope,
        payload=payload,
    )
