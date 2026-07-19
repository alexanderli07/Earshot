"""Runtime-only trained alarm head artifact and rolling evidence gate."""

from __future__ import annotations

import os
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config
from .artifacts import sha256_file


SCHEMA = "earshot.fire_smoke_alarm_head"
SCHEMA_VERSION = 1
FEATURE_DIM = 1024
ARTIFACT_KEYS = frozenset({
    "schema",
    "schema_version",
    "label",
    "urgency",
    "feature_dim",
    "mean",
    "scale",
    "weights",
    "bias",
    "threshold",
    "gate_count",
    "gate_window",
    "yamnet_model_sha256",
    "class_map_sha256",
})

_VECTOR_KEYS = ("mean", "scale", "weights")
_STRING_KEYS = (
    "schema",
    "label",
    "urgency",
    "yamnet_model_sha256",
    "class_map_sha256",
)
_INTEGER_KEYS = ("schema_version", "feature_dim", "gate_count", "gate_window")
_FLOAT_KEYS = ("bias", "threshold")
_HEX_DIGITS = frozenset("0123456789abcdef")


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
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AlarmModelError(
                "embedding must contain 1024 finite values"
            ) from exc
        if vector.shape != (self.feature_dim,) or not np.isfinite(vector).all():
            raise AlarmModelError("embedding must contain 1024 finite values")

        with np.errstate(over="ignore", invalid="ignore"):
            logit = float(
                np.dot((vector - self.mean) / self.scale, self.weights)
                + self.bias
            )
        if logit >= 0:
            return float(1.0 / (1.0 + np.exp(-logit)))
        exponential = np.exp(logit)
        return float(exponential / (1.0 + exponential))


class RollingEvidenceGate:
    def __init__(self, required_count: int, window_size: int):
        if not _is_integer(required_count) or not _is_integer(window_size):
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


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _is_number(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )


def _validate_digest(value, *, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise AlarmModelError(f"{field} must be a lowercase SHA-256 digest")


def _validate_head(head: AlarmHead) -> None:
    if not isinstance(head, AlarmHead):
        raise AlarmModelError("alarm head must be an AlarmHead dataclass")
    if not isinstance(head.label, str) or head.label != config.ALARM_EVENT_LABEL:
        raise AlarmModelError(
            f"alarm head label must be {config.ALARM_EVENT_LABEL!r}"
        )
    if (
        not isinstance(head.urgency, str)
        or head.urgency != config.ALARM_EVENT_URGENCY
    ):
        raise AlarmModelError(
            f"alarm head urgency must be {config.ALARM_EVENT_URGENCY!r}"
        )
    if not _is_integer(head.feature_dim) or int(head.feature_dim) != FEATURE_DIM:
        raise AlarmModelError(f"alarm head feature_dim must be {FEATURE_DIM}")

    for field in _VECTOR_KEYS:
        vector = getattr(head, field)
        if not isinstance(vector, np.ndarray):
            raise AlarmModelError(f"{field} must be a float32 NumPy vector")
        if vector.dtype != np.dtype(np.float32) or vector.shape != (FEATURE_DIM,):
            raise AlarmModelError(
                f"{field} must have dtype float32 and shape ({FEATURE_DIM},)"
            )
        if not np.isfinite(vector).all():
            raise AlarmModelError(f"{field} must contain only finite values")
    if not np.greater(head.scale, 0.0).all():
        raise AlarmModelError("scale must contain only positive values")

    if not _is_number(head.bias) or not np.isfinite(float(head.bias)):
        raise AlarmModelError("bias must be finite")
    if (
        not _is_number(head.threshold)
        or not np.isfinite(float(head.threshold))
        or not 0.0 < float(head.threshold) < 1.0
    ):
        raise AlarmModelError("threshold must be finite and between zero and one")

    if (
        not _is_integer(head.gate_count)
        or int(head.gate_count) != config.ALARM_GATE_COUNT
    ):
        raise AlarmModelError(
            f"gate_count must be {config.ALARM_GATE_COUNT}"
        )
    if (
        not _is_integer(head.gate_window)
        or int(head.gate_window) != config.ALARM_GATE_WINDOW
    ):
        raise AlarmModelError(
            f"gate_window must be {config.ALARM_GATE_WINDOW}"
        )

    _validate_digest(head.yamnet_model_sha256, field="yamnet_model_sha256")
    _validate_digest(head.class_map_sha256, field="class_map_sha256")


def _artifact_payload(head: AlarmHead) -> dict[str, np.ndarray]:
    return {
        "schema": np.array(SCHEMA, dtype=np.str_),
        "schema_version": np.array(SCHEMA_VERSION, dtype=np.int64),
        "label": np.array(head.label, dtype=np.str_),
        "urgency": np.array(head.urgency, dtype=np.str_),
        "feature_dim": np.array(head.feature_dim, dtype=np.int64),
        "mean": np.array(head.mean, dtype=np.float32, copy=True),
        "scale": np.array(head.scale, dtype=np.float32, copy=True),
        "weights": np.array(head.weights, dtype=np.float32, copy=True),
        "bias": np.array(head.bias, dtype=np.float64),
        "threshold": np.array(head.threshold, dtype=np.float64),
        "gate_count": np.array(head.gate_count, dtype=np.int64),
        "gate_window": np.array(head.gate_window, dtype=np.int64),
        "yamnet_model_sha256": np.array(
            head.yamnet_model_sha256, dtype=np.str_
        ),
        "class_map_sha256": np.array(head.class_map_sha256, dtype=np.str_),
    }


def save_alarm_head(path, head: AlarmHead) -> None:
    """Validate and atomically save a trained alarm head artifact."""

    _validate_head(head)
    destination = Path(path)
    part: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".part",
            delete=False,
        ) as output:
            part = Path(output.name)
            np.savez(output, **_artifact_payload(head))
            output.flush()
            os.fsync(output.fileno())
        os.replace(part, destination)
        part = None
    except Exception as exc:
        raise AlarmModelError(
            f"Could not save alarm head to {destination}: {exc}"
        ) from exc
    finally:
        if part is not None:
            try:
                part.unlink()
            except OSError:
                pass


def _copy_artifact_values(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys != ARTIFACT_KEYS or len(archive.files) != len(ARTIFACT_KEYS):
                missing = sorted(ARTIFACT_KEYS - keys)
                extra = sorted(keys - ARTIFACT_KEYS)
                raise AlarmModelError(
                    f"alarm artifact keys are invalid; missing={missing}, extra={extra}"
                )
            return {key: archive[key].copy() for key in ARTIFACT_KEYS}
    except AlarmModelError:
        raise
    except Exception as exc:
        raise AlarmModelError(f"Could not read alarm artifact {path}: {exc}") from exc


def _unicode_scalar(values, key: str) -> str:
    value = values[key]
    if value.shape != () or value.dtype.kind != "U":
        raise AlarmModelError(f"{key} must be a Unicode scalar")
    return str(value.item())


def _integer_scalar(values, key: str) -> int:
    value = values[key]
    if value.shape != () or value.dtype != np.dtype(np.int64):
        raise AlarmModelError(f"{key} must be an int64 scalar")
    return int(value.item())


def _float_scalar(values, key: str) -> float:
    value = values[key]
    if value.shape != () or value.dtype != np.dtype(np.float64):
        raise AlarmModelError(f"{key} must be a float64 scalar")
    return float(value.item())


def _vector(values, key: str) -> np.ndarray:
    value = values[key]
    if value.shape != (FEATURE_DIM,) or value.dtype != np.dtype(np.float32):
        raise AlarmModelError(
            f"{key} must have dtype float32 and shape ({FEATURE_DIM},)"
        )
    return value


def _head_from_values(values: dict[str, np.ndarray]) -> AlarmHead:
    strings = {key: _unicode_scalar(values, key) for key in _STRING_KEYS}
    integers = {key: _integer_scalar(values, key) for key in _INTEGER_KEYS}
    floats = {key: _float_scalar(values, key) for key in _FLOAT_KEYS}
    if strings["schema"] != SCHEMA:
        raise AlarmModelError(f"unsupported alarm artifact schema {strings['schema']!r}")
    if integers["schema_version"] != SCHEMA_VERSION:
        raise AlarmModelError(
            f"unsupported alarm artifact schema version {integers['schema_version']}"
        )

    head = AlarmHead(
        label=strings["label"],
        urgency=strings["urgency"],
        feature_dim=integers["feature_dim"],
        mean=_vector(values, "mean"),
        scale=_vector(values, "scale"),
        weights=_vector(values, "weights"),
        bias=floats["bias"],
        threshold=floats["threshold"],
        gate_count=integers["gate_count"],
        gate_window=integers["gate_window"],
        yamnet_model_sha256=strings["yamnet_model_sha256"],
        class_map_sha256=strings["class_map_sha256"],
    )
    _validate_head(head)
    return head


def _current_digest(path, *, description: str) -> str:
    try:
        return sha256_file(Path(path))
    except Exception as exc:
        raise AlarmModelError(
            f"Could not verify {description} digest at {path}: {exc}"
        ) from exc


def load_alarm_head(path, *, yamnet_model_path, class_map_path) -> AlarmHead:
    """Load and verify an alarm head against the active YAMNet artifacts."""

    artifact_path = Path(path)
    head = _head_from_values(_copy_artifact_values(artifact_path))

    model_digest = _current_digest(
        yamnet_model_path, description="YAMNet model"
    )
    if model_digest != head.yamnet_model_sha256:
        raise AlarmModelError(
            "YAMNet model digest mismatch: alarm head is incompatible"
        )
    map_digest = _current_digest(class_map_path, description="class map")
    if map_digest != head.class_map_sha256:
        raise AlarmModelError(
            "class map digest mismatch: alarm head is incompatible"
        )

    for vector in (head.mean, head.scale, head.weights):
        vector.setflags(write=False)
    return head


def load_optional_alarm_head(
    path,
    *,
    yamnet_model_path,
    class_map_path,
) -> AlarmHead | None:
    """Load an optional alarm head, ignoring only a genuinely absent artifact."""

    if path is None:
        return None
    artifact_path = Path(path)
    if not artifact_path.exists():
        return None
    return load_alarm_head(
        artifact_path,
        yamnet_model_path=yamnet_model_path,
        class_map_path=class_map_path,
    )
