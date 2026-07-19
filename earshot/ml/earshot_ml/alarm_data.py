"""Validated local corpus inventory for the trained alarm detector."""

import hashlib
import json
import os
import shutil
import tempfile
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from . import config
from .pipeline import AudioFileError, load_wav_16k_mono, record


ALARM = "alarm"
NOT_ALARM = "not_alarm"
VALID_LABELS = frozenset({ALARM, NOT_ALARM})
MANIFEST_VERSION = 1
_EXCEPTION_ADD_NOTE = getattr(BaseException, "add_note", None)


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


@dataclass(frozen=True)
class _OwnedTemporaryFile:
    path: Path
    identity: tuple[int, int]


@dataclass(frozen=True)
class _InstallToken:
    destination: Path
    staging: _OwnedTemporaryFile


@dataclass(frozen=True)
class _ClassDirectoryToken:
    root: Path
    label: str
    path: Path
    identity: tuple[int, int]


def _decoded_digest(audio: np.ndarray) -> str:
    canonical = np.asarray(audio, dtype="<f4")
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def _relative_path_sort_key(relative_path: str) -> tuple[str, str]:
    return relative_path.casefold(), relative_path


def _validated_segments(value, *, duration: float, relative_path: str):
    if value is None:
        return ()
    if not isinstance(value, list):
        raise AlarmDataError(f"{relative_path}: segments must be a list")
    result = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise AlarmDataError(
                f"{relative_path}: each segment must be [start, end]"
            )
        try:
            start, end = (float(pair[0]), float(pair[1]))
        except (TypeError, ValueError, OverflowError) as exc:
            raise AlarmDataError(
                f"{relative_path}: each segment must be [start, end]"
            ) from exc
        if (
            not np.isfinite([start, end]).all()
            or start < 0
            or end <= start
            or end > duration
        ):
            raise AlarmDataError(
                f"{relative_path}: invalid segment [{start}, {end}]"
            )
        result.append((start, end))
    return tuple(result)


def _normalized_manifest_path(root: Path, value, *, entry_number: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlarmDataError(
            f"manifest entry {entry_number}: path must be a non-empty relative path"
        )

    raw_path = value.strip()
    windows_path = PureWindowsPath(raw_path)
    posix_path = PurePosixPath(raw_path.replace("\\", "/"))
    if (
        windows_path.is_absolute()
        or posix_path.is_absolute()
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise AlarmDataError(
            f"manifest entry {entry_number}: path must be a relative path "
            "inside the data directory"
        )

    try:
        resolved = root.joinpath(*posix_path.parts).resolve()
        relative_path = resolved.relative_to(root).as_posix()
    except (OSError, ValueError) as exc:
        raise AlarmDataError(
            f"manifest entry {entry_number}: path must be a relative path "
            "inside the data directory"
        ) from exc
    return relative_path


def load_manifest(data_dir):
    """Load and structurally validate ``manifest.json`` when present.

    The return value maps normalized corpus-relative paths to their manifest
    metadata. An absent manifest is represented by an empty mapping so callers
    can still inventory manually placed WAV files.
    """

    root = _collection_root(data_dir)
    _validate_existing_class_directories(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    if not manifest_path.is_file():
        raise AlarmDataError(f"{manifest_path}: manifest must be a JSON file")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AlarmDataError(
            f"{manifest_path}: could not read manifest JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise AlarmDataError(f"{manifest_path}: manifest must be a JSON object")
    version = payload.get("version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != MANIFEST_VERSION
    ):
        raise AlarmDataError(
            f"{manifest_path}: manifest version must be {MANIFEST_VERSION}"
        )
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise AlarmDataError(f"{manifest_path}: entries must be a list")

    entries = {}
    seen_paths = set()
    for entry_number, raw_entry in enumerate(raw_entries, start=1):
        if not isinstance(raw_entry, dict):
            raise AlarmDataError(
                f"manifest entry {entry_number} must be a JSON object"
            )
        relative_path = _normalized_manifest_path(
            root,
            raw_entry.get("path"),
            entry_number=entry_number,
        )
        path_key = relative_path.casefold()
        if path_key in seen_paths:
            raise AlarmDataError(
                f"manifest contains duplicate path: {relative_path}"
            )
        seen_paths.add(path_key)

        label = raw_entry.get("label")
        if not isinstance(label, str) or label not in VALID_LABELS:
            raise AlarmDataError(
                f"{relative_path}: label must be one of {sorted(VALID_LABELS)}"
            )
        source_group = raw_entry.get("source_group")
        if not isinstance(source_group, str) or not source_group.strip():
            raise AlarmDataError(
                f"{relative_path}: source_group must be a non-empty string"
            )

        entry = dict(raw_entry)
        entry["path"] = relative_path
        entry["label"] = label
        entry["source_group"] = source_group.strip()
        entries[relative_path] = entry
    return entries


def _ensure_parent_directory(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AlarmDataError(
            f"could not create parent directory {path}: {exc}"
        ) from exc
    try:
        is_directory = path.is_dir()
    except OSError as exc:
        raise AlarmDataError(
            f"could not inspect parent directory {path}: {exc}"
        ) from exc
    if not is_directory:
        raise AlarmDataError(f"parent path is not a directory: {path}")


def _identity_from_stat(stat_result) -> tuple[int, int]:
    return int(stat_result.st_dev), int(stat_result.st_ino)


def _path_identity(path: Path) -> tuple[int, int]:
    return _identity_from_stat(os.stat(path, follow_symlinks=False))


def _attach_failure_note(error: BaseException, message: str) -> None:
    if callable(_EXCEPTION_ADD_NOTE):
        _EXCEPTION_ADD_NOTE(error, message)
        return

    primary_message = str(error)
    if primary_message:
        error.args = (f"{primary_message}\n{message}",)
    else:
        error.args = (message,)


def _owned_temporary_unlink_error(owned: _OwnedTemporaryFile):
    try:
        current_identity = _path_identity(owned.path)
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc
    if current_identity != owned.identity:
        return AlarmDataError(
            f"refusing to clean temporary path whose identity changed: "
            f"{owned.path}"
        )
    try:
        owned.path.unlink()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return exc
    return None


def _stage_bytes(path: Path, writer) -> _OwnedTemporaryFile:
    _ensure_parent_directory(path.parent)
    try:
        descriptor, raw_staging = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
        )
    except OSError as exc:
        raise AlarmDataError(
            f"could not create staging file for {path}: {exc}"
        ) from exc

    staging = Path(raw_staging)
    try:
        owned = _OwnedTemporaryFile(
            path=staging,
            identity=_identity_from_stat(os.fstat(descriptor)),
        )
    except OSError as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            staging.unlink()
        except OSError:
            pass
        raise AlarmDataError(
            f"could not identify staging file for {path}: {exc}"
        ) from exc

    output = None
    try:
        output = os.fdopen(descriptor, "wb")
        with output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
    except BaseException as exc:
        if output is None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        cleanup_error = _owned_temporary_unlink_error(owned)
        if cleanup_error is not None:
            _attach_failure_note(
                exc,
                f"staging cleanup failed for {staging}: {cleanup_error}",
            )
        if isinstance(exc, AlarmDataError):
            raise
        if isinstance(exc, Exception):
            error = AlarmDataError(f"could not stage {path}: {exc}")
            if cleanup_error is not None:
                _attach_failure_note(error, str(cleanup_error))
            raise error from exc
        raise
    return owned


def _atomic_replace_bytes(path: Path, writer) -> None:
    """Install bytes with an explicit overwrite transaction."""

    path = Path(path)
    staging = _stage_bytes(path, writer)
    try:
        os.replace(staging.path, path)
    except OSError as exc:
        cleanup_error = _owned_temporary_unlink_error(staging)
        message = f"could not install {path}: {exc}"
        if cleanup_error is not None:
            message += (
                f"; staging cleanup failed for {staging.path}: {cleanup_error}"
            )
        raise AlarmDataError(message) from exc


def _atomic_create_bytes(
    path: Path,
    writer,
    *,
    class_token: _ClassDirectoryToken | None = None,
) -> _InstallToken:
    """Atomically create ``path`` without replacing any existing entry."""

    path = Path(path)
    staging = _stage_bytes(path, writer)
    try:
        os.link(staging.path, path)
    except FileExistsError as exc:
        cleanup_error = _owned_temporary_unlink_error(staging)
        message = f"refusing to overwrite existing corpus file: {path}"
        if cleanup_error is not None:
            message += (
                f"; staging cleanup failed for {staging.path}: {cleanup_error}"
            )
        raise AlarmDataError(message) from exc
    except OSError as exc:
        cleanup_error = _owned_temporary_unlink_error(staging)
        message = f"could not install new corpus file {path}: {exc}"
        if cleanup_error is not None:
            message += (
                f"; staging cleanup failed for {staging.path}: {cleanup_error}"
            )
        raise AlarmDataError(message) from exc
    except BaseException as exc:
        if class_token is not None:
            try:
                _verify_class_directory_token(class_token)
            except AlarmDataError as parent_error:
                _attach_failure_note(
                    exc,
                    f"class directory revalidation failed before rollback: "
                    f"{parent_error}",
                )
        rollback_failures = _rollback_install_token(
            _InstallToken(destination=path, staging=staging)
        )
        if rollback_failures:
            _attach_failure_note(
                exc,
                "post-link rollback failed: " + "; ".join(rollback_failures),
            )
        raise
    return _InstallToken(destination=path, staging=staging)


def _finalize_install_token(token: _InstallToken):
    cleanup_error = _owned_temporary_unlink_error(token.staging)
    if cleanup_error is not None:
        return (
            f"staging cleanup failed for {token.staging.path}: "
            f"{cleanup_error}"
        )
    return None


def _rollback_install_token(token: _InstallToken) -> list[str]:
    failures = []
    staging_is_owned = False
    try:
        staging_is_owned = (
            _path_identity(token.staging.path) == token.staging.identity
        )
    except FileNotFoundError:
        pass
    except OSError as exc:
        failures.append(
            f"could not inspect retained staging token "
            f"{token.staging.path}: {exc}"
        )

    destination_present = True
    destination_identity = None
    try:
        destination_identity = _path_identity(token.destination)
    except FileNotFoundError:
        destination_present = False
    except OSError as exc:
        failures.append(
            f"could not inspect rollback destination {token.destination}: {exc}"
        )

    if destination_present:
        safe_to_delete = False
        if staging_is_owned and destination_identity == token.staging.identity:
            try:
                safe_to_delete = os.path.samefile(
                    token.staging.path,
                    token.destination,
                )
            except OSError as exc:
                failures.append(
                    f"could not compare rollback identity for "
                    f"{token.destination}: {exc}"
                )
        if safe_to_delete:
            try:
                token.destination.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                failures.append(f"{token.destination}: {exc}")
        else:
            failures.append(
                f"could not safely delete {token.destination}: current "
                "destination is not the file installed by this transaction"
            )

    cleanup_error = _owned_temporary_unlink_error(token.staging)
    if cleanup_error is not None:
        failures.append(
            f"staging cleanup failed for {token.staging.path}: {cleanup_error}"
        )
    return failures


def _recording_samples(audio) -> np.ndarray:
    try:
        samples = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlarmDataError(
            "recording must be finite mono audio at least one model window long"
        ) from exc
    if (
        samples.ndim != 1
        or samples.size < config.WINDOW_SAMPLES
        or not np.isfinite(samples).all()
    ):
        raise AlarmDataError(
            "recording must be finite mono audio at least one model window long"
        )
    return samples


def _pcm16_samples(samples: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(samples, -1, 1) * 32767).astype("<i2")


def _pcm16_decoded_digest(samples: np.ndarray) -> str:
    decoded = _pcm16_samples(samples).astype(np.float32) / 32768.0
    return _decoded_digest(decoded)


def _write_pcm16_wav_install(
    path: Path,
    audio,
    *,
    class_token: _ClassDirectoryToken | None = None,
) -> _InstallToken:
    samples = _recording_samples(audio)
    pcm = _pcm16_samples(samples)

    def write(output):
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(config.SAMPLE_RATE)
            wav_file.writeframes(pcm.tobytes())

    return _atomic_create_bytes(
        Path(path),
        write,
        class_token=class_token,
    )


def write_pcm16_wav(path: Path, audio) -> None:
    token = _write_pcm16_wav_install(path, audio)
    cleanup_failure = _finalize_install_token(token)
    if cleanup_failure is not None:
        raise AlarmDataError(cleanup_failure)


def _validated_collection_label(label) -> str:
    if not isinstance(label, str) or label not in VALID_LABELS:
        raise AlarmDataError(f"label must be one of {sorted(VALID_LABELS)}")
    return label


def _validated_source_group(source_group):
    if source_group is None:
        return None
    if not isinstance(source_group, str) or not source_group.strip():
        raise AlarmDataError("source_group must be a non-empty string")
    return source_group.strip()


def _collection_root(data_dir) -> Path:
    try:
        root = Path(data_dir).resolve()
        exists = root.exists()
        is_directory = root.is_dir() if exists else False
    except (OSError, TypeError, ValueError) as exc:
        raise AlarmDataError(f"invalid alarm data directory: {data_dir!r}") from exc
    if exists and not is_directory:
        raise AlarmDataError(f"alarm data directory must be a directory: {root}")
    return root


def _resolve_class_directory(path: Path) -> Path:
    """Resolution seam used to detect symlink and junction escapes."""

    return path.resolve(strict=True)


def _validated_class_directory(
    root: Path,
    label: str,
    *,
    create: bool = False,
    required: bool = False,
):
    expected = root / label
    try:
        present = os.path.lexists(expected)
    except OSError as exc:
        raise AlarmDataError(
            f"could not inspect {label} class directory {expected}: {exc}"
        ) from exc

    if not present and create:
        _ensure_parent_directory(root)
        try:
            expected.mkdir()
        except FileExistsError:
            pass
        except OSError as exc:
            raise AlarmDataError(
                f"could not create {label} class directory {expected}: {exc}"
            ) from exc
        try:
            present = os.path.lexists(expected)
        except OSError as exc:
            raise AlarmDataError(
                f"could not recheck {label} class directory {expected}: {exc}"
            ) from exc

    if not present:
        if required:
            raise AlarmDataError(
                f"{label} class directory is missing: {expected}"
            )
        return None

    try:
        resolved = _resolve_class_directory(expected)
    except (OSError, RuntimeError) as exc:
        raise AlarmDataError(
            f"could not resolve {label} class directory {expected}: {exc}"
        ) from exc
    if resolved != expected:
        raise AlarmDataError(
            f"{label} class directory escapes the corpus root: "
            f"{expected} resolves to {resolved}"
        )
    try:
        is_directory = expected.is_dir()
    except OSError as exc:
        raise AlarmDataError(
            f"could not inspect {label} class directory {expected}: {exc}"
        ) from exc
    if not is_directory:
        raise AlarmDataError(f"{label} class path is not a directory: {expected}")
    return expected


def _validate_existing_class_directories(root: Path) -> None:
    for label in (ALARM, NOT_ALARM):
        _validated_class_directory(root, label)


def _class_directory_identity(path: Path) -> tuple[int, int]:
    return _identity_from_stat(path.stat())


def _capture_class_directory_token(
    root: Path,
    label: str,
) -> _ClassDirectoryToken:
    path = _validated_class_directory(
        root,
        label,
        create=True,
        required=True,
    )
    try:
        identity = _class_directory_identity(path)
    except OSError as exc:
        raise AlarmDataError(
            f"could not identify {label} class directory {path}: {exc}"
        ) from exc
    return _ClassDirectoryToken(
        root=root,
        label=label,
        path=path,
        identity=identity,
    )


def _verify_class_directory_token(token: _ClassDirectoryToken) -> Path:
    path = _validated_class_directory(
        token.root,
        token.label,
        required=True,
    )
    try:
        identity = _class_directory_identity(path)
    except OSError as exc:
        raise AlarmDataError(
            f"could not recheck {token.label} class directory {path}: {exc}"
        ) from exc
    if path != token.path or identity != token.identity:
        raise AlarmDataError(
            f"{token.label} class directory identity changed during "
            f"collection: {token.path}"
        )
    return path


def _decode_collection_wav(path: Path, *, display_path=None):
    shown = display_path or str(path)
    if not path.is_file():
        raise AlarmDataError(f"{shown}: source WAV must be a file")
    try:
        audio = load_wav_16k_mono(path)
    except AudioFileError as exc:
        raise AlarmDataError(f"{shown}: could not decode WAV: {exc}") from exc
    try:
        audio = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlarmDataError(f"{shown}: decoded audio must be finite mono audio") from exc
    if audio.ndim != 1:
        raise AlarmDataError(f"{shown}: decoded audio must be mono")
    if audio.size < config.WINDOW_SAMPLES:
        raise AlarmDataError(
            f"{shown}: decoded audio has {audio.size} samples; "
            f"at least {config.WINDOW_SAMPLES} are required"
        )
    if not np.isfinite(audio).all():
        raise AlarmDataError(f"{shown}: decoded audio contains non-finite samples")
    return audio, _decoded_digest(audio)


def _snapshot_collection_wav(source: Path) -> _OwnedTemporaryFile:
    try:
        descriptor, raw_snapshot = tempfile.mkstemp(
            prefix="earshot-source-",
            suffix=".wav",
        )
    except OSError as exc:
        raise AlarmDataError(
            f"{source}: could not create secure source snapshot: {exc}"
        ) from exc

    snapshot = Path(raw_snapshot)
    try:
        owned = _OwnedTemporaryFile(
            path=snapshot,
            identity=_identity_from_stat(os.fstat(descriptor)),
        )
    except OSError as exc:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            snapshot.unlink()
        except OSError:
            pass
        raise AlarmDataError(
            f"{source}: could not identify secure source snapshot: {exc}"
        ) from exc

    output = None
    try:
        output = os.fdopen(descriptor, "wb")
        with output:
            with source.open("rb") as source_file:
                shutil.copyfileobj(source_file, output)
            output.flush()
            os.fsync(output.fileno())
    except BaseException as exc:
        if output is None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        cleanup_error = _owned_temporary_unlink_error(owned)
        message = f"{source}: could not snapshot source WAV: {exc}"
        if cleanup_error is not None:
            _attach_failure_note(
                exc,
                f"snapshot cleanup failed for {snapshot}: {cleanup_error}",
            )
        if isinstance(exc, Exception):
            error = AlarmDataError(message)
            if cleanup_error is not None:
                _attach_failure_note(error, str(cleanup_error))
            raise error from exc
        raise
    return owned


def _cleanup_source_snapshots(snapshots) -> None:
    failures = []
    for snapshot in snapshots:
        cleanup_error = _owned_temporary_unlink_error(snapshot)
        if cleanup_error is not None:
            failures.append(f"{snapshot.path}: {cleanup_error}")
    if failures:
        raise AlarmDataError(
            "source snapshot cleanup failed: " + "; ".join(failures)
        )


def _collection_wavs(root: Path):
    wavs = []
    for label in (ALARM, NOT_ALARM):
        class_dir = _validated_class_directory(root, label)
        if class_dir is None:
            continue
        wavs.extend(
            (path, label)
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".wav"
        )
    wavs.sort(
        key=lambda item: _relative_path_sort_key(
            item[0].relative_to(root).as_posix()
        )
    )
    return wavs


def _existing_collection_digests(root: Path):
    by_digest = {}
    for path, label in _collection_wavs(root):
        relative_path = path.relative_to(root).as_posix()
        _, digest = _decode_collection_wav(
            path,
            display_path=relative_path,
        )
        prior = by_digest.get(digest)
        if prior is not None and prior[1] != label:
            raise AlarmDataError(
                f"{relative_path} and {prior[0]} contain exact decoded "
                "duplicate audio under both labels"
            )
        if prior is None:
            by_digest[digest] = (relative_path, label)
    return by_digest


def _occupied_relative_keys(root: Path, manifest) -> set[str]:
    keys = {relative_path.casefold() for relative_path in manifest}
    if root.is_dir():
        keys.update(
            path.relative_to(root).as_posix().casefold()
            for path in root.rglob("*")
            if path.is_file()
        )
    return keys


def _available_destination(
    root: Path,
    label: str,
    preferred_name: str,
    digest: str,
    occupied: set[str],
) -> tuple[Path, str]:
    preferred = Path(preferred_name)
    stem = preferred.stem
    suffix = preferred.suffix or ".wav"
    names = [
        preferred.name,
        f"{stem}-{digest[:12]}{suffix}",
        f"{stem}-{digest[:16]}{suffix}",
        f"{stem}-{digest}{suffix}",
    ]
    counter = 2
    while True:
        if names:
            name = names.pop(0)
        else:
            name = f"{stem}-{digest[:12]}-{counter}{suffix}"
            counter += 1
        relative_path = (PurePosixPath(label) / name).as_posix()
        key = relative_path.casefold()
        destination = root.joinpath(*PurePosixPath(relative_path).parts)
        if key in occupied or destination.exists():
            continue
        occupied.add(key)
        return destination, relative_path


def _copy_wav_atomic(
    source: Path,
    destination: Path,
    *,
    class_token: _ClassDirectoryToken | None = None,
) -> _InstallToken:
    def write(output):
        with source.open("rb") as source_file:
            shutil.copyfileobj(source_file, output)

    return _atomic_create_bytes(
        destination,
        write,
        class_token=class_token,
    )


def _write_collection_manifest(root: Path, entries) -> None:
    payload = json.dumps(
        {"version": MANIFEST_VERSION, "entries": entries},
        sort_keys=True,
        indent=2,
    ) + "\n"
    _atomic_replace_bytes(
        root / "manifest.json",
        lambda output: output.write(payload.encode("utf-8")),
    )


def _persist_candidates(label, candidates, root, source_group):
    _validate_existing_class_directories(root)
    manifest = load_manifest(root)
    existing_digests = _existing_collection_digests(root)
    occupied = _occupied_relative_keys(root, manifest)
    planned = []

    for preferred_name, digest, install in candidates:
        prior = existing_digests.get(digest)
        if prior is not None:
            if prior[1] != label:
                raise AlarmDataError(
                    f"{preferred_name} and {prior[0]} contain exact decoded "
                    "duplicate audio under both labels"
                )
            continue

        destination, relative_path = _available_destination(
            root,
            label,
            preferred_name,
            digest,
            occupied,
        )
        entry = {
            "path": relative_path,
            "label": label,
            "source_group": source_group or f"source-{digest[:16]}",
            "segments": [],
        }
        planned.append((destination, install, entry))
        existing_digests[digest] = (relative_path, label)

    if not planned:
        return ()

    manifest_entries = [dict(entry) for entry in manifest.values()]
    manifest_entries.extend(entry for _, _, entry in planned)
    manifest_entries.sort(
        key=lambda entry: _relative_path_sort_key(entry["path"])
    )

    class_token = _capture_class_directory_token(root, label)
    _validate_existing_class_directories(root)
    installs = []
    try:
        for destination, install, _ in planned:
            verified_parent = _verify_class_directory_token(class_token)
            if destination.parent != verified_parent:
                raise AlarmDataError(
                    f"planned destination left the verified class directory: "
                    f"{destination}"
                )
            install_token = install(destination, class_token)
            if not isinstance(install_token, _InstallToken):
                raise AlarmDataError(
                    f"install did not return an identity token for {destination}"
                )
            installs.append(install_token)
            post_install_parent = _verify_class_directory_token(class_token)
            if (
                install_token.destination != destination
                or install_token.staging.path.parent != post_install_parent
            ):
                raise AlarmDataError(
                    f"install token left the verified class directory: "
                    f"{destination}"
                )
        _write_collection_manifest(root, manifest_entries)
    except BaseException as exc:
        rollback_failures = []
        if installs:
            try:
                _verify_class_directory_token(class_token)
            except AlarmDataError as rollback_error:
                rollback_failures.append(
                    f"class directory revalidation failed: {rollback_error}"
                )
            for install_token in reversed(installs):
                rollback_failures.extend(
                    _rollback_install_token(install_token)
                )
        if rollback_failures:
            rollback_message = (
                "rollback failed: " + "; ".join(rollback_failures)
            )
            _attach_failure_note(exc, rollback_message)
        raise

    cleanup_failures = []
    for install_token in installs:
        cleanup_failure = _finalize_install_token(install_token)
        if cleanup_failure is not None:
            cleanup_failures.append(cleanup_failure)
    if cleanup_failures:
        raise AlarmDataError(
            "manifest committed but install-token cleanup failed: "
            + "; ".join(cleanup_failures)
        )
    return tuple(destination for destination, _, _ in planned)


def collect_files(
    label,
    paths,
    data_dir,
    source_group=None,
) -> tuple[Path, ...]:
    """Validate and copy WAV files into the local alarm corpus atomically."""

    label = _validated_collection_label(label)
    source_group = _validated_source_group(source_group)
    root = _collection_root(data_dir)
    _validate_existing_class_directories(root)
    try:
        sources = tuple(Path(path).resolve() for path in paths)
    except (OSError, TypeError, ValueError) as exc:
        raise AlarmDataError("paths must contain valid WAV file paths") from exc

    snapshots = []
    try:
        preflighted = []
        for source in sources:
            snapshot = _snapshot_collection_wav(source)
            snapshots.append(snapshot)
            _, digest = _decode_collection_wav(
                snapshot.path,
                display_path=str(source),
            )
            preflighted.append((source.name, snapshot, digest))
        if not preflighted:
            result = ()
        else:
            candidates = []
            for preferred_name, snapshot, digest in preflighted:
                candidates.append(
                    (
                        preferred_name,
                        digest,
                        lambda destination, class_token, snapshot=snapshot.path: (
                            _copy_wav_atomic(
                                snapshot,
                                destination,
                                class_token=class_token,
                            )
                        ),
                    )
                )
            result = _persist_candidates(
                label,
                candidates,
                root,
                source_group,
            )
    except BaseException as primary_error:
        try:
            _cleanup_source_snapshots(snapshots)
        except Exception as cleanup_error:
            _attach_failure_note(
                primary_error,
                f"source snapshot cleanup failed: {cleanup_error}",
            )
        raise
    else:
        _cleanup_source_snapshots(snapshots)
        return result


def _validated_recording_request(count, seconds):
    if (
        isinstance(count, (bool, np.bool_))
        or not isinstance(count, (int, np.integer))
        or count <= 0
    ):
        raise AlarmDataError("count must be a positive integer")
    if isinstance(seconds, (bool, np.bool_)):
        raise AlarmDataError("seconds must be a finite recording duration")
    try:
        duration = float(seconds)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AlarmDataError("seconds must be a finite recording duration") from exc
    minimum = config.WINDOW_SAMPLES / config.SAMPLE_RATE
    if not np.isfinite(duration) or duration < minimum:
        raise AlarmDataError(
            f"seconds must record at least one model window ({minimum:g})"
        )
    return int(count), duration


def collect_recordings(
    label,
    count,
    seconds,
    data_dir,
    source_group=None,
    device=None,
    recorder=record,
    before_capture=None,
    *,
    clock=datetime.now,
) -> tuple[Path, ...]:
    """Capture a microphone batch, then persist valid recordings atomically."""

    label = _validated_collection_label(label)
    count, seconds = _validated_recording_request(count, seconds)
    source_group = _validated_source_group(source_group)
    root = _collection_root(data_dir)
    _validate_existing_class_directories(root)
    if not callable(recorder):
        raise AlarmDataError("recorder must be callable")
    if before_capture is not None and not callable(before_capture):
        raise AlarmDataError("before_capture must be callable")
    if not callable(clock):
        raise AlarmDataError("clock must be callable")
    try:
        timestamp = clock().strftime("%Y%m%d-%H%M%S")
    except (AttributeError, TypeError, ValueError) as exc:
        raise AlarmDataError("clock must return a local datetime value") from exc

    captured = []
    for index in range(1, count + 1):
        if before_capture is not None:
            try:
                before_capture(index, count, seconds)
            except Exception as exc:
                raise AlarmDataError(
                    f"before capture failed for recording {index}: {exc}"
                ) from exc
        try:
            audio = recorder(seconds, device=device)
        except Exception as exc:
            raise AlarmDataError(
                f"recording failed for capture {index}: {exc}"
            ) from exc
        samples = _recording_samples(audio).copy()
        captured.append((index, samples, _pcm16_decoded_digest(samples)))

    candidates = []
    for index, samples, digest in captured:
        preferred_name = f"{label}-{timestamp}-{index:03d}.wav"
        candidates.append(
            (
                preferred_name,
                digest,
                lambda destination, class_token, samples=samples: (
                    _write_pcm16_wav_install(
                        destination,
                        samples,
                        class_token=class_token,
                    )
                ),
            )
        )
    return _persist_candidates(
        label,
        candidates,
        root,
        source_group,
    )


def _class_wavs(root: Path, label: str) -> list[Path]:
    class_dir = _validated_class_directory(root, label, required=True)

    files = sorted(
        (path for path in class_dir.rglob("*") if path.is_file()),
        key=lambda path: _relative_path_sort_key(
            path.relative_to(root).as_posix()
        ),
    )
    non_wavs = [path for path in files if path.suffix.casefold() != ".wav"]
    if non_wavs:
        relative = non_wavs[0].relative_to(root).as_posix()
        raise AlarmDataError(f"{relative}: class directories may contain only WAV files")

    wavs = [path for path in files if path.suffix.casefold() == ".wav"]
    if not wavs:
        raise AlarmDataError(f"{label} class directory is empty: {class_dir}")
    return wavs


def inventory_corpus(data_dir):
    """Validate, decode, deduplicate, and deterministically list the corpus."""

    root = _collection_root(data_dir)
    _validate_existing_class_directories(root)
    corpus_paths = []
    for label in (ALARM, NOT_ALARM):
        corpus_paths.extend((path, label) for path in _class_wavs(root, label))
    corpus_paths.sort(
        key=lambda item: _relative_path_sort_key(
            item[0].relative_to(root).as_posix()
        )
    )

    manifest_path = root / "manifest.json"
    manifest_present = manifest_path.is_file()
    manifest = load_manifest(root)
    corpus_by_relative_path = {
        path.relative_to(root).as_posix(): (path, label)
        for path, label in corpus_paths
    }
    corpus_by_casefolded_path = {}
    for relative_path in corpus_by_relative_path:
        corpus_by_casefolded_path.setdefault(
            relative_path.casefold(), []
        ).append(relative_path)
    manifest_by_corpus_path = {}
    for manifest_relative_path, metadata in manifest.items():
        if manifest_relative_path in corpus_by_relative_path:
            corpus_relative_path = manifest_relative_path
        else:
            candidates = corpus_by_casefolded_path.get(
                manifest_relative_path.casefold(), []
            )
            if not candidates:
                raise AlarmDataError(
                    f"{manifest_relative_path}: manifest path must name an "
                    "existing class WAV"
                )
            if len(candidates) > 1:
                raise AlarmDataError(
                    f"{manifest_relative_path}: ambiguous case-insensitive "
                    f"manifest path matches {', '.join(candidates)}"
                )
            corpus_relative_path = candidates[0]
        _, path_label = corpus_by_relative_path[corpus_relative_path]
        manifest_label = metadata["label"]
        if manifest_label != path_label:
            raise AlarmDataError(
                f"{corpus_relative_path}: manifest label {manifest_label!r} "
                f"and path class {path_label!r} disagree"
            )
        manifest_by_corpus_path[corpus_relative_path] = metadata

    warnings = []
    if not manifest_present:
        warnings.append(
            "manifest.json is missing; using stable manual source groups "
            "for all WAV files"
        )

    entries = []
    digest_entries = {}
    for path, label in corpus_paths:
        relative_path = path.relative_to(root).as_posix()
        metadata = manifest_by_corpus_path.get(relative_path)

        try:
            audio = load_wav_16k_mono(path)
        except AudioFileError as exc:
            raise AlarmDataError(f"{relative_path}: could not decode WAV: {exc}") from exc
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise AlarmDataError(f"{relative_path}: decoded audio must be mono")
        if len(audio) < config.WINDOW_SAMPLES:
            raise AlarmDataError(
                f"{relative_path}: decoded audio has {len(audio)} samples; "
                f"at least {config.WINDOW_SAMPLES} are required"
            )
        if not np.isfinite(audio).all():
            raise AlarmDataError(
                f"{relative_path}: decoded audio contains non-finite samples"
            )

        digest = _decoded_digest(audio)
        duration = len(audio) / config.SAMPLE_RATE
        if metadata is None:
            source_group = f"manual-{digest[:16]}"
            segments = ()
            if manifest_present:
                warnings.append(
                    f"{relative_path} is not listed in manifest.json; using "
                    "a stable manual source group"
                )
        else:
            source_group = metadata["source_group"]
            segments = _validated_segments(
                metadata.get("segments"),
                duration=duration,
                relative_path=relative_path,
            )

        prior = digest_entries.get(digest)
        if prior is not None:
            if prior.label != label:
                raise AlarmDataError(
                    f"{relative_path} and {prior.relative_path} contain exact "
                    "decoded duplicate audio under both labels"
                )
            warnings.append(
                f"{relative_path} duplicates {prior.relative_path}; counting "
                "the decoded audio once"
            )
            continue

        entry = CorpusEntry(
            path=path.resolve(),
            relative_path=relative_path,
            label=label,
            source_group=source_group,
            segments=segments,
            decoded_sha256=digest,
            duration_seconds=duration,
        )
        entries.append(entry)
        digest_entries[digest] = entry

    entries.sort(
        key=lambda entry: _relative_path_sort_key(entry.relative_path)
    )
    return CorpusInventory(
        root=root,
        entries=tuple(entries),
        warnings=tuple(warnings),
    )
