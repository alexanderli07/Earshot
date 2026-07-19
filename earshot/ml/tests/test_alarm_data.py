import json
import os
from datetime import datetime
from pathlib import Path
import wave

import numpy as np
import pytest

from earshot_ml import alarm_data
from earshot_ml.alarm_data import (
    AlarmDataError,
    collect_files,
    collect_recordings,
    inventory_corpus,
    load_manifest,
    write_pcm16_wav,
)


def write_pcm16(path: Path, samples, rate=16_000):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(np.clip(samples, -1, 1) * 32767, dtype="<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(pcm.tobytes())


def write_two_class_corpus(root: Path):
    samples = np.linspace(-0.2, 0.2, 16_000, dtype=np.float32)
    write_pcm16(root / "alarm" / "a.wav", samples)
    write_pcm16(root / "not_alarm" / "n.wav", -samples)


def write_manifest(root: Path, entries):
    (root / "manifest.json").write_text(
        json.dumps({"version": 1, "entries": entries}),
        encoding="utf-8",
    )


def test_inventory_uses_manifest_source_groups_and_segments(tmp_path):
    audio = np.ones(16_000, dtype=np.float32) * 0.2
    write_pcm16(tmp_path / "alarm" / "a.wav", audio)
    write_pcm16(tmp_path / "not_alarm" / "n.wav", audio * 0.5)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/a.wav",
                "label": "alarm",
                "source_group": "alarm-source",
                "segments": [[0.1, 0.8]],
            },
            {
                "path": "not_alarm/n.wav",
                "label": "not_alarm",
                "source_group": "negative-source",
                "segments": [],
            },
        ],
    )

    inventory = inventory_corpus(tmp_path)

    assert [(entry.label, entry.source_group, entry.segments)
            for entry in inventory.entries] == [
        ("alarm", "alarm-source", ((0.1, 0.8),)),
        ("not_alarm", "negative-source", ()),
    ]
    assert inventory.root == tmp_path.resolve()
    assert inventory.entries_for("alarm") == (inventory.entries[0],)


def test_manual_files_receive_stable_in_memory_groups(tmp_path):
    write_two_class_corpus(tmp_path)

    first = inventory_corpus(tmp_path)
    second = inventory_corpus(tmp_path)

    assert [entry.source_group for entry in first.entries] == [
        entry.source_group for entry in second.entries
    ]
    assert all(
        entry.source_group.startswith("manual-")
        for entry in first.entries
    )
    assert first.warnings == second.warnings
    assert "manifest" in " ".join(first.warnings).lower()


def test_inventory_exact_path_tiebreaker_selects_duplicate_deterministically(
    tmp_path,
    monkeypatch,
):
    lower = tmp_path / "alarm" / "a.wav"
    upper = tmp_path / "alarm" / "A.wav"
    negative = tmp_path / "not_alarm" / "n.wav"
    paths_by_label = {
        "alarm": [lower, upper],
        "not_alarm": [negative],
    }

    monkeypatch.setattr(
        alarm_data,
        "_class_wavs",
        lambda root, label: paths_by_label[label],
    )

    def load_case_collision(path):
        relative_path = path.relative_to(tmp_path).as_posix()
        if relative_path.startswith("alarm/"):
            return np.ones(16_000, dtype=np.float32) * 0.25
        return np.ones(16_000, dtype=np.float32) * -0.25

    monkeypatch.setattr(
        alarm_data,
        "load_wav_16k_mono",
        load_case_collision,
    )

    inventory = inventory_corpus(tmp_path)

    assert [entry.relative_path for entry in inventory.entries] == [
        "alarm/A.wav",
        "not_alarm/n.wav",
    ]
    assert "alarm/a.wav duplicates alarm/A.wav" in " ".join(
        inventory.warnings
    )


def install_case_distinct_corpus(tmp_path, monkeypatch, decoded_paths):
    paths_by_label = {
        "alarm": [
            tmp_path / "alarm" / "a.wav",
            tmp_path / "alarm" / "A.wav",
        ],
        "not_alarm": [tmp_path / "not_alarm" / "n.wav"],
    }
    levels = {
        "alarm/A.wav": 0.1,
        "alarm/a.wav": 0.2,
        "not_alarm/n.wav": -0.2,
    }

    monkeypatch.setattr(
        alarm_data,
        "_class_wavs",
        lambda root, label: paths_by_label[label],
    )

    def load_case_distinct(path):
        relative_path = path.relative_to(tmp_path).as_posix()
        decoded_paths.append(relative_path)
        return np.ones(16_000, dtype=np.float32) * levels[relative_path]

    monkeypatch.setattr(
        alarm_data,
        "load_wav_16k_mono",
        load_case_distinct,
    )


def test_manifest_metadata_prefers_exact_case_distinct_path(
    tmp_path,
    monkeypatch,
):
    decoded_paths = []
    install_case_distinct_corpus(tmp_path, monkeypatch, decoded_paths)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/A.wav",
                "label": "alarm",
                "source_group": "uppercase-source",
                "segments": [],
            }
        ],
    )

    inventory = inventory_corpus(tmp_path)

    groups = {
        entry.relative_path: entry.source_group
        for entry in inventory.entries
    }
    assert groups["alarm/A.wav"] == "uppercase-source"
    assert groups["alarm/a.wav"].startswith("manual-")


def test_manifest_rejects_ambiguous_case_fallback_before_decoding(
    tmp_path,
    monkeypatch,
):
    decoded_paths = []
    install_case_distinct_corpus(tmp_path, monkeypatch, decoded_paths)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/a.WAV",
                "label": "alarm",
                "source_group": "ambiguous-source",
                "segments": [],
            }
        ],
    )

    with pytest.raises(AlarmDataError, match="ambiguous"):
        inventory_corpus(tmp_path)

    assert decoded_paths == []


def test_manifest_does_not_hide_unlisted_manual_wavs(tmp_path):
    samples = np.linspace(-0.2, 0.2, 16_000, dtype=np.float32)
    write_pcm16(tmp_path / "alarm" / "listed.wav", samples)
    write_pcm16(tmp_path / "alarm" / "manual.wav", samples * 0.5)
    write_pcm16(tmp_path / "not_alarm" / "negative.wav", -samples)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/listed.wav",
                "label": "alarm",
                "source_group": "listed-source",
                "segments": [],
            },
            {
                "path": "not_alarm/negative.wav",
                "label": "not_alarm",
                "source_group": "negative-source",
                "segments": [],
            },
        ],
    )

    inventory = inventory_corpus(tmp_path)

    assert [entry.relative_path for entry in inventory.entries] == [
        "alarm/listed.wav",
        "alarm/manual.wav",
        "not_alarm/negative.wav",
    ]
    manual = next(
        entry for entry in inventory.entries
        if entry.relative_path == "alarm/manual.wav"
    )
    assert manual.source_group.startswith("manual-")
    assert "manifest" in " ".join(inventory.warnings).lower()


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


@pytest.mark.parametrize("missing_label", ["alarm", "not_alarm"])
def test_inventory_rejects_missing_class_directory(tmp_path, missing_label):
    present_label = "not_alarm" if missing_label == "alarm" else "alarm"
    write_pcm16(
        tmp_path / present_label / "clip.wav",
        np.zeros(16_000, dtype=np.float32),
    )

    with pytest.raises(AlarmDataError, match=rf"{missing_label}.*missing"):
        inventory_corpus(tmp_path)


@pytest.mark.parametrize("empty_label", ["alarm", "not_alarm"])
def test_inventory_rejects_empty_class_directory(tmp_path, empty_label):
    present_label = "not_alarm" if empty_label == "alarm" else "alarm"
    write_pcm16(
        tmp_path / present_label / "clip.wav",
        np.zeros(16_000, dtype=np.float32),
    )
    (tmp_path / empty_label).mkdir()

    with pytest.raises(AlarmDataError, match=rf"{empty_label}.*empty"):
        inventory_corpus(tmp_path)


def test_inventory_rejects_non_wav_files_in_class_directories(tmp_path):
    write_two_class_corpus(tmp_path)
    (tmp_path / "alarm" / "notes.txt").write_text("not audio")

    with pytest.raises(AlarmDataError, match=r"notes\.txt.*WAV"):
        inventory_corpus(tmp_path)


def test_inventory_rejects_clips_shorter_than_one_model_window(tmp_path):
    write_pcm16(
        tmp_path / "alarm" / "short.wav",
        np.zeros(15_599, dtype=np.float32),
    )
    write_pcm16(
        tmp_path / "not_alarm" / "valid.wav",
        np.zeros(16_000, dtype=np.float32),
    )

    with pytest.raises(AlarmDataError, match=r"short\.wav.*15,?600"):
        inventory_corpus(tmp_path)


def test_inventory_wraps_unreadable_wav_errors(tmp_path):
    (tmp_path / "alarm").mkdir()
    (tmp_path / "alarm" / "bad.wav").write_bytes(b"not a wave")
    write_pcm16(
        tmp_path / "not_alarm" / "valid.wav",
        np.zeros(16_000, dtype=np.float32),
    )

    with pytest.raises(AlarmDataError, match=r"bad\.wav"):
        inventory_corpus(tmp_path)


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ("all", "segments must be a list"),
        ([[0.0]], "each segment"),
        ([[-0.1, 0.2]], "invalid segment"),
        ([[0.2, 0.2]], "invalid segment"),
        ([[0.8, 0.2]], "invalid segment"),
        ([[0.0, 1.1]], "invalid segment"),
        ([[float("nan"), 0.2]], "invalid segment"),
    ],
)
def test_inventory_rejects_invalid_segment_ranges(tmp_path, segments, message):
    write_two_class_corpus(tmp_path)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/a.wav",
                "label": "alarm",
                "source_group": "alarm-source",
                "segments": segments,
            }
        ],
    )

    with pytest.raises(AlarmDataError, match=message):
        inventory_corpus(tmp_path)


def test_inventory_rejects_invalid_manifest_label(tmp_path):
    write_two_class_corpus(tmp_path)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/a.wav",
                "label": "smoke",
                "source_group": "source",
                "segments": [],
            }
        ],
    )

    with pytest.raises(AlarmDataError, match="label"):
        inventory_corpus(tmp_path)


@pytest.mark.parametrize(
    "invalid_label",
    [["alarm"], {}],
    ids=["list", "object"],
)
def test_inventory_rejects_non_string_manifest_labels(
    tmp_path,
    invalid_label,
):
    write_two_class_corpus(tmp_path)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/a.wav",
                "label": invalid_label,
                "source_group": "source",
                "segments": [],
            }
        ],
    )

    with pytest.raises(AlarmDataError, match="label"):
        inventory_corpus(tmp_path)


def test_inventory_rejects_manifest_label_path_disagreement_before_decoding(
    tmp_path,
    monkeypatch,
):
    write_two_class_corpus(tmp_path)
    write_manifest(
        tmp_path,
        [
            {
                "path": "not_alarm/n.wav",
                "label": "alarm",
                "source_group": "source",
                "segments": [],
            }
        ],
    )
    decoded_paths = []
    real_load_wav = alarm_data.load_wav_16k_mono

    def tracked_load_wav(path):
        decoded_paths.append(path)
        return real_load_wav(path)

    monkeypatch.setattr(alarm_data, "load_wav_16k_mono", tracked_load_wav)

    with pytest.raises(AlarmDataError, match="disagree"):
        inventory_corpus(tmp_path)

    assert decoded_paths == []


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "JSON object"),
        ({"version": 2, "entries": []}, "version"),
        ({"version": 1, "entries": {}}, "entries must be a list"),
        ({"version": 1, "entries": ["alarm/a.wav"]}, "entry.*object"),
        (
            {
                "version": 1,
                "entries": [
                    {
                        "path": "alarm/a.wav",
                        "label": "alarm",
                        "source_group": "one",
                    },
                    {
                        "path": "alarm/a.wav",
                        "label": "alarm",
                        "source_group": "two",
                    },
                ],
            },
            "duplicate.*path",
        ),
        (
            {
                "version": 1,
                "entries": [
                    {
                        "path": "../outside.wav",
                        "label": "alarm",
                        "source_group": "source",
                    }
                ],
            },
            "relative path",
        ),
    ],
)
def test_load_manifest_rejects_malformed_contracts(tmp_path, payload, message):
    (tmp_path / "manifest.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(AlarmDataError, match=message):
        load_manifest(tmp_path)


def test_load_manifest_rejects_invalid_json(tmp_path):
    (tmp_path / "manifest.json").write_text("{", encoding="utf-8")

    with pytest.raises(AlarmDataError, match="JSON"):
        load_manifest(tmp_path)


def test_inventory_rejects_manifest_path_outside_data_dir(tmp_path):
    write_two_class_corpus(tmp_path)
    outside = tmp_path.parent / "outside.wav"
    write_pcm16(outside, np.zeros(16_000, dtype=np.float32))
    write_manifest(
        tmp_path,
        [
            {
                "path": str(outside),
                "label": "alarm",
                "source_group": "source",
                "segments": [],
            }
        ],
    )

    with pytest.raises(AlarmDataError, match="relative path"):
        inventory_corpus(tmp_path)


def test_inventory_rejects_manifest_path_that_is_not_a_corpus_wav(tmp_path):
    write_two_class_corpus(tmp_path)
    write_manifest(
        tmp_path,
        [
            {
                "path": "manifest.json",
                "label": "alarm",
                "source_group": "source",
                "segments": [],
            }
        ],
    )

    with pytest.raises(AlarmDataError, match="class WAV"):
        inventory_corpus(tmp_path)


def test_inventory_rejects_empty_manifest_source_group(tmp_path):
    write_two_class_corpus(tmp_path)
    write_manifest(
        tmp_path,
        [
            {
                "path": "alarm/a.wav",
                "label": "alarm",
                "source_group": " ",
                "segments": [],
            }
        ],
    )

    with pytest.raises(AlarmDataError, match="source_group"):
        inventory_corpus(tmp_path)


def test_collect_files_validates_then_copies_without_moving_source(tmp_path):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    samples = np.linspace(-0.25, 0.25, 16_000, dtype=np.float32)
    write_pcm16(source, samples)
    before = source.read_bytes()

    stored = collect_files(
        "alarm",
        [source],
        data_dir,
        source_group="smoke-a",
    )

    assert source.read_bytes() == before
    assert len(stored) == 1
    assert stored[0].parent == data_dir / "alarm"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert manifest["entries"][0]["source_group"] == "smoke-a"


def test_collect_files_preflights_every_input_before_mutating_corpus(tmp_path):
    valid = tmp_path / "valid.wav"
    invalid = tmp_path / "invalid.wav"
    data_dir = tmp_path / "data"
    write_pcm16(valid, np.ones(16_000, dtype=np.float32) * 0.1)
    invalid.write_bytes(b"not a PCM WAV")
    valid_before = valid.read_bytes()
    invalid_before = invalid.read_bytes()

    with pytest.raises(AlarmDataError, match=r"invalid\.wav"):
        collect_files("alarm", [valid, invalid], data_dir)

    assert valid.read_bytes() == valid_before
    assert invalid.read_bytes() == invalid_before
    assert not data_dir.exists()


def test_collect_files_rejects_label_before_decoding_or_mutating(
    tmp_path,
    monkeypatch,
):
    decoded = []
    monkeypatch.setattr(
        alarm_data,
        "load_wav_16k_mono",
        lambda path: decoded.append(path),
    )

    with pytest.raises(AlarmDataError, match="label"):
        collect_files("smoke", [tmp_path / "missing.wav"], tmp_path / "data")

    assert decoded == []
    assert not (tmp_path / "data").exists()


def test_collect_collision_uses_hash_suffix_and_never_overwrites(tmp_path):
    first = tmp_path / "first" / "same.wav"
    second = tmp_path / "second" / "same.wav"
    write_pcm16(first, np.ones(16_000, dtype=np.float32) * 0.1)
    write_pcm16(second, np.ones(16_000, dtype=np.float32) * 0.2)

    stored = collect_files("alarm", [first, second], tmp_path / "data")

    assert stored[0].name == "same.wav"
    assert stored[1].stem.startswith("same-")
    assert stored[0].read_bytes() != stored[1].read_bytes()


def test_collect_files_leaves_predictable_part_hardlink_untouched(tmp_path):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    destination = data_dir / "alarm" / source.name
    predictable_part = destination.with_name(destination.name + ".part")
    sentinel_target = tmp_path / "sentinel.bin"
    sentinel_bytes = b"pre-existing staging sentinel"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    destination.parent.mkdir(parents=True)
    sentinel_target.write_bytes(sentinel_bytes)
    os.link(sentinel_target, predictable_part)

    stored = collect_files("alarm", [source], data_dir)

    assert stored == (destination,)
    assert sentinel_target.read_bytes() == sentinel_bytes
    assert predictable_part.read_bytes() == sentinel_bytes


def test_collect_files_persists_validated_snapshot_if_source_changes(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    validated_bytes = source.read_bytes()
    validated_digest = alarm_data._decoded_digest(
        alarm_data.load_wav_16k_mono(source)
    )
    real_persist = alarm_data._persist_candidates

    def mutate_original_then_persist(*args, **kwargs):
        write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.8)
        return real_persist(*args, **kwargs)

    monkeypatch.setattr(
        alarm_data,
        "_persist_candidates",
        mutate_original_then_persist,
    )

    stored = collect_files("alarm", [source], data_dir)

    assert stored[0].read_bytes() == validated_bytes
    assert alarm_data._decoded_digest(
        alarm_data.load_wav_16k_mono(stored[0])
    ) == validated_digest


def test_collect_files_racing_destination_is_untouched_and_not_rolled_back(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    destination = data_dir / "alarm" / source.name
    racing_bytes = b"created by another writer"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    real_copy = alarm_data._copy_wav_atomic

    def create_racing_destination_then_copy(
        snapshot,
        planned_destination,
        *,
        class_token=None,
    ):
        planned_destination.parent.mkdir(parents=True, exist_ok=True)
        planned_destination.write_bytes(racing_bytes)
        return real_copy(
            snapshot,
            planned_destination,
            class_token=class_token,
        )

    monkeypatch.setattr(
        alarm_data,
        "_copy_wav_atomic",
        create_racing_destination_then_copy,
    )

    with pytest.raises(
        AlarmDataError,
        match="refusing to overwrite existing corpus file",
    ):
        collect_files("alarm", [source], data_dir)

    assert destination.read_bytes() == racing_bytes
    assert not (data_dir / "manifest.json").exists()


def test_manifest_failure_preserves_destination_replaced_after_install(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    destination = data_dir / "alarm" / source.name
    foreign_bytes = b"replacement created by another writer"
    primary_error = AlarmDataError("primary manifest failure")
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)

    def replace_destination_then_fail(_root, _entries):
        destination.unlink()
        destination.write_bytes(foreign_bytes)
        raise primary_error

    monkeypatch.setattr(
        alarm_data,
        "_write_collection_manifest",
        replace_destination_then_fail,
    )

    with pytest.raises(AlarmDataError, match="primary manifest failure") as error:
        collect_files("alarm", [source], data_dir)

    assert error.value is primary_error
    assert any(
        "could not safely delete" in note
        for note in getattr(error.value, "__notes__", ())
    )
    assert destination.read_bytes() == foreign_bytes
    assert not list(data_dir.rglob("*.part"))


def test_python310_fallback_preserves_primary_type_and_rollback_details(
    tmp_path,
    monkeypatch,
):
    class PrimaryFailure(AlarmDataError):
        pass

    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    destination = data_dir / "alarm" / source.name
    foreign_bytes = b"foreign replacement"
    primary_error = PrimaryFailure("primary manifest failure")
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)

    def replace_destination_then_fail(_root, _entries):
        destination.unlink()
        destination.write_bytes(foreign_bytes)
        raise primary_error

    monkeypatch.setattr(alarm_data, "_EXCEPTION_ADD_NOTE", None, raising=False)
    monkeypatch.setattr(
        alarm_data,
        "_write_collection_manifest",
        replace_destination_then_fail,
    )

    with pytest.raises(PrimaryFailure) as error:
        collect_files("alarm", [source], data_dir)

    assert error.value is primary_error
    assert type(error.value) is PrimaryFailure
    assert "primary manifest failure" in str(error.value)
    assert "could not safely delete" in str(error.value)
    assert destination.read_bytes() == foreign_bytes


def test_collect_files_rolls_back_owned_wav_if_class_identity_changes(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    class_dir = data_dir / "alarm"
    destination = class_dir / source.name
    outside = tmp_path / "outside"
    outside_sentinel = outside / "sentinel.bin"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    class_dir.mkdir(parents=True)
    outside.mkdir()
    outside_sentinel.write_bytes(b"foreign data")
    identity_calls = 0

    def changing_class_identity(path):
        nonlocal identity_calls
        stat_result = Path(path).stat()
        identity = (stat_result.st_dev, stat_result.st_ino)
        if Path(path) == class_dir:
            identity_calls += 1
            if identity_calls >= 3:
                return identity[0], identity[1] + 1
        return identity

    monkeypatch.setattr(
        alarm_data,
        "_class_directory_identity",
        changing_class_identity,
        raising=False,
    )

    with pytest.raises(AlarmDataError, match="class directory identity changed"):
        collect_files("alarm", [source], data_dir)

    assert identity_calls >= 3
    assert not destination.exists()
    assert outside_sentinel.read_bytes() == b"foreign data"
    assert not list(data_dir.rglob("*.part"))
    assert not (data_dir / "manifest.json").exists()


def test_keyboard_interrupt_after_first_install_rolls_back_owned_wav(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    real_link = alarm_data.os.link

    def link_then_interrupt(staging, destination):
        real_link(staging, destination)
        raise KeyboardInterrupt("capture cancelled")

    monkeypatch.setattr(
        alarm_data.os,
        "link",
        link_then_interrupt,
    )

    with pytest.raises(KeyboardInterrupt, match="capture cancelled"):
        collect_files("alarm", [source], data_dir)

    assert not list(data_dir.rglob("*.wav"))
    assert not list(data_dir.rglob("*.part"))
    assert not (data_dir / "manifest.json").exists()


def test_snapshot_cleanup_failure_does_not_mask_primary_failure(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.wav"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    primary_error = AlarmDataError("primary persistence failure")
    real_cleanup = alarm_data._cleanup_source_snapshots

    def fail_persistence(*_args, **_kwargs):
        raise primary_error

    def cleanup_then_fail(snapshots):
        real_cleanup(snapshots)
        raise AlarmDataError("snapshot cleanup failure")

    monkeypatch.setattr(alarm_data, "_persist_candidates", fail_persistence)
    monkeypatch.setattr(
        alarm_data,
        "_cleanup_source_snapshots",
        cleanup_then_fail,
    )

    with pytest.raises(AlarmDataError, match="primary persistence failure") as error:
        collect_files("alarm", [source], tmp_path / "data")

    assert error.value is primary_error
    assert any(
        "snapshot cleanup failure" in note
        for note in getattr(error.value, "__notes__", ())
    )


def test_python310_fallback_preserves_primary_type_and_cleanup_details(
    tmp_path,
    monkeypatch,
):
    class PrimaryFailure(AlarmDataError):
        pass

    source = tmp_path / "source.wav"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    primary_error = PrimaryFailure("primary persistence failure")
    real_cleanup = alarm_data._cleanup_source_snapshots

    def fail_persistence(*_args, **_kwargs):
        raise primary_error

    def cleanup_then_fail(snapshots):
        real_cleanup(snapshots)
        raise AlarmDataError("snapshot cleanup failure")

    monkeypatch.setattr(alarm_data, "_EXCEPTION_ADD_NOTE", None, raising=False)
    monkeypatch.setattr(alarm_data, "_persist_candidates", fail_persistence)
    monkeypatch.setattr(
        alarm_data,
        "_cleanup_source_snapshots",
        cleanup_then_fail,
    )

    with pytest.raises(PrimaryFailure) as error:
        collect_files("alarm", [source], tmp_path / "data")

    assert error.value is primary_error
    assert type(error.value) is PrimaryFailure
    assert "primary persistence failure" in str(error.value)
    assert "snapshot cleanup failure" in str(error.value)


def test_collect_files_rejects_escaping_class_dir_before_external_mutation(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    class_dir = data_dir / "alarm"
    outside = tmp_path / "outside"
    outside_sentinel = outside / "sentinel.bin"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    class_dir.mkdir(parents=True)
    outside.mkdir()
    outside_sentinel.write_bytes(b"outside must remain untouched")

    def escaped_resolution(path):
        if Path(path) == class_dir:
            return outside.resolve()
        return Path(path).resolve(strict=True)

    monkeypatch.setattr(
        alarm_data,
        "_resolve_class_directory",
        escaped_resolution,
        raising=False,
    )

    with pytest.raises(AlarmDataError, match="escapes the corpus root"):
        collect_files("alarm", [source], data_dir)

    assert outside_sentinel.read_bytes() == b"outside must remain untouched"
    assert list(outside.iterdir()) == [outside_sentinel]
    assert list(class_dir.iterdir()) == []
    assert not (data_dir / "manifest.json").exists()


def test_collect_files_skips_exact_duplicates_in_batch_and_corpus(tmp_path):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.15)

    first = collect_files("alarm", [source, source], data_dir)
    second = collect_files("alarm", [source], data_dir)

    assert len(first) == 1
    assert second == ()
    assert len(list((data_dir / "alarm").glob("*.wav"))) == 1
    manifest = json.loads((data_dir / "manifest.json").read_text())
    assert len(manifest["entries"]) == 1


def test_collect_files_rejects_cross_label_exact_duplicate(tmp_path):
    source = tmp_path / "source.wav"
    data_dir = tmp_path / "data"
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.15)
    collect_files("alarm", [source], data_dir)
    manifest_before = (data_dir / "manifest.json").read_bytes()

    with pytest.raises(AlarmDataError, match="both labels"):
        collect_files("not_alarm", [source], data_dir)

    assert not (data_dir / "not_alarm").exists()
    assert (data_dir / "manifest.json").read_bytes() == manifest_before


def test_collect_manifest_entries_are_sorted_and_preserve_source_groups(tmp_path):
    z_source = tmp_path / "z.wav"
    a_source = tmp_path / "a.wav"
    negative = tmp_path / "n.wav"
    data_dir = tmp_path / "data"
    write_pcm16(z_source, np.ones(16_000, dtype=np.float32) * 0.1)
    write_pcm16(a_source, np.ones(16_000, dtype=np.float32) * 0.2)
    write_pcm16(negative, np.ones(16_000, dtype=np.float32) * -0.2)

    collect_files(
        "alarm",
        [z_source, a_source],
        data_dir,
        source_group="same-smoke-detector",
    )
    collect_files("not_alarm", [negative], data_dir)

    text = (data_dir / "manifest.json").read_text(encoding="utf-8")
    entries = json.loads(text)["entries"]
    assert [entry["path"] for entry in entries] == [
        "alarm/a.wav",
        "alarm/z.wav",
        "not_alarm/n.wav",
    ]
    assert [entry["source_group"] for entry in entries[:2]] == [
        "same-smoke-detector",
        "same-smoke-detector",
    ]
    assert entries[2]["source_group"].startswith("source-")
    assert text.endswith("\n")


def test_collect_recording_writes_atomic_mono_16k_pcm(tmp_path):
    audio = np.linspace(-1, 1, 16_000, dtype=np.float32)
    calls = []
    prompts = []

    stored = collect_recordings(
        "not_alarm",
        1,
        1.0,
        tmp_path / "data",
        device=7,
        recorder=lambda seconds, device=None: calls.append(
            (seconds, device)
        ) or audio,
        before_capture=lambda index, count, seconds: prompts.append(
            (index, count, seconds)
        ),
        clock=lambda: datetime(2026, 7, 18, 13, 14, 15),
    )

    assert calls == [(1.0, 7)]
    assert prompts == [(1, 1, 1.0)]
    assert stored[0].name == "not_alarm-20260718-131415-001.wav"
    with wave.open(str(stored[0]), "rb") as saved:
        assert (
            saved.getnchannels(),
            saved.getframerate(),
            saved.getsampwidth(),
        ) == (1, 16_000, 2)
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert manifest["entries"][0]["source_group"].startswith("source-")
    assert not list((tmp_path / "data").rglob("*.part"))


def test_collect_recordings_skips_duplicate_after_pcm16_round_trip(tmp_path):
    audio = np.linspace(-0.25, 0.25, 16_000, dtype=np.float32)
    data_dir = tmp_path / "data"

    first = collect_recordings(
        "alarm",
        1,
        1.0,
        data_dir,
        recorder=lambda *_args, **_kwargs: audio,
        clock=lambda: datetime(2026, 7, 18, 13, 14, 15),
    )
    second = collect_recordings(
        "alarm",
        1,
        1.0,
        data_dir,
        recorder=lambda *_args, **_kwargs: audio,
        clock=lambda: datetime(2026, 7, 18, 13, 14, 16),
    )

    assert len(first) == 1
    assert second == ()
    assert len(list((data_dir / "alarm").glob("*.wav"))) == 1


def test_collect_recordings_snapshots_each_batch_capture(tmp_path):
    shared = np.zeros(16_000, dtype=np.float32)
    capture_number = 0

    def recorder(_seconds, device=None):
        nonlocal capture_number
        capture_number += 1
        shared.fill(capture_number * 0.1)
        return shared

    stored = collect_recordings(
        "alarm",
        2,
        1.0,
        tmp_path / "data",
        recorder=recorder,
        clock=lambda: datetime(2026, 7, 18, 13, 14, 15),
    )

    assert len(stored) == 2
    assert stored[0].read_bytes() != stored[1].read_bytes()


@pytest.mark.parametrize(
    ("label", "count", "seconds", "message"),
    [
        ("smoke", 1, 1.0, "label"),
        ("alarm", 0, 1.0, "count"),
        ("alarm", True, 1.0, "count"),
        ("alarm", 1.5, 1.0, "count"),
        ("alarm", 1, 0.0, "seconds"),
        ("alarm", 1, float("nan"), "seconds"),
        ("alarm", 1, 0.5, "seconds"),
    ],
)
def test_collect_recordings_rejects_invalid_inputs_before_recorder_use(
    tmp_path,
    label,
    count,
    seconds,
    message,
):
    calls = []

    with pytest.raises(AlarmDataError, match=message):
        collect_recordings(
            label,
            count,
            seconds,
            tmp_path / "data",
            recorder=lambda *_args, **_kwargs: calls.append(True),
        )

    assert calls == []
    assert not (tmp_path / "data").exists()


def test_collect_recordings_captures_and_validates_batch_before_persistence(
    tmp_path,
):
    responses = [
        np.ones(16_000, dtype=np.float32) * 0.1,
        np.ones(100, dtype=np.float32),
    ]
    calls = []

    def recorder(seconds, device=None):
        calls.append((seconds, device))
        return responses.pop(0)

    with pytest.raises(AlarmDataError, match="one model window"):
        collect_recordings(
            "alarm",
            2,
            1.0,
            tmp_path / "data",
            recorder=recorder,
        )

    assert calls == [(1.0, None), (1.0, None)]
    assert not (tmp_path / "data").exists()


def test_collect_recordings_wraps_before_capture_failure(tmp_path):
    backend_error = RuntimeError("prompt backend unavailable")
    recorder_calls = []

    with pytest.raises(AlarmDataError, match="before capture failed") as error:
        collect_recordings(
            "alarm",
            1,
            1.0,
            tmp_path / "data",
            before_capture=lambda *_args: (_ for _ in ()).throw(backend_error),
            recorder=lambda *_args, **_kwargs: recorder_calls.append(True),
        )

    assert error.value.__cause__ is backend_error
    assert recorder_calls == []


def test_collect_recordings_wraps_recorder_failure(tmp_path):
    backend_error = OSError("microphone backend unavailable")

    with pytest.raises(AlarmDataError, match="recording failed") as error:
        collect_recordings(
            "alarm",
            1,
            1.0,
            tmp_path / "data",
            recorder=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                backend_error
            ),
        )

    assert error.value.__cause__ is backend_error


def test_write_pcm16_wav_wraps_parent_directory_failure(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "denied" / "recording.wav"
    real_mkdir = Path.mkdir

    def deny_parent_creation(self, *args, **kwargs):
        if self == path.parent:
            raise PermissionError("parent directory denied")
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", deny_parent_creation)

    with pytest.raises(AlarmDataError, match="parent directory denied") as error:
        write_pcm16_wav(path, np.zeros(16_000, dtype=np.float32))

    assert isinstance(error.value.__cause__, PermissionError)


def test_manifest_failure_rolls_back_every_new_destination(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest.write_text('{"version": 1, "entries": []}', encoding="utf-8")
    before = manifest.read_bytes()
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    write_pcm16(first, np.ones(16_000, dtype=np.float32) * 0.1)
    write_pcm16(second, np.ones(16_000, dtype=np.float32) * 0.2)
    real_replace = alarm_data.os.replace

    def fail_manifest_replace(part, destination):
        if Path(destination) == manifest:
            raise PermissionError("read-only")
        return real_replace(part, destination)

    monkeypatch.setattr(alarm_data.os, "replace", fail_manifest_replace)

    with pytest.raises(AlarmDataError, match="read-only"):
        collect_files("alarm", [first, second], data_dir)

    assert manifest.read_bytes() == before
    assert not list(data_dir.rglob("*.wav"))
    assert not list(data_dir.rglob("*.part"))


def test_manifest_failure_surfaces_destination_rollback_failure(
    tmp_path,
    monkeypatch,
):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    manifest = data_dir / "manifest.json"
    manifest.write_text('{"version": 1, "entries": []}', encoding="utf-8")
    source = tmp_path / "source.wav"
    destination = data_dir / "alarm" / source.name
    write_pcm16(source, np.ones(16_000, dtype=np.float32) * 0.1)
    real_replace = alarm_data.os.replace
    real_unlink = Path.unlink

    def fail_manifest_replace(part, installed_path):
        if Path(installed_path) == manifest:
            raise PermissionError("manifest read-only")
        return real_replace(part, installed_path)

    def fail_destination_rollback(self, *args, **kwargs):
        if self == destination:
            raise PermissionError("rollback denied")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(alarm_data.os, "replace", fail_manifest_replace)
    monkeypatch.setattr(Path, "unlink", fail_destination_rollback)

    with pytest.raises(AlarmDataError, match="manifest read-only") as error:
        collect_files("alarm", [source], data_dir)

    assert "manifest read-only" in str(error.value)
    assert any(
        "rollback denied" in note
        for note in getattr(error.value, "__notes__", ())
    )
    assert destination.exists()


@pytest.mark.parametrize(
    "audio",
    [
        np.zeros((16_000, 1), dtype=np.float32),
        np.zeros(15_599, dtype=np.float32),
        np.full(16_000, np.nan, dtype=np.float32),
    ],
    ids=["not-mono", "too-short", "non-finite"],
)
def test_write_pcm16_wav_rejects_invalid_audio_without_partial_file(
    tmp_path,
    audio,
):
    path = tmp_path / "recording.wav"

    with pytest.raises(AlarmDataError, match="finite mono audio"):
        write_pcm16_wav(path, audio)

    assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()
