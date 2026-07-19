"""Engine-level tests that run without a microphone or YAMNet model."""

import queue
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from earshot_ml import config, core  # noqa: E402
from earshot_ml.alarm_model import AlarmModelError  # noqa: E402
from earshot_ml.core import EarshotML  # noqa: E402


def embedding(index=0):
    value = np.zeros(1024, dtype=np.float32)
    value[index] = 1.0
    return value


class FakeYamNet:
    """Deterministic inference seam for engine tests."""

    def __init__(self, results=None, class_names=None):
        self.class_names = (list(class_names) if class_names is not None else
                            ["Smoke detector, smoke alarm"])
        self.results = list(results or [])
        self.infer_calls = []

    def infer(self, waveform):
        self.infer_calls.append(np.asarray(waveform))
        if self.results:
            result = self.results.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return (np.zeros(len(self.class_names), dtype=np.float32), embedding())


class FakeAlarmHead:
    label = "smoke_alarm"
    urgency = "high"
    threshold = 0.70
    gate_count = 2
    gate_window = 8

    def __init__(self):
        self.score_calls = []

    def score(self, value):
        self.score_calls.append(value)
        return float(np.asarray(value)[0])


def make_engine(yamnet, **kwargs):
    return EarshotML(
        yamnet=yamnet,
        taught_store_path=None,
        alarm_model_path=None,
        **kwargs,
    )


def test_legacy_seventh_positional_yamnet_argument_remains_supported():
    fake = FakeYamNet(class_names=[])

    engine = EarshotML(
        None,
        None,
        None,
        config.MODEL_PATH,
        config.CLASS_MAP_PATH,
        None,
        fake,
        alarm_model_path=None,
    )

    assert engine.yamnet is fake
    assert engine.process_window(
        np.zeros(config.WINDOW_SAMPLES),
        now=0.0,
    ) == []
    assert len(fake.infer_calls) == 1


def test_pretrained_event_fires_after_two_windows():
    result = (np.array([0.90], dtype=np.float32), embedding(1))
    engine = make_engine(FakeYamNet([result, result]))

    assert engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.0) == []
    events = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.5)

    assert [(event.label, event.source) for event in events] == [
        ("smoke_alarm", "pretrained")
    ]


def test_taught_event_fires_after_one_window():
    learned_embedding = embedding(2)
    fake = FakeYamNet(
        [(np.array([], dtype=np.float32), learned_embedding)],
        class_names=[],
    )
    engine = make_engine(fake)
    engine.store.add("clap", learned_embedding)

    events = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=2.0)

    assert [(event.label, event.source) for event in events] == [
        ("clap", "taught")
    ]


def test_fired_event_reaches_callback_and_queue():
    event_queue = queue.Queue()
    callback_payloads = []
    learned_embedding = embedding(3)
    fake = FakeYamNet(
        [(np.array([], dtype=np.float32), learned_embedding)],
        class_names=[],
    )
    engine = make_engine(
        fake,
        on_event=callback_payloads.append,
        event_queue=event_queue,
    )
    engine.store.add("kettle", learned_embedding)

    events = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=12.0)

    expected = events[0].to_dict()
    assert callback_payloads == [expected]
    assert event_queue.get_nowait() == expected
    assert event_queue.empty()


def test_same_label_from_taught_and_pretrained_has_independent_state():
    shared_embedding = embedding(4)
    result = (np.array([0.90], dtype=np.float32), shared_embedding)
    engine = make_engine(FakeYamNet([result, result]))
    # Direct store setup models persisted/internal data independently of teach's
    # reserved-name validation and exercises detector identity precisely.
    engine.store.add("smoke_alarm", shared_embedding)

    first = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=20.0)
    second = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=20.5)

    assert [(event.label, event.source) for event in first] == [
        ("smoke_alarm", "taught")
    ]
    assert [(event.label, event.source) for event in second] == [
        ("smoke_alarm", "pretrained")
    ]


def test_missing_alarm_artifact_preserves_exact_generic_fallback(tmp_path):
    result = (
        np.array([0.99, 0.0, 0.0], dtype=np.float32),
        embedding(1),
    )
    fake = FakeYamNet(
        [result, result],
        class_names=[
            "Fire alarm",
            "Smoke detector, smoke alarm",
            "Doorbell",
        ],
    )
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=None,
        alarm_model_path=tmp_path / "missing.npz",
    )

    assert engine.alarm_head is None
    assert [spec["label"] for spec in engine._specs] == [
        "smoke_alarm",
        "doorbell",
    ]
    assert set(engine._specs[0]["indices"]) == {0, 1}
    assert engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.0) == []
    events = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.5)
    assert [(event.label, event.source) for event in events] == [
        ("smoke_alarm", "pretrained")
    ]


def test_corrupt_existing_alarm_artifact_propagates_at_startup(tmp_path):
    artifact = tmp_path / "corrupt.npz"
    artifact.write_bytes(b"not an alarm artifact")

    with pytest.raises(AlarmModelError, match="Could not read alarm artifact"):
        EarshotML(
            yamnet=FakeYamNet(class_names=[]),
            taught_store_path=None,
            alarm_model_path=artifact,
        )


def test_injected_alarm_head_wins_over_existing_artifact(tmp_path):
    artifact = tmp_path / "corrupt.npz"
    artifact.write_bytes(b"must not be loaded")
    head = FakeAlarmHead()

    engine = EarshotML(
        yamnet=FakeYamNet(class_names=[]),
        taught_store_path=None,
        alarm_model_path=artifact,
        alarm_head=head,
    )

    assert engine.alarm_head is head


def test_trained_head_uses_same_embedding_and_two_of_eight_gate():
    positive = embedding(0)
    negative = embedding(1)
    fake = FakeYamNet(
        [
            (np.array([0.99], np.float32), positive),
            (np.array([0.99], np.float32), negative),
            (np.array([0.99], np.float32), positive),
        ],
        class_names=["Fire alarm"],
    )
    head = FakeAlarmHead()
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=head,
    )

    assert engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.0) == []
    assert engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.5) == []
    events = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=2.0)

    assert [(event.label, event.source) for event in events] == [
        ("smoke_alarm", "trained")
    ]
    assert len(fake.infer_calls) == 3
    assert head.score_calls == [positive, negative, positive]
    assert all(
        scored is returned
        for scored, returned in zip(
            head.score_calls,
            [positive, negative, positive],
        )
    )


def test_trained_head_suppresses_only_generic_alarm_specs():
    fake = FakeYamNet(
        class_names=[
            "Fire alarm",
            "Smoke detector, smoke alarm",
            "Doorbell",
        ]
    )
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
    )

    assert [spec["label"] for spec in engine._specs] == ["doorbell"]


def test_trained_head_preserves_doorbell_and_taught_events():
    positive = embedding(0)
    result = (np.array([0.99, 0.99], np.float32), positive)
    engine = EarshotML(
        yamnet=FakeYamNet(
            [result, result],
            class_names=["Fire alarm", "Doorbell"],
        ),
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
    )
    engine.store.add("kettle", positive)

    first = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=3.0)
    second = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=3.5)

    assert [(event.label, event.source) for event in first] == [
        ("kettle", "taught")
    ]
    assert [(event.label, event.source) for event in second] == [
        ("doorbell", "pretrained"),
        ("smoke_alarm", "trained"),
    ]


def test_trained_alarm_uses_unchanged_ten_second_debounce():
    positive = embedding(0)
    result = (np.array([0.99], np.float32), positive)
    engine = EarshotML(
        yamnet=FakeYamNet(
            [result, result, result, result],
            class_names=["Fire alarm"],
        ),
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
    )

    fired = []
    for now in (0.0, 0.5, 5.0, 10.5):
        fired.extend(
            engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=now)
        )

    assert [(event.label, event.timestamp) for event in fired] == [
        ("smoke_alarm", 0.5),
        ("smoke_alarm", 10.5),
    ]


def test_trained_alarm_gate_state_is_independent_per_engine():
    positive = embedding(0)
    result = (np.array([0.99], np.float32), positive)
    first = EarshotML(
        yamnet=FakeYamNet([result, result], class_names=["Fire alarm"]),
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
    )
    second = EarshotML(
        yamnet=FakeYamNet([result], class_names=["Fire alarm"]),
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
    )

    assert first.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.0) == []
    assert second.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.0) == []
    events = first.process_window(np.zeros(config.WINDOW_SAMPLES), now=1.5)

    assert [(event.label, event.source) for event in events] == [
        ("smoke_alarm", "trained")
    ]


def test_trained_event_callback_and_queue_keep_exact_public_payload():
    event_queue = queue.Queue()
    callback_payloads = []
    positive = embedding(0)
    result = (np.array([0.99], np.float32), positive)
    engine = EarshotML(
        yamnet=FakeYamNet([result, result], class_names=["Fire alarm"]),
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
        on_event=callback_payloads.append,
        event_queue=event_queue,
    )

    assert engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=41.5) == []
    events = engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=42.0)

    expected = {
        "label": "smoke_alarm",
        "urgency": "high",
        "confidence": 1.0,
        "source": "trained",
        "timestamp": 42.0,
    }
    assert [event.to_dict() for event in events] == [expected]
    assert callback_payloads == [expected]
    assert event_queue.get_nowait() == expected
    assert event_queue.empty()
    assert set(expected) == {
        "label",
        "urgency",
        "confidence",
        "source",
        "timestamp",
    }


class CountingScores:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.index_count = 0

    def __getitem__(self, key):
        self.index_count += 1
        return self.values[key]


def test_pretrained_mapped_max_is_computed_once_per_spec():
    scores = CountingScores([0.90])
    engine = make_engine(FakeYamNet([(scores, embedding(5))]))

    engine.process_window(np.zeros(config.WINDOW_SAMPLES), now=30.0)

    assert scores.index_count == 1


@pytest.mark.parametrize("name", [None, 7, "", "   "])
def test_teach_rejects_invalid_names_before_inference(name):
    fake = FakeYamNet(class_names=[])
    engine = make_engine(fake)

    with pytest.raises(ValueError):
        engine.teach(name, [np.ones(config.WINDOW_SAMPLES, dtype=np.float32)])

    assert fake.infer_calls == []
    assert engine.learned_sounds() == []


def test_teach_rejects_pretrained_label_case_insensitively_after_trim():
    fake = FakeYamNet(class_names=[])
    engine = make_engine(fake)

    with pytest.raises(ValueError):
        engine.teach(
            "  SmOkE_AlArM  ",
            [np.ones(config.WINDOW_SAMPLES, dtype=np.float32)],
        )

    assert fake.infer_calls == []
    assert engine.learned_sounds() == []


@pytest.mark.parametrize(
    "reserved_name",
    ["FIRE_ALARM", "FIRE_SMOKE_ALARM"],
)
def test_teach_rejects_legacy_alarm_label_before_files_or_store_mutation(
        tmp_path, monkeypatch, reserved_name):
    store_path = tmp_path / "taught.npz"
    original = core.TeachStore(store_path)
    original.add("keep", embedding())
    original.save()
    original_bytes = store_path.read_bytes()
    fake = FakeYamNet(class_names=[])
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=store_path,
        alarm_model_path=None,
    )

    def forbid_file_load(path):
        pytest.fail(f"reserved name tried to load {path}")

    monkeypatch.setattr(core, "load_wav_16k_mono", forbid_file_load)

    with pytest.raises(ValueError, match="reserved"):
        engine.teach(f"  {reserved_name}  ", [tmp_path / "missing.wav"])

    assert fake.infer_calls == []
    assert engine.learned_sounds() == [{"name": "keep", "clips": 1}]
    assert store_path.read_bytes() == original_bytes


@pytest.mark.parametrize("clips", [None, 42, [], ()])
def test_teach_requires_a_nonempty_clip_iterable(clips):
    fake = FakeYamNet(class_names=[])
    engine = make_engine(fake)

    with pytest.raises(ValueError):
        engine.teach("kettle", clips)

    assert fake.infer_calls == []
    assert engine.learned_sounds() == []


@pytest.mark.parametrize(
    "audio",
    [
        np.array([], dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
        np.array([np.inf], dtype=np.float32),
    ],
)
def test_teach_rejects_invalid_array_audio_before_inference(audio):
    fake = FakeYamNet(class_names=[])
    engine = make_engine(fake)

    with pytest.raises(ValueError):
        engine.teach("kettle", [audio])

    assert fake.infer_calls == []
    assert engine.learned_sounds() == []


def test_teach_validates_every_clip_before_any_inference_or_store_mutation(
        tmp_path):
    fake = FakeYamNet(class_names=[])
    store_path = tmp_path / "taught.npz"
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=store_path,
        alarm_model_path=None,
    )
    valid = np.ones(config.WINDOW_SAMPLES, dtype=np.float32)
    invalid = np.array([np.nan], dtype=np.float32)

    with pytest.raises(ValueError):
        engine.teach("kettle", [valid, invalid])

    assert fake.infer_calls == []
    assert engine.learned_sounds() == []
    assert not store_path.exists()


def test_teach_validates_loaded_path_audio_before_inference(monkeypatch,
                                                            tmp_path):
    fake = FakeYamNet(class_names=[])
    engine = make_engine(fake)
    monkeypatch.setattr(
        core,
        "load_wav_16k_mono",
        lambda _path: np.zeros((2, 2), dtype=np.float32),
    )

    with pytest.raises(ValueError):
        engine.teach("kettle", [tmp_path / "bad.wav"])

    assert fake.infer_calls == []
    assert engine.learned_sounds() == []


def test_teach_preserves_path_and_array_handling_and_stored_count(monkeypatch,
                                                                  tmp_path):
    fake = FakeYamNet(class_names=[])
    engine = make_engine(fake)
    audio = np.ones(config.WINDOW_SAMPLES, dtype=np.float32)
    path = tmp_path / "clip.wav"
    loaded_paths = []

    def fake_load(candidate):
        loaded_paths.append(candidate)
        return audio

    monkeypatch.setattr(core, "load_wav_16k_mono", fake_load)

    stored = engine.teach("kettle", [path, audio])

    assert stored == 2
    assert loaded_paths == [path]
    assert engine.learned_sounds() == [{"name": "kettle", "clips": 2}]


@pytest.mark.parametrize(
    "second_result",
    [
        RuntimeError("second inference failed"),
        (np.array([], dtype=np.float32), np.zeros(3, dtype=np.float32)),
    ],
    ids=["inference-error", "invalid-embedding"],
)
def test_teach_prepares_every_embedding_before_mutating_or_saving(
        tmp_path, monkeypatch, second_result):
    store_path = tmp_path / "taught.npz"
    original = core.TeachStore(store_path)
    original.add("known-good", embedding())
    original.save()
    before_bytes = store_path.read_bytes()

    fake = FakeYamNet(
        [
            (np.array([], dtype=np.float32), embedding(1)),
            second_result,
        ],
        class_names=[],
    )
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=store_path,
        alarm_model_path=None,
    )
    before_vectors = engine.store._vectors.copy()
    save_calls = []
    monkeypatch.setattr(engine.store, "save", lambda: save_calls.append(True))
    audio = np.ones(config.WINDOW_SAMPLES, dtype=np.float32)

    with pytest.raises((RuntimeError, ValueError)):
        engine.teach("kettle", [audio, audio])

    assert len(fake.infer_calls) == 2
    assert save_calls == []
    assert engine.learned_sounds() == [{"name": "known-good", "clips": 1}]
    assert np.array_equal(engine.store._vectors, before_vectors)
    assert engine.store.match(embedding()) == (
        "known-good",
        pytest.approx(1.0),
    )
    assert engine.store.match(embedding(1)) is None
    assert store_path.read_bytes() == before_bytes
    assert not store_path.with_name(store_path.name + ".part").exists()


def test_teach_save_failure_restores_memory_and_existing_store(
        tmp_path, monkeypatch):
    store_path = tmp_path / "taught.npz"
    original = core.TeachStore(store_path)
    original.add("known-good", embedding())
    original.save()
    before_bytes = store_path.read_bytes()

    fake = FakeYamNet(
        [(np.array([], dtype=np.float32), embedding(1))],
        class_names=[],
    )
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=store_path,
        alarm_model_path=None,
    )
    before_vectors = engine.store._vectors.copy()

    def fail_savez(output, **_arrays):
        output.write(b"partial replacement")
        raise OSError("simulated teach write failure")

    monkeypatch.setattr(core.np, "savez", fail_savez)
    audio = np.ones(config.WINDOW_SAMPLES, dtype=np.float32)

    with pytest.raises(
        core.TeachStoreError, match="simulated teach write failure"
    ) as exc_info:
        engine.teach("kettle", [audio])

    assert isinstance(exc_info.value.__cause__, OSError)

    assert engine.learned_sounds() == [{"name": "known-good", "clips": 1}]
    assert np.array_equal(engine.store._vectors, before_vectors)
    assert engine.store.match(embedding()) == (
        "known-good",
        pytest.approx(1.0),
    )
    assert engine.store.match(embedding(1)) is None
    assert store_path.read_bytes() == before_bytes
    assert not store_path.with_name(store_path.name + ".part").exists()


def test_forget_save_failure_restores_memory_and_existing_store(
        tmp_path, monkeypatch):
    store_path = tmp_path / "taught.npz"
    original = core.TeachStore(store_path)
    original.add("keep", embedding())
    original.add("remove", embedding(1))
    original.save()
    before_bytes = store_path.read_bytes()

    engine = EarshotML(
        yamnet=FakeYamNet(class_names=[]),
        taught_store_path=store_path,
        alarm_model_path=None,
    )
    before_vectors = engine.store._vectors.copy()

    def fail_replace(_source, _destination):
        raise PermissionError("simulated forget replace failure")

    monkeypatch.setattr(core.os, "replace", fail_replace)

    with pytest.raises(
        core.TeachStoreError, match="simulated forget replace failure"
    ) as exc_info:
        engine.forget("remove")

    assert isinstance(exc_info.value.__cause__, PermissionError)

    assert engine.learned_sounds() == [
        {"name": "keep", "clips": 1},
        {"name": "remove", "clips": 1},
    ]
    assert np.array_equal(engine.store._vectors, before_vectors)
    assert engine.store.match(embedding()) == ("keep", pytest.approx(1.0))
    assert engine.store.match(embedding(1)) == (
        "remove",
        pytest.approx(1.0),
    )
    assert store_path.read_bytes() == before_bytes
    assert not store_path.with_name(store_path.name + ".part").exists()


def test_teach_success_commits_and_returns_stored_count(tmp_path):
    store_path = tmp_path / "taught.npz"
    fake = FakeYamNet(
        [(np.array([], dtype=np.float32), embedding(1))],
        class_names=[],
    )
    engine = EarshotML(
        yamnet=fake,
        taught_store_path=store_path,
        alarm_model_path=None,
    )
    audio = np.ones(config.WINDOW_SAMPLES, dtype=np.float32)

    assert engine.teach("kettle", [audio]) == 1
    assert engine.learned_sounds() == [{"name": "kettle", "clips": 1}]
    assert core.TeachStore(store_path).learned() == [
        {"name": "kettle", "clips": 1}
    ]


def test_forget_success_commits_and_returns_removed_count(tmp_path):
    store_path = tmp_path / "taught.npz"
    original = core.TeachStore(store_path)
    original.add("keep", embedding())
    original.add("remove", embedding(1))
    original.save()
    engine = EarshotML(
        yamnet=FakeYamNet(class_names=[]),
        taught_store_path=store_path,
        alarm_model_path=None,
    )

    assert engine.forget("remove") == 1
    assert engine.learned_sounds() == [{"name": "keep", "clips": 1}]
    assert core.TeachStore(store_path).learned() == [
        {"name": "keep", "clips": 1}
    ]


def test_run_forwards_stop_event_to_stream_and_exits(monkeypatch):
    stop_event = threading.Event()
    received = {}

    class FakeMicStream:
        def __init__(self, device=None):
            received["device"] = device

        def windows(self, stop_event=None, on_gap=None):
            received["stop_event"] = stop_event
            received["on_gap"] = on_gap
            yield np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
            stop_event.set()

    fake = FakeYamNet(class_names=[])
    engine = make_engine(fake, device="fake-device")
    monkeypatch.setattr(core, "MicStream", FakeMicStream)

    assert engine.run(stop_event=stop_event) is None
    assert received == {
        "device": "fake-device",
        "stop_event": stop_event,
        "on_gap": engine._reset_after_capture_gap,
    }
    assert stop_event.is_set()
    assert len(fake.infer_calls) == 1


def test_run_capture_gap_does_not_combine_any_detection_evidence(monkeypatch):
    positive = embedding(0)
    result = (np.array([0.99, 0.99], np.float32), positive)
    emitted = []

    class GappedMicStream:
        def __init__(self, device=None):
            del device

        def windows(self, stop_event=None, on_gap=None):
            del stop_event
            yield np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
            on_gap()
            yield np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)

    engine = EarshotML(
        yamnet=FakeYamNet(
            [result, result],
            class_names=["Fire alarm", "Doorbell"],
        ),
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
        on_event=emitted.append,
    )
    monkeypatch.setattr(core, "MicStream", GappedMicStream)

    engine.run()

    assert emitted == []


def test_run_capture_gap_preserves_event_debounce(monkeypatch):
    positive = embedding(0)
    result = (np.array([0.99], np.float32), positive)
    emitted = []

    class GappedMicStream:
        def __init__(self, device=None):
            del device

        def windows(self, stop_event=None, on_gap=None):
            del stop_event
            yield np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
            yield np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
            on_gap()
            yield np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
            yield np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)

    engine = EarshotML(
        yamnet=FakeYamNet(
            [result, result, result, result],
            class_names=["Fire alarm"],
        ),
        taught_store_path=None,
        alarm_model_path=None,
        alarm_head=FakeAlarmHead(),
        on_event=emitted.append,
    )
    monkeypatch.setattr(core, "MicStream", GappedMicStream)

    engine.run()

    assert [event["label"] for event in emitted] == ["smoke_alarm"]
