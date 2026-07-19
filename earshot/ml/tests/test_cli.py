import builtins
import hashlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from earshot_ml import alarm_model, cli, core, pipeline
from earshot_ml.artifacts import Artifact
from earshot_ml.core import EarshotML, TeachStore


ROOT = Path(__file__).resolve().parent.parent
COMMANDS = (
    "download",
    "top5",
    "run",
    "teach",
    "sounds",
    "forget",
    "collect",
    "train-alarm",
    "evaluate-alarm",
)
TFLITE_BACKENDS = ("ai_edge_litert", "tensorflow", "tflite_runtime")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _artifact(source, destination, data, *, sha256=None):
    source.write_bytes(data)
    return Artifact(
        url=source.resolve().as_uri(),
        path=destination,
        sha256=sha256 or _digest(data),
    )


def _store_with_sounds(path):
    store = TeachStore(path)
    store.add("kettle", np.eye(1, 1024, 0, dtype=np.float32).ravel())
    store.add("kettle", np.eye(1, 1024, 1, dtype=np.float32).ravel())
    store.add("dryer", np.eye(1, 1024, 2, dtype=np.float32).ravel())
    store.save()


def _forbid_model_use(*args, **kwargs):
    pytest.fail("storage/help actions must not construct a model or engine")


def _forbid_tflite_imports(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith(TFLITE_BACKENDS):
            pytest.fail(f"storage action imported TFLite backend {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


class _FakeTeachYamNet:
    class_names = [
        class_name
        for event in cli.config.EVENT_MAP
        for class_name in event["classes"]
    ]

    def infer(self, waveform):
        embedding = np.zeros(1024, dtype=np.float32)
        embedding[0] = 1.0
        return np.zeros(len(self.class_names), dtype=np.float32), embedding


def _real_teach_engine_factory(store_path, constructed=None):
    def factory(**kwargs):
        engine = EarshotML(
            yamnet=_FakeTeachYamNet(), taught_store_path=store_path
        )
        if constructed is not None:
            constructed.append(engine)
        return engine

    return factory


def test_main_help_lists_all_commands(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    for command in COMMANDS:
        assert command in help_text


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


def test_cold_process_help_works_when_sklearn_is_unavailable():
    script = (
        "import importlib.abc\n"
        "import runpy\n"
        "import sys\n"
        "class BlockSklearn(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'sklearn' or fullname.startswith('sklearn.'):\n"
        "            raise ModuleNotFoundError('blocked sklearn')\n"
        "        return None\n"
        "sys.meta_path.insert(0, BlockSklearn())\n"
        "try:\n"
        "    import sklearn\n"
        "except ModuleNotFoundError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('sklearn blocker did not engage')\n"
        "sys.argv = ['earshot', '--help']\n"
        "runpy.run_module('earshot_ml.cli', run_name='__main__')\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "One CLI for Earshot" in completed.stdout
    for command in COMMANDS:
        assert command in completed.stdout


def test_collect_parser_defaults():
    args = cli._build_parser().parse_args(["collect", "alarm"])

    assert args.label == "alarm"
    assert args.wavs == []
    assert args.record == 0
    assert args.seconds == 5.0
    assert args.device is None
    assert args.data_dir == cli.config.ALARM_DATA_DIR
    assert args.source_group is None


def test_train_alarm_parser_defaults():
    args = cli._build_parser().parse_args(["train-alarm"])

    assert args.data_dir == cli.config.ALARM_DATA_DIR
    assert args.output == cli.config.ALARM_MODEL_PATH
    assert args.seed == 0


def test_evaluate_alarm_parser_defaults():
    args = cli._build_parser().parse_args(["evaluate-alarm"])

    assert args.data_dir == cli.config.ALARM_DATA_DIR
    assert args.model == cli.config.ALARM_MODEL_PATH


def test_collect_combines_files_and_recordings(monkeypatch, tmp_path, capsys):
    imported = []
    captured = []
    monkeypatch.setattr(
        cli,
        "collect_files",
        lambda label, paths, data_dir, source_group=None: (
            imported.extend(paths) or (tmp_path / "a.wav",)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "collect_recordings",
        lambda label, count, seconds, data_dir, **kwargs: (
            captured.append((label, count, seconds, kwargs))
            or (tmp_path / "b.wav",)
        ),
        raising=False,
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    cli.main([
        "collect",
        "alarm",
        "source.wav",
        "--record",
        "1",
        "--seconds",
        "1",
        "--device",
        "1",
        "--data-dir",
        str(tmp_path),
    ])

    assert imported == [Path("source.wav")]
    assert len(captured) == 1
    assert captured[0][1:3] == (1, 1.0)
    assert captured[0][3]["device"] == 1
    assert "stored 2 clips" in capsys.readouterr().out


def test_collect_files_only_skips_microphone(monkeypatch, tmp_path, capsys):
    received = []
    stored = tmp_path / "alarm" / "source.wav"
    monkeypatch.setattr(
        cli,
        "collect_files",
        lambda label, paths, data_dir, source_group=None: (
            received.append((label, tuple(paths), data_dir, source_group))
            or (stored,)
        ),
    )
    monkeypatch.setattr(cli, "collect_recordings", _forbid_model_use)
    monkeypatch.setattr(cli, "record", _forbid_model_use)

    cli.main([
        "collect",
        "alarm",
        "source.wav",
        "--data-dir",
        str(tmp_path),
        "--source-group",
        "office",
    ])

    assert received == [(
        "alarm",
        (Path("source.wav"),),
        tmp_path,
        "office",
    )]
    output = capsys.readouterr().out
    assert str(stored) in output
    assert "stored 1 clips" in output


def test_collect_recordings_only_uses_prompt_delay_and_injected_recorder(
        monkeypatch, tmp_path, capsys):
    received = []
    prompts = []
    delays = []
    stored = tmp_path / "not_alarm" / "capture.wav"

    def fake_collect(label, count, seconds, data_dir, **kwargs):
        received.append((label, count, seconds, data_dir, kwargs))
        kwargs["before_capture"](1, count, seconds)
        return (stored,)

    monkeypatch.setattr(cli, "collect_files", _forbid_model_use)
    monkeypatch.setattr(cli, "collect_recordings", fake_collect)
    monkeypatch.setattr(
        builtins, "input", lambda prompt: prompts.append(prompt) or ""
    )
    monkeypatch.setattr(cli.time, "sleep", delays.append)

    cli.main([
        "collect",
        "not_alarm",
        "--record",
        "2",
        "--seconds",
        "1.5",
        "--data-dir",
        str(tmp_path),
    ])

    assert len(received) == 1
    assert received[0][0:4] == ("not_alarm", 2, 1.5, tmp_path)
    assert received[0][4]["recorder"] is cli.record
    assert prompts and "1/2" in prompts[0]
    assert delays == [0.2]
    assert "stored 1 clips" in capsys.readouterr().out


def test_collect_rejects_invalid_label_before_mutation(monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_files", _forbid_model_use)
    monkeypatch.setattr(cli, "collect_recordings", _forbid_model_use)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["collect", "other", "source.wav"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "invalid choice" in stderr
    assert "traceback" not in stderr


@pytest.mark.parametrize(
    ("arguments", "diagnostic"),
    [
        (["alarm", "source.wav", "--record", "-1"], "--record"),
        (["alarm", "source.wav", "--seconds", "0"], "--seconds"),
        (["alarm", "source.wav", "--seconds", "-1"], "--seconds"),
        (["alarm", "source.wav", "--seconds", "nan"], "--seconds"),
        (["alarm", "source.wav", "--seconds", "inf"], "--seconds"),
        (["alarm", "source.wav", "--seconds", "0.5"], "--seconds"),
    ],
)
def test_collect_rejects_invalid_options_before_mutation(
        monkeypatch, capsys, arguments, diagnostic):
    monkeypatch.setattr(cli, "collect_files", _forbid_model_use)
    monkeypatch.setattr(cli, "collect_recordings", _forbid_model_use)
    monkeypatch.setattr(cli, "record", _forbid_model_use)
    monkeypatch.setattr(builtins, "input", _forbid_model_use)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["collect", *arguments])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert diagnostic in stderr
    assert "traceback" not in stderr


def test_collect_requires_files_or_recording_before_mutation(
        monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_files", _forbid_model_use)
    monkeypatch.setattr(cli, "collect_recordings", _forbid_model_use)
    monkeypatch.setattr(cli, "record", _forbid_model_use)
    monkeypatch.setattr(builtins, "input", _forbid_model_use)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["collect", "alarm"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "wav files or --record" in stderr
    assert "traceback" not in stderr


def test_train_alarm_passes_paths_and_seed(monkeypatch, tmp_path, capsys):
    received = []
    fake_report = SimpleNamespace(
        deployment_threshold=0.73,
        oof_metrics=SimpleNamespace(
            positive_groups_triggered=7,
            positive_groups_total=7,
            negative_groups_triggered=0,
            negative_groups_total=10,
            false_triggers_per_minute=0.0,
        ),
    )
    monkeypatch.setattr(
        cli,
        "_train_alarm",
        lambda data, output, report, seed=0: (
            received.append((data, output, report, seed)) or fake_report
        ),
        raising=False,
    )

    cli.main([
        "train-alarm",
        "--data-dir",
        str(tmp_path / "data"),
        "--output",
        str(tmp_path / "head.npz"),
        "--seed",
        "9",
    ])

    assert received == [(
        tmp_path / "data",
        tmp_path / "head.npz",
        tmp_path / "fire_smoke_alarm_report.json",
        9,
    )]
    assert capsys.readouterr().out.strip() == (
        "trained fire_smoke_alarm threshold 0.730; recall 7/7; "
        "negative groups 0/10; false triggers/min 0.000"
    )


def test_train_alarm_uses_configured_companion_report(
        monkeypatch, tmp_path):
    model_path = tmp_path / "configured-head.npz"
    report_path = tmp_path / "configured-report.json"
    received = []
    fake_report = SimpleNamespace(
        deployment_threshold=0.5,
        oof_metrics=SimpleNamespace(
            positive_groups_triggered=1,
            positive_groups_total=1,
            negative_groups_triggered=0,
            negative_groups_total=1,
            false_triggers_per_minute=0.0,
        ),
    )
    monkeypatch.setattr(cli.config, "ALARM_MODEL_PATH", model_path)
    monkeypatch.setattr(cli.config, "ALARM_REPORT_PATH", report_path)
    monkeypatch.setattr(
        cli,
        "_train_alarm",
        lambda data, output, report, seed=0: (
            received.append((data, output, report, seed)) or fake_report
        ),
    )

    cli.main(["train-alarm"])

    assert received[0][1:3] == (model_path, report_path)


def test_evaluate_alarm_prints_files_and_scope(monkeypatch, tmp_path, capsys):
    received = []
    fake_report = SimpleNamespace(
        metrics=SimpleNamespace(files=({
            "label": "alarm",
            "triggered": True,
            "path": "alarm/a.wav",
        },)),
        payload={"evaluation_scope": "external_corpus"},
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_alarm",
        lambda data, model: received.append((data, model)) or fake_report,
        raising=False,
    )

    cli.main([
        "evaluate-alarm",
        "--data-dir",
        str(tmp_path / "data"),
        "--model",
        str(tmp_path / "head.npz"),
    ])

    output = capsys.readouterr().out
    assert received == [(tmp_path / "data", tmp_path / "head.npz")]
    assert "  alarm     True  alarm/a.wav\n" in output
    assert '"evaluation_scope": "external_corpus"' in output


def test_top5_displays_optional_alarm_score_and_loads_head_once(
        monkeypatch, tmp_path, capsys):
    model_path = tmp_path / "head.npz"
    model_calls = []
    embeddings = []

    class FakeYamNet:
        def __init__(self, model, class_map):
            model_calls.append((model, class_map))

        def infer(self, waveform):
            return np.array([0.8], dtype=np.float32), np.full(
                1024, 0.25, dtype=np.float32
            )

        def top(self, scores, k=5):
            return [("Siren", float(scores[0]))]

    class FakeHead:
        def score(self, embedding):
            embeddings.append(embedding.copy())
            return 0.42

    class FakeMicStream:
        def __init__(self, device=None):
            assert device == 3

        def windows(self):
            yield np.zeros(cli.config.WINDOW_SAMPLES, dtype=np.float32)

    loaded = []
    monkeypatch.setattr(cli.config, "ALARM_MODEL_PATH", model_path)
    monkeypatch.setattr(cli, "YamNet", FakeYamNet)
    monkeypatch.setattr(cli, "MicStream", FakeMicStream)
    monkeypatch.setattr(
        cli,
        "load_optional_alarm_head",
        lambda path, **kwargs: loaded.append((path, kwargs)) or FakeHead(),
        raising=False,
    )

    cli.main(["top5", "--device", "3"])

    output = capsys.readouterr().out
    assert "Siren 0.80" in output
    assert "fire_smoke_alarm 0.42" in output
    assert len(loaded) == 1
    assert loaded[0] == (model_path, {
        "yamnet_model_path": cli.config.MODEL_PATH,
        "class_map_path": cli.config.CLASS_MAP_PATH,
    })
    assert len(embeddings) == 1
    assert embeddings[0].shape == (1024,)


def test_top5_without_alarm_head_omits_trained_score(monkeypatch, capsys):
    loaded = []

    class FakeYamNet:
        def __init__(self, model, class_map):
            pass

        def infer(self, waveform):
            return np.array([0.8], dtype=np.float32), np.zeros(
                1024, dtype=np.float32
            )

        def top(self, scores, k=5):
            return [("Siren", float(scores[0]))]

    class FakeMicStream:
        def __init__(self, device=None):
            pass

        def windows(self):
            yield np.zeros(cli.config.WINDOW_SAMPLES, dtype=np.float32)

    monkeypatch.setattr(cli, "YamNet", FakeYamNet)
    monkeypatch.setattr(cli, "MicStream", FakeMicStream)
    monkeypatch.setattr(
        cli,
        "load_optional_alarm_head",
        lambda *args, **kwargs: loaded.append((args, kwargs)) or None,
        raising=False,
    )

    cli.main(["top5"])

    output = capsys.readouterr().out
    assert "Siren 0.80" in output
    assert "fire_smoke_alarm" not in output
    assert len(loaded) == 1


def test_top5_reports_malformed_alarm_head_before_listening(
        monkeypatch, capsys):
    class FakeYamNet:
        def __init__(self, model, class_map):
            pass

    def fail_load(*args, **kwargs):
        raise alarm_model.AlarmModelError("alarm head schema is malformed")

    monkeypatch.setattr(cli, "YamNet", FakeYamNet)
    monkeypatch.setattr(cli, "MicStream", _forbid_model_use)
    monkeypatch.setattr(
        cli, "load_optional_alarm_head", fail_load, raising=False
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["top5"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "alarm head schema is malformed" in stderr
    assert "traceback" not in stderr


def test_run_passes_configured_alarm_artifact(monkeypatch, tmp_path):
    received = []
    model_path = tmp_path / "fire_smoke_alarm_head.npz"

    class FakeEngine:
        def __init__(self, **kwargs):
            received.append(kwargs)

        def learned_sounds(self):
            return []

        def run(self):
            return None

    monkeypatch.setattr(cli.config, "ALARM_MODEL_PATH", model_path)
    monkeypatch.setattr(cli, "EarshotML", FakeEngine)

    cli.main(["run"])

    assert received[0]["alarm_model_path"] == model_path


@pytest.mark.parametrize(
    ("command", "diagnostic"),
    [
        ("collect", "corpus destination is invalid"),
        ("train", "training corpus has no groups"),
        ("evaluate", "evaluation artifact is incompatible"),
    ],
)
def test_alarm_workflow_errors_are_concise(
        monkeypatch, capsys, command, diagnostic):
    from earshot_ml.alarm_training import TrainingError

    if command == "collect":
        def fail(*args, **kwargs):
            raise cli.AlarmDataError(diagnostic)

        monkeypatch.setattr(cli, "collect_files", fail)
        arguments = ["collect", "alarm", "source.wav"]
    elif command == "train":
        def fail(*args, **kwargs):
            raise TrainingError(diagnostic)

        monkeypatch.setattr(cli, "_train_alarm", fail)
        arguments = ["train-alarm"]
    else:
        def fail(*args, **kwargs):
            raise TrainingError(diagnostic)

        monkeypatch.setattr(cli, "_evaluate_alarm", fail)
        arguments = ["evaluate-alarm"]

    with pytest.raises(SystemExit) as exc_info:
        cli.main(arguments)

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert diagnostic in stderr
    assert "traceback" not in stderr


def test_non_training_commands_do_not_import_sklearn(monkeypatch):
    class FakeYamNet:
        def __init__(self, *args, **kwargs):
            pass

    class FakeMicStream:
        def __init__(self, device=None):
            pass

        def windows(self):
            return iter(())

    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def learned_sounds(self):
            return []

        def run(self):
            return None

        def teach(self, name, clips):
            return len(clips)

    class FakeStore:
        def learned(self):
            return []

        def forget(self, name):
            return 0

        def save(self):
            return None

    monkeypatch.setattr(cli, "download_artifact", lambda artifact: False)
    monkeypatch.setattr(cli, "YamNet", FakeYamNet)
    monkeypatch.setattr(cli, "MicStream", FakeMicStream)
    monkeypatch.setattr(cli, "EarshotML", FakeEngine)
    monkeypatch.setattr(cli, "_teach_store", FakeStore)
    monkeypatch.setattr(
        cli, "load_optional_alarm_head", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli,
        "collect_files",
        lambda *args, **kwargs: (Path("stored.wav"),),
    )
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            pytest.fail(f"non-training command imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)

    cli.main(["download"])
    cli.main(["top5"])
    cli.main(["run"])
    cli.main(["teach", "custom", "clip.wav"])
    cli.main(["sounds"])
    cli.main(["forget", "custom"])
    cli.main(["collect", "alarm", "clip.wav"])


def test_main_parses_supplied_argv(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "cmd_top5", lambda args: received.append(args))

    assert cli.main(["top5", "--device", "7"]) is None

    assert len(received) == 1
    assert received[0].command == "top5"
    assert received[0].device == 7


def test_backend_error_is_concise_without_traceback(monkeypatch, capsys):
    def unavailable_backend(*args, **kwargs):
        raise pipeline.InterpreterBackendError(
            "install tflite-runtime, ai-edge-litert, or tensorflow"
        )

    monkeypatch.setattr(cli, "YamNet", unavailable_backend)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["top5"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "tflite-runtime" in stderr
    assert "ai-edge-litert" in stderr
    assert "tensorflow" in stderr
    assert "traceback" not in stderr


def test_unexpected_os_error_is_not_hidden_as_an_expected_cli_error(
    monkeypatch,
):
    failure = OSError("unexpected native failure")

    def fail_unexpectedly(args):
        raise failure

    monkeypatch.setattr(cli, "cmd_top5", fail_unexpectedly)

    with pytest.raises(OSError) as exc_info:
        cli.main(["top5"])

    assert exc_info.value is failure


def test_recording_device_error_is_concise_without_traceback(
    monkeypatch, capsys
):
    class FakeEngine:
        def __init__(self, **kwargs):
            pass

        def teach(self, name, clips):
            pytest.fail("device failure must stop before teaching")

    def failed_rec(*args, **kwargs):
        raise RuntimeError("USB microphone disconnected")

    monkeypatch.setattr(cli, "EarshotML", FakeEngine)
    monkeypatch.setattr(builtins, "input", lambda prompt: "")
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "sounddevice",
        type("FakeSoundDevice", (), {"rec": staticmethod(failed_rec)})(),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["teach", "kettle", "--record", "1", "--seconds", "0.01"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err
    assert "USB microphone disconnected" in stderr
    assert "python -m sounddevice" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("wav_kind", ["malformed", "unsupported-24-bit"])
def test_teach_reports_wav_errors_concisely(
    tmp_path, monkeypatch, capsys, wav_kind
):
    path = tmp_path / f"{wav_kind}.wav"
    if wav_kind == "malformed":
        path.write_bytes(b"not a RIFF/WAVE file")
        diagnostic = "RIFF"
    else:
        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(3)
            wav_file.setframerate(16_000)
            wav_file.writeframes(b"\0" * 6)
        diagnostic = "unsupported sample width 3"
    monkeypatch.setattr(
        cli,
        "EarshotML",
        _real_teach_engine_factory(tmp_path / "taught.npz"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["teach", "custom", str(path)])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err
    assert path.name in stderr
    assert diagnostic in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize(
    ("class_map_data", "diagnostic"),
    [
        (b"index,mid,name\n0,/m/0,test\n", "display_name"),
        (b"index,mid,display_name\nnot-an-index,,\n", "index"),
    ],
)
def test_download_reports_malformed_class_map_concisely(
    tmp_path, monkeypatch, capsys, class_map_data, diagnostic
):
    model_artifact = _artifact(
        tmp_path / "source-model.tflite",
        tmp_path / "models" / "yamnet.tflite",
        b"placeholder model",
    )
    class_artifact = _artifact(
        tmp_path / "source-map.csv",
        tmp_path / "models" / "yamnet_class_map.csv",
        class_map_data,
    )
    monkeypatch.setattr(cli.config, "MODEL_ARTIFACT", model_artifact)
    monkeypatch.setattr(cli.config, "CLASS_MAP_ARTIFACT", class_artifact)
    monkeypatch.setattr(cli.config, "MODEL_PATH", model_artifact.path)
    monkeypatch.setattr(cli.config, "CLASS_MAP_PATH", class_artifact.path)
    monkeypatch.setattr(
        pipeline,
        "_load_interpreter",
        lambda path: pytest.fail("backend loaded for malformed class map"),
    )

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["download"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err
    assert class_artifact.path.name in stderr
    assert diagnostic in stderr
    assert "Traceback" not in stderr


def test_forget_reports_store_replace_error_concisely(
    tmp_path, monkeypatch, capsys
):
    store_path = tmp_path / "taught.npz"
    _store_with_sounds(store_path)
    before = store_path.read_bytes()
    failure = PermissionError("taught store is read-only")
    monkeypatch.setattr(cli.config, "TAUGHT_STORE_PATH", store_path)

    def fail_replace(source, destination):
        raise failure

    monkeypatch.setattr(core.os, "replace", fail_replace)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["forget", "kettle"])

    assert exc_info.value.code != 0
    store_error = exc_info.value.__context__
    assert isinstance(store_error, core.TeachStoreError)
    assert store_error.__cause__ is failure
    stderr = capsys.readouterr().err
    assert "taught store is read-only" in stderr
    assert "Traceback" not in stderr
    assert store_path.read_bytes() == before
    assert not store_path.with_name(store_path.name + ".part").exists()


def test_teach_reports_store_write_error_concisely_and_rolls_back(
    tmp_path, monkeypatch, capsys
):
    store_path = tmp_path / "taught.npz"
    constructed = []
    failure = OSError("disk full while writing taught store")
    monkeypatch.setattr(
        cli,
        "EarshotML",
        _real_teach_engine_factory(store_path, constructed),
    )
    monkeypatch.setattr(builtins, "input", lambda prompt: "")
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        cli,
        "record",
        lambda *args, **kwargs: np.ones(
            pipeline.WINDOW_SAMPLES, dtype=np.float32
        ),
    )

    def fail_savez(*args, **kwargs):
        raise failure

    monkeypatch.setattr(core.np, "savez", fail_savez)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["teach", "custom", "--record", "1"])

    assert exc_info.value.code != 0
    store_error = exc_info.value.__context__
    assert isinstance(store_error, core.TeachStoreError)
    assert store_error.__cause__ is failure
    stderr = capsys.readouterr().err
    assert "disk full while writing taught store" in stderr
    assert "Traceback" not in stderr
    assert constructed[0].learned_sounds() == []
    assert not store_path.exists()
    assert not store_path.with_name(store_path.name + ".part").exists()


def test_sounds_reads_store_without_model_artifacts_or_backend(
    tmp_path, monkeypatch, capsys
):
    store_path = tmp_path / "state" / "taught_sounds.npz"
    _store_with_sounds(store_path)
    monkeypatch.setattr(cli.config, "TAUGHT_STORE_PATH", store_path)
    monkeypatch.setattr(cli.config, "MODEL_PATH", tmp_path / "missing.tflite")
    monkeypatch.setattr(cli.config, "CLASS_MAP_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(cli, "EarshotML", _forbid_model_use)
    monkeypatch.setattr(cli, "YamNet", _forbid_model_use)
    _forbid_tflite_imports(monkeypatch)

    assert cli.main(["sounds"]) is None

    output = capsys.readouterr().out
    assert "kettle  (2 clips)" in output
    assert "dryer  (1 clips)" in output


def test_sounds_reports_corrupt_store_without_traceback(
    tmp_path, monkeypatch, capsys
):
    store_path = tmp_path / "taught_sounds.npz"
    store_path.write_bytes(b"not an npz archive")
    monkeypatch.setattr(cli.config, "TAUGHT_STORE_PATH", store_path)
    monkeypatch.setattr(cli, "EarshotML", _forbid_model_use)
    monkeypatch.setattr(cli, "YamNet", _forbid_model_use)
    _forbid_tflite_imports(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["sounds"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "taught store" in stderr
    assert "traceback" not in stderr


def test_forget_updates_store_atomically_without_model_artifacts_or_backend(
    tmp_path, monkeypatch, capsys
):
    store_path = tmp_path / "state" / "taught_sounds.npz"
    _store_with_sounds(store_path)
    monkeypatch.setattr(cli.config, "TAUGHT_STORE_PATH", store_path)
    monkeypatch.setattr(cli.config, "MODEL_PATH", tmp_path / "missing.tflite")
    monkeypatch.setattr(cli.config, "CLASS_MAP_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(cli, "EarshotML", _forbid_model_use)
    monkeypatch.setattr(cli, "YamNet", _forbid_model_use)
    _forbid_tflite_imports(monkeypatch)

    assert cli.main(["forget", "kettle"]) is None

    assert "removed 2 clips of 'kettle'" in capsys.readouterr().out
    assert TeachStore(store_path).learned() == [{"name": "dryer", "clips": 1}]
    assert not store_path.with_name(store_path.name + ".part").exists()


def test_teach_rejects_reserved_name_before_engine_recording_or_files(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli, "EarshotML", _forbid_model_use)
    monkeypatch.setattr(cli, "YamNet", _forbid_model_use)
    monkeypatch.setattr(cli, "record", _forbid_model_use)
    monkeypatch.setattr(builtins, "input", _forbid_model_use)
    monkeypatch.setattr(cli.config, "MODEL_PATH", tmp_path / "missing.tflite")
    monkeypatch.setattr(cli.config, "CLASS_MAP_PATH", tmp_path / "missing.csv")

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["teach", "  SMOKE_ALARM  ", "--record", "1"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "smoke_alarm" in stderr
    assert "pretrained label" in stderr
    assert "traceback" not in stderr


def test_teach_rejects_trained_alarm_name_before_any_side_effect(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "EarshotML", _forbid_model_use)
    monkeypatch.setattr(cli, "YamNet", _forbid_model_use)
    monkeypatch.setattr(cli, "record", _forbid_model_use)
    monkeypatch.setattr(builtins, "input", _forbid_model_use)
    monkeypatch.setattr(cli.config, "MODEL_PATH", tmp_path / "missing.tflite")
    monkeypatch.setattr(cli.config, "CLASS_MAP_PATH", tmp_path / "missing.csv")

    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            "teach",
            "  FIRE_SMOKE_ALARM  ",
            str(tmp_path / "missing.wav"),
            "--record",
            "1",
        ])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "fire_smoke_alarm" in stderr
    assert "trained" in stderr
    assert "traceback" not in stderr


@pytest.mark.parametrize(
    ("arguments", "diagnostic"),
    [
        (["custom", "clip.wav", "--record", "-1"], "--record"),
        (["custom", "--record", "1", "--seconds", "0"], "--seconds"),
        (["custom", "--record", "1", "--seconds", "-1"], "--seconds"),
        (["custom", "--record", "1", "--seconds", "nan"], "--seconds"),
        (["custom", "--record", "1", "--seconds", "inf"], "--seconds"),
    ],
)
def test_teach_rejects_invalid_record_options_before_engine_or_recording(
    monkeypatch, capsys, arguments, diagnostic
):
    monkeypatch.setattr(cli, "EarshotML", _forbid_model_use)
    monkeypatch.setattr(cli, "record", _forbid_model_use)
    monkeypatch.setattr(builtins, "input", _forbid_model_use)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["teach", *arguments])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert diagnostic in stderr
    assert "positive" in stderr or "non-negative" in stderr
    assert "traceback" not in stderr


def test_download_checksum_error_is_concise_and_cleans_partial_file(
    tmp_path, monkeypatch, capsys
):
    destination = tmp_path / "models" / "yamnet.tflite"
    bad_artifact = _artifact(
        tmp_path / "source-model.tflite",
        destination,
        b"not the expected model",
        sha256="0" * 64,
    )
    class_artifact = _artifact(
        tmp_path / "source-map.csv",
        tmp_path / "models" / "yamnet_class_map.csv",
        b"index,mid,display_name\n",
    )
    monkeypatch.setattr(cli.config, "MODEL_ARTIFACT", bad_artifact)
    monkeypatch.setattr(cli.config, "CLASS_MAP_ARTIFACT", class_artifact)
    monkeypatch.setattr(cli.config, "MODEL_PATH", bad_artifact.path)
    monkeypatch.setattr(cli.config, "CLASS_MAP_PATH", class_artifact.path)
    monkeypatch.setattr(cli, "YamNet", _forbid_model_use)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["download"])

    assert exc_info.value.code != 0
    stderr = capsys.readouterr().err.lower()
    assert "checksum mismatch" in stderr
    assert "retry" in stderr
    assert "traceback" not in stderr
    assert not destination.exists()
    assert not destination.with_name(destination.name + ".part").exists()


def test_download_installs_both_artifacts_then_validates_configured_paths(
    tmp_path, monkeypatch, capsys
):
    model_data = b"local test model"
    class_map_data = b"index,mid,display_name\n0,/m/0,test class\n"
    model_artifact = _artifact(
        tmp_path / "source-model.tflite",
        tmp_path / "models" / "yamnet.tflite",
        model_data,
    )
    class_artifact = _artifact(
        tmp_path / "source-map.csv",
        tmp_path / "models" / "yamnet_class_map.csv",
        class_map_data,
    )
    monkeypatch.setattr(cli.config, "MODEL_ARTIFACT", model_artifact)
    monkeypatch.setattr(cli.config, "CLASS_MAP_ARTIFACT", class_artifact)
    monkeypatch.setattr(cli.config, "MODEL_PATH", model_artifact.path)
    monkeypatch.setattr(cli.config, "CLASS_MAP_PATH", class_artifact.path)
    validated = []

    class FakeYamNet:
        def __init__(self, model_path, class_map_path):
            assert model_artifact.path.read_bytes() == model_data
            assert class_artifact.path.read_bytes() == class_map_data
            validated.append((model_path, class_map_path))

    monkeypatch.setattr(cli, "YamNet", FakeYamNet)

    assert cli.main(["download"]) is None
    first_output = capsys.readouterr().out.lower()
    assert "downloaded yamnet.tflite" in first_output
    assert "downloaded yamnet_class_map.csv" in first_output
    assert "validated" in first_output
    assert validated == [(model_artifact.path, class_artifact.path)]

    assert cli.main(["download"]) is None
    second_output = capsys.readouterr().out.lower()
    assert "cached yamnet.tflite" in second_output
    assert "cached yamnet_class_map.csv" in second_output
    assert validated == [
        (model_artifact.path, class_artifact.path),
        (model_artifact.path, class_artifact.path),
    ]


def test_top_level_cli_is_exact_compatibility_wrapper():
    wrapper = (ROOT / "cli.py").read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    )

    assert wrapper == (
        "from earshot_ml.cli import main\n"
        "\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    main()\n"
    )
