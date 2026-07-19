import builtins
import hashlib
from pathlib import Path
import wave

import numpy as np
import pytest

from earshot_ml import cli, core, pipeline
from earshot_ml.artifacts import Artifact
from earshot_ml.core import EarshotML, TeachStore


ROOT = Path(__file__).resolve().parent.parent
COMMANDS = ("download", "top5", "run", "teach", "sounds", "forget")
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
