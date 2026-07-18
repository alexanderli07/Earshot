import builtins
import inspect
from pathlib import Path
import sys
import threading
import types
import wave

import numpy as np
import pytest

from earshot_ml import pipeline


WINDOW_SAMPLES = pipeline.WINDOW_SAMPLES


def _write_pcm_wav(path, payload, *, width, channels=1, samplerate=16_000):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(samplerate)
        wav_file.writeframes(payload)
    return path


@pytest.mark.parametrize(
    ("width", "samples", "scale", "offset"),
    [
        (1, np.array([0, 128, 255], dtype=np.uint8), 128.0, 128.0),
        (2, np.array([-32768, 0, 32767], dtype="<i2"), 32768.0, 0.0),
        (
            4,
            np.array([-2147483648, 0, 2147483647], dtype="<i4"),
            2147483648.0,
            0.0,
        ),
    ],
)
def test_load_wav_decodes_supported_uncompressed_pcm_widths(
    tmp_path, width, samples, scale, offset
):
    path = _write_pcm_wav(
        tmp_path / f"pcm-{width}.wav", samples.tobytes(), width=width
    )

    audio = pipeline.load_wav_16k_mono(path)

    expected = (samples.astype(np.float32) - offset) / scale
    assert audio.dtype == np.float32
    np.testing.assert_allclose(audio, expected)


def test_load_wav_downmixes_stereo_channels(tmp_path):
    frames = np.array(
        [[32767, -32768], [16384, 0], [-16384, 16384]], dtype="<i2"
    )
    path = _write_pcm_wav(
        tmp_path / "stereo.wav", frames.tobytes(), width=2, channels=2
    )

    audio = pipeline.load_wav_16k_mono(path)

    expected = (frames.astype(np.float32) / 32768.0).mean(axis=1)
    np.testing.assert_allclose(audio, expected)


def test_load_wav_rejects_24_bit_pcm_with_width_diagnostic(tmp_path):
    path = _write_pcm_wav(tmp_path / "pcm-24.wav", b"\0" * 6, width=3)

    with pytest.raises(pipeline.AudioFileError) as exc_info:
        pipeline.load_wav_16k_mono(path)

    assert "unsupported sample width 3" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_load_wav_wraps_malformed_file_with_original_diagnostic(tmp_path):
    path = tmp_path / "malformed.wav"
    path.write_bytes(b"not a RIFF/WAVE file")

    with pytest.raises(pipeline.AudioFileError) as exc_info:
        pipeline.load_wav_16k_mono(path)

    assert str(path) in str(exc_info.value)
    assert "RIFF" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, wave.Error)


def test_load_wav_preserves_empty_non_16k_audio(tmp_path):
    path = _write_pcm_wav(
        tmp_path / "empty-8k.wav", b"", width=2, samplerate=8_000
    )

    audio = pipeline.load_wav_16k_mono(path)

    assert audio.shape == (0,)
    assert audio.dtype == np.float32


def test_resample_linear_has_scaled_length_and_preserves_endpoints():
    audio = np.array([0.25, 1.0, -0.5], dtype=np.float32)

    result = pipeline.resample_linear(audio, 2, 4)

    assert result.shape == (6,)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(audio[0])
    assert result[-1] == pytest.approx(audio[-1])


@pytest.mark.parametrize("audio", [0.5, np.zeros((2, 2), dtype=np.float32)])
def test_resample_linear_requires_one_dimensional_audio(audio):
    with pytest.raises(ValueError, match="one-dimensional"):
        pipeline.resample_linear(audio, 16_000, 8_000)


@pytest.mark.parametrize("sample_rate", [0, -1, np.nan, np.inf, "invalid"])
def test_resample_linear_requires_positive_finite_sample_rates(sample_rate):
    with pytest.raises(ValueError, match="sample rates.*positive.*finite"):
        pipeline.resample_linear(np.zeros(2), sample_rate, 16_000)


@pytest.mark.parametrize("audio", [np.array([], np.float32), np.array([2.0])])
def test_clip_windows_zero_pads_empty_and_short_audio(audio):
    windows = pipeline.clip_windows(audio, window=4, hop=2)

    assert len(windows) == 1
    np.testing.assert_array_equal(
        windows[0], np.pad(audio, (0, 4 - len(audio))).astype(np.float32)
    )


def test_clip_windows_uses_overlapping_window_and_hop_geometry():
    windows = pipeline.clip_windows(np.arange(10), window=4, hop=2)

    assert [window.tolist() for window in windows] == [
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 5.0],
        [4.0, 5.0, 6.0, 7.0],
        [6.0, 7.0, 8.0, 9.0],
    ]


@pytest.mark.parametrize("audio", [0.5, np.zeros((2, 2), dtype=np.float32)])
def test_clip_windows_requires_one_dimensional_audio(audio):
    with pytest.raises(ValueError, match="one-dimensional"):
        pipeline.clip_windows(audio, window=4, hop=2)


@pytest.mark.parametrize("window", [0, -1, 1.5, True])
def test_clip_windows_requires_a_positive_integer_window(window):
    with pytest.raises(ValueError, match="window must be a positive integer"):
        pipeline.clip_windows(np.zeros(4), window=window, hop=2)


@pytest.mark.parametrize("hop", [0, -1, 1.5, True])
def test_clip_windows_requires_a_positive_integer_hop(hop):
    with pytest.raises(ValueError, match="hop must be a positive integer"):
        pipeline.clip_windows(np.zeros(4), window=4, hop=hop)


def test_latest_block_queue_discards_oldest_when_full():
    blocks = pipeline.LatestBlockQueue(maxsize=2)

    blocks.put_latest(1)
    blocks.put_latest(2)
    blocks.put_latest(3)

    assert blocks.get(timeout=0.01) == 2
    assert blocks.get(timeout=0.01) == 3


def test_mic_stream_clears_overlap_after_a_dropped_block(monkeypatch):
    first_consumed = threading.Event()
    post_gap_blocks_queued = threading.Event()
    producer_threads = []
    real_queue = pipeline.LatestBlockQueue

    class CoordinatedQueue(real_queue):
        def _wait_for_gap_producer(self, item):
            if not first_consumed.is_set():
                first_consumed.set()
                assert post_gap_blocks_queued.wait(timeout=1)
            return item

        def get(self, *args, **kwargs):
            return self._wait_for_gap_producer(super().get(*args, **kwargs))

        def _get_sequenced(self, *args, **kwargs):
            item = super()._get_sequenced(*args, **kwargs)
            return self._wait_for_gap_producer(item)

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            self.callback(
                np.array([[1.0], [2.0]], dtype=np.float32), 2, None, None
            )

            def produce_gap():
                assert first_consumed.wait(timeout=1)
                for values in ([3.0, 4.0], [5.0, 6.0], [7.0, 8.0]):
                    self.callback(
                        np.asarray(values, dtype=np.float32).reshape(-1, 1),
                        2,
                        None,
                        None,
                    )
                post_gap_blocks_queued.set()

            producer = threading.Thread(target=produce_gap, daemon=True)
            producer_threads.append(producer)
            producer.start()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            producer_threads[0].join(timeout=1)
            return False

    monkeypatch.setattr(pipeline, "LatestBlockQueue", CoordinatedQueue)
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeInputStream),
    )
    windows = pipeline.MicStream(window=4, hop=2, queue_size=2).windows()

    result = next(windows)
    windows.close()

    np.testing.assert_array_equal(
        result,
        np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
    )


def test_mic_stream_stop_event_closes_stream_context(monkeypatch):
    lifecycle = []

    class FakeInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            lifecycle.append("enter")
            self.callback(
                np.array([[1.0], [2.0]], dtype=np.float32),
                2,
                None,
                None,
            )
            self.callback(
                np.array([[3.0], [4.0]], dtype=np.float32),
                2,
                None,
                None,
            )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            lifecycle.append("exit")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeInputStream),
    )
    stop_event = threading.Event()
    windows = pipeline.MicStream(window=4, hop=2).windows(
        stop_event=stop_event)

    np.testing.assert_array_equal(
        next(windows),
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
    )
    assert lifecycle == ["enter"]

    stop_event.set()

    with pytest.raises(StopIteration):
        next(windows)
    assert lifecycle == ["enter", "exit"]


def test_mic_stream_stop_event_defaults_to_none():
    stop_event = inspect.signature(
        pipeline.MicStream.windows).parameters["stop_event"]

    assert stop_event.default is None


@pytest.mark.parametrize("failure_stage", ["initialize", "enter"])
def test_mic_stream_wraps_input_stream_failures_with_device_guidance(
    monkeypatch, failure_stage
):
    class FakeInputStream:
        def __init__(self, **kwargs):
            if failure_stage == "initialize":
                raise RuntimeError("no default input device")

        def __enter__(self):
            raise RuntimeError("PortAudio stream unavailable")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeInputStream),
    )

    with pytest.raises(pipeline.AudioDeviceError) as exc_info:
        next(pipeline.MicStream(window=4, hop=2).windows())

    message = str(exc_info.value)
    expected_diagnostic = (
        "no default input device"
        if failure_stage == "initialize"
        else "PortAudio stream unavailable"
    )
    assert expected_diagnostic in message
    assert "python -m sounddevice" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize("failure_stage", ["record", "wait"])
def test_record_wraps_sounddevice_failures_with_device_guidance(
    monkeypatch, failure_stage
):
    def fake_rec(*args, **kwargs):
        if failure_stage == "record":
            raise RuntimeError("invalid input device 9")
        return np.zeros((4, 1), dtype=np.float32)

    def fake_wait():
        if failure_stage == "wait":
            raise RuntimeError("recording stream stopped")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(rec=fake_rec, wait=fake_wait),
    )

    with pytest.raises(pipeline.AudioDeviceError) as exc_info:
        pipeline.record(0.001, samplerate=4_000)

    message = str(exc_info.value)
    assert (
        "invalid input device 9" in message
        or "recording stream stopped" in message
    )
    assert "python -m sounddevice" in message
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_record_wraps_sounddevice_import_failure_with_device_guidance(
    monkeypatch,
):
    real_import = builtins.__import__

    def missing_sounddevice(name, *args, **kwargs):
        if name == "sounddevice":
            raise ModuleNotFoundError("No module named 'sounddevice'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_sounddevice)

    with pytest.raises(pipeline.AudioDeviceError) as exc_info:
        pipeline.record(0.001, samplerate=4_000)

    assert "No module named 'sounddevice'" in str(exc_info.value)
    assert "python -m sounddevice" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


def test_mic_stream_wraps_close_failure_when_generator_exits_normally(
    monkeypatch,
):
    class FakeInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            self.callback(
                np.arange(4, dtype=np.float32).reshape(-1, 1), 4, None, None
            )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            raise RuntimeError("PortAudio close failed")

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeInputStream),
    )
    stop_event = threading.Event()
    windows = pipeline.MicStream(window=4, hop=2).windows(stop_event)
    next(windows)
    stop_event.set()

    with pytest.raises(pipeline.AudioDeviceError) as exc_info:
        next(windows)

    assert "PortAudio close failed" in str(exc_info.value)
    assert "python -m sounddevice" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_mic_stream_does_not_wrap_application_errors_thrown_into_generator(
    monkeypatch,
):
    class FakeInputStream:
        def __init__(self, **kwargs):
            self.callback = kwargs["callback"]

        def __enter__(self):
            self.callback(
                np.arange(4, dtype=np.float32).reshape(-1, 1), 4, None, None
            )
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeInputStream),
    )
    windows = pipeline.MicStream(window=4, hop=2).windows()
    next(windows)
    application_error = RuntimeError("application event callback failed")

    with pytest.raises(RuntimeError) as exc_info:
        windows.throw(application_error)

    assert exc_info.value is application_error
    assert not isinstance(exc_info.value, getattr(pipeline, "AudioDeviceError", ()))


def test_mic_stream_does_not_wrap_keyboard_interrupt(monkeypatch):
    class FakeInputStream:
        def __init__(self, **kwargs):
            raise KeyboardInterrupt

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(InputStream=FakeInputStream),
    )

    with pytest.raises(KeyboardInterrupt):
        next(pipeline.MicStream(window=4, hop=2).windows())


def tensor_detail(index, shape, *, dtype=np.float32, name=None,
                  shape_signature=None):
    shape = np.asarray(shape, dtype=np.int32)
    if shape_signature is None:
        shape_signature = shape
    return {
        "index": index,
        "name": name or f"tensor_{index}",
        "shape": shape.copy(),
        "shape_signature": np.asarray(shape_signature, dtype=np.int32),
        "dtype": dtype,
    }


def _copy_detail(detail):
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in detail.items()
    }


class FakeInterpreter:
    def __init__(self, *, input_detail=None, outputs=None, values=None,
                 resize_applies=True, resize_error=None):
        self.input_detail = input_detail or tensor_detail(
            1, [WINDOW_SAMPLES], name="waveform")
        self.outputs = outputs or [
            tensor_detail(7, [2, 521], name="scores"),
            tensor_detail(8, [1, 1024], name="embeddings"),
        ]
        self.values = values or {
            7: np.zeros((2, 521), dtype=np.float32),
            8: np.zeros((1, 1024), dtype=np.float32),
        }
        self.resize_applies = resize_applies
        self.resize_error = resize_error
        self.resize_calls = []
        self.allocate_calls = 0
        self.get_input_details_calls = 0
        self.get_output_details_calls = 0
        self.set_calls = []
        self.invoke_calls = 0
        self.events = []
        self.allocated = False

    def get_input_details(self):
        self.get_input_details_calls += 1
        return [_copy_detail(self.input_detail)]

    def resize_tensor_input(self, index, shape):
        assert not self.allocated
        resized = [int(value) for value in shape]
        self.resize_calls.append((index, resized))
        self.events.append("resize")
        if self.resize_error is not None:
            raise self.resize_error
        if self.resize_applies:
            self.input_detail["shape"] = np.asarray(resized, dtype=np.int32)

    def allocate_tensors(self):
        self.allocate_calls += 1
        self.events.append("allocate")
        self.allocated = True

    def get_output_details(self):
        assert self.allocated
        self.get_output_details_calls += 1
        return [_copy_detail(detail) for detail in self.outputs]

    def set_tensor(self, index, value):
        assert self.allocated
        self.set_calls.append((index, np.array(value, copy=True)))
        self.events.append("set")

    def invoke(self):
        assert self.set_calls
        self.invoke_calls += 1
        self.events.append("invoke")

    def get_tensor(self, index):
        assert self.invoke_calls
        return np.array(self.values[index], copy=True)


def write_class_map(tmp_path, rows=521):
    path = tmp_path / "class_map.csv"
    lines = ["index,mid,display_name"]
    lines.extend(f"{index},/m/{index},class {index}" for index in range(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("contents", "diagnostic"),
    [
        ("index,mid,name\n0,/m/0,test\n", "display_name"),
        ("index,mid,display_name\nnot-an-index,,\n", "index"),
    ],
)
def test_load_class_names_wraps_schema_and_content_errors(
    tmp_path, contents, diagnostic
):
    path = tmp_path / "bad-class-map.csv"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        pipeline.load_class_names(path)

    message = str(exc_info.value)
    assert str(path) in message
    assert diagnostic in message
    assert isinstance(exc_info.value.__cause__, ValueError)


def make_model(tmp_path, fake, *, rows=521):
    return pipeline.YamNet(
        Path("unused.tflite"),
        write_class_map(tmp_path, rows),
        interpreter=fake,
    )


def test_valid_model_averages_frames_and_invokes_interpreter(tmp_path):
    score_frames = np.stack([
        np.zeros(521, dtype=np.float32),
        np.full(521, 2.0, dtype=np.float32),
    ])
    embedding_frames = np.full((1, 1024), 3.0, dtype=np.float32)
    fake = FakeInterpreter(values={7: score_frames, 8: embedding_frames})

    model = make_model(tmp_path, fake)
    scores, embedding = model.infer(np.arange(WINDOW_SAMPLES))

    assert scores.shape == (521,)
    assert embedding.shape == (1024,)
    np.testing.assert_allclose(scores, 1.0)
    np.testing.assert_allclose(embedding, 3.0)
    assert fake.resize_calls == []
    assert fake.allocate_calls == 1
    assert fake.get_input_details_calls >= 2
    assert fake.get_output_details_calls == 1
    assert fake.events == ["allocate", "set", "invoke"]
    set_index, set_value = fake.set_calls[0]
    assert set_index == 1
    assert set_value.shape == (WINDOW_SAMPLES,)
    assert set_value.dtype == np.float32


def test_infer_serializes_interpreter_mutation_across_threads(tmp_path):
    class CoordinatedInterpreter(FakeInterpreter):
        def __init__(self):
            super().__init__()
            self.state_lock = threading.Lock()
            self.active = 0
            self.maximum_active = 0
            self.call_numbers = {}
            self.get_counts = {}
            self.first_invoke_started = threading.Event()
            self.second_attempted = threading.Event()
            self.second_set = threading.Event()
            self.release_first = threading.Event()

        def set_tensor(self, index, value):
            thread_id = threading.get_ident()
            with self.state_lock:
                call_number = len(self.call_numbers) + 1
                self.call_numbers[thread_id] = call_number
                self.active += 1
                self.maximum_active = max(self.maximum_active, self.active)
                if call_number == 2:
                    self.second_set.set()
            super().set_tensor(index, value)

        def invoke(self):
            call_number = self.call_numbers[threading.get_ident()]
            if call_number == 1:
                self.first_invoke_started.set()
                assert self.second_attempted.wait(timeout=1)
                assert self.release_first.wait(timeout=1)
            super().invoke()

        def get_tensor(self, index):
            value = super().get_tensor(index)
            thread_id = threading.get_ident()
            with self.state_lock:
                count = self.get_counts.get(thread_id, 0) + 1
                self.get_counts[thread_id] = count
                if count == 2:
                    self.active -= 1
            return value

    fake = CoordinatedInterpreter()
    model = make_model(tmp_path, fake)
    errors = []

    def infer(*, second=False):
        if second:
            fake.second_attempted.set()
        try:
            model.infer(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
        except BaseException as exc:  # captured and asserted in the test thread
            errors.append(exc)

    first = threading.Thread(target=infer)
    second = threading.Thread(target=infer, kwargs={"second": True})
    first.start()
    assert fake.first_invoke_started.wait(timeout=1)
    second.start()
    assert fake.second_attempted.wait(timeout=1)

    second_entered_while_first_was_active = fake.second_set.wait(timeout=0.2)
    fake.release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert not second_entered_while_first_was_active
    assert fake.maximum_active == 1


def test_score_only_model_reports_observed_output_contract(tmp_path):
    outputs = [tensor_detail(7, [1, 521], name="score_frames")]
    fake = FakeInterpreter(
        outputs=outputs,
        values={7: np.zeros((1, 521), dtype=np.float32)},
    )

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        make_model(tmp_path, fake)

    message = str(exc_info.value)
    assert "1024" in message
    assert "score_frames" in message
    assert "[1, 521]" in message
    assert "float32" in message


@pytest.mark.parametrize(
    ("quantized_index", "quantized_width", "quantized_name"),
    [
        (7, 521, "quantized_scores"),
        (8, 1024, "quantized_embedding"),
    ],
)
def test_required_model_outputs_reject_quantized_tensor_details(
    tmp_path, quantized_index, quantized_width, quantized_name
):
    outputs = [
        tensor_detail(7, [1, 521], name="scores"),
        tensor_detail(8, [1, 1024], name="embedding"),
    ]
    outputs[quantized_index - 7] = tensor_detail(
        quantized_index,
        [1, quantized_width],
        dtype=np.int8,
        name=quantized_name,
    )
    fake = FakeInterpreter(outputs=outputs)

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        make_model(tmp_path, fake)

    message = str(exc_info.value)
    assert "exactly one float32" in message
    assert quantized_name in message
    assert "int8" in message
    assert f"[1, {quantized_width}]" in message


@pytest.mark.parametrize("duplicate_width", [521, 1024])
def test_required_model_outputs_reject_ambiguous_duplicate_widths(
    tmp_path, duplicate_width
):
    duplicate_name = f"duplicate_{duplicate_width}"
    outputs = [
        tensor_detail(7, [1, 521], name="scores"),
        tensor_detail(8, [1, 1024], name="embedding"),
        tensor_detail(9, [2, duplicate_width], name=duplicate_name),
    ]
    fake = FakeInterpreter(outputs=outputs)

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        make_model(tmp_path, fake)

    message = str(exc_info.value)
    assert "exactly one float32" in message
    assert duplicate_name in message
    assert f"[2, {duplicate_width}]" in message


@pytest.mark.parametrize(
    ("index", "value", "diagnostic"),
    [
        (7, np.zeros((1, 521), dtype=np.float64), "float32"),
        (7, np.zeros((0, 521), dtype=np.float32), "nonempty"),
        (7, np.zeros((1, 520), dtype=np.float32), "521"),
        (8, np.full((1, 1024), np.nan, dtype=np.float32), "finite"),
    ],
)
def test_infer_rejects_invalid_actual_output_values(
    tmp_path, index, value, diagnostic
):
    values = {
        7: np.zeros((1, 521), dtype=np.float32),
        8: np.zeros((1, 1024), dtype=np.float32),
    }
    values[index] = value
    model = make_model(tmp_path, FakeInterpreter(values=values))

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        model.infer(np.zeros(WINDOW_SAMPLES, dtype=np.float32))

    message = str(exc_info.value)
    assert diagnostic in message
    assert "actual" in message
    assert "scores" in message or "embeddings" in message


@pytest.mark.parametrize("rows", [520, 522])
def test_class_map_requires_exactly_521_rows(tmp_path, rows):
    fake = FakeInterpreter()

    with pytest.raises(
            pipeline.ModelContractError,
            match=rf"class map.*521.*{rows}"):
        make_model(tmp_path, fake, rows=rows)


def test_non_float32_model_input_reports_tensor_details(tmp_path):
    fake = FakeInterpreter(input_detail=tensor_detail(
        3, [WINDOW_SAMPLES], dtype=np.int8, name="quantized_waveform"))

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        make_model(tmp_path, fake)

    message = str(exc_info.value)
    assert "float32" in message
    assert "quantized_waveform" in message
    assert f"[{WINDOW_SAMPLES}]" in message
    assert "int8" in message


@pytest.mark.parametrize(
    ("shape", "shape_signature"),
    [([1], [-1]), ([1, WINDOW_SAMPLES], [1, WINDOW_SAMPLES])],
)
def test_dynamic_or_incompatible_input_is_resized_before_allocation(
        tmp_path, shape, shape_signature):
    fake = FakeInterpreter(input_detail=tensor_detail(
        3,
        shape,
        name="resizable_waveform",
        shape_signature=shape_signature,
    ))

    model = make_model(tmp_path, fake)
    model.infer(np.zeros(WINDOW_SAMPLES, dtype=np.float32))

    assert fake.resize_calls == [(3, [WINDOW_SAMPLES])]
    assert fake.events[:2] == ["resize", "allocate"]
    assert fake.get_input_details_calls >= 2


def test_input_that_remains_incompatible_after_resize_is_rejected(tmp_path):
    fake = FakeInterpreter(
        input_detail=tensor_detail(3, [80], name="fixed_chunks"),
        resize_applies=False,
    )

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        make_model(tmp_path, fake)

    message = str(exc_info.value)
    assert "fixed_chunks" in message
    assert "[80]" in message
    assert "float32" in message


def test_unresizable_input_reports_tensor_details(tmp_path):
    fake = FakeInterpreter(
        input_detail=tensor_detail(3, [80], name="fixed_chunks"),
        resize_error=ValueError("cannot resize a fixed tensor"),
    )

    with pytest.raises(pipeline.ModelContractError) as exc_info:
        make_model(tmp_path, fake)

    message = str(exc_info.value)
    assert "resize" in message
    assert "fixed_chunks" in message
    assert "[80]" in message
    assert "float32" in message


@pytest.mark.parametrize("waveform", [np.zeros((1, WINDOW_SAMPLES)), 0.0])
def test_infer_rejects_non_one_dimensional_waveforms(tmp_path, waveform):
    model = make_model(tmp_path, FakeInterpreter())

    with pytest.raises(ValueError, match="one-dimensional"):
        model.infer(waveform)


@pytest.mark.parametrize("length", [WINDOW_SAMPLES - 1, WINDOW_SAMPLES + 1])
def test_infer_rejects_wrong_waveform_length(tmp_path, length):
    model = make_model(tmp_path, FakeInterpreter())

    with pytest.raises(ValueError, match=rf"exactly {WINDOW_SAMPLES} samples"):
        model.infer(np.zeros(length))


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_infer_rejects_non_finite_waveform_values(tmp_path, invalid):
    model = make_model(tmp_path, FakeInterpreter())
    waveform = np.zeros(WINDOW_SAMPLES)
    waveform[123] = invalid

    with pytest.raises(ValueError, match="finite"):
        model.infer(waveform)


def test_infer_rejects_values_that_are_not_float_convertible(tmp_path):
    model = make_model(tmp_path, FakeInterpreter())

    with pytest.raises(ValueError, match="float-convertible"):
        model.infer(["not audio"] * WINDOW_SAMPLES)


def test_missing_interpreter_backends_report_supported_packages_and_cause(
    monkeypatch,
):
    real_import = builtins.__import__
    attempted = []
    backend_modules = {
        "tflite_runtime.interpreter",
        "ai_edge_litert.interpreter",
        "tensorflow.lite",
    }

    def missing_backends(name, *args, **kwargs):
        if name in backend_modules:
            attempted.append(name)
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_backends)

    with pytest.raises(pipeline.InterpreterBackendError) as exc_info:
        pipeline._load_interpreter(Path("unused.tflite"))

    message = str(exc_info.value)
    assert "tflite-runtime" in message
    assert "ai-edge-litert" in message
    assert "tensorflow" in message
    assert "platform" in message
    assert attempted == [
        "tflite_runtime.interpreter",
        "ai_edge_litert.interpreter",
        "tensorflow.lite",
    ]
    assert isinstance(exc_info.value.__cause__, ModuleNotFoundError)


@pytest.mark.parametrize("missing", ["model", "class_map"])
def test_missing_artifact_points_to_download_before_backend_loading(
    tmp_path, monkeypatch, missing
):
    model_path = tmp_path / "yamnet.tflite"
    class_map_path = tmp_path / "yamnet_class_map.csv"
    if missing != "model":
        model_path.write_bytes(b"placeholder")
    if missing != "class_map":
        write_class_map(tmp_path)
        class_map_path = tmp_path / "class_map.csv"

    monkeypatch.setattr(
        pipeline,
        "_load_interpreter",
        lambda path: pytest.fail("backend loaded before artifact preflight"),
    )

    with pytest.raises(FileNotFoundError, match="earshot download"):
        pipeline.YamNet(str(model_path), str(class_map_path))


def test_string_artifact_paths_are_normalized_to_paths(tmp_path, monkeypatch):
    model_path = tmp_path / "yamnet.tflite"
    model_path.write_bytes(b"placeholder")
    class_map_path = write_class_map(tmp_path)
    fake = FakeInterpreter()
    loaded = []

    def fake_loader(path):
        loaded.append(path)
        return fake

    monkeypatch.setattr(pipeline, "_load_interpreter", fake_loader)

    model = pipeline.YamNet(str(model_path), str(class_map_path))

    assert model.interpreter is fake
    assert loaded == [model_path]
    assert isinstance(loaded[0], Path)


def test_without_injection_uses_production_interpreter_loader(tmp_path,
                                                              monkeypatch):
    model_path = tmp_path / "model.tflite"
    model_path.write_bytes(b"placeholder")
    class_map = write_class_map(tmp_path)
    fake = FakeInterpreter()
    loaded = []

    def fake_loader(path):
        loaded.append(path)
        return fake

    monkeypatch.setattr(pipeline, "_load_interpreter", fake_loader)

    model = pipeline.YamNet(model_path, class_map)

    assert model.interpreter is fake
    assert loaded == [model_path]


def test_top_preserves_descending_class_ranking(tmp_path):
    model = make_model(tmp_path, FakeInterpreter())
    scores = np.zeros(521, dtype=np.float32)
    scores[[4, 10, 2]] = [0.7, 0.9, 0.8]

    assert model.top(scores, k=3) == [
        ("class 10", pytest.approx(0.9)),
        ("class 2", pytest.approx(0.8)),
        ("class 4", pytest.approx(0.7)),
    ]


def test_mic_stream_falls_back_to_native_rate_and_resamples(monkeypatch):
    """A laptop mic that rejects 16 kHz is captured at its native rate and
    resampled, so laptop-only (no-Pi) demos survive picky audio hosts."""
    received = {}

    class FakeInputStream:
        def __init__(self, **kwargs):
            received.update(kwargs)
            self.callback = kwargs["callback"]

        def __enter__(self):
            block = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(-1, 1)
            self.callback(block, 12, None, None)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    def check_input_settings(**kwargs):
        raise RuntimeError("16 kHz not supported by this host")

    def query_devices(device, kind):
        assert kind == "input"
        return {"default_samplerate": 48000.0}

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(
            InputStream=FakeInputStream,
            check_input_settings=check_input_settings,
            query_devices=query_devices,
        ),
    )
    windows = pipeline.MicStream(samplerate=16000, window=4, hop=2).windows()

    result = next(windows)
    windows.close()

    assert received["samplerate"] == 48000       # captured natively
    assert received["blocksize"] == 6            # hop scaled 2 -> 6
    assert len(result) == 4                      # resampled back to 16 kHz


def test_mic_stream_keeps_direct_path_when_16k_is_supported(monkeypatch):
    received = {}

    class FakeInputStream:
        def __init__(self, **kwargs):
            received.update(kwargs)
            self.callback = kwargs["callback"]

        def __enter__(self):
            self.callback(
                np.zeros((4, 1), dtype=np.float32), 4, None, None)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(
            InputStream=FakeInputStream,
            check_input_settings=lambda **kwargs: None,
            query_devices=lambda device, kind: {"default_samplerate": 48000.0},
        ),
    )
    windows = pipeline.MicStream(samplerate=16000, window=4, hop=2).windows()

    result = next(windows)
    windows.close()

    assert received["samplerate"] == 16000       # direct path kept
    assert received["blocksize"] == 2
    assert len(result) == 4
