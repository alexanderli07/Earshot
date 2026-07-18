"""The signal path: sound in -> 521 class scores + 1024-dim embedding out.

Regions: audio helpers | mic capture | YAMNet classifier.
sounddevice and tflite are imported lazily, so this module loads anywhere.
"""

import csv
import queue
import sys
import threading
import wave
from pathlib import Path

import numpy as np

from .config import (SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES,
                     MODEL_PATH, CLASS_MAP_PATH)

# ======================================================================
# Audio helpers — wav loading, resampling, windowing (stdlib + numpy only)
# ======================================================================

_PCM_SCALE = {1: 128.0, 2: 32768.0, 4: 2147483648.0}
_PCM_DTYPE = {1: np.uint8, 2: np.int16, 4: np.int32}


class AudioFileError(ValueError):
    """A WAV file is unreadable or incompatible with Earshot's PCM input."""


def _as_float32_audio(audio):
    try:
        result = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("audio must be float-convertible") from exc
    if result.ndim != 1:
        raise ValueError(
            "audio must be one-dimensional; "
            f"received shape {result.shape}"
        )
    return result


def _validated_sample_rate(value):
    try:
        sample_rate = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "sample rates must be positive and finite numbers"
        ) from exc
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample rates must be positive and finite numbers")
    return sample_rate


def _validated_positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def load_wav_16k_mono(path):
    """Load a PCM wav as float32 mono in [-1, 1] at 16 kHz."""
    try:
        with wave.open(str(path), "rb") as w:
            width = w.getsampwidth()
            if width not in _PCM_DTYPE:
                cause = ValueError(f"unsupported sample width {width}")
                raise AudioFileError(f"{path}: {cause}") from cause
            frames = w.readframes(w.getnframes())
            x = np.frombuffer(frames, dtype=_PCM_DTYPE[width]).astype(np.float32)
            if width == 1:  # 8-bit wav is unsigned
                x -= 128.0
            x /= _PCM_SCALE[width]
            if w.getnchannels() > 1:
                x = x.reshape(-1, w.getnchannels()).mean(axis=1)
            return resample_linear(x, w.getframerate(), SAMPLE_RATE)
    except AudioFileError:
        raise
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        raise AudioFileError(f"{path}: could not read PCM WAV: {exc}") from exc


def resample_linear(x, sr_from, sr_to):
    """Linear-interpolation resample. Fine for an MVP; not audiophile grade."""
    x = _as_float32_audio(x)
    sr_from = _validated_sample_rate(sr_from)
    sr_to = _validated_sample_rate(sr_to)
    if sr_from == sr_to:
        return x.astype(np.float32, copy=False)
    n_out = int(round(len(x) * sr_to / sr_from))
    if not len(x) or not n_out:
        return np.zeros(n_out, dtype=np.float32)
    src_t = np.arange(len(x), dtype=np.float64)
    dst_t = np.linspace(0, len(x) - 1, n_out)
    return np.interp(dst_t, src_t, x).astype(np.float32)


def clip_windows(audio, window=WINDOW_SAMPLES, hop=HOP_SAMPLES):
    """Split a clip into model-sized windows, zero-padding if too short."""
    audio = _as_float32_audio(audio)
    window = _validated_positive_integer(window, "window")
    hop = _validated_positive_integer(hop, "hop")
    if len(audio) < window:
        audio = np.pad(audio, (0, window - len(audio)))
    return [audio[i:i + window] for i in range(0, len(audio) - window + 1, hop)]


# ======================================================================
# Mic capture — USB mic -> overlapping 0.975 s float32 windows at 16 kHz
# ======================================================================

class AudioDeviceError(RuntimeError):
    """sounddevice could not initialize or operate the selected input."""


def _audio_device_error(action, exc):
    return AudioDeviceError(
        f"{action}: {exc}. Run `python -m sounddevice` to list and validate "
        "available audio devices"
    )


def _load_sounddevice():
    try:
        import sounddevice as sd
    except Exception as exc:
        raise _audio_device_error("could not import sounddevice", exc) from exc
    return sd


class LatestBlockQueue(queue.Queue):
    """Bounded queue that replaces its oldest block when full."""

    def __init__(self, maxsize=2):
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        super().__init__(maxsize=maxsize)
        self._next_sequence = 0

    def put_latest(self, block):
        """Enqueue block, atomically discarding one oldest block if full."""
        with self.not_full:
            item = (self._next_sequence, block)
            self._next_sequence += 1
            if self._qsize() >= self.maxsize:
                self._get()
                self.unfinished_tasks -= 1
            self._put(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def _get_sequenced(self, block=True, timeout=None):
        return super().get(block=block, timeout=timeout)

    def get(self, block=True, timeout=None):
        """Return only the queued block, preserving the public Queue API."""
        _, value = self._get_sequenced(block=block, timeout=timeout)
        return value


class MicStream:
    def __init__(self, samplerate=SAMPLE_RATE, window=WINDOW_SAMPLES,
                 hop=HOP_SAMPLES, device=None, queue_size=2):
        self.samplerate = samplerate
        self.window = window
        self.hop = hop
        self.device = device
        self.queue_size = queue_size

    def windows(self, stop_event=None, on_gap=None):
        """Blocking generator of float32 arrays, `window` samples each.

        on_gap: optional callable invoked when a capture discontinuity
        (dropped blocks) clears the buffer, so downstream state that spans
        windows — e.g. detector streaks — can reset too.
        """
        sd = _load_sounddevice()
        blocks = LatestBlockQueue(maxsize=self.queue_size)

        def callback(indata, frames, time_info, status):
            if status:
                print(f"[mic] {status}", file=sys.stderr)
            blocks.put_latest(indata[:, 0].copy())

        buf = np.zeros(0, dtype=np.float32)
        previous_sequence = None
        try:
            stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype="float32",
                blocksize=self.hop,
                device=self.device,
                callback=callback,
            )
        except Exception as exc:
            raise _audio_device_error(
                "could not initialize microphone input stream", exc
            ) from exc
        try:
            stream.__enter__()
        except Exception as exc:
            raise _audio_device_error(
                "could not start microphone input stream", exc
            ) from exc
        try:
            while stop_event is None or not stop_event.is_set():
                try:
                    sequence, block = blocks._get_sequenced(timeout=0.1)
                except queue.Empty:
                    continue
                if stop_event is not None and stop_event.is_set():
                    break
                if (previous_sequence is not None
                        and sequence != previous_sequence + 1):
                    buf = np.zeros(0, dtype=np.float32)
                    if on_gap is not None:
                        on_gap()
                previous_sequence = sequence
                buf = np.concatenate([buf, block])
                while (len(buf) >= self.window
                       and (stop_event is None or not stop_event.is_set())):
                    yield buf[:self.window].copy()
                    buf = buf[self.hop:]
        finally:
            active_exception = sys.exc_info()
            try:
                stream.__exit__(*active_exception)
            except Exception as exc:
                if active_exception[0] is None:
                    raise _audio_device_error(
                        "could not close microphone input stream", exc
                    ) from exc


def record(seconds, samplerate=SAMPLE_RATE, device=None):
    """Blocking one-shot recording, returns float32 mono."""
    sd = _load_sounddevice()
    try:
        audio = sd.rec(int(seconds * samplerate), samplerate=samplerate,
                       channels=1, dtype="float32", device=device)
    except Exception as exc:
        raise _audio_device_error(
            "could not start microphone recording", exc
        ) from exc
    try:
        sd.wait()
    except Exception as exc:
        raise _audio_device_error(
            "microphone recording failed", exc
        ) from exc
    return audio[:, 0].copy()


# ======================================================================
# YAMNet classifier — tflite-runtime, one 0.975 s window per inference
# ======================================================================

class InterpreterBackendError(RuntimeError):
    """No supported lazy-loaded TFLite interpreter is available."""


def _load_interpreter(model_path):
    try:
        from tflite_runtime.interpreter import Interpreter  # the Pi
    except ImportError:
        try:
            from ai_edge_litert.interpreter import Interpreter  # dev laptop
        except ImportError:
            try:
                from tensorflow.lite import Interpreter  # last resort
            except ImportError as exc:
                raise InterpreterBackendError(
                    "no supported TFLite interpreter backend is installed; "
                    "install a package available for this platform: "
                    "`tflite-runtime`, `ai-edge-litert`, or `tensorflow`"
                ) from exc
    return Interpreter(model_path=str(model_path), num_threads=1)


class ModelContractError(RuntimeError):
    """The class map or interpreter tensors are incompatible with YAMNet."""


def load_class_names(path=CLASS_MAP_PATH):
    """Load and validate the indexed display names in a YAMNet class map."""
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, strict=True)
            required = {"index", "mid", "display_name"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "missing required column(s): " + ", ".join(sorted(missing))
                )

            names = []
            for line_number, row in enumerate(reader, start=2):
                try:
                    index = int(row["index"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"row {line_number} index must be an integer"
                    ) from exc
                expected_index = len(names)
                if index != expected_index:
                    raise ValueError(
                        f"row {line_number} index must be {expected_index}; "
                        f"found {index}"
                    )
                display_name = row["display_name"]
                if display_name is None or not display_name.strip():
                    raise ValueError(
                        f"row {line_number} display_name must not be empty"
                    )
                names.append(display_name.strip())
        return names
    except ModelContractError:
        raise
    except (
        OSError,
        UnicodeError,
        csv.Error,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ModelContractError(
            f"could not read class map {path}: {exc}"
        ) from exc


def _tensor_shape(detail):
    """Return a tensor detail shape as a regular tuple of integers."""
    shape = detail.get("shape", ())
    return tuple(int(value) for value in np.asarray(shape).reshape(-1))


def _tensor_dtype_name(detail):
    dtype = detail.get("dtype")
    if dtype is None:
        return "unknown"
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype)


def _describe_tensors(details):
    if not details:
        return "none"
    descriptions = []
    for detail in details:
        name = detail.get("name", "<unnamed>")
        index = detail.get("index", "?")
        shape = list(_tensor_shape(detail))
        dtype = _tensor_dtype_name(detail)
        descriptions.append(
            f"{name!r} (index={index}, shape={shape}, dtype={dtype})")
    return "; ".join(descriptions)


def _validated_output_value(value, *, width, label, detail):
    output = np.asarray(value)
    actual_shape = list(output.shape)
    actual_dtype = output.dtype.name
    diagnostic = (
        f"observed tensor metadata: {_describe_tensors([detail])}; "
        f"actual shape={actual_shape}, dtype={actual_dtype}"
    )
    if output.dtype != np.dtype(np.float32):
        raise ModelContractError(
            f"{label} output must be float32; {diagnostic}"
        )
    if output.ndim < 1 or output.size == 0:
        raise ModelContractError(
            f"{label} output must be nonempty with final dimension {width}; "
            f"{diagnostic}"
        )
    if output.shape[-1] != width:
        raise ModelContractError(
            f"{label} output must have final dimension {width}; {diagnostic}"
        )
    if not np.isfinite(output).all():
        raise ModelContractError(
            f"{label} output values must all be finite; {diagnostic}"
        )
    return output.reshape(-1, width).mean(axis=0)


class YamNet:
    def __init__(self, model_path=MODEL_PATH, class_map_path=CLASS_MAP_PATH,
                 interpreter=None):
        model_path = Path(model_path)
        class_map_path = Path(class_map_path)
        missing = []
        if interpreter is None and not model_path.is_file():
            missing.append(str(model_path))
        if not class_map_path.is_file():
            missing.append(str(class_map_path))
        if missing:
            raise FileNotFoundError(
                "required Earshot artifact(s) not found: "
                f"{', '.join(missing)}; run `earshot download` first"
            )
        self.class_names = load_class_names(class_map_path)
        if len(self.class_names) != 521:
            raise ModelContractError(
                "class map must contain exactly 521 rows; "
                f"found {len(self.class_names)} in {class_map_path}")

        self.interpreter = (interpreter if interpreter is not None
                            else _load_interpreter(model_path))
        self._infer_lock = threading.RLock()
        inputs = self.interpreter.get_input_details()
        if len(inputs) != 1:
            raise ModelContractError(
                "model must expose exactly one float32 waveform input shaped "
                f"[{WINDOW_SAMPLES}]; observed inputs: "
                f"{_describe_tensors(inputs)}")

        initial_input = inputs[0]
        if _tensor_dtype_name(initial_input) != "float32":
            raise ModelContractError(
                "model input must be float32 and shaped "
                f"[{WINDOW_SAMPLES}]; observed inputs: "
                f"{_describe_tensors(inputs)}")

        shape = _tensor_shape(initial_input)
        signature = _tensor_shape({
            "shape": initial_input.get("shape_signature", shape)
        })
        if shape != (WINDOW_SAMPLES,) or any(size < 0 for size in signature):
            try:
                self.interpreter.resize_tensor_input(
                    initial_input["index"], [WINDOW_SAMPLES])
            except Exception as exc:
                raise ModelContractError(
                    f"could not resize model input to [{WINDOW_SAMPLES}]; "
                    f"observed inputs: {_describe_tensors(inputs)}") from exc

        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        if (len(inputs) != 1
                or _tensor_dtype_name(inputs[0]) != "float32"
                or _tensor_shape(inputs[0]) != (WINDOW_SAMPLES,)):
            raise ModelContractError(
                "model input must resolve to one float32 tensor shaped "
                f"[{WINDOW_SAMPLES}] after allocation; observed inputs: "
                f"{_describe_tensors(inputs)}")
        self._input = inputs[0]

        # Outputs are identified by shape: 521 = class scores, 1024 = embedding.
        outputs = self.interpreter.get_output_details()
        score_outputs = [
            detail for detail in outputs
            if _tensor_shape(detail) and _tensor_shape(detail)[-1] == 521
        ]
        embedding_outputs = [
            detail for detail in outputs
            if _tensor_shape(detail) and _tensor_shape(detail)[-1] == 1024
        ]
        if (len(score_outputs) != 1
                or _tensor_dtype_name(score_outputs[0]) != "float32"
                or len(embedding_outputs) != 1
                or _tensor_dtype_name(embedding_outputs[0]) != "float32"):
            raise ModelContractError(
                "model must expose exactly one float32 output with final "
                "dimension 521 (scores) and exactly one float32 output with "
                "final dimension 1024 (embedding); observed outputs: "
                f"{_describe_tensors(outputs)}")
        self._scores_output = score_outputs[0]
        self._embedding_output = embedding_outputs[0]
        self._scores_idx = self._scores_output["index"]
        self._embed_idx = self._embedding_output["index"]

    def infer(self, waveform):
        """waveform: float32 [-1, 1], exactly WINDOW_SAMPLES long.

        Returns (scores[521], embedding[1024]).
        """
        try:
            waveform = np.asarray(waveform, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "waveform must be float-convertible to float32") from exc
        if waveform.ndim != 1:
            raise ValueError(
                "waveform must be one-dimensional; "
                f"received shape {waveform.shape}")
        if waveform.shape[0] != WINDOW_SAMPLES:
            raise ValueError(
                f"waveform must contain exactly {WINDOW_SAMPLES} samples; "
                f"received {waveform.shape[0]}")
        if not np.isfinite(waveform).all():
            raise ValueError("waveform samples must all be finite")
        with self._infer_lock:
            self.interpreter.set_tensor(self._input["index"], waveform)
            self.interpreter.invoke()
            scores = self.interpreter.get_tensor(self._scores_idx)
            embedding = self.interpreter.get_tensor(self._embed_idx)
        return (
            _validated_output_value(
                scores,
                width=521,
                label="scores",
                detail=self._scores_output,
            ),
            _validated_output_value(
                embedding,
                width=1024,
                label="embeddings",
                detail=self._embedding_output,
            ),
        )

    def top(self, scores, k=5):
        idx = np.argsort(scores)[::-1][:k]
        return [(self.class_names[i], float(scores[i])) for i in idx]
