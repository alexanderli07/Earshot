"""Decisions + public API: scores/embeddings in -> debounced events out.

Regions: event objects | streak + debounce detector | teach store | engine.
Everything above the engine is pure logic — unit-testable without a mic
or the model.
"""

import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from math import inf
from pathlib import Path

import numpy as np

from . import config
from .pipeline import MicStream, YamNet, clip_windows, load_wav_16k_mono

# ======================================================================
# Event objects — what gets emitted to the backend
# ======================================================================


@dataclass(frozen=True)
class Event:
    label: str
    urgency: str          # "high" | "medium" | "low"
    confidence: float
    source: str           # "pretrained" | "taught"
    timestamp: float      # unix seconds

    def to_dict(self):
        return {
            "label": self.label,
            "urgency": self.urgency,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class Observation:
    """One candidate event's status for a single audio window."""
    label: str
    confidence: float
    above: bool           # confidence cleared its threshold this window
    urgency: str
    source: str
    consecutive: int      # windows above threshold required to fire


# ======================================================================
# Streak + debounce detector — fire after N consecutive windows, then
# suppress repeats of the same label for debounce_s seconds
# ======================================================================

class EventDetector:
    def __init__(self, debounce_s=10.0):
        self.debounce_s = debounce_s
        self._streaks = {}
        self._last_fired = {}

    def update(self, observations, now=None):
        """Feed one window's observations; returns the list of fired Events.

        Any label tracked previously but absent from `observations` has its
        streak reset — a taught sound that stops matching starts over.
        """
        if now is None:
            now = time.time()
        fired = []
        seen = set()
        for obs in observations:
            key = (obs.source, obs.label)
            seen.add(key)
            if not obs.above:
                self._streaks[key] = 0
                continue
            self._streaks[key] = self._streaks.get(key, 0) + 1
            if self._streaks[key] < obs.consecutive:
                continue
            if now - self._last_fired.get(key, -inf) < self.debounce_s:
                continue
            self._last_fired[key] = now
            fired.append(Event(
                label=obs.label,
                urgency=obs.urgency,
                confidence=obs.confidence,
                source=obs.source,
                timestamp=now,
            ))
        for key in self._streaks:
            if key not in seen:
                self._streaks[key] = 0
        return fired


# ======================================================================
# Teach store — one L2-normalized embedding per taught clip; live windows
# matched by cosine nearest neighbor (dot product after normalization)
# ======================================================================

class TeachStoreError(RuntimeError):
    """A taught-sound store could not be loaded, validated, or persisted."""


def _normalize(v):
    """Validate and L2-normalize one 1,024-value embedding."""
    try:
        vector = np.asarray(v, dtype=np.float32).ravel()
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("embedding must be float-convertible") from exc
    if vector.size != 1024:
        raise ValueError(
            f"embedding must contain exactly 1024 values; found {vector.size}")
    if not np.isfinite(vector).all():
        raise ValueError("embedding values must all be finite")
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class TeachStore:
    def __init__(self, path=None, cutoff=0.80):
        self.path = Path(path) if path is not None else None
        self.cutoff = cutoff
        self._lock = threading.RLock()
        self._names = []                              # one entry per vector
        self._vectors = np.zeros((0, 1024), np.float32)
        if self.path is not None and self.path.exists():
            self._load()

    def _load(self):
        """Load and validate a complete store before changing live state."""
        try:
            with np.load(self.path, allow_pickle=False) as data:
                missing = {"names", "vectors"} - set(data.files)
                if missing:
                    missing_text = ", ".join(sorted(missing))
                    raise TeachStoreError(
                        f"invalid taught store {self.path}: missing {missing_text}")
                try:
                    names = np.array(data["names"], copy=True)
                except Exception as exc:
                    raise TeachStoreError(
                        f"invalid taught store {self.path}: names array "
                        "could not be loaded") from exc
                try:
                    vectors = np.array(data["vectors"], copy=True)
                except Exception as exc:
                    raise TeachStoreError(
                        f"invalid taught store {self.path}: vectors array "
                        "could not be loaded") from exc

            if names.ndim != 1:
                raise TeachStoreError(
                    f"invalid taught store {self.path}: names must be "
                    "one-dimensional")
            if names.size and names.dtype.kind not in {"U", "S"}:
                raise TeachStoreError(
                    f"invalid taught store {self.path}: names must use a "
                    "string dtype")
            if vectors.ndim != 2:
                raise TeachStoreError(
                    f"invalid taught store {self.path}: vectors must be "
                    "two-dimensional")
            if vectors.shape[1] != 1024:
                raise TeachStoreError(
                    f"invalid taught store {self.path}: vectors must have "
                    f"width 1024; found {vectors.shape[1]}")
            if len(names) != len(vectors):
                raise TeachStoreError(
                    f"invalid taught store {self.path}: name/vector count "
                    f"mismatch ({len(names)} names, {len(vectors)} vectors)")
            if not (np.issubdtype(vectors.dtype, np.floating)
                    or np.issubdtype(vectors.dtype, np.integer)):
                raise TeachStoreError(
                    f"invalid taught store {self.path}: vectors must use a "
                    "real numeric dtype")
            vectors = vectors.astype(np.float32)
            if not np.isfinite(vectors).all():
                raise TeachStoreError(
                    f"invalid taught store {self.path}: vectors must all be finite")

            loaded_names = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in names
            ]
        except TeachStoreError:
            raise
        except Exception as exc:
            raise TeachStoreError(
                f"could not load taught store {self.path}: {exc}") from exc

        with self._lock:
            self._names = loaded_names
            self._vectors = vectors

    def add(self, name, embedding):
        vector = _normalize(embedding)
        with self._lock:
            vectors = np.vstack([self._vectors, vector])
            self._names.append(name)
            self._vectors = vectors

    @contextmanager
    def transaction(self):
        """Rollback an in-memory update if any operation in it fails."""
        with self._lock:
            names = list(self._names)
            vectors = self._vectors.copy()
            try:
                yield self
            except BaseException:
                self._names = names
                self._vectors = vectors
                raise

    def match(self, embedding):
        """Nearest stored clip by cosine similarity, or None below cutoff.

        Returns (name, similarity).
        """
        vector = _normalize(embedding)
        with self._lock:
            if not self._names:
                return None
            names = list(self._names)
            vectors = self._vectors.copy()
        sims = vectors @ vector
        best = int(np.argmax(sims))
        if sims[best] < self.cutoff:
            return None
        return names[best], float(sims[best])

    def learned(self):
        """[{"name": ..., "clips": ...}], insertion order preserved."""
        with self._lock:
            names = list(self._names)
        counts = {}
        for n in names:
            counts[n] = counts.get(n, 0) + 1
        return [{"name": n, "clips": c} for n, c in counts.items()]

    def forget(self, name):
        with self._lock:
            keep = [i for i, n in enumerate(self._names) if n != name]
            removed = len(self._names) - len(keep)
            self._names = [self._names[i] for i in keep]
            self._vectors = (self._vectors[keep] if keep else
                             np.zeros((0, 1024), np.float32))
            return removed

    def save(self):
        if self.path is None:
            return
        part = self.path.with_name(self.path.name + ".part")
        try:
            with self._lock:
                names = np.asarray(self._names, dtype=np.str_)
                vectors = self._vectors.copy()
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with part.open("wb") as output:
                    np.savez(output, names=names, vectors=vectors)
                    output.flush()
                os.replace(part, self.path)
        except OSError as exc:
            raise TeachStoreError(
                f"could not save taught store {self.path}: {exc}"
            ) from exc
        finally:
            try:
                part.unlink()
            except OSError:
                # Best effort only: never hide the write/replace failure.
                pass


# ======================================================================
# Engine — the whole pipeline behind two calls:
#     engine = EarshotML(on_event=backend_callback)
#     engine.run()                      # blocks: mic -> YAMNet -> events
#     engine.teach("kettle", clips)     # from the backend teach endpoint
# ======================================================================

class EarshotML:
    def __init__(self, on_event=None, event_queue=None, device=None,
                 model_path=config.MODEL_PATH,
                 class_map_path=config.CLASS_MAP_PATH,
                 taught_store_path=config.TAUGHT_STORE_PATH,
                 yamnet=None):
        """on_event: callable(dict) invoked per fired event.
        event_queue: queue.Queue that fired event dicts are put on.
        Either or both; no network involved."""
        self.on_event = on_event
        self.event_queue = event_queue
        self.device = device
        self.yamnet = (yamnet if yamnet is not None else
                       YamNet(model_path, class_map_path))
        self.store = TeachStore(path=taught_store_path,
                                cutoff=config.TAUGHT_SIMILARITY_CUTOFF)
        self.detector = EventDetector(debounce_s=config.DEBOUNCE_SECONDS)
        self._specs = self._resolve_event_map()

    def _resolve_event_map(self):
        """Turn config.EVENT_MAP class names into score indices once."""
        name_to_idx = {n: i for i, n in enumerate(self.yamnet.class_names)}
        specs = []
        for entry in config.EVENT_MAP:
            indices = []
            for cls in entry["classes"]:
                if cls in name_to_idx:
                    indices.append(name_to_idx[cls])
                else:
                    print(f"[earshot] class {cls!r} not in YAMNet class map, "
                          f"skipping", file=sys.stderr)
            if indices:
                specs.append({**entry, "indices": indices})
        return specs

    # ---- live pipeline ----

    def process_window(self, waveform, now=None):
        """Run one 0.975 s window through the pipeline; returns fired Events.

        Split out from run() so it's testable and reusable on file input.
        """
        scores, embedding = self.yamnet.infer(waveform)
        observations = []
        for spec in self._specs:
            confidence = float(scores[spec["indices"]].max())
            observations.append(Observation(
                label=spec["label"],
                confidence=confidence,
                above=bool(confidence >= spec["threshold"]),
                urgency=spec["urgency"],
                source="pretrained",
                consecutive=config.CONSECUTIVE_WINDOWS,
            ))
        match = self.store.match(embedding)
        if match is not None:
            name, similarity = match
            observations.append(Observation(
                label=name,
                confidence=similarity,
                above=True,
                urgency=config.TAUGHT_URGENCY,
                source="taught",
                consecutive=config.TAUGHT_CONSECUTIVE_WINDOWS,
            ))
        events = self.detector.update(observations, now=now)
        for event in events:
            self._emit(event)
        return events

    def run(self, stop_event=None):
        """Block on mic windows until the stream ends or is stopped."""
        for waveform in MicStream(device=self.device).windows(
                stop_event=stop_event):
            self.process_window(waveform)

    def _emit(self, event):
        payload = event.to_dict()
        if self.on_event is not None:
            self.on_event(payload)
        if self.event_queue is not None:
            self.event_queue.put(payload)

    # ---- teach mode (called by the backend teach endpoint) ----

    def teach(self, name, clips):
        """Store one embedding per clip. clips: wav paths and/or float32
        arrays (16 kHz mono). Returns the number of clips stored."""
        if not isinstance(name, str):
            raise ValueError("teach name must be a non-empty string")
        name = name.strip()
        if not name:
            raise ValueError("teach name must be a non-empty string")
        reserved_names = {
            entry["label"].strip().casefold() for entry in config.EVENT_MAP
        }
        if name.casefold() in reserved_names:
            raise ValueError(
                f"teach name {name!r} conflicts with a pretrained label")

        try:
            clip_list = list(clips)
        except TypeError as exc:
            raise ValueError("clips must be a non-empty iterable") from exc
        if not clip_list:
            raise ValueError("clips must be a non-empty iterable")

        audio_clips = []
        for clip in clip_list:
            loaded = (load_wav_16k_mono(clip)
                      if isinstance(clip, (str, bytes))
                      or hasattr(clip, "__fspath__") else clip)
            try:
                audio = np.asarray(loaded, dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("clip audio must be float32-compatible") from exc
            if audio.ndim != 1:
                raise ValueError("clip audio must be one-dimensional")
            if audio.size == 0:
                raise ValueError("clip audio must not be empty")
            if not np.isfinite(audio).all():
                raise ValueError("clip audio values must all be finite")
            audio_clips.append(audio)

        clip_embeddings = []
        for audio in audio_clips:
            embeddings = [self.yamnet.infer(w)[1] for w in clip_windows(audio)]
            clip_embeddings.append(_normalize(np.mean(embeddings, axis=0)))

        with self.store.transaction():
            for clip_embedding in clip_embeddings:
                self.store.add(name, clip_embedding)
            self.store.save()
        return len(audio_clips)

    def learned_sounds(self):
        return self.store.learned()

    def forget(self, name):
        with self.store.transaction():
            removed = self.store.forget(name)
            self.store.save()
        return removed
