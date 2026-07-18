"""Optional bridge to the ML component in the sibling ml/ folder.

The backend runs fine without ML (the debug endpoint fires fake events so the
frontend and hardware can build all day). When earshot_ml is importable, real
mic events flow in and the teach endpoint works.
"""

import asyncio
import sys
import tempfile
import threading
from pathlib import Path

# Make the sibling ml/ package importable: <repo>/ml alongside <repo>/backend.
_ML_DIR = Path(__file__).resolve().parent.parent.parent / "ml"
if _ML_DIR.exists() and str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

REQUIRED_CLIPS = 3


class MLBridge:
    def __init__(self):
        self.engine = None
        self.available = False
        self.last_error = None
        self._thread = None
        self._stop_event = threading.Event()
        try:
            from earshot_ml import EarshotML   # noqa: F401
            self._EarshotML = EarshotML
            self.available = True
        except Exception as exc:
            self.last_error = f"import failed: {exc}"
            print(f"[ml] earshot_ml not available ({exc}); running in "
                  f"debug-only mode", file=sys.stderr)

    @property
    def alive(self):
        """True only while the listener thread is actually running —
        /healthz reports this, not just whether the import worked."""
        return self._thread is not None and self._thread.is_alive()

    def start(self, loop, dispatch):
        """Start live detection in a supervised daemon thread.

        The ML on_event callback is synchronous and fires from the audio
        thread; hand each event to the asyncio loop safely.
        """
        if not self.available or self.engine is not None:
            return

        def on_event(event):
            asyncio.run_coroutine_threadsafe(
                dispatch(event, source_default=event.get("source", "pretrained")),
                loop)

        try:
            self.engine = self._EarshotML(on_event=on_event)
        except Exception as exc:
            # Any construction failure (missing/corrupt model, class map,
            # store, interpreter) degrades to debug-only instead of taking
            # the optional backend down with it.
            self.last_error = f"engine init failed: {exc}"
            self.available = False
            self.engine = None
            print(f"[ml] {exc}; running in debug-only mode", file=sys.stderr)
            return

        def supervised_run():
            try:
                self.engine.run(stop_event=self._stop_event)
            except Exception as exc:
                self.last_error = f"listener died: {exc}"
                print(f"[ml] listener died: {exc}", file=sys.stderr)

        self._thread = threading.Thread(target=supervised_run, daemon=True)
        self._thread.start()
        print("[ml] live detection started", file=sys.stderr)

    def stop(self, timeout=3.0):
        """Signal the listener to stop and join it with a deadline."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                print("[ml] listener did not stop within deadline",
                      file=sys.stderr)

    def teach(self, name, blobs):
        """blobs: list of (filename, bytes). Writes temp wavs, calls ML teach.

        Temp audio is private voice data: it lives in a TemporaryDirectory
        and is deleted in finally, success or not.
        """
        if not self.available or self.engine is None:
            raise RuntimeError("ML not available")
        if len(blobs) != REQUIRED_CLIPS:
            raise ValueError(f"teach requires exactly {REQUIRED_CLIPS} clips")
        with tempfile.TemporaryDirectory(prefix="earshot_teach_") as tmpdir:
            paths = []
            for i, (fname, data) in enumerate(blobs):
                suffix = Path(fname or f"clip{i}.wav").suffix or ".wav"
                p = Path(tmpdir) / f"clip{i}{suffix}"
                p.write_bytes(data)
                paths.append(str(p))
            self.engine.teach(name, paths)
        return self.engine.learned_sounds()

    def learned_sounds(self):
        if not self.available or self.engine is None:
            return []
        return self.engine.learned_sounds()
