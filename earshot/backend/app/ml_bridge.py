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


class MLBridge:
    def __init__(self):
        self.engine = None
        self.available = False
        self._thread = None
        try:
            from earshot_ml import EarshotML   # noqa: F401
            self._EarshotML = EarshotML
            self.available = True
        except Exception as exc:
            self._import_error = exc
            print(f"[ml] earshot_ml not available ({exc}); running in "
                  f"debug-only mode", file=sys.stderr)

    def start(self, loop, dispatch):
        """Start live detection in a daemon thread.

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
        except FileNotFoundError as exc:
            # Model not downloaded yet — stay in debug-only mode.
            print(f"[ml] {exc}", file=sys.stderr)
            self.available = False
            return
        self._thread = threading.Thread(target=self.engine.run, daemon=True)
        self._thread.start()
        print("[ml] live detection started", file=sys.stderr)

    def teach(self, name, blobs):
        """blobs: list of (filename, bytes). Writes temp wavs, calls ML teach.

        NOTE: needs the ML mode-gate fix to run while detection is live —
        engine.teach() currently raises if called during engine.run().
        """
        if not self.available or self.engine is None:
            raise RuntimeError("ML not available")
        paths = []
        tmpdir = tempfile.mkdtemp(prefix="earshot_teach_")
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
