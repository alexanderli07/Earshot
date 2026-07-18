"""Backend logic: events + recent buffer | rules | dispatch fan-out.

Pure-ish and testable — the dispatcher takes its sinks as callables, so the
fan-out can be unit-tested with fakes (no FastAPI, GPIO, or network).
"""

import asyncio
import json
import os
import tempfile
import time
from collections import deque
from itertools import count

from . import config

# ======================================================================
# Events — normalize whatever comes in (ML, debug, replay) to one shape
# ======================================================================

_ids = count(1)
_VALID_URGENCY = set(config.ALERT_PROFILES)


def normalize_event(raw, source_default="pretrained"):
    """Coerce a raw event dict into the canonical broadcast shape.

    Adds a server id + received_at; fills missing fields with sane defaults.
    """
    raw = dict(raw or {})
    urgency = raw.get("urgency", config.DEFAULT_URGENCY)
    if urgency not in _VALID_URGENCY:
        urgency = config.DEFAULT_URGENCY
    label = str(raw.get("label", "unknown"))
    try:
        confidence = round(float(raw.get("confidence", 1.0)), 3)
    except (TypeError, ValueError):
        confidence = 1.0
    try:
        timestamp = float(raw.get("timestamp", time.time()))
    except (TypeError, ValueError):
        timestamp = time.time()
    return {
        "id": f"evt_{next(_ids)}",
        "label": label,
        "urgency": urgency,
        "confidence": confidence,
        "source": str(raw.get("source", source_default)),
        "timestamp": timestamp,
        "received_at": time.time(),
    }


# ======================================================================
# Recent events — in-memory ring buffer (newest first)
# ======================================================================

class RecentEvents:
    def __init__(self, maxlen=config.RECENT_EVENTS_MAX):
        self._events = deque(maxlen=maxlen)

    def add(self, event):
        self._events.appendleft(event)

    def list(self, limit=None):
        events = list(self._events)
        return events[:limit] if limit else events


# ======================================================================
# Rules — per-sound on/off + urgency override, persisted to JSON
# ======================================================================

class Rules:
    """{ "<label>": {"enabled": bool, "urgency": "high"|... or None} }.

    apply() drops a muted sound and applies any urgency override.
    """

    def __init__(self, path=config.RULES_PATH):
        self.path = path
        self._rules = {}
        if path is not None and path.exists():
            try:
                self._rules = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._rules = {}   # a corrupt rules file must not brick startup

    def all(self):
        return dict(self._rules)

    def set(self, label, enabled=True, urgency=None):
        if urgency is not None and urgency not in _VALID_URGENCY:
            raise ValueError(f"invalid urgency {urgency!r}")
        self._rules[str(label)] = {"enabled": bool(enabled), "urgency": urgency}
        self._save()
        return self._rules[str(label)]

    def apply(self, event):
        """Return the event (possibly with overridden urgency), or None if the
        sound is muted."""
        rule = self._rules.get(event["label"])
        if rule is None:
            return event
        if not rule.get("enabled", True):
            return None
        if rule.get("urgency"):
            event = {**event, "urgency": rule["urgency"]}
        return event

    def _save(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write so a crash mid-save can't corrupt the rules file.
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self._rules, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise


# ======================================================================
# Dispatch — one event in, four alerts out (row, light, buzz, push)
# ======================================================================

class Dispatcher:
    """Ties recent + rules to the three sinks. Sinks are injected so this is
    testable with fakes.

      broadcast(event)  -> async, pushes JSON to WebSocket clients (row)
      alert(urgency)    -> sync, drives LED + motor (light + buzz)
      push(event, prof) -> async, ntfy phone push
    """

    def __init__(self, recent, rules, broadcast, alert, push):
        self.recent = recent
        self.rules = rules
        self._broadcast = broadcast
        self._alert = alert
        self._push = push

    async def dispatch(self, raw, source_default="pretrained"):
        """The switchboard. Returns the delivered event, or None if muted."""
        event = self.rules.apply(normalize_event(raw, source_default))
        if event is None:
            return None
        self.recent.add(event)
        profile = config.ALERT_PROFILES.get(
            event["urgency"], config.ALERT_PROFILES[config.DEFAULT_URGENCY])
        loop = asyncio.get_running_loop()
        # Fire all sinks concurrently for the <1 s budget; return_exceptions so
        # one dead phone / loose wire never takes down the others.
        await asyncio.gather(
            self._broadcast(event),
            loop.run_in_executor(None, self._alert, event["urgency"]),
            self._push(event, profile),
            return_exceptions=True,
        )
        return event
