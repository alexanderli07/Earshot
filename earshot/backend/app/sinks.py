"""The outputs: WebSocket hub | GPIO alerts | ntfy push.

GPIO and httpx are used lazily/defensively so the whole app runs on a laptop
(GPIO falls back to a logging mock) for development without a Pi.
"""

import asyncio
import sys
import threading
import time

from . import config

# ======================================================================
# WebSocket hub — broadcast every event to all connected clients
# ======================================================================

class EventHub:
    def __init__(self):
        self._connections = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)

    async def disconnect(self, ws):
        async with self._lock:
            self._connections.discard(ws)

    @property
    def count(self):
        return len(self._connections)

    async def broadcast(self, event):
        """Send to every client; returns how many actually received it.

        Iterates a COPY: clients drop mid-send and one dead phone must not
        crash the loop (per the spec's watch-out).
        """
        delivered = 0
        for ws in list(self._connections):
            try:
                await ws.send_json(event)
                delivered += 1
            except Exception:
                await self.disconnect(ws)
        return delivered


# ======================================================================
# GPIO alerts — one serialized, priority-aware controller.
#
# A single worker thread owns the LED and motor. New alerts are admitted
# only if they rank at least as high as whatever is active, so a low-priority
# chime can never cancel a smoke-alarm strobe, and only the worker ever
# writes hardware state, so a superseded alert can't blank a newer one.
# The named LED patterns (strobe/pulse/blink) are actually played.
# ======================================================================

def _make_devices():
    """Return (RGBLED, motor) on a Pi, or (None, None) to use the mock."""
    try:
        from gpiozero import RGBLED, PWMOutputDevice
        led = RGBLED(config.PIN_R, config.PIN_G, config.PIN_B)
        motor = PWMOutputDevice(config.PIN_MOTOR)
        return led, motor
    except Exception as exc:   # not a Pi, or no pin factory
        print(f"[gpio] hardware unavailable ({exc}); using log mock",
              file=sys.stderr)
        return None, None


def build_timeline(profile):
    """Merge the motor on/off pattern and the LED cycle into one step list.

    Returns [(duration_s, led_on, motor_on), ...] covering the motor
    pattern's total duration; the LED repeats its cycle across it.
    """
    motor_pattern = config.MOTOR_PATTERNS.get(profile["motor"], [0.2])
    led_cycle = config.LED_PATTERNS.get(profile["led"],
                                        {"on": 0.3, "off": 0.3})
    total = sum(motor_pattern)
    period = led_cycle["on"] + led_cycle["off"]

    # Collect every boundary where either output changes state.
    boundaries = {0.0, total}
    t = 0.0
    for duration in motor_pattern:
        t += duration
        boundaries.add(min(t, total))
    t = 0.0
    while t < total:
        boundaries.add(t)
        boundaries.add(min(t + led_cycle["on"], total))
        t += period

    def motor_on_at(instant):
        t, on = 0.0, True
        for duration in motor_pattern:
            if instant < t + duration:
                return on
            t += duration
            on = not on
        return False

    def led_on_at(instant):
        return (instant % period) < led_cycle["on"]

    ordered = sorted(boundaries)
    steps = []
    for start, end in zip(ordered, ordered[1:]):
        if end - start > 1e-9:
            steps.append((end - start, led_on_at(start), motor_on_at(start)))
    return steps


class Alerts:
    """Single-owner alert scheduler with priority latching."""

    def __init__(self, sleep=time.sleep):
        self._led, self._motor = _make_devices()
        self.mock = self._led is None
        self._sleep = sleep
        self._cond = threading.Condition()
        self._pending = None          # (rank, profile) awaiting the worker
        self._active_rank = 0         # rank currently playing (0 = idle)
        self._closing = False
        self.trace = []               # mock/test observability
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    @staticmethod
    def _admit(rank, active_rank, pending_rank):
        """Priority policy: admit only if nothing strictly higher is active
        or already pending. Equal rank preempts (newest alert of the same
        urgency wins); lower rank is dropped — the latch."""
        return rank >= active_rank and rank >= pending_rank

    def alert(self, urgency):
        """Request an alert. Returns "queued" or "dropped:latched"."""
        profile = config.ALERT_PROFILES.get(
            urgency, config.ALERT_PROFILES[config.DEFAULT_URGENCY])
        rank = config.URGENCY_RANK.get(urgency, 2)
        with self._cond:
            pending_rank = self._pending[0] if self._pending else 0
            if not self._admit(rank, self._active_rank, pending_rank):
                return "dropped:latched"
            self._pending = (rank, profile)
            self._cond.notify()
        return "queued"

    def _loop(self):
        while True:
            with self._cond:
                while self._pending is None and not self._closing:
                    self._cond.wait()
                if self._closing:
                    self._set_outputs(None, False)   # always end dark
                    return
                rank, profile = self._pending
                self._pending = None
                self._active_rank = rank
            try:
                self._play(profile)
            finally:
                with self._cond:
                    self._active_rank = 0
                    clear = self._pending is None and not self._closing
                if clear:
                    self._set_outputs(None, False)   # only the owner clears

    def _play(self, profile):
        if self.mock:
            print(f"[gpio-mock] rgb={profile['rgb']} led={profile['led']} "
                  f"motor={profile['motor']}", file=sys.stderr)
        for duration, led_on, motor_on in build_timeline(profile):
            with self._cond:
                if self._pending is not None or self._closing:
                    return               # preempted by an admitted alert
            self._set_outputs(profile["rgb"] if led_on else None, motor_on)
            self._sleep(duration)

    def _set_outputs(self, rgb, motor_on):
        """The only writer of hardware state; called solely by the worker."""
        self.trace.append((rgb, motor_on))
        if self.mock:
            return
        if rgb is None:
            self._led.off()
        else:
            self._led.color = rgb
        self._motor.value = 1.0 if motor_on else 0.0

    def close(self):
        with self._cond:
            self._closing = True
            self._cond.notify()
        self._worker.join(timeout=2.0)
        if not self.mock:
            try:
                self._led.off()
                self._motor.value = 0.0
            except Exception:
                pass
        for dev in (self._led, self._motor):
            try:
                if dev is not None:
                    dev.close()
            except Exception:
                pass


# ======================================================================
# ntfy push — POST the event to a public ntfy.sh topic
# ======================================================================

async def push(client, event, profile):
    """Fire a phone push. Returns True on send, False on failure, and None
    when push isn't configured (no topic or no client)."""
    if client is None or not config.NTFY_TOPIC:
        return None
    url = f"{config.NTFY_SERVER}/{config.NTFY_TOPIC}"
    message = f"{event['label'].replace('_', ' ')} ({event['confidence']:.0%})"
    try:
        response = await client.post(
            url,
            content=message.encode("utf-8"),
            headers={
                "Title": f"Earshot: {event['label'].replace('_', ' ')}",
                "Priority": profile["ntfy"],
                "Tags": ",".join(profile["tags"]),
            },
        )
        return response.status_code < 400
    except Exception as exc:   # no internet, ntfy down — never break dispatch
        print(f"[ntfy] push failed: {exc}", file=sys.stderr)
        return False
