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
        # Iterate a COPY: clients drop mid-send and one dead phone must not
        # crash the loop (per the spec's watch-out).
        for ws in list(self._connections):
            try:
                await ws.send_json(event)
            except Exception:
                await self.disconnect(ws)


# ======================================================================
# GPIO alerts — RGB LED colour by urgency + vibration motor pattern
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


class Alerts:
    """Non-blocking: each alert runs in a background thread and a generation
    counter lets a newer alert cut off an in-flight one."""

    def __init__(self):
        self._led, self._motor = _make_devices()
        self.mock = self._led is None
        self._gen = 0
        self._lock = threading.Lock()

    def alert(self, urgency):
        profile = config.ALERT_PROFILES.get(
            urgency, config.ALERT_PROFILES[config.DEFAULT_URGENCY])
        with self._lock:
            self._gen += 1
            gen = self._gen
        threading.Thread(target=self._run, args=(gen, profile),
                         daemon=True).start()

    def _run(self, gen, profile):
        if self.mock:
            print(f"[gpio-mock] urgency rgb={profile['rgb']} "
                  f"led={profile['led']} motor={profile['motor']}",
                  file=sys.stderr)
            return
        pattern = config.MOTOR_PATTERNS.get(profile["motor"], [0.2])
        self._led.color = profile["rgb"]
        on = True
        for duration in pattern:
            if gen != self._gen:            # a newer alert superseded us
                break
            self._motor.value = 1.0 if on else 0.0
            time.sleep(duration)
            on = not on
        self._motor.value = 0.0
        if gen == self._gen:
            self._led.off()

    def close(self):
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
    """Fire a phone push. No-op when no topic is configured or no client."""
    if client is None or not config.NTFY_TOPIC:
        return
    url = f"{config.NTFY_SERVER}/{config.NTFY_TOPIC}"
    message = f"{event['label'].replace('_', ' ')} ({event['confidence']:.0%})"
    try:
        await client.post(
            url,
            content=message.encode("utf-8"),
            headers={
                "Title": f"Earshot: {event['label'].replace('_', ' ')}",
                "Priority": profile["ntfy"],
                "Tags": ",".join(profile["tags"]),
            },
        )
    except Exception as exc:   # no internet, ntfy down — never break dispatch
        print(f"[ntfy] push failed: {exc}", file=sys.stderr)
