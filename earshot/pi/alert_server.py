"""Earshot wearable alert unit — runs ON the Raspberry Pi itself.

The Pi drives the alert hardware DIRECTLY on its own GPIO pins — there is
no Arduino in the rig:

    RGB LED  R=GPIO17  G=GPIO27  B=GPIO22   (matches backend config.py)
    Motor    GPIO18 (PWM throttle via a TB6612 driver, direction hardwired)
    Buzzer   GPIO23

Color says WHICH sound (see LABEL_COLORS); urgency says HOW HARD to alert
(blink speed, shake pattern, beeps, duration — see PROFILES). So a high-
urgency baby cry is a furious BLUE strobe, and a low-urgency kettle is a
lazy YELLOW blink. Unknown/taught sounds default to white.

The backend's pi_alert sink POSTs {"label": ..., "urgency": ...} to
http://<pi>:8000/alarm on every forwarded event; /stop cancels. A priority
latch keeps a low alert from interrupting an active high one.

Deployed on the Pi at /home/pi/alert_server.py under systemd
(alert-server.service, Restart=always, WorkingDirectory=/home/pi — the
working dir matters: gpiozero's pin backend needs a writable CWD). After
editing:  sudo systemctl restart alert-server
"""

import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread

from gpiozero import RGBLED, PWMOutputDevice, TonalBuzzer
from gpiozero.tones import Tone

led = RGBLED(17, 27, 22)
motor = PWMOutputDevice(18)
buzzer = TonalBuzzer(23)

# Which sound -> which color (R, G, B channels, 0 or 1).
LABEL_COLORS = {
    "smoke_alarm": (1, 0, 0),   # red
    "fire_alarm":  (1, 0, 0),   # legacy alias
    "baby_cry":    (0, 0, 1),   # blue
    "glass_break": (0, 1, 0),   # green
    "doorbell":    (0, 1, 1),   # cyan
    "knock":       (1, 0, 1),   # magenta
    "kettle":      (1, 1, 0),   # yellow
}
DEFAULT_COLOR = (1, 1, 1)       # white: taught/unknown sounds

# How hard to alert, by urgency.
PROFILES = {
    "high":   {"rank": 3, "led": (0.10, 0.10), "motor": (0.60, 0.15),
               "beep": True,  "seconds": 20},
    "medium": {"rank": 2, "led": (0.45, 0.25), "motor": (0.30, 0.50),
               "beep": False, "seconds": 8},
    "low":    {"rank": 1, "led": (0.20, 0.60), "motor": (0.12, 2.00),
               "beep": False, "seconds": 4},
}

state = {"until": 0, "profile": PROFILES["high"]}


def active():
    return time.time() < state["until"]


def vibrator():
    while True:
        p = state["profile"]
        if active():
            motor.value = 1.0; time.sleep(p["motor"][0])
            motor.value = 0.0; time.sleep(p["motor"][1])
        else:
            motor.value = 0.0; time.sleep(0.1)


def sounder():
    while True:
        p = state["profile"]
        if active() and p["beep"]:
            for _ in range(3):
                if not (active() and state["profile"]["beep"]):
                    break
                buzzer.play(Tone(880)); time.sleep(0.5)
                buzzer.stop(); time.sleep(0.5)
            time.sleep(1.0)
        else:
            buzzer.stop(); time.sleep(0.1)


Thread(target=vibrator, daemon=True).start()
Thread(target=sounder, daemon=True).start()


class AlertHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == "/alarm":
            length = int(self.headers.get("Content-Length") or 0)
            body = {}
            if length:
                try:
                    body = json.loads(self.rfile.read(length))
                except ValueError:
                    pass
            label = (body.get("label") or "smoke_alarm").lower()
            urgency = body.get("urgency") or "high"
            color = LABEL_COLORS.get(label, DEFAULT_COLOR)
            profile = PROFILES.get(urgency, PROFILES["medium"])
            if active() and profile["rank"] < state["profile"]["rank"]:
                self.send_response(200); self.end_headers()
                self.wfile.write(b"dropped:latched\n")
                return
            state["profile"] = profile
            state["until"] = time.time() + profile["seconds"]
            on, off = profile["led"]
            led.blink(on_time=on, off_time=off, on_color=color,
                      n=int(profile["seconds"] / (on + off)))
            self.send_response(200); self.end_headers()
            self.wfile.write(f"alerting:{label}:{urgency}\n".encode())
        elif self.path == "/stop":
            state["until"] = 0
            led.off()
            self.send_response(200); self.end_headers()
            self.wfile.write(b"stopped\n")
        else:
            self.send_response(404); self.end_headers()
    do_GET = do_POST


print("Alert server on 8000 - color = which sound, urgency = how hard")
HTTPServer(("0.0.0.0", 8000), AlertHandler).serve_forever()
