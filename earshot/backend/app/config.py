"""Backend tuning knobs — pins, ntfy, alert profiles, paths.

Agree on the pin map once, change never (per the spec).
"""

import os
from pathlib import Path

# --- Hardware pin map (BCM numbering) ---
# R = GPIO17, G = GPIO27, B = GPIO22, motor = GPIO18. Fixed.
PIN_R = 17
PIN_G = 27
PIN_B = 22
PIN_MOTOR = 18

# --- ntfy phone push ---
# Pick a unique, hard-to-guess topic (ntfy topics are public!) and export it:
#   export EARSHOT_NTFY_TOPIC=earshot-7f3a9c
# Push is disabled when unset, so we never spam a guessable public topic.
NTFY_SERVER = os.environ.get("EARSHOT_NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = os.environ.get("EARSHOT_NTFY_TOPIC", "")

# --- Server ---
HOST = os.environ.get("EARSHOT_HOST", "0.0.0.0")   # 0.0.0.0 so phones on the
PORT = int(os.environ.get("EARSHOT_PORT", "8000"))  # hotspot can reach the Pi
RECENT_EVENTS_MAX = 100

# --- Files ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RULES_PATH = DATA_DIR / "rules.json"

# --- Alert profiles: urgency -> how the four sinks react ---
# rgb:     RGB LED colour (0/1 per channel)
# led:     LED pattern name (see sinks.Alerts)
# motor:   vibration pattern name (see MOTOR_PATTERNS)
# ntfy:    ntfy priority (min|low|default|high|urgent)
# tags:    ntfy tags -> emoji on the phone
ALERT_PROFILES = {
    "high":   {"rgb": (1, 0, 0), "led": "strobe", "motor": "long",
               "ntfy": "urgent", "tags": ["rotating_light"]},
    "medium": {"rgb": (1, 1, 0), "led": "pulse",  "motor": "pulse",
               "ntfy": "high",   "tags": ["bell"]},
    "low":    {"rgb": (0, 0, 1), "led": "blink",  "motor": "short",
               "ntfy": "default", "tags": ["information_source"]},
}
DEFAULT_URGENCY = "medium"

# Priority ranks: a lower-ranked alert never interrupts an active higher one
# (a microwave ding must not cancel a smoke-alarm strobe).
URGENCY_RANK = {"low": 1, "medium": 2, "high": 3}

# Vibration patterns as alternating on/off seconds, starting with ON.
#   urgent = strobe plus a long buzz; notice = one pulse.
MOTOR_PATTERNS = {
    "long":  [0.5, 0.12, 0.5, 0.12, 1.6],   # strobe, then long buzz
    "pulse": [0.30],                          # one pulse
    "short": [0.12],                          # brief tick
}

# LED pattern timing: the LED cycles on/off at these periods (seconds) for
# the duration of the motor pattern, then turns off.
LED_PATTERNS = {
    "strobe": {"on": 0.12, "off": 0.12},
    "pulse":  {"on": 0.45, "off": 0.25},
    "blink":  {"on": 0.20, "off": 0.60},
}

# Debug endpoint default when no label/urgency is supplied.
DEBUG_DEFAULT = {"label": "doorbell", "urgency": "medium"}
