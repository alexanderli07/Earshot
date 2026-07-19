"""Central knobs for the Earshot ML component.

Everything you'll want to tune Sunday morning in the venue's actual noise
lives in this file: thresholds, debounce, teach-mode cutoff.
"""

import os
from pathlib import Path

from .artifacts import Artifact

# --- Audio geometry (fixed by the TFLite YAMNet model, don't touch) ---
SAMPLE_RATE = 16_000
WINDOW_SAMPLES = 15_600      # 0.975 s — exactly what TFLite YAMNet expects
HOP_SAMPLES = 8_000          # ~0.5 s hop between windows

# --- Stability ---
CONSECUTIVE_WINDOWS = 2      # windows above threshold before an event fires
DEBOUNCE_SECONDS = 10.0      # suppress repeats of the same event

# --- Teach mode ---
TAUGHT_SIMILARITY_CUTOFF = 0.80   # cosine similarity floor for a taught match
TAUGHT_CONSECUTIVE_WINDOWS = 1    # transients (claps) may only land in one window
TAUGHT_URGENCY = "medium"

# --- Files ---
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR = Path(os.environ.get("EARSHOT_MODEL_DIR") or _DEFAULT_MODEL_DIR)
MODEL_PATH = MODEL_DIR / "yamnet.tflite"
CLASS_MAP_PATH = MODEL_DIR / "yamnet_class_map.csv"
TAUGHT_STORE_PATH = MODEL_DIR / "taught_sounds.npz"

MODEL_ARTIFACT = Artifact(
    url="https://tfhub.dev/google/lite-model/yamnet/tflite/1?lite-format=tflite",
    path=MODEL_PATH,
    sha256="141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3",
)
CLASS_MAP_ARTIFACT = Artifact(
    url=(
        "https://raw.githubusercontent.com/tensorflow/models/"
        "dfffd623b6be8d1d9744b8e261fbac370d17c46d/research/audioset/yamnet/"
        "yamnet_class_map.csv"
    ),
    path=CLASS_MAP_PATH,
    sha256="cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2",
)

# Retired public alarm identities remain reserved so taught sounds cannot be
# canonicalized into the built-in smoke alarm by newer backends.
LEGACY_ALARM_EVENT_LABELS = frozenset({"fire_alarm", "fire_smoke_alarm"})

# --- YAMNet display names -> Earshot events ---
# Class names must match the display_name column of yamnet_class_map.csv
# exactly. Thresholds are starting points; tune with clips played into real
# room noise.
EVENT_MAP = [
    {"label": "smoke_alarm",
     "classes": ["Smoke detector, smoke alarm", "Fire alarm"],
     "threshold": 0.30, "urgency": "high"},
    {"label": "doorbell", "classes": ["Doorbell", "Ding-dong"],
     "threshold": 0.35, "urgency": "medium"},
    {"label": "knock", "classes": ["Knock"],
     "threshold": 0.40, "urgency": "medium"},
    {"label": "baby_cry", "classes": ["Baby cry, infant cry"],
     "threshold": 0.35, "urgency": "high"},
    {"label": "glass_break", "classes": ["Shatter"],
     "threshold": 0.40, "urgency": "high"},
]

RESERVED_EVENT_LABELS = frozenset({
    *(entry["label"].strip().casefold() for entry in EVENT_MAP),
    *(label.casefold() for label in LEGACY_ALARM_EVENT_LABELS),
})