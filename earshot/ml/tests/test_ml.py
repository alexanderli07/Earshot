"""All pure-logic tests. Runs anywhere, no mic or model:

    python tests/test_ml.py

Regions: event streak + debounce | teach-mode matching + persistence.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from earshot_ml.core import EventDetector, Observation, TeachStore  # noqa: E402

# ======================================================================
# Event streak + debounce
# ======================================================================


def obs(label, above, conf=0.9, consecutive=2):
    return Observation(label=label, confidence=conf, above=above,
                       urgency="high", source="pretrained",
                       consecutive=consecutive)


def test_needs_two_consecutive_windows():
    d = EventDetector(debounce_s=10)
    assert d.update([obs("smoke_alarm", True)], now=0.0) == []
    fired = d.update([obs("smoke_alarm", True)], now=0.5)
    assert [e.label for e in fired] == ["smoke_alarm"]


def test_gap_resets_streak():
    d = EventDetector(debounce_s=10)
    d.update([obs("smoke_alarm", True)], now=0.0)
    d.update([obs("smoke_alarm", False)], now=0.5)
    assert d.update([obs("smoke_alarm", True)], now=1.0) == []
    assert d.update([obs("smoke_alarm", True)], now=1.5) != []


def test_debounce_suppresses_repeats_then_refires():
    d = EventDetector(debounce_s=10)
    now, fired_times = 0.0, []
    for _ in range(50):  # 25 s of continuous alarm at 0.5 s hop
        for e in d.update([obs("smoke_alarm", True)], now=now):
            fired_times.append(now)
        now += 0.5
    assert fired_times == [0.5, 10.5, 20.5]


def test_labels_independent():
    d = EventDetector(debounce_s=10)
    d.update([obs("smoke_alarm", True), obs("doorbell", True)], now=0.0)
    fired = d.update([obs("smoke_alarm", True), obs("doorbell", False)], now=0.5)
    assert [e.label for e in fired] == ["smoke_alarm"]


def test_absent_label_resets_streak():
    """A taught sound that stops matching disappears from observations."""
    d = EventDetector(debounce_s=10)
    d.update([obs("kettle", True)], now=0.0)
    d.update([], now=0.5)                          # no match this window
    assert d.update([obs("kettle", True)], now=1.0) == []


def test_consecutive_one_fires_immediately():
    d = EventDetector(debounce_s=10)
    fired = d.update([obs("clap", True, consecutive=1)], now=0.0)
    assert [e.label for e in fired] == ["clap"]


def test_event_dict_shape():
    d = EventDetector(debounce_s=10)
    fired = d.update([obs("clap", True, conf=0.876, consecutive=1)], now=42.0)
    assert fired[0].to_dict() == {
        "label": "clap", "urgency": "high", "confidence": 0.876,
        "source": "pretrained", "timestamp": 42.0,
    }


# ======================================================================
# Teach-mode matching + persistence
# ======================================================================


def vec(seed):
    return np.random.default_rng(seed).normal(size=1024).astype(np.float32)


def test_empty_store_matches_nothing():
    assert TeachStore(cutoff=0.8).match(vec(0)) is None


def test_exact_clip_matches_itself():
    s = TeachStore(cutoff=0.8)
    s.add("kettle", vec(1))
    name, sim = s.match(vec(1))
    assert name == "kettle" and sim > 0.999


def test_unrelated_sound_below_cutoff():
    s = TeachStore(cutoff=0.8)
    s.add("kettle", vec(1))
    assert s.match(vec(2)) is None  # random 1024-dim vectors are near-orthogonal


def test_nearest_neighbor_wins():
    s = TeachStore(cutoff=0.5)
    s.add("kettle", vec(1))
    s.add("dryer", vec(2))
    noisy = vec(2) + 0.1 * vec(3)
    name, _ = s.match(noisy)
    assert name == "dryer"


def test_learned_counts_clips():
    s = TeachStore(cutoff=0.8)
    for seed in (1, 2, 3):
        s.add("kettle", vec(seed))
    s.add("dryer", vec(4))
    assert s.learned() == [{"name": "kettle", "clips": 3},
                           {"name": "dryer", "clips": 1}]


def test_forget():
    s = TeachStore(cutoff=0.8)
    s.add("kettle", vec(1))
    s.add("dryer", vec(2))
    assert s.forget("kettle") == 1
    assert s.match(vec(1)) is None
    assert s.match(vec(2))[0] == "dryer"


def test_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "taught.npz"
        s = TeachStore(path=path, cutoff=0.8)
        s.add("kettle", vec(1))
        s.save()
        reloaded = TeachStore(path=path, cutoff=0.8)
        assert reloaded.match(vec(1))[0] == "kettle"
        assert reloaded.learned() == [{"name": "kettle", "clips": 1}]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")


def test_reset_clears_streaks_after_capture_gap():
    d = EventDetector(debounce_s=10)
    assert d.update([obs("smoke_alarm", True)], now=0.0) == []
    d.reset()   # dropped audio between the two windows
    assert d.update([obs("smoke_alarm", True)], now=0.5) == []
    fired = d.update([obs("smoke_alarm", True)], now=1.0)
    assert [e.label for e in fired] == ["smoke_alarm"]


def test_reset_keeps_debounce_timestamps():
    d = EventDetector(debounce_s=10)
    d.update([obs("smoke_alarm", True)], now=0.0)
    d.update([obs("smoke_alarm", True)], now=0.5)   # fires
    d.reset()   # a gap must not allow an immediate re-fire
    d.update([obs("smoke_alarm", True)], now=1.0)
    assert d.update([obs("smoke_alarm", True)], now=1.5) == []
