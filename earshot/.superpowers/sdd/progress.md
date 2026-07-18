# Earshot SDD Progress

Plan: `docs/superpowers/plans/2026-07-17-earshot-reliability.md`

Baseline: 14 existing logic tests passed with Python 3.14.0; CLI help worked; pip check clean.
Artifact validation: full YAMNet model hash and 521/1024 inference contract verified from temporary official artifacts.

Task 1: complete (review approved; 15 tests pass; editable install verified)
Task 2: complete (review approved; 24 tests pass)
Task 3: complete (review approved; 43 tests pass)
Task 4: complete (review approved after fix; 59 tests pass)
Task 5: complete (review approved after Task 6 integration; 84 tests pass)
Task 6: complete (review approved; 84 tests pass)
Task 7: complete (review approved after fix; 93 tests pass)
Task 8: complete (documentation, CI, packaging, cleanup, and production-model integration verified)

Release hardening: complete

- Actionable CLI diagnostics cover missing interpreter backends, microphone/device
  failures, malformed WAV/class-map input, and taught-store persistence failures;
  unrelated `OSError`s still propagate for debugging.
- YAMNet interpreter access is serialized per instance and output tensors are strictly validated.
- Teach/forget persistence is transactional: failed saves restore the prior in-memory state.
- WAV decoding, resampling, windowing, queue-gap handling, and invalid-input boundaries have regression tests.
- Artifact transfers use bounded, timed, unique same-directory temporary files
  with atomic replacement; destination checksum reads and installation are
  serialized per process to avoid Windows sharing races.
- Raspberry Pi `armv7l` support is explicit: Python 3.10/3.11 only; unsupported versions fail dependency resolution.

Final clean-tree verification (2026-07-18):

- `python -B -m pytest -p no:cacheprovider -q`: 177 passed in 2.13 s, including real-model integration.
- Fast split: 176 passed, 1 integration test deselected in 1.97 s.
- Artifact-focused suite: 15 passed in 0.54 s.
- Both `earshot --help` and `python -m earshot_ml.cli --help` expose all six commands.
- `earshot download` revalidated the cached official model and class map.
- `pip check`: no broken requirements.
- Runtime model contract: scores `(521,)`, embedding `(1024,)`, 521 class names.
- SHA-256: model `141fba1cdaae842c816f28edc4937e8b4f0af4c8df21862ccc6b52dc567993c3`; class map `cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2`.
- A separate non-editable wheel environment also passed the then-current 165 fast tests, the real-model integration test, both installed CLI entry paths, and `pip check` before its temporary environment was removed. After final CLI hardening, a fresh wheel was rebuilt, imported from an isolated target, and passed the real-model `(521,)`/`(1024,)` contract before cleanup.
- Generated build/test/cache artifacts were removed and verified absent outside the preserved project `.venv`.

Manual acceptance remaining: exercise the actual target microphone, room acoustics, and device selection on hardware.
