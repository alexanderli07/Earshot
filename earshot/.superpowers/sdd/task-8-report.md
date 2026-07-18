# Task 8 Report: Real-model test, documentation, and CI

Date: 2026-07-18

## Completed scope

- Added `ml/tests/test_real_model.py` with the approved production contract:
  the test is marked `integration`, skips when either configured artifact is
  absent, infers a zero-valued 15,600-sample float32 waveform when present,
  and checks finite `(521,)` scores, a finite `(1024,)` embedding, and 521
  class names.
- Registered the `integration` marker in `ml/pyproject.toml`.
- Rewrote `ml/README.md` to cover purpose and scope; module architecture;
  pretrained and taught flows; Windows PowerShell and Raspberry Pi setup;
  installed and wrapper commands; verified model acquisition; fast,
  integration, and manual hardware tests; device selection and gain; all CLI
  workflows; `EARSHOT_MODEL_DIR`; callback, queue, and stop-event backend
  integration; payload and confidence semantics; microphone and application
  queue behavior; tuning, calibration, safety limits, and troubleshooting.
- Added repository-root `.github/workflows/tests.yml` with
  `actions/checkout@v4`, `actions/setup-python@v5`, an Ubuntu matrix for
  Python 3.11 and 3.14, `ml` as the run working directory,
  `python -m pip install ".[test]"`, and the fast pytest selection.

## Test-first evidence

The configured `ml/models` directory contained no model or class-map files.
No project model download occurred before the smoke test.

The global Python did not have pytest, so the existing project virtual
environment was used without creating or modifying an environment:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_real_model.py -q -rs
```

Before marker registration:

```text
SKIPPED [1] tests\test_real_model.py:15: run `earshot download` first
1 skipped, 1 warning in 0.13s
```

The one warning was the expected `PytestUnknownMarkWarning` that established
the need for marker registration. After adding the pyproject marker, the same
command reported:

```text
SKIPPED [1] tests\test_real_model.py:15: run `earshot download` first
1 skipped in 0.13s
```

Post-registration smoke summary: 0 passed, 1 skipped, 0 warnings.

## Fast verification

Command:

```powershell
.\.venv\Scripts\python.exe -m pytest -m "not integration" -q
```

Fresh result:

```text
93 passed, 1 deselected in 0.90s
```

Fast-suite totals: 93 passed, 0 failed, 0 skipped, 1 integration test
deselected, and 0 warnings. Exit code: 0.

## Intentionally not performed in this subtask

These remaining Task 8 steps are owned by the parent agent:

1. Run the live checksum-verified model download and capture the model/class-map
   validation result.
2. Run `tests/test_real_model.py` with the downloaded artifacts and capture
   the expected one passing integration test.
3. Create a clean temporary environment, install `.[test]`, run fast tests,
   verify both CLI help entry points, and reuse the verified model directory
   for the real-model smoke.
4. Resolve, print, validate, and remove only the approved `__pycache__`,
   `*.pyc`, and outer `__MACOSX` artifacts.
5. Perform the final combined review and record the model digest and any manual
   hardware limitations.

No network/model download, microphone access, virtual-environment creation,
Git operation, cache deletion, or unrelated production-behavior edit was
performed here.
