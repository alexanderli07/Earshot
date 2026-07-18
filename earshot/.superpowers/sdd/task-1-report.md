# Task 1 Report: Package and Test Harness

## Status

`DONE`

The required package metadata, legacy requirements bridge, ignore rules, and packaging test are implemented. The focused test and all existing ML tests pass. The parent subsequently verified the editable install with `python -m pip install --no-build-isolation -e ".[test]"`.

## Implementation

- Added a setuptools `pyproject.toml` for project `earshot-ml` version `0.1.0`.
- Declared Python support as `>=3.10,<3.15`.
- Declared `numpy` and `sounddevice` runtime dependencies.
- Declared `ai-edge-litert` for every machine except `armv7l`.
- Declared `tflite-runtime==2.14.0` for `armv7l` when Python is below 3.12.
- Added the `test` extra containing `pytest`.
- Added setuptools discovery for `earshot_ml*`.
- Added the required console-script metadata: `earshot = "earshot_ml.cli:main"`.
- Replaced the legacy requirements list with `-e .`, making `requirements.txt` consume the package metadata.
- Added all prescribed repository ignore patterns.
- Added the prescribed TOML packaging test.

## Files

- Created `ml/pyproject.toml`
- Created `ml/tests/test_packaging.py`
- Created `.gitignore`
- Modified `ml/requirements.txt`

No Git repository was initialized and no commit was created.

## TDD Evidence

### Test prerequisite

The first runner attempts could not execute the test because pytest was absent from both the project virtual environment and the system interpreter. Pytest was then made available in `ml/.venv` by the parent. Those missing-runner errors were not counted as RED.

### RED: test added before metadata

Command, from `ml`:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_packaging.py -q
```

Result: exit code 1, with the intended missing-feature failure:

```text
F                                                                        [100%]
E       FileNotFoundError: [Errno 2] No such file or directory: 'C:\Users\alexa\Downloads\earshot\earshot\ml\pyproject.toml'
FAILED tests/test_packaging.py::test_pyproject_declares_package_and_cli - Fil...
1 failed in 0.09s
```

This was the valid RED because the test executed and failed specifically because `pyproject.toml` did not yet exist.

### GREEN: focused test after minimal metadata

Command, from `ml`:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_packaging.py -q
```

Result: exit code 0.

```text
.                                                                        [100%]
1 passed in 0.01s
```

### Final focused and regression verification

Command, from `ml`:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_packaging.py tests/test_ml.py -q
```

Result: exit code 0.

```text
...............                                                          [100%]
15 passed in 0.12s
```

This covers the new packaging test plus all 14 tests in the existing `tests/test_ml.py` suite.

## Editable Install Verification

Attempted command, from `ml`:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -e '.[test]'
```

The sandboxed attempt exited 1 while pip was creating an isolated build environment. It could not reach the package index to obtain `setuptools>=69` and reported `WinError 10013`, followed by `No matching distribution found for setuptools>=69`. This is an environment/network failure rather than evidence of invalid project metadata. An escalated retry hung and was aborted; the parent explicitly directed that it not be run again and will perform the approved install verification.

## Self-Review

- Confirmed the test was written and observed failing for the expected reason before `pyproject.toml` was created.
- Confirmed the exact required project name, version, Python range, script target, test extra, package-discovery pattern, and dependency markers are present.
- Confirmed `requirements.txt` contains only `-e .`.
- Confirmed `.gitignore` contains every entry named in the task brief.
- Confirmed the focused test and the full pre-existing ML test file pass together.
- Confirmed no Git initialization or commit was performed.
- Kept changes restricted to the four files named in Task 1 plus this required report.

## Concerns and Follow-Up

1. The required script metadata targets `earshot_ml.cli:main`, but the current source tree has `ml/cli.py` and does not yet have `ml/earshot_ml/cli.py`. Setuptools can install console metadata without importing the target, so Task 7 must supply that module before the installed `earshot` command can run.

## Review Follow-Up: Python 3.10 Test Compatibility

The review identified that `tests/test_packaging.py` imported standard-library `tomllib` unconditionally even though the package supports Python 3.10. Since `tomllib` is standard-library only on Python 3.11 and newer, the test now falls back to `tomli` after `ModuleNotFoundError`, and the `test` extra conditionally installs that backport below Python 3.11.

### Files Modified

- `ml/tests/test_packaging.py`
  - Added `tomllib` / `tomli` fallback import.
  - Added an assertion for the exact conditional backport dependency.
- `ml/pyproject.toml`
  - Added `tomli; python_version < '3.11'` to the `test` extra.

### Review RED

Command, from `ml`:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_packaging.py -q
```

Result: exit code 1, with the intended missing-metadata assertion:

```text
F                                                                        [100%]
>       assert (
            "tomli; python_version < '3.11'"
            in data["project"]["optional-dependencies"]["test"]
        )
E       assert "tomli; python_version < '3.11'" in ['pytest']

FAILED tests/test_packaging.py::test_pyproject_declares_package_and_cli - ass...
1 failed in 0.06s
```

This RED confirms that the new test detects omission of the Python 3.10 backport from package metadata.

### Review GREEN

Focused command, from `ml`:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_packaging.py -q
```

Result: exit code 0.

```text
.                                                                        [100%]
1 passed in 0.01s
```

Full Task 1 regression command, from `ml`:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest tests/test_packaging.py tests/test_ml.py -q
```

Result: exit code 0.

```text
...............                                                          [100%]
15 passed in 1.77s
```

### Review Self-Check

- Confirmed the dependency assertion failed before package metadata changed.
- Confirmed the dependency marker is restricted to Python versions below 3.11, so Python 3.11+ does not install an unnecessary backport.
- Confirmed the test uses standard-library `tomllib` when available and only falls back when the module itself is absent.
- Confirmed the focused test and all 14 original tests pass together.
- No Git initialization or commit was performed.
