# Task 1 review package (archive without Git metadata)

## Changed files

- Created `ml/pyproject.toml`
- Created `ml/tests/test_packaging.py`
- Created `.gitignore`
- Replaced `ml/requirements.txt`

## Previous `ml/requirements.txt`

```text
numpy
# needs the PortAudio system package: sudo apt install -y libportaudio2
sounddevice
# TFLite interpreter: tflite-runtime on the Pi, ai-edge-litert elsewhere
tflite-runtime; platform_machine == "aarch64" or platform_machine == "armv7l"
ai-edge-litert; platform_machine != "aarch64" and platform_machine != "armv7l"
```

The other three changed files did not previously exist.

## Current `.gitignore`

```text
ml/.venv/
__pycache__/
*.py[cod]
.pytest_cache/
ml/models/*.tflite
ml/models/*.csv
ml/models/*.npz
ml/models/*.part
.DS_Store
__MACOSX/
```

## Current `ml/requirements.txt`

```text
-e .
```

## Current `ml/pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[project]
name = "earshot-ml"
version = "0.1.0"
requires-python = ">=3.10,<3.15"
dependencies = [
    "numpy",
    "sounddevice",
    "ai-edge-litert; platform_machine != 'armv7l'",
    "tflite-runtime==2.14.0; platform_machine == 'armv7l' and python_version < '3.12'",
]

[project.optional-dependencies]
test = ["pytest", "tomli; python_version < '3.11'"]

[project.scripts]
earshot = "earshot_ml.cli:main"

[tool.setuptools.packages.find]
where = ["."]
include = ["earshot_ml*"]
```

## Current `ml/tests/test_packaging.py`

```python
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_declares_package_and_cli():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["requires-python"] == ">=3.10,<3.15"
    assert data["project"]["scripts"]["earshot"] == "earshot_ml.cli:main"
    assert "test" in data["project"]["optional-dependencies"]
    assert (
        "tomli; python_version < '3.11'"
        in data["project"]["optional-dependencies"]["test"]
    )
```

## Parent verification after report

`python -m pip install --no-build-isolation -e ".[test]"` succeeded after installing the declared `setuptools>=69` build backend in the project-local virtual environment. The editable `earshot-ml==0.1.0` package built and installed successfully.

## Review-fix delta

The Python 3.10 compatibility review finding was addressed test-first. `tomli` is now a conditional test dependency for Python below 3.11, and `test_packaging.py` falls back from `tomllib` to `tomli`. The implementer appended RED/GREEN evidence to `task-1-report.md`; focused plus regression verification reports 15 passing tests.
