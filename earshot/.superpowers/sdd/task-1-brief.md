### Task 1: Package and test harness

**Files:**
- Create: `ml/pyproject.toml`
- Create: `.gitignore`
- Create: `ml/tests/test_packaging.py`
- Modify: `ml/requirements.txt`

**Interfaces:**
- Produces: an editable package install, pytest test discovery, and `earshot` console-script metadata.
- Consumes: existing `earshot_ml` package and top-level `cli.py`.

- [ ] **Step 1: Write the failing packaging test**

```python
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parent.parent


def test_pyproject_declares_package_and_cli():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert data["project"]["requires-python"] == ">=3.10,<3.15"
    assert data["project"]["scripts"]["earshot"] == "earshot_ml.cli:main"
    assert "test" in data["project"]["optional-dependencies"]
```

- [ ] **Step 2: Run the test to verify RED**

Run: `python -m pytest tests/test_packaging.py -q`

Expected: failure because `pyproject.toml` does not exist.

- [ ] **Step 3: Add package metadata and ignore rules**

Create a setuptools `pyproject.toml` with project name `earshot-ml`, version `0.1.0`, `requires-python = ">=3.10,<3.15"`, NumPy and sounddevice dependencies, LiteRT for every architecture except 32-bit ARM, `tflite-runtime==2.14.0` for `armv7l` on Python below 3.12, pytest in the `test` extra, package discovery for `earshot_ml*`, and script `earshot = "earshot_ml.cli:main"`.

Update `requirements.txt` to contain `-e .` so the legacy setup command uses package metadata. Add repository-root `.gitignore` entries for `ml/.venv/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `ml/models/*.tflite`, `ml/models/*.csv`, `ml/models/*.npz`, `ml/models/*.part`, `.DS_Store`, and `__MACOSX/`.

- [ ] **Step 4: Verify GREEN and installation metadata**

Run: `python -m pytest tests/test_packaging.py -q`

Expected: one passing test.

Run: `python -m pip install -e ".[test]"`

Expected: editable install succeeds and installs the declared runtime/test dependencies.

---

