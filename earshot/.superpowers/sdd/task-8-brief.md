### Task 8: Real-model integration test, documentation, CI, and hygiene

**Files:**
- Create: `ml/tests/test_real_model.py`
- Modify: `ml/README.md`
- Create: `.github/workflows/tests.yml`
- Delete after exact-path verification: generated `__pycache__`, `*.pyc`, and outer `__MACOSX` artifacts

**Interfaces:**
- Produces: opt-in pytest marker `integration` for the actual downloaded model.
- Produces: complete Windows/Pi operator documentation.

- [ ] **Step 1: Write the integration test before downloading into the project**

```python
from pathlib import Path

import numpy as np
import pytest

from earshot_ml import config
from earshot_ml.pipeline import YamNet


pytestmark = pytest.mark.integration


def test_downloaded_model_contract():
    if not config.MODEL_PATH.exists() or not config.CLASS_MAP_PATH.exists():
        pytest.skip("run `earshot download` first")
    model = YamNet()
    scores, embedding = model.infer(np.zeros(15_600, np.float32))
    assert scores.shape == (521,)
    assert embedding.shape == (1024,)
    assert np.isfinite(scores).all()
    assert np.isfinite(embedding).all()
    assert len(model.class_names) == 521
```

- [ ] **Step 2: Confirm the integration test skips before download**

Run: `python -m pytest tests/test_real_model.py -q`

Expected: one skipped test with the documented reason.

- [ ] **Step 3: Update docs and CI**

Rewrite README setup and validation sections for Windows and Raspberry Pi, explain both data flows, document `EARSHOT_MODEL_DIR`, automation versus hardware tests, stop-event backend usage, score semantics, calibration, and safety limitations. Add a repository-root Ubuntu GitHub Actions workflow with `ml` as its working directory; it installs `.[test]` and runs all tests except the integration marker on Python 3.11 and 3.14.

- [ ] **Step 4: Run full fast verification**

Run: `python -m pytest -m "not integration" -q`

Expected: all fast tests pass with zero failures or warnings.

- [ ] **Step 5: Download and verify the real model**

Run: `python -m earshot_ml.cli download`

Expected: both artifacts pass their fixed SHA-256 checks and the CLI reports a valid 521/1,024 tensor contract.

Run: `python -m pytest tests/test_real_model.py -q`

Expected: one passing integration test.

- [ ] **Step 6: Verify packaging and commands from a clean environment**

Create a fresh temporary virtual environment, install `.[test]`, run all fast tests, run both help entry points, and run the real-model test using the verified model directory.

Expected: installation and every non-hardware check succeed.

- [ ] **Step 7: Remove regenerable archive clutter safely**

Resolve and print every targeted cache and `__MACOSX` path, confirm all targets remain under `C:\Users\alexa\Downloads\earshot`, then remove only those verified paths. Do not remove source, models, tests, documentation, or virtual environments.

- [ ] **Step 8: Final review checkpoint**

Inspect the complete file set, rerun fast and real-model suites, and request a code-quality review. Record the exact test counts, skipped hardware-only checks, model digest, and any remaining limitations for the handoff.
