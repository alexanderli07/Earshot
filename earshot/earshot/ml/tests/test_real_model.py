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
