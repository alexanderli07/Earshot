### Task 3: Model tensor contract and inference validation

**Files:**
- Modify: `ml/earshot_ml/pipeline.py`
- Create: `ml/tests/test_pipeline.py`

**Interfaces:**
- Produces: `YamNet(..., interpreter=None)` for production loading or deterministic fake-interpreter tests.
- Preserves: `infer(waveform) -> tuple[np.ndarray, np.ndarray]` and `top(scores, k=5)`.

- [ ] **Step 1: Write failing fake-interpreter tests**

Create a fake interpreter exposing configurable input/output details and tensor values. Test that a valid model yields `(521,)` scores and `(1024,)` embeddings, while a score-only model raises `ModelContractError` containing observed output shapes. Test class maps with other than 521 rows, non-float inputs, wrong waveform length, and non-finite waveform values.

Representative assertion:

```python
def test_score_only_model_reports_contract_error(tmp_path):
    class_map = write_class_map(tmp_path, 521)
    fake = FakeInterpreter(outputs=[tensor_detail(7, [1, 521])])
    with pytest.raises(ModelContractError, match="1024"):
        YamNet(Path("unused"), class_map, interpreter=fake)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_pipeline.py -q`

Expected: failures because injection and contract errors are not implemented.

- [ ] **Step 3: Implement model validation**

Add `ModelContractError`. Validate class-map length, input dtype and shape, required outputs by last dimension, and diagnostic messages listing tensor names/shapes/dtypes. Resize a dynamic or incompatible input to `[WINDOW_SAMPLES]`, then allocate tensors and refresh details. Make `infer` reject anything except a finite float-convertible one-dimensional 15,600-sample waveform and normalize outputs by averaging frame dimensions.

- [ ] **Step 4: Verify model unit tests**

Run: `python -m pytest tests/test_pipeline.py tests/test_ml.py -q`

Expected: all tests pass.

---

