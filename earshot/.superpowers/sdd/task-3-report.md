# Task 3 Report: Model Tensor Contract and Inference Validation

## Outcome

Implemented the Task 3 YAMNet boundary with test-first development. `YamNet`
now accepts an optional interpreter for deterministic tests while preserving its
production model loader, validates the 521-row class map and TFLite tensor
contract before inference, resizes dynamic or incompatible inputs before tensor
allocation, and reports observed tensor names, shapes, and dtypes when a model is
incompatible.

`infer()` now accepts only finite, float-convertible, one-dimensional waveforms
containing exactly 15,600 samples. Valid score and embedding tensors are reduced
across frame dimensions to stable `(521,)` and `(1024,)` outputs. `top(scores,
k=5)` remains unchanged.

All tests use a deterministic fake interpreter. No network, microphone, model
download, or real TFLite backend was used.

## Files Changed

- Modified `ml/earshot_ml/pipeline.py`
  - Added `ModelContractError`.
  - Added tensor-shape, dtype, and diagnostic formatting helpers that handle
    NumPy-backed TFLite detail fields.
  - Added `YamNet(..., interpreter=None)` injection without changing the
    production loader path.
  - Requires exactly 521 class-map rows.
  - Requires one float32 waveform input and resolves its allocated shape to
    `[15600]`.
  - Resizes dynamic or incompatible inputs before allocation, refreshes input
    details afterward, and chains resize errors into diagnostic contract errors.
  - Finds score and embedding outputs by final widths 521 and 1,024 and reports
    every observed output when either is absent.
  - Rejects non-convertible, wrong-rank, wrong-length, and non-finite waveforms
    before setting or invoking the interpreter.
  - Preserves frame averaging and the public `top()` behavior.
- Created `ml/tests/test_pipeline.py`
  - Added a configurable fake interpreter that verifies resize, allocation,
    detail refresh, tensor set, and invocation behavior.
  - Added 19 tests covering valid frame reduction, missing outputs and
    diagnostics, class-map cardinality, input dtype and layout, dynamic/static
    resizing, failed resize diagnostics, waveform validation, production-loader
    preservation, and `top()` ranking.
- Created `.superpowers/sdd/task-3-report.md`
  - Recorded Task 3 TDD, regression, and self-review evidence.

No Task 2 production or test files were modified by this worker.

## RED Evidence

The complete fake-interpreter test file was created before any production edit.
Run from `ml`:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q
```

The suite failed for the requested missing API and contract behavior:

```text
E   TypeError: YamNet.__init__() got an unexpected keyword argument 'interpreter'
E   AttributeError: module 'earshot_ml.pipeline' has no attribute 'ModelContractError'
17 failed, 1 passed in 0.43s
```

The one passing test was the pre-existing production loading path exercised with
an on-disk placeholder and a monkeypatched lazy loader. The failures were caused
by the absent injection and contract implementation, not a collection,
dependency, or test-fixture error.

During self-review, a second focused TDD cycle covered an interpreter that
rejects resizing:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py::test_unresizable_input_reports_tensor_details -q
```

```text
E   ValueError: cannot resize a fixed tensor
1 failed in 0.27s
```

The implementation was then extended minimally to raise a chained
`ModelContractError` containing the input name, shape, and dtype.

## GREEN Evidence

Focused Task 3 suite after the final production change:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q
```

```text
...................                                                      [100%]
19 passed in 0.59s
```

Required Task 3 plus original regression suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py tests/test_ml.py -q
```

```text
.................................                                        [100%]
33 passed in 0.51s
```

All tests currently present under `ml/tests`, including Tasks 1 and 2:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

```text
...........................................                              [100%]
43 passed in 0.66s
```

Every command exited with status 0 on GREEN runs.

## Requirement Review

- `ModelContractError` is the explicit error type for class-map and tensor
  incompatibilities.
- Interpreter injection bypasses only model-file loading; production calls
  without injection still require the model path and use `_load_interpreter()`.
- Production runtime imports remain lazy and unchanged.
- Class maps with 520 or 522 rows are both rejected; exactly 521 rows are
  accepted.
- The interpreter must expose one float32 waveform input.
- A dynamic shape signature or any actual shape other than `[15600]` triggers
  `resize_tensor_input(index, [15600])` before `allocate_tensors()`.
- Input details are fetched again after allocation and validated against the
  resolved dtype and shape.
- Required outputs are selected by final dimensions 521 and 1,024; singleton
  and multi-frame leading dimensions are accepted.
- Missing-output errors include each observed tensor's name, shape, dtype, and
  index.
- Input errors likewise include the observed input name, shape, dtype, and
  index, including when the interpreter refuses resizing.
- `infer()` converts accepted inputs to float32, then validates rank, exact
  sample count, and finiteness before `set_tensor()` or `invoke()`.
- Score frames and embedding frames are averaged to `(521,)` and `(1024,)`.
- `top(scores, k=5)` is byte-for-byte unchanged and has focused ranking
  coverage.

## Self-Review

- Followed strict RED-GREEN TDD: all primary contract tests were observed
  failing before production was edited, and the resize-exception behavior had
  its own observed RED before implementation.
- The fake asserts that allocation precedes output-detail access and that
  resize precedes allocation. It also records tensor set and invocation calls.
- NumPy arrays are deliberately used for `shape` and `shape_signature`, matching
  real TFLite detail structures.
- Both `[frames, width]` and `[1, width]` output layouts are covered, and the
  numerical mean is asserted rather than only checking output shapes.
- Error tests assert actionable tensor metadata, not merely the exception type.
- The change is confined to the Task 3 pipeline, test, and report files. No
  network access, microphone access, real-model loading, deletion, Git action,
  or out-of-scope edit occurred.

## Concerns and Follow-Up

- No Task 3 blocker remains.
- This task proves the contract with deterministic interpreter doubles. Loading
  the downloaded full YAMNet artifact and checking real finite outputs remains
  the explicitly separate Task 8 integration smoke test.
- Task 6 will extend the same pipeline test module for bounded microphone queues
  and stop-event behavior; those lifecycle changes were intentionally not
  pulled into this task.
