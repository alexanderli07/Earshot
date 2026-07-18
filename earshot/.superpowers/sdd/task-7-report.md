# Task 7 report: package CLI, verified download, and lazy storage commands

## RED evidence

- Command: `ml\.venv\Scripts\python.exe -m pytest tests/test_cli.py -q`
  (run from `ml` before creating `earshot_ml/cli.py`).
- Result: **collection failed with 1 error in 0.24s**.
- Expected cause: `ImportError: cannot import name 'cli' from 'earshot_ml'`
  because the package CLI module did not exist yet.

## Implementation

- Added `earshot_ml.cli.main(argv=None)`, package command handlers, parser
  construction, and the package module main guard while preserving the six
  commands and all existing options.
- Routed `download` through `config.MODEL_ARTIFACT`,
  `config.CLASS_MAP_ARTIFACT`, and `download_artifact()`. It reports downloaded
  versus cached artifacts, then constructs `YamNet` with the configured model
  and class-map paths to validate the installed pair.
- Converted expected artifact, checksum, model-contract, missing-file, and CLI
  validation failures into concise nonzero CLI exits. Artifact failures include
  a retry/source/connectivity action and do not emit a normal-use traceback.
- Made `sounds` and `forget` use `TeachStore(config.TAUGHT_STORE_PATH)` directly.
  `forget` calls the store's atomic `save()` after mutation.
- Added trimmed, case-insensitive reserved-label validation before teach creates
  an engine, opens a clip, prompts, records, or requires model artifacts. Core
  validation remains unchanged as a second boundary.
- Replaced top-level `ml/cli.py` with the exact compatibility import and main
  guard. Existing `pyproject.toml` console metadata already targets
  `earshot_ml.cli:main`, so no metadata edit was needed.

## GREEN evidence

- Focused CLI suite:
  `ml\.venv\Scripts\python.exe -m pytest tests/test_cli.py -q`
  -> **8 passed in 0.30s**.
- Full ML suite:
  `ml\.venv\Scripts\python.exe -m pytest -q`
  -> **92 passed in 0.88s**.

## Help compatibility evidence

Each command exited zero and listed
`download, top5, run, teach, sounds, forget`:

- `ml\.venv\Scripts\python.exe cli.py --help`
- `ml\.venv\Scripts\python.exe -m earshot_ml.cli --help`
- `ml\.venv\Scripts\earshot.exe --help`

## Self-review

- Storage tests use a real temporary NPZ store, missing model/class-map paths,
  constructor sentinels for both `EarshotML` and `YamNet`, and an import guard
  for all supported TFLite backends. They prove listing and atomic forgetting
  do not cross the model/backend boundary.
- Artifact tests use only local `file:` URIs. The checksum failure test proves a
  nonzero concise error and `.part` cleanup; the success test proves both files
  exist before `YamNet` receives the explicitly configured paths and checks both
  downloaded and cached status output.
- The reserved-name test uses whitespace plus different casing and makes engine,
  model, prompt, and record paths fail if reached, proving validation order.
- Help and parser tests exercise `main()` with explicit argv. The wrapper test
  locks the requested five-line compatibility shape, while the pre-existing
  packaging test continues to lock the console entry point metadata.
- Scope review found changes only in `ml/earshot_ml/cli.py`, `ml/cli.py`,
  `ml/tests/test_cli.py`, and this report.

## Concerns

- No known code-level concerns. No live network, microphone, real model, Git,
  or deletion operation was used.

## Review Fix

The independent review found that a corrupt taught store escaped `main()` as a
raw `TeachStoreError`. A new CLI test first reproduced that failure, then the
package CLI was updated to treat `TeachStoreError` as an expected concise,
nonzero error. Fresh verification after the fix produced `9 passed` for the CLI
suite and `93 passed` for the full suite.
