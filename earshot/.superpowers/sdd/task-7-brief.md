### Task 7: Package CLI, verified download, and lazy storage commands

**Files:**
- Create: `ml/earshot_ml/cli.py`
- Replace: `ml/cli.py` with a compatibility wrapper
- Create: `ml/tests/test_cli.py`

**Interfaces:**
- Produces: `earshot_ml.cli.main(argv=None)` and package console entry point.
- Preserves: all existing command names and options.

- [ ] **Step 1: Write failing CLI tests**

Test parser help, storage-only `sounds` and `forget` with no model files and no interpreter, `teach` reserved-name errors, checksum error reporting, and `download` calling `YamNet` after both artifacts are installed. Use temporary `EARSHOT_MODEL_DIR` paths and local artifact URLs; do not use network access.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_cli.py -q`

Expected: import failure because `earshot_ml.cli` does not exist.

- [ ] **Step 3: Move and harden CLI implementation**

Move command handlers and parser creation into `earshot_ml/cli.py`, accept optional argv, use artifact helpers for `download`, validate the actual model after download, and instantiate `TeachStore` directly for `sounds` and `forget`. Keep top-level `cli.py` as:

```python
from earshot_ml.cli import main


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Verify CLI compatibility**

Run: `python -m pytest tests/test_cli.py -q`

Expected: all tests pass.

Run: `python cli.py --help`

Expected: help lists all six commands.

Run: `earshot --help`

Expected: the installed console entry point lists the same commands.

---

