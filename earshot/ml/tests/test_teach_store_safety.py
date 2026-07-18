import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import earshot_ml.core as core
from earshot_ml.core import TeachStore, TeachStoreError


def _vector(index=0):
    value = np.zeros(1024, dtype=np.float32)
    value[index] = 1.0
    return value


def _write_store(path, *, names=None, vectors=None):
    values = {}
    if names is not None:
        values["names"] = names
    if vectors is not None:
        values["vectors"] = vectors
    np.savez(path, **values)


@pytest.mark.parametrize("missing", ["names", "vectors"])
def test_load_rejects_missing_required_keys(tmp_path, missing):
    path = tmp_path / "taught.npz"
    values = {
        "names": np.array(["bell"]),
        "vectors": _vector()[None, :],
    }
    values.pop(missing)
    np.savez(path, **values)

    with pytest.raises(TeachStoreError, match=missing):
        TeachStore(path)


def test_load_rejects_mismatched_names_and_vectors(tmp_path):
    path = tmp_path / "taught.npz"
    _write_store(
        path,
        names=np.array(["bell", "knock"]),
        vectors=_vector()[None, :],
    )

    with pytest.raises(TeachStoreError, match="count"):
        TeachStore(path)


@pytest.mark.parametrize(
    ("names", "vectors", "message"),
    [
        (np.array([["bell"]]), _vector()[None, :], "one-dimensional"),
        (np.array(["bell"]), _vector(), "two-dimensional"),
        (np.array(["bell"]), np.zeros((1, 12), np.float32), "1024"),
        (np.array(["bell"]), np.full((1, 1024), np.nan), "finite"),
    ],
)
def test_load_rejects_incompatible_array_layouts(
    tmp_path, names, vectors, message
):
    path = tmp_path / "taught.npz"
    _write_store(path, names=names, vectors=vectors)

    with pytest.raises(TeachStoreError, match=message):
        TeachStore(path)


def test_load_wraps_unsupported_object_arrays(tmp_path):
    path = tmp_path / "taught.npz"
    _write_store(
        path,
        names=np.array([object()], dtype=object),
        vectors=_vector()[None, :],
    )

    with pytest.raises(TeachStoreError, match="names") as exc_info:
        TeachStore(path)

    assert exc_info.value.__cause__ is not None


@pytest.mark.parametrize(
    "embedding",
    [np.zeros(3), np.full(1024, np.nan), ["not", "numeric"]],
)
def test_add_rejects_invalid_embeddings_without_mutating_store(embedding):
    store = TeachStore()

    with pytest.raises(ValueError):
        store.add("bell", embedding)

    assert store.learned() == []
    assert store.match(_vector()) is None


def test_add_failure_does_not_desynchronize_names_and_vectors(monkeypatch):
    store = TeachStore()

    def fail_vstack(values):
        raise MemoryError("simulated allocation failure")

    monkeypatch.setattr(core.np, "vstack", fail_vstack)

    with pytest.raises(MemoryError, match="simulated allocation failure"):
        store.add("bell", _vector())

    assert store.learned() == []


def test_save_atomically_replaces_store_and_removes_part(tmp_path):
    path = tmp_path / "state" / "taught.npz"
    store = TeachStore(path)
    store.add("bell", _vector())

    store.save()

    assert not path.with_name(path.name + ".part").exists()
    loaded = TeachStore(path)
    assert loaded.learned() == [{"name": "bell", "clips": 1}]
    assert loaded.match(_vector()) == ("bell", pytest.approx(1.0))


def test_save_failure_preserves_existing_store_and_cleans_part(
    tmp_path, monkeypatch
):
    path = tmp_path / "taught.npz"
    original = TeachStore(path)
    original.add("known-good", _vector())
    original.save()
    before = path.read_bytes()

    replacement = TeachStore(path)
    replacement.add("new", _vector(1))

    def fail_savez(*args, **kwargs):
        args[0].write(b"partial")
        raise OSError("simulated write failure")

    monkeypatch.setattr(core.np, "savez", fail_savez)

    with pytest.raises(
        TeachStoreError, match="simulated write failure"
    ) as exc_info:
        replacement.save()

    assert isinstance(exc_info.value.__cause__, OSError)

    assert path.read_bytes() == before
    assert not path.with_name(path.name + ".part").exists()


def test_replace_failure_preserves_existing_store_and_cleans_part(
    tmp_path, monkeypatch
):
    path = tmp_path / "taught.npz"
    original = TeachStore(path)
    original.add("known-good", _vector())
    original.save()
    before = path.read_bytes()

    original.add("new", _vector(1))

    def fail_replace(source, destination):
        raise PermissionError("simulated replace failure")

    monkeypatch.setattr(core.os, "replace", fail_replace)

    with pytest.raises(
        TeachStoreError, match="simulated replace failure"
    ) as exc_info:
        original.save()

    assert isinstance(exc_info.value.__cause__, PermissionError)

    assert path.read_bytes() == before
    assert not path.with_name(path.name + ".part").exists()


def test_match_observes_consistent_state_while_add_is_in_progress(monkeypatch):
    store = TeachStore()
    entered_vstack = threading.Event()
    release_vstack = threading.Event()
    real_vstack = core.np.vstack

    def paused_vstack(values):
        entered_vstack.set()
        assert release_vstack.wait(timeout=2)
        return real_vstack(values)

    monkeypatch.setattr(core.np, "vstack", paused_vstack)

    with ThreadPoolExecutor(max_workers=2) as pool:
        add_future = pool.submit(store.add, "bell", _vector())
        assert entered_vstack.wait(timeout=2)
        match_future = pool.submit(store.match, _vector())
        time.sleep(0.02)
        release_vstack.set()

        add_future.result(timeout=2)
        assert match_future.result(timeout=2) == (
            "bell",
            pytest.approx(1.0),
        )


def test_transaction_hides_transient_state_and_rolls_back_failed_save(
        tmp_path, monkeypatch):
    path = tmp_path / "taught.npz"
    store = TeachStore(path)
    store.add("known-good", _vector())
    store.save()
    before = path.read_bytes()

    entered_save = threading.Event()
    release_save = threading.Event()
    learned_started = threading.Event()
    match_started = threading.Event()
    failure = OSError("simulated transactional save failure")

    def paused_failing_savez(*_args, **_kwargs):
        entered_save.set()
        assert release_save.wait(timeout=2)
        raise failure

    def update_and_save():
        with store.transaction():
            store.add("transient", _vector(1))
            store.save()

    def observe_learned():
        learned_started.set()
        return store.learned()

    def observe_match():
        match_started.set()
        return store.match(_vector(1))

    monkeypatch.setattr(core.np, "savez", paused_failing_savez)

    with ThreadPoolExecutor(max_workers=3) as pool:
        update_future = pool.submit(update_and_save)
        assert entered_save.wait(timeout=2)
        learned_future = pool.submit(observe_learned)
        match_future = pool.submit(observe_match)
        assert learned_started.wait(timeout=2)
        assert match_started.wait(timeout=2)
        time.sleep(0.02)

        assert not learned_future.done()
        assert not match_future.done()
        release_save.set()

        with pytest.raises(
            TeachStoreError,
            match="simulated transactional save failure",
        ) as exc_info:
            update_future.result(timeout=2)
        assert exc_info.value.__cause__ is failure
        assert learned_future.result(timeout=2) == [
            {"name": "known-good", "clips": 1}
        ]
        assert match_future.result(timeout=2) is None

    assert path.read_bytes() == before
    assert not path.with_name(path.name + ".part").exists()
