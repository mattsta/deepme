"""The shared weight store: one physical copy of the weights for N replicas.

The two ways this could go wrong are both silent and both catastrophic:
a second process reading a half-written buffer, or two different tensors
colliding on one key. Everything else can fall back to private memory.
"""

import os

import pytest

from deepme import sharedstore


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(sharedstore, "STORE_DIR", str(tmp_path))
    monkeypatch.setenv(sharedstore.ENABLED_ENV, "1")
    return tmp_path


LAYOUT = [("blocks.0.w", 0, 1024), ("blocks.0.b", 1024, 16)]
KEY_ARGS = {
    "model_id": "/models/MiniMax-H3",
    "component": "transformer",
    "layer_idx": 3,
    "dtype": "BF16",
    "total_numel": 1040,
    "tp_rank": 0,
    "layout": LAYOUT,
}


def key(**over):
    return sharedstore.buffer_key(**{**KEY_ARGS, **over})


def test_identical_deployments_agree_on_a_key():
    """Two replicas of the same model must land on the same buffer."""
    assert key() == key()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_rank", 1),  # a different shard is different bytes
        ("layer_idx", 4),
        ("dtype", "F32"),
        ("total_numel", 2048),
        ("component", "text_encoder"),
        ("model_id", "/models/Other"),
        ("layout", [("blocks.0.w", 0, 512)]),
    ],
)
def test_anything_that_changes_the_bytes_changes_the_key(field, value):
    assert key(**{field: value}) != key()


def _fake(nbytes=64):
    made = {"private": 0, "mapped": []}

    def map_shared(path, n):
        # Stand in for a file-backed tensor: create the file at full size.
        with open(path, "wb") as f:
            f.truncate(n)
        made["mapped"].append(path)
        return f"tensor@{path}"

    def allocate_private():
        made["private"] += 1
        return "private-tensor"

    return made, map_shared, allocate_private


def test_the_first_process_is_told_to_fill_the_buffer():
    made, m, p = _fake()
    got = sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    assert got.shared and not got.prefilled
    assert made["private"] == 0


def test_a_later_process_maps_it_and_skips_the_copy():
    _made, m, p = _fake()
    sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    sharedstore.publish("k1")
    second = sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    assert second.shared and second.prefilled


def test_an_unpublished_buffer_is_never_treated_as_complete():
    """A crash mid-fill must not leave others reading uninitialised weights."""
    _made, m, p = _fake()
    sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    again = sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    assert not again.prefilled


def test_a_wrong_sized_buffer_is_rebuilt_not_reused():
    _made, m, p = _fake()
    sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    sharedstore.publish("k1")
    grown = sharedstore.acquire("k1", 128, map_shared=m, allocate_private=p)
    assert not grown.prefilled


def test_disabled_by_default_falls_back_to_upstream_behaviour(monkeypatch):
    monkeypatch.delenv(sharedstore.ENABLED_ENV, raising=False)
    made, m, p = _fake()
    got = sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    assert not got.shared and made["private"] == 1
    assert not made["mapped"], "must not touch the store when disabled"


def test_a_broken_store_never_stops_a_model_loading():
    made, _m, p = _fake()

    def explode(path, n):
        raise OSError("tmpfs full")

    got = sharedstore.acquire("k1", 64, map_shared=explode, allocate_private=p)
    assert not got.shared and got.prefilled is False
    assert made["private"] == 1


def test_usage_reports_what_the_store_holds():
    _made, m, p = _fake()
    sharedstore.acquire("k1", 4096, map_shared=m, allocate_private=p)
    assert sharedstore.usage()["buffers"] == 1
    assert sharedstore.usage()["bytes"] == 4096


def test_clear_removes_segments_nothing_is_mapping(monkeypatch, store):
    _made, m, p = _fake()
    sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    sharedstore.publish("k1")
    monkeypatch.setattr(sharedstore, "_in_use", lambda path: False)
    assert sharedstore.clear() >= 1
    assert sharedstore.usage()["buffers"] == 0


def test_clear_leaves_segments_that_are_still_mapped(monkeypatch):
    _made, m, p = _fake()
    sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p)
    monkeypatch.setattr(sharedstore, "_in_use", lambda path: True)
    assert sharedstore.clear() == 0
    assert sharedstore.usage()["buffers"] == 1


def test_the_build_is_serialised_so_only_one_process_fills(store):
    """Without the lock two processes could fill the same buffer at once."""
    order = []

    class _Guard:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            order.append(("enter", self.path))
            return self

        def __exit__(self, *exc):
            order.append(("exit", self.path))
            return False

    _made, m, p = _fake()
    sharedstore.acquire("k1", 64, map_shared=m, allocate_private=p, lock=_Guard)
    assert [o[0] for o in order] == ["enter", "exit"]
    assert order[0][1].endswith("k1.lock")


def test_paths_are_namespaced_per_key(store):
    data, ready, lock = sharedstore.paths("abc")
    assert os.path.basename(data) == "abc.bin"
    assert os.path.basename(ready) == "abc.ready"
    assert os.path.basename(lock) == "abc.lock"


def test_orphaned_segments_are_reported_not_silently_held(monkeypatch, store):
    """A key-scheme change strands every previous key: the files stay on
    tmpfs holding RAM and nothing will ever ask for them again. Observed:
    the store grew 123.5GiB -> 202GiB across such a change."""
    _made, m, p = _fake()
    sharedstore.acquire("old_key", 4096, map_shared=m, allocate_private=p)
    monkeypatch.setattr(sharedstore, "_in_use", lambda path: False)
    usage = sharedstore.usage()
    assert usage["orphans"] >= 1
    assert usage["orphan_bytes"] >= 4096


def test_segments_in_use_are_not_reported_as_orphans(monkeypatch, store):
    _made, m, p = _fake()
    sharedstore.acquire("live", 4096, map_shared=m, allocate_private=p)
    monkeypatch.setattr(sharedstore, "_in_use", lambda path: True)
    assert sharedstore.usage()["orphans"] == 0
