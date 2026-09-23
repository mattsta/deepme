"""Identifying a weight component by its content rather than its path.

MiniMax-H3's FL2VA and Ref2VA partitions ship a byte-identical 62.1GiB text
encoder and differ only in the transformer. Keyed by path they occupy two
copies of everything; keyed by content they share what is genuinely the same.
The transformers have IDENTICAL shapes and differ only in values, so
structure alone cannot tell them apart.
"""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="fingerprinting needs real tensors")

_spec = importlib.util.spec_from_file_location(
    "deepme_hooks",
    Path(__file__).resolve().parents[1] / "src" / "deepme" / "hooks" / "deepme_hooks.py",
)
hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks)


def model(values, shape=(4, 4)):
    m = torch.nn.Module()
    m.w = torch.nn.Parameter(torch.full(shape, float(values)))
    m.b = torch.nn.Parameter(torch.full((shape[0],), float(values) / 2))
    return m


def test_identical_weights_share_an_identity():
    """The same encoder in two partitions must map one physical copy."""
    assert hooks.component_fingerprint(model(1.0)) == hooks.component_fingerprint(model(1.0))


def test_same_shapes_different_values_are_distinguished():
    """The two transformers differ ONLY in values — the case that matters."""
    assert hooks.component_fingerprint(model(1.0)) != hooks.component_fingerprint(model(2.0))


def test_different_shapes_are_distinguished():
    assert hooks.component_fingerprint(model(1.0)) != hooks.component_fingerprint(
        model(1.0, shape=(8, 2))
    )


def test_parameter_names_are_part_of_identity():
    a = model(1.0)
    b = model(1.0)
    b.renamed = b.w
    del b.w
    assert hooks.component_fingerprint(a) != hooks.component_fingerprint(b)


def test_an_unfingerprintable_module_says_so_rather_than_colliding():
    class Broken:
        def named_parameters(self):
            raise RuntimeError("no")

    assert hooks.component_fingerprint(Broken()) == "unknown"


def test_an_empty_module_is_stable():
    empty = torch.nn.Module()
    assert hooks.component_fingerprint(empty) == hooks.component_fingerprint(torch.nn.Module())


def test_fingerprinting_is_bounded_for_large_models():
    """A real DiT has thousands of parameters; sampling must stay O(1)-ish."""
    big = torch.nn.Module()
    for i in range(500):
        setattr(big, f"p{i}", torch.nn.Parameter(torch.zeros(4)))
    assert len(hooks.component_fingerprint(big)) == 32
