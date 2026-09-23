"""Vendored patch bookkeeping.

A fork is only maintainable if an engine upgrade that moves the patched code
is DETECTED. The dangerous outcome is not "patch failed" — it is "patch
applied to code that changed meaning", so staleness detection is the part
that matters most here.
"""

import pytest

from deepme import patches


def manifest(tmp_path, body):
    d = tmp_path / "sglang"
    d.mkdir()
    (d / "manifest.toml").write_text(body)
    (d / "001-x.patch").write_text("--- a/x\n+++ b/x\n")
    return tmp_path


ONE = """
[[patch]]
file = "001-x.patch"
description = "share the weight store"
upstream_version = "0.5.17"
targets = ["sglang/mm/offload.py"]
sha256 = {"sglang/mm/offload.py" = "abc123"}
"""


@pytest.fixture
def declared(tmp_path, monkeypatch):
    monkeypatch.setattr(patches, "PATCH_ROOT", manifest(tmp_path, ONE))
    return patches.load("sglang")[0]


def test_a_patch_carries_the_version_it_was_written_against(declared):
    assert declared.upstream_version == "0.5.17"
    assert declared.targets == ("sglang/mm/offload.py",)


def test_no_manifest_means_no_patches(tmp_path, monkeypatch):
    monkeypatch.setattr(patches, "PATCH_ROOT", tmp_path)
    assert patches.load("sglang") == []


def _wire(monkeypatch, *, shas, reverse_ok, forward_ok):
    monkeypatch.setattr(patches, "_remote_shas", lambda e, t: shas)
    monkeypatch.setattr(
        patches,
        "_dry_run",
        lambda p, *, reverse: reverse_ok if reverse else forward_ok,
    )


def test_an_applied_patch_is_detected_from_the_file_not_a_state_file(declared, monkeypatch):
    """Reverse-applying cleanly proves our change is present on disk."""
    _wire(
        monkeypatch, shas={"sglang/mm/offload.py": "different"}, reverse_ok=True, forward_ok=False
    )
    assert patches.inspect(declared).state == patches.APPLIED


def test_pristine_upstream_reports_clean(declared, monkeypatch):
    _wire(monkeypatch, shas={"sglang/mm/offload.py": "abc123"}, reverse_ok=False, forward_ok=True)
    assert patches.inspect(declared).state == patches.CLEAN


def test_an_upgrade_that_moves_the_code_is_stale_not_clean(declared, monkeypatch):
    """The whole point: never apply a diff to code that changed under us."""
    _wire(monkeypatch, shas={"sglang/mm/offload.py": "CHANGED"}, reverse_ok=False, forward_ok=True)
    st = patches.inspect(declared)
    assert st.state == patches.STALE
    assert "re-base" in st.detail


def test_a_patch_that_simply_does_not_apply_is_stale(declared, monkeypatch):
    _wire(monkeypatch, shas={"sglang/mm/offload.py": "abc123"}, reverse_ok=False, forward_ok=False)
    assert patches.inspect(declared).state == patches.STALE


def test_a_missing_target_is_reported_as_missing(declared, monkeypatch):
    _wire(monkeypatch, shas={}, reverse_ok=False, forward_ok=False)
    assert patches.inspect(declared).state == patches.MISSING


def test_apply_skips_already_applied_patches(declared, monkeypatch):
    monkeypatch.setattr(
        patches,
        "status",
        lambda e: [patches.PatchState(declared, patches.APPLIED)],
    )
    monkeypatch.setattr(patches, "_push", lambda p: pytest.fail("should not push"))
    assert "already applied" in patches.apply("sglang")[0]


def test_apply_refuses_a_stale_patch_unless_forced(declared, monkeypatch):
    monkeypatch.setattr(
        patches,
        "status",
        lambda e: [patches.PatchState(declared, patches.STALE, "moved")],
    )
    monkeypatch.setattr(patches, "_push", lambda p: pytest.fail("must not apply"))
    assert patches.apply("sglang")[0].startswith("[!]")


def test_state_ok_is_true_only_for_clean_and_applied(declared):
    ok = {
        s: patches.PatchState(declared, s).ok
        for s in (patches.CLEAN, patches.APPLIED, patches.STALE, patches.MISSING)
    }
    assert ok == {
        patches.CLEAN: True,
        patches.APPLIED: True,
        patches.STALE: False,
        patches.MISSING: False,
    }
