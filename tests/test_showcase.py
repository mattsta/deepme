"""Demo reels.

A reel is validation that leaves assets behind, so the parts that matter are:
every clip is actually attempted, results are reproducible from the manifest,
and one failure does not discard the clips that worked.
"""

from pathlib import Path

import pytest

from deepme import showcase


@pytest.fixture
def reel(tmp_path, monkeypatch):
    monkeypatch.setattr(showcase, "SHOWCASE_DIR", tmp_path)
    (tmp_path / "p.toml").write_text(
        '[[clip]]\nname = "a"\ncaption = "text to video"\nprompt = "cats"\n'
        "seconds = 5\nseed = 7\n\n"
        '[[clip]]\nname = "b"\nprompt = "harbour"\n'
        '[clip.params]\n"target.aspect_ratio" = "9:16"\n'
    )
    return showcase.load("p")


def test_a_reel_is_read_with_its_captions_and_parameters(reel):
    assert [c.name for c in reel] == ["a", "b"]
    assert reel[0].caption == "text to video"
    assert reel[0].seconds == 5 and reel[0].seed == 7
    assert reel[1].params == {"target.aspect_ratio": "9:16"}


def test_a_missing_reel_says_where_to_put_one(tmp_path, monkeypatch):
    monkeypatch.setattr(showcase, "SHOWCASE_DIR", tmp_path)
    with pytest.raises(SystemExit, match="no showcase reel"):
        showcase.load("absent")


def test_dotted_parameters_become_nested_request_objects(reel):
    assert showcase._expand(reel[1].params) == {"target": {"aspect_ratio": "9:16"}}


class _Spec:
    profile = "p"
    repo = "Org/Repo"


def _result(name, ok=True, path=None):
    clip = showcase.Clip(name=name, caption="c", prompt="p")
    return showcase.ClipResult(clip, ok, path, 12.0, "job1", None if ok else "boom")


def test_the_manifest_records_how_each_clip_was_made(tmp_path):
    results = [_result("a", path=tmp_path / "a.mp4")]
    path = showcase.write_manifest(tmp_path, _Spec(), results)  # type: ignore[arg-type]
    import json

    data = json.loads(path.read_text())
    assert data["profile"] == "p"
    assert data["clips"][0]["file"] == "a.mp4"
    assert data["clips"][0]["took_s"] == 12.0


def test_a_failed_clip_is_recorded_without_discarding_the_others(tmp_path):
    import json

    results = [_result("a", path=tmp_path / "a.mp4"), _result("b", ok=False)]
    path = showcase.write_manifest(tmp_path, _Spec(), results)  # type: ignore[arg-type]
    clips = json.loads(path.read_text())["clips"]
    assert clips[0]["ok"] and not clips[1]["ok"]
    assert clips[1]["error"] == "boom"


def test_every_clip_is_attempted_and_results_keep_reel_order(tmp_path, monkeypatch, reel):
    seen = []

    def fake(spec, clip, out_dir, timeout):
        seen.append(clip.name)
        return _result(clip.name, path=Path(out_dir) / f"{clip.name}.mp4")

    monkeypatch.setattr(showcase, "generate_one", fake)
    results = showcase.run(_Spec(), reel, out_dir=tmp_path)  # type: ignore[arg-type]
    assert sorted(seen) == ["a", "b"]
    assert [r.clip.name for r in results] == ["a", "b"]


def test_clips_run_concurrently_not_one_after_another(tmp_path, monkeypatch, reel):
    """Serial generation would take minutes per clip and prove nothing about
    replica packing."""
    import threading

    barrier = threading.Barrier(len(reel), timeout=5)

    def fake(spec, clip, out_dir, timeout):
        barrier.wait()  # deadlocks unless both run at once
        return _result(clip.name)

    monkeypatch.setattr(showcase, "generate_one", fake)
    assert len(showcase.run(_Spec(), reel, out_dir=tmp_path)) == 2  # type: ignore[arg-type]
