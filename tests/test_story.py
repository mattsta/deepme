"""Chained generation: each segment continues from the one before.

Two things make this different from a batch: it is strictly ordered (a
segment needs a frame that does not exist until its predecessor finishes),
and the join has a seam — the first frame of a continuation is a duplicate of
the frame it was seeded from.
"""

from pathlib import Path

import pytest

from deepme import story as story_mod


def _story(tmp_path, statuses):
    segs = [
        story_mod.Segment(
            index=i, prompt=f"p{i}", status=s, file=f"seg{i:03d}.mp4" if s == "done" else None
        )
        for i, s in enumerate(statuses)
    ]
    return story_mod.Story(name="s", profile="h3", out_dir=str(tmp_path), segments=segs)


def test_state_round_trips_so_a_long_story_resumes(tmp_path):
    st = _story(tmp_path, ["done", "pending"])
    st.segments[0].took_s = 850.0
    st.save()
    again = story_mod.Story.load("s", tmp_path)
    assert again is not None
    assert again.segments[0].file == "seg000.mp4"
    assert again.segments[0].took_s == 850.0
    assert again.segments[1].status == "pending"


def test_only_finished_segments_are_joined(tmp_path):
    st = _story(tmp_path, ["done", "failed", "done"])
    assert [p.name for p in st.done_clips()] == ["seg000.mp4", "seg002.mp4"]


def test_joining_nothing_is_an_explicit_error(tmp_path):
    with pytest.raises(SystemExit, match="no finished segments"):
        story_mod.concat([], tmp_path / "out.mp4")


def _fake_ffmpeg(monkeypatch, calls):
    monkeypatch.setattr(story_mod, "_ffmpeg", lambda: "ffmpeg")

    def fake_run(cmd, **kw):
        calls.append(cmd)
        # Emulate ffmpeg producing its output file (last argument).
        Path(cmd[-1]).write_bytes(b"x")

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(story_mod.subprocess, "run", fake_run)


def test_the_seam_frame_is_dropped_from_continuations(tmp_path, monkeypatch):
    """The first frame of a continuation IS the frame it was seeded from, so
    joining naively shows it twice."""
    calls = []
    _fake_ffmpeg(monkeypatch, calls)
    clips = []
    for i in range(3):
        c = tmp_path / f"seg{i:03d}.mp4"
        c.write_bytes(b"x")
        clips.append(c)
    story_mod.concat(clips, tmp_path / "out.mp4")
    trims = [c for c in calls if "-ss" in c]
    assert len(trims) == 2, "first segment keeps its opening frame; others trim"


def test_the_seam_trim_can_be_disabled(tmp_path, monkeypatch):
    calls = []
    _fake_ffmpeg(monkeypatch, calls)
    clips = []
    for i in range(2):
        c = tmp_path / f"seg{i:03d}.mp4"
        c.write_bytes(b"x")
        clips.append(c)
    story_mod.concat(clips, tmp_path / "out.mp4", drop_seam_frame=False)
    assert not [c for c in calls if "-ss" in c]


def test_segments_are_copied_not_re_encoded(tmp_path, monkeypatch):
    """Every segment comes from the same model at the same geometry, so a
    stream copy is lossless and fast."""
    calls = []
    _fake_ffmpeg(monkeypatch, calls)
    clip = tmp_path / "seg000.mp4"
    clip.write_bytes(b"x")
    story_mod.concat([clip], tmp_path / "out.mp4")
    assert any("-c" in c and "copy" in c for c in calls)


def test_the_last_frame_is_taken_by_seeking_from_the_end(tmp_path, monkeypatch):
    """Decoding a whole clip to reach its final frame would be wasteful."""
    calls = []
    _fake_ffmpeg(monkeypatch, calls)
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    story_mod.last_frame(src, tmp_path / "f.png")
    assert "-sseof" in calls[0]


def test_a_missing_ffmpeg_is_reported_not_crashed(monkeypatch, tmp_path):
    monkeypatch.setattr(story_mod.shutil, "which", lambda _n: None)
    with pytest.raises(SystemExit, match="ffmpeg is required"):
        story_mod.last_frame(tmp_path / "a.mp4", tmp_path / "f.png")


def test_the_first_segment_is_plain_text_to_video(tmp_path, monkeypatch):
    """Only continuations carry a keyframe condition."""
    seen = []
    st = _story(tmp_path, ["pending"])

    def fake(spec, prompt, **kw):
        seen.append(kw.get("params"))
        out = Path(kw["out_dir"]) / "x.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 1.0, 1.0, None

        return R()

    monkeypatch.setattr(story_mod.media, "generate_video", fake)
    story_mod.run(object(), st)
    assert "conditions" not in (seen[0] or {})


def test_a_failed_segment_stops_the_chain(tmp_path, monkeypatch):
    """Continuing would need a frame from a clip that was never produced."""
    st = _story(tmp_path, ["pending", "pending"])

    def fake(spec, prompt, **kw):
        class R:
            ok, path, total_s, peak_memory_mb = False, None, 1.0, 1.0
            error = "boom"

        return R()

    monkeypatch.setattr(story_mod.media, "generate_video", fake)
    with pytest.raises(SystemExit, match="chain stops here"):
        story_mod.run(object(), st)
    assert st.segments[1].status == "pending", "later segments untouched"


def test_finished_segments_are_not_regenerated(tmp_path, monkeypatch):
    st = _story(tmp_path, ["done"])
    (tmp_path / "seg000.mp4").write_bytes(b"x")

    def boom(*a, **kw):
        raise AssertionError("must not regenerate a finished segment")

    monkeypatch.setattr(story_mod.media, "generate_video", boom)
    story_mod.run(object(), st)
