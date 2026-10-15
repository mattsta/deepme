"""Conditioning modes: keyframes vs references.

These are different TASKS backed by different weight partitions, not two
spellings of the same thing. fl2va pins exact frames (first, last, or both);
ref2va takes images/video/audio as guidance only. Getting the condition shape
wrong is a 400 after a round trip, so the shape is built and tested locally.
"""

import base64

import pytest

from deepme import media


def _img(tmp_path, name="a.png"):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return str(p)


def test_a_local_file_is_inlined_because_the_server_cannot_see_your_disk(tmp_path):
    uri = media.as_uri(_img(tmp_path))
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1]) == b"\x89PNG\r\n\x1a\nfake"


@pytest.mark.parametrize(
    "src", ["https://x/y.png", "http://x/y.mp4", "file:///srv/a.png", "data:image/png;base64,AA"]
)
def test_remote_and_inline_uris_pass_through_untouched(src):
    assert media.as_uri(src) == src


def test_a_missing_local_file_fails_before_the_request(tmp_path):
    with pytest.raises(media.RequestInvalid, match="no such file"):
        media.as_uri(str(tmp_path / "nope.png"))


def test_unknown_media_types_are_rejected_with_the_supported_list():
    with pytest.raises(media.RequestInvalid, match="supported extensions"):
        media.media_type_of("notes.txt")


def test_first_and_last_keyframes_get_the_engines_frame_indices(tmp_path):
    conds = media.keyframe_conditions(_img(tmp_path, "a.png"), _img(tmp_path, "b.png"))
    assert [c["frame_index"] for c in conds] == [0, -1]
    assert {c["role"] for c in conds} == {"keyframe"}
    assert {c["type"] for c in conds} == {"image"}


def test_a_single_keyframe_is_a_legal_signature(tmp_path):
    """(0,), (-1,) and (0,-1) are the accepted signatures — no middle frames."""
    assert [c["frame_index"] for c in media.keyframe_conditions(_img(tmp_path), None)] == [0]
    assert [c["frame_index"] for c in media.keyframe_conditions(None, _img(tmp_path))] == [-1]


def test_no_keyframes_means_no_conditions():
    assert media.keyframe_conditions(None, None) == []


def test_a_video_keyframe_is_rejected(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    with pytest.raises(media.RequestInvalid, match="keyframes must be images"):
        media.keyframe_conditions(str(vid), None)


def test_references_carry_no_frame_index(tmp_path):
    """A reference guides the whole clip; pinning it to a frame is a keyframe."""
    conds = media.reference_conditions([_img(tmp_path)])
    assert conds[0]["role"] == "reference"
    assert "frame_index" not in conds[0]


def test_a_video_reference_also_supplies_its_soundtrack(tmp_path):
    vid = tmp_path / "v.mp4"
    vid.write_bytes(b"x")
    assert media.reference_conditions([str(vid)])[0]["type"] == "video_audio"


def test_an_audio_reference_stays_audio(tmp_path):
    aud = tmp_path / "a.wav"
    aud.write_bytes(b"x")
    assert media.reference_conditions([str(aud)])[0]["type"] == "audio"


def test_condition_keys_stay_inside_the_engine_allowlist(tmp_path):
    """The validator rejects any unknown condition field outright."""
    allowed = {"type", "uri", "role", "frame_index", "start_time_seconds"}
    conds = media.keyframe_conditions(_img(tmp_path), None)
    conds += media.reference_conditions([_img(tmp_path, "c.png")])
    for c in conds:
        assert set(c) <= allowed
