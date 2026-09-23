"""Building a generation request body.

MiniMax-H3's validator (request_validation.py in the engine) accepts only
{task, prompt, conditions, target, flow_shift, audio_flow_shift, seed} and
DISCARDS everything else, with target's own required projection. So the body
we assemble has to be right by construction, not by trial.
"""

from typing import Any, ClassVar

import pytest

from deepme import media


def test_dotted_override_reaches_a_nested_field():
    body = {}
    media.set_path(body, "target.short_edge", 768)
    assert body == {"target": {"short_edge": 768}}


def test_dotted_override_keeps_its_siblings():
    """Overriding one target field must not drop aspect_ratio or duration."""
    base = {"target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5}}
    over = {}
    media.set_path(over, "target.aspect_ratio", "9:16")
    merged = media.deep_merge(base, over)
    assert merged["target"] == {
        "short_edge": 768,
        "aspect_ratio": "9:16",
        "duration_seconds": 5,
    }


def test_merge_replaces_scalars_and_lists_wholesale():
    assert media.deep_merge({"a": 1, "c": [1, 2]}, {"a": 2, "c": [3]}) == {"a": 2, "c": [3]}


def test_set_path_replaces_a_non_object_on_the_way_down():
    body = {"target": "nonsense"}
    media.set_path(body, "target.short_edge", 768)
    assert body == {"target": {"short_edge": 768}}


class _Spec:
    """Minimal stand-in: generate_video only reads these."""

    profile = "h3"
    served_name = "h3-video"
    api_base = "http://h"
    gen_defaults: ClassVar[dict[str, Any]] = {
        "task": "t2va",
        "conditions": [],
        "target": {"short_edge": 768, "aspect_ratio": "16:9", "duration_seconds": 5},
    }


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def fake_submit(spec, body, timeout):
        seen.update(body)
        return {"id": "vid_1", "status": "failed", "error": {"stop": "here"}}

    monkeypatch.setattr(media, "_submit", fake_submit)
    return seen


def test_profile_defaults_are_sent(captured):
    media.generate_video(_Spec(), "a cat")  # type: ignore[arg-type]
    assert captured["task"] == "t2va"
    assert captured["conditions"] == []
    assert captured["prompt"] == "a cat"


def test_seconds_lands_where_the_model_keeps_duration(captured):
    """--seconds must reach target.duration_seconds, not just a flat field."""
    media.generate_video(_Spec(), "a cat", seconds=4)  # type: ignore[arg-type]
    assert captured["target"]["duration_seconds"] == 4
    assert captured["target"]["short_edge"] == 768


def test_params_win_over_everything(captured):
    media.generate_video(  # type: ignore[arg-type]
        _Spec(), "a cat", seconds=4, params={"target": {"duration_seconds": 9}}
    )
    assert captured["target"]["duration_seconds"] == 9
    assert captured["target"]["aspect_ratio"] == "16:9"


def test_a_failed_job_reports_the_error_without_a_file(captured):
    result = media.generate_video(_Spec(), "a cat")  # type: ignore[arg-type]
    assert not result.ok
    assert result.path is None
    assert result.error == {"stop": "here"}


class _Floored(_Spec):
    min_steps = 2


def test_a_step_count_below_the_models_floor_fails_before_submitting(captured):
    """steps=1 makes a 1-point sigma schedule; the server blames the schedule."""
    with pytest.raises(SystemExit, match="below this model's floor of 2"):
        media.generate_video(_Floored(), "a cat", steps=1)  # type: ignore[arg-type]
    assert not captured, "must not reach the server"


def test_the_floor_itself_is_allowed(captured):
    media.generate_video(_Floored(), "a cat", steps=2)  # type: ignore[arg-type]
    assert captured["num_inference_steps"] == 2


def test_no_declared_floor_means_no_client_side_step_check(captured):
    media.generate_video(_Spec(), "a cat", steps=1)  # type: ignore[arg-type]
    assert captured["num_inference_steps"] == 1
