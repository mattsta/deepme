"""Dispatching a request to the fleet that can serve it.

One model's capabilities can be split across deployments — H3's FL2VA serves
t2va/fl2va while Ref2VA serves ref2va — so the model name alone does not say
where a request goes. These refusal messages are part of the contract the
service layer builds against, so they are asserted, not just the happy path.
"""

import pytest

from deepme.gateway.route import RoutingError, build

PRODUCTION = [
    {
        "model": "h3-video",
        "profile": "h3-video",
        "tasks": ["t2va", "fl2va"],
        "url": "http://127.0.0.1:8090",
    },
    {"model": "h3-video", "profile": "h3-ref", "tasks": ["ref2va"], "url": "http://127.0.0.1:8091"},
]
BASE_ONLY = PRODUCTION[:1]


def test_a_task_selects_between_fleets_backing_one_model():
    t = build(PRODUCTION)
    assert t.resolve("h3-video", "t2va").profile == "h3-video"
    assert t.resolve("h3-video", "fl2va").profile == "h3-video"
    assert t.resolve("h3-video", "ref2va").profile == "h3-ref"


def test_the_caller_never_needs_to_know_the_port():
    t = build(PRODUCTION)
    assert t.resolve("h3-video", "ref2va").url.endswith("8091")


def test_a_single_fleet_does_not_require_a_task():
    """Demanding one when there is no ambiguity is pedantry."""
    assert build(BASE_ONLY).resolve("h3-video", None).profile == "h3-video"


def test_an_ambiguous_model_requires_a_task_and_says_which():
    t = build(PRODUCTION)
    with pytest.raises(RoutingError, match="task is required"):
        t.resolve("h3-video", None)
    try:
        t.resolve("h3-video", None)
    except RoutingError as exc:
        assert "ref2va" in str(exc) and "t2va" in str(exc)


def test_an_undeployed_task_is_an_operations_problem_with_the_fix():
    """ref2va is real but its fleet is not running — tell the operator how."""
    t = build(BASE_ONLY)
    with pytest.raises(RoutingError, match="not running") as exc:
        t.resolve("h3-video", "ref2va")
    assert "fleet apply" in str(exc.value)


def test_a_misspelt_task_is_a_client_bug_and_reads_differently():
    t = build(PRODUCTION)
    with pytest.raises(RoutingError, match="unknown task") as exc:
        t.resolve("h3-video", "t2vaa")
    assert "not running" not in str(exc.value)


def test_an_unknown_model_lists_what_is_available():
    t = build(PRODUCTION)
    with pytest.raises(RoutingError, match="h3-video"):
        t.resolve("some-other-model", "t2va")


def test_an_empty_table_says_nothing_is_running():
    t = build([])
    with pytest.raises(RoutingError, match="no fleet is running"):
        t.resolve("h3-video", "t2va")


def test_a_job_is_routed_back_to_the_fleet_that_minted_it():
    """Ids travel through two hops now and mean nothing to the other fleet."""
    t = build(PRODUCTION)
    t.remember("vid_1", "http://127.0.0.1:8091")
    assert t.owner("vid_1") == "http://127.0.0.1:8091"


def test_an_unknown_job_has_no_owner_so_the_caller_can_probe():
    assert build(PRODUCTION).owner("vid_absent") is None


def test_job_tracking_is_bounded():
    t = build(PRODUCTION)
    t.max_jobs = 3
    for i in range(5):
        t.remember(f"vid_{i}", "http://x")
    assert t.snapshot()["tracked_jobs"] == 3


def test_tasks_are_reported_per_model_for_capability_discovery():
    assert build(PRODUCTION).tasks_for("h3-video") == {"t2va", "fl2va", "ref2va"}


def test_the_table_is_inspectable_for_observability():
    snap = build(PRODUCTION).snapshot()
    assert len(snap["routes"]) == 2
    assert {r["profile"] for r in snap["routes"]} == {"h3-video", "h3-ref"}
