"""Health watching: what counts as a change worth recording.

The failure this guards against is a service that dies while nobody is
watching — and its subtler cousin, a service that crash-loops fast enough to
answer a probe between restarts.
"""

import pytest

from deepme import health


def uh(profile="m", active="active", api_ok=True, detail="HTTP 200", restarts=0):
    return health.UnitHealth(
        profile=profile,
        unit=f"deepme-llm-{profile}",
        active=active,
        api_ok=api_ok,
        detail=detail,
        restarts=restarts,
    )


def test_a_crash_loop_is_a_change_even_while_probes_succeed():
    """Restarts are part of the state, so a loop cannot hide behind an up probe."""
    before = {"m": uh(restarts=3).state}
    changes = health.diff(before, [uh(restarts=7)])
    assert len(changes) == 1
    assert changes[0][0].restarts == 7


def test_steady_state_records_nothing():
    assert health.diff({"m": uh().state}, [uh()]) == []


def test_going_down_is_reported_with_the_previous_state():
    before = {"m": uh().state}
    ((h, was),) = health.diff(before, [uh(active="failed", api_ok=False, detail="ConnectError")])
    assert not h.healthy
    assert was == "active/up/0"


def test_first_sample_is_a_baseline_not_an_outage():
    """An empty history means we have no evidence anything changed."""
    assert health.diff({}, [uh(active="failed", api_ok=False)]) == []


@pytest.mark.parametrize(
    "active,api_ok,healthy",
    [
        ("active", True, True),
        ("active", False, False),
        ("failed", False, False),
        ("absent", False, False),
    ],
)
def test_healthy_requires_both_running_and_answering(active, api_ok, healthy):
    assert uh(active=active, api_ok=api_ok).healthy is healthy


def test_props_parse_one_batched_systemctl_call(monkeypatch):
    class R:
        stdout = (
            "Id=deepme-llm-a.service\nActiveState=active\nNRestarts=0\n"
            "Id=deepme-llm-b.service\nActiveState=failed\nNRestarts=12\n"
        )

    calls = []
    monkeypatch.setattr(health.remote, "run", lambda cmd, **kw: calls.append(cmd) or R())
    props = health._unit_props(["deepme-llm-a", "deepme-llm-b"])
    assert props == {"deepme-llm-a": ("active", 0), "deepme-llm-b": ("failed", 12)}
    assert len(calls) == 1, "one round trip regardless of unit count"


def test_no_units_means_no_remote_call(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("must not touch the server for an empty fleet")

    monkeypatch.setattr(health.remote, "run", boom)
    assert health._unit_props([]) == {}


def test_watch_records_only_transitions(monkeypatch):
    samples = [[uh()], [uh()], [uh(active="failed", api_ok=False)]]
    monkeypatch.setattr(health, "snapshot", lambda *a, **kw: samples.pop(0))
    monkeypatch.setattr(health.time, "sleep", lambda _s: None)
    recorded = []
    monkeypatch.setattr(
        health.runs,
        "record",
        lambda action, spec, **kw: recorded.append(kw["detail"]) or "id",
    )
    health.watch(0.0, cycles=3)
    assert len(recorded) == 1
    assert recorded[0]["was"] == "active/up/0"
    assert recorded[0]["api_ok"] is False


def test_inactive_units_are_not_probed(monkeypatch):
    """Probing a stopped unit burns a timeout every cycle for no information."""
    spec = type("S", (), {"unit": "deepme-llm-m", "profile": "m"})()
    monkeypatch.setattr(health, "_unit_props", lambda units: {"deepme-llm-m": ("inactive", 2)})
    monkeypatch.setattr(
        health.serve,
        "health",
        lambda s: (_ for _ in ()).throw(AssertionError("probed an inactive unit")),
    )
    (result,) = health.snapshot([spec])  # type: ignore[list-item]
    assert result.detail == "inactive"
    assert not result.healthy
