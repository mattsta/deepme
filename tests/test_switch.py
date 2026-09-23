"""The convergence engine, under fakes — the platform's headline claim.

"A resize converges from any intermediate state" lived untested inside a
215-line CLI function. Now every side effect goes through SwitchOps, so the
drain rules, the reconcile pass, orphan cleanup and the gateway decision run
here in milliseconds against the exact intermediate states an interrupted
switch leaves behind.
"""

from types import SimpleNamespace

import pytest

from deepme import fleet as fleet_mod
from deepme import switch as switch_mod


def _spec(unit, profile, gpus, port):
    return SimpleNamespace(unit=unit, profile=profile, gpus=gpus, port=port)


V0 = _spec("deepme-llm-h3-video", "h3-video", (0, 1), 8100)
V1 = _spec("deepme-llm-h3-video-r1", "h3-video", (2, 3), 8110)
# Squats on V1's TARGET GPUs, so any plan containing V1 must displace it —
# a claim on unrelated GPUs is (correctly) not this switch's business.
REF = _spec("deepme-llm-h3-ref", "h3-ref", (2, 3), 8200)


def _claim(spec):
    from deepme import resources as res

    return res.Claim(unit=spec.unit, profile=spec.profile, gpus=tuple(spec.gpus), port=spec.port)


class FakeOps(switch_mod.SwitchOps):
    """Records every action; state is mutable per scenario."""

    def __init__(
        self, *, claims, memberships, queues=None, units=None, drain_fails=False, drained=True
    ):
        self.log: list[tuple] = []
        self.memberships = memberships  # profile -> list of backend dicts
        self._queues = queues or {}
        self._drain_fails = drain_fails
        self._drained = drained

        def rec(name):
            def _f(*a, **k):
                self.log.append((name, *a))
                return {}

            return _f

        super().__init__(
            current_claims=lambda: claims,
            active_units=lambda: set(units or []),
            queue=lambda p: self._queues.get(p, {"waiting": 0, "next": []}),
            stats=self._stats,
            drain=self._drain,
            undrain=rec("undrain"),
            wait_drained=lambda *a, **k: self._drained,
            detach=rec("detach"),
            attach=lambda p, u: self.log.append(("attach", p, u)) or {"backends": [u]},
            start_balancer=rec("start_balancer"),
            stop_balancer=rec("stop_balancer"),
            stop_units=rec("stop_units"),
            start_replica=rec("start_replica"),
            wait_healthy=lambda s, timeout_s=0: (True, "ok"),
            discover_routes=lambda: [
                {"model": "m", "profile": "h3-video", "tasks": ["t2va"], "url": "u"}
            ],
            gateway_routes=lambda: {"routes": []},
            repoint=lambda routes: self.log.append(("repoint",)) or True,
            gateway_start=rec("gateway_start"),
        )

    def _stats(self, profile):
        if profile not in self.memberships:
            raise ConnectionError("no balancer")
        return {"backends": self.memberships[profile]}

    def _drain(self, profile, url):
        self.log.append(("drain", profile, url))
        if self._drain_fails:
            raise ConnectionError("balancer down")
        return {}

    def did(self, name):
        return [entry for entry in self.log if entry[0] == name]


@pytest.fixture
def production(monkeypatch, tmp_path):
    """A declared mix + a deterministic plan, isolated from real profiles."""
    monkeypatch.setattr(
        fleet_mod, "load", lambda name: SimpleNamespace(name=name, members=(), total_replicas=2)
    )
    monkeypatch.setattr(fleet_mod, "plan", lambda f, releasing_extra=None: [V0, V1])
    return


def _run(ops, name="mix", force=False):
    audit = {"fleet": name, "abandoned_tickets": []}
    switch_mod.execute_switch(name, timeout=1, drain_timeout=1, force=force, audit=audit, ops=ops)
    return audit


def test_a_noop_delta_still_reconciles_membership(production):
    """THE convergence property: an arrival whose attach failed last run is
    a 'keep' this run — a switch that only attaches d.add would strand it
    out of rotation forever."""
    ops = FakeOps(
        claims=[_claim(V0), _claim(V1)],
        memberships={
            "h3-video": [{"url": "http://127.0.0.1:8100", "draining": False}]
        },  # r1 missing!
    )
    _run(ops)
    assert ("attach", "h3-video", "http://127.0.0.1:8110") in ops.log


def test_a_leftover_drain_is_unwound(production):
    """An aborted switch left a kept replica draining: visibly 'up',
    silently starved. The reconcile pass re-arms it."""
    ops = FakeOps(
        claims=[_claim(V0), _claim(V1)],
        memberships={
            "h3-video": [
                {"url": "http://127.0.0.1:8100", "draining": False},
                {"url": "http://127.0.0.1:8110", "draining": True},
            ]
        },
    )
    _run(ops)
    assert ("undrain", "h3-video", "http://127.0.0.1:8110") in ops.log


def test_a_stale_backend_is_detached(production):
    """A stopped replica whose detach was interrupted stays in rotation,
    eating one failed submission per pick — until reconcile removes it."""
    ops = FakeOps(
        claims=[_claim(V0), _claim(V1)],
        memberships={
            "h3-video": [
                {"url": "http://127.0.0.1:8100", "draining": False},
                {"url": "http://127.0.0.1:8110", "draining": False},
                {"url": "http://127.0.0.1:8130", "draining": False},  # stale
            ]
        },
    )
    _run(ops)
    assert ("detach", "h3-video", "http://127.0.0.1:8130") in ops.log


def test_an_orphan_balancer_is_stopped(production):
    """Derived from what RUNS, not this run's delta — an interrupted earlier
    switch's leftover balancer is invisible to the delta forever."""
    ops = FakeOps(
        claims=[_claim(V0), _claim(V1)],
        memberships={
            "h3-video": [
                {"url": "http://127.0.0.1:8100", "draining": False},
                {"url": "http://127.0.0.1:8110", "draining": False},
            ]
        },
        units={"deepme-balancer-h3-ref", "deepme-balancer-h3-video"},
    )
    _run(ops)
    assert ("stop_balancer", "h3-ref") in ops.log
    assert ("stop_balancer", "h3-video") not in ops.log


def test_a_down_balancer_for_a_target_profile_is_started_fresh(production):
    ops = FakeOps(claims=[_claim(V0), _claim(V1)], memberships={})
    _run(ops)
    assert ops.did("start_balancer"), "nothing to preserve when it is down"


def test_a_drain_that_cannot_be_issued_aborts_without_force(production):
    """Stopping an undrained replica kills whatever it is generating —
    'the balancer was down' is a reason to stop, not to guess."""
    ops = FakeOps(
        claims=[_claim(V0), _claim(REF)],
        memberships={"h3-video": []},
        drain_fails=True,
    )
    with pytest.raises(SystemExit, match="cannot drain"):
        _run(ops)
    assert not ops.did("stop_units"), "nothing was stopped"


def test_force_stops_without_drain_and_names_abandoned_tickets(production):
    ops = FakeOps(
        claims=[_claim(V0), _claim(REF)],
        memberships={
            "h3-video": [
                {"url": "http://127.0.0.1:8100", "draining": False},
                {"url": "http://127.0.0.1:8110", "draining": False},
            ]
        },
        queues={"h3-ref": {"waiting": 2, "next": [{"ticket": "q_a"}, {"ticket": "q_b"}]}},
        drain_fails=True,
    )
    audit = _run(ops, force=True)
    assert ops.did("stop_units")
    assert audit["abandoned_tickets"] == ["q_a", "q_b"], "the casualties are on the record"
    assert ("stop_balancer", "h3-ref") in ops.log


def test_leaving_profile_with_queue_refuses_without_force(production):
    ops = FakeOps(
        claims=[_claim(V0), _claim(REF)],
        memberships={"h3-video": []},
        queues={"h3-ref": {"waiting": 3, "next": []}},
    )
    with pytest.raises(SystemExit, match="no remaining fleet can serve"):
        _run(ops)


def test_the_gateway_is_repointed_in_place_when_supported(production):
    ops = FakeOps(
        claims=[_claim(V0), _claim(V1)],
        memberships={
            "h3-video": [
                {"url": "http://127.0.0.1:8100", "draining": False},
                {"url": "http://127.0.0.1:8110", "draining": False},
            ]
        },
    )
    _run(ops)
    assert ops.did("repoint")
    assert not ops.did("gateway_start"), "no restart when reload works"
