"""The derived service catalog.

Three facts lived in three places — the port registry (declared), the
allocator (running) and profiles (capable) — and each consumer stitched them
together itself. The catalog derives one answer, and deliberately keeps
"declared" and "running" apart: mistaking configuration for reality is the
expensive failure here.
"""

import pytest

from deepme import discovery
from deepme import resources as res


@pytest.fixture(autouse=True)
def no_service_units(monkeypatch):
    """Default to no service units, so tests never reach the real server.

    Services and balancers hold no resource claim, so their liveness comes
    from their own systemd units rather than from current_claims().
    """
    monkeypatch.setattr(res, "active_units", set)


@pytest.fixture
def nothing_running(monkeypatch):
    monkeypatch.setattr(res, "current_claims", lambda: [])


@pytest.fixture
def service_units(monkeypatch):
    """The service layer up: gateway plus both fleet balancers."""
    monkeypatch.setattr(
        res,
        "active_units",
        lambda: {"deepme-gateway", "deepme-balancer-h3-video", "deepme-balancer-h3-ref"},
    )


@pytest.fixture
def production_running(monkeypatch):
    """3x h3-video + 1x h3-ref, on their declared ports."""
    monkeypatch.setattr(
        res,
        "current_claims",
        lambda: [
            res.Claim(unit="deepme-llm-h3-video", profile="h3-video", gpus=(0, 1), port=8100),
            res.Claim(unit="deepme-llm-h3-video-r1", profile="h3-video", gpus=(2, 3), port=8110),
            res.Claim(unit="deepme-llm-h3-video-r2", profile="h3-video", gpus=(4, 5), port=8120),
            res.Claim(unit="deepme-llm-h3-ref", profile="h3-ref", gpus=(6, 7), port=8200),
        ],
    )


def test_declared_is_not_confused_with_running(nothing_running):
    """Configuration is not reality — the distinction has to survive."""
    eps = discovery.endpoints()
    assert eps, "the registry declares endpoints even when nothing runs"
    assert all(e.declared for e in eps)
    assert not any(e.running for e in eps)
    assert {e.state for e in eps} == {"declared"}


def test_running_endpoints_are_marked_from_live_claims(production_running):
    running = {e.name for e in discovery.endpoints() if e.running}
    assert "h3-video/replica-0" in running
    assert "h3-ref/replica-0" in running
    assert "h3-video/replica-3" not in running  # declared, not deployed


def test_a_balancer_is_live_only_when_its_own_unit_is(production_running, service_units):
    """Its liveness is its own, never inferred from the replicas.

    Inferring it from the fleet reported a DEAD balancer as running whenever
    the replicas behind it were healthy — hiding precisely the failure a
    caller would hit, since every request goes through it.
    """
    by_name = {e.name: e for e in discovery.endpoints()}
    assert by_name["h3-video/balancer"].running
    assert by_name["dsv4-gguf/balancer"].running is False


def test_a_dead_balancer_over_a_healthy_fleet_is_reported_dead(production_running):
    """The replicas are all up and the balancer is not — say so."""
    by_name = {e.name: e for e in discovery.endpoints()}
    assert by_name["h3-video/replica-0"].running
    assert by_name["h3-video/balancer"].running is False


def test_the_gateway_is_seen_through_its_unit(production_running, service_units):
    """It holds no port claim, so nothing else can reveal that it is serving."""
    by_name = {e.name: e for e in discovery.endpoints()}
    assert by_name["gateway"].running
    assert by_name["gateway"].state == "running"


def test_partitions_merge_into_one_caller_facing_model(production_running):
    """A caller asks for the model and a task, not for a checkpoint."""
    entries = discovery.models()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.model == "minimax-h3"
    assert entry.tasks == ("fl2va", "ref2va", "t2va")
    assert set(entry.profiles) == {"h3-video", "h3-ref"}


def test_replica_counts_are_summed_across_partitions(production_running):
    assert discovery.models()[0].replicas == 4


def test_capabilities_come_from_the_profile_that_enforces_them(production_running):
    """A limit documented twice is a limit that will disagree with itself."""
    entry = discovery.models()[0]
    assert entry.limits.get("min_steps") == 2
    assert entry.generate_defaults.get("task")


def test_nothing_deployed_means_no_models_offered(nothing_running):
    assert discovery.models() == []
    assert discovery.catalog()["summary"]["models"] == 0


def test_a_model_is_unavailable_with_no_replicas(nothing_running):
    assert all(not m.available for m in discovery.models())


def test_the_catalog_reports_both_counts_so_drift_is_visible(production_running):
    summary = discovery.catalog()["summary"]
    assert summary["running_endpoints"] < summary["declared_endpoints"]
    assert summary["running_endpoints"] > 0


def test_endpoints_carry_addressable_urls(production_running):
    urls = {e.name: e.url for e in discovery.endpoints()}
    assert urls["h3-video/replica-0"].endswith(":8100")
    assert urls["gateway"].endswith(":8080")
