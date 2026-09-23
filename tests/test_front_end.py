"""Which endpoint a request is sent to.

A profile's port addresses replica 0 specifically. Defaulting there sends
every CLI generation to one replica while the rest of the fleet idles, and
queues work behind generations that can run for minutes.
"""

import pytest
from test_subfolder_routing import spec as make_spec

from deepme import cli


@pytest.fixture
def spec():
    return make_spec(port=8030, profile="h3-video")


def _balancer(monkeypatch, backends):
    """Each model fleet has its own balancer, so stats are per-profile."""
    from deepme import balancerctl

    def stats(profile):
        if backends is None:
            raise OSError("balancer not running")
        return {"backends": backends}

    monkeypatch.setattr(balancerctl, "stats", stats)
    monkeypatch.setattr(balancerctl, "port_for", lambda profile: 9030)


@pytest.fixture(autouse=True)
def _no_gateway(monkeypatch):
    """Default every test to "no gateway running".

    The gateway is preferred when it is up, so without this each test would
    depend on whether something happens to be listening on 8080 on the
    machine running the suite.
    """
    from deepme import gatewayctl

    def down():
        raise OSError("gateway not running")

    monkeypatch.setattr(gatewayctl, "routes", down)


def _gateway(monkeypatch, models):
    from deepme import gatewayctl

    monkeypatch.setattr(
        gatewayctl,
        "routes",
        lambda: {"routes": [{"model": m} for m in models]},
    )


def test_a_running_balancer_is_used_by_default(monkeypatch, spec):
    _balancer(monkeypatch, [{"url": "http://127.0.0.1:8030"}])
    target, via = cli.front_end(spec)
    assert target.port == 9030
    assert via == "balancer"


def test_direct_bypasses_the_balancer_on_request(monkeypatch, spec):
    _balancer(monkeypatch, [{"url": "http://127.0.0.1:8030"}])
    target, via = cli.front_end(spec, direct=True)
    assert target.port == 8030
    assert "directly" in via


def test_no_balancer_falls_back_to_the_profile_port(monkeypatch, spec):
    """A single-replica deployment must work with no extra setup."""
    _balancer(monkeypatch, None)
    target, via = cli.front_end(spec)
    assert target.port == 8030
    assert "no balancer" in via


def test_a_balancer_with_no_backends_is_not_used(monkeypatch, spec):
    _balancer(monkeypatch, [])
    assert cli.front_end(spec)[0].port == 8030


def test_everything_else_about_the_request_is_unchanged(monkeypatch, spec):
    _balancer(monkeypatch, [{"url": "x"}])
    target, _ = cli.front_end(spec)
    assert target.served_name == spec.served_name
    assert target.profile == spec.profile


@pytest.fixture
def h3():
    """A spec whose public name is real, since routing is by that name."""
    return make_spec(
        port=8030, profile="h3-video", repo="MiniMaxAI/MiniMax-H3", served_name="h3-video"
    )


def test_the_gateway_is_preferred_when_it_serves_this_model(monkeypatch, h3):
    """Highest layer wins: it hides which checkpoint serves which task."""
    _balancer(monkeypatch, [{"url": "http://127.0.0.1:8030"}])
    _gateway(monkeypatch, ["minimax-h3"])
    target, via = cli.front_end(h3)
    assert target.port == 8080
    assert via == "gateway"


def test_the_gateway_rewrites_the_model_to_its_public_name(monkeypatch, h3):
    """served_name goes into the request body, and the gateway routes on it.

    The profile name ("h3-video") is exactly what the gateway exists to hide,
    and it does not appear in the routing table — sending it would be refused
    by the layer meant to make deployment invisible.
    """
    _balancer(monkeypatch, [{"url": "http://127.0.0.1:8030"}])
    _gateway(monkeypatch, ["minimax-h3"])
    target, _ = cli.front_end(h3)
    assert target.served_name == "minimax-h3"


def test_a_model_the_gateway_does_not_serve_falls_to_the_balancer(monkeypatch, spec):
    """Routing there would be a 400 from a layer that never had the route."""
    _balancer(monkeypatch, [{"url": "http://127.0.0.1:8030"}])
    _gateway(monkeypatch, ["some-other-model"])
    target, via = cli.front_end(spec)
    assert target.port == 9030
    assert via == "balancer"


def test_direct_bypasses_the_gateway_too(monkeypatch, h3):
    """With a gateway that WOULD serve this model, so the bypass is real."""
    _balancer(monkeypatch, [{"url": "http://127.0.0.1:8030"}])
    _gateway(monkeypatch, ["minimax-h3"])
    target, via = cli.front_end(h3, direct=True)
    assert target.port == 8030
    assert "directly" in via
