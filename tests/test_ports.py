"""The declared port registry.

Ports used to be derived (replica N at base + N*stride, balancer at port +
1000). Derivation fails silently because nothing compares the results: a
diffusion server binds its port AND the next one, so h3-video replica 0 held
8030-8031 while h3-ref was declared at 8031. That collision was live and
would have surfaced as a replica mysteriously relocating.
"""

import pytest

from deepme import ports


def write(tmp_path, body):
    p = tmp_path / "ports.toml"
    p.write_text(body)
    return p


CLEAN = """
[services]
gateway = 8080

[profiles.a]
span = 2
replicas = [8100, 8110]
balancer = 9100

[profiles.b]
span = 2
replicas = [8200]
balancer = 9200
"""


def test_the_real_registry_has_no_conflicts():
    """The shipped file must be valid — this is the regression guard."""
    assert ports.conflicts(ports.load()) == []


def test_ports_are_looked_up_not_computed(tmp_path):
    reg = ports.load(write(tmp_path, CLEAN))
    assert reg.replica_port("a", 0) == 8100
    assert reg.replica_port("a", 1) == 8110
    assert reg.balancer_port("a") == 9100
    assert reg.service_port("gateway") == 8080


def test_a_span_makes_neighbouring_ports_conflict(tmp_path):
    """The exact bug: two allocations collide WITHOUT sharing a number."""
    reg = ports.load(
        write(
            tmp_path,
            """
[profiles.a]
span = 2
replicas = [8030]

[profiles.b]
span = 2
replicas = [8031]
""",
        )
    )
    found = ports.conflicts(reg)
    assert len(found) == 1
    assert "8030-8031" in found[0] and "8031-8032" in found[0]
    assert "binds the NEXT port" in found[0]


def test_an_exact_duplicate_is_a_conflict(tmp_path):
    reg = ports.load(
        write(
            tmp_path,
            """
[services]
one = 8080
two = 8080
""",
        )
    )
    assert len(ports.conflicts(reg)) == 1


def test_adjacent_single_port_services_do_not_conflict(tmp_path):
    reg = ports.load(
        write(
            tmp_path,
            """
[services]
one = 8080
two = 8081
""",
        )
    )
    assert ports.conflicts(reg) == []


def test_a_balancer_colliding_with_a_replica_is_caught(tmp_path):
    reg = ports.load(
        write(
            tmp_path,
            """
[profiles.a]
replicas = [9100]
balancer = 9100
""",
        )
    )
    assert len(ports.conflicts(reg)) == 1


def test_asking_for_an_undeclared_profile_says_where_to_add_it(tmp_path):
    reg = ports.load(write(tmp_path, CLEAN))
    with pytest.raises(SystemExit, match="no replica ports declared"):
        reg.replica_ports("absent")


def test_running_out_of_declared_replica_ports_is_explicit(tmp_path):
    """Better than silently reusing one and colliding at runtime."""
    reg = ports.load(write(tmp_path, CLEAN))
    with pytest.raises(SystemExit, match="declares 1 replica port"):
        reg.replica_port("b", 3)


def test_a_missing_registry_is_reported(tmp_path):
    with pytest.raises(SystemExit, match="missing port registry"):
        ports.load(tmp_path / "nope.toml")


def test_allocations_are_listed_in_port_order_for_review(tmp_path):
    reg = ports.load(write(tmp_path, CLEAN))
    got = [a.port for a in reg.allocations()]
    assert got == sorted(got)
