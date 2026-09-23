"""Fleet composition: which models share the machine, and in what proportion.

GPUs are the hard constraint, so this split is a pricing decision expressed
as config. The failure that matters is two members being handed the same
GPUs, because the allocator only knows what is already RUNNING and a fleet is
placed before any of it starts.
"""

import re

import pytest
from test_subfolder_routing import spec as make_spec

from deepme import fleet as fleet_mod
from deepme import resources as res
from deepme import serve

CAP = res.Capacity(
    gpus=tuple(range(8)),
    numa={0: tuple(range(0, 64)), 1: tuple(range(64, 128))},
    ram_gib=503.0,
)


@pytest.fixture
def empty_machine(monkeypatch):
    monkeypatch.setattr(res, "probe_capacity", lambda **kw: CAP)
    monkeypatch.setattr(res, "current_claims", lambda: [])


@pytest.fixture
def fleets(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_mod, "FLEETS_DIR", tmp_path)
    (tmp_path / "production.toml").write_text(
        '[[member]]\nprofile = "h3-video"\nreplicas = 3\n\n'
        '[[member]]\nprofile = "h3-ref"\nreplicas = 1\n'
    )
    (tmp_path / "max-base.toml").write_text('[[member]]\nprofile = "h3-video"\nreplicas = 4\n')
    return tmp_path


def test_a_mix_is_read_in_declaration_order(fleets):
    f = fleet_mod.load("production")
    assert [(m.profile, m.replicas) for m in f.members] == [("h3-video", 3), ("h3-ref", 1)]
    assert f.total_replicas == 4


def test_an_unknown_fleet_lists_what_exists(fleets):
    with pytest.raises(SystemExit, match="max-base"):
        fleet_mod.load("nope")


def test_a_fleet_with_no_members_is_rejected(fleets, tmp_path):
    (tmp_path / "empty.toml").write_text("# nothing\n")
    with pytest.raises(SystemExit, match=re.escape("no [[member]] entries")):
        fleet_mod.load("empty")


def test_zero_replicas_is_rejected(fleets, tmp_path):
    (tmp_path / "bad.toml").write_text('[[member]]\nprofile = "h3-video"\nreplicas = 0\n')
    with pytest.raises(SystemExit, match="0 replicas"):
        fleet_mod.load("bad")


def _stub_profiles(monkeypatch):
    def load(name, **over):
        port = {"h3-video": 8030, "h3-ref": 8031}.get(name, 8030)
        return make_spec(profile=name, gpus=(6, 7), port=port)

    monkeypatch.setattr(fleet_mod.profiles, "load", load)


def test_members_never_share_gpus(fleets, empty_machine, monkeypatch):
    """The allocator only sees RUNNING claims, so without reservation every
    member would be handed the same free GPUs."""
    _stub_profiles(monkeypatch)
    specs = fleet_mod.plan(fleet_mod.load("production"))
    seen: set[int] = set()
    for s in specs:
        assert not (set(s.gpus) & seen), f"{s.unit} overlaps an earlier member"
        seen |= set(s.gpus)
    assert len(seen) == 8


def test_the_whole_machine_is_used_by_a_full_mix(fleets, empty_machine, monkeypatch):
    _stub_profiles(monkeypatch)
    specs = fleet_mod.plan(fleet_mod.load("production"))
    assert len(specs) == 4
    assert sum(len(s.gpus) for s in specs) == 8


def test_each_profile_keeps_its_own_replica_numbering(fleets, empty_machine, monkeypatch):
    _stub_profiles(monkeypatch)
    specs = fleet_mod.plan(fleet_mod.load("production"))
    video = [s for s in specs if s.profile == "h3-video"]
    ref = [s for s in specs if s.profile == "h3-ref"]
    assert [s.replica for s in video] == [0, 1, 2]
    assert [s.replica for s in ref] == [0]


def test_a_single_member_fleet_still_places(fleets, empty_machine, monkeypatch):
    _stub_profiles(monkeypatch)
    assert len(fleet_mod.plan(fleet_mod.load("max-base"))) == 4


def test_reserved_gpus_are_excluded_from_a_later_placement(empty_machine):
    """The primitive the fleet planner relies on."""
    # Both profiles must exist in the port registry, since ports are declared.
    first = serve.replica_specs(make_spec(gpus=(6, 7), profile="h3-video"), 2)
    second = serve.replica_specs(make_spec(gpus=(6, 7), profile="h3-ref"), 2, reserved=first)
    taken = {g for s in first for g in s.gpus}
    assert all(not (set(s.gpus) & taken) for s in second)


def test_a_mix_that_does_not_fit_says_so(fleets, empty_machine, monkeypatch, tmp_path):
    _stub_profiles(monkeypatch)
    (tmp_path / "toobig.toml").write_text(
        '[[member]]\nprofile = "h3-video"\nreplicas = 3\n\n'
        '[[member]]\nprofile = "h3-ref"\nreplicas = 3\n'
    )
    with pytest.raises(SystemExit, match="GPUs"):
        fleet_mod.plan(fleet_mod.load("toobig"))


def test_a_mix_can_be_planned_while_the_previous_mix_still_runs(monkeypatch, fleets):
    """`fleet plan` is a dry run. Requiring an idle machine would mean tearing
    down the fleet you are only trying to describe."""
    _stub_profiles(monkeypatch)
    monkeypatch.setattr(res, "probe_capacity", lambda **kw: CAP)
    # All 8 GPUs held by the outgoing 4x h3-video fleet.
    monkeypatch.setattr(
        res,
        "current_claims",
        lambda: [
            res.Claim(
                unit=f"deepme-llm-h3-video{'' if i == 0 else f'-r{i}'}",
                profile="h3-video",
                gpus=(i * 2, i * 2 + 1),
                port=8030 + i * 10,
            )
            for i in range(4)
        ],
    )
    specs = fleet_mod.plan(fleet_mod.load("production"))
    assert len(specs) == 4
    assert sum(len(s.gpus) for s in specs) == 8


def test_another_models_gpus_are_still_respected_when_planning(monkeypatch, fleets):
    """Only profiles the fleet will STOP are treated as free."""
    _stub_profiles(monkeypatch)
    monkeypatch.setattr(res, "probe_capacity", lambda **kw: CAP)
    monkeypatch.setattr(
        res,
        "current_claims",
        lambda: [res.Claim(unit="deepme-llm-other", profile="other", gpus=(0, 1, 2, 3), port=8000)],
    )
    with pytest.raises(SystemExit, match="GPUs"):
        fleet_mod.plan(fleet_mod.load("production"))


# --------------------------------------------------------------------------
# The switch delta: what separates the running fleet from a target mix


def _spec(unit, profile, gpus, port):
    """The four fields delta() reads, shaped like a RunSpec."""
    import types

    return types.SimpleNamespace(unit=unit, profile=profile, gpus=gpus, port=port)


V0 = _spec("deepme-llm-h3-video", "h3-video", (0, 1), 8100)
V1 = _spec("deepme-llm-h3-video-r1", "h3-video", (2, 3), 8110)
V2 = _spec("deepme-llm-h3-video-r2", "h3-video", (4, 5), 8120)
V3 = _spec("deepme-llm-h3-video-r3", "h3-video", (6, 7), 8130)
REF = _spec("deepme-llm-h3-ref", "h3-ref", (6, 7), 8200)


def _claim(spec):
    return res.Claim(unit=spec.unit, profile=spec.profile, gpus=tuple(spec.gpus), port=spec.port)


def test_matching_replicas_are_kept_not_restarted():
    """The point of the delta: 3+1 -> 4+0 must not touch the three video
    replicas that are identical in both mixes. Restarting an unchanged
    replica costs ~5 minutes of its revenue for nothing."""
    d = fleet_mod.delta([V0, V1, V2, V3], [_claim(V0), _claim(V1), _claim(V2), _claim(REF)])
    assert {s.unit for s in d.keep} == {V0.unit, V1.unit, V2.unit}
    assert [s.unit for s in d.add] == [V3.unit]
    assert d.remove == (REF.unit,)
    assert d.remove_ports[REF.unit] == 8200


def test_the_reverse_switch_is_the_mirror_delta():
    d = fleet_mod.delta([V0, V1, V2, REF], [_claim(V0), _claim(V1), _claim(V2), _claim(V3)])
    assert {s.unit for s in d.keep} == {V0.unit, V1.unit, V2.unit}
    assert [s.unit for s in d.add] == [REF.unit]
    assert d.remove == (V3.unit,)


def test_a_moved_replica_is_remove_plus_add_never_kept():
    """Same unit name on different GPUs is a different deployment — keeping
    it would leave the unit running somewhere the plan does not expect."""
    moved = _claim(_spec(V2.unit, "h3-video", (6, 7), 8120))
    d = fleet_mod.delta([V0, V1, V2], [_claim(V0), _claim(V1), moved])
    assert {s.unit for s in d.keep} == {V0.unit, V1.unit}
    assert [s.unit for s in d.add] == [V2.unit]
    assert d.remove == (V2.unit,)


def test_an_unrelated_model_on_other_gpus_is_left_alone():
    """A coexisting deployment the mix does not mention and whose GPUs the
    mix does not need is not this switch's business."""
    other = res.Claim(unit="deepme-llm-lfm-cpu", profile="lfm-cpu", gpus=(), port=8010)
    d = fleet_mod.delta([V0, V1], [_claim(V0), _claim(V1), other])
    assert d.is_noop


def test_a_conflicting_model_on_needed_gpus_is_removed():
    """...but one squatting on GPUs the target places onto must go."""
    squatter = res.Claim(unit="deepme-llm-dsv4", profile="dsv4", gpus=(6, 7), port=8300)
    d = fleet_mod.delta([V3], [squatter])
    assert d.remove == ("deepme-llm-dsv4",)


def test_an_identical_fleet_is_a_noop():
    d = fleet_mod.delta([V0, V1, V2, REF], [_claim(V0), _claim(V1), _claim(V2), _claim(REF)])
    assert d.is_noop
    assert len(d.keep) == 4
