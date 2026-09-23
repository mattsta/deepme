"""Laying out N copies of a model across free GPUs.

Replicas are how throughput is bought from engines that serve one request at
a time. They must land on DISJOINT GPUs with distinct ports and unit names,
or they displace each other instead of adding capacity — and replica 0 must
keep the plain unit name so scaling does not orphan an existing deployment.
"""

import dataclasses

import pytest
from test_subfolder_routing import spec as make_spec

from deepme import resources as res
from deepme import serve

CAP = res.Capacity(
    gpus=tuple(range(8)),
    numa={0: tuple(range(64)), 1: tuple(range(64, 128))},
    ram_gib=503.0,
)


@pytest.fixture
def empty_machine(monkeypatch):
    monkeypatch.setattr(res, "probe_capacity", lambda **kw: CAP)
    monkeypatch.setattr(res, "current_claims", list)


def base(**over):
    return make_spec(**{"gpus": (6, 7), "port": 8030, "profile": "h3-video", **over})


def test_replica_zero_keeps_the_plain_unit_name(empty_machine):
    """An existing single deployment must not be orphaned by scaling."""
    specs = serve.replica_specs(base(), 3)
    assert specs[0].unit == "deepme-llm-h3-video"
    assert [s.unit for s in specs[1:]] == [
        "deepme-llm-h3-video-r1",
        "deepme-llm-h3-video-r2",
    ]


def test_replicas_get_disjoint_gpus(empty_machine):
    specs = serve.replica_specs(base(), 3)
    seen: set[int] = set()
    for s in specs:
        assert len(s.gpus) == 2
        assert not (set(s.gpus) & seen), "a GPU was handed out twice"
        seen |= set(s.gpus)


def test_replicas_get_distinct_ports(empty_machine):
    ports = [s.port for s in serve.replica_specs(base(), 4)]
    assert len(set(ports)) == 4


def test_each_replica_is_socket_local(empty_machine):
    """A replica split across sockets pays for every all-reduce."""
    for s in serve.replica_specs(base(), 4):
        assert set(s.gpus) <= {0, 1, 2, 3} or set(s.gpus) <= {4, 5, 6, 7}


def test_asking_for_more_than_fits_says_what_does_fit(empty_machine):
    with pytest.raises(SystemExit, match=r"-n 4"):
        serve.replica_specs(base(), 5)


def test_rescaling_reclaims_this_profiles_own_gpus(monkeypatch):
    """Going 3 -> 2 must reuse its own GPUs, not refuse for lack of space."""
    monkeypatch.setattr(res, "probe_capacity", lambda **kw: CAP)
    monkeypatch.setattr(
        res,
        "current_claims",
        lambda: [
            res.Claim(unit="deepme-llm-h3-video", profile="h3-video", gpus=(0, 1), port=8030),
            res.Claim(unit="deepme-llm-h3-video-r1", profile="h3-video", gpus=(2, 3), port=8031),
        ],
    )
    assert len(serve.replica_specs(base(), 4)) == 4


def test_another_models_gpus_are_never_taken(monkeypatch):
    monkeypatch.setattr(res, "probe_capacity", lambda **kw: CAP)
    monkeypatch.setattr(
        res,
        "current_claims",
        lambda: [res.Claim(unit="deepme-llm-other", profile="other", gpus=(0, 1, 2, 3), port=8000)],
    )
    specs = serve.replica_specs(base(), 2)
    assert all(not (set(s.gpus) & {0, 1, 2, 3}) for s in specs)


def test_a_profile_with_no_gpus_cannot_be_replicated_this_way(empty_machine):
    with pytest.raises(SystemExit, match="no GPUs to replicate"):
        serve.replica_specs(base(gpus=()), 2)


def test_zero_replicas_is_rejected(empty_machine):
    with pytest.raises(SystemExit, match="at least 1"):
        serve.replica_specs(base(), 0)


def test_everything_else_about_the_deployment_is_unchanged(empty_machine):
    """Replicas differ only in identity and placement — same model, same args."""
    original = base()
    for s in serve.replica_specs(original, 2):
        assert s.engine_args == original.engine_args
        assert s.subfolder == original.subfolder
        assert s.weight_gib == original.weight_gib


def test_only_the_first_replica_pays_for_shared_weights(empty_machine):
    """With a shared store the weights exist once; charging every replica the
    full amount is what refuses replicas 3 and 4 when the RAM is right there."""
    specs = serve.replica_specs(base(ram_gib=200.0, extra_replica_ram_gib=24.0), 4)
    # The CAP is unchanged (load peak); the allocator's CHARGE is what drops.
    assert [s.ram_gib for s in specs] == [200.0] * 4
    assert [s.accounted_ram_gib for s in specs] == [200.0, 24.0, 24.0, 24.0]


def test_without_sharing_every_replica_is_charged_in_full(empty_machine):
    specs = serve.replica_specs(base(ram_gib=200.0), 3)
    assert [s.ram_gib for s in specs] == [200.0, 200.0, 200.0]


def test_the_ram_claim_reaches_the_unit(empty_machine, monkeypatch):
    specs = serve.replica_specs(base(ram_gib=200.0, extra_replica_ram_gib=24.0), 2)
    monkeypatch.setattr(serve, "build_command", lambda _s: ["/bin/true"])
    # Both units are capped for the load peak; they differ in what the
    # allocator counts against the machine.
    assert "MemoryMax=200G" in serve.render_unit(specs[1])
    assert "X-DEEPME-RAM=24" in serve.render_unit(specs[1])


def test_gpu_release_wait_allows_for_slow_pinned_teardown():
    """Unregistering tens of GiB of page-locked shared memory takes minutes;
    a short wait aborts a scale-up halfway and leaves one replica running."""
    assert serve.GPU_RELEASE_TIMEOUT_S >= 300


def test_a_sharing_replica_is_not_charged_for_weights_it_only_maps():
    """The offload fit check charges host RAM for the weights. Replica 2+ with
    a shared store maps replica 0's copy, so charging it again refuses
    placements that demonstrably fit."""
    offload = ("--layerwise-offload-components", "dit,text_encoder,vae")
    first = base(
        weight_gib=62.0,
        tp=2,
        engine_args=offload,
        ram_gib=200.0,
        extra_replica_ram_gib=24.0,
        replica=0,
    )
    later = dataclasses.replace(first, replica=1, ram_gib=24.0)
    assert first.fit_check() is None
    assert later.fit_check() is None


def test_without_sharing_a_later_replica_is_still_checked():
    offload = ("--layerwise-offload-components", "dit,text_encoder,vae")
    later = base(weight_gib=62.0, tp=2, engine_args=offload, ram_gib=24.0, replica=1)
    err = later.fit_check()
    assert err is not None and "host RAM" in err


def test_the_gpu_release_wait_does_not_match_sibling_replicas(monkeypatch):
    """Waiting on "deepme-llm-h3-video" must not see a running
    "deepme-llm-h3-video-r1" as the stopped unit's leftover process."""
    captured = {}

    class R:
        stdout = "RELEASED"

    def fake(script, **kw):
        captured["s"] = script
        return R()

    monkeypatch.setattr(serve.remote, "run_script", fake)
    serve._wait_gpus_released(["deepme-llm-h3-video"])
    import re

    assert re.escape("deepme-llm-h3-video.service") in captured["s"]
    # The sibling's name must NOT be what we wait on.
    assert re.escape("deepme-llm-h3-video-r1.service") not in captured["s"]


def test_replicas_get_distinct_rendezvous_ports():
    """Every replica runs its own torch.distributed rendezvous. Sharing the
    default port kills replica 2 with EADDRINUSE before it ever loads."""
    from deepme import engines

    seen = set()
    for i in range(4):
        s = dataclasses.replace(base(), replica=i, port=8030 + i)
        cmd = engines.get("sglang").build_command(s, "key")
        master = cmd[cmd.index("--master-port") + 1]
        sched = cmd[cmd.index("--scheduler-port") + 1]
        assert (master, sched) not in seen
        seen.add((master, sched))
    assert len(seen) == 4


def test_internal_ports_come_from_the_registry_per_profile():
    """Deriving these from the replica index alone collided ACROSS profiles:
    h3-video r0 and h3-ref r0 both wanted scheduler 5644."""
    from deepme import engines
    from deepme import ports as ports_mod

    registry = ports_mod.load()
    cmd = engines.get("sglang").build_command(base(), "key")
    assert cmd[cmd.index("--master-port") + 1] == str(registry.master_port("h3-video", 0))
    assert cmd[cmd.index("--scheduler-port") + 1] == str(registry.scheduler_port("h3-video", 0))
    # And the other profile's replica 0 must differ, which is the whole point.
    assert registry.scheduler_port("h3-ref", 0) != registry.scheduler_port("h3-video", 0)


def test_the_cgroup_cap_covers_the_load_peak_not_the_resting_size(empty_machine, monkeypatch):
    """A sharing replica rests at a few GiB but still materialises every
    component before offloading it. Capping at the resting figure OOM-kills it
    mid-load — measured, at "Loading safetensors checkpoint shards 4/12"."""
    specs = serve.replica_specs(base(ram_gib=200.0, extra_replica_ram_gib=24.0), 2)
    monkeypatch.setattr(serve, "build_command", lambda _s: ["/bin/true"])
    for s in specs:
        assert "MemoryMax=200G" in serve.render_unit(s)


def test_the_allocator_is_charged_the_resting_size(empty_machine, monkeypatch):
    """...while the allocator counts the resting cost, or it refuses replicas
    that comfortably fit once loaded."""
    specs = serve.replica_specs(base(ram_gib=200.0, extra_replica_ram_gib=24.0), 2)
    monkeypatch.setattr(serve, "build_command", lambda _s: ["/bin/true"])
    assert "X-DEEPME-RAM=200" in serve.render_unit(specs[0])
    assert "X-DEEPME-RAM=24" in serve.render_unit(specs[1])


def test_scaling_stops_the_whole_previous_generation_first(monkeypatch):
    """Rewriting units one at a time leaves older replicas crash-looping
    against the new layout and stealing the rendezvous port."""
    stopped = []
    monkeypatch.setattr(
        res,
        "current_claims",
        lambda: [
            res.Claim(unit="deepme-llm-h3-video", profile="h3-video", gpus=(0, 1)),
            res.Claim(unit="deepme-llm-h3-video-r1", profile="h3-video", gpus=(2, 3)),
            res.Claim(unit="deepme-llm-other", profile="other", gpus=(4, 5)),
        ],
    )
    monkeypatch.setattr(serve.remote, "run", lambda cmd, **kw: stopped.append(cmd))
    monkeypatch.setattr(serve, "_wait_gpus_released", lambda units, **kw: None)
    units = serve.stop_profile(base())
    assert units == ["deepme-llm-h3-video", "deepme-llm-h3-video-r1"]
    assert not any("other" in c for c in stopped), "must not stop another model"


def test_a_replica_must_bind_the_port_we_assigned_it():
    """sglang auto-selects a free port when the requested one is busy and
    reports success. Every address here is the ASSIGNED port, so a drifted
    replica serves on a port nobody knows and reads as unhealthy forever."""
    from deepme import engines

    cmd = engines.get("sglang").build_command(base(), "key")
    assert cmd[cmd.index("--strict-ports") + 1] == "true"


def test_replica_ports_come_from_the_declared_registry(empty_machine):
    """Measured: replica 0 held 8030 AND 8031. Derived ports cannot see that,
    so allocations are declared and validated in deploy/ports.toml."""
    from deepme import ports as ports_mod

    declared = ports_mod.load().replica_ports("h3-video")
    ports = [s.port for s in serve.replica_specs(base(), 4)]
    assert ports == declared[:4]
    import itertools

    assert all(b - a > 1 for a, b in itertools.pairwise(ports))


def test_the_serving_port_is_identified_by_asking_not_by_guessing(monkeypatch):
    """A unit holds several sockets (API, rendezvous, an internal store on
    2379). Picking "the lowest" reported 2379 as the API port."""
    ss = "2379 100\n8030 100\n30005 100\n"

    def fake_run(cmd, **kw):
        class R:
            pass

        r = R()
        if "ss -lntp" in cmd:
            r.stdout = ss
        elif "/proc/" in cmd:
            r.stdout = "0::/system.slice/deepme-llm-h3-video.service"
        else:  # curl probe: only 8030 answers the model API
            r.stdout = "200" if ":8030/" in cmd else "000"
        return r

    monkeypatch.setattr(serve.remote, "run", fake_run)
    assert serve.serving_port(base()) == 8030


def test_no_listening_sockets_means_still_loading_not_drifted(monkeypatch):
    class R:
        stdout = ""

    monkeypatch.setattr(serve.remote, "run", lambda cmd, **kw: R())
    assert serve.serving_port(base()) is None


def test_placement_uses_the_accounted_ram_not_the_cgroup_cap(monkeypatch):
    """Third place the cap/charge split matters. Checking the 200GiB cap here
    refused replica 3 with "needs 200GiB RAM, only 200GiB free"."""
    spec = dataclasses.replace(
        base(ram_gib=200.0, extra_replica_ram_gib=24.0),
        replica=3,
        claim_ram_gib=24.0,
        mode="gpu",
    )
    avail = res.Availability(free_gpus=(6, 7), free_cpus=(), free_ram_gib=100.0)
    monkeypatch.setattr(res, "availability", lambda *a, **kw: avail)
    assert res.check_placement(spec, CAP, []) is None


def test_the_first_replica_is_charged_its_resting_size_not_its_load_peak():
    """Loads are staggered, so a peak is transient and never overlaps another
    profile's. Charging it permanently reserves memory that is briefly used
    and refuses placements the machine can hold — which is what blocked the
    3+1 mix (h3-ref needed a 320GiB cap but rests near 100GiB)."""
    spec = base(ram_gib=320.0, resting_ram_gib=100.0)
    assert spec.ram_gib == 320.0  # cgroup cap: survives the load
    assert spec.accounted_ram_gib == 100.0  # allocator: what it occupies


def test_without_a_measured_resting_size_the_cap_is_charged():
    """The safe default for a profile nobody has measured."""
    assert base(ram_gib=320.0).accounted_ram_gib == 320.0


def test_later_replicas_still_use_the_extra_replica_charge(empty_machine):
    specs = serve.replica_specs(
        base(ram_gib=320.0, resting_ram_gib=100.0, extra_replica_ram_gib=24.0), 3
    )
    assert [s.accounted_ram_gib for s in specs] == [100.0, 24.0, 24.0]
    assert [s.ram_gib for s in specs] == [320.0] * 3
