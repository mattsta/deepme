"""Resource allocator: capacity math, NUMA-aware packing, conflict rules."""

from dataclasses import dataclass

from deepme import resources
from deepme.resources import Capacity, Claim


def cap8() -> Capacity:
    # Mirrors the reference machine: 2 NUMA nodes, 4 GPUs each, SMT pairs.
    return Capacity(
        gpus=tuple(range(8)),
        numa={
            0: tuple(range(64)) + tuple(range(128, 192)),
            1: tuple(range(64, 128)) + tuple(range(192, 256)),
        },
        ram_gib=503.0,
    )


@dataclass
class FakeSpec:
    unit: str = "deepme-llm-x"
    mode: str = "gpu"
    gpus: tuple = ()
    cpus: int = 0
    ram_gib: float = 0.0
    port: int = 8000

    @property
    def accounted_ram_gib(self) -> float:
        """What the allocator charges — the cap and the charge differ once
        replicas share a weight store (see RunSpec.accounted_ram_gib)."""
        return self.ram_gib


def test_cpu_list_roundtrip():
    parsed = resources._parse_cpu_list("0-3,8,10-11")
    assert parsed == (0, 1, 2, 3, 8, 10, 11)
    assert resources.format_cpu_list(parsed) == "0-3,8,10-11"


def test_availability_subtracts_claims_and_reserves():
    cap = cap8()
    claims = [Claim(unit="a", gpus=(0, 1), cpus=(10, 11), ram_gib=64, port=8000)]
    avail = resources.availability(cap, claims)
    assert 0 not in avail.free_gpus and 1 not in avail.free_gpus
    assert 2 in avail.free_gpus
    assert 10 not in avail.free_cpus
    # system reservation carved off the front
    assert 0 not in avail.free_cpus
    assert avail.free_ram_gib == 503.0 - resources.RESERVED_RAM_GIB - 64


def test_pick_cpus_packs_within_one_numa_node():
    cap = cap8()
    avail = resources.availability(cap, [])
    picked = resources.pick_cpus(avail, cap, 32)
    assert len(picked) == 32
    node0 = set(cap.numa[0])
    node1 = set(cap.numa[1])
    assert set(picked) <= node0 or set(picked) <= node1


def test_pick_cpus_spills_when_no_single_node_fits():
    cap = cap8()
    # occupy most of node 0 and node 1, leaving a split remainder
    claims = [
        Claim(unit="a", cpus=tuple(cap.numa[0][:120])),
        Claim(unit="b", cpus=tuple(cap.numa[1][:120])),
    ]
    avail = resources.availability(cap, claims)
    picked = resources.pick_cpus(avail, cap, 16)
    assert len(picked) == 16


def test_cpu_model_does_not_displace_gpu_model():
    gpu_claim = Claim(unit="deepme-llm-big", gpus=(0, 1, 2, 3, 4, 5, 6, 7), port=8000)
    cpu_spec = FakeSpec(unit="deepme-llm-small", mode="cpu", gpus=(), port=8010)
    assert resources.conflicts_for(cpu_spec, [gpu_claim]) == []


def test_gpu_overlap_conflicts():
    claim = Claim(unit="deepme-llm-a", gpus=(0, 1), port=8000)
    spec = FakeSpec(unit="deepme-llm-b", gpus=(1, 2), port=8001)
    assert resources.conflicts_for(spec, [claim]) == ["deepme-llm-a"]


def test_disjoint_gpus_coexist():
    claim = Claim(unit="deepme-llm-a", gpus=(0, 1, 2, 3), port=8000)
    spec = FakeSpec(unit="deepme-llm-b", gpus=(4, 5, 6, 7), port=8001)
    assert resources.conflicts_for(spec, [claim]) == []


def test_port_collision_conflicts_even_without_gpu_overlap():
    claim = Claim(unit="deepme-llm-a", gpus=(0,), port=8000)
    spec = FakeSpec(unit="deepme-llm-b", gpus=(1,), port=8000)
    assert resources.conflicts_for(spec, [claim]) == ["deepme-llm-a"]


def test_check_placement_rejects_insufficient_cpus():
    cap = cap8()
    claims = [Claim(unit="hog", cpus=tuple(cap.cpus[:250]))]
    spec = FakeSpec(unit="new", mode="cpu", cpus=32, port=8010)
    err = resources.check_placement(spec, cap, claims)
    assert err and "CPUs" in err


def test_check_placement_rejects_insufficient_ram():
    cap = cap8()
    claims = [Claim(unit="hog", ram_gib=460)]
    spec = FakeSpec(unit="new", mode="cpu", cpus=8, ram_gib=64, port=8010)
    err = resources.check_placement(spec, cap, claims)
    assert err and "RAM" in err


def test_check_placement_allows_colocation():
    cap = cap8()
    claims = [Claim(unit="deepme-llm-gpu", gpus=tuple(range(8)), port=8000)]
    spec = FakeSpec(unit="deepme-llm-cpu", mode="cpu", cpus=32, ram_gib=80, port=8010)
    assert resources.check_placement(spec, cap, claims) is None


def test_check_placement_excludes_units_it_will_stop():
    # Replacing a model with itself must not report its own resources as taken
    cap = cap8()
    claims = [
        Claim(
            unit="deepme-llm-x",
            gpus=tuple(range(8)),
            cpus=tuple(cap.cpus[:200]),
            ram_gib=400,
            port=8000,
        )
    ]
    spec = FakeSpec(unit="deepme-llm-x", gpus=tuple(range(8)), cpus=64, ram_gib=200, port=8000)
    assert resources.check_placement(spec, cap, claims) is None


def test_gpu_numa_node_mapping():
    cap = cap8()
    assert cap.gpu_numa_node(0) == 0
    assert cap.gpu_numa_node(3) == 0
    assert cap.gpu_numa_node(4) == 1
    assert cap.gpu_numa_node(7) == 1
