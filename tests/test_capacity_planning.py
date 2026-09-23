"""Planning a placement into the space that is free RIGHT NOW.

The fit check on a profile answers "could this run on an empty machine".
This answers the operational question instead: given what is already running,
is there room, and which GPUs would it use. Getting it wrong wastes a
multi-minute load before the engine refuses.
"""

import pytest

from deepme import config
from deepme import resources as res

# Two sockets, four GPUs each — the shape of the real box.
CAP = res.Capacity(
    gpus=tuple(range(8)),
    numa={0: tuple(range(64)), 1: tuple(range(64, 128))},
    ram_gib=503.0,
)


def avail(free_gpus, ram=400.0):
    return res.Availability(free_gpus=tuple(free_gpus), free_cpus=(), free_ram_gib=ram)


@pytest.fixture(autouse=True)
def _fixed_budget(monkeypatch):
    """~28.8GiB usable of a 32GiB card at 0.90."""
    monkeypatch.setattr(config, "per_gpu_weight_budget_gib", lambda frac: 32.0 * frac)


def test_gpus_split_by_socket():
    assert CAP.numa_gpus == {0: (0, 1, 2, 3), 1: (4, 5, 6, 7)}


def test_ranks_are_placed_inside_one_socket_when_possible():
    """Tensor parallel all-reduces every layer; crossing sockets taxes each."""
    plans = res.gpu_plans(weight_gib=40.0, gpu_mem=0.9, cap=CAP, avail=avail(range(8)))
    best = plans[0]
    assert set(best.gpus) <= {0, 1, 2, 3}


def test_cheapest_placement_comes_first_so_room_is_left_for_the_next_model():
    plans = res.gpu_plans(weight_gib=40.0, gpu_mem=0.9, cap=CAP, avail=avail(range(8)))
    assert [p.tp for p in plans] == sorted(p.tp for p in plans)
    assert plans[0].tp == 2  # 40GiB needs 2 ranks at 28.8GiB budget


def test_only_free_gpus_are_offered():
    """H3 holding 6 and 7 must never be handed out to something else."""
    plans = res.gpu_plans(weight_gib=20.0, gpu_mem=0.9, cap=CAP, avail=avail([0, 1, 2, 3, 4, 5]))
    assert all(6 not in p.gpus and 7 not in p.gpus for p in plans)


def test_a_model_that_cannot_be_split_evenly_skips_those_rank_counts():
    """64 heads: tp=6 is arithmetically tempting but structurally invalid."""
    plans = res.gpu_plans(
        weight_gib=155.0,
        gpu_mem=0.9,
        cap=CAP,
        avail=avail([0, 1, 2, 3, 4, 5]),
        attn_heads=64,
    )
    assert plans == []


def test_the_same_model_fits_once_all_eight_are_free():
    plans = res.gpu_plans(
        weight_gib=155.0,
        gpu_mem=0.9,
        cap=CAP,
        avail=avail(range(8)),
        attn_heads=64,
    )
    assert [p.tp for p in plans] == [8]


def test_no_free_gpus_yields_no_placements():
    assert res.gpu_plans(weight_gib=1.0, gpu_mem=0.9, cap=CAP, avail=avail([])) == []


def test_headroom_reports_what_is_left_for_kv_and_activations():
    (plan,) = res.gpu_plans(
        weight_gib=57.0, gpu_mem=0.9, cap=CAP, avail=avail([0, 1]), attn_heads=64
    )
    assert plan.per_gpu_gib == pytest.approx(28.5)
    assert plan.headroom_gib == pytest.approx(28.8 - 28.5, abs=1e-6)


def test_max_tp_caps_the_search():
    plans = res.gpu_plans(weight_gib=10.0, gpu_mem=0.9, cap=CAP, avail=avail(range(8)), max_tp=2)
    assert max(p.tp for p in plans) == 2


def test_spilling_across_sockets_beats_refusing_to_place():
    """Only 3 free in one socket and 2 in the other, but 4 ranks are needed."""
    plans = res.gpu_plans(weight_gib=100.0, gpu_mem=0.9, cap=CAP, avail=avail([1, 2, 3, 6, 7]))
    assert plans and len(plans[0].gpus) == 4
    assert set(plans[0].gpus) == {1, 2, 3, 6}


def test_layer_split_engines_are_derated_and_labelled_honestly():
    """llama.cpp splits by layer, packs unevenly, and ignores tp entirely —
    calling its GPU count "tp" and skipping the derate overstates capacity."""
    tp_plans = res.gpu_plans(weight_gib=151.0, gpu_mem=0.9, cap=CAP, avail=avail(range(6)))
    split_plans = res.gpu_plans(
        weight_gib=151.0,
        gpu_mem=0.9,
        cap=CAP,
        avail=avail(range(6)),
        derate=0.85,
        how="layer-split",
    )
    assert tp_plans, "undented budget accepts it"
    assert not split_plans, "derated budget must reject it"


def test_the_derated_label_reaches_the_operator():
    (plan,) = res.gpu_plans(
        weight_gib=40.0,
        gpu_mem=0.9,
        cap=CAP,
        avail=avail([0, 1]),
        derate=0.85,
        how="layer-split",
    )
    assert plan.describe().startswith("layer-split=2 ")


def test_headroom_reflects_the_derated_budget():
    (plan,) = res.gpu_plans(
        weight_gib=48.0,
        gpu_mem=0.9,
        cap=CAP,
        avail=avail([0, 1]),
        derate=0.85,
        how="layer-split",
    )
    assert plan.headroom_gib == pytest.approx(32.0 * 0.9 * 0.85 - 24.0)
