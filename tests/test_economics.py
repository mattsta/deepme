"""The pricing derivation: measured occupancy x server cost -> prices.

Every number a billing layer sees comes from derive(), so the invariants
pinned here are business invariants: costs trace to measurements, floors
cover every task, the surge ladder is monotonic, and the fleet-mix ranking
answers the question it exists for (is a specialised replica worth more than
another generalist?).
"""

import pytest

from deepme import economics

CFG = economics.Config(
    cost_per_hour=3.20,
    replicas=4,
    rate_per_output_second=0.13,
    spot_floor_margin=1.4,
    task_multipliers={"ref2va": 2.0},
    surge=[
        {"queued_per_replica": 0.0, "multiplier": 1.0},
        {"queued_per_replica": 1.0, "multiplier": 1.8},
    ],
    mixes={"4+0": {"t2va": 4}, "3+1": {"t2va": 3, "ref2va": 1}},
    measured_tasks={
        "t2va": {"gpu_seconds": 800.0, "output_seconds": 5.0, "runs": "reel"},
        "ref2va": {"gpu_seconds": 1254.0, "output_seconds": 5.0, "runs": "gen"},
    },
    steps_points=[
        {"steps": 20, "gpu_seconds": 283.0, "output_seconds": 4.0},
        {"steps": 50, "gpu_seconds": 659.0, "output_seconds": 4.0},
    ],
)


@pytest.fixture
def econ():
    return economics.derive(CFG)


def test_cost_traces_from_server_rate_through_occupancy(econ):
    """$3.20/hr over 4 replicas, 800 GPU-s for 5s of output -> $0.0356/s."""
    t = econ.tasks["t2va"]
    assert t.cost_per_output_second == pytest.approx(0.0356, abs=1e-4)
    assert t.cost_per_clip == pytest.approx(0.1778, abs=1e-3)
    assert econ.cost_per_replica_second == pytest.approx(3.20 / 3600 / 4)


def test_the_floor_always_covers_the_task_it_prices(econ):
    """A flat spot price loses money on the expensive task; a derived one
    cannot, because it is cost x margin per task, rounded UP."""
    for t in econ.tasks.values():
        assert t.spot_floor >= t.cost_per_output_second * 1.4 - 1e-9
    # The concrete case that motivated this: $0.05 flat spot is below
    # ref2va's $0.0557 cost. The derived floor is not.
    assert econ.tasks["ref2va"].spot_floor > 0.0557


def test_task_multiplier_moves_market_price_not_cost(econ):
    assert econ.tasks["ref2va"].market_price == pytest.approx(0.26)
    assert econ.tasks["t2va"].market_price == pytest.approx(0.13)
    # Margin reflects both: ref2va at 2x price / 1.57x cost is the BETTER
    # margin, which is exactly what the mix ranking rests on.
    assert econ.tasks["ref2va"].margin_at_market > econ.tasks["t2va"].margin_at_market


def test_mix_ranking_answers_the_fleet_question(econ):
    """3+1 with ref2va at 2x out-earns 4+0 — the specialised replica yields
    more revenue per hour than another generalist would."""
    by_name = {m.name: m for m in econ.mixes}
    assert by_name["3+1"].revenue_per_hour > by_name["4+0"].revenue_per_hour
    assert econ.mixes[0].name == "3+1", "ranked by profit"
    # And break-even utilization is the fraction of the day that must be sold.
    assert by_name["4+0"].break_even_utilization == pytest.approx(
        3.20 / by_name["4+0"].revenue_per_hour, abs=1e-3
    )


def test_surge_ladder_is_resolved_to_absolute_prices(econ):
    assert econ.surge[0]["prices"]["t2va"] == pytest.approx(0.13)
    assert econ.surge[1]["prices"]["ref2va"] == pytest.approx(0.468)
    # Monotonic: a deeper queue never gets cheaper.
    for a, b in zip(econ.surge, econ.surge[1:], strict=False):
        for task in a["prices"]:
            assert b["prices"][task] >= a["prices"][task]


def test_unmeasured_duration_is_declared_not_assumed_silently(econ):
    assert "NOT yet measured" in econ.duration_note
    assert econ.duration_points == []


def test_duration_curve_reports_deviation_from_linear():
    cfg = economics.Config(
        **{
            **CFG.__dict__,
            "duration_points": [
                {"seconds": 4, "gpu_seconds": 700.0},
                {"seconds": 15, "gpu_seconds": 1800.0},  # sublinear
            ],
        },
    )
    econ = economics.derive(cfg)
    long = econ.duration_points[-1]
    # Linear from the 4s point would be 700 * 15/4 = 2625s; 1800 is 0.686x.
    assert long["vs_linear"] == pytest.approx(0.686, abs=0.01)
    assert long["cost_per_output_second"] < econ.duration_points[0]["cost_per_output_second"], (
        "longer clips cheaper per second"
    )


def test_steps_note_prices_a_draft_tier(econ):
    """20 vs 50 steps at 283/659s -> ~12.5s/step + ~32s fixed."""
    assert "12.5s/step" in econ.steps_note


def test_pricing_json_carries_the_billing_contract(econ):
    p = economics.pricing_json(econ)
    assert p["lock_at"] == "submitted"
    assert p["unit"] == "per_output_second"
    assert p["tasks"]["ref2va"]["spot_floor"] > p["tasks"]["t2va"]["spot_floor"]
    assert p["surge"][0]["multiplier"] == 1.0


def test_the_floor_covers_the_worst_case_duration():
    """The measured curve is SUPERLINEAR: a 15s clip costs 1.51x linear.

    A floor derived from short clips alone ($0.05 for t2va) is below the
    long-clip cost ($0.054/s at 15s) — the same trap the flat floor had,
    reintroduced through the duration dimension. The floor must price the
    most expensive clip a customer can legally request.
    """
    cfg = economics.Config(
        **{
            **CFG.__dict__,
            "duration_points": [
                {"seconds": 4, "gpu_seconds": 644.2},
                {"seconds": 10, "gpu_seconds": 1796.8},
                {"seconds": 15, "gpu_seconds": 3644.7},
            ],
        },
    )
    econ = economics.derive(cfg)
    worst = max(p["cost_per_output_second"] for p in econ.duration_points)
    for t in econ.tasks.values():
        # cost scaled to the worst duration, with the margin, for every task.
        assert (
            t.spot_floor
            >= worst * 1.4 * (t.cost_per_output_second / econ.tasks["t2va"].cost_per_output_second)
            - 0.005
        )
    # The concrete number: t2va floor rises from $0.05 to cover $0.054 cost.
    assert econ.tasks["t2va"].spot_floor > 0.054
    # And the curve itself reports the superlinearity.
    assert econ.duration_points[-1]["vs_linear"] == pytest.approx(1.509, abs=0.01)


def test_the_duration_curve_travels_with_the_prices(econ):
    """An app layer pricing by duration needs the curve in-band."""
    p = economics.pricing_json(econ)
    assert "duration_curve" in p
