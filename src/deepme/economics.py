"""Derive prices from what the server costs and what the fleet measures.

One direction only: measured occupancy + configured costs -> derived prices.
Nothing here is typed in as a price; every figure traces back to
deploy/economics.toml, which separates DECISIONS (server cost, market rate,
margins, surge ladder) from FACTS (measured gpu-seconds per task, written
from named runs). Change a decision or re-measure a fact and every derived
number moves together — the CLI (`deepme econ`), the gateway's /v1/pricing,
and any service layer consuming either.

The economics are utilization economics. The server is a fixed $/hour whether
it generates or idles, so marginal cost of a job is ~0 and the questions that
matter are: what does a busy replica-second cost when fully allocated, how
much of the day must be sold to break even, and how should price move with
demand so the queue clears at the revenue-maximising depth.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import config

ECONOMICS_FILE = config.REPO_ROOT / "deploy" / "economics.toml"


@dataclass(frozen=True)
class TaskEconomics:
    """Everything derived for one generation task."""

    task: str
    gpu_seconds: float  # replica-seconds one request occupies
    output_seconds: float  # seconds of video it produces
    cost_per_clip: float  # fully-allocated, at 100% fleet utilization
    cost_per_output_second: float
    clips_per_replica_hour: float
    output_seconds_per_replica_hour: float
    spot_floor: float  # per output second; cost * floor margin
    market_price: float  # per output second; rate * task multiplier
    margin_at_market: float  # market_price / cost_per_output_second
    runs: str  # provenance: which measurements back this


@dataclass(frozen=True)
class MixEconomics:
    """Revenue shape of one fleet mix at the configured prices."""

    name: str
    replicas: dict[str, int]
    output_seconds_per_hour: float
    revenue_per_hour: float  # at 100% utilization, configured prices
    profit_per_hour: float
    break_even_utilization: float  # fraction of the day that must be sold


@dataclass(frozen=True)
class Economics:
    cost_per_hour: float
    replicas: int
    cost_per_replica_second: float
    tasks: dict[str, TaskEconomics]
    mixes: list[MixEconomics]
    surge: list[dict]  # resolved ladder, absolute per-task prices
    duration_points: list[dict]
    duration_note: str
    steps_note: str

    def as_dict(self) -> dict:
        return {
            "cost_per_hour": self.cost_per_hour,
            "replicas": self.replicas,
            "cost_per_replica_second": round(self.cost_per_replica_second, 7),
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
            "mixes": [asdict(m) for m in self.mixes],
            "surge": self.surge,
            "duration_points": self.duration_points,
            "duration_note": self.duration_note,
            "steps_note": self.steps_note,
        }


@dataclass(frozen=True)
class Config:
    """deploy/economics.toml, parsed but not yet derived."""

    cost_per_hour: float
    replicas: int
    rate_per_output_second: float
    spot_floor_margin: float
    task_multipliers: dict[str, float]
    surge: list[dict]
    mixes: dict[str, dict[str, int]]
    measured_tasks: dict[str, dict]
    duration_points: list[dict] = field(default_factory=list)
    steps_points: list[dict] = field(default_factory=list)
    default_steps: int = 49


def load(path: Path | None = None) -> Config:
    raw = tomllib.loads((path or ECONOMICS_FILE).read_text())
    server, market = raw["server"], raw["market"]
    pricing = raw["pricing"]
    measured = raw.get("measured", {})
    return Config(
        cost_per_hour=float(server["cost_per_hour"]),
        replicas=int(server["gpus"]) // int(server["gpus_per_replica"]),
        rate_per_output_second=float(market["rate_per_output_second"]),
        spot_floor_margin=float(pricing["spot_floor_margin"]),
        task_multipliers={k: float(v) for k, v in pricing.get("task_multipliers", {}).items()},
        surge=list(pricing.get("surge", [])),
        mixes={k: dict(v) for k, v in pricing.get("mixes", {}).items()},
        measured_tasks=dict(measured.get("tasks", {})),
        duration_points=list(measured.get("duration", {}).get("points", [])),
        steps_points=list(measured.get("steps", {}).get("points", [])),
        default_steps=int(measured.get("steps", {}).get("default_steps", 49)),
    )


def _round_price(value: float) -> float:
    """Up to the next half-cent — prices are quoted, not computed, by users."""
    import math

    return math.ceil(value * 200) / 200


def derive(cfg: Config) -> Economics:
    cost_replica_s = cfg.cost_per_hour / 3600 / cfg.replicas

    # Duration scaling is SUPERLINEAR (measured: 15s costs 1.51x linear), so
    # the per-output-second cost of the WORST legal duration exceeds the cost
    # measured on short clips. A floor derived from short clips alone would
    # lose money on long ones — scale it by the worst point on the curve.
    per_out_s = [float(p["gpu_seconds"]) / float(p["seconds"]) for p in cfg.duration_points]
    duration_worst_ratio = max(per_out_s) / min(per_out_s) if per_out_s else 1.0

    tasks: dict[str, TaskEconomics] = {}
    for name, m in sorted(cfg.measured_tasks.items()):
        gpu_s, out_s = float(m["gpu_seconds"]), float(m["output_seconds"])
        cost_clip = gpu_s * cost_replica_s
        cost_out_s = cost_clip / out_s
        market = cfg.rate_per_output_second * cfg.task_multipliers.get(name, 1.0)
        tasks[name] = TaskEconomics(
            task=name,
            gpu_seconds=gpu_s,
            output_seconds=out_s,
            cost_per_clip=round(cost_clip, 4),
            cost_per_output_second=round(cost_out_s, 4),
            clips_per_replica_hour=round(3600 / gpu_s, 2),
            output_seconds_per_replica_hour=round(3600 / gpu_s * out_s, 1),
            spot_floor=_round_price(cost_out_s * duration_worst_ratio * cfg.spot_floor_margin),
            market_price=round(market, 4),
            margin_at_market=round(market / cost_out_s, 2),
            runs=str(m.get("runs", "")),
        )

    mixes: list[MixEconomics] = []
    for mix_name, alloc in cfg.mixes.items():
        out_s_hr = sum(
            tasks[t].output_seconds_per_replica_hour * n for t, n in alloc.items() if t in tasks
        )
        revenue = sum(
            tasks[t].output_seconds_per_replica_hour * n * tasks[t].market_price
            for t, n in alloc.items()
            if t in tasks
        )
        mixes.append(
            MixEconomics(
                name=mix_name,
                replicas=dict(alloc),
                output_seconds_per_hour=round(out_s_hr, 1),
                revenue_per_hour=round(revenue, 2),
                profit_per_hour=round(revenue - cfg.cost_per_hour, 2),
                break_even_utilization=round(cfg.cost_per_hour / revenue, 3)
                if revenue
                else float("inf"),
            )
        )
    mixes.sort(key=lambda m: -m.profit_per_hour)

    surge = [
        {
            "queued_per_replica": float(tier["queued_per_replica"]),
            "multiplier": float(tier["multiplier"]),
            "prices": {
                t: round(te.market_price * float(tier["multiplier"]), 4) for t, te in tasks.items()
            },
        }
        for tier in sorted(cfg.surge, key=lambda t: float(t["queued_per_replica"]))
    ]

    duration_points = []
    base = None
    for p in sorted(cfg.duration_points, key=lambda p: float(p["seconds"])):
        secs, gpu_s = float(p["seconds"]), float(p["gpu_seconds"])
        base = base or (secs, gpu_s)
        duration_points.append(
            {
                "seconds": secs,
                "gpu_seconds": gpu_s,
                "cost_per_output_second": round(gpu_s * cost_replica_s / secs, 4),
                # vs linear scaling from the shortest measured point: 1.0 = linear,
                # below = sublinear (longer clips are cheaper per second).
                "vs_linear": round((gpu_s / base[1]) / (secs / base[0]), 3),
            }
        )
    duration_note = (
        "duration cost curve measured — see vs_linear per point"
        if duration_points
        else "NOT yet measured; pricing assumes cost LINEAR in duration (the "
        "conservative reading). Run the duration-curve batch to replace the "
        "assumption with data."
    )

    steps_note = ""
    if len(cfg.steps_points) >= 2:
        lo, hi = (sorted(cfg.steps_points, key=lambda p: p["steps"])[i] for i in (0, -1))
        per_step = (hi["gpu_seconds"] - lo["gpu_seconds"]) / (hi["steps"] - lo["steps"])
        overhead = hi["gpu_seconds"] - per_step * hi["steps"]
        steps_note = (
            f"~{per_step:.1f}s/step + {overhead:.0f}s fixed: a reduced-step "
            f"draft tier at half the default steps costs ~"
            f"{(overhead + per_step * cfg.default_steps / 2) / (overhead + per_step * cfg.default_steps):.0%} "
            f"of a full-quality run"
        )

    return Economics(
        cost_per_hour=cfg.cost_per_hour,
        replicas=cfg.replicas,
        cost_per_replica_second=cost_replica_s,
        tasks=tasks,
        mixes=mixes,
        surge=surge,
        duration_points=duration_points,
        duration_note=duration_note,
        steps_note=steps_note,
    )


def pricing_json(econ: Economics) -> dict:
    """The static half of the gateway's /v1/pricing.

    Frozen into the gateway environment at `gateway start` — the same way
    routes are — so the gateway can quote without reading files. The LIVE
    half (which surge tier applies right now) comes from the gateway's own
    queue signal at request time.
    """
    import hashlib
    import time

    table = {
        t: {
            "cost": te.cost_per_output_second,
            "spot_floor": te.spot_floor,
            "market": te.market_price,
        }
        for t, te in econ.tasks.items()
    }
    return {
        "currency": "USD",
        "unit": "per_output_second",
        # Consumers detect a stale gateway table by comparing this hash with
        # a fresh local derivation (`deepme econ --json`) — the table freezes
        # at `gateway start` and editing economics.toml does NOT refresh it.
        "derived_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "table_hash": hashlib.sha256(
            json.dumps({"tasks": table, "surge": econ.surge}, sort_keys=True).encode()
        ).hexdigest()[:16],
        "tasks": table,
        "surge": econ.surge,
        # Cost is superlinear in clip length (measured), so under flat
        # per-second pricing a long clip carries the thinnest margin. The
        # curve travels with the prices so an app layer CAN price by duration;
        # vs_linear = 1.0 means a clip costs its length, above means more.
        "duration_curve": econ.duration_points,
        # The contract a billing layer must honour: quotes are indicative,
        # the price locks at the `submitted` event (billable=true), because
        # queue depth moves while a request waits.
        "lock_at": "submitted",
    }


def record_duration_batch(batch_name: str = "duration-curve") -> list[dict]:
    """Write measured duration points from a finished batch into the config.

    Reads the batch ledger (took_s per item, seconds per item) and rewrites
    the `points = []` line of [measured.duration]. Refuses partial batches:
    a curve missing its long end would silently under-price long clips.
    """
    from . import batch as batch_mod

    batch = batch_mod.Batch.load(batch_name)
    if batch is None:
        raise SystemExit(f"no batch named {batch_name!r}")
    bad = [i for i in batch.items if i.status != "done"]
    if bad:
        raise SystemExit(
            f"batch {batch_name!r} has {len(bad)} unfinished/failed item(s) — "
            "a partial curve would misprice whatever is missing"
        )
    no_secs = [i.name for i in batch.items if i.seconds is None]
    if no_secs:
        raise SystemExit(
            f"item(s) {no_secs} in {batch_name!r} do not set `seconds` — a "
            "duration curve needs the duration as the independent variable"
        )
    points = sorted(
        (
            {"seconds": float(i.seconds or 0), "gpu_seconds": round(i.took_s, 1)}
            for i in batch.items
        ),
        key=lambda p: p["seconds"],
    )
    text = ECONOMICS_FILE.read_text()
    marker = "points = []"
    if marker not in text:
        raise SystemExit(
            f"{ECONOMICS_FILE} has no empty duration points to fill — "
            "already recorded? Edit [measured.duration] deliberately instead."
        )
    rendered = (
        "points = [\n"
        + "".join(
            f"  {{ seconds = {p['seconds']}, gpu_seconds = {p['gpu_seconds']} }},\n" for p in points
        )
        + "]"
    )
    ECONOMICS_FILE.write_text(text.replace(marker, rendered, 1))
    return points
