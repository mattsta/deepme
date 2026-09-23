"""Fleet composition: how many replicas of which models share the machine.

GPUs are the hard constraint — 8 cards at 2 per replica is four replicas in
any arrangement — so the split between capability tiers is a COMMERCIAL
decision, not an engineering one. A premium surface (H3's ref2va, which needs
a different checkpoint) costs base-tier throughput one for one, and whether
that trade is worth making depends on pricing and demand, both of which
change faster than code should.

So the mix is declared in TOML and this module places it. Adding a tier is an
edit to a file, not a deployment script.

    [[member]]
    profile  = "h3-video"   # t2va + fl2va
    replicas = 3

    [[member]]
    profile  = "h3-ref"     # ref2va (premium)
    replicas = 1
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import config, profiles, serve
from .profiles import RunSpec

FLEETS_DIR = config.REPO_ROOT / "deploy" / "fleets"


@dataclass(frozen=True)
class Member:
    profile: str
    replicas: int


@dataclass(frozen=True)
class Fleet:
    name: str
    members: tuple[Member, ...]

    @property
    def total_replicas(self) -> int:
        return sum(m.replicas for m in self.members)


def fleet_path(name: str) -> Path:
    return FLEETS_DIR / f"{name}.toml"


def load(name: str) -> Fleet:
    path = fleet_path(name)
    if not path.exists():
        raise SystemExit(
            f"no fleet named {name!r} — expected {path}. Available: "
            f"{sorted(p.stem for p in FLEETS_DIR.glob('*.toml'))}"
        )
    raw = tomllib.loads(path.read_text())
    members = tuple(
        Member(profile=m["profile"], replicas=int(m.get("replicas", 1)))
        for m in raw.get("member", [])
    )
    if not members:
        raise SystemExit(f"{path} declares no [[member]] entries")
    for m in members:
        if m.replicas < 1:
            raise SystemExit(f"{path}: {m.profile!r} declares {m.replicas} replicas")
    return Fleet(name=name, members=members)


def plan(fleet: Fleet, *, releasing_extra: set[str] | None = None) -> list[RunSpec]:
    """Resolve a fleet into concrete, placed deployments.

    Members are laid out in declaration order against ONE shared GPU pool, so
    an earlier member's replicas are already claimed when the next is placed
    and the sets cannot overlap. Ports are separated per profile by the
    profile's own base port, and per replica by the existing stride.

    `releasing_extra` names profiles whose GPUs the caller intends to free —
    `fleet switch` passes every running model profile, because its delta
    DISPLACES whatever squats on GPUs the target needs. Without it, a mix
    that does not mention a profile could never take that profile's GPUs:
    switching 3+1 -> 4+0 would fail planning because h3-ref (absent from the
    4+0 mix) still holds two cards.
    """
    # Applying this fleet stops every profile it names, so their GPUs count as
    # available while planning. Otherwise a mix could only be planned from an
    # idle machine, and `fleet plan` — a dry run — would have to tear down the
    # fleet it is describing.
    releasing = {m.profile for m in fleet.members} | (releasing_extra or set())
    placed: list[RunSpec] = []
    for member in fleet.members:
        spec = profiles.load(member.profile)
        specs = serve.replica_specs(
            spec, member.replicas, reserved=list(placed), releasing=releasing
        )
        placed.extend(specs)
    return placed


@dataclass(frozen=True)
class Delta:
    """What separates the running fleet from a target mix.

    The whole point of computing this: `switch` touches ONLY these. A mix
    change that restarts an unchanged replica costs ~5 minutes of that
    replica's revenue for nothing.
    """

    keep: tuple[RunSpec, ...]  # running, healthy placement matches target
    add: tuple[RunSpec, ...]  # in the target, not running
    remove: tuple[str, ...]  # running units with no place in the target
    remove_ports: dict[str, int]  # unit -> port, to address its backend URL

    @property
    def is_noop(self) -> bool:
        return not self.add and not self.remove


def delta(target: list[RunSpec], claims) -> Delta:
    """Diff a planned mix against what the machine is actually running.

    Auto-detecting, deliberately: the current state comes from live unit
    claims, never from remembering what was applied last — an operator who
    hand-stopped a replica yesterday must not confuse today's switch.

    A running unit is KEPT only when the target places the same unit on the
    same GPUs and port. A profile match with different placement is a
    remove+add — restarting it is the only honest way to move a replica.
    """
    running = {c.unit: c for c in claims if c.profile and c.unit.startswith("deepme-llm")}
    target_units = {s.unit: s for s in target}

    keep, add = [], []
    for spec in target:
        current = running.get(spec.unit)
        if (
            current is not None
            and tuple(current.gpus) == tuple(spec.gpus)
            and current.port == spec.port
        ):
            keep.append(spec)
        else:
            add.append(spec)

    target_gpus = {g for s in target for g in s.gpus}
    remove, remove_ports = [], {}
    for unit, claim in running.items():
        if unit in target_units and any(
            tuple(c.gpus) == tuple(target_units[unit].gpus) and c.port == target_units[unit].port
            for c in [claim]
        ):
            continue
        # Anything else goes if the target claims its unit name OR its GPUs.
        # A profile the target does not mention, on GPUs the target does not
        # touch (e.g. a text model on other cards), is left alone.
        if unit in target_units or (set(claim.gpus) & target_gpus):
            remove.append(unit)
            if claim.port:
                remove_ports[unit] = claim.port
    return Delta(
        keep=tuple(keep),
        add=tuple(add),
        remove=tuple(sorted(remove)),
        remove_ports=remove_ports,
    )


def describe(specs: list[RunSpec]) -> str:
    by_profile: dict[str, list[RunSpec]] = {}
    for spec in specs:
        by_profile.setdefault(spec.profile, []).append(spec)
    lines = []
    for profile, group in by_profile.items():
        gpus = ", ".join(str(list(s.gpus)) for s in group)
        ports = ", ".join(str(s.port) for s in group)
        lines.append(f"  {profile} x{len(group)}: GPUs {gpus} ports {ports}")
    return "\n".join(lines)
