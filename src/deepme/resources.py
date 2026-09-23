"""Resource model: capacity, claims, and placement across GPU + CPU + RAM.

The server is a fixed pool (GPUs, logical CPUs grouped by NUMA node, RAM).
Every running deepme unit holds a CLAIM recorded in its systemd unit file
(X-DEEPME-GPUS / X-DEEPME-CPUS / X-DEEPME-RAM / X-DEEPME-PORT), so the live
allocation is read from the server's own state — never guessed locally.

This enables colocation: a CPU-mode GGUF model can run alongside GPU models
as long as cores/RAM/ports don't collide, and only genuinely-overlapping
units are displaced on launch.

NUMA matters: on this class of machine each socket owns half the GPUs and
half the cores, so CPU placements are packed within a single NUMA node
(cross-socket memory traffic dominates CPU inference otherwise).
"""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass, field

from . import config, remote

CAPACITY_CACHE = config.STATE_DIR / "capacity.json"

# Headroom never handed to models: the OS, SSH, the guard proxy, and the host
# processes of GPU engines all need cores and RAM.
RESERVED_CPUS = 8
RESERVED_RAM_GIB = 32.0


@dataclass(frozen=True)
class Capacity:
    gpus: tuple[int, ...]
    numa: dict[int, tuple[int, ...]]  # node -> logical CPUs
    ram_gib: float

    @property
    def cpus(self) -> tuple[int, ...]:
        return tuple(sorted(c for cpus in self.numa.values() for c in cpus))

    def gpu_numa_node(self, gpu: int) -> int | None:
        """Which NUMA node a GPU is attached to (even split by index)."""
        if not self.numa or not self.gpus:
            return None
        per = max(1, len(self.gpus) // max(1, len(self.numa)))
        node = gpu // per
        nodes = sorted(self.numa)
        return nodes[min(node, len(nodes) - 1)]

    @property
    def numa_gpus(self) -> dict[int, tuple[int, ...]]:
        """node -> GPUs attached to it, for socket-local rank placement."""
        out: dict[int, list[int]] = {}
        for g in self.gpus:
            node = self.gpu_numa_node(g)
            if node is not None:
                out.setdefault(node, []).append(g)
        return {k: tuple(v) for k, v in out.items()}


@dataclass(frozen=True)
class Claim:
    """Resources one running unit holds."""

    unit: str
    profile: str = ""
    gpus: tuple[int, ...] = ()
    cpus: tuple[int, ...] = ()
    ram_gib: float = 0.0
    port: int | None = None


def probe_capacity(*, refresh: bool = False) -> Capacity:
    """Read the machine's real topology (cached; `refresh` re-probes)."""
    if CAPACITY_CACHE.exists() and not refresh:
        d = json.loads(CAPACITY_CACHE.read_text())
        return Capacity(
            gpus=tuple(d["gpus"]),
            numa={int(k): tuple(v) for k, v in d["numa"].items()},
            ram_gib=d["ram_gib"],
        )
    script = """\
nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\\n' ',' | sed 's/,$//'
echo
lscpu | grep -E '^NUMA node[0-9]+ CPU' | sed 's/NUMA node//; s/ CPU(s): */|/'
echo '---'
awk '/MemTotal/{printf "%.1f\\n", $2/1048576}' /proc/meminfo
"""
    out = remote.run_script(script, check=False, timeout=120).stdout.splitlines()
    gpus: tuple[int, ...] = ()
    numa: dict[int, tuple[int, ...]] = {}
    ram = 0.0
    section = "gpus"
    for line in out:
        line = line.strip()
        if not line:
            continue
        if line == "---":
            section = "ram"
            continue
        if section == "gpus" and not gpus and "|" not in line and re.fullmatch(r"[\d,]+", line):
            gpus = tuple(int(x) for x in line.split(",") if x.strip().isdigit())
            continue
        if "|" in line:
            node, cpus = line.split("|", 1)
            numa[int(node.strip())] = _parse_cpu_list(cpus.strip())
        elif section == "ram":
            with contextlib.suppress(ValueError):
                ram = float(line)
    cap = Capacity(gpus=gpus, numa=numa, ram_gib=ram)
    config.STATE_DIR.mkdir(exist_ok=True)
    CAPACITY_CACHE.write_text(
        json.dumps(
            {
                "gpus": list(cap.gpus),
                "numa": {str(k): list(v) for k, v in cap.numa.items()},
                "ram_gib": cap.ram_gib,
            }
        )
    )
    return cap


def _parse_cpu_list(s: str) -> tuple[int, ...]:
    """Expand a cpuset string like '0-63,128-191' into explicit CPU ids."""
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if a.strip().isdigit() and b.strip().isdigit():
                out.extend(range(int(a), int(b) + 1))
        elif part.isdigit():
            out.append(int(part))
    return tuple(out)


def format_cpu_list(cpus: tuple[int, ...] | list[int]) -> str:
    """Compact a CPU id list back into systemd cpuset syntax ('0-7,16-23')."""
    if not cpus:
        return ""
    ids = sorted(set(cpus))
    ranges, start, prev = [], ids[0], ids[0]
    for c in ids[1:]:
        if c == prev + 1:
            prev = c
            continue
        ranges.append((start, prev))
        start = prev = c
    ranges.append((start, prev))
    return ",".join(f"{a}-{b}" if a != b else str(a) for a, b in ranges)


def active_units() -> set[str]:
    """Names of every active deepme unit, model and service alike.

    current_claims() deliberately reads only `deepme-llm*` because it is about
    GPU/CPU/RAM accounting, and the service units hold none of those. But the
    catalog still has to report whether the gateway and the balancers are up,
    and there is no way to infer that from resource claims — a balancer's
    liveness is its own, not its fleet's.
    """
    script = (
        "systemctl list-units 'deepme-*' --state=active --no-legend "
        "--no-pager --plain 2>/dev/null | awk '{print $1}'"
    )
    out = remote.run_script(script, check=False, timeout=60).stdout
    return {
        line.strip().removesuffix(".service")
        for line in out.splitlines()
        if line.strip().startswith("deepme-")
    }


def current_claims() -> list[Claim]:
    """Resource claims of every deepme unit currently loaded on the server."""
    script = """\
for f in /etc/systemd/system/deepme-llm*.service; do
  [ -e "$f" ] || continue
  u=$(basename "$f" .service)
  state=$(systemctl is-active "$u" 2>/dev/null)
  [ "$state" = "active" ] || [ "$state" = "activating" ] || continue
  g=$(grep -oP '(?<=X-DEEPME-GPUS=)[0-9,]*' "$f" | head -1)
  c=$(grep -oP '(?<=X-DEEPME-CPUS=)[0-9,-]*' "$f" | head -1)
  r=$(grep -oP '(?<=X-DEEPME-RAM=)[0-9.]*' "$f" | head -1)
  p=$(grep -oP '(?<=X-DEEPME-PORT=)[0-9]*' "$f" | head -1)
  pr=$(grep -oP '(?<=X-DEEPME-PROFILE=)[^ ]*' "$f" | head -1)
  echo "$u|$pr|$g|$c|$r|$p"
done
"""
    out = remote.run_script(script, check=False, timeout=120).stdout
    claims = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 6 or not parts[0]:
            continue
        unit, prof, g, c, r, p = parts
        claims.append(
            Claim(
                unit=unit,
                profile=prof,
                gpus=tuple(int(x) for x in g.split(",") if x.strip().isdigit()),
                cpus=_parse_cpu_list(c),
                ram_gib=float(r) if r.strip() else 0.0,
                port=int(p) if p.strip().isdigit() else None,
            )
        )
    return claims


@dataclass
class Availability:
    free_gpus: tuple[int, ...]
    free_cpus: tuple[int, ...]
    free_ram_gib: float
    used_ports: tuple[int, ...] = field(default_factory=tuple)


def availability(
    cap: Capacity, claims: list[Claim], *, exclude: frozenset[str] | set[str] = frozenset()
) -> Availability:
    """What's left after existing claims (optionally ignoring some units)."""
    live = [c for c in claims if c.unit not in exclude]
    used_gpus = {g for c in live for g in c.gpus}
    used_cpus = {p for c in live for p in c.cpus}
    used_ram = sum(c.ram_gib for c in live)
    # Reserve headroom from the front of the CPU list (OS/SSH/guard live there)
    reserved = set(cap.cpus[:RESERVED_CPUS])
    return Availability(
        free_gpus=tuple(g for g in cap.gpus if g not in used_gpus),
        free_cpus=tuple(c for c in cap.cpus if c not in used_cpus and c not in reserved),
        free_ram_gib=max(0.0, cap.ram_gib - RESERVED_RAM_GIB - used_ram),
        used_ports=tuple(c.port for c in live if c.port is not None),
    )


@dataclass(frozen=True)
class Placement:
    """A concrete way to run a model in the space that is free right now."""

    tp: int
    gpus: tuple[int, ...]
    per_gpu_gib: float  # weights each rank holds
    headroom_gib: float  # budget left per GPU for KV/activations
    how: str = "tp"  # how the engine divides them, for an honest label

    def describe(self) -> str:
        return (
            f"{self.how}={self.tp} on GPUs {list(self.gpus)} "
            f"({self.per_gpu_gib:.1f}GiB/GPU, {self.headroom_gib:.1f}GiB spare)"
        )


def pick_gpus(avail: Availability, cap: Capacity, count: int) -> tuple[int, ...]:
    """Choose `count` free GPUs, packed within ONE NUMA node when possible.

    Tensor parallelism is all-reduce heavy every layer, so ranks split across
    sockets pay for it on every step. Falls back to any free GPUs rather than
    refusing a placement that would work.
    """
    if count <= 0:
        return ()
    free = set(avail.free_gpus)
    for _node, node_gpus in sorted(cap.numa_gpus.items()):
        here = tuple(g for g in node_gpus if g in free)
        if len(here) >= count:
            return here[:count]
    return tuple(avail.free_gpus[:count]) if len(avail.free_gpus) >= count else ()


def gpu_plans(
    *,
    weight_gib: float,
    gpu_mem: float,
    cap: Capacity,
    avail: Availability,
    attn_heads: int = 0,
    max_tp: int = 0,
    derate: float = 1.0,
    how: str = "tp",
) -> list[Placement]:
    """Every way this model fits in the CURRENTLY FREE GPUs, cheapest first.

    "Cheapest" means fewest GPUs consumed, so planning one model leaves the
    most room for the next. Candidate rank counts are constrained by tensor
    parallelism's real rules: tp must divide the attention head count, and
    each rank must hold its shard within the per-GPU budget.

    `derate` and `how` come from the ENGINE's declared placement — layer-split
    engines pack unevenly and must be derated, and calling their GPU count
    "tp" would be a lie (tp is inert for them).
    """
    from . import config

    budget = config.per_gpu_weight_budget_gib(gpu_mem) * derate
    limit = min(max_tp or len(avail.free_gpus), len(avail.free_gpus))
    out = []
    for tp in range(1, limit + 1):
        if attn_heads and attn_heads % tp:
            continue
        per_gpu = weight_gib / tp
        if per_gpu > budget:
            continue
        gpus = pick_gpus(avail, cap, tp)
        if not gpus:
            continue
        out.append(
            Placement(
                tp=tp,
                gpus=gpus,
                per_gpu_gib=per_gpu,
                headroom_gib=budget - per_gpu,
                how=how,
            )
        )
    return out


def pick_cpus(avail: Availability, cap: Capacity, count: int) -> tuple[int, ...]:
    """Choose `count` free CPUs packed within ONE NUMA node when possible
    (cross-socket memory traffic dominates CPU inference)."""
    if count <= 0:
        return ()
    free = set(avail.free_cpus)
    best: tuple[int, ...] = ()
    for _node, node_cpus in sorted(cap.numa.items()):
        here = tuple(c for c in node_cpus if c in free)
        if len(here) >= count:
            return here[:count]
        if len(here) > len(best):
            best = here
    # No single node has enough — spill across nodes (still deterministic).
    if len(avail.free_cpus) >= count:
        return tuple(avail.free_cpus[:count])
    return best


def conflicts_for(spec, claims: list[Claim]) -> list[str]:
    """Units that MUST stop for this spec to run: GPU overlap or port
    collision. CPU/RAM pressure never displaces a running model — it fails
    the placement instead (an operator decision, not a silent eviction)."""
    want_gpus = set(spec.gpus) if spec.mode == "gpu" else set()
    out = []
    for c in claims:
        if c.unit == spec.unit:
            out.append(c.unit)  # replacing itself
            continue
        if (want_gpus and (set(c.gpus) & want_gpus)) or (
            c.port is not None and c.port == spec.port
        ):
            out.append(c.unit)
    return out


def check_placement(spec, cap: Capacity, claims: list[Claim]) -> str | None:
    """Can this spec be placed? Returns an operator-readable error or None.

    Conflicting units (which the launcher will stop) are excluded from the
    availability calculation — otherwise replacing a model would always
    look like it doesn't fit.
    """
    freeing = set(conflicts_for(spec, claims))
    avail = availability(cap, claims, exclude=freeing)
    if spec.mode == "gpu":
        missing = [g for g in spec.gpus if g not in cap.gpus]
        if missing:
            return f"profile requests GPUs {missing} that don't exist (have {list(cap.gpus)})"
    if spec.cpus and len(avail.free_cpus) < spec.cpus:
        return (
            f"needs {spec.cpus} CPUs, only {len(avail.free_cpus)} free "
            f"(reserved {RESERVED_CPUS} for the system)"
        )
    # The ACCOUNTED figure, not the cgroup cap: a replica sharing a weight
    # store is capped for its load peak but only occupies its resting size.
    # Using the cap here refuses replicas the machine demonstrably fits.
    want_ram = spec.accounted_ram_gib
    if want_ram and want_ram > avail.free_ram_gib:
        return (
            f"needs {want_ram:.0f}GiB RAM, only {avail.free_ram_gib:.0f}GiB free "
            f"(reserved {RESERVED_RAM_GIB:.0f}GiB for the system)"
        )
    if spec.port in avail.used_ports:
        return f"port {spec.port} already held by another unit"
    return None
