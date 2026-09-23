"""Where a deployment's host RAM actually goes.

Layerwise offload keeps model weights in host RAM, which makes host RAM — not
VRAM — the limit on how many replicas fit. Before changing anything, this
answers the only question that matters: is that memory WEIGHTS (identical
across replicas of a model, therefore shareable) or per-process overhead
(never shareable)?

Predicted per-rank weight bytes come from the checkpoint's own safetensors
headers; measured bytes come from the live processes. The gap is the answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from . import config, remote
from .profiles import RunSpec

# Components whose weights a tensor-parallel launch splits across ranks. The
# rest are replicated in full on every rank.
SHARDED = ("transformer",)


@dataclass(frozen=True)
class Component:
    name: str
    total_gib: float
    per_rank_gib: float
    sharded: bool
    dtypes: dict[str, int]


@dataclass(frozen=True)
class ProcMemory:
    pid: int
    label: str
    rss_gib: float
    pss_gib: float  # << rss means pages are shared with other processes
    private_gib: float

    @property
    def shared_gib(self) -> float:
        return max(self.rss_gib - self.private_gib, 0.0)


@dataclass(frozen=True)
class Profile:
    components: tuple[Component, ...]
    procs: tuple[ProcMemory, ...]
    tp: int

    @property
    def predicted_per_rank_gib(self) -> float:
        return sum(c.per_rank_gib for c in self.components)

    @property
    def measured_per_rank_gib(self) -> float:
        """Largest worker RSS.

        Matched on the engine's worker prefix, NOT on "scheduler": /proc comm
        is capped at 15 characters, so `sgl_diffusion::scheduler_TP0` arrives
        truncated to `sgl_diffusion::` and a match on the suffix silently
        finds nothing.
        """
        ranks = [p for p in self.procs if p.label.startswith("sgl_diffusion")]
        return max((p.rss_gib for p in ranks), default=0.0)

    @property
    def unexplained_gib(self) -> float:
        """Measured minus predicted: overhead that sharing weights cannot fix."""
        return self.measured_per_rank_gib - self.predicted_per_rank_gib


def _weight_sizes(spec: RunSpec) -> dict:
    from . import doctor

    script = doctor.push_script("weight_sizes.py")
    root = spec.model_path + (f"/{spec.subfolder}" if spec.subfolder else "")
    r = remote.run(f"{config.venv_python(spec.engine)} {script} {root}", check=False)
    line = next((ln for ln in r.stdout.splitlines() if ln.startswith("{")), "")
    return json.loads(line) if line else {"components": {}}


def _process_memory(spec: RunSpec) -> list[ProcMemory]:
    """RSS/PSS per worker of this deployment.

    PSS divides each shared page by the number of processes mapping it, so
    PSS ≈ RSS proves nothing is shared and PSS ≈ RSS/N proves it is.
    """
    script = f"""
for p in $(pgrep -f 'sgl_diffusion|sglang' 2>/dev/null); do
  cg=$(cat /proc/$p/cgroup 2>/dev/null | tr -d '\\n')
  # Match the FULL unit name including .service: "deepme-llm-h3-video" is a
  # prefix of "deepme-llm-h3-video-r1", so a substring match silently mixes
  # a replica's processes into its neighbour's measurement.
  case "$cg" in *{spec.unit}.service*) ;; *) continue ;; esac
  lbl=$(tr -d '\\000' < /proc/$p/comm 2>/dev/null)
  rss=$(awk '/^Rss:/{{print $2}}' /proc/$p/smaps_rollup 2>/dev/null)
  pss=$(awk '/^Pss:/{{print $2}}' /proc/$p/smaps_rollup 2>/dev/null)
  prv=$(awk '/^Private_Dirty:|^Private_Clean:/{{s+=$2}} END{{print s}}' /proc/$p/smaps_rollup 2>/dev/null)
  [ -n "$rss" ] && echo "$p|$lbl|$rss|$pss|$prv"
done
"""
    out = []
    for line in remote.run_script(script, check=False).stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) != 5 or not parts[2]:
            continue
        pid, label, rss, pss, prv = parts
        out.append(
            ProcMemory(
                pid=int(pid),
                label=label,
                rss_gib=int(rss) / 1024**2,
                pss_gib=int(pss or 0) / 1024**2,
                private_gib=int(prv or 0) / 1024**2,
            )
        )
    return sorted(out, key=lambda p: -p.rss_gib)


def profile(spec: RunSpec) -> Profile:
    raw = _weight_sizes(spec)
    tp = max(spec.tp, 1)
    comps = []
    for name, info in sorted(raw.get("components", {}).items()):
        total = info["bytes"] / 1024**3
        sharded = any(name.startswith(s) for s in SHARDED)
        comps.append(
            Component(
                name=name if name != "." else "(root)",
                total_gib=total,
                per_rank_gib=total / tp if sharded else total,
                sharded=sharded,
                dtypes=info.get("dtypes", {}),
            )
        )
    return Profile(components=tuple(comps), procs=tuple(_process_memory(spec)), tp=tp)
