"""Declared port registry — the single place ports are assigned.

Every port used to be derived: replica N at base + N*stride, a balancer at
profile port + 1000. Derivation looks tidy and fails silently, because
nothing checks the derived values against each other. It had already broken
here: a diffusion server binds its port AND the next one, so h3-video's first
replica occupies 8030-8031, and h3-ref's declared port was 8031. The 3+1
fleet would have collided the first time it deployed, and the symptom would
have been a replica quietly relocating (or refusing to start) rather than
anything naming the real cause.

So ports are declared, not computed, and the registry is validated: every
allocation states how many consecutive ports it consumes, and overlaps are an
error you get from `deepme ports check` instead of from a confusing outage.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import config

REGISTRY = config.REPO_ROOT / "deploy" / "ports.toml"

# A sglang diffusion server binds its API port and the next one. Anything
# claiming a port must therefore reserve a SPAN, or neighbours collide.
DIFFUSION_SPAN = 2
DEFAULT_SPAN = 1


class PortsError(Exception):
    """The registry cannot answer a port question.

    An ordinary exception, never SystemExit: this is reached from BATCH
    WORKER THREADS (progress ticks resolve a job's replica through the
    registry). A BaseException there passes through every `except
    Exception` guard, aborts the pool from a worker, and leaves the
    remaining generations running headless — and callers that survey many
    profiles have to be able to skip one that lacks a balancer port.
    """


@dataclass(frozen=True)
class Allocation:
    """One reserved range and who holds it."""

    owner: str  # "service:gateway" or "h3-video/replica-0"
    port: int
    span: int = DEFAULT_SPAN

    @property
    def last(self) -> int:
        return self.port + self.span - 1

    def overlaps(self, other: Allocation) -> bool:
        return self.port <= other.last and other.port <= self.last


@dataclass(frozen=True)
class Registry:
    services: dict[str, int]
    replicas: dict[str, list[int]]
    balancers: dict[str, int]
    spans: dict[str, int]
    # A distributed engine also binds PRIVATE ports (torch.distributed
    # rendezvous, an internal scheduler). Those defaulted to fixed values, so
    # replica 0 of every profile wanted the same ones — h3-video and h3-ref
    # collided on scheduler 5644 the first time they ran together.
    masters: dict[str, list[int]]
    schedulers: dict[str, list[int]]

    def _indexed(self, table: dict[str, list[int]], kind: str, profile: str, replica: int) -> int:
        ports = table.get(profile)
        if not ports:
            raise PortsError(f"no {kind} ports declared for {profile!r} in {REGISTRY}")
        if replica >= len(ports):
            raise PortsError(
                f"{profile!r} declares {len(ports)} {kind} port(s) but replica "
                f"{replica} was requested — add more in {REGISTRY}."
            )
        return ports[replica]

    def master_port(self, profile: str, replica: int) -> int:
        return self._indexed(self.masters, "master", profile, replica)

    def scheduler_port(self, profile: str, replica: int) -> int:
        return self._indexed(self.schedulers, "scheduler", profile, replica)

    def replica_ports(self, profile: str) -> list[int]:
        ports = self.replicas.get(profile)
        if not ports:
            raise PortsError(
                f"no replica ports declared for {profile!r} in {REGISTRY}. "
                f"Add a [profiles.{profile}] section with `replicas = [...]`."
            )
        return list(ports)

    def replica_port(self, profile: str, replica: int) -> int:
        ports = self.replica_ports(profile)
        if replica >= len(ports):
            raise PortsError(
                f"{profile!r} declares {len(ports)} replica port(s) but replica "
                f"{replica} was requested — add more to [profiles.{profile}] "
                f"in {REGISTRY}."
            )
        return ports[replica]

    def balancer_port(self, profile: str) -> int:
        port = self.balancers.get(profile)
        if port is None:
            raise PortsError(f"no balancer port declared for {profile!r} in {REGISTRY}")
        return port

    def service_port(self, name: str) -> int:
        port = self.services.get(name)
        if port is None:
            raise PortsError(f"no port declared for service {name!r} in {REGISTRY}")
        return port

    def span_for(self, profile: str) -> int:
        return self.spans.get(profile, DEFAULT_SPAN)

    def allocations(self) -> list[Allocation]:
        out = [
            Allocation(owner=f"service:{name}", port=port) for name, port in self.services.items()
        ]
        for profile, ports in self.replicas.items():
            span = self.span_for(profile)
            out.extend(
                Allocation(owner=f"{profile}/replica-{i}", port=port, span=span)
                for i, port in enumerate(ports)
            )
        out.extend(
            Allocation(owner=f"{profile}/balancer", port=port)
            for profile, port in self.balancers.items()
        )
        for kind, table in (("master", self.masters), ("scheduler", self.schedulers)):
            for profile, ports in table.items():
                out.extend(
                    Allocation(owner=f"{profile}/{kind}-{i}", port=port)
                    for i, port in enumerate(ports)
                )
        return sorted(out, key=lambda a: a.port)


def load(path: Path | None = None) -> Registry:
    source = path or REGISTRY
    if not source.exists():
        raise PortsError(f"missing port registry: {source}")
    raw = tomllib.loads(source.read_text())
    profiles_raw = raw.get("profiles", {})
    return Registry(
        services={k: int(v) for k, v in raw.get("services", {}).items()},
        replicas={
            name: [int(p) for p in cfg.get("replicas", [])] for name, cfg in profiles_raw.items()
        },
        balancers={
            name: int(cfg["balancer"]) for name, cfg in profiles_raw.items() if "balancer" in cfg
        },
        spans={name: int(cfg.get("span", DEFAULT_SPAN)) for name, cfg in profiles_raw.items()},
        masters={
            name: [int(p) for p in cfg.get("master", [])] for name, cfg in profiles_raw.items()
        },
        schedulers={
            name: [int(p) for p in cfg.get("scheduler", [])] for name, cfg in profiles_raw.items()
        },
    )


def conflicts(registry: Registry) -> list[str]:
    """Every overlapping pair, described in full.

    Spans are what make this necessary: two allocations can conflict without
    sharing a number, because one of them quietly consumes its neighbour.
    """
    allocations = registry.allocations()
    out = []
    for i, a in enumerate(allocations):
        for b in allocations[i + 1 :]:
            if b.port > a.last:
                break  # sorted, so nothing further can overlap
            if a.overlaps(b):
                out.append(
                    f"{a.owner} holds {a.port}-{a.last} and {b.owner} holds "
                    f"{b.port}-{b.last}"
                    + (
                        " (a diffusion server also binds the NEXT port)"
                        if a.span > 1 or b.span > 1
                        else ""
                    )
                )
    return out
