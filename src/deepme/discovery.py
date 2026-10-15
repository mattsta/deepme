"""Service discovery: one derived answer to "what is here, and what can it do".

Three facts had been living in three places — ports.toml says what is
DECLARED, the allocator says what is RUNNING, and profiles say what each
deployment is CAPABLE of — and every consumer stitched them together itself.
That is how hand-maintained service maps begin: each one drifts separately,
and the drift only shows up as a request going somewhere that no longer
exists.

So the catalog is computed from those three sources, and the distinction
between them is preserved rather than flattened. "Declared" is not "running"
and "running" is not "healthy"; conflating them produces the most expensive
kind of wrong answer, where configuration is mistaken for reality.

Consumed by internal services over HTTP (the gateway serves it) and by
operators through `deepme discover`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from . import balancerctl, config, ports, profiles, resources

# What a caller may ask for, read from the profile that enforces it rather
# than restated here — a limit documented in two places is a limit that will
# disagree with itself.
LIMIT_FIELDS = ("min_steps",)


@dataclass(frozen=True)
class Endpoint:
    """One addressable thing, and how sure we are that it exists."""

    name: str
    kind: str  # gateway | balancer | replica | service
    url: str
    declared: bool  # present in the port registry
    running: bool  # a unit currently claims it
    profile: str = ""

    @property
    def state(self) -> str:
        if self.running:
            return "running"
        return "declared" if self.declared else "unknown"


@dataclass(frozen=True)
class ModelEntry:
    """A model as a CALLER sees it: one name, the tasks it can serve now."""

    model: str
    tasks: tuple[str, ...]
    profiles: tuple[str, ...]
    replicas: int
    balancers: tuple[str, ...]
    limits: dict = field(default_factory=dict)
    generate_defaults: dict = field(default_factory=dict)

    @property
    def available(self) -> bool:
        return self.replicas > 0


def _public_model(spec) -> str:
    return spec.repo.rsplit("/", 1)[-1].lower()


def endpoints() -> list[Endpoint]:
    """Every address the platform knows about, declared and/or running."""
    registry = ports.load()
    claims = resources.current_claims()
    running_ports = {c.port for c in claims if c.port}
    units = resources.active_units()
    host = config.SSH_HOST

    # Services carry no resource claim, so their unit is the only evidence.
    # Matching on claims alone reported the gateway as merely "declared" while
    # it was serving — the catalog denying that its own entry point was up.
    out = [
        Endpoint(
            name=name,
            kind="service",
            url=f"http://{host}:{port}",
            declared=True,
            running=port in running_ports or f"deepme-{name}" in units,
        )
        for name, port in sorted(registry.services.items())
    ]
    for profile, replica_ports in sorted(registry.replicas.items()):
        for i, port in enumerate(replica_ports):
            out.append(
                Endpoint(
                    name=f"{profile}/replica-{i}",
                    kind="replica",
                    url=f"http://{host}:{port}",
                    declared=True,
                    running=port in running_ports,
                    profile=profile,
                )
            )
        balancer = registry.balancers.get(profile)
        if balancer is not None:
            out.append(
                Endpoint(
                    name=f"{profile}/balancer",
                    kind="balancer",
                    url=f"http://{host}:{balancer}",
                    declared=True,
                    # Its OWN unit, not its fleet's. Inferring liveness from the
                    # replicas reported a dead balancer as running whenever the
                    # replicas behind it were fine — hiding exactly the failure a
                    # caller would hit, since every request goes through it.
                    running=balancerctl.unit_for(profile) in units,
                    profile=profile,
                )
            )
    return out


def models() -> list[ModelEntry]:
    """Models as capabilities, merged across the profiles that implement them.

    A model's tasks can be split across checkpoints (H3: FL2VA serves
    t2va/fl2va, Ref2VA serves ref2va), so the caller-facing entry is the UNION
    of what is currently deployed — which is why this is derived from running
    claims rather than from the profile list on disk.
    """
    registry = ports.load()
    claims = resources.current_claims()
    live: dict[str, int] = {}
    for claim in claims:
        if claim.profile:
            live[claim.profile] = live.get(claim.profile, 0) + 1

    merged: dict[str, dict] = {}
    for profile in sorted(live):
        try:
            spec = profiles.load(profile)
        except Exception:
            continue
        if not spec.tasks:
            continue  # not a caller-facing generation model
        entry = merged.setdefault(
            _public_model(spec),
            {
                "tasks": set(),
                "profiles": [],
                "replicas": 0,
                "balancers": [],
                "limits": {},
                "defaults": {},
            },
        )
        entry["tasks"].update(spec.tasks)
        entry["profiles"].append(profile)
        entry["replicas"] += live[profile]
        with_balancer = registry.balancers.get(profile)
        if with_balancer:
            entry["balancers"].append(f"http://{config.SSH_HOST}:{with_balancer}")
        for name in LIMIT_FIELDS:
            value = getattr(spec, name, None)
            if value:
                entry["limits"][name] = value
        entry["defaults"].update(spec.gen_defaults)

    return [
        ModelEntry(
            model=name,
            tasks=tuple(sorted(v["tasks"])),
            profiles=tuple(v["profiles"]),
            replicas=v["replicas"],
            balancers=tuple(v["balancers"]),
            limits=v["limits"],
            generate_defaults=v["defaults"],
        )
        for name, v in sorted(merged.items())
    ]


def catalog() -> dict:
    """The whole picture, in the shape internal services consume."""
    eps = endpoints()
    return {
        "host": config.SSH_HOST,
        "models": [asdict(m) for m in models()],
        "endpoints": [asdict(e) | {"state": e.state} for e in eps],
        "summary": {
            "models": len(models()),
            "running_endpoints": sum(1 for e in eps if e.running),
            "declared_endpoints": len(eps),
        },
    }
