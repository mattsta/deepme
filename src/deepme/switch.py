"""Fleet resize orchestration — the platform's convergence engine.

Extracted from the CLI so the headline invariant ("a resize converges from
any intermediate state") is TESTABLE: every side effect goes through a
SwitchOps port, so the drain rules, the reconcile pass, orphan cleanup and
the repoint-vs-restart decision all run against fakes in milliseconds.
The command in cli.py is a thin shell: build the real ops, call this.

The shape of the algorithm:
  0. plan the target, diff it against LIVE unit claims (never memory)
  1. drain leaving replicas (abort if a drain cannot even be issued)
  2. wait for their jobs, stop them, detach them
  3. start arrivals, wait healthy
  4. RECONCILE every target balancer's membership (attach missing, undrain
     leftovers, detach strays) — this is what makes re-runs converge
  4b. stop orphan balancers (derived from what RUNS, not from this delta)
  5. re-point the gateway — in place when it supports reload
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import fleet as fleet_mod


@dataclass
class SwitchOps:
    """Every side effect a switch performs, as an injectable port."""

    current_claims: Callable[[], list]
    active_units: Callable[[], set]
    # balancer control
    queue: Callable[[str], dict]
    stats: Callable[[str], dict]
    drain: Callable[[str, str], dict]
    undrain: Callable[[str, str], dict]
    wait_drained: Callable[..., bool]
    detach: Callable[[str, str], dict]
    attach: Callable[[str, str], dict]
    start_balancer: Callable[[list], str]
    stop_balancer: Callable[[str], None]
    # replica control
    stop_units: Callable[[list], None]
    start_replica: Callable[[Any], list]
    wait_healthy: Callable[..., tuple]
    # gateway control
    discover_routes: Callable[[], list]
    gateway_routes: Callable[[], dict]
    repoint: Callable[[list], bool]
    gateway_start: Callable[[list], str]
    # reporting
    say: Callable[[str], None] = field(default=lambda msg: None)
    stamp: Callable[[], str] = field(default=lambda: "")


def default_ops(say: Callable[[str], None], stamp: Callable[[], str]) -> SwitchOps:
    from . import balancerctl, gatewayctl, resources, serve

    return SwitchOps(
        current_claims=resources.current_claims,
        active_units=resources.active_units,
        queue=balancerctl.queue,
        stats=balancerctl.stats,
        drain=balancerctl.drain,
        undrain=balancerctl.undrain,
        wait_drained=balancerctl.wait_drained,
        detach=balancerctl.detach,
        attach=balancerctl.attach,
        start_balancer=balancerctl.start,
        stop_balancer=balancerctl.stop,
        stop_units=serve.stop_units,
        start_replica=serve.start,
        wait_healthy=serve.wait_healthy,
        discover_routes=gatewayctl.discover_routes,
        gateway_routes=lambda: gatewayctl.routes(),
        repoint=gatewayctl.repoint,
        gateway_start=gatewayctl.start,
        say=say,
        stamp=stamp,
    )


def execute_switch(
    name: str,
    *,
    timeout: float,
    drain_timeout: float,
    force: bool,
    audit: dict,
    ops: SwitchOps,
) -> None:
    say, stamp = ops.say, ops.stamp
    f = fleet_mod.load(name)
    claims = ops.current_claims()
    # Every running model profile is displaceable — the delta below decides
    # which actually leave. Without this, a mix that omits a profile could
    # never reclaim its GPUs.
    target = fleet_mod.plan(f, releasing_extra={c.profile for c in claims if c.profile})
    d = fleet_mod.delta(target, claims)
    profile_of = {c.unit: c.profile for c in claims}
    audit.update(
        kept=[s.unit for s in d.keep],
        added=[s.unit for s in d.add],
        removed=list(d.remove),
        forced=force,
    )

    say(f"[bold]{f.name}[/bold] — delta against the running fleet:")
    say(f"  keep    {len(d.keep)}: {', '.join(s.unit for s in d.keep) or '-'}")
    say(f"  add     {len(d.add)}: {', '.join(s.unit for s in d.add) or '-'}")
    say(f"  remove  {len(d.remove)}: {', '.join(d.remove) or '-'}")
    if d.is_noop:
        say("[green]already in this shape[/green] — reconciling front ends only")

    # A profile leaving ENTIRELY takes its queue with it. Tickets that can
    # never be served must be an explicit operator decision, not a side
    # effect of a resize.
    target_profiles = {s.profile for s in target}
    leaving = {
        profile_of[u]
        for u in d.remove
        if profile_of.get(u) and profile_of[u] not in target_profiles
    }
    for profile in sorted(leaving):
        with contextlib.suppress(Exception):
            waiting = int(ops.queue(profile).get("waiting") or 0)
            if waiting and not force:
                raise SystemExit(
                    f"{profile} is leaving but still has {waiting} queued "
                    f"ticket(s) that no remaining fleet can serve — cancel "
                    f"them, let them drain, or pass --force to abandon them"
                )

    # 1. Drain leaving replicas: out of rotation now, jobs run to completion.
    # A drain that cannot be ISSUED aborts the switch: proceeding to stop an
    # undrained replica kills whatever it is generating, and "the balancer
    # was down so we could not ask" is a reason to stop, not to guess.
    for unit in d.remove:
        prof, port = profile_of.get(unit), d.remove_ports.get(unit)
        if not prof or not port:
            continue
        url = f"http://127.0.0.1:{port}"
        try:
            ops.drain(prof, url)
            say(f"[dim]{stamp()} draining {unit}[/dim]")
        except Exception as exc:
            if not force:
                raise SystemExit(
                    f"cannot drain {unit}: {prof}'s balancer did not answer "
                    f"({exc}). Fix it (`deepme balancer start -m {prof}` "
                    f"re-freezes, dropping its queue) or pass --force to "
                    f"stop the replica WITHOUT draining."
                ) from exc
            say(f"[yellow]{stamp()} {unit}: drain skipped (--force), stopping without it[/yellow]")
    for unit in d.remove:
        prof, port = profile_of.get(unit), d.remove_ports.get(unit)
        if not prof or not port:
            continue
        if (
            not ops.wait_drained(
                prof,
                f"http://127.0.0.1:{port}",
                timeout_s=drain_timeout,
                on_tick=lambda inflight, u=unit: say(
                    f"[dim]{stamp()} {u}: {inflight} job(s) still running, waiting[/dim]"
                ),
            )
            and not force
        ):
            raise SystemExit(
                f"{unit} still has work in flight (or its balancer is "
                f"unreachable) after {drain_timeout:.0f}s — retry later, "
                f"raise --drain-timeout, or --force to stop it anyway"
            )

    # 2. Stop them (GPUs release), detach them from their balancer.
    if d.remove:
        say(f"{stamp()} stopping: {', '.join(d.remove)}")
        ops.stop_units(list(d.remove))
        for unit in d.remove:
            prof, port = profile_of.get(unit), d.remove_ports.get(unit)
            if prof and port and prof in target_profiles:
                with contextlib.suppress(Exception):
                    ops.detach(prof, f"http://127.0.0.1:{port}")
    for profile in sorted(leaving):
        # Anything still waiting here dies with the balancer (--force is the
        # only way to reach this with a non-empty queue). Name the
        # casualties — "which tickets did we abandon" must be answerable.
        with contextlib.suppress(Exception):
            abandoned = [t["ticket"] for t in ops.queue(profile).get("next", [])]
            if abandoned:
                audit["abandoned_tickets"].extend(abandoned)
                say(
                    f"[yellow]{stamp()} abandoning {len(abandoned)} queued "
                    f"ticket(s) on {profile}: {', '.join(abandoned)}[/yellow]"
                )
        ops.stop_balancer(profile)
        say(f"[dim]{stamp()} stopped {profile}'s balancer[/dim]")

    # 3. Start arrivals. A profile with kept replicas already has its shared
    # weight store, so its arrivals go in parallel; a NEW profile's first
    # replica builds the store and goes alone.
    kept_profiles = {s.profile for s in d.keep}
    firsts: dict[str, Any] = {}
    rest: list[Any] = []
    for spec in d.add:
        if spec.profile not in kept_profiles and spec.profile not in firsts:
            firsts[spec.profile] = spec
        else:
            rest.append(spec)
    for spec in firsts.values():
        say(f"[dim]{stamp()} starting {spec.unit} (builds its store)[/dim]")
        ops.start_replica(spec)
        ok, detail = ops.wait_healthy(spec, timeout_s=timeout)
        if not ok:
            raise SystemExit(
                f"{spec.unit}: {detail} — diagnose with `deepme serve diag -m {spec.profile}`"
            )
        say(f"{stamp()} {spec.unit}: [green]healthy[/green]")
    for spec in rest:
        say(f"[dim]{stamp()} starting {spec.unit}[/dim]")
        ops.start_replica(spec)
    if rest:
        import concurrent.futures as _f

        with _f.ThreadPoolExecutor(max_workers=len(rest)) as pool:
            for fut, spec in [(pool.submit(ops.wait_healthy, s, timeout), s) for s in rest]:
                ok, detail = fut.result()
                if not ok:
                    raise SystemExit(
                        f"{spec.unit}: {detail} — diagnose with "
                        f"`deepme serve diag -m {spec.profile}`"
                    )
                say(f"{stamp()} {spec.unit}: [green]healthy[/green]")

    # 4. RECONCILE every target balancer against the target membership —
    # not just "attach the arrivals". This is what makes a re-run converge
    # from EVERY intermediate state: an arrival whose attach failed last
    # time is now a "keep" and would otherwise never be attached; a stopped
    # replica whose detach was interrupted stays in rotation forever; a
    # drain left over from an aborted switch starves its replica silently.
    # Claims describe units; balancer membership is a second, independent
    # state store, and only diffing it here closes the gap between them.
    by_profile: dict[str, list] = {}
    for spec in target:
        by_profile.setdefault(spec.profile, []).append(spec)
    for profile, specs in by_profile.items():
        wanted_urls = {f"http://127.0.0.1:{s.port}" for s in specs}
        try:
            snapshot = ops.stats(profile)
        except Exception:
            # No live balancer for this profile: start one fresh. Nothing to
            # preserve — a balancer that is down holds no queue.
            ops.start_balancer(specs)
            say(f"{stamp()} started {profile}'s balancer x{len(specs)}")
            continue
        live = {b["url"]: b for b in snapshot.get("backends", [])}
        for url in sorted(wanted_urls - set(live)):
            out = ops.attach(profile, url)
            say(
                f"{stamp()} attached {url} -> "
                f"{len(out.get('backends', []))} backend(s), queue intact"
            )
        for url, b in live.items():
            if url in wanted_urls and b.get("draining"):
                ops.undrain(profile, url)
                say(f"{stamp()} un-drained {url} (leftover drain)")
        for url in sorted(set(live) - wanted_urls):
            try:
                ops.detach(profile, url)
                say(f"{stamp()} detached stale backend {url}")
            except Exception as exc:
                say(
                    f"[yellow]{stamp()} could not detach {url}: {exc} — it "
                    f"may still be finishing a job; re-run later[/yellow]"
                )

    # 4b. Stop balancers for profiles the target does not serve at all —
    # derived from what is RUNNING, not from this run's d.remove, so an
    # orphan left by an interrupted earlier switch is still cleaned up.
    for unit in sorted(ops.active_units()):
        if not unit.startswith("deepme-balancer-"):
            continue
        prof = unit.removeprefix("deepme-balancer-")
        if prof not in by_profile:
            ops.stop_balancer(prof)
            say(f"[dim]{stamp()} stopped orphan balancer for {prof}[/dim]")

    # 5. Re-point the gateway whenever the fleet changed shape: routes drive
    # routing, but the frozen catalog and replica counts in /gateway/discover
    # go stale on ANY resize, so only a true no-op skips the restart.
    wanted = ops.discover_routes()
    if d.is_noop:
        with contextlib.suppress(Exception):
            current = ops.gateway_routes().get("routes", [])
            same = {
                (r["model"], r["profile"], tuple(sorted(r["tasks"])), r["url"]) for r in current
            } == {(r["model"], r["profile"], tuple(sorted(r["tasks"])), r["url"]) for r in wanted}
            if same:
                say(f"{stamp()} gateway routes unchanged — not touched")
                return
    # Re-point WITHOUT a restart when the live gateway supports it — a
    # restart severs every in-flight SSE stream and MP4 download; a resize
    # must not break unrelated customers' transfers.
    if ops.repoint(wanted):
        say(f"{stamp()} gateway re-pointed in place (no restart)")
    else:
        say(ops.gateway_start(wanted))
