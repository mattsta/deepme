"""deepme — config-driven multi-model LLM deployment CLI.

Model profiles live in deploy/models/*.toml (repo + serving intent only —
weight sizes/shards come from the Hugging Face API). Most commands accept
`-m/--model <profile>`; without it they use the persisted default
(`deepme model use <name>`).

    uv run deepme model list|show|use|add    # manage profiles
    uv run deepme status                     # fleet: GPUs, units, APIs
    uv run deepme setup / gpu-fix            # provision server (idempotent)
    uv run deepme download start|status      # per-model weights (resumable)
    uv run deepme serve start|stop|logs      # per-model systemd service
    uv run deepme test / test-tools / bench  # verify + measure
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import batch as batch_mod
from . import bench as bench_mod
from . import (
    client,
    config,
    download,
    hfmeta,
    media,
    ports,
    profiles,
    provision,
    remote,
    runs,
    serve,
)
from . import gpus as gpus_mod

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
serve_app = typer.Typer(no_args_is_help=True)
download_app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
app.add_typer(serve_app, name="serve", help="Manage inference services (systemd, per model)")
app.add_typer(download_app, name="download", help="Manage model weight downloads")
app.add_typer(config_app, name="config", help="Configure the target server")
app.add_typer(model_app, name="model", help="Manage model profiles")
console = Console()

ModelOpt = typer.Option(None, "--model", "-m", help="Model profile (default: `deepme model use`)")


# ---------------------------------------------------------------------------
# model profiles
# ---------------------------------------------------------------------------


@model_app.command("list")
def model_list() -> None:
    """List profiles; * marks the default."""
    default = profiles.default_profile()
    for name in profiles.list_profiles():
        mark = "*" if name == default else " "
        console.print(f"{mark} {name}")


@model_app.command("show")
def model_show(model: str = ModelOpt) -> None:
    """Show the fully-resolved RunSpec for a profile (incl. HF-fetched facts)."""
    spec = profiles.load(model)
    for k, v in spec.__dict__.items():
        console.print(f"{k} = {v!r}")
    console.print(f"model_path = {spec.model_path!r}")
    console.print(f"unit = {spec.unit!r}   api_base = {spec.api_base!r}")
    err = spec.fit_check()
    console.print(f"fit_check = {'[red]' + err + '[/red]' if err else '[green]ok[/green]'}")


@model_app.command("use")
def model_use(name: str) -> None:
    """Set the default profile for all commands."""
    profiles.set_default_profile(name)
    console.print(f"default model profile: {name}")


@model_app.command("refresh")
def model_refresh(model: str = ModelOpt) -> None:
    """Re-fetch HF metadata (weight size, shard count) for a profile."""
    spec = profiles.load(model)
    meta = hfmeta.get(spec.repo, refresh=True)
    console.print(f"{spec.repo}: {meta.weight_gib:.1f}GiB weights in {meta.weight_shards} shards")


@model_app.command("add")
def model_add(
    repo: str = typer.Argument(..., help="Hugging Face repo, e.g. org/Model-Name"),
    name: str = typer.Option(None, help="Profile name (default: derived from repo)"),
    port: int = typer.Option(8000),
) -> None:
    """Create a profile skeleton for an HF repo (facts auto-fetched;
    engine picked by weight format; context clamped to the model's max)."""
    from . import policy as policy_mod

    pname = name or repo.rsplit("/", 1)[-1].lower()
    path = profiles.PROFILES_DIR / f"{pname}.toml"
    if path.exists():
        raise SystemExit(f"profile {pname!r} already exists: {path}")
    from . import catalog

    meta = hfmeta.get(repo)
    fmt = meta.formats[0] if meta.formats else "safetensors"
    # Proof-chain engine selection: probed format evidence x arch registries,
    # production engines preferred. Never assumed.
    candidates = catalog.candidate_engines(
        set(meta.formats or ["safetensors"]),
        meta.architectures,
        catalog.engine_formats(),
        catalog.engine_archs(),
        kind=meta.kind,
    )
    engine = candidates[0] if candidates else "vllm"
    if not candidates:
        console.print(
            "[yellow]warning: no installed engine proves support for "
            f"kind={meta.kind} formats={list(meta.formats)} "
            f"archs={list(meta.architectures)} — defaulting to vllm; "
            "expect bring-up work[/yellow]"
        )
    model_max = int((meta.config_dims or {}).get("max_position_embeddings") or 0)
    max_len = min(131072, model_max) if model_max else 131072
    size_line = f"{meta.deployable_gib:.0f}GiB deployable, {meta.weight_shards} files"
    variant_block = ""
    if fmt == "gguf" and meta.gguf_variants:
        allowed = policy_mod.current().filter_variants(
            policy_mod.full_model_variants(meta.gguf_variants, meta.total_params)
        )
        auto = min(allowed, key=lambda k: allowed[k]) if allowed else "(none pass policy)"
        inventory = "\n".join(
            f"#   {v}  ({b / 1024**3:.1f}GiB)" for v, b in sorted(meta.gguf_variants.items())
        )
        size_line = f"gguf: {len(meta.gguf_variants)} variants; auto-selected: {auto}"
        variant_block = f'variant = ""   # empty = policy auto-selection; available:\n{inventory}\n'
    path.write_text(f"""\
# Model profile: {repo}
# ({size_line}, per HF API)
[model]
repo = "{repo}"
served_name = "{pname}"
{variant_block}
[serve]
engine = "{engine}"
tp = 8
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
port = {port}
max_len = {max_len}
gpu_mem = 0.90
kv_ram_gb = 0
spec_decode = false

[serve.engine_args]
vllm = []
sglang = []
llamacpp = []

[sampling]
temperature = 1.0
top_p = 1.0
""")
    console.print(f"wrote {path} ({meta.deployable_gib:.1f}GiB deployable, kind={meta.kind})")
    spec = profiles.load(pname)
    err = spec.fit_check()
    console.print(f"fit_check = {'[red]' + err + '[/red]' if err else '[green]ok[/green]'}")


@model_app.command("variants")
def model_variants(model: str = ModelOpt) -> None:
    """Quantization inventory for a profile's repo: per-variant size, policy
    verdict, cluster fit, and which the auto-selection picks. Choose one by
    setting `variant = "..."` in the profile."""
    from . import catalog
    from . import policy as policy_mod

    spec = profiles.load(model)
    meta = hfmeta.get(spec.repo)
    if not meta.gguf_variants:
        console.print(
            f"{spec.repo}: no GGUF variants (formats: {', '.join(meta.formats) or 'none'})"
        )
        return
    pol = policy_mod.current()
    full = policy_mod.full_model_variants(meta.gguf_variants, meta.total_params)
    allowed = pol.filter_variants(full)
    auto = min(allowed, key=lambda k: allowed[k]) if allowed else None
    capacity = catalog.cluster_capacity_gib()
    table = Table(title=f"{spec.repo} — GGUF variants (policy: >= {pol.min_gguf_bits:g}-bit)")
    for col in ("variant", "size", "policy", "fits", "selected"):
        table.add_column(col, overflow="fold")
    for v, b in sorted(meta.gguf_variants.items(), key=lambda kv: kv[1]):
        gib = b / 1024**3
        if v not in full:
            verdict = "[dim]aux/implausible (not a full model)[/dim]"
        else:
            ok, reason = pol.variant_allowed(v)
            verdict = "[green]allowed[/green]" if ok else f"[dim]{reason}[/dim]"
        table.add_row(
            v,
            f"{gib:.1f}GiB",
            verdict,
            "[green]yes[/green]" if gib <= capacity else f"[red]> {capacity:.0f}GiB[/red]",
            "[bold]<- auto[/bold]"
            if v == auto
            else ("[bold]<- profile[/bold]" if v == spec.variant else ""),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# server config / provisioning
# ---------------------------------------------------------------------------


@config_app.command("init")
def config_init(
    host: str = typer.Option(..., help="Server IP or hostname"),
    user: str = typer.Option("ubuntu", help="SSH user"),
) -> None:
    """Set the target server in .deepme/config.toml (gitignored), preserving
    any existing [cluster]/[policy]/[model] sections."""
    config.update_config("server", host=host, user=user)
    console.print(f"wrote server config to {config.CONFIG_FILE}")


@config_app.command("show")
def config_show() -> None:
    src = (
        "env"
        if os.environ.get("DEEPME_HOST")
        else (str(config.CONFIG_FILE) if config.CONFIG_FILE.exists() else "unset")
    )
    console.print(
        f"host={config.SSH_HOST or '(unset)'} user={config.SSH_USER} (source: {src}) "
        f"default_model={profiles.default_profile()}"
    )


@app.command()
def setup(
    engine: list[str] = typer.Option(
        None, "--engine", "-e", help="Engines (default: production; add -e llamacpp for GGUF)"
    ),
    upgrade: bool = typer.Option(
        False,
        "--upgrade",
        help="Re-resolve deps within pyproject ranges and refresh the committed lock "
        "(default: frozen to the validated lock). Bench + validate after.",
    ),
) -> None:
    """Provision server: build tools + engine uv projects (frozen to lock)."""
    from . import catalog

    engines = tuple(engine) if engine else None
    label = engines or "production engines"
    mode = "UPGRADE (re-resolving)" if upgrade else "frozen to committed lock"
    console.print(f"Provisioning {label} on {config.SSH_TARGET} ({mode}, several minutes)...")
    console.print(provision.provision(engines, upgrade=upgrade)[-4000:])
    # Engine capabilities may have changed — invalidate probe caches so the
    # next browse/model-add re-probes the freshly-installed binaries.
    for cache in (catalog.ARCH_CACHE, catalog.FORMATS_CACHE):
        cache.unlink(missing_ok=True)
    console.print("[dim]invalidated engine arch/format probe caches[/dim]")


@app.command("upgrade-preview")
def upgrade_preview(
    engine: str = typer.Option("vllm", "--engine", "-e"),
) -> None:
    """Resolution-only preview of an engine upgrade: what versions a fresh
    resolve would pull (committed lock vs new), with NO install and NO touch
    to the live venv. Safe while production serves."""
    console.print(provision.preview_upgrade(engine))


@app.command("gpu-fix")
def gpu_fix() -> None:
    """Align NVIDIA driver userland + open kernel module; reload if needed."""
    console.print(provision.gpu_stack())
    runs.record("gpu-fix", None, outcome="ok")


@app.command("engines")
def engines_matrix() -> None:
    """Engine capability matrix: formats + declared feature flags."""
    from . import catalog
    from . import engines as engines_mod

    probed = catalog.engine_formats()
    table = Table(
        title="engine capability matrix (declared: engines.py; "
        "probed: evidence from installed binaries)"
    )
    for col in ("engine", "declared formats", "probed formats", "effective", "status", "features"):
        table.add_column(col, overflow="fold")
    for e in engines_mod.REGISTRY.values():
        eff = catalog.effective_formats(e.name, probed)
        table.add_row(
            e.name,
            ",".join(e.formats),
            ",".join(probed.get(e.name) or []) or "[dim]no evidence[/dim]",
            ",".join(sorted(eff)),
            "experimental" if e.experimental else "production",
            ", ".join(e.features.names()) or "-",
        )
    console.print(table)


@app.command()
def versions() -> None:
    """Show installed engine/toolchain versions on the server."""
    console.print(provision.versions())


@app.command()
def reboot(
    force: bool = typer.Option(False, "--force", help="Reboot even with generations running"),
) -> None:
    """Reboot the server and wait until SSH is back (last resort).

    The most destructive command carries the strongest guard: it refuses
    while any generation runs (a reboot kills them all mid-denoise), asks
    for confirmation, and lands in the runs ledger like every mutation.
    """
    import time as _time

    if not force:
        fleets = None
        with contextlib.suppress(Exception):
            from . import gatewayctl

            fleets = gatewayctl.capacity().get("fleets", [])
        # An UNREACHABLE fleet reports busy=0 — it is "no answer", not "no
        # work". Summing it as zero let the most destructive command
        # conclude the box was idle and reboot into live generations. The
        # safe reading of an unanswered question is to refuse.
        blind = [f for f in (fleets or []) if not f.get("reachable", True)]
        if fleets is None:
            raise SystemExit(
                "cannot reach the gateway to check for running generations — "
                "a reboot would kill any mid-denoise. Check `deepme gateway "
                "status`, or pass --force to reboot anyway."
            )
        if blind:
            names = ", ".join(str(f.get("profile") or "?") for f in blind)
            raise SystemExit(
                f"{len(blind)} fleet(s) could not be asked what they are "
                f"running ({names}) — a reboot kills anything mid-denoise. "
                f"Retry, or pass --force to reboot anyway."
            )
        busy = sum(int(f.get("busy") or 0) for f in fleets)
        if busy:
            raise SystemExit(
                f"{busy} generation(s) running — a reboot kills them "
                f"mid-denoise. Wait for them (`deepme capacity --watch`) "
                f"or pass --force to reboot anyway."
            )
    if not typer.confirm("Reboot the server now?", default=False):
        raise SystemExit("aborted")
    runs.record("reboot", None, outcome="began", detail={"forced": force})
    remote.run("sudo systemctl reboot", check=False, timeout=30)
    console.print("rebooting; waiting for SSH ...")
    _time.sleep(20)
    for _ in range(60):
        if remote.run("echo up", check=False, timeout=10).ok:
            console.print("server is back")
            return
        _time.sleep(10)
    console.print("[red]server did not come back within 10 min[/red]")
    raise typer.Exit(1)


# ---------------------------------------------------------------------------
# status / GPUs
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Fleet view: GPUs, disk, every deepme unit, and API health per profile."""
    gpu = remote.run(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader",
        check=False,
    ).stdout.strip()
    disk = remote.run("df -h / | tail -1", check=False).stdout.strip()
    console.print("[bold]GPUs[/bold] (idx, util, mem used/total, temp)")
    console.print(gpu or "[red]unreachable[/red]")
    console.print(f"[bold]Disk[/bold] {disk}")
    console.print(f"[bold]Units[/bold]\n{serve.fleet_status()}")
    from . import guardctl

    g_ok, g_detail = guardctl.health()
    console.print(
        f"[bold]Guard[/bold] {guardctl.UNIT}: {guardctl.status()}   "
        f"http={'[green]OK[/green]' if g_ok else f'[red]{g_detail}[/red]'} "
        f"({guardctl.api_base()}/v1)"
    )
    for name in profiles.list_profiles():
        # Resolving a full RunSpec fetches HF metadata; a cache miss while
        # offline or a gated repo must not crash the whole fleet view.
        try:
            spec = profiles.load(name)
        except Exception as exc:
            console.print(
                f"[bold]{name}[/bold] [dim](metadata unavailable: {type(exc).__name__})[/dim]"
            )
            continue
        ok, detail = serve.health(spec)
        health_txt = "[green]healthy[/green]" if ok else f"[red]down[/red] ({detail})"
        console.print(f"[bold]{name}[/bold] {spec.api_base}/v1 ({spec.served_name}): {health_txt}")


@app.command()
def watch(
    interval: float = typer.Option(60.0, "--interval", "-i", help="Seconds between samples"),
    cycles: int = typer.Option(0, help="Stop after N samples (0 = run until interrupted)"),
) -> None:
    """Watch every deployment and record up/down transitions to the ledger."""
    from . import health as health_mod

    def report(sample, changes, first) -> None:
        stamp = time.strftime("%H:%M:%S")
        if first:
            for h in sample:
                mark = "[green]healthy[/green]" if h.healthy else f"[red]{h.detail}[/red]"
                console.print(f"{stamp} {h.profile}: {mark} (restarts={h.restarts})")
            console.print("[dim]watching — transitions only from here[/dim]")
            return
        for h, was in changes:
            colour = "green" if h.healthy else "red"
            console.print(
                f"{stamp} [{colour}]{h.profile}[/{colour}] {was} -> {h.state}  {h.detail}"
            )

    try:
        health_mod.watch(interval, cycles=cycles, on_event=report)
    except KeyboardInterrupt:
        console.print("[dim]stopped[/dim]")
        # 130 like every interrupted command — scripts must be able to tell
        # "watch finished its cycles" from "someone stopped it".
        raise SystemExit(130) from None


balancer_app = typer.Typer(help="Load balancer across replicas of one model")
app.add_typer(balancer_app, name="balancer")


@balancer_app.command("start")
def balancer_start(
    model: str = ModelOpt,
    setup: bool = typer.Option(False, "--setup", help="Sync the balancer venv first"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Restart even if the live balancer holds queued tickets or "
        "running jobs (DESTROYS them)",
    ),
) -> None:
    """Route one API to every running replica of this model."""
    from . import balancerctl
    from . import resources as res

    spec = profiles.load(model)
    claims = [c for c in res.current_claims() if c.profile == spec.profile]
    if not claims:
        raise SystemExit(
            f"no running replicas of {spec.profile!r} — start them first: "
            f"uv run deepme serve scale -m {spec.profile} -n <count>"
        )
    if setup:
        console.print(balancerctl.setup())
    # Address replicas by the ports they actually hold, not the profile's.
    specs = [
        profiles.load(model, port=c.port)
        for c in sorted(claims, key=lambda c: c.port or 0)
        if c.port
    ]
    console.print(balancerctl.start(specs, force=force))
    # A (re)start freezes membership and, with --force, destroys a queue —
    # the fleet mutation most likely to be asked about afterwards.
    base = balancerctl.api_base(spec.profile)
    # PROVE it came up. A green banner printed straight after systemctl
    # reported success for a service that was crash-looping on a bad
    # environment — the operator only found out when work stopped moving.
    healthy = _await_service(
        lambda: balancerctl.stats(spec.profile),
        what=f"balancer for {spec.profile}",
    )
    runs.record(
        "balancer-start",
        spec,
        outcome="ok" if healthy else "failed",
        detail={"replicas": len(specs), "forced": force, "healthy": healthy},
    )
    if not healthy:
        raise SystemExit(
            f"balancer unit started but {base}/balancer/stats never answered — "
            f"read `deepme balancer logs -m {spec.profile}`"
        )
    console.print(
        f"[green]balancer[/green] {base}/v1 -> "
        f"{len(specs)} replica(s); stats at {base}/balancer/stats"
    )


def _await_service(probe, *, what: str, tries: int = 12, every: float = 2.5) -> bool:
    """Poll a service's own endpoint until it answers. True if it did.

    `systemctl restart` returning 0 means the unit was FORKED, not that the
    process is serving — and several commands treated the two as the same
    thing, reporting success for services that never came up. Boot work
    (occupancy recovery) makes the first seconds legitimately silent, so
    this waits rather than probing once.
    """
    import time as _time

    for attempt in range(tries):
        with contextlib.suppress(Exception):
            probe()
            return True
        if attempt == 0:
            console.print(f"[dim]waiting for {what} to answer…[/dim]")
        _time.sleep(every)
    return False


@balancer_app.command("stop")
def balancer_stop(model: str = ModelOpt) -> None:
    """Stop one model's load balancer (its replicas keep running)."""
    from . import balancerctl

    spec = profiles.load(model)
    balancerctl.stop(spec.profile)
    runs.record("balancer-stop", spec, outcome="ok")
    console.print(f"stopped {balancerctl.unit_for(spec.profile)}")


@balancer_app.command("audit")
def balancer_audit(
    model: str = ModelOpt,
    lines: int = typer.Option(200, "-n", help="Most recent entries"),
    grep: str = typer.Option(
        "", "--grep", help="Only entries containing this text (e.g. a job id)"
    ),
) -> None:
    """The platform's persistent lifecycle trail: queued, handover, started,
    terminal, fetch, reclaimed, retention, owner_rejected.

    The live event ring holds 500 entries in memory and dies with the
    process — this file is the only thing that can answer what happened
    before the last restart. Pass a job id to --grep to follow one job end
    to end.
    """
    from . import balancerctl

    spec = profiles.load(model)
    body = balancerctl.audit(spec.profile, lines, grep)
    if not body.strip():
        console.print("[dim]no audit entries (is the balancer new, or WORK_DIR unset?)[/dim]")
        return
    console.print(body)


@balancer_app.command("logs")
def balancer_logs(
    model: str = ModelOpt,
    lines: int = typer.Option(100, "-n", help="Journal lines to fetch"),
) -> None:
    """The balancer's own journal — strikes, reaps, revives, retention.

    This is where lost-job forensics live: every `presumed lost after N
    probes` line names the job, its replica, and WHY (404 vs unreachable).
    """
    from . import balancerctl

    spec = profiles.load(model)
    console.print(balancerctl.logs(spec.profile, lines) or "[dim]no output[/dim]")


@balancer_app.command("status")
def balancer_status(
    model: str = ModelOpt,
    as_json: bool = typer.Option(False, "--json", help="Machine-readable, for scripts"),
) -> None:
    """Per-replica load as this model's balancer sees it."""
    from . import balancerctl

    spec = profiles.load(model)
    if as_json:
        # The whole snapshot, unflattened — a watchdog wants the numbers,
        # not the table.
        console.print_json(json.dumps(balancerctl.stats(spec.profile)))
        return
    console.print(
        f"unit={balancerctl.unit_for(spec.profile)} state={balancerctl.status(spec.profile)}"
    )
    try:
        snap = balancerctl.stats(spec.profile)
    except Exception as exc:
        console.print(f"[red]stats unavailable[/red]: {type(exc).__name__}")
        return
    table = Table(box=None, pad_edge=False)
    for col in ("replica", "health", "inflight", "started", "done", "failed", "mean job"):
        table.add_column(col)
    for b in snap["backends"]:
        # Draining is the state an operator most needs to SEE: the replica
        # reads "up" with inflight 0 while receiving no work — invisible, it
        # looks exactly like healthy-and-unlucky.
        health = (
            "[yellow]draining[/yellow]"
            if b.get("draining")
            else "up"
            if b["healthy"]
            else "[red]down[/red]"
        )
        table.add_row(
            b["url"],
            health,
            str(b["inflight"]),
            str(b["started"]),
            str(b["completed"]),
            str(b["failed"]),
            f"{b['mean_job_s']:.0f}s",
        )
    console.print(table)
    console.print(
        f"[dim]tracked jobs: {snap['tracked_jobs']} · reaped: {snap.get('reaped_jobs', 0)}[/dim]"
    )
    # The queue an operator was previously told to curl for.
    queue = snap.get("queue") or {}
    if queue:
        by_state = ", ".join(f"{k}={v}" for k, v in sorted((queue.get("by_state") or {}).items()))
        console.print(
            f"[dim]queue: {queue.get('waiting', 0)} waiting"
            + (f" · {by_state}" if by_state else "")
            + "[/dim]"
        )
    # Is the platform's own machinery alive? A dead maintenance loop looks
    # exactly like a healthy idle one unless the failures are counted AND
    # shown; they were counted but lived only in /metrics, which no command
    # reads.
    maint = snap.get("maintenance") or {}
    if maint:
        errs = int(maint.get("errors_total", 0))
        rerrs = int(maint.get("retention_errors_total", 0))
        style = "red" if errs or rerrs else "dim"
        audit = "on" if maint.get("audit_enabled") else "[yellow]OFF[/yellow]"
        console.print(
            f"[{style}]maintenance: {errs} error(s), retention {rerrs} error(s)"
            f"[/{style}] [dim]· audit {audit} · boot {maint.get('boot', '?')}[/dim]"
        )
        if errs or rerrs:
            console.print(
                "[dim]  the maintenance beat enforces the liveness invariant — "
                "read `deepme balancer logs -m <model>`[/dim]"
            )


patches_app = typer.Typer(help="Vendored engine patches (apply, verify, re-base)")
app.add_typer(patches_app, name="patches")

_COLOUR = {"applied": "green", "clean": "cyan", "stale": "red", "missing": "yellow"}


@patches_app.command("status")
def patches_status(
    engine: str = typer.Option("sglang", "--engine", "-e"),
) -> None:
    """Where each vendored patch stands against the installed engine."""
    from . import patches

    states = patches.status(engine)
    if not states:
        console.print(f"[dim]no patches declared for {engine}[/dim]")
        return
    table = Table(box=None, pad_edge=False)
    for col in ("patch", "state", "written against", "detail"):
        table.add_column(col, overflow="fold")
    for st in states:
        colour = _COLOUR.get(st.state, "white")
        table.add_row(
            st.patch.name,
            f"[{colour}]{st.state}[/{colour}]",
            st.patch.upstream_version or "-",
            st.detail,
        )
    console.print(table)
    if any(s.state == "stale" for s in states):
        console.print(
            "[red]stale[/red] means the engine moved under a patch. Re-base it "
            "against the new source, refresh its checksums with "
            f"`deepme patches record -e {engine}`, then re-run the correctness gate."
        )


@patches_app.command("apply")
def patches_apply(
    engine: str = typer.Option("sglang", "--engine", "-e"),
    force: bool = typer.Option(False, "--force", help="Apply even if stale (dangerous)"),
) -> None:
    """Apply every unapplied patch. Idempotent; safe to re-run."""
    from . import patches

    console.print("\n".join(patches.apply(engine, force=force)))
    runs.record("patches-apply", None, outcome="ok", detail={"engine": engine, "forced": force})


@patches_app.command("revert")
def patches_revert(engine: str = typer.Option("sglang", "--engine", "-e")) -> None:
    """Return the engine to pristine upstream (incident escape hatch)."""
    from . import patches

    console.print("\n".join(patches.revert(engine)))
    runs.record("patches-revert", None, outcome="ok", detail={"engine": engine})


@patches_app.command("install-hooks")
def patches_install_hooks(
    engine: str = typer.Option("sglang", "--engine", "-e"),
) -> None:
    """Install runtime hooks (shared weight store) into the engine venv."""
    from . import patches

    console.print("\n".join(patches.install_hooks(engine)))
    console.print(
        "[dim]Hooks are inert until a deployment sets DEEPME_SHARED_WEIGHTS=1 "
        "in its profile [serve.env].[/dim]"
    )


@patches_app.command("remove-hooks")
def patches_remove_hooks(
    engine: str = typer.Option("sglang", "--engine", "-e"),
) -> None:
    """Remove runtime hooks, returning the engine to stock behaviour."""
    from . import patches

    console.print("\n".join(patches.remove_hooks(engine)))


@patches_app.command("record")
def patches_record(
    engine: str = typer.Option("sglang", "--engine", "-e"),
    target: list[str] = typer.Option(None, "--target", "-t", help="Path under site-packages"),
) -> None:
    """Print pristine checksums for a manifest. Run on an UNPATCHED engine."""
    from . import patches

    targets = list(target or [])
    if not targets:
        targets = [t for p in patches.load(engine) for t in p.targets]
    console.print(json.dumps(patches.record_pristine(engine, targets), indent=2))


@app.command("job-progress")
def job_progress(
    job_id: str = typer.Argument(..., help="Server job id (or q_ ticket)"),
    model: str = ModelOpt,
) -> None:
    """One job's live stage, via the exact path every progress display uses.

    Runs the same chain as batch/generate rows — capacity lookup for the
    owning replica, that unit's journal, the bar parse — so when a display
    seems silent, this answers 'is the pipeline broken or is the job simply
    between stages' in one command.
    """
    from . import media

    # check=False: this only resolves an ADDRESS to read progress from. A
    # servability check would refuse to tell an operator what happened to a
    # job on a fleet that has since gone down — exactly when they need it.
    spec, _via = front_end(profiles.load(model), check=False)
    unit = None
    try:
        unit = media._owner_unit(spec, job_id)
    except Exception as exc:
        console.print(f"owner lookup failed: {_safe(str(exc))}")
    console.print(f"owner unit: {unit or '(not running on any replica)'}")
    if unit:
        stage = serve.stage_progress(spec, 3600, unit=unit)
        if stage is None:
            console.print("no stage reported in the last hour's journal")
        elif not stage.counted:
            console.print(f"stage: {stage.label} (no step counter)")
        else:
            console.print(
                f"stage: {stage.label} {stage.step}/{stage.total} · "
                f"{stage.rate_s:.1f}s/step · ~{_hms(stage.eta_s)} left"
            )


artifacts_app = typer.Typer(help="Server-side generated files: audit and sweep")
app.add_typer(artifacts_app, name="artifacts")


@artifacts_app.command("audit")
def artifacts_audit(model: str = ModelOpt) -> None:
    """What generated output is sitting on the server, and how much disk.

    Nothing deletes outputs automatically — every completed clip stays on
    the work volume until swept, and a full disk fails GENERATIONS. This is
    the read-only view; `artifacts sweep` is the janitor.
    """
    spec = profiles.load(model)
    work = serve.work_dir(spec)
    # Both accumulation surfaces: generated outputs AND staged condition
    # uploads (inputs/) — keyframes and reference media land there per
    # request and were invisible to the first version of this audit.
    out = remote.run(
        f"du -sh {work}/outputs {work}/inputs 2>/dev/null; "
        f"find {work}/outputs -name '*.mp4' 2>/dev/null | wc -l; "
        f"find {work}/outputs -name '*.mp4' -mtime +2 2>/dev/null | wc -l; "
        f"find {work}/inputs -type f 2>/dev/null | wc -l; "
        f"df -h {work} | tail -1",
        check=False,
    ).stdout.splitlines()
    console.print("\n".join(f"  {line}" for line in out) if out else "no work directory yet")


@artifacts_app.command("sweep")
def artifacts_sweep(
    model: str = ModelOpt,
    older_than_hours: int = typer.Option(48, help="Only files at least this old are candidates"),
    delete: bool = typer.Option(
        False,
        "--delete",
        help="Actually remove them (default is a dry-run listing)",
    ),
) -> None:
    """Remove old generated files from the server's work volume.

    DRY-RUN BY DEFAULT: it lists what would go and how much space returns.
    Age-based, deliberately conservative — a clip a client never downloaded
    is indistinguishable from one it did, so the threshold is the policy.
    The durable fix (delete-after-verified-download) is a product decision;
    until then this is the only thing standing between the disk and full.
    """
    spec = profiles.load(model)
    work = serve.work_dir(spec)
    mins = older_than_hours * 60
    # Outputs AND staged inputs: condition uploads accumulate identically
    # and a sweep that missed them only postponed the full disk.
    listing = remote.run(
        f"find {work}/outputs -name '*.mp4' -mmin +{mins} 2>/dev/null "
        f"-printf '%s %p\\n'; "
        f"find {work}/inputs -type f -mmin +{mins} 2>/dev/null "
        f"-printf '%s %p\\n'",
        check=False,
    ).stdout.strip()
    if not listing:
        console.print("nothing old enough to sweep")
        return
    files = listing.splitlines()
    total_gb = sum(int(line.split()[0]) for line in files) / 1e9
    console.print(f"{len(files)} file(s), {total_gb:.1f} GB, older than {older_than_hours}h")
    if not delete:
        for line in files[:20]:
            console.print(f"  [dim]{line.split(maxsplit=1)[1]}[/dim]")
        if len(files) > 20:
            console.print(f"  [dim]... and {len(files) - 20} more[/dim]")
        console.print("re-run with --delete to remove them")
        return
    remote.run(
        f"find {work}/outputs -name '*.mp4' -mmin +{mins} -delete; "
        f"find {work}/inputs -type f -mmin +{mins} -delete",
        check=False,
    )
    runs.record(
        "artifacts-sweep",
        spec,
        outcome="ok",
        detail={
            "files": len(files),
            "gb": round(total_gb, 1),
            "older_than_hours": older_than_hours,
        },
    )
    console.print(f"[green]removed {len(files)} file(s), {total_gb:.1f} GB reclaimed[/green]")


@app.command("econ")
def econ_cmd(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable"),
    record_duration: str = typer.Option(
        None,
        "--record-duration",
        help="Write a finished duration batch's timings into economics.toml",
    ),
) -> None:
    """Costs, break-even and derived prices — from deploy/economics.toml.

    Nothing here is typed in as a price: measured occupancy x server cost
    gives per-task cost, and every price (spot floor, market, surge tiers)
    is derived from that. The gateway serves the same derivation to other
    API layers at /v1/pricing.
    """
    from . import economics

    if record_duration:
        points = economics.record_duration_batch(record_duration)
        console.print(f"recorded {len(points)} duration point(s):")
        for p in points:
            console.print(f"  {p['seconds']:>4.0f}s clip -> {p['gpu_seconds']:.0f}s GPU")
        console.print("re-run `deepme econ` to see the updated curve")
        return

    econ = economics.derive(economics.load())
    if as_json:
        console.print_json(json.dumps(econ.as_dict()))
        return

    # Drift checks against the LIVE system — the derivation is only as good
    # as its inputs staying true.
    with contextlib.suppress(Exception):
        from . import gatewayctl

        live = gatewayctl.pricing()
        fresh = economics.pricing_json(econ)
        if live.get("table_hash") and live["table_hash"] != fresh["table_hash"]:
            console.print(
                "[yellow]the gateway is serving an OLDER price table "
                f"(derived {live.get('derived_at')}) — restart it "
                "(`deepme gateway start`) to publish this one[/yellow]"
            )
        cap = gatewayctl.capacity()
        live_replicas = sum(int(f.get("replicas") or 0) for f in cap["fleets"])
        if live_replicas and live_replicas != econ.replicas:
            console.print(
                f"[yellow]fleet is running {live_replicas} replica(s) but "
                f"costs assume {econ.replicas} — per-replica cost and floors "
                f"are wrong until [server] in economics.toml matches or the "
                f"fleet does[/yellow]"
            )

    console.print(
        f"server ${econ.cost_per_hour:.2f}/hr over {econ.replicas} replicas "
        f"= ${econ.cost_per_replica_second * 3600:.3f}/replica-hr"
    )
    tasks = Table(box=None, pad_edge=False, title="per task (per output second)")
    for col in ("task", "gpu s", "cost", "spot floor", "market", "margin", "out-s/replica-hr"):
        tasks.add_column(col, overflow="fold")
    for te in econ.tasks.values():
        tasks.add_row(
            te.task,
            f"{te.gpu_seconds:.0f}",
            f"${te.cost_per_output_second:.4f}",
            f"${te.spot_floor:.3f}",
            f"${te.market_price:.3f}",
            f"{te.margin_at_market:.2f}x",
            f"{te.output_seconds_per_replica_hour:.1f}",
        )
    console.print(tasks)

    mixes = Table(
        box=None, pad_edge=False, title="fleet mixes at configured prices (100% utilization)"
    )
    for col in ("mix", "out-s/hr", "revenue/hr", "profit/hr", "break-even util"):
        mixes.add_column(col, overflow="fold")
    for m in econ.mixes:
        mixes.add_row(
            m.name,
            f"{m.output_seconds_per_hour:.0f}",
            f"${m.revenue_per_hour:.2f}",
            f"${m.profit_per_hour:.2f}",
            f"{m.break_even_utilization:.1%}",
        )
    console.print(mixes)

    surge = Table(box=None, pad_edge=False, title="surge ladder (per output second)")
    surge.add_column("queued/replica")
    for t in econ.tasks:
        surge.add_column(t)
    for tier in econ.surge:
        surge.add_row(
            f">= {tier['queued_per_replica']:.1f} (x{tier['multiplier']:.1f})",
            *(f"${tier['prices'][t]:.3f}" for t in econ.tasks),
        )
    console.print(surge)
    console.print(f"[dim]duration: {econ.duration_note}[/dim]")
    if econ.steps_note:
        console.print(f"[dim]steps: {econ.steps_note}[/dim]")


@app.command("events")
def events_cmd(
    raw: bool = typer.Option(False, "--raw", help="JSON lines, for piping"),
) -> None:
    """Subscribe to the platform firehose: every job transition, live.

    The push counterpart of `top` — queued/started/handover/terminal/
    cancelled/rejected plus fleet membership, retention and outage events,
    tagged per fleet. ^C ends the subscription; the platform never notices
    a slow or absent consumer.
    """
    import httpx

    from . import gatewayctl

    try:
        with httpx.stream("GET", f"{gatewayctl.api_base()}/v1/events", timeout=None) as r:
            if r.status_code in (400, 404):
                raise SystemExit(
                    "the DEPLOYED gateway predates /v1/events — ships with "
                    "the next production recycle"
                )
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if raw:
                    print(line[len("data: ") :], flush=True)
                    continue
                with contextlib.suppress(ValueError, TypeError):
                    e = json.loads(line[len("data: ") :])
                    bits = " ".join(
                        f"{k}={v}"
                        for k, v in e.items()
                        if k not in ("seq", "ts", "type", "fleet") and v is not None
                    )
                    console.print(
                        f"[dim]{time.strftime('%H:%M:%S', time.localtime(e.get('ts', 0)))}[/dim] "
                        f"[cyan]{e.get('fleet', '')}[/cyan] "
                        f"[bold]{e.get('type')}[/bold] {_safe(bits[:160])}"
                    )
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except httpx.HTTPError as exc:
        raise SystemExit(f"gateway unreachable: {exc}") from None


@app.command("top")
def top_cmd(
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh until ^C"),
    every: float = typer.Option(5.0, "--every", help="Seconds between refreshes"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable"),
    recent: int = typer.Option(
        3,
        "--recent",
        help="Terminal jobs to show per fleet (the server keeps 50) — raise "
        "this to see who closed what where when hunting lost jobs",
    ),
) -> None:
    """Everything running on the platform, with stages — the operator view.

    Independent of any client: batch runners see their own items, but the
    operator owns the SYSTEM. One screen: every job (what it is, where it
    runs, denoise step and countdown), every queued ticket (position,
    priority), and what just finished. The same payload serves as JSON for
    scrapers; /metrics carries the Prometheus form.
    """
    import time as _time

    from . import gatewayctl, media

    def fetch() -> dict:
        import httpx

        try:
            r = httpx.get(f"{gatewayctl.api_base()}/v1/jobs", timeout=15)
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"gateway unreachable ({exc}) — is it running? `deepme gateway status`"
            ) from None
        if r.status_code in (400, 404):
            # The deployed gateway predates this endpoint: its catch-all
            # tried to ROUTE the request and refused. A version-drift
            # condition must explain itself, not stack-trace.
            raise SystemExit(
                "the DEPLOYED gateway predates /v1/jobs — the observability "
                "surface ships with the next production recycle "
                "(`deepme gateway start`, guarded). Until then: "
                "`deepme capacity` for slots and `deepme job-progress <id>` "
                "for a single job's stage."
            )
        r.raise_for_status()
        return dict(r.json())

    if as_json:
        console.print_json(json.dumps(fetch()))
        return

    def render() -> Table:
        data = fetch()
        table = Table(box=None, pad_edge=False, title="platform jobs")
        for col in ("fleet", "job", "task", "prompt", "phase", "elapsed"):
            table.add_column(col, overflow="fold")
        for fleet in data.get("fleets", []):
            prof = fleet["profile"]
            if not fleet.get("reachable", True):
                table.add_row(prof, "[red]unreachable[/red]", "", "", "", "")
                continue
            spec = None
            with contextlib.suppress(Exception):
                spec = profiles.load(prof)
            for j in fleet.get("running", []):
                phase = ""
                if spec is not None:
                    with contextlib.suppress(Exception):
                        port = int(str(j.get("backend", "")).rsplit(":", 1)[-1])
                        unit = media._unit_for_port(spec, port)
                        stage = (
                            serve.stage_progress(spec, j.get("elapsed_s", 3600), unit=unit)
                            if unit
                            else None
                        )
                        if stage and stage.counted:
                            phase = (
                                f"{stage.label} {stage.step}/{stage.total}"
                                f" · ~{_hms(stage.eta_s)} left"
                            )
                        elif stage:
                            phase = stage.label
                if not phase and j.get("eta_s") is not None:
                    phase = f"~{_hms(j['eta_s'])} left"
                table.add_row(
                    prof,
                    str(j.get("job_id", ""))[:12],
                    str(j.get("task") or ""),
                    _safe(str(j.get("prompt") or "")[:36]),
                    phase,
                    _hms(j.get("elapsed_s", 0)),
                )
            for q in fleet.get("queued", []):
                table.add_row(
                    prof,
                    str(q.get("ticket", ""))[:16],
                    str(q.get("task") or ""),
                    _safe(str(q.get("prompt") or "")[:36]),
                    f"[yellow]queued #{q.get('position')}[/yellow] p{q.get('priority')}",
                    "",
                )
            for r_ in fleet.get("recent", [])[:recent]:
                # Backend and closer travel together: "failed (reaper) @8100"
                # is a diagnosis; "failed" alone is a mystery.
                where = str(r_.get("backend") or "").rsplit(":", 1)[-1]
                table.add_row(
                    prof,
                    str(r_.get("job_id", ""))[:12],
                    str(r_.get("task") or ""),
                    _safe(str(r_.get("prompt") or "")[:36]),
                    f"[dim]{r_.get('status')} ({r_.get('closed_by')})"
                    f"{' @' + where if where else ''}[/dim]",
                    "",
                )
        return table

    if not watch:
        console.print(render())
        return
    from rich.live import Live

    try:
        with Live(render(), console=console, refresh_per_second=2) as live:
            while True:
                _time.sleep(every)
                live.update(render())
    except KeyboardInterrupt:
        raise SystemExit(130) from None


@app.command("capacity")
def capacity_cmd(
    watch: bool = typer.Option(False, "--watch", "-w", help="Refresh until Ctrl-C"),
    every: float = typer.Option(2.0, "--every", help="Seconds between refreshes"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable"),
) -> None:
    """Where work can go right now — the scheduling signal.

    The same answer an external scheduler polls (`GET /v1/capacity`): `free`
    is how many requests will start immediately. Shown here so an operator
    sees exactly what the app layer sees.
    """
    import time as _time

    import httpx

    from . import gatewayctl

    def _fetch() -> dict:
        """Explain, never stack-trace. This is the command the runbook
        recommends for watching a drain, so a down gateway must produce a
        next action — `top` already does this and `capacity` did not."""
        try:
            return gatewayctl.capacity()
        except httpx.HTTPError as exc:
            raise SystemExit(
                f"gateway unreachable ({exc}) — check `deepme gateway status`, "
                f"or read `deepme gateway logs`"
            ) from None

    def render() -> Table:
        body = _fetch()
        table = Table(
            box=None,
            pad_edge=False,
            title=f"free {body['free']} — accepting "
            f"{', '.join(sorted(set(body['accepting']))) or 'nothing'}",
        )
        for col in ("fleet", "tasks", "free", "busy", "queued", "mean job", "running"):
            table.add_column(col, overflow="fold")
        for f in body["fleets"]:
            if not f.get("reachable"):
                table.add_row(
                    _safe(f["profile"]),
                    ", ".join(f["tasks"]),
                    "[red]unreachable[/red]",
                    "-",
                    "-",
                    "-",
                    "-",
                )
                continue
            running = " ".join(
                f"{_hms(j['elapsed_s'])}"
                + (f"/~{_hms(j['eta_s'])}" if j.get("eta_s") is not None else "")
                for j in f.get("running", [])
            )
            table.add_row(
                _safe(f["profile"]),
                ", ".join(f["tasks"]),
                f"[green]{f['free']}[/green]" if f["free"] else "0",
                str(f["busy"]),
                str(f.get("queued", 0)),
                _hms(f["mean_job_s"]) if f.get("mean_job_s") else "-",
                running or "-",
            )
        return table

    if as_json:
        console.print_json(json.dumps(_fetch()))
        return
    if not watch:
        console.print(render())
        return
    from rich.live import Live

    try:
        with Live(render(), console=console, refresh_per_second=4) as live:
            while True:
                _time.sleep(every)
                live.update(render())
    except KeyboardInterrupt:
        # 130, like every other interrupt path: exiting 0 made
        # `deepme capacity -w && deploy` treat an abandoned watch as a
        # successful precondition.
        raise SystemExit(130) from None


@app.command("discover")
def discover_cmd(
    as_json: bool = typer.Option(False, "--json", help="Machine-readable catalog"),
) -> None:
    """What is deployed and what it can serve — the internal service catalog."""
    from . import discovery

    if as_json:
        console.print_json(json.dumps(discovery.catalog()))
        return

    entries = discovery.models()
    models_table = Table(box=None, pad_edge=False, title="models")
    for col in ("model", "tasks", "replicas", "served by", "limits"):
        models_table.add_column(col, overflow="fold")
    for m in entries:
        models_table.add_row(
            m.model,
            ", ".join(m.tasks),
            str(m.replicas),
            ", ".join(m.profiles),
            ", ".join(f"{k}={v}" for k, v in m.limits.items()) or "-",
        )
    console.print(models_table)
    if not entries:
        console.print("[yellow]no generation model is deployed[/yellow]")

    eps = Table(box=None, pad_edge=False, title="endpoints")
    for col in ("endpoint", "kind", "url", "state"):
        eps.add_column(col)
    for e in discovery.endpoints():
        colour = "green" if e.running else "dim"
        eps.add_row(e.name, e.kind, e.url, f"[{colour}]{e.state}[/{colour}]")
    console.print(eps)
    console.print(
        "[dim]declared = in the port registry; running = a unit claims it. "
        "They are deliberately different.[/dim]"
    )


@app.command("ports")
def ports_cmd(
    check: bool = typer.Option(
        False, "--check", help="Quiet mode for scripts: no table, exit status only"
    ),
) -> None:
    """The declared port registry, and whether anything overlaps.

    Always exits non-zero on a conflict; `--check` just suppresses the
    table for CI use. (The old flag changed nothing — an inert option is
    worse than none.)
    """
    from . import ports as ports_mod

    registry = ports_mod.load()
    problems = ports_mod.conflicts(registry)
    if not check:
        table = Table(box=None, pad_edge=False)
        for col in ("ports", "owner"):
            table.add_column(col)
        for a in registry.allocations():
            span = f"{a.port}-{a.last}" if a.span > 1 else str(a.port)
            table.add_row(span, a.owner)
        console.print(table)
    if problems:
        console.print("[red]conflicts[/red]:")
        for p in problems:
            console.print(f"  {p}")
        raise SystemExit(1)
    if not check:
        console.print(
            f"\n[green]no conflicts[/green] — "
            f"{len(registry.allocations())} allocations "
            f"[dim]({ports_mod.REGISTRY})[/dim]"
        )


gateway_app = typer.Typer(help="One address in front of every deployed fleet")
app.add_typer(gateway_app, name="gateway")


@gateway_app.command("start")
def gateway_start(
    setup: bool = typer.Option(False, "--setup", help="Sync the gateway venv first"),
) -> None:
    """Route model+task to the fleet that serves it. Run after a fleet change."""
    from . import gatewayctl

    if setup:
        console.print(gatewayctl.setup())
    console.print(gatewayctl.start())
    # PROVE it. A gateway crash-looping on a bad routes environment used to
    # print this same green banner: the unit forked, so systemctl returned
    # 0, and nothing ever asked the service itself.
    healthy = _await_service(gatewayctl.routes, what="gateway")
    runs.record("gateway-start", None, outcome="ok" if healthy else "failed")
    if not healthy:
        raise SystemExit(
            f"gateway unit started but {gatewayctl.api_base()}/gateway/routes "
            f"never answered — read `deepme gateway logs`"
        )
    console.print(
        f"[green]gateway[/green] {gatewayctl.api_base()}/v1 — "
        f"routes at {gatewayctl.api_base()}/gateway/routes"
    )


@gateway_app.command("stop")
def gateway_stop() -> None:
    """Stop the gateway (fleets keep serving on their own balancers)."""
    from . import gatewayctl

    gatewayctl.stop()
    runs.record("gateway-stop", None, outcome="ok")
    console.print("stopped")


@gateway_app.command("status")
def gateway_status() -> None:
    """What the gateway is routing, and where."""
    from . import gatewayctl

    console.print(f"unit={gatewayctl.UNIT} state={gatewayctl.status()}")
    try:
        snap = gatewayctl.routes()
    except Exception as exc:
        console.print(f"[red]routes unavailable[/red]: {type(exc).__name__}")
        return
    table = Table(box=None, pad_edge=False)
    for col in ("model", "tasks", "fleet", "balancer"):
        table.add_column(col)
    for r in snap["routes"]:
        table.add_row(r["model"], ", ".join(r["tasks"]), r["profile"], r["url"])
    console.print(table)
    console.print(f"[dim]tracked jobs: {snap['tracked_jobs']}[/dim]")


@gateway_app.command("logs")
def gateway_logs(lines: int = typer.Option(100, "-n", help="Journal lines to fetch")) -> None:
    """The gateway's journal — routing refusals, vanished jobs, unreachable
    fleets, ambiguous owner probes.

    This is where "the customer's job disappeared" is answered: the gateway
    decides which fleet holds every id.
    """
    from . import gatewayctl

    console.print(gatewayctl.logs(lines) or "[dim]no output[/dim]")


fleet_app = typer.Typer(help="Fleet composition: which models share the machine")
app.add_typer(fleet_app, name="fleet")


@fleet_app.command("list")
def fleet_list() -> None:
    """Available fleet mixes."""
    from . import fleet as fleet_mod

    table = Table(box=None, pad_edge=False)
    for col in ("fleet", "composition", "replicas"):
        table.add_column(col)
    for path in sorted(fleet_mod.FLEETS_DIR.glob("*.toml")):
        f = fleet_mod.load(path.stem)
        table.add_row(
            f.name,
            " + ".join(f"{m.profile} x{m.replicas}" for m in f.members),
            str(f.total_replicas),
        )
    console.print(table)


@fleet_app.command("status")
def fleet_status_cmd() -> None:
    """The running fleet against every declared mix — which one are we in?

    Auto-detected from live unit claims and balancer membership, never from
    what was applied last. The one command that answers "what shape is the
    machine in right now" without stitching together status/discover/capacity.
    """
    from . import balancerctl, resources
    from . import fleet as fleet_mod

    claims = [c for c in resources.current_claims() if c.profile]
    table = Table(box=None, pad_edge=False, title="running replicas")
    for col in ("unit", "profile", "gpus", "port", "balancer sees it"):
        table.add_column(col, overflow="fold")
    membership: dict[str, dict] = {}
    for profile in sorted({c.profile for c in claims}):
        # SystemExit too: a profile with no balancer port declared (a CPU
        # model, say) raises it from the registry, and status must describe
        # such deployments rather than die on them.
        with contextlib.suppress(Exception):
            membership[profile] = {
                b["url"]: b for b in balancerctl.stats(profile).get("backends", [])
            }
    for c in sorted(claims, key=lambda c: c.unit):
        b = membership.get(c.profile, {}).get(f"http://127.0.0.1:{c.port}")
        if b is None:
            seen = (
                "[red]NOT ATTACHED[/red]" if c.profile in membership else "[dim]no balancer[/dim]"
            )
        elif b.get("draining"):
            seen = "[yellow]draining[/yellow]"
        else:
            seen = f"inflight {b['inflight']}"
        table.add_row(c.unit, c.profile, str(list(c.gpus)), str(c.port), seen)
    console.print(table)
    # Which declared mix, if any, matches what is running?
    for path in sorted(fleet_mod.FLEETS_DIR.glob("*.toml")):
        with contextlib.suppress(Exception):
            f = fleet_mod.load(path.stem)
            d = fleet_mod.delta(
                fleet_mod.plan(f, releasing_extra={c.profile for c in claims}),
                claims,
            )
            if d.is_noop:
                console.print(f"matches mix: [bold]{f.name}[/bold]")
                return
    console.print("[dim]matches no declared mix[/dim]")


@fleet_app.command("plan")
def fleet_plan(name: str = typer.Argument(..., help="Fleet name")) -> None:
    """Show where a fleet would place, without changing anything."""
    from . import fleet as fleet_mod

    f = fleet_mod.load(name)
    specs = fleet_mod.plan(f)
    console.print(f"[bold]{f.name}[/bold] — {f.total_replicas} replica(s)")
    console.print(fleet_mod.describe(specs))


@fleet_app.command("switch")
def fleet_switch(
    name: str = typer.Argument(..., help="Target fleet mix"),
    timeout: float = typer.Option(2400, help="Per-replica start timeout"),
    drain_timeout: float = typer.Option(
        3600, help="How long to wait for a leaving replica's job to finish"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Proceed even if a leaving profile still has queued tickets",
    ),
) -> None:
    """Change the fleet mix by its DELTA — running work and queues survive.

    Unlike `fleet apply` (full stop of everything the mix names), this
    auto-detects what is running, keeps replicas whose placement already
    matches, drains the ones that leave (no new work, running jobs finish),
    and attaches the ones that arrive to their balancer AT RUNTIME — no
    balancer restart, so waiting tickets and job tracking survive. The
    gateway is re-pointed on any shape change (its frozen catalog goes
    stale on any resize); only a true no-op switch skips it.

    Ctrl-C is safe and re-running is always safe: every abort lands in the
    ledger with what completed, and the reconcile pass converges from any
    intermediate state.
    """
    # A switch MUTATES the fleet, so it must be on the record even when it
    # aborts — an interrupted resize with nothing in the ledger is exactly
    # how "why is the fleet in this shape" becomes unanswerable.
    from . import switch as switch_mod

    audit: dict = {"fleet": name, "abandoned_tickets": []}
    runs.record("fleet-switch", None, outcome="began", detail={"fleet": name})
    outcome = "aborted"
    try:
        switch_mod.execute_switch(
            name,
            timeout=timeout,
            drain_timeout=drain_timeout,
            force=force,
            audit=audit,
            ops=switch_mod.default_ops(say=console.print, stamp=_stamp),
        )
        outcome = "ok"
    except KeyboardInterrupt:
        console.print(
            f"\n{_stamp()} interrupted — the fleet is mid-transition. "
            f"Re-run `uv run deepme fleet switch {name}`: the reconcile "
            f"pass converges from any intermediate state."
        )
        outcome = "interrupted"
        raise SystemExit(130) from None
    finally:
        runs.record("fleet-switch", None, outcome=outcome, detail=audit)


# fleet switch orchestration lives in switch.execute_switch (testable, with
# every side effect behind an injectable SwitchOps port).


@fleet_app.command("apply")
def fleet_apply(
    name: str = typer.Argument(..., help="Fleet name"),
    timeout: float = typer.Option(2400),
) -> None:
    """Deploy a fleet mix: stop what it replaces, place it, wait for health."""
    from . import fleet as fleet_mod

    f = fleet_mod.load(name)
    specs = fleet_mod.plan(f)
    console.print(f"[bold]{f.name}[/bold]\n{fleet_mod.describe(specs)}")

    # Stop every profile the fleet mentions before placing any of it, so a
    # previous generation cannot crash-loop into the new layout.
    for profile in {s.profile for s in specs}:
        serve.stop_profile(profiles.load(profile))

    # The first replica of each profile builds that model's shared store, so
    # it goes alone; the rest map it and can come up together.
    firsts: dict[str, profiles.RunSpec] = {}
    for spec in specs:
        firsts.setdefault(spec.profile, spec)
    for spec in firsts.values():
        console.print(f"[dim]{_stamp()} starting {spec.unit} (builds its store)[/dim]")
        serve.start(spec)
        ok, detail = serve.wait_healthy(spec, timeout_s=timeout)
        console.print(
            f"{_stamp()} {spec.unit}: "
            + ("[green]healthy[/green]" if ok else f"[red]{_safe(detail)}[/red]")
        )
        if not ok:
            raise SystemExit(
                f"{spec.unit} did not come up — `deepme serve diag -m {spec.profile}` has the traceback"
            )

    rest = [s for s in specs if s is not firsts.get(s.profile)]
    for spec in rest:
        console.print(f"[dim]{_stamp()} starting {spec.unit}[/dim]")
        serve.start(spec)
    if rest:
        import concurrent.futures as _f

        with _f.ThreadPoolExecutor(max_workers=len(rest)) as pool:
            futures = {pool.submit(serve.wait_healthy, s, timeout): s for s in rest}
            failed = []
            for fut in _f.as_completed(futures):
                s = futures[fut]
                ok, detail = fut.result()
                console.print(
                    f"{_stamp()} {s.unit}: "
                    + ("[green]healthy[/green]" if ok else f"[red]{_safe(detail)}[/red]")
                )
                if not ok:
                    failed.append(s.unit)
        if failed:
            raise SystemExit(
                f"replicas did not come up: {failed} — `deepme serve diag -m` each for the traceback"
            )

    runs.record(
        "fleet",
        None,
        outcome="ok",
        detail={
            "fleet": f.name,
            "members": [{"profile": m.profile, "replicas": m.replicas} for m in f.members],
        },
    )
    console.print(
        "[dim]put an API in front of each model: uv run deepme balancer start -m <profile>[/dim]"
    )


story_app = typer.Typer(help="Chain clips: each continues from the last frame")
app.add_typer(story_app, name="story")


@story_app.command("run")
def story_run(
    prompts: str = typer.Argument(..., help="One prompt per segment (or JSONL)"),
    model: str = ModelOpt,
    name: str = typer.Option(None, help="Story name (default: the file stem)"),
    seconds: int = typer.Option(None, help="Length of each segment"),
    out: str = typer.Option(None, "--out", "-o"),
    direct: bool = typer.Option(False, "--direct", help="Bypass the load balancer"),
    join: bool = typer.Option(True, "--join/--no-join", help="Concatenate at the end"),
    seam: bool = typer.Option(
        True,
        "--seam/--keep-seam",
        help="Drop each continuation's first frame — it duplicates the frame it was seeded from",
    ),
    timeout: float = typer.Option(7200),
) -> None:
    """Generate a continuous sequence, each segment seeded by the previous."""
    from . import batch as batch_mod
    from . import story as story_mod

    spec = profiles.load(model)
    path = Path(prompts)
    label = name or path.stem
    out_dir = Path(out) if out else story_mod.STORY_DIR / label

    story = story_mod.Story.load(label, out_dir)
    if story is None:
        items = batch_mod.load_items(path)
        story = story_mod.Story(
            name=label,
            profile=spec.profile,
            out_dir=str(out_dir),
            segments=[
                story_mod.Segment(
                    index=i.index,
                    prompt=i.prompt,
                    seconds=i.seconds or seconds,
                    seed=i.seed,
                    steps=i.steps,
                    params=i.params,
                )
                for i in items
            ],
        )
    else:
        console.print(f"[dim]resuming {label!r} from its recorded state[/dim]")

    target, via = front_end(spec, direct=direct)
    pending = sum(1 for s in story.segments if s.status != "done")
    console.print(
        f"{_stamp()} {len(story.segments)} segment(s), {pending} to generate, "
        f"via {via} [dim](serial by nature — each needs the frame before it)[/dim]"
    )

    def report(seg, st) -> None:
        mark = "[green]ok[/green]" if seg.status == "done" else f"[red]{_safe(seg.error)}[/red]"
        console.print(
            f"{_stamp()} {seg.name} {mark} [dim]{seg.took_s:.0f}s · {seg.prompt[:52]}[/dim]"
        )

    story_mod.run(target, story, timeout=timeout, on_done=report)

    clips = story.done_clips()
    console.print(f"[green]{len(clips)}[/green] segment(s) in {story.dir}")
    if join and clips:
        final = story_mod.concat(clips, story.final_path, drop_seam_frame=seam)
        console.print(f"[green]joined[/green] {final}")
    runs.record(
        "story",
        spec,
        outcome="ok",
        detail={
            "name": story.name,
            "segments": len(story.segments),
            "joined": str(story.final_path) if join and clips else None,
        },
    )


@story_app.command("join")
def story_join(
    name: str = typer.Argument(..., help="Story name"),
    seam: bool = typer.Option(True, "--seam/--keep-seam"),
) -> None:
    """Re-join an existing story's segments without regenerating them."""
    from . import story as story_mod

    story = story_mod.Story.load(name)
    if story is None:
        existing = sorted(p.name for p in story_mod.STORY_DIR.glob("*") if p.is_dir())
        raise SystemExit(
            f"no story named {name!r} under {story_mod.STORY_DIR}"
            + (f" — existing: {', '.join(existing)}" if existing else "")
        )
    final = story_mod.concat(story.done_clips(), story.final_path, drop_seam_frame=seam)
    console.print(f"[green]joined[/green] {final}")


batch_app = typer.Typer(help="Run a backlog of generations unattended")
app.add_typer(batch_app, name="batch")


@batch_app.command("run")
def batch_run(
    prompts: str = typer.Argument(..., help="File: one prompt per line, or JSONL"),
    model: str = ModelOpt,
    name: str = typer.Option(None, help="Batch name (default: the file stem)"),
    out: str = typer.Option(None, "--out", "-o", help="Where clips are written"),
    concurrency: int = typer.Option(0, help="In flight at once (default: one per healthy replica)"),
    direct: bool = typer.Option(False, "--direct", help="Bypass the load balancer"),
    resume: bool = typer.Option(
        True, "--resume/--restart", help="Continue an existing batch of this name"
    ),
    timeout: float = typer.Option(
        0,
        help="Optional per-item hard cap, seconds (0 = none; stuck items "
        "are detected from server evidence instead)",
    ),
    fetch_existing: bool = typer.Option(
        False,
        "--fetch-existing",
        help="Collect ONLY: adopt and download items that already own a "
        "server job, submit NOTHING new, then stop. Pending items stay "
        "pending for a later run.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show exactly what this run would do — what it skips as an "
        "existing duplicate, what it adopts, what it would submit — and "
        "exit without contacting a GPU.",
    ),
) -> None:
    """Generate every prompt in a file, resumably, keeping the fleet busy.

    Ctrl-C stops CLEANLY: in-flight jobs keep running server-side, their ids
    are already on disk, and re-running the same command re-attaches to them
    instead of duplicating.

    --fetch-existing winds a batch down instead of continuing it: the
    in-flight tail is harvested, the backlog is left parked, and the fleet
    receives no new work.
    """
    from . import balancerctl
    from . import batch as batch_mod

    spec = profiles.load(model)
    path = Path(prompts)
    label = name or path.stem
    existing = batch_mod.Batch.load(label) if resume else None
    if fetch_existing and existing is None:
        raise SystemExit(
            f"no batch state named {label!r} — --fetch-existing collects "
            f"jobs an earlier run already submitted, so it needs that "
            f"run's state (see `deepme batch list`)"
        )
    if existing:
        batch = existing
        # RECONCILE against the file. Taking the saved state and discarding
        # the file meant a resume ran the OLD prompts with their OLD job
        # ids — and an item with a job id is ADOPTED, not generated, so the
        # old video was delivered as the new prompt's output.
        try:
            merged = batch_mod.merge_items(existing.items, batch_mod.load_items(path))
        except batch_mod.PromptsChanged as exc:
            raise SystemExit(f"{path} does not match batch {label!r}: {exc}") from None
        added = len(merged) - len(existing.items)
        batch.items = merged
        if added:
            batch.save()
        console.print(
            f"[dim]resuming batch {label!r}: {batch.counts()}"
            + (f" (+{added} new from {path.name})" if added else "")
            + "[/dim]"
        )
    else:
        batch = batch_mod.Batch(
            name=label,
            profile=spec.profile,
            items=batch_mod.load_items(path),
            out_dir=str(Path(out) if out else batch_mod.BATCH_DIR / label / "clips"),
        )

    if dry_run:
        _batch_dry_run(batch, fetch_existing=fetch_existing)
        return

    target, via = front_end(spec, direct=direct)
    # One in flight per replica: more just queues INSIDE an engine, where the
    # backlog can no longer be reordered, rebalanced or cancelled.
    replicas = 1
    mean = 0.0
    with contextlib.suppress(Exception):
        backends = balancerctl.stats(spec.profile)["backends"]
        replicas = max(len([b for b in backends if b["healthy"]]), 1)
        means = [b["mean_job_s"] for b in backends if b["mean_job_s"]]
        mean = sum(means) / len(means) if means else 0.0
    workers = concurrency or replicas

    if fetch_existing:
        # Adoptable = owns a server job (running, or failed on a cut
        # download that reconciliation will retry). Counted here only for
        # the banner; batch.run applies the same rule authoritatively.
        adoptable = [i for i in batch.items if i.job_id and i.status in ("running", "failed")]
        parked = sum(
            1
            for i in batch.items
            if i.status == "pending" or (i.status == "running" and not i.job_id)
        )
        pending = len(adoptable)
        console.print(
            f"{_stamp()} fetch-existing: adopting {len(adoptable)} job(s) "
            f"via {via}, submitting nothing — "
            f"{parked} item(s) stay parked for a later run"
        )
    else:
        pending = len(batch.todo())
        est = batch_mod.estimate(pending, workers, mean)
        console.print(
            f"{_stamp()} {pending} item(s) via {via}, {workers} in flight"
            + (
                f" — ~{est['hours']:.1f}h at {est['per_hour']:.1f}/hour "
                f"(~{est['overnight_10h']:.0f} in a 10h night)"
                if mean
                else " — no timing history yet, so no estimate"
            )
        )

    done = {"n": 0}
    done_counter = itertools.count(1)
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
    )

    # One live row per in-flight item. Items run for many minutes, so printing
    # only on completion leaves the terminal silent long enough to look hung —
    # which is exactly how it looked the first time this ran for real.
    progress = Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(bar_width=24),
        TextColumn("{task.fields[steps]}"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[note]}"),
        console=console,
        transient=True,
        disable=not console.is_terminal,
    )
    tasks: dict[int, TaskID] = {}

    def started(item) -> None:
        tasks[item.index] = progress.add_task(
            f"[cyan]{item.name}[/cyan] {item.prompt[:34]}",
            total=None,
            steps="",
            note="",
        )

    def ticked(item, tick) -> None:
        task = tasks.get(item.index)
        if task is None:
            return
        f = _tick_fields(tick)
        # The server's id travels on every row: "which job is this item"
        # must be readable off the screen, not asked afterwards.
        jid = f" [dim]{tick.job_id[:12]}[/dim]" if tick.job_id else ""
        progress.update(
            task,
            total=f["total"],
            completed=f["completed"],
            steps=f["steps"],
            note=f"[dim]{f['note']}[/dim]",
            description=f"[cyan]{item.name}[/cyan]{jid} "
            f"{_safe(item.prompt[:30])} · {f['description']}",
        )

    def report(item, b) -> None:
        # next() on a shared count is atomic; `done["n"] += 1` is
        # load/add/store and is called from every worker OUTSIDE the batch
        # lock, so simultaneous completions lost counts — under-reporting
        # progress and permanently inflating the remaining-time estimate.
        done["n"] = next(done_counter)
        task = tasks.pop(item.index, None)
        if task is not None:
            progress.remove_task(task)
        mark = "[green]ok[/green]" if item.status == "done" else f"[red]{_safe(item.error)}[/red]"
        # Estimate from THIS batch's own completions — a mean carried over
        # from unrelated jobs (a 2-step probe, a different duration) is worse
        # than no estimate at all.
        finished = [i for i in b.items if i.status == "done" and i.took_s]
        left = ""
        if finished and pending > done["n"]:
            mean_here = sum(i.took_s for i in finished) / len(finished)
            remaining = (pending - done["n"]) * mean_here / max(workers, 1)
            left = f" · ~{_hms(remaining)} left"
        progress.console.print(
            f"{_stamp()} [{done['n']}/{pending}] {item.name} {mark} "
            f"[dim]{item.took_s:.0f}s{left} · {item.prompt[:40]}[/dim]"
        )

    try:
        with progress:
            batch_mod.run(
                target,
                batch,
                concurrency=workers,
                timeout=timeout,
                on_start=started,
                on_tick=ticked,
                on_done=report,
                fetch_only=fetch_existing,
            )
    except (KeyboardInterrupt, SystemExit):
        # BOTH stages of the interrupt protocol. Naming only
        # KeyboardInterrupt meant a second ^C — which the escalation banner
        # actively invites — skipped this entire block, so the operator
        # lost the re-attach list and the ledger entry for an interrupted
        # overnight run: the single most important output there is.
        counts = batch.counts()
        attached = [i for i in batch.items if i.status == "running" and i.job_id]
        console.print(
            f"\n{_stamp()} interrupted — {counts['done']} done, "
            f"{counts['failed']} failed, {len(attached)} job(s) still "
            f"running on the server."
        )
        for item in attached:
            console.print(f"  [dim]{item.name} -> {item.job_id}[/dim]")
        console.print(
            f"re-run the same command to re-attach to them:\n"
            f"  uv run deepme batch run {prompts} -m {spec.profile} "
            f"--name {batch.name}"
        )
        runs.record("batch", spec, outcome="interrupted", detail={"name": batch.name, **counts})
        raise SystemExit(130) from None
    counts = batch.counts()
    runs.record(
        "batch",
        spec,
        outcome="ok" if not counts["failed"] else "partial",
        detail={"name": batch.name, **counts, "out_dir": batch.out_dir},
    )
    console.print(
        f"\n[green]{counts['done']}[/green] done, "
        f"[red]{counts['failed']}[/red] failed — {batch.out_dir}"
    )
    if fetch_existing and counts["pending"]:
        console.print(
            f"[dim]{counts['pending']} item(s) still parked — resume "
            f"submissions with: uv run deepme batch run {prompts} "
            f"-m {spec.profile} --name {batch.name}[/dim]"
        )
    if counts["failed"]:
        # NON-ZERO on failures. Exiting 0 with failed items meant an
        # overnight `batch run && publish` could not tell a clean night from
        # a half-failed one — the same rule `bench` and `ports` already
        # follow. The re-run command is on screen, so this is actionable.
        console.print(
            f"[dim]retry them with: uv run deepme batch run {prompts} "
            f"-m {spec.profile} --name {batch.name}[/dim]"
        )
        raise typer.Exit(1)


def _print_spec_census(census, *, full: bool) -> None:
    """The GPU verdict: what to never run again, in a dozen lines. The raw
    group-by-group listing only appears with --full."""
    from . import reconcile

    if not census:
        return
    if full:
        console.print(
            f"\n[bold]{len(census)} identical-request group(s)[/bold] across "
            f"batches (same prompt+seed+params):"
        )
        for g in census:
            seed = f"seed {g.seed}" if g.deterministic else "[yellow]no seed[/yellow]"
            note = "" if g.deterministic else " — reruns produce DIFFERENT bytes"
            who = ", ".join(f"{m.batch}[{m.index:04d} {m.status}]" for m in g.members)
            console.print(f"  {len(g.members)}x  {seed}  {_safe(g.prompt[:48])!r}{note}")
            console.print(f"      [dim]{who}[/dim]")

    s = reconcile.summarize_census(census)
    console.print(f"\n[bold]GPU verdict[/bold] ({len(census)} identical-request group(s)):")
    if s.already_generated:
        console.print(
            f"  [red]NEVER RUN[/red] {len(s.already_generated)} pending item(s): "
            f"seeded twins of content that ALREADY EXISTS — running them "
            f"recreates identical bytes:"
        )
        for name, idxs in s.by_batch(s.already_generated).items():
            console.print(f"    {name}: " + ", ".join(f"{i:04d}" for i in idxs))
    if s.redundant:
        console.print(
            f"  [yellow]RUN ONE PER SET[/yellow] — {len(s.redundant)} pending "
            f"item(s) are seeded twins of another pending item; only one of "
            f"each set is real work:"
        )
        for name, idxs in s.by_batch(s.redundant).items():
            console.print(
                f"    {name}: " + ", ".join(f"{i:04d}" for i in idxs) + "  (the skippable side)"
            )
    if s.seedless:
        console.print(
            f"  [cyan]YOUR CALL[/cyan] — {len(s.seedless)} pending item(s) repeat "
            f"a request WITHOUT a seed (each run differs; duplicates of "
            f"intent, not of content):"
        )
        for name, idxs in s.by_batch(s.seedless).items():
            console.print(f"    {name}: " + ", ".join(f"{i:04d}" for i in idxs))
    if not (s.already_generated or s.redundant or s.seedless):
        console.print("  [green]none pending[/green] — every pending item is unique work.")
    if not full:
        console.print("  [dim](--full lists every group with prompts and seeds)[/dim]")


@batch_app.command("journal")
def batch_journal(
    name: str = typer.Argument(..., help="Batch name"),
    item: int = typer.Option(None, "--item", help="Only this item index"),
    lines: int = typer.Option(200, "-n", help="Most recent entries to show"),
) -> None:
    """The append-only history of a batch: every job id an item ever held,
    every retry, every delivered file and its MD5.

    state.json is only CURRENT state — a resubmit overwrites job_id — so
    this is the only record that can answer "which server job produced the
    file sitting in slot N, and are those bytes the ones it delivered".
    """
    from . import batch as batch_mod

    path = batch_mod.BATCH_DIR / name / "journal.jsonl"
    if not path.exists():
        raise SystemExit(
            f"no journal for {name!r} at {path} — batches run before "
            f"journalling was added have none (see `deepme batch list`)"
        )
    rows = []
    for raw in path.read_text().splitlines():
        with contextlib.suppress(ValueError):
            rec = json.loads(raw)
            if item is None or int(rec.get("item", -1)) == item:
                rows.append(rec)
    if not rows:
        console.print(f"no entries{'' if item is None else f' for item {item:04d}'}")
        return
    table = Table(box=None, pad_edge=False)
    for col in ("when", "item", "event", "detail"):
        table.add_column(col, overflow="fold")
    for rec in rows[-lines:]:
        detail = " ".join(
            f"{k}={v}"
            for k, v in rec.items()
            if k not in ("ts", "item", "event") and v not in (None, "")
        )
        idx = rec.get("item")
        table.add_row(
            str(rec.get("ts", ""))[:19],
            f"{int(idx):04d}" if isinstance(idx, int) else "-",
            str(rec.get("event", "")),
            _safe(detail[:160]),
        )
    console.print(table)
    console.print(f"[dim]{len(rows)} entry(ies) · {path}[/dim]")


@batch_app.command("audit")
def batch_audit(
    name: str = typer.Option(None, help="One batch (default: every batch on disk)"),
) -> None:
    """Find items whose delivered content may not be their own.

    Answers the misattribution question from records already on disk:
    items that a resume would ADOPT instead of generating, items sharing
    one server job, prompts that changed under an index-matched batch, and
    done-but-missing files.
    """
    from . import reconcile

    batches = reconcile.load_batches()
    if name:
        batches = [b for b in batches if b.name == name]
        if not batches:
            raise SystemExit(f"no batch named {name!r} — see `deepme batch list`")
    found = [s for b in batches for s in reconcile.audit_batch(b)]
    bad = reconcile.unreadable_batches()
    if bad:
        console.print(f"[red]unreadable state[/red]: {', '.join(bad)}")
    if not found:
        console.print(f"{len(batches)} batch(es) checked — nothing suspicious.")
        return
    table = Table(box=None, pad_edge=False)
    for col in ("batch", "item", "finding", "detail"):
        table.add_column(col, overflow="fold")
    for s in found:
        colour = (
            "red" if s.kind in ("stale-adoption", "shared-job-id", "prompt-drift") else "yellow"
        )
        table.add_row(s.batch, f"{s.index:04d}", f"[{colour}]{s.kind}[/{colour}]", s.detail)
    console.print(table)
    counts: dict[str, int] = {}
    for s in found:
        counts[s.kind] = counts.get(s.kind, 0) + 1
    console.print(
        "\n" + " · ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + f" — {len(found)} total"
    )
    raise typer.Exit(1)


@batch_app.command("dedupe")
def batch_dedupe(
    apply: bool = typer.Option(
        False,
        "--apply",
        help="Actually delete and reset. Without it: full dry run, no changes.",
    ),
    extra: list[str] = typer.Option(
        [], "--dir", help="Additional clip directories to include (repeatable)"
    ),
    full: bool = typer.Option(
        False, "--full", help="Also list every identical-request group in detail"
    ),
) -> None:
    """Incident repair: hash every clip, keep only the OLDEST copy of any
    byte-identical content, delete the newer copies, and reset their items
    to pending with job ids CLEARED so re-running regenerates from the
    prompt — never re-adopts the wrong server job.

    Dry run by default; nothing is touched until --apply.
    """
    from . import reconcile

    batches = reconcile.load_batches()
    dirs, owners = reconcile.batch_clip_index(batches)
    dirs.extend(Path(d) for d in extra)
    console.print(f"[dim]hashing clips in {len(dirs)} director(ies)…[/dim]")
    plan = reconcile.build_plan(reconcile.scan_dirs(dirs), owners)
    _print_spec_census(reconcile.spec_census(batches), full=full)

    if not plan.groups:
        console.print(f"scanned {plan.scanned} clip(s) — no duplicated content. Nothing to do.")
        return
    table = Table(box=None, pad_edge=False)
    for col in ("content", "keep (oldest)", "delete (newer copies)"):
        table.add_column(col, overflow="fold")
    for g in plan.groups:
        table.add_row(g.digest[:10], str(g.keep), "\n".join(str(p) for p in g.drop))
    console.print(table)
    console.print(
        f"\nscanned {plan.scanned} clip(s): {len(plan.groups)} duplicated "
        f"content group(s), {len(plan.drops)} file(s) to delete, "
        f"{len(plan.skips)} item(s) -> SKIPPED (their output is the kept "
        f"copy; they will NEVER regenerate), {len(plan.resets)} item(s) -> "
        f"pending (different request, so the file was never their output)"
    )
    for name, index, path, keeper in sorted(plan.skips):
        console.print(
            f"  [green]skip[/green] {name}[{index:04d}] — deletes {path}, [dim]keeps {keeper}[/dim]"
        )
    for name, index, path in sorted(plan.resets):
        console.print(f"  [yellow]reset[/yellow] {name}[{index:04d}] — loses {path}")

    # The operator's next move, aggregated: which batches need a re-run,
    # which items each regenerates, and the exact commands — no manual
    # cross-comparison of fifty scattered reset lines.
    by_batch: dict[str, list[int]] = {}
    for name, index, _path in plan.resets:
        by_batch.setdefault(name, []).append(index)
    if by_batch:
        console.print("\nresets by batch:")
        for name in sorted(by_batch):
            idxs = sorted(by_batch[name])
            console.print(f"  {name}  {len(idxs)} item(s): " + ", ".join(f"{i:04d}" for i in idxs))
        console.print("\nre-run these after --apply (resumes regenerate only the resets):")
        from . import batch as batch_mod

        for name in sorted(by_batch):
            b = batch_mod.Batch.load(name)
            prof = b.profile if b else "<profile>"
            console.print(f"  uv run deepme batch run {name}.jsonl -m {prof} --name {name}")
    if not apply:
        console.print(
            "\n[yellow]dry run[/yellow] — nothing was changed. "
            "Re-run with --apply to execute, then run the commands above."
        )
        return
    deleted, reset = reconcile.apply_plan(plan)
    runs.record(
        "batch-dedupe",
        None,
        outcome="ok",
        detail={
            "deleted": deleted,
            "reset": reset,
            "groups": len(plan.groups),
            "batches": {k: sorted(v) for k, v in by_batch.items()},
        },
    )
    console.print(
        f"\n[green]done[/green]: deleted {deleted} file(s), reset {reset} "
        f"item(s) across {len(by_batch)} batch(es). Run the commands above "
        f"to regenerate."
    )


@batch_app.command("fixup")
def batch_fixup(
    apply: bool = typer.Option(False, "--apply", help="Write the skips. Without it: full dry run."),
    skip_seedless: bool = typer.Option(
        False,
        "--skip-seedless",
        help="Also skip seedless duplicate requests (same prompt, no seed — "
        "each run differs, so this is an intent decision)",
    ),
) -> None:
    """Make the GPU verdict executable: mark every duplicate generation
    SKIPPED in its batch state, so `batch run` only ever runs unique work.

    Skipped items carry their reason ("twin of generated longer2[0019]"),
    are journaled, and show in `batch list` — no invisible backlog. Then
    just re-run your batches: nothing duplicated will generate.
    """
    from . import batch as batch_mod
    from . import reconcile

    batches = reconcile.load_batches()
    summary = reconcile.summarize_census(reconcile.spec_census(batches))
    targets = list(summary.already_generated) + list(summary.redundant)
    if skip_seedless:
        targets += list(summary.seedless)
    if not targets:
        console.print("nothing to fix up — every pending item is unique work.")
        return
    for m in sorted(targets, key=lambda m: (m.batch, m.index)):
        reason = summary.sources.get((m.batch, m.index), "duplicate request")
        console.print(f"  skip {m.batch}[{m.index:04d}] — {reason}")
    if summary.seedless and not skip_seedless:
        console.print(
            f"  [dim]{len(summary.seedless)} seedless duplicate request(s) left "
            f"alone (--skip-seedless to include them)[/dim]"
        )
    if not apply:
        console.print(
            f"\n[yellow]dry run[/yellow] — {len(targets)} item(s) would be "
            f"skipped. Re-run with --apply."
        )
        return
    skipped = reconcile.apply_fixup(summary, skip_seedless=skip_seedless)
    runs.record("batch-fixup", None, outcome="ok", detail={"skipped": skipped})
    console.print(f"\n[green]done[/green]: {sum(skipped.values())} item(s) skipped.")
    console.print("remaining unique work per batch:")
    for name in sorted(skipped):
        b = batch_mod.Batch.load(name)
        if b is None:
            continue
        c = b.counts()
        left = c["pending"] + c["running"]
        console.print(
            f"  {name}: {left} to run, {c['skipped']} skipped, {c['done']} done"
            + (
                f" — uv run deepme batch run {name}.jsonl -m {b.profile} --name {name}"
                if left
                else ""
            )
        )


@batch_app.command("list")
def batch_list() -> None:
    """Every batch on disk with its counts — the names `status` accepts."""
    from . import batch as batch_mod

    rows = sorted(batch_mod.BATCH_DIR.glob("*/state.json"))
    if not rows:
        console.print(f"no batches under {batch_mod.BATCH_DIR}")
        return
    table = Table(box=None, pad_edge=False)
    for col in ("batch", "profile", "done", "failed", "pending", "skipped", "running"):
        table.add_column(col)
    for path in rows:
        b = None
        with contextlib.suppress(Exception):
            b = batch_mod.Batch.load(path.parent.name)
        if b is None:
            # A row that cannot load is still a fact worth showing — a
            # silently skipped batch reads as "does not exist".
            table.add_row(path.parent.name, "[red]unreadable state[/red]", "-", "-", "-", "-", "-")
            continue
        c = b.counts()
        table.add_row(
            b.name,
            b.profile,
            str(c["done"]),
            str(c["failed"]),
            str(c["pending"]),
            str(c["skipped"]),
            str(c["running"]),
        )
    console.print(table)


@batch_app.command("status")
def batch_status(
    name: str = typer.Argument(..., help="Batch name"),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable, for scripts"),
) -> None:
    """Progress of a batch — safe to run while it is going."""
    from . import batch as batch_mod

    batch = batch_mod.Batch.load(name)
    if batch is None:
        raise SystemExit(f"no batch named {name!r} — see `deepme batch list`")
    counts = batch.counts()
    if as_json:
        # Every item with its identity and outcome: the shape an overnight
        # watchdog needs to decide whether to page someone.
        console.print_json(
            json.dumps(
                {
                    "name": batch.name,
                    "profile": batch.profile,
                    "out_dir": batch.out_dir,
                    "counts": counts,
                    "items": [
                        {
                            "index": i.index,
                            "status": i.status,
                            "job_id": i.job_id,
                            "file": i.file,
                            "took_s": round(i.took_s, 1),
                            "started_at": i.started_at,
                            "error": str(i.error) if i.error else None,
                        }
                        for i in batch.items
                        if i.status != batch_mod.PENDING
                    ],
                }
            )
        )
        return
    console.print(f"[bold]{name}[/bold] ({batch.profile}) -> {batch.out_dir}")
    console.print(
        f"done={counts['done']} failed={counts['failed']} "
        f"pending={counts['pending']} running={counts['running']}"
    )
    # In-flight work, with how long it has been going. A count alone cannot
    # distinguish a fleet that is busy from one that is wedged, which is the
    # question anyone running this from a second shell is actually asking.
    import time as _time

    # Cross-check against the SERVER, not just our own counters: "running"
    # in the state file with nothing on the server is exactly the stall that
    # burned an overnight run — make it visible from any shell.
    on_server: set[str] = set()
    queued_tickets: set[str] = set()
    server_ok = False
    queue_ok = False
    with contextlib.suppress(Exception):
        from . import gatewayctl, media

        # ONE parser, shared with the stall verdict. A fleet that could not
        # be asked reports NO running list, and reading its silence as "these
        # jobs are gone" printed "NOT on server" for every healthy item on
        # it. `complete` is the trust bit — it is false for an unreachable
        # fleet AND for a reachable one that reported no running list at all,
        # which the check written here missed.
        view = media.capacity_view(gatewayctl.capacity())
        server_ok = view.complete
        on_server.update(view.running)
    # The QUEUE too. Checking only the running set meant a `q_` ticket
    # destroyed by a balancer restart — the one state where a batch is
    # silently dead — was reported as a confident "queued".
    with contextlib.suppress(Exception):
        from . import gatewayctl

        qfleets = gatewayctl.queue().get("fleets", [])
        queue_ok = bool(qfleets) and all(f.get("reachable", True) for f in qfleets)
        for f in qfleets:
            queued_tickets.update(str(t.get("ticket")) for t in f.get("next", []))
    for item in batch.items:
        if item.status == "running":
            elapsed = _time.time() - item.started_at if item.started_at else None
            for_how_long = f"[cyan]{_hms(elapsed)}[/cyan]" if elapsed else "[dim]?[/dim]"
            if not server_ok:
                where = "[dim]server unreachable[/dim]"
            elif item.job_id and item.job_id in on_server:
                where = "[green]generating[/green]"
            elif item.job_id and item.job_id.startswith("q_"):
                # A ticket lives in balancer MEMORY: a restart destroys it,
                # and reporting that as "queued" hid the one state where an
                # overnight batch is silently dead.
                if queue_ok and item.job_id not in queued_tickets:
                    where = "[red]TICKET GONE — re-run to resubmit[/red]"
                else:
                    where = "[yellow]queued[/yellow]"
            elif item.job_id:
                where = "[red]NOT on server[/red]"
            else:
                where = "[red]never submitted[/red]"
            console.print(
                f"[cyan]{item.name}[/cyan] running {for_how_long} {where} "
                f"[dim]{item.job_id or ''} {_safe(item.prompt[:44])}[/dim]"
            )
    for item in batch.items:
        if item.status == "failed":
            console.print(
                f"[red]{item.name}[/red] {_safe(str(item.error))} "
                f"[dim]{_safe(item.prompt[:60])}[/dim]"
            )
    if counts["running"]:
        console.print(
            f"[dim]updated {_batch_age(batch)} ago — "
            f"if this stops moving, check `deepme serve logs -m "
            f"{batch.profile}`[/dim]"
        )


showcase_app = typer.Typer(help="Demo reels — validation that leaves assets behind")
app.add_typer(showcase_app, name="showcase")


@showcase_app.command("list")
def showcase_list(model: str = ModelOpt) -> None:
    """Show the clips in this profile's reel."""
    from . import showcase

    spec = profiles.load(model)
    table = Table(box=None, pad_edge=False)
    for col in ("clip", "demonstrates", "length", "seed"):
        table.add_column(col, overflow="fold")
    for clip in showcase.load(spec.profile):
        table.add_row(
            clip.name,
            clip.caption,
            f"{clip.seconds}s" if clip.seconds else "default",
            str(clip.seed) if clip.seed is not None else "-",
        )
    console.print(table)
    console.print(f"[dim]{showcase.reel_path(spec.profile)}[/dim]")


@showcase_app.command("run")
def showcase_run(
    model: str = ModelOpt,
    out: str = typer.Option(None, "--out", "-o", help="Gallery directory"),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Bypass the balancer and send everything to one replica",
    ),
    concurrency: int = typer.Option(0, help="Simultaneous requests (default: all clips at once)"),
    timeout: float = typer.Option(5400),
) -> None:
    """Generate the whole reel at once — the concurrency proof, kept as assets."""

    from . import showcase

    spec = profiles.load(model)
    clips = showcase.load(spec.profile)
    # Through the balancer, so clips spread across replicas — which is
    # precisely the behaviour being demonstrated.
    front, _via = front_end(spec, direct=direct)
    console.print(
        f"{_stamp()} generating {len(clips)} clips via {front.api_base} "
        f"[dim](this is minutes of GPU time — the output is the reel)[/dim]"
    )
    done = {"n": 0}

    def report(result) -> None:
        done["n"] += 1
        mark = "[green]ok[/green]" if result.ok else f"[red]{_safe(result.error)}[/red]"
        console.print(
            f"{_stamp()} [{done['n']}/{len(clips)}] {result.clip.name}: {mark} "
            f"[dim]{result.seconds_taken:.0f}s[/dim]"
        )

    results = showcase.run(
        front,
        clips,
        out_dir=Path(out) if out else None,
        concurrency=concurrency,
        timeout=timeout,
        on_done=report,
    )
    ok = [r for r in results if r.ok]
    gallery = results[0].path.parent if ok and results[0].path else showcase.DEFAULT_GALLERY
    runs.record(
        "showcase",
        spec,
        outcome="ok" if len(ok) == len(results) else "partial",
        detail={"clips": len(results), "ok": len(ok), "gallery": str(gallery)},
    )
    console.print(
        f"\n[green]{len(ok)}/{len(results)}[/green] clips in {gallery} "
        f"(manifest.json records prompts, seeds and timings)"
    )
    if len(ok) != len(results):
        # A partial reel is not a success: exit non-zero so a scripted
        # `showcase run && publish` stops here.
        raise typer.Exit(1)


store_app = typer.Typer(help="Shared weight store (/dev/shm) — usage and cleanup")
app.add_typer(store_app, name="store")


def _run_store(action: str) -> str:
    """Run the store tool on the server from this repo's own source."""
    from . import config as cfg
    from . import remote as rmt
    from . import sharedstore

    src = Path(sharedstore.__file__).read_text()
    path = f"{cfg.REMOTE_HOME}/deepme-tools/sharedstore.py"
    rmt.run(f"mkdir -p {cfg.REMOTE_HOME}/deepme-tools")
    rmt.write_remote_file(path, src)
    return rmt.run(f"{cfg.venv_python('sglang')} {path} {action}", check=False).stdout.strip()


@store_app.command("status")
def store_status() -> None:
    """How much RAM the shared weight store holds."""
    out = _run_store("status")
    try:
        data = json.loads(out)
    except ValueError:
        console.print(out or "[red]no response[/red]")
        return
    console.print(
        f"{data['dir']}: [bold]{data['buffers']}[/bold] buffers, "
        f"[bold]{data['bytes'] / 1024**3:.1f}GiB[/bold]"
    )
    if data.get("orphans"):
        console.print(
            f"[yellow]{data['orphans']} orphaned[/yellow] "
            f"({data['orphan_bytes'] / 1024**3:.1f}GiB) — mapped by nothing. "
            f"Left behind by stopped models or a key-scheme change; "
            f"`deepme store clear` reclaims them."
        )
    console.print(
        "[dim]Paid once for all replicas of a model. Segments outlive the "
        "processes that made them — `deepme store clear` reclaims any that "
        "no live process is mapping.[/dim]"
    )


@store_app.command("clear")
def store_clear(
    all_segments: bool = typer.Option(
        False,
        "--all",
        help="Also remove segments that ARE mapped (frees nothing; breaks "
        "running replicas). Only for cleaning up after a crash.",
    ),
) -> None:
    """Reclaim shared-store RAM that no running replica is using."""
    # Its own help text says --all breaks running replicas — a flag that
    # destructive cannot be one keystroke from a habit-typed command.
    if all_segments and not typer.confirm(
        "--all removes MAPPED segments and breaks running replicas. Continue?",
        default=False,
    ):
        raise SystemExit("aborted")
    out = _run_store("clear --all" if all_segments else "clear")
    console.print(out)
    runs.record("store-clear", None, outcome="ok", detail={"all": all_segments})


@app.command("memprofile")
def memprofile_cmd(model: str = ModelOpt) -> None:
    """Break down a deployment's host RAM: weights vs per-process overhead."""
    from . import memprofile

    spec = profiles.load(model)
    prof = memprofile.profile(spec)

    table = Table(box=None, pad_edge=False, title="checkpoint weights")
    for col in ("component", "total", f"per rank (tp={prof.tp})", "dtypes"):
        table.add_column(col)
    for c in prof.components:
        table.add_row(
            c.name + (" [dim](sharded)[/dim]" if c.sharded else ""),
            f"{c.total_gib:.1f}GiB",
            f"{c.per_rank_gib:.1f}GiB",
            ", ".join(sorted(c.dtypes)),
        )
    console.print(table)

    if not prof.procs:
        console.print(f"[dim]{spec.unit} is not running — no measurement[/dim]")
        return
    live = Table(box=None, pad_edge=False, title="live processes")
    for col in ("pid", "process", "rss", "pss", "private", "shared"):
        live.add_column(col)
    for p in prof.procs:
        live.add_row(
            str(p.pid),
            p.label,
            f"{p.rss_gib:.1f}GiB",
            f"{p.pss_gib:.1f}GiB",
            f"{p.private_gib:.1f}GiB",
            f"{p.shared_gib:.1f}GiB",
        )
    console.print(live)

    predicted, measured = prof.predicted_per_rank_gib, prof.measured_per_rank_gib
    console.print(
        f"\npredicted weights/rank [bold]{predicted:.1f}GiB[/bold] vs "
        f"measured RSS/rank [bold]{measured:.1f}GiB[/bold] — "
        f"unexplained [bold]{prof.unexplained_gib:.1f}GiB[/bold]"
    )
    console.print(
        "[dim]Weights are identical across replicas and can be shared; the "
        "unexplained remainder cannot. PSS well below RSS would mean sharing "
        "is already happening.[/dim]"
    )


@app.command()
def plan(
    repo: str = typer.Option(
        None, "--repo", help="Plan an arbitrary HF repo instead of the profiles"
    ),
    gpu_mem: float = typer.Option(0.90, help="GPU memory fraction to budget"),
    include_running: bool = typer.Option(
        False,
        "--include-running",
        help="Also plan models that are already deployed (as if stopped)",
    ),
) -> None:
    """What fits in the space that is FREE right now, and where it would go."""
    from . import hfmeta
    from . import resources as res

    cap = res.probe_capacity()
    claims = res.current_claims()
    avail = res.availability(cap, claims)
    running = {c.profile for c in claims if c.profile}

    console.print(
        f"[bold]Free now[/bold]  GPUs {list(avail.free_gpus)} · "
        f"{len(avail.free_cpus)} CPUs · {avail.free_ram_gib:.0f}GiB RAM"
    )
    if claims:
        held = ", ".join(
            f"{c.profile or c.unit} (GPUs {list(c.gpus)}"
            + (f", {c.ram_gib:.0f}GiB RAM" if c.ram_gib else "")
            + ")"
            for c in claims
        )
        console.print(f"[dim]held by: {held}[/dim]")

    table = Table(box=None, pad_edge=False)
    for col in ("model", "weights", "verdict"):
        table.add_column(col, overflow="fold")

    from . import engines as engines_mod

    # (name, weight_gib, heads, spec-or-None). A bare repo has no profile, so
    # it is planned under the default tensor-parallel assumption.
    candidates: list[tuple[str, float, int, profiles.RunSpec | None]] = []
    if repo:
        meta = hfmeta.get(repo)
        candidates.append(
            (
                repo,
                meta.deployable_gib,
                int((meta.config_dims or {}).get("num_attention_heads", 0)),
                None,
            )
        )
    else:
        for name in profiles.list_profiles():
            if name in running and not include_running:
                continue
            try:
                spec = profiles.load(name)
            except Exception:
                continue
            candidates.append((name, spec.weight_gib, spec.attn_heads, spec))

    for name, weight_gib, heads, spec_obj in candidates:
        # How the ENGINE divides weights decides the whole calculation.
        derate, how, placement = 1.0, "tp", None
        if spec_obj is not None:
            edef = engines_mod.get(spec_obj.engine)
            placement = edef.placement_for(spec_obj)
            if not edef.supports(engines_mod.Feature.TENSOR_PARALLEL):
                derate, how = 0.85, "layer-split"
        if placement is not None and not placement.resident:
            # Weights stream from host RAM: GPUs are cheap, RAM is the limit.
            need = weight_gib * max(placement.host_copies, 1)
            ok = need <= avail.free_ram_gib
            colour = "green" if ok else "red"
            table.add_row(
                name,
                f"{weight_gib:.0f}GiB",
                f"[{colour}]{placement.how}[/{colour}] [dim]— needs "
                f"{need:.0f}GiB host RAM ({placement.host_copies} ranks), "
                f"{avail.free_ram_gib:.0f}GiB free[/dim]",
            )
            continue
        if placement is not None and placement.divisor == 1 and spec_obj is not None:
            # Replicated parallelism: every GPU holds a full copy.
            heads = 0
        plans = res.gpu_plans(
            weight_gib=weight_gib,
            gpu_mem=gpu_mem,
            cap=cap,
            avail=avail,
            attn_heads=heads,
            derate=derate,
            how=how,
        )
        if plans:
            best = plans[0]
            extra = (
                f"  [dim](or {', '.join(f'{how}={p.tp}' for p in plans[1:4])})[/dim]"
                if len(plans) > 1
                else ""
            )
            verdict = f"[green]{best.describe()}[/green]{extra}"
        elif not avail.free_gpus:
            verdict = "[red]no free GPUs[/red]"
        else:
            budget = config.per_gpu_weight_budget_gib(gpu_mem) * derate
            biggest = len(avail.free_gpus)
            why = (
                f"needs >{budget:.0f}GiB/GPU even at tp={biggest}"
                if weight_gib / biggest > budget
                else f"tp must divide {heads} attention heads; "
                f"no divisor <= {biggest} leaves it under {budget:.0f}GiB/GPU"
            )
            verdict = f"[red]does not fit[/red] [dim]({why})[/dim]"
        table.add_row(name, f"{weight_gib:.0f}GiB", verdict)

    console.print(table)
    console.print(
        "[dim]Placements are socket-local where possible and listed "
        "fewest-GPUs-first, so planning one model leaves room for the next. "
        "Running models are excluded (--include-running to plan them too).[/dim]"
    )


@app.command("resources")
def resources_cmd(
    refresh: bool = typer.Option(False, help="Re-probe machine topology"),
) -> None:
    """Resource allocation: capacity, live claims per unit, and what's free.

    GPU models claim GPUs; CPU-mode models claim only cores + RAM, so they
    colocate alongside GPU work."""
    from . import resources

    cap = resources.probe_capacity(refresh=refresh)
    claims = resources.current_claims()
    avail = resources.availability(cap, claims)
    console.print(
        f"[bold]Capacity[/bold] gpus={len(cap.gpus)} cpus={len(cap.cpus)} "
        f"ram={cap.ram_gib:.0f}GiB  "
        f"numa={{{', '.join(f'{n}:{len(c)}cpu' for n, c in sorted(cap.numa.items()))}}}"
    )
    table = Table(title="live claims")
    for col in ("unit", "profile", "gpus", "cpus", "ram", "port"):
        table.add_column(col, overflow="fold")
    for c in claims:
        table.add_row(
            c.unit,
            c.profile or "-",
            resources.format_cpu_list(c.gpus) or "-",
            resources.format_cpu_list(c.cpus) or "(unpinned)",
            f"{c.ram_gib:.0f}GiB" if c.ram_gib else "(uncapped)",
            str(c.port or "-"),
        )
    console.print(table)
    console.print(
        f"[bold]Free[/bold] gpus={list(avail.free_gpus)}  "
        f"cpus={len(avail.free_cpus)} ({resources.format_cpu_list(avail.free_cpus)[:60]}…)  "
        f"ram={avail.free_ram_gib:.0f}GiB  "
        f"[dim](reserved {resources.RESERVED_CPUS} cpus / "
        f"{resources.RESERVED_RAM_GIB:.0f}GiB for the system)[/dim]"
    )


@app.command()
def gpus() -> None:
    """Per-GPU stats and every GPU compute process with its command."""
    console.print(gpus_mod.summary())


@app.command()
def reclaim(
    kill: bool = typer.Option(False, "--kill", help="Actually kill them (default lists only)"),
) -> None:
    """Free GPUs of processes that aren't deepme services.

    Lists the candidates by default; `--kill` acts and records to the
    ledger. A kill command with zero friction was the audit's finding — it
    is exactly the command someone runs in a hurry.
    """
    if not kill:
        console.print(gpus_mod.list_foreign())
        console.print("[dim]re-run with --kill to remove them[/dim]")
        return
    out = gpus_mod.reclaim()
    console.print(out)
    runs.record("reclaim", None, outcome="ok", detail={"log": out[-1000:]})


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@download_app.command("start")
def download_start(model: str = ModelOpt) -> None:
    """Start (or resume) the weight download for a profile."""
    spec = profiles.load(model)
    console.print(download.start(spec))
    runs.record(
        "download-start",
        spec,
        outcome="started",
        detail={"format": spec.format, "variant": spec.variant},
    )


@download_app.command("restart")
def download_restart(model: str = ModelOpt) -> None:
    spec = profiles.load(model)
    console.print(download.restart(spec))
    runs.record("download-restart", spec, outcome="restarted")


@download_app.command("status")
def download_status(model: str = ModelOpt) -> None:
    console.print(download.status(profiles.load(model)))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@serve_app.command("wait")
def serve_wait(
    model: str = ModelOpt,
    timeout: float = typer.Option(2400, help="Give up after this many seconds"),
    replicas: bool = typer.Option(
        True,
        "--replicas/--single",
        help="Wait for every running replica of the profile, not just replica 0",
    ),
) -> None:
    """Block until the model answers, then exit. Use this instead of polling.

    Readiness is a first-class operation: ad-hoc `until curl ...; sleep` loops
    around other commands get the poll interval, the failure cases and the
    timeout wrong every time they are rewritten.
    """
    from . import resources as res

    spec = profiles.load(model)
    specs = [spec]
    if replicas:
        ports = sorted(c.port for c in res.current_claims() if c.profile == spec.profile and c.port)
        specs = [profiles.load(model, port=p) for p in ports] or [spec]
    failed = False
    for s in specs:
        ok, detail = serve.wait_healthy(s, timeout_s=timeout)
        mark = "[green]healthy[/green]" if ok else f"[red]{_safe(detail)}[/red]"
        console.print(f"{_stamp()} {s.api_base}: {mark}")
        failed |= not ok
    if failed:
        raise SystemExit(1)


@serve_app.command("scale")
def serve_scale(
    model: str = ModelOpt,
    count: int = typer.Option(..., "-n", "--replicas", help="How many copies to run"),
    stagger: bool = typer.Option(
        True,
        "--stagger/--no-stagger",
        help="Wait for each replica to load before starting the next. Loading "
        "several at once stages every copy's weights through host RAM "
        "simultaneously, which is how a machine gets OOM-killed.",
    ),
    timeout: float = typer.Option(2400, help="Per-replica readiness timeout"),
    force: bool = typer.Option(False, "--force", help="Re-layout even with jobs running/queued"),
) -> None:
    """Run N copies of a model across free GPUs, for concurrent requests."""
    spec = profiles.load(model)
    _assert_profile_drained(spec, force=force)
    specs = serve.replica_specs(spec, count)
    # Stop the WHOLE previous generation first. A staggered bring-up rewrites
    # units one at a time, so any replica still running an older unit keeps
    # crash-looping against the new layout (systemd restarts it, it grabs the
    # rendezvous port, the incoming replica dies with EADDRINUSE) for as long
    # as the scale takes.
    serve.stop_profile(spec)
    console.print(
        "\n".join(
            f"  replica {s.replica}: GPUs {list(s.gpus)} port {s.port} ({s.unit})" for s in specs
        )
    )
    # Only the FIRST replica needs to go alone: it builds the shared weight
    # store and carries the big load peak. The rest map pages it already
    # created (the store's flock makes that safe), so serialising them just
    # multiplies the wait — 4 replicas went from ~10 minutes to ~4.
    first, rest = specs[0], specs[1:]
    console.print(f"[dim]{_stamp()} starting {first.unit} (builds the store)[/dim]")
    serve.start(first)
    if stagger:
        ok, detail = serve.wait_healthy(first, timeout_s=timeout)
        console.print(
            f"{_stamp()} {first.unit}: "
            + ("[green]healthy[/green]" if ok else f"[red]{_safe(detail)}[/red]")
        )
        if not ok:
            raise SystemExit(
                f"{first.unit} did not come up — `deepme serve diag -m {first.profile}` has the traceback"
            )

    for s in rest:
        console.print(f"[dim]{_stamp()} starting {s.unit}[/dim]")
        serve.start(s)
    if stagger and rest:
        import concurrent.futures as _f

        with _f.ThreadPoolExecutor(max_workers=len(rest)) as pool:
            futures = {pool.submit(serve.wait_healthy, s, timeout): s for s in rest}
            failed = []
            for fut in _f.as_completed(futures):
                s = futures[fut]
                ok, detail = fut.result()
                console.print(
                    f"{_stamp()} {s.unit}: "
                    + ("[green]healthy[/green]" if ok else f"[red]{_safe(detail)}[/red]")
                )
                if not ok:
                    failed.append(s.unit)
        if failed:
            raise SystemExit(
                f"replicas did not come up: {failed} — `deepme serve diag -m` each for the traceback"
            )
    runs.record(
        "scale",
        spec,
        outcome="ok",
        detail={
            "replicas": [{"unit": s.unit, "gpus": list(s.gpus), "port": s.port} for s in specs],
        },
    )
    console.print(
        f"[dim]put one API in front of them: "
        f"uv run deepme balancer start -m {spec.profile} --setup[/dim]"
    )


@serve_app.command("start")
def serve_start(
    model: str = ModelOpt,
    engine: str = typer.Option(None, "--engine", "-e", help="Override profile engine"),
    tp: int = typer.Option(None, help="Override tensor parallel size"),
    max_len: int = typer.Option(None, help="Override max context length"),
    gpu_mem: float = typer.Option(None, help="Override GPU memory fraction"),
    kv_ram: int = typer.Option(None, "--kv-ram", help="Override RAM KV tier GB"),
    spec_decode: bool = typer.Option(
        None, "--spec/--no-spec", help="Override speculative decoding"
    ),
    extra: list[str] = typer.Option(None, "--extra", help="Extra raw engine args (repeatable)"),
    wait: bool = typer.Option(True, help="Wait for the API to become healthy"),
) -> None:
    """Deploy a model profile (stops conflicting units automatically)."""
    spec = profiles.load(
        model,
        engine=engine,
        tp=tp,
        max_len=max_len,
        gpu_mem=gpu_mem,
        kv_ram_gb=kv_ram,
        spec_decode=spec_decode,
        extra_args=tuple(extra) if extra else None,
    )
    if not download.is_complete(spec):
        console.print(
            f"[red]weights not fully downloaded[/red] — run: uv run deepme download start -m {spec.profile}"
        )
        raise typer.Exit(1)
    console.print(
        f"Launching [bold]{spec.profile}[/bold] ({spec.engine}, tp={spec.tp}, "
        f"gpus={list(spec.gpus)}, max_len={spec.max_len}, port={spec.port}) ..."
    )
    stopped = serve.start(spec)
    if stopped:
        console.print(f"stopped conflicting units: {stopped}")
    if wait:
        console.print("Waiting for weights to load (can take ~5-15 min) ...")
        ok, reason = serve.wait_healthy(spec)
        runs.record(
            "serve-start",
            spec,
            outcome="healthy" if ok else "failed",
            detail={"stopped_units": stopped, "reason": reason},
        )
        if ok:
            console.print(f"[green]API is up[/green]: {spec.api_base}/v1")
        else:
            console.print(f"[red]Service did not become healthy[/red]: {reason}")
            raise typer.Exit(1)
    else:
        runs.record(
            "serve-start",
            spec,
            outcome="launched-nowait",
            detail={"stopped_units": stopped},
        )


@serve_app.command("stop")
def serve_stop(
    model: str = ModelOpt,
    force: bool = typer.Option(False, "--force", help="Stop even with jobs running/queued on it"),
) -> None:
    spec = profiles.load(model)
    _assert_profile_drained(spec, force=force)
    serve.stop(spec)
    runs.record("serve-stop", spec, outcome="stopped")
    console.print("stopped")


def _assert_profile_drained(spec, *, force: bool) -> None:
    """The drain discipline, on EVERY stop path.

    `fleet switch` drains before stopping and the balancer refuses unsafe
    restarts — but `serve stop`/`restart`/`scale` killed replicas
    mid-generation with no check at all. A rule enforced on one path and
    left to memory on the others is not a rule.
    """
    from . import balancerctl

    if force:
        return
    busy = waiting = 0
    probed = False
    # Probe inside the suppress; VERDICT outside it — a raise inside its own
    # suppress block is a guard that never fires.
    with contextlib.suppress(Exception):
        snap = balancerctl.stats(spec.profile)
        busy = sum(b.get("inflight", 0) for b in snap.get("backends", []))
        waiting = int(balancerctl.queue(spec.profile).get("waiting") or 0)
        probed = True
    if not probed:
        # NO ANSWER IS NOT "NOTHING RUNNING". The counters start at zero, so
        # an unreachable or crash-looping balancer silently satisfied this
        # guard and the stop went through into live generations — the same
        # trap `reboot` refuses on.
        raise SystemExit(
            f"cannot ask {spec.profile}'s balancer what it is running "
            f"(unreachable). Stopping blind can kill paying work — retry, "
            f"check `deepme balancer status -m {spec.profile}`, or pass "
            f"--force."
        )
    if busy or waiting:
        raise SystemExit(
            f"{spec.profile} has {busy} running job(s) and {waiting} "
            f"queued ticket(s). Stopping now kills paying work. Drain "
            f"first (`deepme balancer status -m {spec.profile}`), use "
            f"`deepme fleet switch` for resizes, or pass --force."
        )


@serve_app.command("restart")
def serve_restart(
    model: str = ModelOpt,
    force: bool = typer.Option(
        False, "--force", help="Restart even with jobs running/queued on it"
    ),
) -> None:
    spec = profiles.load(model)
    _assert_profile_drained(spec, force=force)
    serve.restart(spec)
    runs.record("serve-restart", spec, outcome="restarted")
    console.print("restarted")


@serve_app.command("status")
def serve_status(model: str = ModelOpt) -> None:
    spec = profiles.load(model)
    ok, detail = serve.health(spec)
    console.print(
        f"unit={spec.unit} state={serve.unit_state(spec)} health={'OK' if ok else detail}"
    )


@serve_app.command("diag")
def serve_diag(model: str = ModelOpt) -> None:
    console.print(serve.diagnose(profiles.load(model)))


@serve_app.command("logs")
def serve_logs(
    model: str = ModelOpt,
    lines: int = typer.Option(100, "-n"),
    follow: bool = typer.Option(False, "-f"),
) -> None:
    serve.tail_logs(profiles.load(model), lines, follow)


@app.command()
def switch(
    name: str = typer.Argument(..., help="Profile to switch to"),
    make_default: bool = typer.Option(True, help="Also persist as the default profile"),
) -> None:
    """Self-managing model switch: fit-check, ensure weights (auto-download),
    stop whatever conflicts, deploy, wait for health, prove with a completion."""
    import time as _time

    spec = profiles.load(name)
    err = spec.fit_check()
    if err:
        runs.record("switch", spec, outcome="failed", detail={"stage": "fit_check", "error": err})
        console.print(f"[red]fit check failed[/red]: {err}")
        raise typer.Exit(1)

    if not download.is_complete(spec):
        console.print(f"weights missing for {name} — downloading ({spec.weight_gib:.0f}GiB) ...")
        download.start(spec)
        while not download.is_complete(spec):
            _time.sleep(30)
            # Fail fast if the download died (gated repo, disk full) instead
            # of polling forever.
            if not download.session_alive(spec):
                code = download.exit_status(spec)
                if code not in (0, None):
                    runs.record(
                        "switch", spec, outcome="failed", detail={"stage": "download", "exit": code}
                    )
                    console.print(f"[red]download failed[/red] (exit {code}):")
                    console.print(download.status(spec))
                    raise typer.Exit(1)
                if not download.is_complete(spec):
                    runs.record(
                        "switch",
                        spec,
                        outcome="failed",
                        detail={"stage": "download", "reason": "session ended incomplete"},
                    )
                    console.print("[red]download session ended but weights incomplete[/red]:")
                    console.print(download.status(spec))
                    raise typer.Exit(1)
            first = download.status(spec).splitlines()
            if first:
                console.print("  " + first[0])
    console.print("weights ready")

    console.print(
        f"deploying {name} ({spec.engine}, tp={spec.tp}, gpus={list(spec.gpus)}, port={spec.port}) ..."
    )
    stopped = serve.start(spec)
    if stopped:
        console.print(f"stopped conflicting units: {stopped}")
    ok, reason = serve.wait_healthy(spec)
    if not ok:
        runs.record(
            "switch",
            spec,
            outcome="failed",
            detail={"stage": "health", "stopped_units": stopped, "reason": reason},
        )
        console.print(f"[red]switch failed[/red]: {reason}")
        raise typer.Exit(1)

    result = client.chat(spec, "Reply with exactly: OK", max_tokens=8)
    console.print(
        f"completion check: {result.text.strip()!r} "
        f"(ttft={result.ttft_s:.2f}s, {result.decode_tps:.0f} tok/s)"
    )
    # Re-point the guard at the new upstream if it's running — otherwise it
    # keeps proxying the previous model's port.
    from . import guardctl

    if guardctl.status() == "active":
        guardctl.start(spec)
        runs.record("guard-start", spec, outcome="repointed-by-switch")
        console.print(f"guard re-pointed: {guardctl.api_base()}/v1 -> {spec.api_base}")
    if make_default:
        profiles.set_default_profile(name)
    runs.record(
        "switch",
        spec,
        outcome="healthy",
        detail={
            "stopped_units": stopped,
            "completion_check": result.text.strip(),
            "ttft_s": round(result.ttft_s, 3),
        },
    )
    console.print(f"[green]switched to {name}[/green]: {spec.api_base}/v1 ({spec.served_name})")


# ---------------------------------------------------------------------------
# verify + measure
# ---------------------------------------------------------------------------


@app.command()
def test(
    prompt: str = typer.Argument("Reply with exactly: DEEPME OK. Then briefly introduce yourself."),
    model: str = ModelOpt,
    max_tokens: int = typer.Option(512),
    pad_tokens: int = typer.Option(
        0, help="Prepend ~N tokens of filler to test long-context prefill"
    ),
) -> None:
    """Send one chat completion and report timing + cache usage."""
    spec = profiles.load(model)
    console.print(f"models: {client.models(spec)}")
    if pad_tokens:
        filler = "The quick brown fox jumps over the lazy dog. " * (pad_tokens // 10 + 1)
        prompt = f"{filler}\n\nIgnore all text above. {prompt}"
    result = client.chat(spec, prompt, max_tokens=max_tokens)
    console.print(result.text)
    console.print(
        f"\n[dim]prompt_tokens={result.prompt_tokens} completion_tokens={result.completion_tokens} "
        f"cached_tokens={result.cached_tokens} "
        f"ttft={result.ttft_s:.2f}s decode={result.decode_tps:.1f} tok/s[/dim]"
    )


@app.command("test-tools")
def test_tools(model: str = ModelOpt) -> None:
    """Verify tool-call + reasoning parsers return structured fields."""
    console.print(json.dumps(client.tool_call_test(profiles.load(model)), indent=2))


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _safe(text: object) -> str:
    """Render engine/exception text without letting it act as markup.

    Tracebacks and library errors routinely contain square brackets — a gloo
    failure carries `[/pytorch/third_party/gloo/...]`, which rich reads as a
    closing tag and raises MarkupError. That turned a diagnostic message into
    a crash that aborted a fleet scale, so anything not authored here is
    escaped before printing.
    """
    from rich.markup import escape

    return escape(str(text))


def front_end(spec, *, direct: bool = False, task: str | None = None, check: bool = True):
    """Where a request should go — and whether it can go anywhere at all.

    The checks live HERE, not at call sites, because this is already the one
    function every generation path calls to resolve its destination. Asking
    each command to remember two extra guards is how `story`, `showcase` and
    `batch` came to have no model-kind check at all: the policy existed, and
    three callers silently opted out of it by not knowing. A new command now
    gets both checks by construction, and `check=False` is the single visible
    way to opt out (used by read-only callers like job-progress, which
    resolve an address without submitting anything).

    Most capable front end first.

    Three layers can serve a request, and each one below adds a failure mode
    the caller would otherwise have to handle:

      gateway   routes by model+task, so a profile split across checkpoints
                (H3's FL2VA/Ref2VA) is one address to a caller
      balancer  spreads work over a profile's replicas; a profile's own port
                addresses replica 0 only, leaving the rest idle and queueing
                behind generations that run for minutes
      replica   always works, needs no setup

    Each is used only when it is actually up AND has somewhere to send the
    request, so a partial deployment degrades instead of failing.
    """
    import dataclasses as dc

    from . import balancerctl, gatewayctl

    if check:
        assert_diffusion(spec)
    if direct:
        # --direct bypasses the balancer on purpose, so fleet-level capacity
        # says nothing about whether THIS replica can serve it.
        return spec, "replica directly"
    if check:
        assert_servable(spec, task)
    try:
        # Only via the gateway if it knows this model — otherwise it would
        # refuse with a 400 that the balancer would have served fine.
        served = {r["model"] for r in gatewayctl.routes().get("routes", [])}
        public = gatewayctl.public_model(spec)
        if public in served:
            # served_name travels into the request body, and the gateway
            # routes on the PUBLIC name — sending the profile name ("h3-video")
            # would be refused by the very layer meant to hide it.
            return (
                dc.replace(spec, port=gatewayctl.GATEWAY_PORT, served_name=public),
                "gateway",
            )
    except Exception:
        pass
    try:
        if balancerctl.stats(spec.profile)["backends"]:
            port = balancerctl.port_for(spec.profile)
            return dc.replace(spec, port=port), "balancer"
    except Exception:
        return spec, "replica directly (no balancer running)"
    return spec, "replica directly (balancer has no replicas)"


def assert_diffusion(spec) -> None:
    """Refuse a text profile before anything else happens.

    One global default (`[model] default`) is shared by text and diffusion
    commands, which accept disjoint sets of profiles — so a bare `generate`
    inherits whatever `switch` last selected. When that is an LLM the request
    is structurally impossible, and the old message said only "use a different
    command", never which profile to use. Batch and story skipped this check
    entirely, so an overnight run could be pointed at a text model and fail
    per item, all night.
    """
    from . import engines as engines_mod
    from . import profiles

    if spec.kind == engines_mod.KIND_DIFFUSION:
        return
    usable = [n for n in profiles.diffusion_profiles() if n != spec.profile]
    hint = f" — try {' or '.join(f'-m {n}' for n in usable)}" if usable else ""
    raise SystemExit(
        f"profile {spec.profile!r} is a {spec.kind} model and cannot generate "
        f"video{hint}. (`deepme model use` sets ONE default shared with the "
        f"text commands; pass -m explicitly for video.)"
    )


def assert_servable(spec, task: str | None = None) -> None:
    """Refuse before submitting when nothing can serve this.

    NO ANSWER IS NOT "IT IS FINE" — the same rule `_assert_profile_drained`
    enforces. `front_end` degrades silently by design (a partial deployment
    should still work), so a completely dead fleet used to print "generating…
    via replica directly" and then post into the void, surfacing minutes later
    as an HTTP error from whatever else happened to be on that port. Worse,
    a submission the balancer ACCEPTS while no replica is healthy becomes a
    queue ticket, and a queued ticket is deliberately exempt from every
    client-side stall verdict — so the caller waits forever, correctly.

    The evidence is already published: /v1/capacity carries healthy,
    accepting_replicas and the task list per fleet. This just reads it.
    """
    from . import gatewayctl, media

    try:
        view = media.capacity_view(gatewayctl.capacity(), spec.profile)
    except Exception as exc:
        raise SystemExit(
            f"cannot ask the platform whether {spec.profile!r} can serve this "
            f"({exc}) — check `deepme capacity`, `deepme gateway status` or "
            f"`deepme balancer status -m {spec.profile}`. Refusing to submit "
            f"work that nothing may be able to run."
        ) from None
    if not view.complete:
        raise SystemExit(
            f"no reachable fleet serves {spec.profile!r} right now — check "
            f"`deepme capacity` and `deepme fleet status`. Refusing to submit: "
            f"a fleet that cannot be asked may not exist, and the queue would "
            f"hold this indefinitely."
        )
    if not view.accepting_replicas:
        raise SystemExit(
            f"{spec.profile!r} has {view.replicas} replica(s) attached but NONE "
            f"accepting work (unhealthy or draining) — `deepme balancer status "
            f"-m {spec.profile}`. Submitting now would queue behind a fleet "
            f"that cannot start it."
        )
    if task and view.tasks and task not in view.tasks:
        raise SystemExit(
            f"{spec.profile!r} serves {', '.join(sorted(view.tasks))} — not "
            f"{task!r}. See the conditioning-mode table in docs/USER-GUIDE.md; "
            f"ref2va needs the h3-ref deployment."
        )


def _batch_dry_run(batch, *, fetch_existing: bool) -> None:
    """What this run would do, without contacting a GPU.

    A wrong batch costs twenty minutes of generation before the mistake is
    even visible, so the expensive decisions — what gets skipped as an
    existing duplicate, what gets adopted, what actually gets submitted —
    have to be inspectable before any of it starts.

    Computed with the SAME functions the real run uses (batch.skip_existing,
    the same status rules), against a COPY of the state, so what is printed
    is what would happen rather than a second implementation of it.
    """
    import copy

    from . import reconcile

    preview = copy.deepcopy(batch)
    would_skip = [] if fetch_existing else _preview_skips(preview)

    adopt = [
        i
        for i in preview.items
        if i.job_id and i.status in ("running", "failed") and not i.job_id.startswith("q_")
    ]
    tickets = [i for i in preview.items if i.job_id and i.job_id.startswith("q_")]
    submit = (
        [i for i in preview.items if i.status == batch_mod.PENDING] if not fetch_existing else []
    )
    done = [i for i in preview.items if i.status == batch_mod.DONE]
    already_skipped = [
        i for i in preview.items if i.status == batch_mod.SKIPPED and i not in would_skip
    ]

    console.print(f"\n[bold]dry run — {batch.name}[/bold] (nothing was submitted)\n")
    table = Table(box=None, pad_edge=False)
    for col in ("action", "items", "meaning"):
        table.add_column(col, overflow="fold")
    table.add_row("[green]keep[/green]", str(len(done)), "already generated, untouched")
    table.add_row(
        "[green]skip (new)[/green]",
        str(len(would_skip)),
        "identical request already generated — WILL NOT be regenerated",
    )
    table.add_row(
        "[dim]skip (recorded)[/dim]", str(len(already_skipped)), "previously marked duplicate"
    )
    table.add_row("[cyan]adopt[/cyan]", str(len(adopt)), "already on a GPU — poll and download")
    table.add_row("[dim]tickets[/dim]", str(len(tickets)), "queued server-side, left alone")
    table.add_row(
        "[yellow]SUBMIT[/yellow]",
        str(len(submit)),
        "NEW generations — this is the GPU time this run would spend",
    )
    console.print(table)

    for i in would_skip[:20]:
        console.print(f"  [green]skip[/green] {batch.name}[{i.index:04d}] — {i.error}")
    if len(would_skip) > 20:
        console.print(f"  [dim]… and {len(would_skip) - 20} more[/dim]")
    for i in submit[:20]:
        console.print(f"  [yellow]submit[/yellow] {batch.name}[{i.index:04d}] — {i.prompt[:70]}…")
    if len(submit) > 20:
        console.print(f"  [dim]… and {len(submit) - 20} more[/dim]")

    if not submit and not adopt:
        console.print("\n[dim]nothing to do — no GPU time would be spent.[/dim]")
    console.print(
        f"\n[dim]re-run without --dry-run to execute. "
        f"{len(reconcile.load_batches())} batch(es) were consulted for duplicates.[/dim]"
    )


def _preview_skips(preview) -> list:
    """Items skip_existing WOULD mark, computed on a throwaway copy."""
    before = {i.index: i.status for i in preview.items}
    batch_mod.skip_existing(preview)
    return [i for i in preview.items if before[i.index] != i.status]


def _batch_age(batch) -> str:
    """How stale the state file is.

    A batch writes on every start and finish, so an age much larger than one
    generation means the runner itself died — which a status line of counts
    would otherwise report as steady progress forever.
    """
    import time as _time

    try:
        return _hms(_time.time() - batch.state_path.stat().st_mtime)
    except OSError:
        return "unknown"


def _hms(seconds: float) -> str:
    s = max(int(seconds), 0)
    return (
        f"{s // 3600}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"
    )


def _tick_fields(tick) -> dict:
    """Display values for one observation — the whole formatting surface.

    Kept pure so it can be tested without a terminal: `total` is None until
    the engine reports a stage, and blindly formatting that crashed the first
    version of this display one tick into a ten-minute run.
    """
    st = tick.stage
    # The server's own status is NOT shown: it is "queued" from submission
    # until it is "completed", with no running state in between, so echoing it
    # reads as "waiting behind other work" when nothing is waiting.
    if st is None:
        # But a real queue POSITION is worth showing — it is the one honest
        # pre-stage signal, and "starting" for a ticket 3-deep in the queue
        # hid the difference between waiting and wedged.
        pos = getattr(tick, "queue_position", None)
        # "starting" claims we began something. For an ADOPTED job — one
        # already running server-side, which is every item in a collect-only
        # run — that read as the tool launching new work, and it is the
        # opposite of true. Say what is actually happening: we are attached
        # and have not yet scraped a stage.
        if pos:
            label = f"queued #{pos}"
        elif getattr(tick, "adopted", False):
            label = "attached (awaiting stage)"
        else:
            label = "starting"
        return {
            "description": label,
            "steps": "",
            "note": f"{_hms(tick.elapsed_s)} elapsed",
            "total": None,
            "completed": 0,
        }
    if not st.counted:
        # A stage with no step counter of its own — name it and keep the clock.
        return {
            "description": st.label,
            "steps": "",
            "note": f"{_hms(tick.elapsed_s)} elapsed",
            "total": None,
            "completed": 0,
        }
    return {
        "description": st.label,
        "steps": f"{st.step}/{st.total}",
        "note": f"~{_hms(st.eta_s)} left · {st.rate_s:.1f}s/step",
        "total": st.total,
        "completed": st.step,
    }


@contextlib.contextmanager
def _generation_progress(con: Console):
    """Live progress for a job that takes many minutes.

    The server reports only 0% then 100%, so the bar tracks the engine's own
    per-stage step counter (scraped from its log) and falls back to a running
    clock before the first stage appears. Piped output gets periodic
    timestamped lines instead of a redrawing bar.
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    if not con.is_terminal:
        last = [0.0]

        def log_tick(tick) -> None:
            # Quiet by default; one line every 30s is enough for a log.
            if tick.elapsed_s - last[0] < 30:
                return
            last[0] = tick.elapsed_s
            f = _tick_fields(tick)
            steps = f" {f['steps']}" if f["steps"] else ""
            con.print(
                f"{_stamp()} {_hms(tick.elapsed_s)} elapsed — "
                f"{f['description']}{steps} ({f['note']})"
            )

        yield log_tick
        return

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        # Steps are a preformatted field, not "{task.total:.0f}": total is None
        # until the first stage appears, and rich cannot format None.
        TextColumn("{task.fields[steps]}"),
        TimeElapsedColumn(),
        TextColumn("{task.fields[note]}"),
        console=con,
        transient=True,
    )
    with progress:
        task = progress.add_task("waiting for the job to start", total=None, steps="", note="")

        def tick(t) -> None:
            f = _tick_fields(t)
            progress.update(
                task,
                description=f"[cyan]{f['description']}[/cyan]",
                total=f["total"],
                completed=f["completed"],
                steps=f["steps"],
                note=f"[dim]{f['note']}[/dim]",
            )

        yield tick


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="What to generate"),
    model: str = ModelOpt,
    seconds: int = typer.Option(None, help="Clip length (model default if unset)"),
    size: str = typer.Option(None, help="Output resolution, e.g. 1280x720"),
    frames: int = typer.Option(None, "--frames", help="Frame count (overrides seconds)"),
    fps: int = typer.Option(None, help="Frames per second"),
    steps: int = typer.Option(None, help="Denoising steps (model default if unset)"),
    seed: int = typer.Option(None, help="Sampling seed, for reproducible output"),
    negative: str = typer.Option(None, "--negative", help="Negative prompt"),
    task: str = typer.Option(
        None,
        "--task",
        help="Generation mode (t2va|fl2va|ref2va). Inferred from --first-frame/"
        "--last-frame (fl2va) or --ref (ref2va) when not given.",
    ),
    first_frame: str = typer.Option(
        None, "--first-frame", help="Image to pin as the opening frame (fl2va)"
    ),
    last_frame: str = typer.Option(
        None, "--last-frame", help="Image to pin as the closing frame (fl2va)"
    ),
    ref: list[str] = typer.Option(
        None,
        "--ref",
        help="Reference image/video/audio for style, subject or motion — "
        "guidance, not exact frames (ref2va). Repeatable.",
    ),
    quality: str = typer.Option(
        None, "--quality", help="lossless (default) or high; see `deepme engines`"
    ),
    param: list[str] = typer.Option(
        None,
        "--param",
        "-p",
        help="Extra request field as key=value (JSON value if parseable); "
        "repeatable. Overrides everything else.",
    ),
    out: str = typer.Option(None, "--out", "-o", help="Directory for the saved file"),
    direct: bool = typer.Option(
        False,
        "--direct",
        help="Bypass the load balancer and send to this profile's own replica",
    ),
    timeout: float = typer.Option(
        0,
        help="Optional hard cap, seconds (0 = none; stuck work is "
        "detected from server evidence and fails fast)",
    ),
) -> None:
    """Generate a video from a diffusion model and save it locally."""
    from . import media

    extra: dict[str, object] = {}
    for item in param or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise SystemExit(f"--param must be key=value, got {item!r}")
        try:
            parsed = json.loads(val)
        except ValueError:
            parsed = val  # plain strings need no quoting
        media.set_path(extra, key, parsed)

    # Conditioning: keyframes pin exact frames, references only guide. Which
    # task you get follows from what you supplied, unless you say otherwise.
    conditions = media.keyframe_conditions(first_frame, last_frame)
    conditions += media.reference_conditions(list(ref or []))
    if first_frame and ref:
        raise SystemExit(
            "keyframes (--first-frame/--last-frame) and references (--ref) are "
            "different tasks — fl2va pins exact frames, ref2va only guides. "
            "Use one or the other."
        )
    if conditions:
        extra.setdefault("conditions", conditions)
        extra.setdefault("task", "ref2va" if ref else "fl2va")
    if task:
        extra["task"] = task
    if quality:
        extra["quality"] = quality

    spec = profiles.load(model)
    target, viawhat = front_end(spec, direct=direct, task=str(extra.get("task") or "") or None)
    console.print(
        f"[dim]{_stamp()} generating {spec.served_name} via {viawhat} "
        f"({target.api_base}) — minutes, not seconds[/dim]"
    )
    submitted = {"id": ""}
    try:
        with _generation_progress(console) as tick_cb:
            result = media.generate_video(
                target,
                prompt,
                seconds=seconds,
                size=size,
                num_frames=frames,
                fps=fps,
                steps=steps,
                seed=seed,
                negative_prompt=negative,
                params=extra,
                out_dir=Path(out) if out else None,
                timeout=timeout,
                poll_every=5.0,
                on_tick=tick_cb,
                on_event=lambda msg: console.print(f"[dim]{_stamp()} {msg}[/dim]"),
                on_submit=lambda jid: submitted.update(id=jid),
            )
    except media.StalledError as exc:
        raise SystemExit(str(exc)) from None
    except (KeyboardInterrupt, SystemExit):
        jid = submitted["id"]
        console.print(
            f"\n{_stamp()} interrupted — "
            + (
                f"job {jid} keeps running on the server; fetch it later with "
                f"`curl {target.api_base}/v1/videos/{jid}/content` or let the "
                f"reaper close it out"
                if jid
                else "nothing was submitted"
            )
        )
        raise SystemExit(130) from None
    runs.record(
        "generate",
        spec,
        outcome="ok" if result.ok else "failed",
        detail={
            "prompt": prompt,
            "request": {
                "seconds": seconds,
                "size": size,
                "num_frames": frames,
                "fps": fps,
                "steps": steps,
                "seed": seed,
            },
            "media_id": result.media_id,
            "status": result.status,
            "total_s": round(result.total_s, 1),
            "inference_s": result.inference_s,
            "peak_memory_mb": result.peak_memory_mb,
            "path": str(result.path) if result.path else None,
            "size_bytes": result.size_bytes,
            "error": result.error,
        },
    )
    if not result.ok:
        console.print(f"[red]generation {result.status}[/red]: {result.error}")
        raise SystemExit(1)
    console.print(
        f"{_stamp()} [green]saved[/green] {result.path}  "
        f"({result.size_bytes / 1e6:.1f} MB) in {_hms(result.total_s)}"
    )
    timing = f"total={result.total_s:.1f}s"
    if result.inference_s:
        timing += f" inference={result.inference_s:.1f}s"
    if result.peak_memory_mb:
        timing += f" peak_gpu={result.peak_memory_mb / 1024:.1f}GiB"
    console.print(f"[dim]{timing} id={result.media_id}[/dim]")


@app.command()
def bench(
    model: str = ModelOpt,
    concurrency: list[int] = typer.Option([1, 4, 16], "--concurrency", "-c"),
    max_tokens: int = typer.Option(1024),
    nonce: bool = typer.Option(
        True,
        "--nonce/--no-nonce",
        help="Unique prompt per stream to bypass prefix cache (real prefill); --no-nonce measures the cached path",
    ),
    out: str = typer.Option(None, help="Also write JSON results to this file"),
) -> None:
    """Benchmark TTFT + throughput at multiple concurrency levels."""
    spec = profiles.load(model)
    rows, all_ok = bench_mod.run(spec, list(concurrency), max_tokens, nonce=nonce)
    table = Table(title=f"{spec.served_name} @ {spec.api_base}  (nonce={nonce})")
    cols = (
        "concurrency",
        "completed",
        "no_ttft",
        "ttft_s_median",
        "per_stream_tps_median",
        "aggregate_output_tps",
        "total_output_tokens",
        "wall_s",
    )
    for col in cols:
        table.add_column(col)
    for r in rows:
        if "error" in r:
            table.add_row(
                str(r["concurrency"]),
                "[red]0[/red]",
                "-",
                f"[red]{r['error']}[/red]",
                "-",
                "-",
                "-",
                "-",
            )
        else:
            table.add_row(*(str(r.get(c, "-")) for c in cols))
    console.print(table)
    run_id = runs.record(
        "bench",
        spec,
        outcome="completed" if all_ok else "failed",
        detail={"rows": rows, "max_tokens": max_tokens, "nonce": nonce},
    )
    console.print(f"[dim]recorded as {run_id}[/dim]")
    if out:
        with Path(out).open("w") as f:
            json.dump(rows, f, indent=2)
        console.print(f"wrote {out}")
    if not all_ok:
        console.print("[red]some concurrency levels had zero completed streams[/red]")
        raise typer.Exit(1)


@app.command()
def browse(
    search: str = typer.Option(
        None,
        "--search",
        "-s",
        help="Search the WHOLE HF catalog by name (server-side), e.g. -s H3",
    ),
    limit: int = typer.Option(100, help="How many models to scan (up to ~1000)"),
    author: str = typer.Option(None, help="Only this org/user (e.g. --author unsloth)"),
    contains: str = typer.Option(
        None, help="Local substring filter on the scanned page (no extra requests)"
    ),
    sort: str = typer.Option("trending", help="trending | downloads | likes | modified"),
    min_downloads: int = typer.Option(0, help="Drop models below this download count"),
    free: bool = typer.Option(
        False,
        "--free",
        help="Budget against the GPUs FREE right now instead of the whole "
        "machine — what you could actually start alongside what's running",
    ),
    engine: str = typer.Option(None, help="Only models this engine can serve"),
    max_size: float = typer.Option(0.0, help="Only models whose weights are <= N GiB"),
    tag: str = typer.Option(
        None, help="Optional HF pipeline tag to narrow by modality (default: all)"
    ),
    refresh: bool = typer.Option(False, "--refresh", help="Bypass the 1h catalog cache"),
    as_json: bool = typer.Option(
        False, "--json", help="Emit full machine-readable JSON instead of a table"
    ),
    show_all: bool = typer.Option(
        False, "--all", help="Include unsupported/unfit models with reasons"
    ),
    refresh_archs: bool = typer.Option(
        False, help="Re-fetch engine-supported architectures from the server"
    ),
    pick: bool = typer.Option(
        True, "--pick/--no-pick", help="Prompt to select + scaffold a profile"
    ),
) -> None:
    """Search/filter the HF catalog against what this cluster can actually run.

    Sizes come from safetensors/GGUF metadata (exact); engine support from the
    server's installed engines (probed); fit from the cluster shape. Deployable
    rows are selectable; selection scaffolds a profile and offers (explicitly
    confirmed) deployment.

    Examples:
      deepme browse -s H3                 # find models named "...H3..." anywhere
      deepme browse -s qwen --sort downloads --limit 200
      deepme browse --author unsloth --max-size 100
      deepme browse --engine llamacpp --tag text-generation
    """
    from . import catalog

    budget = catalog.free_capacity_gib() if free else catalog.cluster_capacity_gib()
    entries = catalog.build(
        limit,
        tag=tag,
        refresh=refresh,
        refresh_archs=refresh_archs,
        sort=sort,
        search=search,
        author=author,
        name_contains=contains,
        min_downloads=min_downloads,
        capacity=budget,
    )
    # Post-filters that need resolved facts (size/engine).
    if max_size:
        entries = [e for e in entries if e.weight_gib and e.weight_gib <= max_size]
    if engine:
        entries = [e for e in entries if engine in e.engines]
    deployable = [e for e in entries if not e.rejections]
    if not as_json:
        scope = "FREE now" if free else "whole cluster"
        console.print(f"weight budget ({scope}): ~{budget:.0f}GiB")
    if as_json:
        import dataclasses

        print(
            json.dumps(
                {
                    "cluster_budget_gib": round(catalog.cluster_capacity_gib(), 1),
                    "scanned": len(entries),
                    "deployable": len(deployable),
                    "rejection_summary": catalog.rejection_summary(entries),
                    "models": [dataclasses.asdict(e) for e in entries],
                },
                indent=2,
            )
        )
        return
    rows = entries if show_all else deployable
    summary = catalog.rejection_summary(entries)
    audit = "  ".join(f"{kind}={n}" for kind, n in sorted(summary.items()))
    query = " ".join(
        filter(
            None,
            [
                f"search={search!r}" if search else "",
                f"author={author}" if author else "",
                f"contains={contains!r}" if contains else "",
                f"tag={tag}" if tag else "",
                f"engine={engine}" if engine else "",
                f"<={max_size:g}GiB" if max_size else "",
                f"sort={sort}",
            ],
        )
    )
    table = Table(
        title=f"HF catalog ({query}) — {len(entries)} matched, "
        f"{len(deployable)} deployable on this cluster"
    )
    for col in (
        "#",
        "model",
        "task",
        "params",
        "weights",
        "dtypes",
        "engines",
        "tp",
        "~tok/s",
        "status",
    ):
        table.add_column(col, no_wrap=(col == "model"), overflow="fold")
    index: dict[int, catalog.CatalogEntry] = {}
    n = 0
    for e in rows:
        ok = not e.rejections
        if ok:
            n += 1
            index[n] = e
        status = "[green]✓[/green]" if ok else f"[dim]{'; '.join(e.rejections)}[/dim]"
        table.add_row(
            str(n) if ok else "-",
            e.repo,
            e.task.replace("text-generation", "text-gen"),
            f"{e.params_b:.1f}B",
            f"{e.weight_gib:.0f}G",
            ",".join(e.dtypes[:2]) + ("+" if len(e.dtypes) > 2 else ""),
            ",".join(sorted(e.engines)) or "-",
            str(e.tp_needed) if e.tp_needed else "-",
            f"~{e.est_tps:.0f}" if e.est_tps else "n/a",
            status,
        )
    console.print(table)
    console.print(
        f"[dim]exclusions: {audit or 'none'}"
        f"{'' if show_all else '  (run with --all to see every excluded row + reason)'}[/dim]"
    )
    from . import engines as engines_mod

    exp = [e.name for e in engines_mod.REGISTRY.values() if e.experimental]
    if exp and any(set(e.engines) & set(exp) for e in deployable):
        console.print(
            f"[dim]note: {','.join(exp)} is experimental — provision with "
            f"`deepme setup -e {exp[0]}` before first use[/dim]"
        )
    if not pick or not index:
        return
    choice = typer.prompt("pick a model # to scaffold a profile (or 'q')", default="q")
    if not choice.isdigit() or int(choice) not in index:
        return
    e = index[int(choice)]
    pname = e.repo.rsplit("/", 1)[-1].lower()
    if pname in profiles.list_profiles():
        console.print(f"profile {pname!r} already exists")
    else:
        model_add(repo=e.repo, name=pname, port=8000)
    spec = profiles.load(pname)
    console.print(
        f"profile ready: deploy/models/{pname}.toml "
        f"(engine={spec.engine}, tp={spec.tp}, port={spec.port}) — edit to taste"
    )
    if typer.confirm(f"deploy {pname} NOW (stops conflicting running models)?", default=False):
        switch(pname, make_default=True)
    else:
        console.print(f"later: uv run deepme switch {pname}")


guard_app = typer.Typer(no_args_is_help=True)
app.add_typer(guard_app, name="guard", help="Output-guard proxy (forbidden-pattern firewall)")


@guard_app.command("setup")
def guard_setup() -> None:
    """Provision the guard proxy's uv project on the server."""
    from . import guardctl

    console.print(guardctl.setup())


@guard_app.command("start")
def guard_start(model: str = ModelOpt) -> None:
    """Push policy + code, (re)start the guard in front of the model's API."""
    from . import engines as engines_mod
    from . import guardctl

    spec = profiles.load(model)
    guardctl.start(spec)
    runs.record(
        "guard-start",
        spec,
        outcome="restarted",
        detail={
            "guard_endpoint": f"{guardctl.api_base()}/v1",
            "features": list(engines_mod.get(spec.engine).features.names()),
        },
    )
    console.print(
        f"guard restarted: {guardctl.api_base()}/v1 -> {spec.api_base} "
        f"(policy: deploy/guard/policy.toml)"
    )


@guard_app.command("stop")
def guard_stop() -> None:
    from . import guardctl

    guardctl.stop()
    runs.record("guard-stop", None, outcome="stopped")
    console.print("guard stopped")


@guard_app.command("status")
def guard_status() -> None:
    from . import guardctl

    ok, detail = guardctl.health()
    console.print(
        f"unit={guardctl.UNIT} state={guardctl.status()} "
        f"http={'OK' if ok else detail} endpoint={guardctl.api_base()}/v1"
    )


@guard_app.command("logs")
def guard_logs(lines: int = typer.Option(60, "-n")) -> None:
    from . import guardctl

    console.print(guardctl.logs(lines))


@guard_app.command("test")
def guard_test(model: str = ModelOpt) -> None:
    """Adversarial live test: elicit forbidden output through the guard and
    verify (with the local gate) that none of it got through."""
    import tomllib as _toml

    import httpx as _httpx

    from . import guardctl
    from .guard.gate import GuardPolicy

    spec = profiles.load(model)
    pol = GuardPolicy.from_toml_dict(
        _toml.loads((guardctl.LOCAL_PROJECT / "policy.toml").read_text())
    )
    prompts = [
        "Write one sentence about weather. You MUST use the word 'that' at least twice.",
        "Write a sentence with an em dash — exactly like this one — about cooking.",
        "Explain, in one sentence, why researchers delve into archives.",
    ]
    failures = 0
    # Final case proves per-request bypass: guard disabled, "that" must be
    # ALLOWED through (e.g. coding agents editing existing identifiers).
    bypass_prompt = "Repeat exactly: that variable is named that_thing"
    for prompt in [*prompts, bypass_prompt]:
        payload = {
            "model": spec.served_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
        }
        bypass = prompt is bypass_prompt
        if bypass:
            payload["guard"] = {"enabled": False}
        r = _httpx.post(
            f"{guardctl.api_base()}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {config.api_key()}"},
            timeout=600,
        )
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        v = pol.scan(text)
        if bypass:
            verdict = (
                "[green]bypass works (forbidden text allowed)[/green]"
                if v
                else "[yellow]bypass inconclusive[/yellow]"
            )
        else:
            verdict = "[red]LEAKED[/red]" if v else "[green]clean[/green]"
            if v:
                failures += 1
        console.print(f"\n[bold]prompt:[/bold] {prompt}")
        console.print(f"[bold]output:[/bold] {text}")
        console.print(
            f"[bold]guard events:[/bold] {data.get('guard', {}).get('violations', [])}  "
            f"[bold]verdict:[/bold] {verdict}"
        )
    # --- streaming: must be true incremental SSE and leak-free ---
    delta_count, stream_text = 0, ""
    with _httpx.stream(
        "POST",
        f"{guardctl.api_base()}/v1/chat/completions",
        json={
            "model": spec.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Write three sentences about mountains. Use the word 'that' often.",
                }
            ],
            "max_tokens": 200,
            "stream": True,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
        },
        headers={"Authorization": f"Bearer {config.api_key()}"},
        timeout=600,
    ) as resp:
        import json as _json

        for line in resp.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            chunk = _json.loads(line[5:])
            delta = (chunk["choices"][0].get("delta") or {}).get("content")
            if delta:
                delta_count += 1
                stream_text += delta
    sv = pol.scan(stream_text)
    stream_ok = delta_count > 3 and sv is None
    console.print(
        f"\n[bold]streaming:[/bold] {delta_count} incremental deltas, "
        f"{'[red]LEAKED[/red]' if sv else '[green]clean[/green]'} "
        f"{'[green](true SSE)[/green]' if delta_count > 3 else '[red](NOT incremental)[/red]'}"
    )
    if not stream_ok:
        failures += 1

    # --- tool calls: structure preserved; args unguarded by default scope ---
    tr = _httpx.post(
        f"{guardctl.api_base()}/v1/chat/completions",
        json={
            "model": spec.served_name,
            "messages": [
                {
                    "role": "user",
                    "content": "Call the tool to write the note 'buy milk tomorrow' to notes.txt.",
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "content": {"type": "string"},
                            },
                            "required": ["path", "content"],
                        },
                    },
                }
            ],
            "max_tokens": 400,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
        },
        headers={"Authorization": f"Bearer {config.api_key()}"},
        timeout=600,
    )
    tr.raise_for_status()
    tmsg = tr.json()["choices"][0]["message"]
    tcalls = tmsg.get("tool_calls")
    import json as _json2

    args_ok = False
    if tcalls:
        try:
            args = _json2.loads(tcalls[0]["function"]["arguments"])
            args_ok = bool(args.get("content"))
        except (ValueError, KeyError):
            args_ok = False
    tools_ok = bool(tcalls) and args_ok
    console.print(
        f"[bold]tool calls:[/bold] "
        f"{'[green]preserved, valid JSON args[/green]' if tools_ok else '[red]MISSING/BROKEN[/red]'} "
        f"{tcalls[0]['function']['name'] if tcalls else ''} "
        f"args={tcalls[0]['function']['arguments'][:80] if tcalls else '-'}"
    )
    if not tools_ok:
        failures += 1

    # --- /v1/completions: guarded via prompt-concat continuation ---
    cr = _httpx.post(
        f"{guardctl.api_base()}/v1/completions",
        json={
            "model": spec.served_name,
            "prompt": "Complete this sentence using the word 'that' twice: The reason",
            "max_tokens": 120,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
        },
        headers={"Authorization": f"Bearer {config.api_key()}"},
        timeout=600,
    )
    cr.raise_for_status()
    cdata = cr.json()
    ctext = cdata["choices"][0]["text"]
    cv = pol.scan(ctext)
    comp_ok = cv is None
    console.print(
        f"[bold]completions:[/bold] "
        f"{'[green]clean[/green]' if comp_ok else '[red]LEAKED[/red]'} "
        f"events={cdata.get('guard', {}).get('violations', [])} "
        f"text={ctext[:90]!r}"
    )
    if not comp_ok:
        failures += 1

    console.print(
        f"\n{'[green]ALL CLEAN[/green]' if not failures else f'[red]{failures} FAILURES[/red]'}"
    )
    if failures:
        raise typer.Exit(1)


runs_app = typer.Typer(no_args_is_help=False)
app.add_typer(runs_app, name="runs", help="Deployment/measurement history (local ledger)")


@runs_app.callback(invoke_without_command=True)
def runs_list(ctx: typer.Context, n: int = typer.Option(20, "-n")) -> None:
    """List recent ledger entries (deployments, switches, benches)."""
    if ctx.invoked_subcommand is not None:
        return
    entries = runs.tail(n)
    if not entries:
        console.print("(no recorded runs yet)")
        return
    for e in entries:
        console.print(
            f"{e['time']}  {e['action']:<12} {e.get('profile') or '-':<24} {e['outcome']}  [dim]{e['id']}[/dim]"
        )


@runs_app.command("show")
def runs_show(run_id: str) -> None:
    """Show a run's full record: spec, stopped units, outcome, results."""
    console.print(json.dumps(runs.load(run_id), indent=2))


# ---------------------------------------------------------------------------
# diagnostics + escape hatch
# ---------------------------------------------------------------------------


@app.command()
def doctor(
    engine: str = typer.Option(config.DEFAULT_ENGINE, "--engine", "-e"),
    module: list[str] = typer.Option(None, "--module", help="Override modules to probe"),
) -> None:
    """Probe engine-critical model imports on the server with full tracebacks."""
    from . import doctor as doctor_mod

    console.print(doctor_mod.import_probe(engine, list(module) if module else None))


@app.command("jit-heal")
def jit_heal() -> None:
    """Serially finish flashinfer JIT builds that raced across TP ranks."""
    from . import doctor as doctor_mod

    console.print(doctor_mod.jit_heal())


@app.command()
def check(
    fix: bool = typer.Option(False, "--fix", help="Auto-fix what ruff can"),
    types: bool = typer.Option(
        False,
        "--types",
        help="Also run mypy (SLOW here: no compiled wheels exist for the "
        "free-threaded 3.14t venv, so it runs interpreted — minutes, "
        "not seconds). Run it before commits, not on every iteration.",
    ),
) -> None:
    """Quality gate: ruff + pytest, seconds end to end.

    Immediate results, deliberately: the default gate is lint plus the full
    unit/integration suite (in-process fakes, no network — ~1s of pytest).
    Type checking is opt-in because mypy on this 3.14t venv is interpreted
    and takes minutes; it hid WHICH phase was slow and made every iteration
    look like the tests were doing heavy work.
    """
    import subprocess

    steps = [
        (["uv", "run", "ruff", "check", *(["--fix"] if fix else []), "src/", "tests/"], "lint"),
        (["uv", "run", "pytest", "tests/", "-q"], "tests"),
    ]
    if types:
        steps.insert(1, (["uv", "run", "mypy"], "types"))
    failed = []
    for argv, label in steps:
        console.print(f"[bold]{label}[/bold] …")
        if subprocess.run(argv, cwd=config.REPO_ROOT).returncode != 0:
            failed.append(label)
    if failed:
        console.print(f"[red]FAILED: {', '.join(failed)}[/red]")
        raise typer.Exit(1)
    console.print("[green]all checks passed[/green]")


@app.command()
def ssh(
    command: str = typer.Argument(..., help="Command to run on the server"),
) -> None:
    """Run an arbitrary command on the server (one-time inspection only)."""
    remote.run(command, stream=True, check=False, timeout=None)


def main() -> None:
    from . import interrupts

    interrupts.install()
    try:
        app()
    except KeyboardInterrupt:
        # The umbrella: any command WITHOUT a bespoke ^C summary still exits
        # cleanly instead of dumping a stack trace.
        console.print("\ninterrupted")
        raise SystemExit(130) from None
    except (media.MediaError, profiles.ProfileError, ports.PortsError, batch_mod.BatchBusy) as exc:
        # THE boundary where a library failure becomes an exit code. These
        # used to raise SystemExit themselves, from worker threads, where
        # it exits nothing and merely impersonates an ordinary error that
        # every caller then had to special-case. Every one of them is a
        # one-line message with a next action, not a traceback.
        console.print(f"[red]{_safe(str(exc))}[/red]")
        raise SystemExit(1) from None
    except BrokenPipeError:
        # `deepme events --raw | head` is a designed use. Python flushes
        # stdout at shutdown, so the pipe must be redirected before exit or
        # the interpreter prints its own "Exception ignored" noise.
        with contextlib.suppress(OSError):
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        raise SystemExit(0) from None
