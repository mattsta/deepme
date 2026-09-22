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

import json
import os

import typer
from rich.console import Console
from rich.table import Table

from . import bench as bench_mod
from . import client, config, download, hfmeta, profiles, provision, remote, runs, serve
from . import gpus as gpus_mod

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
serve_app = typer.Typer(no_args_is_help=True)
download_app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
app.add_typer(
    serve_app, name="serve", help="Manage inference services (systemd, per model)"
)
app.add_typer(download_app, name="download", help="Manage model weight downloads")
app.add_typer(config_app, name="config", help="Configure the target server")
app.add_typer(model_app, name="model", help="Manage model profiles")
console = Console()

ModelOpt = typer.Option(
    None, "--model", "-m", help="Model profile (default: `deepme model use`)"
)


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
    console.print(
        f"fit_check = {'[red]' + err + '[/red]' if err else '[green]ok[/green]'}"
    )


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
    console.print(
        f"{spec.repo}: {meta.weight_gib:.1f}GiB weights in {meta.weight_shards} shards"
    )


@model_app.command("add")
def model_add(
    repo: str = typer.Argument(..., help="Hugging Face repo, e.g. org/Model-Name"),
    name: str = typer.Option(None, help="Profile name (default: derived from repo)"),
    port: int = typer.Option(8000),
) -> None:
    """Create a profile skeleton for an HF repo (facts auto-fetched)."""
    pname = name or repo.rsplit("/", 1)[-1].lower()
    path = profiles.PROFILES_DIR / f"{pname}.toml"
    if path.exists():
        raise SystemExit(f"profile {pname!r} already exists: {path}")
    meta = hfmeta.get(repo)
    tp = 8
    path.write_text(f"""\
# Model profile: {repo}
# ({meta.weight_gib:.0f}GiB weights in {meta.weight_shards} shards, per HF API)
[model]
repo = "{repo}"
served_name = "{pname}"

[serve]
engine = "vllm"
tp = {tp}
gpus = [0, 1, 2, 3, 4, 5, 6, 7]
port = {port}
max_len = 131072
gpu_mem = 0.90
kv_ram_gb = 0
spec_decode = false

[serve.engine_args]
vllm = []
sglang = []

[sampling]
temperature = 1.0
top_p = 1.0
""")
    console.print(
        f"wrote {path} ({meta.weight_gib:.1f}GiB in {meta.weight_shards} shards)"
    )
    spec = profiles.load(pname)
    err = spec.fit_check()
    console.print(
        f"fit_check = {'[red]' + err + '[/red]' if err else '[green]ok[/green]'}"
    )


# ---------------------------------------------------------------------------
# server config / provisioning
# ---------------------------------------------------------------------------


@config_app.command("init")
def config_init(
    host: str = typer.Option(..., help="Server IP or hostname"),
    user: str = typer.Option("ubuntu", help="SSH user"),
    port: int = typer.Option(8000, help="Default API port"),
) -> None:
    """Write .deepme/config.toml (gitignored) with the target server."""
    config.STATE_DIR.mkdir(exist_ok=True)
    config.CONFIG_FILE.write_text(
        f'[server]\nhost = "{host}"\nuser = "{user}"\nport = {port}\n'
    )
    console.print(f"wrote {config.CONFIG_FILE}")


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
        None, "--engine", "-e", help="Engines (default: all)"
    ),
) -> None:
    """Provision server: build tools + engine uv projects (idempotent)."""
    engines = tuple(engine) if engine else config.ENGINES
    console.print(f"Provisioning {engines} on {config.SSH_TARGET} (several minutes)...")
    console.print(provision.provision(engines)[-4000:])


@app.command("gpu-fix")
def gpu_fix() -> None:
    """Align NVIDIA driver userland + open kernel module; reload if needed."""
    console.print(provision.gpu_stack())


@app.command("engines")
def engines_matrix() -> None:
    """Engine capability matrix: formats + declared feature flags."""
    from . import engines as engines_mod

    table = Table(title="engine capability matrix (src/deepme/engines.py)")
    for col in ("engine", "formats", "status", "features"):
        table.add_column(col, overflow="fold")
    for e in engines_mod.REGISTRY.values():
        table.add_row(
            e.name,
            ",".join(e.formats),
            "experimental" if e.experimental else "production",
            ", ".join(e.features.names()) or "-",
        )
    console.print(table)


@app.command()
def versions() -> None:
    """Show installed engine/toolchain versions on the server."""
    console.print(provision.versions())


@app.command()
def reboot() -> None:
    """Reboot the server and wait until SSH is back (last resort)."""
    import time as _time

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
    for name in profiles.list_profiles():
        spec = profiles.load(name)
        ok, detail = serve.health(spec)
        health_txt = "[green]healthy[/green]" if ok else f"[red]down[/red] ({detail})"
        console.print(
            f"[bold]{name}[/bold] {spec.api_base}/v1 ({spec.served_name}): {health_txt}"
        )


@app.command()
def gpus() -> None:
    """Per-GPU stats and every GPU compute process with its command."""
    console.print(gpus_mod.summary())


@app.command()
def reclaim() -> None:
    """Kill GPU processes that aren't deepme services; clear failed state."""
    console.print(gpus_mod.reclaim())


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


@download_app.command("start")
def download_start(model: str = ModelOpt) -> None:
    """Start (or resume) the weight download for a profile."""
    console.print(download.start(profiles.load(model)))


@download_app.command("restart")
def download_restart(model: str = ModelOpt) -> None:
    console.print(download.restart(profiles.load(model)))


@download_app.command("status")
def download_status(model: str = ModelOpt) -> None:
    console.print(download.status(profiles.load(model)))


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


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
    extra: list[str] = typer.Option(
        None, "--extra", help="Extra raw engine args (repeatable)"
    ),
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
def serve_stop(model: str = ModelOpt) -> None:
    spec = profiles.load(model)
    serve.stop(spec)
    runs.record("serve-stop", spec, outcome="stopped")
    console.print("stopped")


@serve_app.command("restart")
def serve_restart(model: str = ModelOpt) -> None:
    serve.restart(profiles.load(model))
    console.print("restarted")


@serve_app.command("status")
def serve_status(model: str = ModelOpt) -> None:
    spec = profiles.load(model)
    ok, detail = serve.health(spec)
    console.print(
        f"unit={spec.unit} state={serve.unit_state(spec)} health={'OK' if ok else detail}"
    )


@serve_app.command("wait")
def serve_wait(model: str = ModelOpt, timeout: float = typer.Option(1800)) -> None:
    ok, reason = serve.wait_healthy(profiles.load(model), timeout_s=timeout)
    console.print(
        "[green]healthy[/green]" if ok else f"[red]not healthy[/red]: {reason}"
    )
    if not ok:
        raise typer.Exit(1)


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
        console.print(f"[red]fit check failed[/red]: {err}")
        raise typer.Exit(1)

    if not download.is_complete(spec):
        console.print(
            f"weights missing for {name} — downloading ({spec.weight_gib:.0f}GiB) ..."
        )
        download.start(spec)
        while not download.is_complete(spec):
            _time.sleep(30)
            console.print("  " + download.status(spec).splitlines()[0])
    console.print(f"weights ready ({spec.shards} shards)")

    console.print(
        f"deploying {name} ({spec.engine}, tp={spec.tp}, gpus={list(spec.gpus)}, port={spec.port}) ..."
    )
    stopped = serve.start(spec)
    if stopped:
        console.print(f"stopped conflicting units: {stopped}")
    ok, reason = serve.wait_healthy(spec)
    if not ok:
        console.print(f"[red]switch failed[/red]: {reason}")
        raise typer.Exit(1)

    result = client.chat(spec, "Reply with exactly: OK", max_tokens=8)
    console.print(
        f"completion check: {result.text.strip()!r} "
        f"(ttft={result.ttft_s:.2f}s, {result.decode_tps:.0f} tok/s)"
    )
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
    console.print(
        f"[green]switched to {name}[/green]: {spec.api_base}/v1 ({spec.served_name})"
    )


# ---------------------------------------------------------------------------
# verify + measure
# ---------------------------------------------------------------------------


@app.command()
def test(
    prompt: str = typer.Argument(
        "Reply with exactly: DEEPME OK. Then briefly introduce yourself."
    ),
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
        filler = "The quick brown fox jumps over the lazy dog. " * (
            pad_tokens // 10 + 1
        )
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


@app.command()
def bench(
    model: str = ModelOpt,
    concurrency: list[int] = typer.Option([1, 4, 16], "--concurrency", "-c"),
    max_tokens: int = typer.Option(1024),
    out: str = typer.Option(None, help="Also write JSON results to this file"),
) -> None:
    """Benchmark TTFT + throughput at multiple concurrency levels."""
    spec = profiles.load(model)
    rows = bench_mod.run(spec, list(concurrency), max_tokens)
    table = Table(title=f"{spec.served_name} @ {spec.api_base}")
    cols = (
        "concurrency",
        "completed",
        "ttft_s_median",
        "per_stream_tps_median",
        "aggregate_output_tps",
        "total_output_tokens",
        "wall_s",
    )
    for col in cols:
        table.add_column(col)
    for r in rows:
        table.add_row(*(str(r.get(c, "-")) for c in cols))
    console.print(table)
    run_id = runs.record(
        "bench",
        spec,
        outcome="completed",
        detail={"rows": rows, "max_tokens": max_tokens},
    )
    console.print(f"[dim]recorded as {run_id}[/dim]")
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        console.print(f"wrote {out}")


@app.command()
def browse(
    limit: int = typer.Option(100, help="How many top trending models to scan"),
    tag: str = typer.Option(
        None, help="Optional HF pipeline tag to narrow by modality (default: all)"
    ),
    refresh: bool = typer.Option(
        False, "--refresh", help="Bypass the 1h catalog cache"
    ),
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
    """Interactive picker: HF top models x installed-engine support x cluster fit.

    Sizes come from safetensors metadata (exact, per-dtype); engine support
    from the server's installed vllm/sglang registries; fit from the cluster
    shape. Deployable rows are selectable; selection scaffolds a profile and
    offers (explicitly confirmed) deployment via `switch`.
    """
    from . import catalog

    entries = catalog.build(
        limit, tag=tag, refresh=refresh, refresh_archs=refresh_archs
    )
    deployable = [e for e in entries if not e.rejections]
    if not as_json:
        console.print(
            f"cluster weight budget: ~{catalog.cluster_capacity_gib():.0f}GiB"
        )
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
    table = Table(
        title=f"HF top {limit} trending models{f' [{tag}]' if tag else ''} "
        f"({len(deployable)} deployable on this cluster)"
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
    index: dict[int, object] = {}
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
    if typer.confirm(
        f"deploy {pname} NOW (stops conflicting running models)?", default=False
    ):
        switch(pname, make_default=True)
    else:
        console.print(f"later: uv run deepme switch {pname}")


guard_app = typer.Typer(no_args_is_help=True)
app.add_typer(
    guard_app, name="guard", help="Output-guard proxy (forbidden-pattern firewall)"
)


@guard_app.command("setup")
def guard_setup() -> None:
    """Provision the guard proxy's uv project on the server."""
    from . import guardctl

    console.print(guardctl.setup())


@guard_app.command("start")
def guard_start(model: str = ModelOpt) -> None:
    """Push policy + code, (re)start the guard in front of the model's API."""
    from . import guardctl

    spec = profiles.load(model)
    guardctl.start(spec)
    console.print(
        f"guard restarted: {guardctl.api_base()}/v1 -> {spec.api_base} "
        f"(policy: deploy/guard/policy.toml)"
    )


@guard_app.command("stop")
def guard_stop() -> None:
    from . import guardctl

    guardctl.stop()
    console.print("guard stopped")


@guard_app.command("status")
def guard_status() -> None:
    from . import guardctl

    console.print(f"unit={guardctl.UNIT} state={guardctl.status()}")


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
    for prompt in prompts + [bypass_prompt]:
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
        except ValueError, KeyError:
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
app.add_typer(
    runs_app, name="runs", help="Deployment/measurement history (local ledger)"
)


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
    module: list[str] = typer.Option(
        None, "--module", "-m", help="Override modules to probe"
    ),
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
def ssh(
    command: str = typer.Argument(..., help="Command to run on the server"),
) -> None:
    """Run an arbitrary command on the server (one-time inspection only)."""
    remote.run(command, stream=True, check=False, timeout=None)


def main() -> None:
    app()
