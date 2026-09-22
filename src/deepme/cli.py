"""deepme — infrastructure CLI for the 8x5090 DeepSeek-V4-Flash server.

Every operation on the remote box goes through here:

    uv run deepme status                 # GPUs, disk, service, API health
    uv run deepme setup                  # install engine venvs (sglang, vllm)
    uv run deepme download start|status  # fetch model weights (tmux, resumable)
    uv run deepme serve start|stop|logs  # manage the inference service
    uv run deepme test "prompt"          # smoke-test the OpenAI API
    uv run deepme bench -c 1 -c 8 -c 32  # throughput/latency benchmark
    uv run deepme ssh "command"          # escape hatch
"""

from __future__ import annotations

import json
import os

import typer
from rich.console import Console
from rich.table import Table

from . import bench as bench_mod
from . import client, config, download, provision, remote, serve
from . import gpus as gpus_mod

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
serve_app = typer.Typer(no_args_is_help=True)
download_app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
app.add_typer(serve_app, name="serve", help="Manage the inference service (systemd)")
app.add_typer(download_app, name="download", help="Manage model weight download")
app.add_typer(config_app, name="config", help="Configure the target server")
console = Console()


@config_app.command("init")
def config_init(
    host: str = typer.Option(..., help="Server IP or hostname"),
    user: str = typer.Option("ubuntu", help="SSH user"),
    port: int = typer.Option(8000, help="API port the service listens on"),
) -> None:
    """Write .deepme/config.toml (gitignored) with the target server."""
    config.STATE_DIR.mkdir(exist_ok=True)
    config.CONFIG_FILE.write_text(
        f'[server]\nhost = "{host}"\nuser = "{user}"\nport = {port}\n'
    )
    console.print(f"wrote {config.CONFIG_FILE}")


@config_app.command("show")
def config_show() -> None:
    """Show the effective server configuration and where it came from."""
    src = (
        "env"
        if os.environ.get("DEEPME_HOST")
        else (str(config.CONFIG_FILE) if config.CONFIG_FILE.exists() else "unset")
    )
    console.print(
        f"host={config.SSH_HOST or '(unset)'} user={config.SSH_USER} "
        f"port={config.API_PORT} (source: {src})"
    )


@app.command()
def status() -> None:
    """Full picture: GPUs, disk, model files, service state, API health."""
    gpu = remote.run(
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu "
        "--format=csv,noheader",
        check=False,
    ).stdout.strip()
    disk = remote.run("df -h / | tail -1", check=False).stdout.strip()
    console.print("[bold]GPUs[/bold] (idx, util, mem used/total, temp)")
    console.print(gpu or "[red]unreachable[/red]")
    console.print(f"[bold]Disk[/bold] {disk}")
    console.print(f"[bold]Model[/bold] {download.status().strip()}")
    state = serve.unit_state()
    ok, detail = serve.health()
    health_txt = "[green]healthy[/green]" if ok else f"[red]down[/red] ({detail})"
    console.print(
        f"[bold]Service[/bold] {config.SYSTEMD_UNIT}: {state}   API: {health_txt}"
    )
    console.print(
        f"[bold]Endpoint[/bold] {config.API_BASE}/v1  (key in .deepme/api_key)"
    )


@app.command()
def setup(
    engine: list[str] = typer.Option(
        None, "--engine", "-e", help="Engines to install (default: all)"
    ),
) -> None:
    """Provision uv venvs + inference engines on the server."""
    engines = tuple(engine) if engine else config.ENGINES
    console.print(f"Provisioning {engines} on {config.SSH_TARGET} (several minutes)...")
    out = provision.provision(engines)
    console.print(out[-4000:])


@app.command()
def reboot() -> None:
    """Reboot the server and wait until SSH is back."""
    import time as _time

    remote.run("sudo systemctl reboot", check=False, timeout=30)
    console.print("rebooting; waiting for SSH ...")
    _time.sleep(20)
    for _ in range(60):
        r = remote.run("echo up", check=False, timeout=10)
        if r.ok:
            console.print("server is back")
            return
        _time.sleep(10)
    console.print("[red]server did not come back within 10 min[/red]")
    raise typer.Exit(1)


@app.command("gpu-fix")
def gpu_fix() -> None:
    """Align NVIDIA driver userland + kernel module versions; reload if needed."""
    console.print(provision.gpu_stack())


@app.command()
def versions() -> None:
    """Show installed engine/torch versions on the server."""
    console.print(provision.versions())


@download_app.command("start")
def download_start() -> None:
    """Start (or resume) the weight download in a detached tmux session."""
    download.start()


@download_app.command("restart")
def download_restart() -> None:
    """Kill and restart the download session (resumes; picks up new settings)."""
    download.restart()


@download_app.command("status")
def download_status() -> None:
    console.print(download.status())


@serve_app.command("start")
def serve_start(
    engine: str = typer.Option(config.DEFAULT_ENGINE, "--engine", "-e"),
    tp: int = typer.Option(8, help="Tensor parallel size"),
    max_len: int = typer.Option(262_144, help="Max context length"),
    gpu_mem: float = typer.Option(0.90, help="GPU memory fraction"),
    no_spec: bool = typer.Option(
        False, "--no-spec", help="Disable DSpark speculative decoding"
    ),
    kv_ram: int = typer.Option(
        0, "--kv-ram", help="Tier KV cache into N GB of system RAM (0=off)"
    ),
    extra: list[str] = typer.Option(
        None, "--extra", help="Extra raw engine args (repeatable)"
    ),
    wait: bool = typer.Option(True, help="Wait for the API to become healthy"),
) -> None:
    """(Re)deploy the inference service with the given configuration."""
    if not download.is_complete():
        console.print(
            "[red]Model weights are not fully downloaded[/red] — run: uv run deepme download start"
        )
        raise typer.Exit(1)
    opts = config.ServeOptions(
        engine=engine,
        tp=tp,
        max_model_len=max_len,
        gpu_mem_util=gpu_mem,
        spec_decode=not no_spec,
        kv_ram_gb=kv_ram,
        extra_args=list(extra or []),
    )
    console.print(
        f"Launching [bold]{engine}[/bold] tp={tp} max_len={max_len} "
        f"spec_decode={opts.spec_decode} on {config.SSH_TARGET} ..."
    )
    serve.start(opts)
    if wait:
        console.print("Waiting for weights to load (can take ~5-15 min) ...")
        ok, reason = serve.wait_healthy()
        if ok:
            console.print(f"[green]API is up[/green]: {config.API_BASE}/v1")
        else:
            console.print(f"[red]Service did not become healthy[/red]: {reason}")
            raise typer.Exit(1)


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
def gpus() -> None:
    """Per-GPU utilization/memory and every GPU compute process with its command."""
    console.print(gpus_mod.summary())


@app.command()
def reclaim() -> None:
    """Kill all GPU processes that aren't our service; clear systemd failed state."""
    console.print(gpus_mod.reclaim())


@serve_app.command("wait")
def serve_wait(timeout: float = typer.Option(1800, help="Seconds to wait")) -> None:
    """Block until the API is healthy or the service fails (prints diagnosis)."""
    ok, reason = serve.wait_healthy(timeout_s=timeout)
    console.print(
        "[green]healthy[/green]" if ok else f"[red]not healthy[/red]: {reason}"
    )
    if not ok:
        raise typer.Exit(1)


@serve_app.command("restart")
def serve_restart() -> None:
    """Restart the unit as-is (reset-failed + restart), without rewriting config."""
    serve.restart()
    console.print("restarted")


@serve_app.command("diag")
def serve_diag() -> None:
    """Show recent error-ish lines from the service journal."""
    console.print(serve.diagnose())


@serve_app.command("stop")
def serve_stop() -> None:
    serve.stop()
    console.print("stopped")


@serve_app.command("logs")
def serve_logs(
    lines: int = typer.Option(100, "-n"),
    follow: bool = typer.Option(False, "-f"),
) -> None:
    serve.tail_logs(lines, follow)


@serve_app.command("status")
def serve_status() -> None:
    state = serve.unit_state()
    ok, detail = serve.health()
    console.print(f"unit: {state}   health: {'OK' if ok else detail}")


@app.command()
def test(
    prompt: str = typer.Argument(
        "Reply with exactly: DEEPME OK. Then briefly introduce yourself."
    ),
    max_tokens: int = typer.Option(512),
    pad_tokens: int = typer.Option(
        0, help="Prepend ~N tokens of filler to test long-context prefill"
    ),
) -> None:
    """Send one chat completion and report timing."""
    console.print(f"models: {client.models()}")
    if pad_tokens:
        filler = "The quick brown fox jumps over the lazy dog. " * (
            pad_tokens // 10 + 1
        )
        prompt = f"{filler}\n\nIgnore all text above. {prompt}"
    result = client.chat(prompt, max_tokens=max_tokens)
    console.print(result.text)
    console.print(
        f"\n[dim]prompt_tokens={result.prompt_tokens} completion_tokens={result.completion_tokens} "
        f"ttft={result.ttft_s:.2f}s decode={result.decode_tps:.1f} tok/s[/dim]"
    )


@app.command()
def bench(
    concurrency: list[int] = typer.Option([1, 4, 16], "--concurrency", "-c"),
    max_tokens: int = typer.Option(1024),
    out: str = typer.Option(None, help="Also write JSON results to this file"),
) -> None:
    """Benchmark TTFT + throughput at multiple concurrency levels."""
    rows = bench_mod.run(list(concurrency), max_tokens)
    table = Table(title=f"{config.SERVED_MODEL_NAME} @ {config.API_BASE}")
    for col in (
        "concurrency",
        "completed",
        "ttft_s_median",
        "per_stream_tps_median",
        "aggregate_output_tps",
        "total_output_tokens",
        "wall_s",
    ):
        table.add_column(col)
    for r in rows:
        table.add_row(
            *(
                str(r.get(c, "-"))
                for c in (
                    "concurrency",
                    "completed",
                    "ttft_s_median",
                    "per_stream_tps_median",
                    "aggregate_output_tps",
                    "total_output_tokens",
                    "wall_s",
                )
            )
        )
    console.print(table)
    if out:
        with open(out, "w") as f:
            json.dump(rows, f, indent=2)
        console.print(f"wrote {out}")


@app.command()
def ssh(
    command: str = typer.Argument(..., help="Command to run on the server"),
) -> None:
    """Run an arbitrary command on the server (escape hatch)."""
    remote.run(command, stream=True, check=False, timeout=None)


def main() -> None:
    app()
