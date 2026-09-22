"""Build engine launch commands and manage the inference service via systemd.

The service runs as a transient-free, persistent systemd unit so it survives
SSH disconnects and can be inspected with journalctl. Each `start` rewrites
the unit file from the current ServeOptions, so iterating on flags is just
`uv run deepme serve start --engine ... --extra ...` again.
"""

from __future__ import annotations

import json
import shlex
import time

import httpx

from . import config, remote
from .config import ServeOptions

UNIT_PATH = f"/etc/systemd/system/{config.SYSTEMD_UNIT}.service"


def build_command(opts: ServeOptions) -> list[str]:
    """Translate ServeOptions into the engine's CLI invocation.

    Flag choices follow the DeepSeek-V4-Flash-0731 model card, adapted for
    8x RTX 5090: weights are ~167GB FP8 so we shard across all 8 GPUs with
    tensor parallelism (the card's examples assume larger datacenter GPUs).
    """
    key = config.api_key()
    if opts.engine == "sglang":
        cmd = [
            config.venv_python("sglang"),
            "-m",
            "sglang.launch_server",
            "--model-path",
            config.MODEL_PATH,
            "--served-model-name",
            config.SERVED_MODEL_NAME,
            "--trust-remote-code",
            "--tp",
            str(opts.tp),
            "--context-length",
            str(opts.max_model_len),
            "--mem-fraction-static",
            str(opts.gpu_mem_util),
            "--chunked-prefill-size",
            "4096",
            "--moe-runner-backend",
            "flashinfer_mxfp4",
            "--swa-full-tokens-ratio",
            "0.1",
            "--host",
            "0.0.0.0",
            "--port",
            str(config.API_PORT),
            "--api-key",
            key,
        ]
        if opts.spec_decode:
            cmd += ["--speculative-algorithm", "DSPARK"]
        if opts.kv_ram_gb:
            # HiCache: hierarchical KV cache tiered into system RAM.
            cmd += [
                "--enable-hierarchical-cache",
                "--hicache-size",
                str(opts.kv_ram_gb),
            ]
    elif opts.engine == "vllm":
        cmd = [
            f"{config.engine_venv('vllm')}/bin/vllm",
            "serve",
            config.MODEL_PATH,
            "--served-model-name",
            config.SERVED_MODEL_NAME,
            "--trust-remote-code",
            "--tensor-parallel-size",
            str(opts.tp),
            "--enable-expert-parallel",
            "--max-model-len",
            str(opts.max_model_len),
            "--gpu-memory-utilization",
            str(opts.gpu_mem_util),
            "--kv-cache-dtype",
            "fp8",
            "--host",
            "0.0.0.0",
            "--port",
            str(config.API_PORT),
            "--api-key",
            key,
        ]
        if opts.spec_decode:
            spec = {
                "method": "dspark",
                "num_speculative_tokens": opts.spec_tokens,
                "draft_sample_method": "greedy",
            }
            cmd += ["--speculative-config", json.dumps(spec)]
        if opts.kv_ram_gb:
            # OffloadingConnector: spill evicted KV blocks to CPU RAM and
            # restore on prefix-cache hits (needs prefix caching, default on).
            kv_cfg = {
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "cpu_bytes_to_use": opts.kv_ram_gb * 1024**3,
                },
            }
            cmd += ["--kv-transfer-config", json.dumps(kv_cfg)]
    else:
        raise ValueError(
            f"unknown engine {opts.engine!r}; choose from {config.ENGINES}"
        )
    cmd += opts.extra_args
    return cmd


def start(opts: ServeOptions) -> None:
    exec_line = " ".join(shlex.quote(a) for a in build_command(opts))
    unit = f"""\
[Unit]
Description=deepme LLM inference ({opts.engine}, {config.SERVED_MODEL_NAME})
After=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
User={config.SSH_USER}
Environment=HF_HOME={config.REMOTE_HOME}/.cache/huggingface
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_HOME={config.venv_cuda_home(opts.engine)}
Environment=PATH={config.engine_venv(opts.engine)}/bin:{config.venv_cuda_home(opts.engine)}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart={exec_line}
Restart=on-failure
RestartSec=10
LimitNOFILE=1048576
KillMode=mixed
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
"""
    remote.write_remote_file(UNIT_PATH, unit, sudo=True)
    remote.run(
        f"sudo systemctl daemon-reload && sudo systemctl restart {config.SYSTEMD_UNIT}"
    )


def stop() -> None:
    remote.run(f"sudo systemctl stop {config.SYSTEMD_UNIT}", check=False)


def unit_state() -> str:
    r = remote.run(f"systemctl is-active {config.SYSTEMD_UNIT}", check=False)
    return r.stdout.strip() or r.stderr.strip() or "unknown"


def health() -> tuple[bool, str]:
    """Probe the OpenAI-compatible server's health endpoint from here."""
    try:
        r = httpx.get(
            f"{config.API_BASE}/health",
            timeout=5,
            headers={"Authorization": f"Bearer {config.api_key()}"},
        )
        return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001 - report any probe failure as text
        return False, type(exc).__name__


def wait_healthy(timeout_s: float = 1800, poll_s: float = 10) -> tuple[bool, str]:
    """Wait for the server to come up (weight loading takes minutes).

    Returns (ok, reason); on failure the reason includes a log excerpt.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ok, _ = health()
        if ok:
            return True, "healthy"
        state = unit_state()
        if state in ("failed", "inactive", "dead"):
            return False, f"service is {state}\n" + diagnose()
        time.sleep(poll_s)
    return False, f"timed out after {timeout_s}s\n" + diagnose()


def diagnose(lines: int = 3000) -> str:
    """Summarize service health: restart count and the real crash evidence.

    The engines log benign lines containing 'error' (e.g. "Ignore import
    error"), so we filter those out and surface tracebacks/exit reasons.
    """
    script = f"""\
echo "restarts=$(systemctl show -p NRestarts --value {config.SYSTEMD_UNIT}) state=$(systemctl is-active {config.SYSTEMD_UNIT})"
sudo journalctl -u {config.SYSTEMD_UNIT} -n {lines} --no-pager -o cat \\
  | grep -v 'Ignore import error' \\
  | awk '/Main process exited|Failed with result/ {{n=NR}} {{l[NR]=$0}} END {{ if (n==0) n=NR; s=n-45; if (s<1) s=1; for (i=s; i<=n; i++) print l[i] }}'
"""
    r = remote.run_script(script, check=False)
    return r.stdout.strip() or "(no logs)"


def restart() -> None:
    """Clear failure state and restart the existing unit unchanged."""
    remote.run(
        f"sudo systemctl reset-failed {config.SYSTEMD_UNIT} 2>/dev/null;"
        f" sudo systemctl restart {config.SYSTEMD_UNIT}"
    )


def tail_logs(lines: int = 100, follow: bool = False) -> None:
    flags = f"-n {lines}" + (" -f" if follow else "")
    remote.run(
        f"sudo journalctl -u {config.SYSTEMD_UNIT} {flags} --no-pager",
        stream=True,
        timeout=None,
        check=False,
    )
