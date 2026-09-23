"""Central configuration for the deepme inference infrastructure.

Everything about the remote server and the model lives here so every
command shares one source of truth. Values can be overridden via
DEEPME_* environment variables where noted.
"""

from __future__ import annotations

import os
import secrets
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / ".deepme"
CONFIG_FILE = STATE_DIR / "config.toml"

# ---------------------------------------------------------------------------
# Remote server — configured, never hardcoded.
#
# Precedence: DEEPME_* environment variables > .deepme/config.toml (created
# with `deepme config init`, gitignored) > defaults. The repo itself carries
# no server identities so it can be published as-is.
# ---------------------------------------------------------------------------


def _file_config() -> dict:
    if CONFIG_FILE.exists():
        return tomllib.loads(CONFIG_FILE.read_text())
    return {}


def _toml_value(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    return str(v)


def write_config(data: dict) -> None:
    """Serialize a nested {section: {k: v}} dict to .deepme/config.toml,
    preserving every section. The single writer for all config mutations."""
    lines = []
    for section, body in data.items():
        lines.append(f"[{section}]")
        for k, v in body.items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    STATE_DIR.mkdir(exist_ok=True)
    CONFIG_FILE.write_text("\n".join(lines))


def update_config(section: str, **kv) -> None:
    """Merge keys into one section of config.toml, keeping all other sections."""
    data = _file_config()
    data.setdefault(section, {}).update(kv)
    write_config(data)


_server = _file_config().get("server", {})

SSH_HOST = os.environ.get("DEEPME_HOST") or _server.get("host", "")
SSH_USER = os.environ.get("DEEPME_USER") or _server.get("user", "ubuntu")
SSH_TARGET = f"{SSH_USER}@{SSH_HOST}"


def require_host() -> None:
    """Fail with guidance when no server is configured."""
    if not SSH_HOST:
        raise SystemExit(
            "no server configured — run: uv run deepme config init "
            "--host <ip-or-hostname> [--user <ssh-user>] [--port <api-port>] "
            "(or set DEEPME_HOST / DEEPME_USER)"
        )


# Remote filesystem layout (all under the SSH user's home)
REMOTE_HOME = f"/home/{SSH_USER}"
REMOTE_MODELS_DIR = f"{REMOTE_HOME}/models"
# Each engine is a uv project (pyproject in this repo under deploy/engines/,
# synced to the server and materialized with `uv sync` — never `uv pip`).
REMOTE_RUNTIME_DIR = f"{REMOTE_HOME}/deepme-runtime"
REMOTE_UV = f"{REMOTE_HOME}/.local/bin/uv"


def engine_dir(engine: str) -> str:
    return f"{REMOTE_RUNTIME_DIR}/{engine}"


def engine_venv(engine: str) -> str:
    return f"{engine_dir(engine)}/.venv"


# Python version used for the engine venvs (engines lag CPython releases).
ENGINE_PYTHON = "3.12"


# CUDA toolchain. The engine venvs bundle a COMPLETE pip CUDA toolkit
# (nvidia/cu13: nvcc, ptxas, headers) that flashinfer's JIT builds against.
# CUDA_HOME must point at that same toolkit — mixing it with a system
# /usr/local/cuda of a different minor version makes nvcc emit PTX that the
# PATH-resolved ptxas rejects ("Unsupported .version").
def venv_cuda_home(engine: str) -> str:
    return f"{engine_venv(engine)}/lib/python{ENGINE_PYTHON}/site-packages/nvidia/cu13"


# ---------------------------------------------------------------------------
# Engines
#
# Model identity/serving parameters live in profiles (deploy/models/*.toml,
# see profiles.py) — nothing model-specific belongs in this module.
# ---------------------------------------------------------------------------

# Engines we know how to install and launch. Each is a uv project on the
# server so they never fight over torch/flashinfer versions.
# NOTE: the engine list is the registry in engines.py — use
# engines.names(include_experimental=False) for the provisionable default and
# engines.names() for all. config must not import engines (circular), so this
# alias is intentionally not defined here.
# vllm is the production engine: sglang 0.5.17's DSpark draft autotune hits an
# unsupported sparse-MLA kernel shape on SM120 (as does vllm 0.26's DSpark) —
# revisit spec decode when either engine ships the SM120 kernels.
DEFAULT_ENGINE = "vllm"


def venv_python(engine: str) -> str:
    return f"{engine_venv(engine)}/bin/python"


# ---------------------------------------------------------------------------
# Cluster shape — single source of truth for VRAM budget math (fit checks,
# catalog capacity, perf estimates all derive from these). Override in
# .deepme/config.toml: [cluster] gpus = 8, vram_gb = 32
# ---------------------------------------------------------------------------

# Usable VRAM fraction and fixed per-GPU runtime overhead used by fit/capacity
# math when a profile's own gpu_mem isn't the reference point.
USABLE_VRAM_FRACTION = 0.90
PER_GPU_OVERHEAD_GIB = 3.0


def cluster_gpu_count() -> int:
    return int(_file_config().get("cluster", {}).get("gpus", 8))


def cluster_vram_gib() -> float:
    return float(_file_config().get("cluster", {}).get("vram_gb", 32))


def cluster_ram_gib() -> float:
    """System RAM (for CPU-mode placements). Probed value wins when cached."""
    cfg = _file_config().get("cluster", {})
    if "ram_gb" in cfg:
        return float(cfg["ram_gb"])
    cache = STATE_DIR / "capacity.json"
    if cache.exists():
        import json

        try:
            return float(json.loads(cache.read_text()).get("ram_gib") or 503)
        except Exception:
            pass
    return 503.0


def per_gpu_weight_budget_gib(gpu_mem: float | None = None) -> float:
    """GiB available for WEIGHTS on one GPU after runtime overhead. gpu_mem
    (a profile's memory fraction) overrides the default usable fraction."""
    frac = gpu_mem if gpu_mem is not None else USABLE_VRAM_FRACTION
    return cluster_vram_gib() * frac - PER_GPU_OVERHEAD_GIB


# ---------------------------------------------------------------------------
# API key (generated once, stored locally, pushed to the server env)
# ---------------------------------------------------------------------------


def api_key() -> str:
    """Return the shared API key, generating and persisting it on first use."""
    STATE_DIR.mkdir(exist_ok=True)
    key_file = STATE_DIR / "api_key"
    if key_file.exists():
        return key_file.read_text().strip()
    key = "dm-" + secrets.token_urlsafe(32)
    key_file.write_text(key + "\n")
    key_file.chmod(0o600)
    return key
