"""Central configuration for the deepme inference infrastructure.

Everything about the remote server and the model lives here so every
command shares one source of truth. Values can be overridden via
DEEPME_* environment variables where noted.
"""

from __future__ import annotations

import os
import secrets
import tomllib
from dataclasses import dataclass, field
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
            "(or set DEEPME_HOST / DEEPME_USER / DEEPME_PORT)"
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
# Model
# ---------------------------------------------------------------------------

MODEL_REPO = "deepseek-ai/DeepSeek-V4-Flash-0731"
MODEL_NAME = "DeepSeek-V4-Flash-0731"  # directory name on the server
SERVED_MODEL_NAME = "deepseek-v4-flash"  # name exposed via the OpenAI API
MODEL_PATH = f"{REMOTE_MODELS_DIR}/{MODEL_NAME}"
MODEL_SIZE_GB = 167  # approximate total safetensors size, used for progress

# Recommended sampling defaults from the model card
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95  # agentic scenarios; 1.0 otherwise

# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

API_PORT = int(os.environ.get("DEEPME_PORT") or _server.get("port", 8000))
API_BASE = f"http://{SSH_HOST}:{API_PORT}"
SYSTEMD_UNIT = "deepme-llm"

# Engines we know how to install and launch. Each gets its own uv-managed
# venv on the server so they never fight over torch/flashinfer versions.
ENGINES = ("sglang", "vllm")
# vllm is the production engine: sglang 0.5.17's DSpark draft autotune hits an
# unsupported sparse-MLA kernel shape on SM120 (as does vllm 0.26's DSpark) —
# revisit spec decode when either engine ships the SM120 kernels.
DEFAULT_ENGINE = "vllm"


def venv_python(engine: str) -> str:
    return f"{engine_venv(engine)}/bin/python"


@dataclass
class ServeOptions:
    """Tunable knobs for a serve deployment; defaults follow the model card
    adapted to 8x RTX 5090 (32GB, PCIe, no NVLink -> full TP across 8)."""

    engine: str = DEFAULT_ENGINE
    tp: int = 8
    max_model_len: int = 262_144  # 1M supported; capped for KV headroom
    gpu_mem_util: float = 0.90
    spec_decode: bool = True  # DSpark, num_speculative_tokens=7
    spec_tokens: int = 7
    kv_ram_gb: int = 0  # tier KV cache into system RAM (0=off)
    extra_args: list[str] = field(default_factory=list)


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
