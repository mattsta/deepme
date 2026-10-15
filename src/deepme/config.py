"""Central configuration for the deepme inference infrastructure.

Everything about the remote server and the model lives here so every
command shares one source of truth. Values can be overridden via
DEEPME_* environment variables where noted.
"""

from __future__ import annotations

import functools
import json
import os
import secrets
import tempfile
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = REPO_ROOT / ".deepme"
# Modules the balancer and gateway BOTH run. They are deployed as flat files
# importing siblings by bare name, so anything shared cannot live inside
# either service's directory — it is copied into both by their ctl modules.
SHARED_DIR = REPO_ROOT / "src" / "deepme" / "shared"
CONFIG_FILE = STATE_DIR / "config.toml"

# ---------------------------------------------------------------------------
# Remote server — configured, never hardcoded.
#
# Precedence: DEEPME_* environment variables > .deepme/config.toml (created
# with `deepme config init`, gitignored) > defaults. The repo itself carries
# no server identities so it can be published as-is.
# ---------------------------------------------------------------------------


def write_text_atomic(path: Path, text: str) -> None:
    """Publish `text` at `path` or leave the previous content untouched.

    THE write-then-rename implementation for the whole project. Six sites
    hand-rolled this and three of them were not actually atomic — including
    the story runner's state file, the sole record of a multi-hour run.
    `os.replace` is atomic within a filesystem, and the fsyncs mean the
    guarantee survives a power loss, not just a killed process: without
    them the rename can land before the data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        # os.fdopen, not Path.open: mkstemp already returns an OPEN fd, and
        # re-opening by name would race a concurrent writer for the path.
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
        # The rename itself has to reach the disk, or a crash can resurrect
        # the old entry with the new file's data already gone.
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _file_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return tomllib.loads(CONFIG_FILE.read_text())
        except (tomllib.TOMLDecodeError, OSError) as exc:
            raise SystemExit(
                f"{CONFIG_FILE} is unreadable ({exc}) — fix or delete it, then "
                f"re-run `uv run deepme config init --host <ip>`"
            ) from None
    return {}


def section(name: str) -> dict:
    """One section of the config file. The public read half — `profiles`
    was reaching into `_file_config` directly, which is why no caching or
    validation could ever be added here."""
    return dict(_file_config().get(name, {}))


def _toml_value(v) -> str:
    if isinstance(v, str):
        # TOML basic strings share JSON's escape grammar, and stdlib ships
        # no TOML writer — so json.dumps IS the correct escaper. Wrapping
        # raw text in quotes produced a file tomllib then refused to parse
        # for any value containing a quote, backslash or newline, and
        # `_file_config` is called by every command including the one that
        # would repair it.
        return json.dumps(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(f"cannot serialise {type(v).__name__} to TOML: {v!r}")


def write_config(data: dict) -> None:
    """Serialize a nested {section: {k: v}} dict to .deepme/config.toml,
    preserving every section. The single writer for all config mutations."""
    lines = []
    for section_name, body in data.items():
        lines.append(f"[{section_name}]")
        for k, v in body.items():
            lines.append(f"{k} = {_toml_value(v)}")
        lines.append("")
    write_text_atomic(CONFIG_FILE, "\n".join(lines))


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
    return int(section("cluster").get("gpus", 8))


def cluster_vram_gib() -> float:
    return float(section("cluster").get("vram_gb", 32))


def cluster_ram_gib() -> float:
    """System RAM (for CPU-mode placements). Probed value wins when cached."""
    cfg = section("cluster")
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


@functools.cache
def api_key() -> str:
    """The shared API key, generated and persisted on first use.

    Cached: this is called on EVERY request header build, and each call
    was three syscalls (mkdir, exists, read) on the hot path of a poll
    loop running per in-flight item.

    Created with O_EXCL, not check-then-write: two batch worker threads
    reaching a fresh checkout together both saw no file, generated
    DIFFERENT keys and both wrote — so one thread's requests carried a key
    the other had already replaced on disk, and only one was ever
    provisioned to the server. Sporadic 401s on the first run. O_EXCL also
    creates at 0600 directly, closing the window where the key was
    world-readable between write and chmod.
    """
    STATE_DIR.mkdir(exist_ok=True)
    key_file = STATE_DIR / "api_key"
    key = "dm-" + secrets.token_urlsafe(32)
    try:
        fd = os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return key_file.read_text().strip()
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    return key
