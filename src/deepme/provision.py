"""Provision the remote server.

Each engine is a uv *project*: its pyproject.toml lives in this repo under
deploy/engines/<engine>/, gets pushed to ~/deepme-runtime/<engine> on the
server, and is materialized with `uv sync`. All dependency changes go through
those pyprojects (edit + re-run `deepme setup`) — never ad-hoc pip installs.
"""

from __future__ import annotations

from . import config, remote

# NVIDIA driver series (userland + -open kernel module). Override per
# machine in .deepme/config.toml: [server] driver_series = "600"
DRIVER_SERIES = str(config._file_config().get("server", {}).get("driver_series", "595"))

ENGINE_PROJECTS_DIR = config.REPO_ROOT / "deploy" / "engines"


def _post_sync_fixups(engine: str) -> str:
    """Filesystem fixups the wheels themselves can't express.

    The merged nvidia/cu13 toolkit dir ships only `lib/` with versioned
    sonames; JIT builders link with `-L $CUDA_HOME/lib64 -lcudart`.
    """
    cuda = config.venv_cuda_home(engine)
    return (
        f"if [ -d {cuda}/lib ]; then "
        f"ln -sfn lib {cuda}/lib64; "
        f"for so in {cuda}/lib/lib*.so.*; do "
        f'base=$(basename "$so"); ln -sf "$base" "{cuda}/lib/${{base%%.so.*}}.so"; '
        f"done; fi"
    )


def provision(engines: tuple[str, ...] | None = None, *, upgrade: bool = False) -> str:
    """Sync engine venvs. By default FROZEN to the committed uv.lock (exact,
    validated versions — reproducible on any fresh system). `upgrade=True`
    re-resolves within the pyproject ranges and refreshes the committed lock;
    use it deliberately (then bench + validate before trusting)."""
    from . import engines as engines_mod

    # Default: production engines only (llamacpp is experimental, opt-in via -e)
    if engines is None:
        engines = engines_mod.names(include_experimental=False)
    outputs = []
    # System build tools for the remaining JIT compilers (tvm_ffi, tilelang,
    # deep_gemm JIT-compile model-specific kernels at startup and shell out
    # to ninja + a host compiler). No system CUDA — the engine venv's pip
    # toolkit is the single toolchain; a second one poisons auto-detection.
    base = "\n".join(
        [
            "set -euo pipefail",
            f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH",
            f"mkdir -p {config.REMOTE_MODELS_DIR} {config.REMOTE_RUNTIME_DIR}",
            "command -v ninja >/dev/null || sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y -qq ninja-build",
            "command -v g++ >/dev/null || sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y -qq build-essential",
            # Media pipelines mux their output and then VALIDATE it with
            # ffprobe; without it a generation runs to completion and is
            # thrown away at the last step.
            "command -v ffprobe >/dev/null || sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y -qq ffmpeg",
        ]
    )
    outputs.append(remote.run_script(base, timeout=600).stdout)

    from . import engines as engines_mod

    for eng in engines:
        edef = engines_mod.get(eng)
        pyproject = ENGINE_PROJECTS_DIR / eng / "pyproject.toml"
        if not pyproject.exists():
            raise FileNotFoundError(f"no engine project: {pyproject}")
        edir = config.engine_dir(eng)
        lockfile = ENGINE_PROJECTS_DIR / eng / "uv.lock"
        remote.run(f"mkdir -p {edir}")
        remote.write_remote_file(f"{edir}/pyproject.toml", pyproject.read_text())
        # Reproducible by default: push the committed lock and sync --frozen
        # (exact versions, no resolution). Only re-resolve on explicit upgrade.
        if lockfile.exists() and not upgrade:
            remote.write_remote_file(f"{edir}/uv.lock", lockfile.read_text())
            sync_cmd = f"uv sync -p {config.ENGINE_PYTHON} --frozen"
        else:
            sync_cmd = f"uv sync -p {config.ENGINE_PYTHON}" + (" --upgrade" if upgrade else "")
        # {cuda} in sync_env resolves to the venv's own pip CUDA toolkit.
        cuda_home = config.venv_cuda_home(eng)
        uvbin = f"{config.REMOTE_HOME}/.local/bin"
        sync_env = " ".join(
            f"{k}={v.format(cuda=cuda_home, uvbin=uvbin)!r}" for k, v in edef.sync_env.items()
        )
        prepare_steps = []
        if edef.sync_prepare_group:
            # Declarative first pass: install ONLY the build toolchain group so
            # the engine's compile step finds nvcc/headers (no ad-hoc installs).
            prepare_steps = [
                f"echo '=== build toolchain ({edef.sync_prepare_group}) for {eng} ==='",
                f"uv sync -p {config.ENGINE_PYTHON} --only-group "
                f"{edef.sync_prepare_group} 2>&1 | tail -2",
                _post_sync_fixups(eng),
            ]
        sync = "\n".join(
            [
                "set -euo pipefail",
                f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH",
                f"cd {edir}",
                *prepare_steps,
                f"echo '=== uv sync {eng} ({'frozen' if (lockfile.exists() and not upgrade) else 'resolve'}) ==='",
                f"env {sync_env} {sync_cmd} 2>&1 | tail -3",
                _post_sync_fixups(eng),
                f"echo '=== {eng} key packages ==='",
                "uv pip list --python .venv/bin/python | "
                "grep -E '^(vllm|sglang|torch|flashinfer|nvidia-cuda-(nvcc|runtime)) ' || true",
            ]
        )
        outputs.append(remote.run_script(sync, timeout=3600).stdout)
        # Vendored patches go on AFTER the sync that would overwrite them, so
        # a fresh machine and a long-lived one end up with identical source.
        # Idempotent, and refuses to apply to code an upgrade has moved.
        from . import patches as patches_mod

        if patches_mod.load(eng):
            outputs.append(f"=== {eng} vendored patches ===\n" + "\n".join(patches_mod.apply(eng)))
        # Runtime hooks ride along for the same reason: a sync rebuilds the
        # venv, so a fresh machine must not need a separate install-hooks
        # step to match a long-lived one. Idempotent; the hook itself is
        # env-gated and no-ops if the engine internals it names have moved.
        outputs.append(f"=== {eng} runtime hooks ===\n" + "\n".join(patches_mod.install_hooks(eng)))
        # On upgrade, pull the freshly-resolved lock back so the repo records
        # the new validated versions.
        if upgrade:
            got = remote.run(f"cat {edir}/uv.lock", check=False)
            if got.ok and got.stdout.strip():
                lockfile.write_text(got.stdout)
                outputs.append(f"(updated committed lock: {lockfile})")
    return "\n".join(outputs)


def preview_upgrade(engine: str) -> str:
    """Resolution-only upgrade preview: what would a newer resolution pull,
    and does it even resolve — WITHOUT installing or touching the live venv.

    Runs `uv lock --upgrade` in a throwaway staging dir (resolves + writes a
    lockfile; never builds a venv or installs), then diffs key package
    versions committed-lock vs freshly-resolved. Safe while production serves."""
    from . import engines as engines_mod

    engines_mod.get(engine)  # validates the engine name
    pyproject = ENGINE_PROJECTS_DIR / engine / "pyproject.toml"
    lockfile = ENGINE_PROJECTS_DIR / engine / "uv.lock"
    if not pyproject.exists():
        raise FileNotFoundError(f"no engine project: {pyproject}")
    stage = f"{config.REMOTE_RUNTIME_DIR}/{engine}-preview"
    remote.run(f"rm -rf {stage} && mkdir -p {stage}")
    remote.write_remote_file(f"{stage}/pyproject.toml", pyproject.read_text())
    if lockfile.exists():
        remote.write_remote_file(f"{stage}/uv.lock", lockfile.read_text())
    script = "\n".join(
        [
            "set -euo pipefail",
            f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH",
            f"cd {stage}",
            "echo '=== resolving upgrade (no install) ==='",
            f"uv lock --upgrade -p {config.ENGINE_PYTHON} 2>&1 | tail -4",
            "echo '=== resolved key versions ==='",
            "for pkg in vllm sglang torch flashinfer-python nvidia-cuda-nvcc "
            "nvidia-cuda-runtime transformers; do "
            r"""v=$(awk -v p="$pkg" '$0=="name = \""p"\""{f=1} f&&/^version/{gsub(/[",]/,"",$3);print $3;f=0}' uv.lock); """
            'echo "  $pkg = ${v:-(absent)}"; done',
        ]
    )
    out = remote.run_script(script, check=False, timeout=600).stdout
    remote.run(f"rm -rf {stage}", check=False)
    return out


def gpu_stack() -> str:
    """Ensure a consistent NVIDIA driver stack: userland and kernel module at
    the same version, module reloaded if they diverge. RTX 5090 (Blackwell)
    REQUIRES the -open kernel module variant. Idempotent."""
    script = f"""\
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y \\
  libnvidia-compute-{DRIVER_SERIES} nvidia-utils-{DRIVER_SERIES} \\
  nvidia-compute-utils-{DRIVER_SERIES} \\
  > /tmp/deepme-gpu-stack.log 2>&1 || {{ tail -5 /tmp/deepme-gpu-stack.log; exit 1; }}
sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y \\
  "linux-modules-nvidia-{DRIVER_SERIES}-open-$(uname -r)" >> /tmp/deepme-gpu-stack.log 2>&1 \\
  || sudo DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l apt-get install -y nvidia-dkms-{DRIVER_SERIES}-open \\
     >> /tmp/deepme-gpu-stack.log 2>&1 || true
ondisk=$(modinfo nvidia | awk '/^version:/{{print $2}}')
loaded=$(cat /proc/driver/nvidia/version 2>/dev/null | grep -oE '[0-9]+\\.[0-9]+(\\.[0-9]+)?' | head -1 || echo none)
echo "ondisk=$ondisk loaded=$loaded"
if [ "$ondisk" != "$loaded" ]; then
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
    echo "GPU busy; skipping module reload (stop workloads first)"; exit 1
  fi
  echo "reloading nvidia modules to $ondisk"
  sudo systemctl stop nvidia-persistenced 2>/dev/null || true
  for m in nvidia_uvm nvidia_drm nvidia_modeset nvidia; do sudo rmmod $m 2>/dev/null || true; done
  sudo modprobe nvidia nvidia_uvm
  sudo systemctl start nvidia-persistenced 2>/dev/null || true
fi
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader | head -2
echo GPU_STACK_OK
"""
    return remote.run_script(script, check=False, timeout=900).stdout


def versions() -> str:
    """Report installed engine + toolchain versions per engine project."""
    from . import engines as engines_mod

    steps = ["set -u", f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH"]
    for eng in engines_mod.names():
        steps += [
            f"echo '--- {eng} ---'",
            f"uv pip list --python {config.venv_python(eng)} 2>/dev/null"
            " | grep -E '^(vllm|sglang|torch|flashinfer|transformers|nvidia-cuda-(nvcc|runtime)) '"
            " || echo 'not installed'",
        ]
    return remote.run_script("\n".join(steps), check=False).stdout
