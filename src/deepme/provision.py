"""Provision the remote server.

Each engine is a uv *project*: its pyproject.toml lives in this repo under
deploy/engines/<engine>/, gets pushed to ~/deepme-runtime/<engine> on the
server, and is materialized with `uv sync`. All dependency changes go through
those pyprojects (edit + re-run `deepme setup`) — never ad-hoc pip installs.
"""

from __future__ import annotations

from . import config, remote

DRIVER_SERIES = "595"

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


def provision(engines: tuple[str, ...] = config.ENGINES) -> str:
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
            "command -v ninja >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ninja-build",
            "command -v g++ >/dev/null || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential",
        ]
    )
    outputs.append(remote.run_script(base, timeout=600).stdout)

    for eng in engines:
        pyproject = ENGINE_PROJECTS_DIR / eng / "pyproject.toml"
        if not pyproject.exists():
            raise FileNotFoundError(f"no engine project: {pyproject}")
        edir = config.engine_dir(eng)
        remote.run(f"mkdir -p {edir}")
        remote.write_remote_file(f"{edir}/pyproject.toml", pyproject.read_text())
        sync = "\n".join(
            [
                "set -euo pipefail",
                f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH",
                f"cd {edir}",
                f"echo '=== uv sync {eng} ==='",
                f"uv sync -p {config.ENGINE_PYTHON} 2>&1 | tail -3",
                _post_sync_fixups(eng),
                f"echo '=== {eng} key packages ==='",
                "uv pip list --python .venv/bin/python | "
                "grep -E '^(vllm|sglang|torch|flashinfer|nvidia-cuda-(nvcc|runtime)) ' || true",
            ]
        )
        outputs.append(remote.run_script(sync, timeout=3600).stdout)
    return "\n".join(outputs)


def gpu_stack() -> str:
    """Ensure a consistent NVIDIA driver stack: userland and kernel module at
    the same version, module reloaded if they diverge. RTX 5090 (Blackwell)
    REQUIRES the -open kernel module variant. Idempotent."""
    script = f"""\
set -euo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \\
  libnvidia-compute-{DRIVER_SERIES} nvidia-utils-{DRIVER_SERIES} \\
  nvidia-compute-utils-{DRIVER_SERIES} \\
  > /tmp/deepme-gpu-stack.log 2>&1 || {{ tail -5 /tmp/deepme-gpu-stack.log; exit 1; }}
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \\
  "linux-modules-nvidia-{DRIVER_SERIES}-open-$(uname -r)" >> /tmp/deepme-gpu-stack.log 2>&1 \\
  || sudo DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-dkms-{DRIVER_SERIES}-open \\
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
    steps = ["set -u", f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH"]
    for eng in config.ENGINES:
        steps += [
            f"echo '--- {eng} ---'",
            f"uv pip list --python {config.venv_python(eng)} 2>/dev/null"
            " | grep -E '^(vllm|sglang|torch|flashinfer|transformers|nvidia-cuda-(nvcc|runtime)) '"
            " || echo 'not installed'",
        ]
    return remote.run_script("\n".join(steps), check=False).stdout
