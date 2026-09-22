"""Deep diagnostics: run probe scripts inside the engine venvs on the server."""

from __future__ import annotations

from importlib import resources

from . import config, remote

TOOLS_DIR = f"{config.REMOTE_HOME}/deepme-tools"

# The modules whose silent import failure makes an engine fall back to the
# (non-working) transformers implementation for our model.
CRITICAL_MODULES = {
    "sglang": [
        "sglang.srt.models.deepseek_v4",
        "sglang.srt.models.deepseek_v4_dspark",
        "sglang.srt.models.deepseek_v4_nextn",
    ],
    "vllm": [
        "vllm.model_executor.models.deepseek_v2",
        "vllm.model_executor.models.deepseek_v4",
    ],
}


def push_script(name: str) -> str:
    """Push a script from remote_scripts/ to the server; return remote path."""
    content = (resources.files("deepme.remote_scripts") / name).read_text()
    remote.run(f"mkdir -p {TOOLS_DIR}")
    path = f"{TOOLS_DIR}/{name}"
    remote.write_remote_file(path, content)
    return path


def jit_heal(engine: str = config.DEFAULT_ENGINE) -> str:
    """Finish any incomplete flashinfer JIT kernel caches, serially, with the
    engine venv's own CUDA toolkit (consistent nvcc+ptxas). Prints 'no work
    to do' where a cache is already complete."""
    cuda = config.venv_cuda_home(engine)
    script = f"""\
set -u
export CUDA_HOME={cuda} PATH={cuda}/bin:/usr/bin:/bin
found=0
for f in $(find ~/.cache/flashinfer -name build.ninja 2>/dev/null); do
  found=1
  d=$(dirname "$f")
  echo "=== $d"
  (cd "$d" && ninja 2>&1 | tail -4)
done
[ "$found" = 1 ] || echo "no flashinfer build dirs found"
"""
    return remote.run_script(script, check=False, timeout=1200).stdout


def import_probe(engine: str, modules: list[str] | None = None) -> str:
    """Import engine-critical modules in the engine's venv, with tracebacks."""
    mods = modules or CRITICAL_MODULES.get(engine, [])
    if not mods:
        return f"no critical modules defined for {engine}"
    path = push_script("import_probe.py")
    python = config.venv_python(engine)
    cuda = config.venv_cuda_home(engine)
    env = f"CUDA_HOME={cuda} PATH={cuda}/bin:/usr/bin:/bin"
    r = remote.run(f"{env} {python} {path} {' '.join(mods)}", check=False, timeout=300)
    return (r.stdout + r.stderr).strip()
