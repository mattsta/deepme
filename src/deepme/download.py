"""Download model weights onto the server, driven by a RunSpec.

Runs on the server inside a detached tmux session (per model — downloads
survive our SSH session and can proceed in parallel), using `hf download`
with hf_transfer. Resumes cleanly; re-running is always safe. Completeness
is judged against the shard count reported by the Hugging Face API.
"""

from __future__ import annotations

import shlex

from . import config, remote
from .profiles import RunSpec


def _session(spec: RunSpec) -> str:
    return f"deepme-dl-{spec.profile}"


def _log(spec: RunSpec) -> str:
    return f"{config.REMOTE_HOME}/deepme-download-{spec.profile}.log"


# HF CDN throttles to ~5MB/s per connection from this box; high worker count
# (plus hf_transfer's chunked transfers) is what makes 100GB+ fetches tolerable.
MAX_WORKERS = 24


def _selection_args(spec: RunSpec) -> list[str]:
    """Download only what this deployment needs.

    GGUF: the resolved quant variant (+ its multi-part siblings) plus repo
    metadata — never every quantization in the repo (10x+ waste).
    Safetensors: whole repo minus any bundled .gguf variants."""
    # NOTE: each pattern needs its OWN --include flag. `hf download --include
    # A B C` silently DROPS pattern A (it downloads B and C only), which
    # quietly skipped the actual weights while reporting success.
    if spec.format == "gguf":
        stem = spec.variant.removesuffix(".gguf")
        patterns = [f"{stem}*.gguf", "*.json", "*tokenizer*", "*.txt", "*.md", "*.py"]
        return [arg for p in patterns for arg in ("--include", p)]
    return ["--exclude", "*.gguf"]


def _runner_path(spec: RunSpec) -> str:
    return f"{config.REMOTE_HOME}/deepme-download-{spec.profile}.sh"


def start(spec: RunSpec) -> str:
    """Launch the download in a detached tmux session.

    The command is written to a SCRIPT FILE and tmux runs that file — passing
    a quoted command string through tmux -> sh -c mangles nested quoting and
    silently drops glob patterns (that bug cost a whole download).
    """
    session, log = _session(spec), _log(spec)
    args = " ".join(shlex.quote(a) for a in _selection_args(spec))
    runner = _runner_path(spec)
    remote.write_remote_file(
        runner,
        f"""\
#!/usr/bin/env bash
export PATH={config.REMOTE_HOME}/.local/bin:$PATH
export HF_HUB_ENABLE_HF_TRANSFER=1
uvx --from 'huggingface_hub[hf_transfer]' hf download {shlex.quote(spec.repo)} \\
  --local-dir {shlex.quote(spec.download_path)} --max-workers {MAX_WORKERS} \\
  {args} > {shlex.quote(log)} 2>&1
echo "DOWNLOAD_EXIT=$?" >> {shlex.quote(log)}
""",
    )
    script = f"""\
set -euo pipefail
mkdir -p {shlex.quote(config.REMOTE_MODELS_DIR)}
chmod +x {shlex.quote(runner)}
if tmux has-session -t {shlex.quote(session)} 2>/dev/null; then
  echo "download session already running"
else
  tmux new-session -d -s {shlex.quote(session)} {shlex.quote(runner)}
  echo "download started in tmux session {session}"
fi
"""
    return remote.run_script(script).stdout.strip()


def restart(spec: RunSpec) -> str:
    """Kill any running session for this model and start fresh (resumes)."""
    remote.run(f"tmux kill-session -t {_session(spec)} 2>/dev/null || true", check=False)
    return start(spec)


def _weight_glob(spec: RunSpec) -> str:
    """Glob (relative to model_path) for this deployment's weight files."""
    if spec.format == "gguf":
        stem = spec.variant.removesuffix(".gguf")
        return f"{stem}*.gguf"
    return "*.safetensors"


def _completeness_script(spec: RunSpec) -> str:
    """Shell that echoes 'have=<n> want=<n>'. For multi-part GGUF the expected
    count is read from the -of-NNNNN suffix of the downloaded parts (so a
    single finished part never reads as complete)."""
    qpath = shlex.quote(spec.download_path)
    glob = _weight_glob(spec)
    if spec.format == "gguf":
        return f"""\
files=$(ls {qpath}/{glob} 2>/dev/null)
have=$(echo "$files" | grep -c . )
want=1
part=$(echo "$files" | grep -oE '\\-of-[0-9]+\\.gguf$' | head -1 | grep -oE '[0-9]+' | head -1)
[ -n "$part" ] && want=$part
echo "have=$have want=$want"
"""
    # Pipeline models (diffusers) keep weights in component SUBFOLDERS
    # (transformer/, vae/, text_encoder/...), so count recursively — a
    # top-level glob finds nothing and reads as "0 downloaded".
    return f"""\
have=$(find {qpath} -name '{glob}' -type f 2>/dev/null | grep -c .)
want={spec.shards}
echo "have=$have want=$want"
"""


def session_alive(spec: RunSpec) -> bool:
    r = remote.run(
        f"tmux has-session -t {shlex.quote(_session(spec))} 2>/dev/null && echo YES || echo NO",
        check=False,
    )
    return "YES" in r.stdout


def status(spec: RunSpec) -> str:
    qpath = shlex.quote(spec.download_path)
    script = f"""\
active=no
tmux has-session -t {shlex.quote(_session(spec))} 2>/dev/null && active=yes
size=$(du -s --block-size=1G {qpath} 2>/dev/null | cut -f1 || echo 0)
{_completeness_script(spec)}
echo "session_active=$active downloaded=${{size}}GiB (~{spec.weight_gib:.0f}GiB expected) format={spec.format} weights=$have/$want"
tail -2 {shlex.quote(_log(spec))} 2>/dev/null || true
"""
    return remote.run_script(script, check=False).stdout


def is_complete(spec: RunSpec) -> bool:
    script = _completeness_script(spec) + (
        '\n[ "$have" -ge "$want" ] && [ "$want" -gt 0 ] && echo COMPLETE || echo INCOMPLETE\n'
    )
    return "COMPLETE" in remote.run_script(script, check=False).stdout


def exit_status(spec: RunSpec) -> int | None:
    """Recorded DOWNLOAD_EXIT from the log, or None if not yet finished."""
    r = remote.run(
        f"grep -oE 'DOWNLOAD_EXIT=[0-9]+' {shlex.quote(_log(spec))} 2>/dev/null "
        "| tail -1 | grep -oE '[0-9]+'",
        check=False,
    )
    s = r.stdout.strip()
    return int(s) if s.isdigit() else None


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
