"""Download model weights onto the server, driven by a RunSpec.

Runs on the server inside a detached tmux session (per model — downloads
survive our SSH session and can proceed in parallel), using `hf download`
with hf_transfer. Resumes cleanly; re-running is always safe. Completeness
is judged against the shard count reported by the Hugging Face API.
"""

from __future__ import annotations

from . import config, remote
from .profiles import RunSpec


def _session(spec: RunSpec) -> str:
    return f"deepme-dl-{spec.profile}"


def _log(spec: RunSpec) -> str:
    return f"{config.REMOTE_HOME}/deepme-download-{spec.profile}.log"


# HF CDN throttles to ~5MB/s per connection from this box; high worker count
# (plus hf_transfer's chunked transfers) is what makes 100GB+ fetches tolerable.
MAX_WORKERS = 24


def start(spec: RunSpec) -> str:
    session, log = _session(spec), _log(spec)
    inner = (
        f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1; "
        f"uvx --from 'huggingface_hub[hf_transfer]' hf download {spec.repo} "
        f"--local-dir {spec.model_path} --max-workers {MAX_WORKERS} 2>&1 | tee {log}; "
        f"echo DOWNLOAD_EXIT=$? >> {log}"
    )
    script = f"""\
set -euo pipefail
mkdir -p {config.REMOTE_MODELS_DIR}
if tmux has-session -t {session} 2>/dev/null; then
  echo "download session already running"
else
  tmux new-session -d -s {session} {_quote(inner)}
  echo "download started in tmux session {session}"
fi
"""
    return remote.run_script(script).stdout.strip()


def restart(spec: RunSpec) -> str:
    """Kill any running session for this model and start fresh (resumes)."""
    remote.run(
        f"tmux kill-session -t {_session(spec)} 2>/dev/null || true", check=False
    )
    return start(spec)


def status(spec: RunSpec) -> str:
    script = f"""\
active=no
tmux has-session -t {_session(spec)} 2>/dev/null && active=yes
size=$(du -s --block-size=1G {spec.model_path} 2>/dev/null | cut -f1 || echo 0)
n=$(ls {spec.model_path}/*.safetensors 2>/dev/null | wc -l)
echo "session_active=$active downloaded=${{size}}GiB (~{spec.weight_gib:.0f}GiB weights expected) shards=$n/{spec.shards}"
tail -2 {_log(spec)} 2>/dev/null || true
"""
    return remote.run_script(script, check=False).stdout


def is_complete(spec: RunSpec) -> bool:
    script = f"""\
n=$(ls {spec.model_path}/*.safetensors 2>/dev/null | wc -l)
[ "$n" -ge {spec.shards} ] && [ "{spec.shards}" -gt 0 ] && echo COMPLETE || echo INCOMPLETE
"""
    return "COMPLETE" in remote.run_script(script, check=False).stdout


def _quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"
