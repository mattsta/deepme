"""Download model weights onto the server.

The download runs on the server inside a detached tmux session (167GB —
it must survive our SSH session), using `hf download` with hf_transfer
for maximum throughput. `hf download` resumes cleanly, so re-running is
always safe.
"""

from __future__ import annotations

from . import config, remote

TMUX_SESSION = "deepme-download"
LOG = f"{config.REMOTE_HOME}/deepme-download.log"

# HF CDN throttles to ~5MB/s per connection from this box; high worker count
# (plus hf_transfer's chunked transfers) is what makes the 167GB fetch tolerable.
MAX_WORKERS = 24


def restart() -> None:
    """Kill any running download session and start fresh (download resumes)."""
    remote.run(f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null || true", check=False)
    start()


def start() -> None:
    inner = (
        f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH HF_HUB_ENABLE_HF_TRANSFER=1; "
        f"uvx --from 'huggingface_hub[hf_transfer]' hf download {config.MODEL_REPO} "
        f"--local-dir {config.MODEL_PATH} --max-workers {MAX_WORKERS} 2>&1 | tee {LOG}; "
        f"echo DOWNLOAD_EXIT=$? >> {LOG}"
    )
    script = f"""\
set -euo pipefail
mkdir -p {config.REMOTE_MODELS_DIR}
if tmux has-session -t {TMUX_SESSION} 2>/dev/null; then
  echo "download session already running"
else
  tmux new-session -d -s {TMUX_SESSION} {remote_quote(inner)}
  echo "download started in tmux session {TMUX_SESSION}"
fi
"""
    print(remote.run_script(script).stdout.strip())


def remote_quote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def status() -> str:
    script = f"""\
active=no
tmux has-session -t {TMUX_SESSION} 2>/dev/null && active=yes
size=$(du -sBG {config.MODEL_PATH} 2>/dev/null | cut -f1 || echo 0G)
echo "session_active=$active downloaded=$size target=~{config.MODEL_SIZE_GB}G"
tail -2 {LOG} 2>/dev/null || true
n=$(ls {config.MODEL_PATH}/*.safetensors 2>/dev/null | wc -l)
echo "safetensors_shards=$n/48"
"""
    return remote.run_script(script, check=False).stdout


def is_complete() -> bool:
    script = f"""\
n=$(ls {config.MODEL_PATH}/*.safetensors 2>/dev/null | wc -l)
grep -q DOWNLOAD_EXIT=0 {LOG} 2>/dev/null && [ "$n" -ge 48 ] && echo COMPLETE || echo INCOMPLETE
"""
    return "COMPLETE" in remote.run_script(script, check=False).stdout
