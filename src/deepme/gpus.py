"""GPU inspection and reclamation on the server."""

from __future__ import annotations

from . import config, remote


def summary() -> str:
    """Per-GPU utilization/memory plus every compute process with its command."""
    script = """\
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
echo "--- compute processes ---"
pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u)
if [ -z "$pids" ]; then
  echo "(none)"
else
  nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader
  for p in $pids; do ps -o pid=,user=,etime=,args= -p "$p" 2>/dev/null | cut -c1-160; done
fi
"""
    return remote.run_script(script, check=False).stdout


def reclaim() -> str:
    """Kill every GPU compute process that is NOT our systemd service, freeing
    all 8 GPUs for the inference deployment. Also clears systemd failure state.
    """
    script = f"""\
svc_pids=$(systemctl show -p MainPID --value {config.SYSTEMD_UNIT} 2>/dev/null)
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
  # keep anything in our service's cgroup
  if grep -q {config.SYSTEMD_UNIT} /proc/$p/cgroup 2>/dev/null; then
    echo "keeping $p (our service)"
    continue
  fi
  echo "killing $p: $(ps -o args= -p $p 2>/dev/null | cut -c1-120)"
  kill "$p" 2>/dev/null || true
done
sleep 5
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sort -u); do
  grep -q {config.SYSTEMD_UNIT} /proc/$p/cgroup 2>/dev/null || kill -9 "$p" 2>/dev/null || true
done
sudo systemctl reset-failed {config.SYSTEMD_UNIT} 2>/dev/null || true
sleep 2
echo "--- after ---"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
"""
    return remote.run_script(script, check=False, timeout=120).stdout
