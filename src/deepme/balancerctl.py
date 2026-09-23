"""Deploy and control the replica load balancer (see balancer/server.py).

Mirrors guardctl: a small FastAPI app in its own uv project on the server,
run by systemd. It fronts N replicas of ONE model profile and is rebuilt
from the live unit list, so scaling up or down is just re-running `start`.
"""

from __future__ import annotations

import json

from . import config, remote
from .profiles import RunSpec

BALANCER_DIR = f"{config.REMOTE_RUNTIME_DIR}/balancer"
# Every model fleet gets its OWN balancer, because balancing is per-model:
# replicas of one model are interchangeable, replicas of different models are
# not. A mixed fleet (h3-video x3 + h3-ref x1) therefore runs two balancers,
# and the gateway fans out to them.


def unit_for(profile: str) -> str:
    return f"deepme-balancer-{profile}"


def port_for(profile: str) -> int:
    """Declared, never derived — see deploy/ports.toml."""
    from . import ports

    return ports.load().balancer_port(profile)


LOCAL = config.REPO_ROOT / "src" / "deepme" / "balancer"
# The guard's uv project already pins fastapi+uvicorn+httpx; the balancer needs
# exactly the same three, so it reuses that pyproject rather than a second copy.
LOCAL_PROJECT = config.REPO_ROOT / "deploy" / "guard"


def setup() -> str:
    remote.run(f"mkdir -p {BALANCER_DIR}")
    remote.write_remote_file(
        f"{BALANCER_DIR}/pyproject.toml", (LOCAL_PROJECT / "pyproject.toml").read_text()
    )
    script = "\n".join(
        [
            "set -euo pipefail",
            f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH",
            f"cd {BALANCER_DIR}",
            f"uv sync -p {config.ENGINE_PYTHON} 2>&1 | tail -2",
        ]
    )
    return remote.run_script(script, timeout=600).stdout


def backends(specs: list[RunSpec]) -> list[str]:
    """Replica base URLs, addressed on the loopback (the balancer runs there)."""
    return [f"http://127.0.0.1:{s.port}" for s in specs]


def _assert_safe_to_freeze(profile: str, *, force: bool) -> None:
    """Refuse to restart a balancer that is holding live state.

    A restart destroys every waiting ticket and the in-memory job map. This
    exact mistake stalled two 2000-item overnight batches: the operator
    checked inflight (0 at that instant) but not the QUEUE, and the deploy
    wiped the tickets their pollers were following. The rule lived in the
    runbook; rules that only live in runbooks get broken, so now the tool
    enforces it.
    """
    import httpx

    try:
        waiting = int(queue(profile).get("waiting") or 0)
        active = int(stats(profile).get("active_jobs") or 0)
    except (httpx.HTTPError, ValueError, SystemExit):
        return  # not running or unreachable: nothing to preserve
    if (waiting or active) and not force:
        raise SystemExit(
            f"{profile}'s balancer holds live state: {waiting} queued "
            f"ticket(s), {active} running job(s). Restarting DESTROYS the "
            f"queue and their callers poll dead ids forever. Wait for it to "
            f"drain (`deepme balancer status -m {profile}`), or pass --force "
            f"to destroy it deliberately."
        )


def start(specs: list[RunSpec], *, force: bool = False) -> str:
    """Point this model's balancer at its replicas and (re)start it."""
    if not specs:
        raise SystemExit("no replicas to route to")
    profile = specs[0].profile
    _assert_safe_to_freeze(profile, force=force)
    unit_name, port = unit_for(profile), port_for(profile)
    for name in ("balance.py", "admission.py", "server.py"):
        remote.write_remote_file(f"{BALANCER_DIR}/{name}", (LOCAL / name).read_text())
    urls = backends(specs)
    unit = f"""\
[Unit]
Description=deepme replica load balancer ({profile} x{len(specs)})
After=network-online.target

[Service]
User={config.SSH_USER}
Environment=BALANCER_BACKENDS={json.dumps(json.dumps(urls))}
Environment=BALANCER_NAME={profile}
Environment=DEEPME_WORK_DISK={config.REMOTE_HOME}/deepme-work
Environment=DEEPME_WORK_DIR={config.REMOTE_HOME}/deepme-work/{profile}
Environment=BALANCER_API_KEY={config.api_key()}
WorkingDirectory={BALANCER_DIR}
ExecStart={BALANCER_DIR}/.venv/bin/uvicorn server:app --host 0.0.0.0 --port {port}
Restart=on-failure
RestartSec=5
"""
    remote.write_remote_file(f"/etc/systemd/system/{unit_name}.service", unit, sudo=True)
    # An explicit start is a FRESH FREEZE of the membership: drop whatever
    # runtime attach/drain state a previous generation persisted, or the
    # restored file would override the very list this deploy declares.
    remote.run(
        f"rm -f {BALANCER_DIR}/backends-{profile}.json && "
        f"sudo systemctl daemon-reload && sudo systemctl reset-failed {unit_name} "
        f"2>/dev/null; sudo systemctl restart {unit_name}"
    )
    return "\n".join(urls)


def stop(profile: str) -> None:
    remote.run(f"sudo systemctl stop {unit_for(profile)}", check=False)


def status(profile: str) -> str:
    r = remote.run(f"systemctl is-active {unit_for(profile)}", check=False)
    return r.stdout.strip() or r.stderr.strip() or "unknown"


def api_base(profile: str) -> str:
    return f"http://{config.SSH_HOST}:{port_for(profile)}"


def stats(profile: str) -> dict:
    """Live per-replica load for one model, as its balancer sees it."""
    import httpx

    r = httpx.get(f"{api_base(profile)}/balancer/stats", timeout=10)
    r.raise_for_status()
    return dict(r.json())


def queue(profile: str) -> dict:
    """This model's admission queue, as its balancer reports it."""
    import httpx

    r = httpx.get(f"{api_base(profile)}/v1/queue", timeout=10)
    r.raise_for_status()
    return dict(r.json())


# -- runtime membership -------------------------------------------------
# The counterpart of the balancer's /balancer/backends API: change which
# replicas a live balancer routes to WITHOUT restarting it, so a fleet
# transition never discards waiting tickets or the job-owner map.


def attach(profile: str, replica_url: str) -> dict:
    import httpx

    r = httpx.post(f"{api_base(profile)}/balancer/backends", json={"url": replica_url}, timeout=30)
    r.raise_for_status()
    return dict(r.json())


def drain(profile: str, replica_url: str) -> dict:
    import httpx

    r = httpx.post(
        f"{api_base(profile)}/balancer/backends/drain", json={"url": replica_url}, timeout=10
    )
    r.raise_for_status()
    return dict(r.json())


def detach(profile: str, replica_url: str) -> dict:
    import httpx

    r = httpx.delete(
        f"{api_base(profile)}/balancer/backends", params={"url": replica_url}, timeout=10
    )
    r.raise_for_status()
    return dict(r.json())


def undrain(profile: str, replica_url: str) -> dict:
    import httpx

    r = httpx.post(
        f"{api_base(profile)}/balancer/backends/undrain", json={"url": replica_url}, timeout=10
    )
    r.raise_for_status()
    return dict(r.json())


def wait_drained(
    profile: str, replica_url: str, *, timeout_s: float, poll_s: float = 10.0, on_tick=None
) -> bool:
    """True once the replica has nothing in flight; False on timeout OR when
    the balancer cannot be asked.

    The wait IS the drain: stopping the unit any earlier kills a paying
    customer's generation mid-flight. A transport error is therefore NOT
    "drained" — the old behaviour returned True on the first failed poll,
    and a balancer restarting (systemd gives it 5s) had the caller conclude
    a busy replica was idle and kill its job. Unreachable means UNKNOWN, and
    unknown work is treated as running.
    """
    import time

    import httpx

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        inflight: int | str = "?"
        try:
            r = httpx.get(f"{api_base(profile)}/balancer/stats", timeout=10)
            r.raise_for_status()
            backends = {b["url"]: b for b in r.json().get("backends", [])}
            st = backends.get(replica_url)
            if st is None or st.get("inflight", 0) == 0:
                return True
            inflight = int(st.get("inflight", 0))
        except httpx.HTTPError:
            pass  # balancer unreachable right now — retry until the deadline
        if on_tick:
            on_tick(inflight)
        time.sleep(poll_s)
    return False


def logs(profile: str, lines: int = 100) -> str:
    return remote.run(
        f"sudo journalctl -u {unit_for(profile)} -n {lines} --no-pager -o cat",
        check=False,
    ).stdout
