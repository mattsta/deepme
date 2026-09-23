"""Operate the output-guard proxy on the server.

The guard is a separate systemd unit (deepme-guard) in front of the model
API — deploying/stopping it never touches the inference service. Policy
lives in deploy/guard/policy.toml (versioned); every `start` pushes the
current policy + code, so policy iteration is edit → `deepme guard start`.
"""

from __future__ import annotations

from . import config, remote
from .profiles import RunSpec

UNIT = "deepme-guard"
GUARD_DIR = f"{config.REMOTE_RUNTIME_DIR}/guard"
# Guard listen port — override in .deepme/config.toml: [server] guard_port
GUARD_PORT = int(config._file_config().get("server", {}).get("guard_port", 8001))

LOCAL_PROJECT = config.REPO_ROOT / "deploy" / "guard"
LOCAL_GATE = config.REPO_ROOT / "src" / "deepme" / "guard" / "gate.py"
LOCAL_SERVER = config.REPO_ROOT / "src" / "deepme" / "guard" / "server.py"


def setup() -> str:
    """Sync the guard uv project (fastapi/uvicorn/httpx) on the server."""
    remote.run(f"mkdir -p {GUARD_DIR}")
    remote.write_remote_file(
        f"{GUARD_DIR}/pyproject.toml", (LOCAL_PROJECT / "pyproject.toml").read_text()
    )
    script = "\n".join(
        [
            "set -euo pipefail",
            f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH",
            f"cd {GUARD_DIR}",
            f"uv sync -p {config.ENGINE_PYTHON} 2>&1 | tail -2",
        ]
    )
    return remote.run_script(script, timeout=600).stdout


def start(spec: RunSpec) -> None:
    """Push code + policy and (re)start the guard in front of spec's API."""
    from . import engines as engines_mod

    for name, path in (
        ("gate.py", LOCAL_GATE),
        ("adapters.py", config.REPO_ROOT / "src" / "deepme" / "guard" / "adapters.py"),
        ("server.py", LOCAL_SERVER),
        ("policy.toml", LOCAL_PROJECT / "policy.toml"),
    ):
        remote.write_remote_file(f"{GUARD_DIR}/{name}", path.read_text())
    # Capability matrix flows from the central engine registry — the guard
    # only enables operations the upstream engine declares.
    features = ",".join(engines_mod.get(spec.engine).features.names())
    unit = f"""\
[Unit]
Description=deepme output-guard proxy (upstream: {spec.profile})
After=network-online.target

[Service]
User={config.SSH_USER}
Environment=GUARD_UPSTREAM=http://127.0.0.1:{spec.port}
Environment=GUARD_POLICY={GUARD_DIR}/policy.toml
Environment=GUARD_ENGINE={spec.engine}
Environment=GUARD_FEATURES={features}
WorkingDirectory={GUARD_DIR}
ExecStart={GUARD_DIR}/.venv/bin/uvicorn server:app --host 0.0.0.0 --port {GUARD_PORT}
Restart=on-failure
RestartSec=5
"""
    remote.write_remote_file(f"/etc/systemd/system/{UNIT}.service", unit, sudo=True)
    remote.run(
        f"sudo systemctl daemon-reload && sudo systemctl reset-failed {UNIT} "
        f"2>/dev/null; sudo systemctl restart {UNIT}"
    )


def stop() -> None:
    remote.run(f"sudo systemctl stop {UNIT}", check=False)


def status() -> str:
    r = remote.run(f"systemctl is-active {UNIT}", check=False)
    return r.stdout.strip() or "unknown"


def health() -> tuple[bool, str]:
    """HTTP reachability of the guard itself (proxied /v1/models)."""
    import httpx

    try:
        r = httpx.get(
            f"{api_base()}/v1/models",
            headers={"Authorization": f"Bearer {config.api_key()}"},
            timeout=5,
        )
        return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as exc:
        return False, type(exc).__name__


def logs(lines: int = 60) -> str:
    r = remote.run(f"sudo journalctl -u {UNIT} -n {lines} --no-pager -o cat", check=False)
    return r.stdout


def api_base() -> str:
    return f"http://{config.SSH_HOST}:{GUARD_PORT}"
