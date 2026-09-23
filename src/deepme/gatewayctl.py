"""Deploy and control the gateway (see gateway/server.py).

The routing table is built from what is ACTUALLY deployed — running units and
their profiles' declared tasks — rather than from a hand-maintained list, so
changing the fleet mix cannot leave the gateway pointing at something that is
no longer there.
"""

from __future__ import annotations

import json

from . import balancerctl, config, profiles, remote, resources

UNIT = "deepme-gateway"
GATEWAY_DIR = f"{config.REMOTE_RUNTIME_DIR}/gateway"


def _gateway_port() -> int:
    """From the declared registry, never a constant.

    A hardcoded 8080 next to `[services] gateway = 8080` in ports.toml was
    two sources of one truth: edit the TOML and the catalog reports one
    port while the unit binds another — the exact silent drift the registry
    exists to prevent.
    """
    from . import ports

    return int(ports.load().services.get("gateway", 8080))


GATEWAY_PORT = _gateway_port()
LOCAL = config.REPO_ROOT / "src" / "deepme" / "gateway"
LOCAL_PROJECT = config.REPO_ROOT / "deploy" / "guard"


def setup() -> str:
    remote.run(f"mkdir -p {GATEWAY_DIR}")
    remote.write_remote_file(
        f"{GATEWAY_DIR}/pyproject.toml", (LOCAL_PROJECT / "pyproject.toml").read_text()
    )
    script = "\n".join(
        [
            "set -euo pipefail",
            f"export PATH={config.REMOTE_HOME}/.local/bin:$PATH",
            f"cd {GATEWAY_DIR}",
            f"uv sync -p {config.ENGINE_PYTHON} 2>&1 | tail -2",
        ]
    )
    return remote.run_script(script, timeout=600).stdout


def public_model(spec) -> str:
    """The name a caller uses.

    Partitions of one model must present as ONE model with different tasks —
    a caller asks for MiniMax-H3 and a task, not for the checkpoint that
    happens to implement it.
    """
    return spec.repo.rsplit("/", 1)[-1].lower()


def catalog_json() -> str:
    """The service catalog, frozen into the gateway so it can serve it.

    Internal services read this instead of maintaining their own map of what
    is deployed where. Refreshed whenever the gateway is (re)started, which
    is required after any fleet change anyway.
    """
    from . import discovery

    return json.dumps(discovery.catalog())


def discover_routes() -> list[dict]:
    """Routing entries for every profile that currently has replicas running.

    A profile with no running unit is deliberately absent: routing to it would
    produce a connection error where a clear "not deployed" refusal belongs.
    """
    claims = resources.current_claims()
    live = {c.profile for c in claims if c.profile}
    out = []
    for name in sorted(live):
        try:
            spec = profiles.load(name)
        except (Exception, SystemExit):
            continue
        if not spec.tasks:
            continue
        out.append(
            {
                "model": public_model(spec),
                "profile": spec.profile,
                "tasks": list(spec.tasks),
                # Each model fleet has its own balancer; the gateway fans out to
                # them rather than to replicas directly.
                "url": f"http://127.0.0.1:{balancerctl.port_for(spec.profile)}",
            }
        )
    return out


def start(routes: list[dict] | None = None) -> str:
    entries = routes if routes is not None else discover_routes()
    if not entries:
        raise SystemExit(
            "no deployed profile declares tasks — deploy a fleet first "
            "(`deepme fleet apply production`)"
        )
    for name in ("route.py", "server.py"):
        remote.write_remote_file(f"{GATEWAY_DIR}/{name}", (LOCAL / name).read_text())
    # Pricing is frozen in at start, exactly like routes: the derivation
    # (economics.toml x measured occupancy) happens HERE, where the repo is,
    # and the gateway only applies the live surge tier to a static table.
    from . import economics

    pricing = economics.pricing_json(economics.derive(economics.load()))
    unit = f"""\
[Unit]
Description=deepme gateway (model/task -> fleet)
After=network-online.target

[Service]
User={config.SSH_USER}
Environment=GATEWAY_ROUTES={json.dumps(json.dumps(entries))}
Environment=GATEWAY_CATALOG={json.dumps(catalog_json())}
Environment=GATEWAY_PRICING={json.dumps(json.dumps(pricing))}
Environment=GATEWAY_API_KEY={config.api_key()}
WorkingDirectory={GATEWAY_DIR}
ExecStart={GATEWAY_DIR}/.venv/bin/uvicorn server:app --host 0.0.0.0 --port {GATEWAY_PORT}
Restart=on-failure
RestartSec=5
"""
    remote.write_remote_file(f"/etc/systemd/system/{UNIT}.service", unit, sudo=True)
    remote.run(
        f"sudo systemctl daemon-reload && sudo systemctl reset-failed {UNIT} "
        f"2>/dev/null; sudo systemctl restart {UNIT}"
    )
    return "\n".join(
        f"  {e['model']} {sorted(e['tasks'])} -> {e['profile']} @ {e['url']}" for e in entries
    )


def stop() -> None:
    remote.run(f"sudo systemctl stop {UNIT}", check=False)


def status() -> str:
    r = remote.run(f"systemctl is-active {UNIT}", check=False)
    return r.stdout.strip() or r.stderr.strip() or "unknown"


def api_base() -> str:
    return f"http://{config.SSH_HOST}:{GATEWAY_PORT}"


def routes() -> dict:
    import httpx

    r = httpx.get(f"{api_base()}/gateway/routes", timeout=10)
    r.raise_for_status()
    return dict(r.json())


def capacity() -> dict:
    """Where work can go right now (see gateway/server.py:/v1/capacity)."""
    import httpx

    r = httpx.get(f"{api_base()}/v1/capacity", timeout=10)
    r.raise_for_status()
    return dict(r.json())


def repoint(routes_list: list[dict]) -> bool:
    """Swap the live gateway's routes/catalog/pricing in place — no restart.

    False when the gateway is down or predates /gateway/reload; the caller
    falls back to a full start. Preferred because a restart severs every
    in-flight SSE stream and download.
    """
    import httpx

    from . import economics

    try:
        payload = {
            "routes": routes_list,
            "catalog": json.loads(catalog_json()),
            "pricing": economics.pricing_json(economics.derive(economics.load())),
        }
        r = httpx.post(f"{api_base()}/gateway/reload", json=payload, timeout=15)
        return r.status_code == 200
    except Exception:
        return False


def pricing() -> dict:
    """The price table the gateway is currently serving."""
    import httpx

    r = httpx.get(f"{api_base()}/v1/pricing", timeout=10)
    r.raise_for_status()
    return dict(r.json())
