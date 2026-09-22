"""Build engine launch commands and manage inference services via systemd.

Each model profile gets its own unit (deepme-llm-<profile>) pinned to its
GPU set and port, so multiple models can coexist when their GPU sets don't
overlap. Every `start` rewrites the unit from the resolved RunSpec —
iterating on flags is just editing the profile (or CLI overrides) and
re-running `deepme serve start`.
"""

from __future__ import annotations

import shlex
import time

import httpx

from . import config, remote
from .profiles import RunSpec

UNIT_PREFIX = "deepme-llm"
LEGACY_UNITS = ("deepme-llm",)  # pre-profile single unit; stopped on conflict


def _resolve_model_file(spec: RunSpec) -> RunSpec:
    """For file-based formats (gguf), resolve the concrete weight file on the
    server (variant name, or the single .gguf present) into spec.model_file."""
    if spec.format != "gguf":
        return spec
    import dataclasses

    pattern = spec.variant or "*.gguf"
    r = remote.run(f"ls {spec.model_path}/{pattern} 2>/dev/null | head -1", check=False)
    path = r.stdout.strip()
    if not path:
        raise SystemExit(
            f"no gguf file matching {pattern!r} under {spec.model_path} — "
            f"is the download complete?"
        )
    return dataclasses.replace(spec, model_file=path)


def build_command(spec: RunSpec) -> list[str]:
    """Delegate to the engine registry; append profile + CLI args.

    Engine-specific logic lives in engines.py EngineDefs — this layer only
    validates and composes.
    """
    from . import engines as engines_mod

    eng = engines_mod.get(spec.engine)
    err = eng.validate_spec(spec)
    if err:
        raise SystemExit(f"invalid spec for engine {spec.engine}: {err}")
    cmd = eng.build_command(spec, config.api_key())
    cmd += list(spec.engine_args)
    cmd += list(spec.extra_args)
    return cmd


def _running_units() -> list[str]:
    r = remote.run(
        f"systemctl list-units --no-legend --plain '{UNIT_PREFIX}*' "
        "| awk '{print $1}' | sed 's/\\.service//'",
        check=False,
    )
    return [u for u in r.stdout.split() if u]


def _stop_conflicting(spec: RunSpec) -> list[str]:
    """Stop units whose GPU sets (or port) overlap the one we're starting.

    Unit files carry an X-DEEPME-GPUS comment so overlap is decided from the
    server's actual state, not local assumptions. Legacy units count as
    all-GPUs.
    """
    stopped = []
    for unit in _running_units():
        if unit == spec.unit or unit in LEGACY_UNITS:
            overlap = True
        else:
            r = remote.run(
                f"grep -oP '(?<=X-DEEPME-GPUS=)[0-9,]+' "
                f"/etc/systemd/system/{unit}.service 2>/dev/null",
                check=False,
            )
            gpus = {int(g) for g in r.stdout.strip().split(",") if g.strip().isdigit()}
            overlap = bool(gpus & set(spec.gpus)) if gpus else True
        if overlap:
            remote.run(f"sudo systemctl stop {unit} 2>/dev/null || true", check=False)
            stopped.append(unit)
    return stopped


def start(spec: RunSpec) -> list[str]:
    """Deploy a RunSpec; returns any units stopped to free its GPUs/port."""
    err = spec.fit_check()
    if err:
        raise SystemExit(f"fit check failed for {spec.profile}: {err}")
    spec = _resolve_model_file(spec)
    stopped = _stop_conflicting(spec)
    exec_line = " ".join(shlex.quote(a) for a in build_command(spec))
    gpus_csv = ",".join(str(g) for g in spec.gpus)
    unit = f"""\
# X-DEEPME-GPUS={gpus_csv}
# X-DEEPME-PROFILE={spec.profile}
[Unit]
Description=deepme LLM inference ({spec.profile}: {spec.engine}, {spec.served_name})
After=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
User={config.SSH_USER}
Environment=HF_HOME={config.REMOTE_HOME}/.cache/huggingface
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES={gpus_csv}
Environment=CUDA_HOME={config.venv_cuda_home(spec.engine)}
Environment=PATH={config.engine_venv(spec.engine)}/bin:{config.venv_cuda_home(spec.engine)}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart={exec_line}
Restart=on-failure
RestartSec=10
LimitNOFILE=1048576
KillMode=mixed
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
"""
    remote.write_remote_file(
        f"/etc/systemd/system/{spec.unit}.service", unit, sudo=True
    )
    remote.run(
        f"sudo systemctl daemon-reload && sudo systemctl reset-failed {spec.unit} "
        f"2>/dev/null; sudo systemctl restart {spec.unit}"
    )
    return stopped


def stop(spec: RunSpec) -> None:
    remote.run(f"sudo systemctl stop {spec.unit}", check=False)


def restart(spec: RunSpec) -> None:
    remote.run(
        f"sudo systemctl reset-failed {spec.unit} 2>/dev/null;"
        f" sudo systemctl restart {spec.unit}"
    )


def unit_state(spec: RunSpec) -> str:
    r = remote.run(f"systemctl is-active {spec.unit}", check=False)
    return r.stdout.strip() or r.stderr.strip() or "unknown"


def health(spec: RunSpec) -> tuple[bool, str]:
    try:
        r = httpx.get(
            f"{spec.api_base}/health",
            timeout=5,
            headers={"Authorization": f"Bearer {config.api_key()}"},
        )
        return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as exc:  # noqa: BLE001 - report any probe failure as text
        return False, type(exc).__name__


def wait_healthy(
    spec: RunSpec, timeout_s: float = 1800, poll_s: float = 10
) -> tuple[bool, str]:
    """Wait for the server to come up; on failure, include a log diagnosis."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ok, _ = health(spec)
        if ok:
            return True, "healthy"
        state = unit_state(spec)
        if state in ("failed", "inactive", "dead"):
            return False, f"service is {state}\n" + diagnose(spec)
        time.sleep(poll_s)
    return False, f"timed out after {timeout_s}s\n" + diagnose(spec)


def diagnose(spec: RunSpec, lines: int = 3000) -> str:
    """Restart count + the log window before the last crash (noise filtered)."""
    script = f"""\
echo "restarts=$(systemctl show -p NRestarts --value {spec.unit}) state=$(systemctl is-active {spec.unit})"
sudo journalctl -u {spec.unit} -n {lines} --no-pager -o cat \\
  | grep -v 'Ignore import error' \\
  | awk '/Main process exited|Failed with result/ {{n=NR}} {{l[NR]=$0}} END {{ if (n==0) n=NR; s=n-45; if (s<1) s=1; for (i=s; i<=n; i++) print l[i] }}'
"""
    r = remote.run_script(script, check=False)
    return r.stdout.strip() or "(no logs)"


def tail_logs(spec: RunSpec, lines: int = 100, follow: bool = False) -> None:
    flags = f"-n {lines}" + (" -f" if follow else "")
    remote.run(
        f"sudo journalctl -u {spec.unit} {flags} --no-pager",
        stream=True,
        timeout=None,
        check=False,
    )


def fleet_status() -> str:
    """One line per deepme unit on the server: profile, state, gpus."""
    script = f"""\
for f in /etc/systemd/system/{UNIT_PREFIX}*.service; do
  [ -e "$f" ] || continue
  u=$(basename "$f" .service)
  gpus=$(grep -oP '(?<=X-DEEPME-GPUS=)[0-9,]+' "$f" 2>/dev/null || echo '?')
  echo "$u state=$(systemctl is-active $u) gpus=$gpus"
done
"""
    return remote.run_script(script, check=False).stdout.strip() or "(no units)"
