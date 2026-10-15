"""Build engine launch commands and manage inference services via systemd.

Each model profile gets its own unit (deepme-llm-<profile>) pinned to its
GPU set and port, so multiple models can coexist when their GPU sets don't
overlap. Every `start` rewrites the unit from the resolved RunSpec —
iterating on flags is just editing the profile (or CLI overrides) and
re-running `deepme serve start`.
"""

from __future__ import annotations

import re
import shlex
import time
from dataclasses import dataclass

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

    # Variant keys strip the multi-part suffix and may include a subdir, so
    # the concrete file is `{stem}*.gguf` (part 00001 for multi-part). Match
    # the same way download.py fetched it; pick the first part (sorted).
    if spec.variant:
        stem = spec.variant[: -len(".gguf")]
        pattern = f"{stem}*.gguf"
    else:
        pattern = "*.gguf"
    r = remote.run(
        f"ls {shlex.quote(spec.model_path)}/{pattern} 2>/dev/null | sort | head -1",
        check=False,
    )
    path = r.stdout.strip()
    if not path:
        raise SystemExit(
            f"no gguf file matching {pattern!r} under {spec.model_path} — "
            f"is the download complete? (deepme download status -m {spec.profile})"
        )
    return dataclasses.replace(spec, model_file=path)


def replica_specs(
    spec: RunSpec,
    count: int,
    *,
    reserved: list[RunSpec] | None = None,
    releasing: set[str] | frozenset[str] = frozenset(),
) -> list[RunSpec]:
    """Lay out `count` copies of a model across the GPUs that are free.

    Each replica gets its own GPU set, port and unit, so they coexist rather
    than displacing one another. GPUs come from the allocator (socket-local
    where it can), NOT from the profile's hand-written list — that list only
    says how many GPUs one copy needs.

    `reserved` carries deployments that are PLANNED but not yet running, so a
    mixed fleet can be laid out in one pass: the allocator only knows what is
    currently claimed, and without this every member of a fleet would be
    handed the same free GPUs.
    """
    import dataclasses

    from . import ports, resources

    registry = ports.load()
    if count < 1:
        raise SystemExit("replica count must be at least 1")
    per_replica = len(spec.gpus)
    if not per_replica:
        raise SystemExit(f"profile {spec.profile!r} declares no GPUs to replicate")

    cap = resources.probe_capacity()
    claims = resources.current_claims()
    # This profile's own replicas are being replaced, so their GPUs are free
    # for the new layout. `releasing` extends that to the OTHER profiles a
    # fleet change will also stop — without it, planning a mix while the
    # previous mix is still running reports no free GPUs, and a dry run would
    # require destroying the fleet it is only trying to describe.
    freeing = {spec.profile} | set(releasing)
    mine = {c.unit for c in claims if c.profile in freeing}
    avail = resources.availability(cap, claims, exclude=mine)

    taken = {g for s in (reserved or []) for g in s.gpus}
    free = [g for g in avail.free_gpus if g not in taken]
    if len(free) < per_replica * count:
        raise SystemExit(
            f"{count} replicas need {per_replica * count} GPUs, "
            f"{len(free)} free ({list(free)}). Free some, or use "
            f"-n {len(free) // per_replica}."
        )
    out = []
    for i in range(count):
        here = resources.pick_gpus(
            resources.Availability(tuple(free), (), avail.free_ram_gib), cap, per_replica
        )
        for g in here:
            free.remove(g)
        # With a shared weight store, only the first replica pays for the
        # weights; the rest map the same pages and need private overhead
        # only. Charging every replica the full amount is what would refuse
        # replicas 3 and 4 despite the RAM being there.
        # The CAP stays at the full figure for every replica — loading
        # materialises each component before offloading it, so the peak is
        # unchanged by sharing and a small cap is an OOM kill mid-load. Only
        # the allocator's CHARGE drops.
        claim = spec.extra_replica_ram_gib if i else 0.0
        out.append(
            dataclasses.replace(
                spec,
                replica=i,
                gpus=here,
                port=registry.replica_port(spec.profile, i),
                claim_ram_gib=claim,
            )
        )
    return out


# Replica ports come from the declared registry (deploy/ports.toml), not from
# arithmetic on a base port. Derivation had already produced a live collision:
# a diffusion server binds its API port AND the next one, so a stride that
# looks safe can hand one profile a port its neighbour silently consumes.


def stop_profile(spec: RunSpec) -> list[str]:
    """Stop every replica of a profile, so a re-layout starts from nothing."""
    from . import resources

    units = [c.unit for c in resources.current_claims() if c.profile == spec.profile]
    for unit in units:
        remote.run(f"sudo systemctl stop {unit} 2>/dev/null || true", check=False)
    if units:
        _wait_gpus_released(units)
    return units


def stop_units(units: list[str]) -> None:
    """Stop SPECIFIC units and wait for their GPUs to release.

    The delta-aware path: a fleet switch removes individual replicas, and
    stop_profile would take its healthy siblings down with them.
    """
    for unit in units:
        remote.run(f"sudo systemctl stop {unit} 2>/dev/null || true", check=False)
    if units:
        _wait_gpus_released(units)


def work_dir(spec: RunSpec) -> str:
    """Writable scratch directory for one deployment.

    systemd starts units in `/`, which is not writable. Engines that stage
    uploads or outputs relative to the CWD (sglang's diffusion server creates
    `inputs/`) fail with EACCES there, at REQUEST time rather than at launch —
    the server looks perfectly healthy until the first generation.
    """
    return f"{config.REMOTE_HOME}/deepme-work/{spec.profile}"


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
    if eng.build_command is None:
        raise SystemExit(f"engine {spec.engine} declares no launch builder")
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
    """Stop only the units that genuinely conflict with this placement.

    Conflict = GPU overlap or port collision (see resources.conflicts_for).
    A CPU-mode model never displaces a GPU model — colocation is the point.
    Legacy (pre-annotation) units are treated as holding everything.
    """
    from . import resources

    claims = resources.current_claims()
    known = {c.unit for c in claims}
    stopped = list(resources.conflicts_for(spec, claims))
    # Units without claim annotations (legacy) can't be reasoned about — stop
    # them if they're running and we're taking GPUs.
    for unit in _running_units():
        if unit in known or unit in stopped:
            continue
        if unit in LEGACY_UNITS or spec.mode == "gpu":
            stopped.append(unit)
    for unit in stopped:
        remote.run(f"sudo systemctl stop {unit} 2>/dev/null || true", check=False)
    if stopped:
        _wait_gpus_released(stopped)
    return stopped


# Teardown is slower than it looks with the shared weight store: each rank
# has cudaHostRegister'd tens of GiB, and un-registering page-locked memory
# costs time proportional to its size. 90s was enough before sharing and is
# not now — a timeout here aborts a scale-up midway, leaving one replica up.
GPU_RELEASE_TIMEOUT_S = 300


def _wait_gpus_released(units: list[str], timeout_s: int = GPU_RELEASE_TIMEOUT_S) -> None:
    """After stopping specific units, wait until THOSE units' processes release
    GPU memory. Scoped to the stopped units by cgroup so a coexisting model on
    other GPUs is never waited on (that would spin the full timeout)."""
    if not units:
        return
    # Match the FULL unit name including .service. "deepme-llm-h3-video" is a
    # prefix of "deepme-llm-h3-video-r1", so a bare-name match sees a sibling
    # replica that is still (correctly) running as the stopped unit's process
    # and waits out the whole timeout.
    alt = "|".join(re.escape(f"{u}.service") for u in units)
    script = f"""\
for i in $(seq {timeout_s}); do
  busy=0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do
    grep -qE '{alt}' /proc/$p/cgroup 2>/dev/null && busy=1
  done
  [ "$busy" = 0 ] && echo RELEASED && exit 0
  sleep 1
done
echo "TIMEOUT: processes for stopped units still present after {timeout_s}s"
exit 1
"""
    r = remote.run_script(script, check=False, timeout=timeout_s + 30)
    if "RELEASED" not in r.stdout:
        raise SystemExit("stopped units did not release GPU memory: " + r.stdout.strip())


def start(spec: RunSpec) -> list[str]:
    """Deploy a RunSpec; returns any units stopped to free its resources."""
    err = spec.fit_check()
    if err:
        raise SystemExit(f"fit check failed for {spec.profile}: {err}")
    from . import resources

    # Placement check against the live allocation (GPU/CPU/RAM/port).
    cap = resources.probe_capacity()
    claims = resources.current_claims()
    perr = resources.check_placement(spec, cap, claims)
    if perr:
        raise SystemExit(f"placement failed for {spec.profile}: {perr}")
    spec = _resolve_model_file(spec)
    stopped = _stop_conflicting(spec)

    # CPU pinning: pack within one NUMA node (cross-socket traffic dominates
    # CPU inference). GPU-mode units may also pin if they declared cpus.
    avail = resources.availability(cap, claims, exclude=set(stopped))
    chosen_cpus = resources.pick_cpus(avail, cap, spec.cpus) if spec.cpus else ()
    unit = render_unit(spec, cpus_csv=resources.format_cpu_list(chosen_cpus))
    # WorkingDirectory must exist or systemd refuses to start the unit.
    remote.run(f"mkdir -p {shlex.quote(work_dir(spec))}")
    # Deliberately NO [Install]/enable: models are operator-started, never
    # auto-launched on boot (a 150GB model should not surprise-start).
    remote.write_remote_file(f"/etc/systemd/system/{spec.unit}.service", unit, sudo=True)
    remote.run(
        f"sudo systemctl daemon-reload && sudo systemctl reset-failed {spec.unit} "
        f"2>/dev/null; sudo systemctl restart {spec.unit}"
    )
    return stopped


def render_unit(spec: RunSpec, *, cpus_csv: str = "") -> str:
    """The systemd unit for one deployment — pure text, no server contact.

    The X-DEEPME-* header comments are the allocator's record of what this
    unit claims; resources.py reads them back to compute live claims.
    """
    from . import engines as engines_mod

    resource_lines = ""
    if cpus_csv:
        resource_lines += f"AllowedCPUs={cpus_csv}\n"
    if spec.ram_gib:
        resource_lines += f"MemoryMax={int(spec.ram_gib)}G\n"

    exec_line = " ".join(shlex.quote(a) for a in build_command(spec))
    # CPU mode claims no GPUs and is hidden from CUDA entirely.
    gpus_csv = ",".join(str(g) for g in spec.gpus) if spec.mode == "gpu" else ""
    # Engine-declared runtime environment (e.g. loader path for venv CUDA libs)
    cuda_home = config.venv_cuda_home(spec.engine)
    extra_env = "".join(
        f"Environment={k}={v.format(cuda=cuda_home)}\n"
        for k, v in engines_mod.get(spec.engine).runtime_env.items()
    )
    # Profile env last so a deployment can override an engine default.
    extra_env += "".join(f'Environment="{k}={v}"\n' for k, v in spec.env)
    return f"""\
# X-DEEPME-GPUS={gpus_csv}
# X-DEEPME-CPUS={cpus_csv}
# X-DEEPME-RAM={spec.accounted_ram_gib}
# X-DEEPME-PORT={spec.port}
# X-DEEPME-PROFILE={spec.profile}
# X-DEEPME-MODE={spec.mode}
[Unit]
Description=deepme LLM inference ({spec.profile}: {spec.engine}, {spec.served_name})
After=network-online.target
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
User={config.SSH_USER}
WorkingDirectory={work_dir(spec)}
Environment=HF_HOME={config.REMOTE_HOME}/.cache/huggingface
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_VISIBLE_DEVICES={gpus_csv}
Environment=CUDA_HOME={config.venv_cuda_home(spec.engine)}
Environment=PATH={config.engine_venv(spec.engine)}/bin:{config.venv_cuda_home(spec.engine)}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
{extra_env}{resource_lines}ExecStart={exec_line}
Restart=on-failure
RestartSec=10
LimitNOFILE=1048576
KillMode=mixed
TimeoutStopSec=60
"""


def stop(spec: RunSpec) -> None:
    remote.run(f"sudo systemctl stop {spec.unit}", check=False)


def restart(spec: RunSpec) -> None:
    remote.run(
        f"sudo systemctl reset-failed {spec.unit} 2>/dev/null; sudo systemctl restart {spec.unit}"
    )


@dataclass(frozen=True)
class StageProgress:
    """A running stage's own progress, scraped from the engine's log.

    Long generations report nothing useful over HTTP — sglang's video job
    endpoint only ever returns progress 0 then 100 — but the engine prints a
    tqdm bar per pipeline stage. That bar is the only real signal there is,
    so we read it rather than showing an 11-minute blank.
    """

    label: str  # e.g. "minimax_h3 denoise" or "MiniMaxH3TextEncodingStage"
    step: int
    total: int  # 0 when the stage reports no step counter
    rate_s: float  # seconds per step (0 if the engine reported it/s)

    @property
    def counted(self) -> bool:
        return self.total > 0

    @property
    def fraction(self) -> float:
        return self.step / self.total if self.total else 0.0

    @property
    def eta_s(self) -> float:
        return max(self.total - self.step, 0) * self.rate_s


# tqdm renders as: "minimax_h3 denoise:  96%|#####| 47/49 [09:02<00:23, 11.75s/it]"
_TQDM = re.compile(
    r"(?P<label>[A-Za-z0-9_][A-Za-z0-9_ .-]*):\s*\d+%\|[^|]*\|\s*"
    r"(?P<step>\d+)/(?P<total>\d+)\s*\[[^<]*<[^,]*,\s*"
    r"(?P<rate>[\d.]+)(?P<unit>s/it|it/s)\]"
)
# Uncounted stages announce themselves: "[MiniMaxH3TextEncodingStage] started..."
_STAGE = re.compile(r"\[(?P<label>[A-Za-z0-9_]*Stage)\] started")


def stage_progress(
    spec: RunSpec, within_s: float, *, unit: str | None = None, timeout: float = 600
) -> StageProgress | None:
    """Latest stage progress from the last `within_s` seconds of the journal.

    Bounded by time so a bar left over from an EARLIER generation is never
    reported as this one's progress.

    `unit` names WHICH replica's journal to read. Behind a balancer the job
    runs on whichever replica was free, and each replica logs only its own
    bar — the default (replica 0) is only right for single-replica and
    --direct runs; callers who know the owner must say so (media._owner_unit).

    `timeout` bounds the SSH round trip. It defaults to remote.run's own 600s,
    which is right for an operator running this once and catastrophic for the
    poll loop, where this is called inside the tick callback on every cycle:
    an unresponsive replica host could park a generation's progress display —
    and the poll loop with it — for ten minutes per tick, uninterruptibly.
    Callers on a clock must pass their own bound (see media._STAGE_PROBE_S).
    """
    window = max(int(within_s) + 5, 10)
    r = remote.run(
        f"journalctl -u {unit or spec.unit} --since '-{window}s' --no-pager "
        "-o cat 2>/dev/null | grep -aE 'it\\]|Stage\\] started' | tail -40",
        check=False,
        timeout=timeout,
    )
    # Whichever appeared LAST is what the pipeline is doing now: a step bar
    # while one is running, otherwise the name of the current stage. The job
    # API cannot answer this — its status is "queued" until it is "completed".
    bar = stage = None
    for bar in _TQDM.finditer(r.stdout):  # noqa: B007 - want the LAST match
        pass
    for stage in _STAGE.finditer(r.stdout):  # noqa: B007 - want the LAST match
        pass
    if bar and (not stage or bar.start() > stage.start()):
        rate = float(bar["rate"])
        return StageProgress(
            label=bar["label"].strip(),
            step=int(bar["step"]),
            total=int(bar["total"]),
            rate_s=rate if bar["unit"] == "s/it" else (1 / rate if rate else 0.0),
        )
    if stage:
        return StageProgress(label=stage["label"], step=0, total=0, rate_s=0.0)
    return None


def unit_state(spec: RunSpec) -> str:
    r = remote.run(f"systemctl is-active {spec.unit}", check=False)
    return r.stdout.strip() or r.stderr.strip() or "unknown"


def health(spec: RunSpec) -> tuple[bool, str]:
    """Is the API serving? Tries /health, then /v1/models — not every engine
    implements /health (llama.cpp's server doesn't), but any OpenAI-compatible
    server answers /v1/models."""
    headers = {"Authorization": f"Bearer {config.api_key()}"}
    last = "no response"
    for path in ("/health", "/v1/models"):
        try:
            r = httpx.get(f"{spec.api_base}{path}", timeout=5, headers=headers)
            if r.status_code == 200:
                return True, f"HTTP 200 ({path})"
            last = f"HTTP {r.status_code} ({path})"
        except Exception as exc:
            last = type(exc).__name__
    return False, last


def wait_healthy(spec: RunSpec, timeout_s: float = 1800, poll_s: float = 10) -> tuple[bool, str]:
    """Wait for the server to come up; on failure, include a log diagnosis."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ok, _ = health(spec)
        if ok:
            return True, "healthy"
        state = unit_state(spec)
        if state in ("failed", "inactive", "dead"):
            return False, f"service is {state}\n" + diagnose(spec)
        # A unit can be "running" and serving perfectly on a port we are not
        # watching. Waiting out a 30-minute timeout on that is indistinguishable
        # from a slow load, so check for it explicitly and fail immediately.
        drifted = serving_port(spec)
        if drifted and drifted != spec.port:
            return False, (
                f"{spec.unit} is serving on port {drifted}, not the {spec.port} "
                f"it was assigned — the engine auto-selected a free port. "
                f"Something else already holds {spec.port} (a diffusion server "
                f"binds its port AND the next one, so replicas must be spaced "
                f"by more than 1)."
            )
        time.sleep(poll_s)
    return False, f"timed out after {timeout_s}s\n" + diagnose(spec)


def serving_port(spec: RunSpec) -> int | None:
    """The port this unit's processes are ACTUALLY listening on.

    Reported state lies in a specific way: systemd says `running`, the engine
    says `fired up and ready`, and both are true — of a port nobody is
    watching. This reads the socket table instead of believing either.
    """
    r = remote.run(
        "sudo ss -lntp 2>/dev/null | sed -E 's/.*:([0-9]+) .*pid=([0-9]+).*/\\1 \\2/'",
        check=False,
    )
    candidates: set[int] = set()
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        port, pid = int(parts[0]), parts[1]
        cg = remote.run(f"cat /proc/{pid}/cgroup 2>/dev/null", check=False).stdout
        if f"{spec.unit}.service" in cg:
            candidates.add(port)
    if not candidates:
        return None
    # A unit holds several sockets — the API port, the rendezvous port, an
    # internal store on 2379, and more. Guessing which is the API (e.g. "the
    # lowest") picks the wrong one; ASK them instead.
    for port in sorted(candidates):
        code = remote.run(
            f"curl -s -o /dev/null -w '%{{http_code}}' -m 3 http://127.0.0.1:{port}/v1/models",
            check=False,
        ).stdout.strip()
        if code == "200":
            return port
    return None


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
