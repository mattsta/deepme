"""Generation client for media models (video today, image next).

Text models answer over /v1/chat/completions and stream tokens (see client.py).
Media models take minutes per output and return a JOB, so the shape is
different: submit, poll to a terminal status, then fetch bytes. This module
hides that difference behind one call, and always lands the result as a real
file on disk — a generation you cannot open is not a result.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from . import config
from .profiles import RunSpec

# Job states the server will not move out of on its own.
TERMINAL = frozenset({"completed", "failed", "cancelled", "error"})

DEFAULT_OUT_DIR = config.STATE_DIR / "generated"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key()}"}


# Condition media types, by extension. The engine accepts image / video /
# video_audio / audio; `video_audio` means "use this video's own soundtrack
# as the audio reference too".
_EXT_TYPE = {
    **dict.fromkeys((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"), "image"),
    **dict.fromkeys((".mp4", ".mov", ".webm", ".mkv", ".avi"), "video"),
    **dict.fromkeys((".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"), "audio"),
}
_MIME = {"image": "image/png", "video": "video/mp4", "audio": "audio/wav"}


def media_type_of(src: str) -> str:
    ext = Path(src.split("?", 1)[0]).suffix.lower()
    kind = _EXT_TYPE.get(ext)
    if not kind:
        raise SystemExit(
            f"cannot tell what kind of media {src!r} is — supported extensions: {sorted(_EXT_TYPE)}"
        )
    return kind


def as_uri(src: str) -> str:
    """A condition URI the server can read.

    http(s) URLs and file:// pass through; a LOCAL path is inlined as a
    base64 data: URI, because the server is a different machine and has no
    access to your filesystem.
    """
    if src.startswith(("http://", "https://", "data:", "file://")):
        return src
    path = Path(src).expanduser()
    if not path.is_file():
        raise SystemExit(f"no such file: {path}")
    import base64

    payload = base64.b64encode(path.read_bytes()).decode()
    return f"data:{_MIME[media_type_of(src)]};base64,{payload}"


def keyframe_conditions(first: str | None, last: str | None) -> list[dict[str, Any]]:
    """fl2va conditioning: an image pinned to the first and/or last frame.

    The engine accepts exactly the signatures (0,), (-1,) and (0, -1) — a
    keyframe in the middle is not a supported mode.
    """
    out = []
    for src, index in ((first, 0), (last, -1)):
        if not src:
            continue
        kind = media_type_of(src)
        if kind != "image":
            raise SystemExit(f"keyframes must be images, got {kind} for {src!r}")
        out.append({"type": "image", "uri": as_uri(src), "role": "keyframe", "frame_index": index})
    return out


def reference_conditions(refs: list[str]) -> list[dict[str, Any]]:
    """ref2va conditioning: style/subject/motion/audio guidance, NOT exact
    frames. A video reference contributes its soundtrack as well."""
    return [
        {
            "type": "video_audio" if media_type_of(r) == "video" else media_type_of(r),
            "uri": as_uri(r),
            "role": "reference",
        }
        for r in refs
    ]


def set_path(body: dict[str, Any], dotted: str, value: Any) -> None:
    """Assign into a nested request body: set_path(b, "target.short_edge", 768).

    Media APIs group related fields into objects, so a flat key=value override
    could not reach half the request.
    """
    keys = dotted.split(".")
    node = body
    for k in keys[:-1]:
        child = node.get(k)
        if not isinstance(child, dict):
            child = {}
            node[k] = child
        node = child
    node[keys[-1]] = value


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """Merge `over` into `base`, recursing into nested objects.

    A plain update would replace a whole object, so overriding one field of
    `target` would silently drop its siblings.
    """
    out = dict(base)
    for k, v in over.items():
        cur = out.get(k)
        out[k] = deep_merge(cur, v) if isinstance(cur, dict) and isinstance(v, dict) else v
    return out


@dataclass(frozen=True)
class Tick:
    """One observation while a generation runs, for progress display.

    elapsed_s is the JOB's age whenever the server reports created_at — not
    this process's polling time — so a runner that re-attaches shows how
    long the work has really been going instead of counting up from zero.
    """

    elapsed_s: float
    status: str
    stage: Any | None = None  # serve.StageProgress, when the engine reports one
    job_id: str = ""  # the server's effective id (post-handover)
    queue_position: int | None = None


# job id -> unit name. An owner never changes, so each job is resolved once;
# without the cache every progress tick would cost a capacity round-trip.
_OWNER_UNITS: dict[str, str] = {}
# unit -> (monotonic, stage). The journal probe is an SSH round-trip; doing
# it on every 5s poll tick multiplied ssh sessions by every in-flight item.
_STAGE_CACHE: dict[str, tuple[float, Any]] = {}
_STAGE_PROBE_EVERY_S = 20.0


def _owner_unit(spec: RunSpec, media_id: str) -> str | None:
    """Which replica's journal holds this job's progress bar.

    Behind a balancer the job runs on WHICHEVER replica was free — reading
    replica 0's journal (the old behaviour) reported nothing when replica 0
    was idle and, worse, replica 0's bar when it was busy with someone
    else's work. The capacity signal already names each running job's
    backend, so the owner is one cheap lookup away.
    """
    cached = _OWNER_UNITS.get(media_id)
    if cached:
        return cached
    r = httpx.get(f"{spec.api_base}/v1/capacity", headers=_headers(), timeout=5)
    r.raise_for_status()
    data = dict(r.json())
    # The gateway aggregates fleets; a balancer answers flat. Same job shape.
    fleets = data.get("fleets") or [data]
    for fleet in fleets:
        if fleet.get("profile") not in (None, spec.profile):
            continue
        for job in fleet.get("running", []):
            if job.get("job_id") != media_id or not job.get("backend"):
                continue
            port = int(str(job["backend"]).rsplit(":", 1)[-1])
            unit = _unit_for_port(spec, port)
            if unit:
                _OWNER_UNITS[media_id] = unit
                if len(_OWNER_UNITS) > 500:
                    _OWNER_UNITS.pop(next(iter(_OWNER_UNITS)))
                return unit
    return None


def _unit_for_port(spec: RunSpec, port: int) -> str | None:
    """Replica port -> unit name, from the declared registry."""
    from . import ports

    declared = ports.load().replicas.get(spec.profile, [])
    if port not in declared:
        return None
    index = declared.index(port)
    base = f"deepme-llm-{spec.profile}"
    return base if index == 0 else f"{base}-r{index}"


def probe_stage(spec: RunSpec, elapsed_s: float, media_id: str = "") -> Any | None:
    """Best-effort stage progress; never let observability break a generation."""
    from . import serve

    try:
        if media_id:
            try:
                unit = _owner_unit(spec, media_id)
            except Exception:
                # No capacity endpoint at all: a --direct run against the
                # replica itself, whose own journal IS the job's journal.
                unit = spec.unit
            else:
                if unit is None:
                    # A front end answered but the job is not running on any
                    # replica (still queued, or between hops). No bar exists;
                    # showing replica 0's would be someone else's progress.
                    return None
        else:
            unit = spec.unit
        # Throttled: the journal read is a REMOTE round-trip, and per-tick
        # probing multiplied ssh sessions by every in-flight batch item.
        cached = _STAGE_CACHE.get(unit)
        if cached and time.monotonic() - cached[0] < _STAGE_PROBE_EVERY_S:
            return cached[1]
        stage = serve.stage_progress(spec, elapsed_s, unit=unit)
        _STAGE_CACHE[unit] = (time.monotonic(), stage)
        return stage
    except Exception:
        return None


@dataclass
class MediaResult:
    """One generation, start to saved file."""

    media_id: str
    status: str
    total_s: float  # wall clock, submit -> bytes on disk
    path: Path | None  # local file, None if the job failed
    size_bytes: int
    inference_s: float | None = None  # server-reported compute time
    peak_memory_mb: float | None = None
    error: dict[str, Any] | None = None
    server_paths: tuple[str, ...] = ()  # where the server kept its copies

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.path is not None


def _submit(spec: RunSpec, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    # A submission returns in milliseconds; `timeout or 300` because the
    # caller's 0 means "no overall deadline", not "no patience for HTTP".
    r = httpx.post(
        f"{spec.api_base}/v1/videos",
        json=body,
        headers=_headers(),
        timeout=timeout or 300,
    )
    if r.status_code >= 400:
        raise SystemExit(f"generation rejected ({r.status_code}): {r.text[:500]}")
    return dict(r.json())


class TicketLostError(RuntimeError):
    """The queue no longer knows this ticket — it died with a balancer.

    A ticket lives in balancer memory. A balancer restart (deploy, crash)
    destroys waiting tickets, and polling the dead id 404s forever — which,
    swallowed as a transient error, looked like a generation that never
    starts while the elapsed counter climbed. 2000-item overnight batches
    stalled exactly this way. Nothing was billed (tickets are pre-billable),
    so the caller can and should simply resubmit.
    """


# Consecutive not-found polls before a ticket is declared dead. One 404 can
# be a race (gateway probing during a balancer's 5s systemd restart); half a
# minute of them is a verdict.
_TICKET_LOST_POLLS = 6


class StalledError(RuntimeError):
    """The server is verifiably NOT working on this id, and it is not
    waiting in any queue either.

    This is what replaces blanket deadlines. A 2-hour per-item timeout was
    never derived from anything and wedged a worker slot for 2 hours per
    dead item — at concurrency 4 that is the whole fleet idle while the
    display counts up. Failure is decided from EVIDENCE instead:

      queued ticket  -> exists server-side (a dead one 404s -> resubmitted)
      accepted job   -> appears in the server's running set within minutes
      generating     -> finishes within 3x the fleet's own mean (the same
                        staleness rule the balancer's reaper applies)

    A RuntimeError, deliberately not SystemExit: a batch worker catches
    Exception per item, so one stalled item fails visibly and the other
    1999 continue.
    """


# "Accepted but nowhere to be seen" grace: many multiples of the seconds a
# submit->running handoff actually takes, nowhere near a generation time.
_STALL_QUEUED_S = 300.0
# Overrun rule for work the server confirms is running: identical to the
# balancer reaper's, so client and server agree on what "stuck" means.
_STALL_RUNNING_MULTIPLE = 3.0
_STALL_RUNNING_FLOOR_S = 1800.0
# How often to cross-check the server's running set while status is opaque.
_EVIDENCE_EVERY_S = 60.0


def _server_view(spec: RunSpec, media_id: str) -> tuple[bool, float | None]:
    """(is the server running this id right now, fleet mean job seconds).

    Reads /v1/capacity — the same shape from a balancer (flat) or the
    gateway (fleets list). The status endpoint cannot answer this: the
    engine reports "queued" from submission to completion, so status alone
    can never distinguish generating from abandoned.
    """
    r = httpx.get(f"{spec.api_base}/v1/capacity", headers=_headers(), timeout=10)
    r.raise_for_status()
    data = dict(r.json())
    fleets = data.get("fleets") or [data]
    running, mean = False, None
    for fleet in fleets:
        if fleet.get("profile") not in (None, spec.profile):
            continue
        if fleet.get("mean_job_s"):
            mean = float(fleet["mean_job_s"])
        if any(j.get("job_id") == media_id for j in fleet.get("running", [])):
            running = True
    return running, mean


# Positive absence reads required before verdict 1 (spanning >= this many
# evidence intervals on top of the initial grace window).
_STALL_ABSENT_CONFIRMATIONS = 5


def _poll(
    spec: RunSpec,
    media_id: str,
    *,
    deadline: float,
    every: float,
    started: float,
    on_tick: Callable[[Tick], None] | None = None,
    cancel: Any | None = None,
    on_real_id: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Follow a job to a terminal status.

    Transient transport errors are tolerated (a long generation can outlive a
    proxy's idle timeout); only the overall deadline ends the wait — except a
    ticket that the platform has verifiably forgotten, which raises
    TicketLostError so the caller can resubmit instead of waiting out a
    timeout for work that will never run.

    ONE IDENTITY RULE. The caller polls the id it was handed, but after a
    queue handover the server works under the id it MINTED — the poll bodies
    carry it, and the balancer rewrites ticket polls transparently. Every
    piece of truth here (evidence, stall verdicts, progress attribution, the
    eventual download, and what gets persisted for adoption) keys on that
    harvested `effective` id. Keying on the handed id made 202-admitted
    items invisible to both stall verdicts (infinite hang on a wedged
    engine) and, after a balancer restart killed the ticket, resubmitted
    work that was still running — generated and billed twice.
    """
    last: dict[str, Any] = {}
    lost = 0
    # Evidence-of-life bookkeeping. The engine reports "queued" from
    # submission until completion, so the status endpoint alone can NEVER
    # distinguish generating from abandoned — the server's running set can.
    began = time.monotonic()
    last_check = 0.0
    running_since: float | None = None
    fleet_mean: float | None = None
    # Verdict 1 needs POSITIVE proof of absence: successful capacity reads
    # that lacked the id. Treating "could not read capacity" as absence
    # falsely killed every --direct run (replicas serve no capacity
    # endpoint) and every job that outlived a 5-minute network blip.
    absent_confirmed = 0
    effective = media_id
    server_started: float | None = None
    while time.monotonic() < deadline:
        if cancel is not None and cancel.is_set():
            # Clean interruption: the JOB keeps running server-side and can
            # be adopted later — only this poller stops.
            raise InterruptedError(effective)
        try:
            r = httpx.get(
                f"{spec.api_base}/v1/videos/{media_id}",
                headers=_headers(),
                timeout=30,
            )
            if r.status_code == 404:
                lost += 1
                if lost >= _TICKET_LOST_POLLS:
                    if effective != media_id:
                        # The HANDLE died (ticket gone with a balancer
                        # restart) but the job it named is known — follow
                        # the job. Resubmitting here is the double-billing
                        # defect.
                        media_id = effective
                        lost = 0
                        continue
                    raise TicketLostError(media_id)
            elif r.status_code < 400:
                # Only a real answer clears the streak: 5xx interleaved with
                # 404s (a crash-looping balancer) must not keep a dead id
                # alive forever.
                lost = 0
                last = dict(r.json())
                real = str(last.get("id") or "")
                if real and real != effective and not real.startswith("q_"):
                    effective = real
                    if on_real_id:
                        on_real_id(real)  # persist: adoption needs THIS id
                with contextlib.suppress(ValueError, TypeError):
                    if last.get("created_at"):
                        server_started = float(last["created_at"])
                if last.get("status") in TERMINAL:
                    return last
        except httpx.HTTPError:
            pass
        now = time.monotonic()
        if now - last_check >= _EVIDENCE_EVERY_S:
            last_check = now
            with contextlib.suppress(httpx.HTTPError, ValueError, TypeError):
                running, mean = _server_view(spec, effective)
                fleet_mean = mean or fleet_mean
                if running:
                    running_since = running_since or now
                    absent_confirmed = 0
                else:
                    absent_confirmed += 1
        # Verdict 1 — accepted but ownerless, CONFIRMED absent repeatedly.
        # (Un-handed tickets are exempt: waiting in the queue is a
        # legitimate unbounded state, and a dead ticket 404s above.)
        if (
            running_since is None
            and not effective.startswith("q_")
            and absent_confirmed >= _STALL_ABSENT_CONFIRMATIONS
            and now - began > _STALL_QUEUED_S
        ):
            raise StalledError(
                f"{effective}: accepted {int(now - began)}s ago and the "
                f"server confirms {absent_confirmed}x that no replica is "
                f"running it — it will never complete. Check "
                f"`deepme balancer status -m {spec.profile}`."
            )
        # Verdict 2 — running far beyond what this fleet takes: mirror of
        # the balancer reaper's rule, so client and server agree on 'stuck'.
        if running_since is not None:
            limit = max(
                (fleet_mean or 0.0) * _STALL_RUNNING_MULTIPLE,
                _STALL_RUNNING_FLOOR_S,
            )
            if now - running_since > limit:
                raise StalledError(
                    f"{effective}: running for {int(now - running_since)}s, "
                    f"over {int(limit)}s (3x this fleet's own mean) — the "
                    f"engine has almost certainly wedged. Check "
                    f"`deepme serve logs -m {spec.profile}`."
                )
        if on_tick:
            # The JOB's age when the server told us its birth time — a
            # re-attached runner shows the true clock, not zero.
            elapsed = time.time() - server_started if server_started else now - started
            on_tick(
                Tick(
                    elapsed_s=max(elapsed, 0.0),
                    status=str(last.get("status") or "queued"),
                    stage=probe_stage(spec, elapsed, effective),
                    job_id=effective,
                    queue_position=last.get("queue_position"),
                )
            )
        time.sleep(every)
    raise SystemExit(
        f"generation {effective} did not finish before the timeout "
        f"(last status: {last.get('status', 'unknown')!r}) — the job may still "
        f"be running; check `deepme serve logs -m {spec.profile}`"
    )


def _download(spec: RunSpec, media_id: str, dest: Path, timeout: float) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with (
        httpx.Client(timeout=timeout or 900) as c,
        c.stream(
            "GET",
            f"{spec.api_base}/v1/videos/{media_id}/content",
            headers=_headers(),
        ) as r,
    ):
        if r.status_code >= 400:
            r.read()
            raise SystemExit(f"content fetch failed ({r.status_code}): {r.text[:300]}")
        with dest.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
                written += len(chunk)
    return written


def generate_video(
    spec: RunSpec,
    prompt: str,
    *,
    seconds: int | None = None,
    size: str | None = None,
    num_frames: int | None = None,
    fps: int | None = None,
    steps: int | None = None,
    seed: int | None = None,
    negative_prompt: str | None = None,
    params: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    timeout: float = 0,
    poll_every: float = 5.0,
    on_tick: Callable[[Tick], None] | None = None,
    on_event: Callable[[str], None] | None = None,
    on_submit: Callable[[str], None] | None = None,
    adopt: str | None = None,
    cancel: Any | None = None,
) -> MediaResult:
    """Generate one video and save it locally.

    Only parameters the caller actually set are sent — the model's own
    defaults (resolution, step count, scheduler shift) are tuned per release
    and are better than anything we would invent here.

    Body precedence, weakest first: the profile's [generate] defaults (fields
    a model requires, like MiniMax-H3's `task`), then the named arguments,
    then `params` — the caller's explicit last word.

    `timeout` is an OPTIONAL absolute cap, off (0) by default. Stuck work is
    detected from evidence instead (see StalledError) — a blanket deadline
    both wasted hours on dead items and hid the absence of real detection.

    The job outlives this process: it runs server-side, and the client is
    only a poller. So `on_submit` reports the server's id the moment it is
    known — persist it — and `adopt` picks up an existing id instead of
    submitting anew. Together they make reconnection idempotent: a restarted
    caller re-attaches to (or downloads the finished result of) the job it
    already paid for, instead of quietly generating it twice. If the server
    verifiably no longer knows the adopted id, a fresh submission happens —
    with `on_submit` fired again so the stored id stays current.

    `cancel` (a threading.Event) makes Ctrl-C clean: the poll loop notices
    within one cycle and raises InterruptedError, leaving the job running
    server-side for the next `adopt` to pick up. It DEFAULTS to the
    process-wide interrupts.CANCEL, so every caller — story, showcase,
    plain generate — is interrupt-aware without wiring anything.
    """
    if cancel is None:
        from . import interrupts

        cancel = interrupts.CANCEL
    floor = getattr(spec, "min_steps", 0)
    if steps is not None and floor and steps < floor:
        raise SystemExit(
            f"--steps {steps} is below this model's floor of {floor}: the "
            f"denoise schedule is built as {steps} "
            f"point{'' if steps == 1 else 's'} from sigma 1.0 to 0.0, and "
            f"fewer than {floor} leaves nothing to step between. The server "
            f"reports this as a sigma-schedule error, which names the symptom "
            f"rather than the cause."
        )
    body: dict[str, Any] = {
        **spec.gen_defaults,
        "model": spec.served_name,
        "prompt": prompt,
    }
    optional = {
        "seconds": seconds,
        "size": size,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": steps,
        "seed": seed,
        "negative_prompt": negative_prompt,
    }
    body.update({k: v for k, v in optional.items() if v is not None})
    # Some models describe the output shape as a nested object instead of flat
    # fields (H3's `target`). Duration belongs wherever that model keeps it,
    # or --seconds would be silently ignored.
    target = body.get("target")
    if seconds is not None and isinstance(target, dict):
        body["target"] = {**target, "duration_seconds": seconds}
    body = deep_merge(body, params or {})

    start = time.monotonic()
    deadline = start + timeout if timeout else float("inf")
    if adopt:
        job: dict[str, Any] = {"id": adopt, "status": "queued"}
        media_id = adopt
        if on_event:
            on_event(f"re-attaching to existing job {media_id}")
    else:
        job = _submit(spec, body, timeout)
        media_id = job.get("id") or ""
        if not media_id:
            raise SystemExit(f"server returned no video id: {job}")
        if on_submit:
            on_submit(media_id)
        if on_event:
            on_event(f"submitted job {media_id}")
    # An id can die server-side: a queued ticket with a balancer restart, an
    # adopted job with a replica restart. Nothing completed was lost either
    # way, so the honest recovery is to resubmit — not to poll a dead id
    # forever. Bounded, so a crash-looping server cannot keep one item
    # resubmitting all night.
    resubmits = 0
    while job.get("status") not in TERMINAL:
        try:
            job = _poll(
                spec,
                media_id,
                deadline=deadline,
                every=poll_every,
                started=start,
                on_tick=on_tick,
                cancel=cancel,
                # The server's own id, harvested mid-poll: persisted so a
                # future adoption follows the JOB, not a mortal ticket.
                on_real_id=on_submit,
            )
        except TicketLostError:
            resubmits += 1
            if resubmits > 3:
                raise SystemExit(
                    f"{media_id} was lost {resubmits} times — the platform "
                    f"keeps forgetting it; check "
                    f"`deepme balancer status -m {spec.profile}`"
                ) from None
            if on_event:
                on_event(f"{media_id} no longer exists on the server — resubmitting")
            job = _submit(spec, body, timeout)
            media_id = job.get("id") or media_id
            if on_submit:
                on_submit(media_id)
            continue
        break

    # Download and report under the server's OWN id when one was harvested —
    # the handle we were given may be a ticket that dies with a balancer.
    media_id = str(job.get("id") or media_id)
    paths = tuple(job.get("file_paths") or ([job["file_path"]] if job.get("file_path") else []))
    result = MediaResult(
        media_id=media_id,
        status=str(job.get("status", "unknown")),
        total_s=time.monotonic() - start,
        path=None,
        size_bytes=0,
        inference_s=job.get("inference_time_s"),
        peak_memory_mb=job.get("peak_memory_mb"),
        error=job.get("error"),
        server_paths=paths,
    )
    if result.status != "completed":
        return result

    if on_event:
        on_event("generation complete — downloading")
    dest = (out_dir or DEFAULT_OUT_DIR) / f"{spec.profile}-{media_id}.mp4"
    size_bytes = _download(spec, media_id, dest, timeout)
    result.path, result.size_bytes = dest, size_bytes
    result.total_s = time.monotonic() - start
    return result
