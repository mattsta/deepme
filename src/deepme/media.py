"""Generation client for media models (video today, image next).

Text models answer over /v1/chat/completions and stream tokens (see client.py).
Media models take minutes per output and return a JOB, so the shape is
different: submit, poll to a terminal status, then fetch bytes. This module
hides that difference behind one call, and always lands the result as a real
file on disk — a generation you cannot open is not a result.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from . import config
from .profiles import RunSpec

# Job states the server will not move out of on its own.
TERMINAL = frozenset({"completed", "failed", "cancelled", "error"})

DEFAULT_OUT_DIR = config.STATE_DIR / "generated"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key()}"}


# Per-operation HTTP bounds, INDEPENDENT of any generation deadline. These
# used to be derived from the caller's `timeout` — which is the overall
# deadline (batch defaults to 7200) — so `timeout or 300` evaluated to two
# hours and a wedged gateway pinned a worker thread for that long on a call
# that normally takes 30ms. Worse, the item never reached the poll loop, so
# none of the evidence-based stall detection below ever ran.
_SUBMIT_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
_POLL_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_PROBE_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# A result is hundreds of MB: generous to read, still bounded.
_DOWNLOAD_TIMEOUT = httpx.Timeout(900.0, connect=10.0)
# Wall clock for the WHOLE transfer. httpx timeouts are per-operation, so the
# one above bounds the gap between chunks and nothing else — a server emitting
# one byte every 899s never trips it. 30 minutes is far beyond any honest
# fetch here (clips are single-digit MB) so this only ever fires on a
# transfer that has stopped making progress.
_DOWNLOAD_TOTAL_S = 1800.0


# KNOWN GAP (not a guess — measured shape): every call below is a
# module-level httpx.get/post, so each builds and discards its own
# connection pool. A 2000-item run polling every 5s per in-flight item
# opens sockets by the tens of thousands against one host. Fixing it means
# one thread-safe Client per endpoint, which the test suite's
# function-level httpx monkeypatching cannot intercept — so it lands with
# the transport-level test seam, not before it.


class MediaError(RuntimeError):
    """Base for every failure this client reports.

    A plain exception, never SystemExit: this module is a LIBRARY called
    from batch/story/showcase worker threads. `SystemExit` is a
    BaseException, so it sailed through `except Exception`, aborted the
    pool from a worker thread, and forced every caller to grow its own
    `except (Exception, SystemExit)` compensation — inconsistently, so
    story and showcase never got one. The CLI converts these to exit codes
    at its own boundary, which is the only place that decision belongs.
    """


class RequestInvalid(MediaError):
    """The request cannot be built — a caller/argument error."""


# Engine failures that depend on CONCURRENT LOAD rather than on the request:
# the same body succeeds once the fleet is less busy or less fragmented.
# Matched on the engine's message because that is all it gives us — there is
# no error code in the body.
_TRANSIENT_MARKERS = (
    "out of memory",
    "cuda error: out of memory",
    "no kernel image is available",  # transient only under memory pressure
    "cublas_status_alloc_failed",
    "cudnn_status_alloc_failed",
)


def is_transient(error: Any) -> bool:
    """Would this failure plausibly succeed on a retry?

    A CUDA OOM is the motivating case: a 15s clip failed at allocation
    while four other clips ran, on a replica whose allocator had 6GiB
    stranded in fragmented reserved segments. Nothing about the REQUEST was
    wrong — it needed a less-loaded replica. Recording that as a permanent
    item failure discards work the platform could have completed.

    Deliberately narrow: a validation error, a sigma-schedule floor or a
    bad task must NEVER be retried, because every attempt costs a slot and
    fails identically.
    """
    if error is None:
        return False
    text = (error.get("message", "") if isinstance(error, dict) else str(error)).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


class GenerationRejected(MediaError):
    """The server refused the submission."""


class GenerationTimeout(MediaError):
    """The caller's absolute deadline passed with the job unfinished."""


class DownloadFailed(MediaError):
    """The result exists but its bytes could not be fetched."""


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
        raise RequestInvalid(
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
        raise RequestInvalid(f"no such file: {path}")
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
            raise RequestInvalid(f"keyframes must be images, got {kind} for {src!r}")
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
    # True when this call ATTACHED to work that already existed rather than
    # creating any. The display must not claim to be "starting" something
    # it did not start — that is what made a collect-only run look like it
    # was launching new generations.
    adopted: bool = False


class _BoundedCache:
    """A small id->value cache shared by every batch worker thread.

    Locked, and bounded with OrderedDict rather than a hand-rolled
    len/next(iter)/pop dance. That dance was a read-modify-write across
    three statements: on free-threaded 3.14 two workers crossing the bound
    together computed the same victim key and the loser's `pop` raised
    KeyError — swallowed upstream into a fallback that reported REPLICA
    0's progress bar for a job running elsewhere, which is precisely the
    bug the cache's owner lookup exists to prevent.
    """

    def __init__(self, maxsize: int):
        self._data: OrderedDict[str, Any] = OrderedDict()
        self._max = maxsize
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def discard(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# job id -> unit name. An owner never changes, so each job is resolved once;
# without the cache every progress tick would cost a capacity round-trip.
_OWNER_UNITS = _BoundedCache(500)
# Jobs with no owning replica YET (queued), so the lookup is not repeated on
# every tick. Separate from _OWNER_UNITS because these entries expire and
# those do not: an owner, once known, cannot change.
_OWNER_MISSES = _BoundedCache(500)
_OWNER_MISS_EVERY_S = 15.0
# unit -> (monotonic, stage). The journal probe is an SSH round-trip; doing
# it on every 5s poll tick multiplied ssh sessions by every in-flight item.
_STAGE_CACHE = _BoundedCache(64)
_STAGE_PROBE_EVERY_S = 20.0
# Hard bound on the progress probe's SSH round trip. It runs inside the tick
# callback, so anything slow here stalls the POLL LOOP — and the loop's own
# cancel checks sit outside it, which means an unresponsive host also swallows
# Ctrl-C. Well under the poll interval on purpose: progress is decoration, and
# a decoration must never outrank the work or the operator's ability to stop
# it. Inheriting remote.run's 600s default made one dead host a ten-minute
# freeze per tick.
_STAGE_PROBE_S = 8.0


@dataclass(frozen=True)
class FleetView:
    """One reading of /v1/capacity, with an explicit trust bit.

    Every client-side caller was re-deriving this shape, and the rule that
    matters — when absence counts as EVIDENCE — held in some of them and not
    others. The gateway answers 200 even when a fleet's balancer could not be
    asked: that entry says `reachable: false` and carries no `running` list.
    Reading that silence as "the job is absent" laundered a failed read into
    positive proof of death, and capacity probes time out exactly while a
    finished clip's MP4 saturates the data path — so clients killed their own
    about-to-complete jobs at the moment sibling results downloaded.

    `complete` is false unless EVERY matching fleet was reachable AND actually
    reported a running list. Nothing may conclude a job is gone without it.
    """

    running: dict[str, dict]  # job_id -> the fleet's record of it
    mean_job_s: float | None
    complete: bool
    # Servability, for callers deciding whether to submit AT ALL. A fleet with
    # replicas attached but none accepting still queues work — and a queued
    # ticket is exempt from every stall verdict below, so submitting into that
    # state is how a caller waits forever.
    replicas: int = 0
    accepting_replicas: int = 0
    tasks: frozenset[str] = frozenset()

    def holds(self, media_id: str) -> bool:
        return media_id in self.running

    def absent(self, media_id: str) -> bool:
        """Not running AND we can trust the reading. Never `not holds()`."""
        return self.complete and media_id not in self.running


def capacity_view(payload: dict, profile: str | None = None) -> FleetView:
    """Parse /v1/capacity from a balancer (flat) or the gateway (fleets list).

    `profile` narrows to one fleet; None accepts every fleet, which is what
    a whole-platform view (top, batch status) wants.
    """
    fleets = payload.get("fleets") or [payload]
    running: dict[str, dict] = {}
    mean: float | None = None
    complete = True
    matched = False
    replicas = accepting = 0
    tasks: set[str] = set()
    for fleet in fleets:
        if profile is not None and fleet.get("profile") not in (None, profile):
            continue
        matched = True
        replicas += int(fleet.get("replicas") or 0)
        accepting += int(fleet.get("accepting_replicas") or 0)
        tasks.update(str(t) for t in (fleet.get("tasks") or []))
        if fleet.get("mean_job_s"):
            mean = float(fleet["mean_job_s"])
        # BOTH conditions. "reachable" alone let a reachable fleet that
        # reported no running key at all count as positive evidence of an
        # empty fleet.
        if fleet.get("reachable", True) and isinstance(fleet.get("running"), list):
            for job in fleet["running"]:
                job_id = str(job.get("job_id") or "")
                if job_id:
                    running[job_id] = job
        else:
            complete = False
    return FleetView(
        running=running,
        mean_job_s=mean,
        complete=complete and matched,
        replicas=replicas,
        accepting_replicas=accepting,
        tasks=frozenset(tasks),
    )


def fetch_capacity_view(spec: RunSpec, profile: str | None = None) -> FleetView:
    """capacity_view over a live read, for callers that just want the answer."""
    r = httpx.get(f"{spec.api_base}/v1/capacity", headers=_headers(), timeout=_PROBE_TIMEOUT)
    r.raise_for_status()
    return capacity_view(dict(r.json()), profile)


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
        return str(cached)
    # NEGATIVE results are cached too, briefly. A job with no owner yet — the
    # entire time it sits in the queue — returns None, and probe_stage returns
    # before reaching its own throttle, so every 5s tick issued a fresh
    # /v1/capacity GET with a 10s timeout: the poll interval silently became
    # 15s and one socket pool was built and discarded per tick, per item.
    # Short enough that a bar still appears promptly once work starts.
    missed = _OWNER_MISSES.get(media_id)
    if missed is not None and time.monotonic() - float(missed) < _OWNER_MISS_EVERY_S:
        return None
    job = fetch_capacity_view(spec, spec.profile).running.get(media_id)
    if not job or not job.get("backend"):
        _OWNER_MISSES.put(media_id, time.monotonic())
        return None
    # urlsplit, not rsplit(":"): a backend without an explicit port (or an
    # IPv6 literal) made this raise ValueError, which the caller's blanket
    # except turned into "no stage" — a progress bar that silently loses its
    # step counter forever.
    port = urlsplit(str(job["backend"])).port
    unit = _unit_for_port(spec, port) if port else None
    if unit:
        _OWNER_UNITS.put(media_id, unit)
    return unit


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
        # STAMPED BEFORE THE PROBE, so a failed or slow one backs off exactly
        # like a successful one. Caching only on success meant an
        # unresponsive host was never cached, so every tick re-probed and
        # re-waited — the throttle protected the healthy case and abandoned
        # the sick one.
        _STAGE_CACHE.put(unit, (time.monotonic(), None))
        stage = serve.stage_progress(spec, elapsed_s, unit=unit, timeout=_STAGE_PROBE_S)
        _STAGE_CACHE.put(unit, (time.monotonic(), stage))
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


def _submit(spec: RunSpec, body: dict[str, Any]) -> dict[str, Any]:
    """Hand the request to the platform. Bounded independently of any
    generation deadline — a submission answers in milliseconds."""
    r = httpx.post(
        f"{spec.api_base}/v1/videos",
        json=body,
        headers=_headers(),
        timeout=_SUBMIT_TIMEOUT,
    )
    if r.status_code >= 400:
        raise GenerationRejected(f"generation rejected ({r.status_code}): {r.text[:500]}")
    return dict(r.json())


class TicketLostError(MediaError):
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


class StalledError(MediaError):
    """The server is verifiably NOT working on this id, and it is not
    waiting in any queue either.

    This is what replaces blanket deadlines. A 2-hour per-item timeout was
    never derived from anything and wedged a worker slot for 2 hours per
    dead item — at concurrency 4 that is the whole fleet idle while the
    display counts up. Failure is decided from EVIDENCE instead:

      queued ticket  -> exists server-side (a dead one 404s -> resubmitted)
      accepted job   -> appears in the server's running set within minutes
      generating     -> finishes within 3x what this fleet/duration takes

    That last limit is a CLIENT policy with no server counterpart: the
    balancer's reconciler closes a job only on a terminal status from its
    replica, never on age, so nothing server-side will ever agree or
    disagree with this number. (An earlier comment claimed the two mirrored
    each other; the server rule it named had already been replaced.)
    """


# "Accepted but nowhere to be seen" grace: many multiples of the seconds a
# submit->running handoff actually takes, nowhere near a generation time.
_STALL_QUEUED_S = 300.0
# Overrun rule for work the server confirms is running. A CLIENT policy:
# the balancer's reconciler closes a job only on a terminal status from its
# replica, never on age, so nothing server-side mirrors this number.
_STALL_RUNNING_MULTIPLE = 3.0
_STALL_RUNNING_FLOOR_S = 1800.0
# How often to cross-check the server's running set while status is opaque.
_EVIDENCE_EVERY_S = 60.0
# Longest a worker may go without noticing a cancel. Teardown time is
# bounded by this, and the interpreter's thread-join at shutdown is what
# the user actually waits on.
_CANCEL_SLICE_S = 0.25
# How long the PLATFORM needs to heal a job it has lost track of, before a
# client is entitled to call it gone. Mirrors the balancer's own constants:
# RECONCILE_MIN_AGE_S (60s) + JOB_LOST_STRIKES (5) * DRAIN_S (10s), plus a
# beat of slack. Resubmitting sooner races the server's recovery and bills
# the same clip twice. Raise this if those server constants grow.
_SERVER_RECOVERY_S = 150.0


def _server_view(spec: RunSpec, media_id: str) -> tuple[bool, float | None, bool]:
    """(running now, fleet mean job seconds, is this actual EVIDENCE).

    Reads /v1/capacity — the same shape from a balancer (flat) or the
    gateway (fleets list). The status endpoint cannot answer this: the
    engine reports "queued" from submission to completion, so status alone
    can never distinguish generating from abandoned.

    The third value is the trust bit, and it is what keeps verdict 1
    honest. The gateway answers 200 even when a fleet's balancer could not
    be asked — the entry says `reachable: false` and carries NO `running`
    list. Reading that as "the job is absent" laundered a failed read into
    positive proof of death: capacity probes time out exactly while a
    finished clip's MP4 saturates the data path, so clients killed their
    own about-to-complete jobs at the moment sibling results downloaded.
    Absence is evidence ONLY from a reachable fleet that actually reported
    its running set.
    """
    view = fetch_capacity_view(spec, spec.profile)
    return view.holds(media_id), view.mean_job_s, view.complete


# Positive absence reads required before verdict 1 (spanning >= this many
# evidence intervals on top of the initial grace window).
_STALL_ABSENT_CONFIRMATIONS = 5

# Consecutive unreadable polls (5xx and friends) before we stop believing the
# platform is merely having a moment. 12 x 5s poll = ~1 minute, comfortably
# longer than a systemd restart of a balancer or gateway.
_UNREADABLE_POLLS = 12
# Consecutive unreachable polls. Same window, same reasoning: a restart is
# seconds, a vanished host is not.
_TRANSPORT_FAILS = 12
# Consecutive polls of a ticket whose fleet reports zero accepting replicas.
# Longer (~2.5 min at 5s) than the others on purpose: the balancer's own
# fleet-dead verdict runs at ~5 min, so the client speaks first only when it
# has watched the same positive fact for a sustained window, and a fleet
# switch or a replica restart that resolves inside it costs nobody anything.
_UNSERVABLE_POLLS = 30


def _expected_job_s(seconds: float | None) -> float | None:
    """What this request should take, from the measured duration curve.

    Delegates to `economics`, which owns the curve's meaning — this module
    had its own piecewise-linear copy, so the two could drift on what the
    same measurements meant. A missing/unparseable economics file is not
    silently absorbed here: the stall limit would quietly fall back to the
    short-clip fleet mean, reinstating the very bug the curve prevents.
    """
    from . import economics

    if not seconds:
        return None
    try:
        return economics.expected_gpu_seconds(seconds)
    except (OSError, ValueError, KeyError, TypeError):
        # A malformed economics.toml must NOT take generation down with a
        # raw TOMLDecodeError: this file is an operator-editable config
        # that every generate/batch item now reads. Fall back to fleet
        # evidence alone, exactly as when no measurements are deployed.
        return None


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
    expected_s: float | None = None,
    adopted: bool = False,
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
    # When a queued ticket handed over to a real job — the moment its life
    # as schedulable work actually began. None until then (or forever, for
    # a direct/adopted real id, where `began` already IS that moment).
    handed_over: float | None = None
    # Consecutive polls we could not read at all (non-2xx that is not 404/409)
    # or could not reach. Each was previously ignored, which meant an
    # unreadable or unreachable platform was indistinguishable from a healthy
    # one that had nothing to report — the caller waited forever.
    unreadable = 0
    transport_fails = 0
    # Consecutive polls showing this ticket queued while the fleet reports it
    # cannot accept work at all. Not a timeout: zero accepting replicas is a
    # positive fact about servability, the same kind of evidence the other
    # verdicts use.
    unservable = 0
    while time.monotonic() < deadline:
        if cancel is not None and cancel.is_set():
            # Clean interruption: the JOB keeps running server-side and can
            # be adopted later — only this poller stops.
            raise InterruptedError(effective)
        try:
            r = httpx.get(
                f"{spec.api_base}/v1/videos/{media_id}",
                headers=_headers(),
                timeout=_POLL_TIMEOUT,
            )
            if r.status_code == 409:
                # AMBIGUOUS ownership: the platform can see this id in more
                # than one place and refuses to guess. Resubmitting would
                # bill the work twice, and the streak counter must not treat
                # it as absence — keep polling and let the platform settle.
                lost = 0
            elif r.status_code == 404:
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
                    # 404 means UNROUTABLE, which is not the same as gone —
                    # for a ticket OR a real job. A ticket 404s not only
                    # when a balancer restart destroyed it (pre-billable,
                    # safe to resubmit) but also when it has ALREADY handed
                    # over to a running job the gateway cannot currently
                    # resolve, or when a prune dropped the ticket while its
                    # job generates. Resubmitting those bills the same clip
                    # twice, so both branches now demand the same positive
                    # evidence: the fleet must SAY nothing is running.
                    #
                    # A fleet that cannot be asked at all (--direct: no
                    # capacity endpoint) leaves the 404s as the only truth
                    # there is, so it still resubmits.
                    try:
                        running, mean, ev = _server_view(spec, effective)
                    except (httpx.HTTPError, ValueError, TypeError):
                        raise TicketLostError(media_id) from None
                    fleet_mean = mean or fleet_mean
                    if not ev:
                        raise TicketLostError(media_id)
                    if not running:
                        absent_confirmed += 1
                    # The bar is the SERVER's recovery clock, not a couple
                    # of reads. The balancer waits RECONCILE_MIN_AGE_S (60s)
                    # before its first probe, then JOB_LOST_STRIKES (5) x
                    # DRAIN_S (10s) before it releases the slot — ~110s. A
                    # client that resubmits after ~10s of 404s delivers its
                    # verdict INSIDE that window every time, duplicating
                    # work the platform was about to heal.
                    if (
                        absent_confirmed >= _STALL_ABSENT_CONFIRMATIONS
                        and time.monotonic() - began > _SERVER_RECOVERY_S
                    ):
                        raise TicketLostError(media_id)
                    # Alive: keep polling. Routing noise must never mint a
                    # duplicate generation.
                    lost = 0
            elif r.status_code < 400:
                # Only a real answer clears the streak: 5xx interleaved with
                # 404s (a crash-looping balancer) must not keep a dead id
                # alive forever.
                lost = 0
                last = dict(r.json())
                # Verdict 0 — a QUEUED ticket the fleet says it can never
                # start. The balancer now reports accepting_replicas on every
                # ticket poll precisely so this is answerable: a ticket is
                # exempt from verdicts 1 and 2 (waiting is legitimate and
                # unbounded, and killing queued work on a timer is what this
                # platform refuses to do), so without this the client waits
                # forever and is RIGHT to. Zero accepting replicas is not a
                # guess about elapsed time — it is the fleet stating that
                # nothing can take the work.
                if last.get("status") == "queued" and last.get("accepting_replicas") == 0:
                    unservable += 1
                    if unservable >= _UNSERVABLE_POLLS:
                        raise StalledError(
                            f"{effective}: queued at position "
                            f"{last.get('queue_position')} on a fleet with NO "
                            f"replica accepting work, confirmed {unservable}x — "
                            f"nothing can start this. It was never billed; it "
                            f"resubmits when capacity returns. Check "
                            f"`deepme balancer status -m {spec.profile}`."
                        )
                else:
                    unservable = 0
                real = str(last.get("id") or "")
                if real and real != effective and not real.startswith("q_"):
                    effective = real
                    # A NEW identity gets a CLEAN evidence sheet. The absence
                    # counter accumulated while this was a queued ticket —
                    # legitimately absent from the running set for however
                    # long the queue was (13x-39x on busy nights) — and
                    # carrying it across the handover made verdict 1 fire
                    # the instant the exemption lifted, killing jobs that
                    # had JUST reached a replica, on evidence about an id
                    # they no longer were.
                    absent_confirmed = 0
                    handed_over = time.monotonic()
                    if on_real_id:
                        on_real_id(real)  # persist: adoption needs THIS id
                with contextlib.suppress(ValueError, TypeError):
                    if last.get("created_at"):
                        server_started = float(last["created_at"])
                if last.get("status") in TERMINAL:
                    return last
                unreadable = 0
                transport_fails = 0
            elif r.status_code in (401, 403):
                # NEVER retried. A rejected credential is not a transient
                # condition — it is the same answer every time — and with no
                # else-branch here it fell through every case: `lost` was not
                # incremented, so the ticket-lost verdict never fired, and the
                # caller polled a 401 forever without ever being told the key
                # was wrong.
                raise GenerationRejected(
                    f"the platform rejected our credentials ({r.status_code}) polling "
                    f"{effective} — check .deepme/api_key against the deployed "
                    f"BALANCER_API_KEY/GATEWAY_API_KEY"
                )
            else:
                # Any OTHER non-2xx (500/502/503 from a crash-looping service).
                # Counted, not ignored: the comment on the 404 branch says 5xx
                # "must not keep a dead id alive forever", but falling through
                # without incrementing anything achieved precisely that.
                unreadable += 1
                if unreadable >= _UNREADABLE_POLLS:
                    raise StalledError(
                        f"{effective}: the platform has answered "
                        f"{r.status_code} to {unreadable} consecutive polls — "
                        f"it is not serving this job. Check "
                        f"`deepme balancer status -m {spec.profile}` and "
                        f"`deepme gateway logs`."
                    )
        except httpx.HTTPError as exc:
            # A gateway that goes away mid-run used to leave the caller
            # polling a dead host indefinitely: no response means the 404
            # branch is unreachable, evidence reads fail and are suppressed,
            # and nothing counted the failures.
            transport_fails += 1
            if transport_fails >= _TRANSPORT_FAILS:
                raise StalledError(
                    f"{effective}: {transport_fails} consecutive polls could not "
                    f"reach {spec.api_base} ({exc!r}) — the job may still be "
                    f"running server-side and will be re-adopted on the next "
                    f"run. Check `deepme gateway status`."
                ) from None
        now = time.monotonic()
        if now - last_check >= _EVIDENCE_EVERY_S:
            last_check = now
            with contextlib.suppress(httpx.HTTPError, ValueError, TypeError):
                running, mean, evidence = _server_view(spec, effective)
                fleet_mean = mean or fleet_mean
                if running:
                    running_since = running_since or now
                    absent_confirmed = 0
                elif evidence and not effective.startswith("q_"):
                    # Only a reachable fleet's actual running set counts —
                    # an unreachable fleet is a failed read wearing a 200.
                    # And only about a REAL job id: a queued ticket is
                    # rightly absent, and that absence says nothing about
                    # the job it will hand over to.
                    absent_confirmed += 1
        # Verdict 1 — accepted but ownerless, CONFIRMED absent repeatedly.
        # (Un-handed tickets are exempt: waiting in the queue is a
        # legitimate unbounded state, and a dead ticket 404s above.) The
        # grace clock starts at HANDOVER, not submission: a job that spent
        # 30 minutes queued has been a real job for seconds.
        held_since = handed_over if handed_over is not None else began
        if (
            running_since is None
            and not effective.startswith("q_")
            and absent_confirmed >= _STALL_ABSENT_CONFIRMATIONS
            and now - held_since > _STALL_QUEUED_S
        ):
            raise StalledError(
                f"{effective}: a replica took it {int(now - held_since)}s "
                f"ago and the server has confirmed {absent_confirmed}x "
                f"since that no replica is running it — it will never "
                f"complete. Check `deepme balancer status -m {spec.profile}`."
            )
        # Verdict 2 — running far beyond what THIS REQUEST should take:
        # mirror of the balancer reaper's rule, lifted by the measured
        # duration curve. The fleet mean is dominated by short clips, and
        # 3x that mean sits BELOW the honest wall time of a 15s clip
        # (measured 5.7x a 4s clip, superlinear) — every long clip on a
        # mixed fleet was executed as "wedged" mid-generation.
        if running_since is not None:
            limit = max(
                (fleet_mean or 0.0) * _STALL_RUNNING_MULTIPLE,
                (expected_s or 0.0) * _STALL_RUNNING_MULTIPLE,
                _STALL_RUNNING_FLOOR_S,
            )
            if now - running_since > limit:
                raise StalledError(
                    f"{effective}: running for {int(now - running_since)}s, "
                    f"over {int(limit)}s (3x what this fleet/duration "
                    f"takes) — the engine has almost certainly wedged. "
                    f"Check `deepme serve logs -m {spec.profile}`."
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
                    adopted=adopted,
                )
            )
        # CANCELLABLE, in slices. A single `time.sleep(every)` could not be
        # interrupted, so every worker slept out its whole poll interval
        # before noticing a ^C — and the interpreter's shutdown then JOINS
        # those threads with no timeout, which is why holding ^C down was
        # the only way to stop. Slices keep `time.sleep` as the clock (a
        # bare Event.wait ignores a patched clock and makes the suite sleep
        # for real) while bounding the notice period to _CANCEL_SLICE_S.
        waited = 0.0
        while waited < every:
            if cancel is not None and cancel.is_set():
                raise InterruptedError(effective)
            step = min(_CANCEL_SLICE_S, every - waited)
            time.sleep(step)
            waited += step
    raise GenerationTimeout(
        f"generation {effective} did not finish before the timeout "
        f"(last status: {last.get('status', 'unknown')!r}) — the job may still "
        f"be running; check `deepme serve logs -m {spec.profile}`"
    )


def _download(spec: RunSpec, media_id: str, dest: Path, cancel: Any = None) -> int:
    """Fetch the result to `dest`, atomically.

    Streamed to a sibling `.part` and renamed only after the last byte:
    writing straight to the final path left a TRUNCATED mp4 in the clips
    directory whenever a download was cut (a gateway restart does this
    routinely). Batch reconciliation only checks `.exists()`, so the
    fragment counted as a finished deliverable — and the dedupe tool
    hashed it as a real clip.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A UNIQUE temp name. A fixed `{dest}.part` is shared by any two
    # processes fetching the same job into one directory (a
    # `--fetch-existing` run beside a `generate --adopt`), where the first
    # to finish unlinks the file the second is still writing.
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f"{dest.name}.", suffix=".part")
    os.close(fd)
    part = Path(tmp_name)
    written = 0
    try:
        with (
            httpx.Client(timeout=_DOWNLOAD_TIMEOUT) as c,
            c.stream(
                "GET",
                f"{spec.api_base}/v1/videos/{media_id}/content",
                headers=_headers(),
            ) as r,
        ):
            if r.status_code >= 400:
                r.read()
                raise DownloadFailed(f"content fetch failed ({r.status_code}): {r.text[:300]}")
            began = time.monotonic()
            with part.open("wb") as f:
                for chunk in r.iter_bytes():
                    # A TOTAL bound, not just a per-read one. _DOWNLOAD_TIMEOUT
                    # bounds the gap between chunks; a server dribbling one
                    # chunk every 899s satisfies it forever. Generous enough
                    # that no honest transfer can trip it (clips are single-
                    # digit MB and arrive in seconds), so tripping it means
                    # the transfer is not progressing.
                    if time.monotonic() - began > _DOWNLOAD_TOTAL_S:
                        raise DownloadFailed(
                            f"content fetch for {media_id} exceeded "
                            f"{_DOWNLOAD_TOTAL_S:.0f}s after {written} bytes — "
                            f"the transfer is not progressing. The artifact is "
                            f"still on the server (it is reclaimed only on ack) "
                            f"and can be re-fetched."
                        )
                    # Cancellable PER CHUNK. Without this the only wait in
                    # the client that ignores CANCEL was a multi-hundred-MB
                    # transfer with a 900s timeout — so a second ^C blocked
                    # on the worker join for up to fifteen minutes, making
                    # the interrupt protocol's "a few seconds of orderly
                    # teardown" untrue exactly when it is read.
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError(media_id)
                    f.write(chunk)
                    written += len(chunk)
        # DURABLE BEFORE ACK. fsync the bytes AND the directory entry, so the
        # ack means "this survives a power loss", not "this reached the page
        # cache". Acking on a buffered write would let the server reclaim the
        # only copy of a file we could still lose.
        with part.open("rb+") as f:
            os.fsync(f.fileno())
        part.replace(dest)
        dir_fd = os.open(dest.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        # Only on FAILURE: after a successful replace the name is gone, and
        # unlinking unconditionally would race a concurrent writer that has
        # since created it.
        part.unlink(missing_ok=True)
        raise
    _ack(spec, media_id)
    return written


def _ack(spec: RunSpec, media_id: str) -> None:
    """Tell the platform we have the bytes; it may reclaim its copy.

    Sent ONLY after the file is durably in place. The server holds the
    artifact until this arrives (or its retention window expires), which is
    what makes a failed local write recoverable — under delete-on-delivery a
    caller that got a 200 and then could not write had lost the work for
    good, with no way to ask again.

    BEST EFFORT, and deliberately so: we already have the file, so the
    caller's result must not fail because cleanup did not. A lost ack costs
    one artifact one retention window, which is what the window is for.
    """
    with contextlib.suppress(httpx.HTTPError):
        httpx.delete(
            f"{spec.api_base}/v1/videos/{media_id}/content",
            headers=_headers(),
            timeout=_PROBE_TIMEOUT,
        )


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
    resubmit: bool = True,
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
        raise RequestInvalid(
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
    # What THIS request should take, from the measured duration curve — the
    # stall detector's per-request baseline (a 15s clip is not a wedged 5s
    # clip). Read from the final body so --param overrides count too.
    req_seconds: float | None = seconds
    if req_seconds is None:
        with contextlib.suppress(TypeError, ValueError):
            tgt = body.get("target")
            if isinstance(tgt, dict) and tgt.get("duration_seconds"):
                req_seconds = float(tgt["duration_seconds"])
    expected_s = _expected_job_s(req_seconds)

    start = time.monotonic()
    deadline = start + timeout if timeout else float("inf")
    if adopt:
        job: dict[str, Any] = {"id": adopt, "status": "queued"}
        media_id = adopt
        if on_event:
            on_event(f"re-attaching to existing job {media_id}")
    elif not resubmit:
        # STRUCTURALLY collection-only. `resubmit=False` used to guard only
        # the RE-submission path, so any caller reaching here with no id to
        # adopt still created brand-new server work — a collector that
        # announced "submitting nothing" and then commissioned generations.
        # One flag, one meaning: this call may never create work.
        raise TicketLostError(
            "collect-only: nothing to adopt for this item and submission is "
            "disabled, so no work was created"
        )
    else:
        job = _submit(spec, body)
        media_id = job.get("id") or ""
        if not media_id:
            raise GenerationRejected(f"server returned no video id: {job}")
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
                expected_s=expected_s,
                adopted=bool(adopt),
            )
        except TicketLostError:
            if not resubmit:
                # Collect-only callers (batch --fetch-existing) must never
                # create new server work: a lost job is reported, not redone.
                raise TicketLostError(
                    f"{media_id} no longer exists on the server and "
                    f"resubmission is disabled — it must be regenerated"
                ) from None
            resubmits += 1
            if resubmits > 3:
                raise TicketLostError(
                    f"{media_id} was lost {resubmits} times — the platform "
                    f"keeps forgetting it; check "
                    f"`deepme balancer status -m {spec.profile}`"
                ) from None
            if cancel is not None and cancel.is_set():
                # Teardown must not create billable work. A ^C during a
                # balancer restart — the very event that kills tickets —
                # let this path post a brand-new job while the operator
                # was being told the run was interrupted.
                raise InterruptedError(media_id) from None
            if on_event:
                on_event(f"{media_id} no longer exists on the server — resubmitting")
            job = _submit(spec, body)
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
    size_bytes = _download(spec, media_id, dest, cancel)
    result.path, result.size_bytes = dest, size_bytes
    result.total_s = time.monotonic() - start
    return result
