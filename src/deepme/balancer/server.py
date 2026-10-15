"""Replica load balancer: one API in front of N identical model deployments.

A media replica serves one job at a time and can hold it for tens of minutes,
so capacity comes from running several and steering each request to whichever
is free. Two rules do the work (see balance.py):

  new request  -> the replica with the least outstanding work
  existing job -> the replica that created it, always

Job ids are minted per replica, so the second rule is not an optimisation:
polling the wrong replica returns 404 and the result is stranded. If the
balancer has forgotten a job (restart, eviction), it probes the replicas to
find the owner again rather than failing.

Environment: BALANCER_BACKENDS (JSON list of base URLs), BALANCER_PORT,
BALANCER_API_KEY. Deployed to ~/deepme-runtime/balancer alongside balance.py.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
from collections import deque
from pathlib import Path

import httpx

# NOT named queue.py: that shadows the stdlib `queue` module, and which one
# wins depends on sys.path order at import time.
from admission import (  # deployed next to this
    CANCELLED,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    REJECTED,
    SUBMITTED,
    AdmissionQueue,
)
from balance import Balancer  # deployed next to this file
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Shared with the gateway — see proxycore's docstring for why the owner
# probe must not be maintained twice. Deployed next to this file too.
from proxycore import (
    TERMINAL_STATUSES,
    AmbiguousOwner,
    auth_headers,
    collection_of,
    content_headers,
    find_owner,
    is_content,
    job_id_of,
    safe_headers,
    status_path,
)

BACKENDS = json.loads(os.environ.get("BALANCER_BACKENDS", "[]"))
API_KEY = os.environ.get("BALANCER_API_KEY", "")
# A generation can run for tens of minutes; only the connect phase is quick.
# This unbounded-read client is for PROXYING CLIENT TRAFFIC ONLY.
TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
# Control-plane calls — pump submissions, reconcile probes, owner probes,
# health probes — are all sub-second against a healthy replica and MUST be
# bounded: they run inside the maintenance loop, and one wedged replica
# holding a read open forever would freeze the very task that exists to heal
# wedges (pump, reconcile and prune all die together).
CTL_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
# Boot recovery blocks the port from binding, so it gets a much tighter
# bound than ordinary control-plane work: a replica that cannot answer in
# this window is reconciled by the maintenance beat instead of holding the
# whole service in connection-refused.
RECOVERY_TIMEOUT = httpx.Timeout(5.0, connect=2.0)

DRAIN_S = 10.0

# Maintenance failures are OPERATIONAL FACTS, exported in /metrics: the
# liveness invariant is enforced by that loop, so a beat that dies must be
# visible rather than merely absent.
maintenance_errors = 0
retention_errors = 0
# Operational counters an incident actually asks for. /metrics carried only
# gauges plus reaped/retention, so "why did submissions start failing" and
# "which replica is degrading" had no alertable series at all.
submit_rejected = 0  # replica said no (4xx/5xx)
submit_ambiguous = 0  # transport died mid-accept; outcome unknown
submit_requeued = 0  # connect-phase failure, safely returned to the queue
owner_conflicts = 0  # two replicas claimed one id
owner_rejections = 0  # a remembered binding failed verification
# Running on a replica but absent from our job map when the sweep looked. Any
# non-zero value means we were double-booking a GPU; a rising one means the
# submit path is losing acceptances (chase submit_ambiguous first).
orphans_adopted = 0
fetches = 0
fetch_failures = 0

# Strong references to in-flight pump tasks (see _safe_pump).
_pending_pumps: set = set()

# ---------------------------------------------------------------------------
# Artifact retention (the user-set policy, enforced by the platform):
#   * an artifact never fetched is kept 6h from creation — the initial-fetch
#     window;
#   * once fetched, it is kept 1h from the FIRST fetch — a re-fetch window —
#     then reclaimed. Space comes back on the platform's own clock; nothing
#     accumulates until a disk full fails generations.
# The balancer owns this because it already witnesses both events: it serves
# every /content fetch, and it runs on the host that holds the files.
RETENTION_TTL_S = float(os.environ.get("RETENTION_TTL_S", 6 * 3600))
RETENTION_FETCHED_TTL_S = float(os.environ.get("RETENTION_FETCHED_TTL_S", 3600))
# The profile's work dir (outputs/ + inputs/). Retention is OFF when unset —
# a balancer that does not know where the files live must not guess.
WORK_DIR = os.environ.get("DEEPME_WORK_DIR", "")
# Every ~6 maintenance beats (~1min): a directory scan is cheap but not free.
_RETENTION_EVERY_BEATS = 6
# Ditto for the orphan sweep, which costs one /v1/videos listing per replica.
# Slower than the reap because a ghost wastes a slot, not correctness, and the
# listing grows with each replica's whole history.
_ORPHAN_SWEEP_EVERY_BEATS = 6


# Consecutive maintenance beats with ZERO accepting replicas before waiting
# tickets are refused. ~5 minutes at DRAIN_S=10.
#
# The waiting ticket was the only unbounded object in this system — jobs are
# bounded by strikes and stale_jobs, the owner map by max_jobs, the event ring
# by maxlen, artifacts by retention, settled tickets by prune. A ticket on a
# dead fleet was not: nothing could move it out of WAITING, because every path
# to REJECTED lives inside _pump AFTER pick() returns a target, and pick()
# returns None on a dead fleet — so the pump bails before reaching the code
# that could refuse anything. Client-side, a queued ticket is deliberately
# exempt from every stall verdict (waiting is legitimate, and killing queued
# work on a timer is exactly what this platform refuses to do). The two
# together deadlock permanently.
#
# This is a STRIKE COUNT, not a timeout, and the evidence is positive: zero
# accepting replicas means no replica can EVER take this work, which is a
# fact, not a guess about elapsed time. A single accepting replica resets it,
# so a fleet switch or a blip cannot refuse legitimately queued work — the
# drain would have to outlast every beat in the window. Set to 0 to disable
# for a migration longer than the window.
FLEET_DEAD_STRIKES = int(os.environ.get("FLEET_DEAD_STRIKES", "30"))
_fleet_dead_beats = 0
# Waiting tickets refused because nothing could ever serve them.
submit_unservable = 0


def _fail_unservable_tickets() -> None:
    """Refuse waiting tickets once the fleet is provably unable to serve them.

    Only NEVER-BILLED tickets can be touched: mark_rejected refuses to move a
    SUBMITTING or SUBMITTED ticket, so work that reached a replica is safe by
    construction.
    """
    global _fleet_dead_beats, submit_unservable
    if not FLEET_DEAD_STRIKES:
        return
    if bal.capacity()["accepting_replicas"]:
        _fleet_dead_beats = 0  # one live replica clears the whole count
        return
    _fleet_dead_beats += 1
    if _fleet_dead_beats < FLEET_DEAD_STRIKES:
        return
    waiting = list(admission.waiting())
    if not waiting:
        return
    detail = (
        f"no replica has been able to accept work for "
        f"{int(FLEET_DEAD_STRIKES * DRAIN_S)}s ({len(bal.stats)} attached, "
        f"0 accepting) — this was never submitted and never billed; "
        f"resubmit when capacity returns"
    )
    for ticket in waiting:
        if admission.mark_rejected(ticket.ticket_id, detail):
            submit_unservable += 1
            _publish("rejected", ticket=ticket.ticket_id, error=detail[:120])
    print(f"refused {len(waiting)} waiting ticket(s): {detail}", flush=True)


@contextlib.asynccontextmanager
async def _lifespan(_app):
    """Drain the queue on a timer as well as on traffic.

    Pumping only when a request arrives starves: a client that submits and
    walks away leaves its ticket waiting even as replicas go idle, because
    nothing is arriving to trigger the drain. That is precisely the unattended
    overnight case the queue exists for — the fleet would sit at 0% with work
    outstanding until someone happened to poll.
    """
    import asyncio

    async def drain() -> None:
        # Declared once, up front: Python scopes `global` per FUNCTION, so a
        # second declaration lower down is a SyntaxError, not a redundancy.
        global maintenance_errors, retention_errors
        beats = 0
        while True:
            await asyncio.sleep(DRAIN_S)
            beats += 1
            try:
                # Order matters: reconcile FIRST (frees slots the replicas
                # say are done), revive lapsed backends, then pump into
                # whatever is free, then prune. Together these enforce the
                # liveness invariant — with zero client traffic, finished
                # slots still free, failed backends still return, and the
                # queue still drains.
                await _reap_stale_jobs()
                await _revive_backends()
                await _pump()
                # AFTER revive and pump, so a returning replica always gets
                # first refusal on the work before anything is refused.
                _fail_unservable_tickets()
                # Tickets whose job is still running keep their binding —
                # the caller was promised that id keeps working.
                admission.prune(live_jobs={j for j, _u, _t, _c in bal.open_jobs()})
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # COUNTED and NAMED, never silently swallowed. This
                # sequence IS the liveness invariant; a blanket suppress
                # turned a permanent violation (the pump never running
                # again, every ticket waiting forever on an idle fleet)
                # into something indistinguishable from a healthy beat.
                maintenance_errors += 1
                print(f"maintenance beat failed: {exc!r}", flush=True)
            if beats % _ORPHAN_SWEEP_EVERY_BEATS == 0:
                try:
                    await _recover_occupancy(boot=False)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    maintenance_errors += 1
                    print(f"orphan sweep failed: {exc!r}", flush=True)
            if beats % _RETENTION_EVERY_BEATS == 0:
                try:
                    # The WALK goes to a thread (slow, pure); the deletes
                    # and the publish stay on the loop, which owns the
                    # event ring, the subscriber queues and `_fetched`.
                    _retention_apply(await asyncio.to_thread(_retention_plan))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    retention_errors += 1
                    print(f"retention pass failed: {exc!r}", flush=True)

    # BEFORE serving: occupancy must reflect replica truth or the first
    # request routes onto a replica that is already generating. A
    # create_task here only SCHEDULED the recovery, so startup completed
    # with an empty job map — the double-booking this exists to prevent.
    with contextlib.suppress(Exception):
        await _recover_occupancy()
    task = asyncio.create_task(drain())
    try:
        yield
    finally:
        # Cancel AND await: cancellation is only a request, and a pump
        # abandoned mid-POST never ran its cleanup, leaving a ticket stuck
        # in SUBMITTING and a reserved slot leaked across the restart.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await client.aclose()
        await ctl.aclose()


app = FastAPI(lifespan=_lifespan)
bal = Balancer(list(BACKENDS))
# Work waits HERE when every replica is busy, rather than inside an engine.
# A request that has reached a replica is consuming a GPU and is billable, so
# cancellation and reprioritisation are only possible while it is still ours.
#
# Ticket ids are salted with this balancer's name AND a per-boot nonce.
# Without the salt, every balancer mints q_00000001 and the gateway — which
# keys job ownership by bare id — routed one customer's poll (and worse,
# their CANCEL) to another customer's ticket on a different fleet. The boot
# nonce keeps a restarted balancer from re-minting ids the gateway may still
# map to the previous incarnation's customers.
# 8 random bytes: the 2-byte nonce this started with left a 1-in-65536
# cross-boot collision on the exact cross-customer routing the salt closes.
admission = AdmissionQueue(salt=f"{os.environ.get('BALANCER_NAME', 'b')}-{os.urandom(8).hex()}-")
client = httpx.AsyncClient(timeout=TIMEOUT)
ctl = httpx.AsyncClient(timeout=CTL_TIMEOUT)

# One vocabulary for "this job is over", shared by every code path. Two of
# the four paths previously listed only two of the four words, so inline
# cancellations opened occupancy that nothing would ever close.


def require_key(request: Request) -> None:
    """Gate the CONTROL PLANE (/balancer/*). Not /v1/*.

    These routes change the fleet: attach, drain, detach. They were open to
    anyone who could reach the port — and the balancer binds 0.0.0.0 — so a
    single unauthenticated DELETE could tear the fleet down. `/v1/*` stays
    ungated deliberately: it forwards the CALLER's credentials to the
    engine, and gating it here would break the service layer's own auth.

    Off when no key is configured, so a local/dev deployment is unchanged.
    """
    if not API_KEY:
        return
    header = request.headers.get("authorization", "")
    if header.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="control-plane routes require the platform key")


def _log_bad_binding(job_id: str, target: str) -> None:
    global owner_rejections
    owner_rejections += 1
    print(
        f"owner map was WRONG: {target} does not hold {job_id} — binding "
        f"dropped, re-probing (this is the misattribution guard firing)",
        flush=True,
    )
    _publish("owner_rejected", job_id=job_id, backend=target)


async def _owns(target: str, job_id: str, headers: dict) -> bool:
    """Does this replica actually hold this id? Cheap status GET.

    Unreachable counts as NOT verified: serving bytes on an unverified
    binding is what delivered another job's video.
    """
    with contextlib.suppress(httpx.HTTPError, ValueError, TypeError, KeyError):
        r = await ctl.get(f"{target}{status_path('/v1/videos/' + job_id, job_id)}", headers=headers)
        return r.status_code == 200 and str(dict(r.json()).get("id") or "") == job_id
    return False


async def _find_owner(job_id: str, path: str, headers: dict) -> str | None:
    """Ask every replica which one holds this job.

    Only needed after the balancer lost its mapping. Sequential and cheap: a
    status GET is milliseconds, and a wrong guess would strand the result.

    Recovery restores OCCUPANCY too, not just routing: after a restart the
    fleet is full of jobs the balancer no longer counts, and healing only the
    route left pick() double-booking busy replicas while _all_busy() waved
    the whole former queue into engine-internal FIFOs.
    """

    def _ambiguous(jid: str, claimants: int, _unreachable: int) -> None:
        global owner_conflicts
        owner_conflicts += 1
        _publish("owner_ambiguous", job_id=jid, claimants=claimants)

    found = await find_owner(
        job_id,
        path,
        headers,
        # EVERY attached replica, including ones currently marked unhealthy —
        # see find_owner on why filtering them breaks the guard's argument.
        candidates=[st.url for st in list(bal.stats.values())],
        client=ctl,
        log=lambda msg: print(msg, flush=True),
        on_ambiguous=_ambiguous,
    )
    if found is None:
        return None
    url, status = found
    bal.remember(job_id, url)
    if status not in TERMINAL_STATUSES:
        bal.begin_job(job_id, url)
    return url


async def _safe_pump() -> None:
    """Run the pump shielded from the caller's cancellation.

    Request handlers trigger the pump inline, and a disconnecting client
    (an SSE watcher closing its tab, say) cancels its handler — which used
    to cancel the pump MID-POST, leaking a SUBMITTING ticket and a reserved
    slot forever. The pump finishes on the loop regardless; only the wait
    for it is abandoned.
    """
    import asyncio

    task = asyncio.ensure_future(_pump())
    # A STRONG reference for the task's whole life. The event loop keeps
    # only a weak one, so a pump suspended inside its submission POST — a
    # ticket already claimed, a slot already reserved — could be garbage
    # collected the moment this frame returned on the cancelled path,
    # freezing that ticket in SUBMITTING and leaking the slot forever.
    _pending_pumps.add(task)
    task.add_done_callback(_pending_pumps.discard)
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.shield(task)


@app.get("/v1/videos/{job_id}/events")
async def job_events(job_id: str, request: Request) -> Response:
    """Server-sent progress for one job, so a caller is never left guessing.

    The upstream job API reports "queued" from submission until "completed"
    with nothing in between, which is indistinguishable from being stuck.
    This emits queue position while waiting behind other work, then liveness
    and an ETA once running, then a terminal event.
    """
    headers = auth_headers(request, API_KEY)

    async def stream():
        import asyncio

        last = None
        # A ticket id addresses work that has NOT reached a replica. Following
        # it has to start before there is a job at all, then continue through
        # the same stream once one exists — otherwise a caller who was queued
        # would have to notice the handover and resubscribe.
        ticket_id = job_id if job_id.startswith("q_") else None
        current = job_id
        while True:
            if ticket_id:
                ticket = admission.get(ticket_id)
                if ticket is None:
                    yield _sse({"status": "unknown", "job_id": ticket_id})
                    return
                if ticket.state == CANCELLED:
                    yield _sse({"job_id": ticket_id, "status": "cancelled", "billable": False})
                    return
                if ticket.state == REJECTED:
                    # Terminal and unpaid: the replica refused the request.
                    # Reporting this as "submitted, billable" (the old
                    # retirement path) was a lie in both fields.
                    yield _sse(
                        {
                            "job_id": ticket_id,
                            "status": "failed",
                            # Same error shape as the poll path: one wire
                            # vocabulary, whichever way a client watches.
                            "error": {"message": ticket.error or "the replica refused it"},
                            "billable": False,
                        }
                    )
                    return
                if ticket.state == SUBMITTED:
                    # Handover: a replica has it, and from here it is billable.
                    yield _sse(
                        {
                            "job_id": ticket.job_id or ticket_id,
                            "ticket": ticket_id,
                            "status": "submitted",
                            "queue_position": 0,
                            "billable": True,
                        }
                    )
                    if not ticket.job_id:
                        # The replica answered synchronously and minted no job
                        # id, so there is nothing left to follow.
                        return
                    last = None
                    current = ticket.job_id
                    ticket_id = None
                    continue
                position = admission.position(ticket_id)
                means = [s.mean_job_s for s in bal.stats.values() if s.mean_job_s]
                mean = sum(means) / len(means) if means else 0.0
                # Divided by replicas that can actually DRAIN the queue, not by
                # everything attached: a fleet with dead replicas that once
                # completed work still has a mean, so the old divisor produced
                # a confident finite ETA that could never come true.
                accepting = bal.capacity()["accepting_replicas"]
                wait = admission.wait_estimate_s(ticket_id, mean, accepting or 1)
                event = {
                    "job_id": ticket_id,
                    "status": "queued",
                    "queue_position": position,
                    "accepting_replicas": accepting,
                    "eta_s": round(wait) if wait is not None else None,
                    "billable": False,
                }
                if event != last:
                    yield _sse(event)
                    last = event
                if await request.is_disconnected():
                    return
                await asyncio.sleep(POLL_S)
                await _safe_pump()
                continue

            try:
                target = bal.owner(current) or await _find_owner(
                    current, f"/v1/videos/{current}", headers
                )
            except AmbiguousOwner:
                # Say so on the stream too: a watcher must not read this as
                # "gone" and resubmit.
                yield _sse({"status": "ambiguous", "job_id": current, "retryable": False})
                return
            if target is None:
                yield _sse({"status": "unknown", "job_id": current})
                return
            status = "queued"
            with contextlib.suppress(httpx.HTTPError, ValueError):
                # ctl, not client: this is a status poll, and `client` has
                # read=None. A wedged replica blocked every SSE watcher
                # here forever — never reaching the disconnect check, so
                # the coroutine outlived the client that opened it.
                r = await ctl.get(f"{target}/v1/videos/{current}", headers=headers)
                status = str(dict(r.json()).get("status") or "queued")
            ahead = bal.queue_position(current)
            eta = bal.eta_hint_s(current)
            event = {
                "job_id": current,
                "status": "running" if ahead == 0 and status == "queued" else status,
                "queue_position": ahead,
                "eta_s": round(eta) if eta is not None else None,
            }
            if event != last:
                yield _sse(event)
                last = event
            if status in TERMINAL_STATUSES:
                # `current`, not the id we were subscribed with: after a ticket
                # hands over, the occupancy is held against the JOB id, and
                # ending the ticket id instead leaks the replica slot forever.
                # Note only on a REAL close: whoever closes first owns the
                # recent-ring entry (see Balancer.end_job).
                if bal.end_job(current, ok=status == "completed"):
                    _note_terminal(current, str(status), closed_by="client-sse")
                return
            if await request.is_disconnected():
                return
            await asyncio.sleep(POLL_S)

    return StreamingResponse(stream(), media_type="text/event-stream")


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# How often a subscribed client is updated. Generations run for minutes, so
# a slow cadence is plenty and keeps the replicas free of poll traffic.
POLL_S = 5.0


def _all_busy() -> bool:
    """Is there NO free slot right now — including "no slots at all"?

    `bool(live) and ...` made a fully drained or fully unhealthy fleet
    report NOT busy, so admission waved every submission straight through
    to `pick()`, which had nothing to give, and the caller got a 503
    instead of a ticket. During a `fleet switch` — the exact window the
    queue exists to cover — the queue could never even bootstrap, because
    the "someone is already waiting" path needs a first ticket to exist.
    Delegates to free_slots() so the queue and the capacity endpoint can
    never disagree about what "free" means.
    """
    return bal.free_slots() == 0


def _retry_hint() -> int:
    """Seconds until a replica is likely free, for a caller we just refused.

    The soonest-finishing job is the answer; without history there is nothing
    to base it on, so fall back to a poll interval rather than invent a wait.
    """
    etas = [j["eta_s"] for j in bal.running_jobs() if j["eta_s"] is not None]
    return max(int(min(etas)), 1) if etas else 30


# Query parameters that belong to THIS layer and must not reach an engine.
_INTERNAL_PARAMS = frozenset({"queue"})


def _upstream_params(request: Request) -> dict[str, str]:
    return {k: v for k, v in request.query_params.items() if k not in _INTERNAL_PARAMS}


async def _admit(path: str, request: Request, headers: dict, body: bytes):
    """Hold a submission when every replica is busy; None to submit now.

    The fast path is deliberately unchanged: with a free replica the request
    goes straight through and the caller gets a job id, exactly as before.
    Queuing only happens when it would otherwise pile up INSIDE an engine,
    where it could no longer be cancelled, reprioritised or counted.
    """
    # LATENT GAP — other modalities bypass the queue entirely.
    # Only "/v1/videos" is held. COLLECTIONS already names videos, images and
    # meshes, and the moment an image or mesh model is deployed its requests
    # will be submitted on arrival: no ticket, no position, no cancellation,
    # and no billing line. Harmless while video is the only deployment.
    # To extend: match the collection out of the path (see _job_id) rather
    # than hardcoding one, and give each collection its own busy predicate —
    # "all replicas busy" is the right admission rule for one-job-at-a-time
    # diffusion, but an image model that batches will want a depth limit
    # instead, so the predicate belongs with the model profile, not here.
    # Queue when the fleet is full — OR when anyone is already waiting. A
    # fresh POST taking a just-freed slot while tickets wait (high-priority
    # ones included) let the fast path starve the queue: every reaper-freed
    # slot went to the newest arrival, and (priority, arrival) ordering held
    # only among tickets, never against walk-ins.
    if not path.startswith("/v1/videos") or (not _all_busy() and not admission.waiting()):
        return None
    # `?queue=false` — the caller runs its own queue and wants this refused
    # rather than held. Two queues for the same work is the failure mode here:
    # ours would hold requests the caller's scheduler still believes are its
    # own to order, cancel and charge for. Refusing keeps one authority.
    if request.query_params.get("queue", "").lower() in ("false", "0", "no"):
        return JSONResponse(
            {
                "detail": "all replicas busy or queued work is ahead of you — "
                "queueing was declined",
                "free": bal.free_slots(),
                "waiting": len(admission.waiting()),
                "retry_hint_s": _retry_hint(),
            },
            status_code=503,
            headers={"Retry-After": str(_retry_hint())},
        )
    payload: dict = {}
    with contextlib.suppress(ValueError, TypeError):
        payload = json.loads(body) if body else {}
    # A client's junk must not become a 500. `{"priority": "high"}` raised
    # ValueError straight out of the handler; an unparseable priority is a
    # request the caller can fix, so it takes the documented default.
    priority = PRIORITY_NORMAL
    with contextlib.suppress(ValueError, TypeError):
        # CLAMPED to the documented scale. Unbounded, any caller sending
        # `priority: -1000` sorted ahead of everything forever — including
        # the service layer's own PRIORITY_HIGH — and the queue has no
        # ageing, so one such client starves it indefinitely.
        priority = min(
            max(int(payload.get("priority", PRIORITY_NORMAL)), PRIORITY_HIGH), PRIORITY_LOW
        )
    ticket = admission.enqueue(
        {"path": path, "body": payload, "headers": dict(headers)},
        priority=priority,
    )
    target_shape = payload.get("target") or {}
    _publish(
        "queued",
        ticket=ticket.ticket_id,
        priority=priority,
        task=payload.get("task"),
        prompt=str(payload.get("prompt") or "")[:120],
        seconds=target_shape.get("duration_seconds"),
    )
    # A slot may be free right now (that is HOW the ticket can exist: the
    # queue-jump guard above). Drain immediately so it goes to the queue
    # head in priority order — which may or may not be this ticket.
    await _safe_pump()
    position = admission.position(ticket.ticket_id)
    means = [s.mean_job_s for s in bal.stats.values() if s.mean_job_s]
    mean = sum(means) / len(means) if means else 0.0
    wait = admission.wait_estimate_s(
        ticket.ticket_id, mean, bal.capacity()["accepting_replicas"] or 1
    )
    # Shaped like a job so a client polls one way regardless of which side of
    # the billing line the request is currently on.
    return JSONResponse(
        {
            "id": ticket.ticket_id,
            "object": "video",
            "status": "queued",
            "queue_position": position,
            "estimated_wait_s": round(wait) if wait is not None else None,
            "billable": False,
        },
        status_code=202,
    )


async def _pump() -> None:
    """Submit waiting tickets while replicas are free.

    Called opportunistically rather than on a timer: every request that
    touches the balancer is a chance to drain, and a queue with nothing
    arriving has nothing to drain into.
    """
    while True:
        ticket = admission.next_ready()
        if ticket is None:
            return
        path = ticket.body.get("path")
        if not path:
            # Nothing to replay: terminal, and it says so.
            admission.mark_rejected(ticket.ticket_id, "ticket carries no request")
            continue
        target = bal.pick()
        if target is None or _all_busy():
            return
        # Claim BEFORE the await. The POST below suspends this coroutine, and
        # the pump is entered from six call sites — without the claim, a
        # concurrent pump saw the same WAITING ticket and submitted it to a
        # second replica (one generation, run and billed twice), and a cancel
        # arriving mid-POST "succeeded" while the job ran anyway.
        if not admission.claim(ticket.ticket_id):
            continue  # someone else has it; look at the next ticket
        began = bal.reserve(target)
        try:
            resp = await ctl.post(
                f"{target}{path}",
                headers=ticket.body.get("headers") or {},
                json=ticket.body.get("body") or {},
            )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            bal.release(target, failed=True)
            bal.mark(target, healthy=False)
            _publish("fleet", action="unhealthy", url=target)
            # CONNECT-phase failure: the request provably never reached the
            # replica, so the ticket safely goes back to the queue.
            global submit_requeued
            submit_requeued += 1
            admission.release_claim(ticket.ticket_id)
            return
        except httpx.HTTPError as exc:
            # AMBIGUOUS transport failure (read timeout, broken pipe): the
            # engine may have ACCEPTED and queued the job with only the
            # response lost. Requeuing here generated the same prompt twice
            # — a ghost with no occupancy, no owner, no meta (replica-0's
            # failed=41 was this, one ghost per timeout). The only honest
            # move is a LOUD terminal refusal: the caller decides whether
            # to resubmit, knowing a duplicate is possible.
            bal.release(target, failed=True)
            detail = (
                f"submission outcome unknown ({type(exc).__name__} against "
                f"{target}) — not retried: the engine may already hold this "
                f"job, and a blind retry can generate it twice"
            )
            global submit_ambiguous
            submit_ambiguous += 1
            admission.mark_rejected(ticket.ticket_id, detail)
            _publish("rejected", ticket=ticket.ticket_id, error=detail[:120])
            continue
        except BaseException:
            # Cancellation (or anything else non-HTTP) mid-POST: put back
            # BOTH pieces of held state, or the ticket freezes in SUBMITTING
            # and the reserved slot flips _all_busy() true on an idle fleet
            # — forever, since nothing else can see either leak.
            bal.release(target, failed=True)
            admission.release_claim(ticket.ticket_id)
            raise
        if resp.status_code >= 400:
            bal.release(target, failed=True)
            # NO RETRY, DELIBERATELY — and this is where one would go.
            # The common rejection is a bad request that would fail the same
            # on every replica; retrying would occupy the fleet in a loop.
            # To add one: retry only 5xx, only on a replica not yet tried,
            # bounded — needs attempts/tried fields on Ticket.
            detail = resp.text[:300] if resp.content else f"HTTP {resp.status_code}"
            global submit_rejected
            submit_rejected += 1
            admission.mark_rejected(ticket.ticket_id, detail)
            _publish("rejected", ticket=ticket.ticket_id, error=detail[:120])
            continue
        job_id, status = "", ""
        with contextlib.suppress(ValueError, KeyError, TypeError):
            payload = json.loads(resp.content)
            job_id = str(payload.get("id") or "")
            status = str(payload.get("status") or "")
        # FIRST, before any side effect. Everything below can raise, and
        # this is the only call that moves the ticket out of SUBMITTING —
        # which `prune` spares and `cancel` refuses. A failure after
        # `begin_job` froze the ticket there permanently: the customer
        # polled "queued" forever while being billed for a job that ran to
        # completion, and nothing could ever settle it.
        admission.mark_submitted(ticket.ticket_id, job_id, target)
        if job_id and status not in TERMINAL_STATUSES:
            # Same rule as the direct path: the slot becomes the job's.
            bal.release(target)
            bal.begin_job(job_id, target, collection=collection_of(path))
            _capture_meta(job_id, ticket.body.get("body") or {}, via="queue", backend=target)
            _publish("handover", ticket=ticket.ticket_id, job_id=job_id, backend=target)
        else:
            # Finished inside the call — a real, short job, counted as one.
            bal.record_inline(target, began, ok=True)


@app.get("/v1/queue")
async def queue_state() -> JSONResponse:
    """What is waiting, for operators and the service layer."""
    return JSONResponse(
        {
            **admission.snapshot(),
            "next": [
                {"ticket": t.ticket_id, "priority": t.priority, "position": position}
                for position, t in enumerate(admission.waiting()[:20])
            ],
        }
    )


@app.delete("/v1/queue/{ticket_id}")
async def cancel_ticket(ticket_id: str) -> JSONResponse:
    """Cancel work that has not reached a replica yet.

    Refused once submitted: the GPU time is already being spent, so stopping
    recovers nothing and the request is billable either way.
    """
    ok, why = admission.cancel(ticket_id)
    if ok:
        _publish("cancelled", ticket=ticket_id)
    # Unknown is 404 — the same verdict this id gets from every other
    # route. 409 is reserved for "exists, but already on a GPU".
    code = 200 if ok else (404 if why == "no such ticket" else 409)
    return JSONResponse(
        {"ticket": ticket_id, "cancelled": ok, "detail": why},
        status_code=code,
    )


# Younger than this and a job is left alone — it cannot plausibly have
# finished, and probing it buys nothing. NOT a staleness heuristic: closure
# is decided by the REPLICA saying terminal, never by age, so probing often
# is always safe. The old design probed only jobs past 3x-mean/1800s AND
# only when traffic happened to arrive — finished jobs whose clients died
# then blocked their slots for half an hour of a fully idle, fully blocked
# fleet.
RECONCILE_MIN_AGE_S = 60.0
# Consecutive failed probes (404 / unreachable) before a job is presumed
# lost and its slot released. A replica that crashed and restarted forgets
# its jobs; under a terminal-only close rule those slots were IMMORTAL —
# occupancy stuck, detach refused, the fleet shrinking with every crash.
JOB_LOST_STRIKES = 5
_job_strikes: dict[str, int] = {}


def _strike(job_id: str, url: str, why: str) -> None:
    _job_strikes[job_id] = _job_strikes.get(job_id, 0) + 1
    if _job_strikes[job_id] >= JOB_LOST_STRIKES:
        # Counted ONLY on a real close. `reaped` is the number a billing
        # layer watches for walked-away work, and sitting outside this
        # guard it was incremented once per concurrent reconcile pass over
        # the same job.
        if bal.end_job(job_id, ok=False):
            # "failed", not "lost": the wire vocabulary is the documented
            # four, and a consumer switching on it silently ignored every
            # strike-closed job. The nuance lives in closed_by/reason,
            # which are fields, not a fifth status.
            _note_terminal(job_id, "failed", closed_by="strikes", reason=why)
            bal.reaped += 1
        _job_strikes.pop(job_id, None)
        print(
            f"presumed lost after {JOB_LOST_STRIKES} probes: {job_id} on "
            f"{url} ({why}) — slot released",
            flush=True,
        )


async def _reap_stale_jobs() -> None:
    """Reconcile occupancy against the replicas — the owners of the truth.

    Every open job old enough to plausibly matter is asked about, every
    time. A job is only closed on a TERMINAL status from its own replica;
    one that is still running stays open, no matter how old. This runs on
    the maintenance timer, so a fleet with zero client traffic still frees
    finished slots and drains its queue: no external request is ever
    required for the platform to make progress on admitted work.
    """
    import asyncio
    import time as _time

    now = _time.monotonic()
    auth = {"authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    due = [j for j in bal.open_jobs() if now - j[2] >= RECONCILE_MIN_AGE_S]

    async def _probe(job_id: str, url: str, collection: str) -> None:
        try:
            r = await ctl.get(f"{url}/v1/{collection}/{job_id}", headers=auth)
        except httpx.HTTPError:
            _strike(job_id, url, "replica unreachable")
            return
        if r.status_code == 404:
            _strike(job_id, url, "unknown to its replica")
            return
        if r.status_code >= 400:
            return
        status = ""
        with contextlib.suppress(ValueError, TypeError, KeyError):
            status = str(dict(r.json()).get("status") or "")
        if status in TERMINAL_STATUSES and bal.end_job(job_id, ok=status == "completed"):
            # Everything below is guarded by end_job's dedupe: it is the
            # single winner across concurrent reconcile passes. `reaped`
            # sat outside it and was incremented once per pass, inflating
            # the counter a billing layer watches for walked-away work.
            _note_terminal(job_id, status, closed_by="reconciler")
            bal.reaped += 1
            print(
                f"reconciled {job_id} ({status}) on {url}: closed without a client poll",
                flush=True,
            )
        _job_strikes.pop(job_id, None)

    # CONCURRENTLY. Sequentially, each probe could take the full control
    # timeout, so one wedged replica turned a 10-second beat into minutes —
    # making JOB_LOST_STRIKES mean "5 x however long the beat took" and
    # stalling reconcile, revive, pump and prune together on the slowest
    # replica. That is exactly what CTL_TIMEOUT's docstring says must not
    # happen.
    await asyncio.gather(*(_probe(j, u, c) for j, u, _t, c in due))

    # Strikes belong to OPEN jobs only. Entries were removed on threshold or
    # on a healthy answer, so a job closed by any other path (client poll,
    # SSE, DELETE) left its counter behind forever — an unbounded dict on a
    # long-lived balancer, and stale strikes re-applied to a recycled id.
    live = {j for j, _u, _t, _c in bal.open_jobs()}
    for gone in [j for j in _job_strikes if j not in live]:
        _job_strikes.pop(gone, None)

    # A replica that keeps ANSWERING for work it will never finish is
    # invisible to everything above: closure is terminal-only, and strikes
    # fire only on 404/unreachable. The client gives up at its own floor
    # and fails the item, while the slot stays occupied forever — a
    # permanent capacity loss per wedge. This rule existed as dead code in
    # balance.py the whole time.
    for job_id, url in bal.stale_jobs():
        if not bal.end_job(job_id, ok=False):
            continue
        _note_terminal(
            job_id, "failed", closed_by="stale", reason="exceeded this fleet's overrun limit"
        )
        bal.reaped += 1
        print(f"stale: {job_id} on {url} exceeded the overrun limit — slot released", flush=True)
        # Tell the replica too, so it stops spending on work nobody awaits.
        with contextlib.suppress(httpx.HTTPError):
            await ctl.delete(f"{url}/v1/videos/{job_id}", headers=auth)


async def _revive_backends() -> None:
    """Re-arm backends that failed once and recovered.

    healthy=False was a ONE-WAY latch: nothing ever probed an unhealthy
    backend again, so a 30-second network blip permanently removed replicas
    — a fleet-wide blip starved every ticket forever with all GPUs idle,
    recoverable only by operator surgery.
    """
    for st in list(bal.stats.values()):
        if st.healthy:
            continue
        with contextlib.suppress(httpx.HTTPError):
            r = await ctl.get(f"{st.url}/health")
            if r.status_code < 500:
                bal.mark(st.url, healthy=True)
                _publish("fleet", action="revived", url=st.url)
                print(f"revived {st.url}: health probe answered", flush=True)


# ---------------------------------------------------------------------------
# Event firehose: every lifecycle transition, published once, streamed to any
# number of subscribers. Dashboards and other consumers subscribe instead of
# polling; a slow consumer can NEVER back-pressure the platform — its queue
# drops oldest and receives a `gap` marker, and the bounded replay ring lets
# a reconnecting subscriber catch up with `?since=<seq>`.

# Identifies THIS process lifetime in every event and in the audit log.
BOOT_ID = os.urandom(4).hex()

EVENTS_RING_MAX = 500
# deque(maxlen=...) IS a ring: the hand-rolled list + `del ring[:-500]` did
# an O(n) slice-delete on every published event, and the same hand-rolled
# shape appeared again for `_recent` with `insert(0, ...)`, which is O(n)
# too. itertools.count is the sequence generator, already used in the two
# sibling modules.
_events_ring: deque[dict] = deque(maxlen=EVENTS_RING_MAX)
_event_seq = itertools.count(1)
_subscribers: set = set()


# Persistent audit trail: the SAME events the firehose streams, appended to
# disk. The ring answers "what just happened"; this answers "what happened
# last Tuesday" — the question every duplicate/misattribution incident asks
# and the 500-event ring could not. JSONL, append-only, rotated once at
# _AUDIT_MAX_BYTES (current + .1 = bounded at ~2x).
AUDIT_LOG = os.environ.get(
    "DEEPME_AUDIT_LOG", str(Path(WORK_DIR) / "audit.jsonl") if WORK_DIR else ""
)
_AUDIT_MAX_BYTES = 64 * 1024 * 1024


def _audit_write(evt: dict) -> None:
    if not AUDIT_LOG:
        return
    with contextlib.suppress(OSError):
        path = Path(AUDIT_LOG)
        if path.exists() and path.stat().st_size > _AUDIT_MAX_BYTES:
            path.replace(f"{AUDIT_LOG}.1")
        with path.open("a") as f:
            f.write(json.dumps(evt) + "\n")


def _publish(kind: str, **fields) -> None:
    import asyncio
    import time as _time

    # `boot` partitions the log. `seq` restarts at 1 on every process start,
    # so across a restart — the event that destroys waiting tickets and
    # re-mints ticket ids — sequence numbers COLLIDE, and a reconstruction
    # cannot order or separate the two lifetimes.
    evt = {
        "seq": next(_event_seq),
        "boot": BOOT_ID,
        "ts": round(_time.time(), 3),
        "type": kind,
        **fields,
    }
    _audit_write(evt)
    _events_ring.append(evt)
    for q in list(_subscribers):
        try:
            q.put_nowait(evt)
        except asyncio.QueueFull:
            with contextlib.suppress(Exception):
                q.get_nowait()
                q.put_nowait({"seq": evt["seq"], "ts": evt["ts"], "type": "gap"})


# job id -> request metadata, captured at submission. The BALANCER is the
# only component that sees every request body, and the engine's own job
# records carry no prompt/task — without this an operator can see THAT four
# jobs run but never WHAT they are.
_job_meta: dict[str, dict] = {}
# Last N terminal jobs, newest first — the operator's "what just happened".
_RECENT_MAX = 50
_recent: deque[dict] = deque(maxlen=_RECENT_MAX)


def _capture_meta(job_id: str, payload: dict, *, via: str, backend: str = "") -> None:
    if not job_id or not isinstance(payload, dict):
        return
    import hashlib

    target = payload.get("target") or {}
    prompt = str(payload.get("prompt") or "")
    _job_meta[job_id] = {
        "task": payload.get("task"),
        "prompt": prompt[:120],
        # The two fields a duplicate incident turns on. Two items with the
        # same 120-char prompt prefix and DIFFERENT seeds were
        # indistinguishable in the audit log — which is exactly the
        # question `batch dedupe` exists to answer after the fact.
        "seed": payload.get("seed"),
        "prompt_sha": hashlib.sha1(prompt.encode()).hexdigest()[:12],
        "seconds": target.get("duration_seconds") or payload.get("seconds"),
        "priority": payload.get("priority"),
        "via": via,
        # WHERE it runs, recorded at start. Previously the job->replica
        # binding first appeared at `terminal`, so a job that never closed
        # cleanly had no recorded owner at all.
        "backend": backend or bal.owner(job_id) or "",
    }
    while len(_job_meta) > 2000:
        _job_meta.pop(next(iter(_job_meta)))
    _publish("started", job_id=job_id, **_job_meta[job_id])


def _note_terminal(job_id: str, status: str, *, closed_by: str, reason: str = "") -> None:
    import time as _time

    entry = {
        "job_id": job_id,
        "status": status,
        "closed_by": closed_by,
        # WHY, when the closer knows: a strike's "404 vs unreachable" is the
        # whole diagnosis and it used to live only in the journal.
        **({"reason": reason} if reason else {}),
        # WHERE it ran — "failed (strikes) @8100" is a diagnosis, "failed"
        # alone is a mystery. The owner map outlives occupancy, so this is
        # still known at close time.
        "backend": bal.owner(job_id),
        "finished_at": _time.time(),
        **_job_meta.pop(job_id, {}),
    }
    _recent.appendleft(entry)
    _publish("terminal", **{k: v for k, v in entry.items() if k != "finished_at"})


# job id -> unix time of FIRST content fetch. Persisted with membership so a
# restart keeps the 1h re-fetch windows honest.
_fetched: dict[str, float] = {}
retention_deleted = 0


def _retention_candidates(
    entries: list[tuple[str, str, float]],
    fetched: dict[str, float],
    open_ids: set[str],
    now: float,
    ttl_s: float,
    fetched_ttl_s: float,
) -> list[str]:
    """Pure policy: which files are past their window.

    entries: (path, artifact id or '' for input files, mtime).
      * never fetched -> keep ttl_s from mtime (the initial-fetch window)
      * fetched       -> keep fetched_ttl_s from the FIRST fetch
      * NEVER a file whose job is still open — running work is not garbage.
    """
    out = []
    for path, art_id, mtime in entries:
        if art_id and art_id in open_ids:
            continue
        first_fetch = fetched.get(art_id) if art_id else None
        expiry = (first_fetch + fetched_ttl_s) if first_fetch else (mtime + ttl_s)
        if now > expiry:
            out.append(path)
    return out


def _artifact_id(path) -> str:
    """The job id a file belongs to, from its name or its directory.

    Engines write `{id}.mp4`, `{id}.0.mp4`-style multi-outputs, and
    per-job subdirectories. Matching `p.stem` exactly missed everything
    but the first form, so fetched/open lookups silently degraded to the
    6h mtime window.
    """
    # A per-job SUBDIRECTORY names the job; the file inside it usually does
    # not (`outputs/<job_id>/output.mp4`). Taking the stem unconditionally
    # gave every such job the artifact id "output", so no job's id ever
    # matched the open set — running work was not protected from deletion —
    # and the first fetch of ANY job collapsed every other job's 6h window
    # to 1h. Prefer the directory whenever the file is nested.
    outputs = Path(WORK_DIR) / "outputs" if WORK_DIR else None
    if outputs is not None and path.parent != outputs and path.parent.name:
        return path.parent.name
    return path.stem.partition(".")[0] or path.parent.name


def _scan_dir(root, *, with_ids: bool) -> list[tuple[str, str, float]]:
    """(path, artifact id, mtime) for every file under `root`.

    A file that vanishes mid-walk drops out; it never aborts the scan.
    """
    out: list[tuple[str, str, float]] = []
    try:
        walk = list(root.rglob("*"))
    except OSError:
        return out
    for p in walk:
        with contextlib.suppress(OSError):
            if p.is_file():
                out.append((str(p), _artifact_id(p) if with_ids else "", p.stat().st_mtime))
    return out


def _delete_artifacts(job_id: str) -> int:
    """Remove every file belonging to a job. Returns how many went.

    Called the instant a download completes: an artifact that has been
    delivered has no remaining purpose, and keeping it is what allows a
    later request to be served someone else's video.
    """
    if not (WORK_DIR and job_id):
        return 0
    outputs = Path(WORK_DIR) / "outputs"
    gone = 0
    for path, art_id, _mtime in _scan_dir(outputs, with_ids=True):
        if art_id != job_id:
            continue
        with contextlib.suppress(OSError):
            Path(path).unlink()
            gone += 1
            print(f"delivered and reclaimed: {path}", flush=True)
    # A now-empty per-job directory goes too, or the tree grows forever.
    with contextlib.suppress(OSError):
        job_dir = outputs / job_id
        if job_dir.is_dir() and not any(job_dir.iterdir()):
            job_dir.rmdir()
    return gone


def _retention_plan() -> list[str]:
    """Which files are expired. PURE and thread-safe: reads the filesystem
    and a snapshot of shared state, mutates nothing, publishes nothing.

    The scan is the slow part (a full recursive walk with a stat per file),
    so it belongs off the event loop — but everything it used to do
    afterwards did not. `_publish` appends to a deque the SSE replay reader
    iterates and calls `Queue.put_nowait`, which is not thread-safe: from a
    worker thread it can wake a loop waiter incorrectly, and an append that
    evicts under `maxlen` raises `deque mutated during iteration` in any
    dashboard reconnecting with `?since=`. So: plan here, apply on the loop.
    """
    import pathlib
    import time as _time

    if not WORK_DIR:
        return []
    entries: list[tuple[str, str, float]] = []
    outputs = pathlib.Path(WORK_DIR) / "outputs"
    inputs = pathlib.Path(WORK_DIR) / "inputs"
    # RECURSIVE: engines nest outputs in per-replica/per-job subdirs, and a
    # flat iterdir() skipped directories entirely — the 6h/1h policy ran
    # against a directory level nothing wrote to, which is how fetched
    # files survived for days.
    #
    # Suppression is PER FILE, deliberately. Wrapping the whole walk meant
    # one vanishing file (routine: an engine writes and moves it while the
    # scan runs) abandoned the entire traversal, so a recurring race
    # silently disabled retention altogether while /balancer/stats kept
    # reporting it enabled.
    entries.extend(_scan_dir(outputs, with_ids=True))
    entries.extend(_scan_dir(inputs, with_ids=False))
    open_ids = {j for j, _u, _t, _c in bal.open_jobs()}
    return _retention_candidates(
        entries,
        dict(_fetched),  # a SNAPSHOT: the loop may be mutating the original
        open_ids,
        _time.time(),
        RETENTION_TTL_S,
        RETENTION_FETCHED_TTL_S,
    )


def _retention_apply(doomed: list[str]) -> None:
    """Delete the planned files and record it. LOOP-SIDE ONLY — it
    publishes events and mutates `_fetched`."""
    global retention_deleted
    reclaimed = 0
    for path in doomed:
        with contextlib.suppress(OSError):
            Path(path).unlink()
            retention_deleted += 1
            reclaimed += 1
            print(f"retention: reclaimed {path}", flush=True)
    if reclaimed:
        _publish("retention", reclaimed=reclaimed, paths=doomed[:50])
    # Fetch records for deleted artifacts have done their job — pruned via
    # the SAME id extraction the scan used, or records leak forever.
    kept = {_artifact_id(Path(p)) for p in doomed}
    dropped = [a for a in _fetched if a in kept]
    for art_id in dropped:
        _fetched.pop(art_id, None)
    if reclaimed or dropped:
        with contextlib.suppress(OSError):
            _persist_membership()


def _retention_pass() -> None:
    """Plan and apply in one call — for tests and any synchronous caller."""
    _retention_apply(_retention_plan())


async def _recover_occupancy(*, boot: bool = True) -> int:
    """Rebuild occupancy from replica truth. Returns the number adopted.

    Runs at boot AND on a slow beat (see _ORPHAN_SWEEP_EVERY_BEATS). The
    periodic pass is what closes the ghost job: a generation the engine
    ACCEPTED but we never recorded, because our POST timed out at 60s while
    the engine kept going, or because an ambiguous submit was rejected after
    the work had already started. Those jobs occupy a GPU that pick() still
    believes is free, so the replica gets double-booked and the second job
    queues inside the engine where nothing can cancel or reorder it. Only
    boot recovery ever looked for them, so a ghost born at 03:00 stayed a
    ghost until the next restart.

    Adoption is safe to repeat: begin_job on an id we already hold is a
    no-op rebinding to the same replica, and TERMINAL ids are never bound
    (see the note below) so a replica's history cannot resurrect anything.

    A restart empties the job map while the replicas keep generating. Left
    unrecovered, pick() double-books busy replicas and _all_busy() reads
    false — waving the entire former queue into engine-internal FIFOs where
    nothing can reorder or cancel it. The replicas own the truth, so ask
    them: every non-terminal job they report re-opens its slot.

    Listing is /v1/videos only — the sole modality deployed; extend per
    collection when another ships (see the _admit latent-gap note).
    """
    import asyncio

    auth = {"authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    known = {j for j, _u, _t, _c in bal.open_jobs()}
    adopted = 0

    async def _one(st) -> None:
        nonlocal adopted
        with contextlib.suppress(httpx.HTTPError, ValueError, TypeError, KeyError):
            r = await ctl.get(f"{st.url}/v1/videos", headers=auth, timeout=RECOVERY_TIMEOUT)
            if r.status_code >= 400:
                return
            for job in dict(r.json()).get("data", []) or []:
                job_id = str(job.get("id") or "")
                status = str(job.get("status") or "")
                if not job_id:
                    continue
                if status not in TERMINAL_STATUSES:
                    if job_id in known:
                        continue  # already accounted for; nothing to adopt
                    bal.begin_job(job_id, st.url)
                    adopted += 1
                    if boot:
                        print(f"recovered occupancy: {job_id} running on {st.url}", flush=True)
                    else:
                        # NOT routine. A ghost at runtime means a generation
                        # was burning a GPU that we believed was free, so it
                        # is worth an event and a counter, not just a log.
                        global orphans_adopted
                        orphans_adopted += 1
                        _publish("job", action="orphan-adopted", job=job_id, url=st.url)
                        print(
                            f"adopted orphan {job_id} on {st.url}: running but untracked",
                            flush=True,
                        )
                # TERMINAL ids are deliberately NOT re-bound here. A replica's
                # /v1/videos is its whole HISTORY, and engine ids are minted
                # PER REPLICA and re-minted after an engine restart — so
                # adopting every terminal id from every replica overwrote live
                # bindings with stale ones whenever two replicas had ever used
                # the same number. The owner map then pointed at the replica
                # holding the OLD artifact, /content streamed it, and the
                # binding was persisted and taught to the gateway. A
                # completed-but-unfetched job is resolved correctly by the
                # VERIFIED probe instead, which requires the body to name the
                # id.

    # CONCURRENTLY and on a SHORT timeout. This runs before the lifespan
    # yields, so uvicorn does not bind the port until it returns: probing
    # replicas one at a time at the 30s control timeout meant a fleet of
    # slow-answering replicas held the balancer in connection-refused for
    # minutes, which is how a restart turned into an outage. Recovery is
    # best-effort — the maintenance beat reconciles anything missed within
    # one beat — so a replica that cannot answer FAST is skipped, not
    # waited on.
    await asyncio.gather(*(_one(st) for st in list(bal.stats.values())))
    return adopted


@app.get("/v1/capacity")
async def capacity() -> JSONResponse:
    """Can I send work right now, and how much?

    Built for a caller that keeps its OWN priority queue and only submits when
    a GPU is free. That caller needs one number per fleet — `free` — plus what
    is running, so it can plan the next minute.

    Deliberately cheap: answered from memory, with no call to any replica and
    no stale-job reaping, so polling it once a second costs nothing. Contrast
    /balancer/stats, which reaps before reporting and is a diagnostic, not a
    scheduling signal.
    """
    snapshot = bal.capacity()
    snapshot["queued"] = len(admission.waiting())
    snapshot["accepting"] = snapshot["free"] > 0
    # Disk is part of capacity: outputs accumulate on the work volume and a
    # full disk fails GENERATIONS, so a scheduler must see it coming.
    with contextlib.suppress(OSError):
        import shutil

        du = shutil.disk_usage(os.environ.get("DEEPME_WORK_DISK", "."))
        snapshot["disk"] = {
            "free_gb": round(du.free / 1e9, 1),
            "used_pct": round(du.used / du.total * 100, 1),
            # Above the watermark, stop selling spot and start sweeping.
            "watermark_exceeded": du.used / du.total > 0.85,
        }
    return JSONResponse(snapshot)


# ---------------------------------------------------------------------------
# Fleet membership at runtime. A mix change (3+1 -> 4+0) must not restart
# this process: a restart discards every waiting ticket and the owner map,
# punishing queued customers for an operation that is about replicas.
# Membership changes are persisted next to the process so a crash-restart
# resumes the CURRENT fleet, not the one frozen in at deploy time.

# Per balancer, because every profile's balancer shares one WorkingDirectory
# — a single shared file would have h3-video and h3-ref overwriting each
# other's fleets.
MEMBERSHIP_FILE = f"backends-{os.environ.get('BALANCER_NAME', 'default')}.json"


def _persist_membership() -> None:
    """Atomically: write-then-rename. A crash mid-write must corrupt nothing
    — a truncated file silently reverted the fleet to the deploy-time list,
    which is exactly the drift this file exists to prevent."""
    import pathlib

    state = {
        "backends": bal.backend_urls(),
        # Draining survives a restart too: a crash-restart that forgot a
        # drain put the replica back in rotation, and the operator's planned
        # unit stop then killed the 40-minute job pick() had just handed it.
        "draining": [u for u, s in bal.stats.items() if s.draining],
        # First-fetch times survive too, or a restart would silently reset
        # every artifact's 1h re-fetch window back to the 6h global one.
        "fetched": dict(_fetched),
        # Job ownership survives: a restart that forgot who owns what sent
        # every stranded poll through the owner probe — the misattribution
        # path this file now closes at the source.
        "owners": bal.owners_snapshot(),
    }
    tmp = pathlib.Path(f"{MEMBERSHIP_FILE}.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(MEMBERSHIP_FILE)


def _restore_membership() -> None:
    """Prefer the persisted membership over the deploy-time env list.

    `balancer start` deletes the file (a fresh freeze); anything else that
    restarts this process — a crash, a reboot — comes back with the fleet
    as it actually was.
    """
    import pathlib

    path = pathlib.Path(MEMBERSHIP_FILE)
    if not path.exists():
        return
    with contextlib.suppress(ValueError, OSError):
        state = json.loads(path.read_text())
        # Legacy format: a bare list of URLs.
        urls = state if isinstance(state, list) else state.get("backends", [])
        draining = set() if isinstance(state, list) else set(state.get("draining", []))
        for url in urls:
            bal.attach(str(url))
        for url in list(bal.stats):
            if url not in urls:
                bal.stats.pop(url)
        for url in draining:
            bal.drain(str(url))
        if not isinstance(state, list):
            _fetched.update({str(k): float(v) for k, v in (state.get("fetched") or {}).items()})
            for job_id, url in (state.get("owners") or {}).items():
                if str(url) in bal.stats:
                    bal.remember(str(job_id), str(url))


_restore_membership()


@app.post("/balancer/backends", dependencies=[Depends(require_key)])
async def attach_backend(request: Request) -> JSONResponse:
    """Attach a replica at runtime — probed first, so a typo cannot join.

    An unprobed attach would put a dead URL into pick() rotation and turn
    some fraction of submissions into 502s until someone noticed.
    """
    body = {}
    with contextlib.suppress(ValueError, TypeError):
        body = json.loads(await request.body())
    url = str(body.get("url") or "").rstrip("/")
    if not url.startswith("http"):
        return JSONResponse({"detail": "body must carry {url}"}, status_code=400)
    try:
        r = await ctl.get(f"{url}/health", timeout=10)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        return JSONResponse(
            {"detail": f"{url} failed its health probe, not attached: {exc}"},
            status_code=400,
        )
    fresh = bal.attach(url)
    _publish("fleet", action="attached", url=url)
    _persist_membership()
    await _safe_pump()  # a new slot may free queued work immediately
    return JSONResponse(
        {"url": url, "attached": True, "was_new": fresh, "backends": bal.backend_urls()}
    )


@app.post("/balancer/backends/drain", dependencies=[Depends(require_key)])
async def drain_backend(request: Request) -> JSONResponse:
    """Take a replica out of rotation without touching its running work.

    The drain completes when its inflight reaches 0 (watch /balancer/stats);
    only then is a detach — and the unit stop behind it — safe.
    """
    body = {}
    with contextlib.suppress(ValueError, TypeError):
        body = json.loads(await request.body())
    url = str(body.get("url") or "").rstrip("/")
    if not bal.drain(url):
        return JSONResponse({"detail": f"{url} is not attached"}, status_code=404)
    _persist_membership()  # a restart must not put it back in rotation
    _publish("fleet", action="drained", url=url)
    st = bal.stats[url]
    return JSONResponse({"url": url, "draining": True, "inflight": st.inflight})


@app.post("/balancer/backends/undrain", dependencies=[Depends(require_key)])
async def undrain_backend(request: Request) -> JSONResponse:
    """Put a drained replica back in rotation.

    The recovery path for an aborted resize: without it, a drain left over
    from a Ctrl-C'd `fleet switch` starves that replica forever — visibly
    "up", inflight 0, receiving nothing.
    """
    body = {}
    with contextlib.suppress(ValueError, TypeError):
        body = json.loads(await request.body())
    url = str(body.get("url") or "").rstrip("/")
    if not bal.undrain(url):
        return JSONResponse({"detail": f"{url} is not attached"}, status_code=404)
    _persist_membership()
    _publish("fleet", action="undrained", url=url)
    await _safe_pump()  # a slot just came back
    return JSONResponse({"url": url, "draining": False})


@app.delete("/balancer/backends", dependencies=[Depends(require_key)])
async def detach_backend(request: Request) -> JSONResponse:
    url = str(request.query_params.get("url") or "").rstrip("/")
    ok, why = bal.detach(url)
    if not ok:
        # 404 only for "no such backend"; every other refusal is a conflict
        # with work in progress and deserves a retry, not a shrug.
        return JSONResponse(
            {"detail": f"{url}: {why}"},
            status_code=404 if why == "not attached" else 409,
        )
    _persist_membership()
    _publish("fleet", action="detached", url=url)
    return JSONResponse({"url": url, "detached": True, "backends": bal.backend_urls()})


@app.get("/v1/events")
async def event_firehose(request: Request) -> Response:
    """Every lifecycle transition, as SSE — subscribe instead of polling.

    `?since=<seq>` replays the bounded ring first, so a reconnecting
    dashboard misses nothing recent; a `gap` event means this SUBSCRIBER
    fell behind and dropped (the platform never blocks on a slow consumer).
    Heartbeat comments every 15s keep proxies from killing idle streams.
    """
    import asyncio

    since = limit = 0
    with contextlib.suppress(ValueError, TypeError):
        since = int(request.query_params.get("since") or 0)
        limit = int(request.query_params.get("limit") or 0)
    q: asyncio.Queue = asyncio.Queue(maxsize=256)

    async def stream():
        _subscribers.add(q)
        sent = 0
        try:
            for evt in [e for e in _events_ring if e["seq"] > since]:
                yield _sse(evt)
                sent += 1
                if limit and sent >= limit:
                    return
            while True:
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=15)
                    yield _sse(evt)
                    sent += 1
                    # `limit` makes the stream a bounded tap — curl-friendly,
                    # and the only way a transport that cannot signal
                    # disconnect (in-process tests included) ends it cleanly.
                    if limit and sent >= limit:
                        return
                except TimeoutError:
                    yield ": heartbeat\n\n"
                if await request.is_disconnected():
                    return
        finally:
            _subscribers.discard(q)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/v1/jobs")
async def jobs() -> JSONResponse:
    """The operator's global view: every job this fleet knows about, WITH
    what it is — running (parameters + elapsed + ETA), queued (position +
    priority), and the recent-terminal ring. Batch clients see their own
    items; operators own the SYSTEM, and this answers "what is running
    right now" without owning any client.
    """
    running = [{**j, **_job_meta.get(j["job_id"], {})} for j in bal.running_jobs()]
    queued = []
    # ONE sorted pass, enumerated. Asking `position()` per ticket re-sorted
    # the entire queue each time — O(n^2 log n) with no await anywhere in
    # the loop, so at the 5000-ticket prune bound a single dashboard poll
    # pinned the event loop for seconds: heartbeats stopped, /health timed
    # out, and the gateway marked a perfectly healthy fleet down.
    for position, t in enumerate(admission.waiting()):
        body = t.body.get("body") or {}
        target = body.get("target") or {}
        queued.append(
            {
                "ticket": t.ticket_id,
                "position": position,
                "priority": t.priority,
                "task": body.get("task"),
                "prompt": str(body.get("prompt") or "")[:120],
                "seconds": target.get("duration_seconds"),
            }
        )
    return JSONResponse(
        {
            "running": running,
            "queued": queued,
            "recent": list(_recent),
        }
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus text exposition — point any scraper here. No deps, no
    push agents: the numbers a dashboard needs, in the format it eats."""
    cap = bal.capacity()
    lines = [
        "# TYPE deepme_replicas gauge",
        f"deepme_replicas {cap['replicas']}",
        f"deepme_replicas_healthy {cap['healthy']}",
        f"deepme_replicas_accepting {cap['accepting_replicas']}",
        f"deepme_replicas_draining {cap['draining']}",
        f"deepme_slots_free {cap['free']}",
        f"deepme_jobs_running {cap['busy']}",
        f"deepme_queue_waiting {len(admission.waiting())}",
        "# TYPE deepme_jobs_reaped_total counter",
        f"deepme_jobs_reaped_total {bal.reaped}",
        "# TYPE deepme_retention_deleted_total counter",
        f"deepme_retention_deleted_total {retention_deleted}",
        # A dead maintenance loop looks exactly like a healthy idle one from
        # the outside — unless it is counted.
        "# TYPE deepme_maintenance_errors_total counter",
        f"deepme_maintenance_errors_total {maintenance_errors}",
        "# TYPE deepme_retention_errors_total counter",
        f"deepme_retention_errors_total {retention_errors}",
        # Submission outcomes BY CAUSE — "submissions are failing" and
        # "which cause" were previously the same unanswerable question.
        "# TYPE deepme_submit_rejected_total counter",
        f"deepme_submit_rejected_total {submit_rejected}",
        "# TYPE deepme_submit_ambiguous_total counter",
        f"deepme_submit_ambiguous_total {submit_ambiguous}",
        "# TYPE deepme_submit_requeued_total counter",
        f"deepme_submit_requeued_total {submit_requeued}",
        # Identity integrity: both of these firing means the platform caught
        # itself about to serve the wrong job.
        "# TYPE deepme_owner_conflicts_total counter",
        f"deepme_owner_conflicts_total {owner_conflicts}",
        "# TYPE deepme_owner_rejections_total counter",
        f"deepme_owner_rejections_total {owner_rejections}",
        "# HELP deepme_orphans_adopted_total Running jobs found untracked by the sweep.",
        "# TYPE deepme_orphans_adopted_total counter",
        f"deepme_orphans_adopted_total {orphans_adopted}",
        "# HELP deepme_submit_unservable_total Waiting tickets refused: fleet could never serve them.",
        "# TYPE deepme_submit_unservable_total counter",
        f"deepme_submit_unservable_total {submit_unservable}",
        "# TYPE deepme_queue_tickets gauge",
        f"deepme_queue_tickets {admission.snapshot()['tickets']}",
    ]
    # PER REPLICA, so a single degrading replica is alertable — these exist
    # in /balancer/stats but never reached a scrape target.
    for b in bal.snapshot()["backends"]:
        label = f'{{replica="{b["url"]}"}}'
        lines += [
            f"deepme_replica_completed_total{label} {b['completed']}",
            f"deepme_replica_failed_total{label} {b['failed']}",
            f"deepme_replica_inflight{label} {b['inflight']}",
        ]
    if cap.get("mean_job_s"):
        lines += [
            "# TYPE deepme_job_mean_seconds gauge",
            f"deepme_job_mean_seconds {cap['mean_job_s']}",
        ]
    with contextlib.suppress(OSError):
        import shutil

        du = shutil.disk_usage(os.environ.get("DEEPME_WORK_DISK", "."))
        lines += [
            "# TYPE deepme_disk_used_ratio gauge",
            f"deepme_disk_used_ratio {du.used / du.total:.4f}",
        ]
    return Response("\n".join(lines) + "\n", media_type="text/plain")


@app.get("/balancer/stats", dependencies=[Depends(require_key)])
async def stats() -> JSONResponse:
    # Reap before reporting, so the numbers an operator (or a billing layer)
    # reads are not inflated by clients that walked away.
    await _reap_stale_jobs()
    snap = bal.snapshot()
    snap["retention"] = {
        "deleted_total": retention_deleted,
        "tracked_fetches": len(_fetched),
        "ttl_s": RETENTION_TTL_S,
        "fetched_ttl_s": RETENTION_FETCHED_TTL_S,
        "enabled": bool(WORK_DIR),
    }
    # Health of the platform's own machinery, where an OPERATOR will look.
    # These counters exist because a dead maintenance loop is indis-
    # tinguishable from a healthy idle one — but they lived only in
    # /metrics, which no command reads, so in practice nobody could see
    # them. Same for the queue: the runbook told operators to curl it.
    snap["queue"] = admission.snapshot()
    snap["maintenance"] = {
        "errors_total": maintenance_errors,
        "retention_errors_total": retention_errors,
        "audit_enabled": bool(AUDIT_LOG),
        "boot": BOOT_ID,
    }
    return JSONResponse(snap)


@app.get("/health")
async def health() -> JSONResponse:
    healthy = [s.url for s in bal.stats.values() if s.healthy]
    return JSONResponse(
        {
            "status": "ok" if healthy else "no healthy backends",
            "healthy": len(healthy),
            "total": len(bal.stats),
        },
        status_code=200 if healthy else 503,
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy(path: str, request: Request) -> Response:
    full = "/" + path
    headers = auth_headers(request, API_KEY)
    body = await request.body()

    # 1. A request about an EXISTING job goes to its owner, wherever it is.
    job_id = job_id_of(full)

    # A ticket is not a job yet, but a 202 hands the caller an id shaped like
    # one and invites it to poll. Until a replica takes the work there is
    # nothing to forward, so answer from the queue; once there is, rewrite the
    # path to the real job so status AND content follow through untouched.
    if job_id and job_id.startswith("q_") and request.method in ("GET", "DELETE"):
        ticket = admission.get(job_id)
        if ticket is None:
            return JSONResponse({"detail": f"no such ticket {job_id}"}, status_code=404)
        # A SUB-RESOURCE (/content) on a ticket that has no job yet has no
        # bytes to serve. This branch used to answer every q_ path with a
        # JSON status body at 200 — so a client fetching /content early
        # streamed that JSON into its .mp4, renamed it, and recorded a
        # successful deliverable. There is no status code a body-shaped
        # answer is correct for here.
        if request.method == "GET" and not full.rstrip("/").endswith(job_id) and not ticket.job_id:
            return JSONResponse(
                {
                    "detail": f"{job_id} has not reached a replica yet — "
                    f"no artifact exists; poll {job_id} until it reports "
                    f"completed",
                    "status": "queued",
                },
                status_code=409,
            )
        if request.method == "DELETE":
            ok, why = admission.cancel(job_id)
            return JSONResponse(
                {"id": job_id, "status": "cancelled" if ok else ticket.state, "detail": why},
                status_code=200 if ok else 409,
            )
        if ticket.state == REJECTED:
            # Terminal, and it must SAY so: retiring a refused ticket as
            # 'submitted' had callers polling "queued" forever for work that
            # never ran and never will.
            return JSONResponse(
                {
                    "id": job_id,
                    "object": "video",
                    "status": "failed",
                    "error": {"message": ticket.error or "the replica refused it"},
                    "billable": False,
                }
            )
        if not ticket.job_id:
            if ticket.state == SUBMITTED:
                # Accepted and finished inside the submission call, with no
                # artifact id minted — nothing further to follow.
                return JSONResponse(
                    {
                        "id": job_id,
                        "object": "video",
                        "status": "completed",
                        "billable": True,
                    }
                )
            await _safe_pump()  # a replica may have freed since the last poll
            position = admission.position(job_id)
            return JSONResponse(
                {
                    "id": job_id,
                    "object": "video",
                    "status": "cancelled" if ticket.state == CANCELLED else "queued",
                    "queue_position": position,
                    # THE EVIDENCE A WAITING CLIENT NEEDS. Without this, a poll
                    # of a ticket on a dead fleet is byte-identical to one that
                    # is next in line behind a long generation: same 200, same
                    # "queued", same position. A client cannot distinguish "wait,
                    # this is working" from "there is no fleet and never will
                    # be", so it waits forever — and it is right to, because
                    # queueing is legitimate and killing queued work on a timer
                    # is what this platform refuses to do. `accepting_replicas:
                    # 0` alongside a position is unambiguous: nothing can ever
                    # take this. capacity() is pure in-memory, so this is free.
                    "accepting_replicas": bal.capacity()["accepting_replicas"],
                    "billable": False,
                }
            )
        full = full.replace(job_id, ticket.job_id, 1)
        job_id = ticket.job_id

    if job_id and request.method == "DELETE" and is_content(full):
        # THE ACK. "I have persisted these bytes; you may reclaim them."
        #
        # Distinct from DELETE on the job itself, which CANCELS work — acking
        # a delivered artifact must never be mistaken for cancelling the job
        # that produced it, so this is handled before any cancel path and is
        # never forwarded to the replica.
        #
        # Idempotent and safe to call for an unknown id: the client may retry
        # an ack whose response was lost, and a second one must not error.
        # Deliberately NOT gated on _fetched — an operator or a caller that
        # already holds the file may reclaim it either way.
        removed = _delete_artifacts(job_id)
        if removed:
            _publish("reclaimed", job_id=job_id, files=removed, reason="acked")
        return JSONResponse(
            {"id": job_id, "reclaimed": removed, "object": "video.ack"},
            status_code=200,
        )

    if job_id and request.method in ("GET", "DELETE"):
        target = bal.owner(job_id)
        # VERIFY the remembered binding before serving BYTES. The verified
        # probe used to run only on a cache MISS — the rare path — while the
        # common remembered path streamed from whatever the map said, with
        # nothing checking that the replica owns the id or that the bytes
        # belong to it. A single stale binding therefore served one job's
        # video for another's id, indefinitely.
        if target is not None and is_content(full) and not await _owns(target, job_id, headers):
            _log_bad_binding(job_id, target)
            bal.forget(job_id)
            target = None
        if target is None:
            try:
                target = await _find_owner(job_id, full, headers)
            except AmbiguousOwner:
                return JSONResponse(
                    {
                        "detail": f"job {job_id} is claimed by more than one "
                        f"replica — refusing to guess which output is yours",
                        "retryable": False,
                    },
                    status_code=409,
                )
        if target is None:
            return JSONResponse({"detail": f"no replica holds job {job_id}"}, status_code=404)
        if request.method == "DELETE":
            # Forward FIRST. Freeing occupancy before the replica confirms
            # means a failed delete leaves a job still running on a replica
            # the balancer now believes is idle — it routes new work there,
            # which is exactly what job-scoped occupancy exists to prevent.
            resp = await _passthrough(target, full, request, headers, body)
            if resp.status_code < 400:
                if bal.end_job(job_id, ok=False):
                    _note_terminal(job_id, "cancelled", closed_by="delete")
                bal.forget(job_id)
                await _safe_pump()  # a slot just freed
            return resp
        # A status poll is how we learn a job finished — the replica never
        # tells us. Reading the terminal status here is what closes out its
        # occupancy and records how long it really took.
        if full.rstrip("/").endswith(job_id):
            resp = await _forward(target, full, request, headers, body)
            with contextlib.suppress(ValueError, KeyError, TypeError):
                status = json.loads(resp.content).get("status")
                if status in TERMINAL_STATUSES and bal.end_job(job_id, ok=status == "completed"):
                    _note_terminal(job_id, str(status), closed_by="client")
                    # A replica just freed — hand it the next waiting ticket.
                    await _safe_pump()
            out_headers = safe_headers(resp.headers)
            if resp.status_code < 400:
                # ONLY on success. Stamping it on the replica's 404 told the
                # gateway "this fleet owns the job" about a fleet that had
                # just denied it — and the gateway caches that binding until
                # a 502, so one 404 poisoned the mapping for the process's
                # whole life.
                out_headers["x-deepme-job-id"] = job_id
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=out_headers,
                media_type=resp.headers.get("content-type"),
            )
        # A sub-resource fetch. /content starts the artifact's re-fetch
        # retention window on its FIRST COMPLETED read — a download cut
        # mid-stream must not burn the only re-fetch window (marking at
        # request time deleted billed work whose delivery a gateway restart
        # had interrupted; the re-adopting client then fetched a 404).
        fetched_id = job_id if full.rstrip("/").endswith("/content") else ""
        return await _passthrough(
            target, full, request, headers, body, fetched_id=fetched_id, job_header=job_id
        )

    # 2. Everything else is new work: give it to the least busy replica.
    # Reap first — routing on stale occupancy would treat an idle replica as
    # busy and send work to a genuinely busier one.
    if request.method == "POST":
        # NOT reaped here. The maintenance beat reconciles every 10s, and
        # doing it inline put an unbounded fan-out of probes on the
        # latency-critical path: one wedged replica delayed a client's
        # submission by the whole pass, the client timed out and retried,
        # and each retry started another pass.
        queued = await _admit(full, request, headers, body)
        if queued is not None:
            return queued
    target = bal.pick()
    if target is None:
        return JSONResponse({"detail": "no healthy replica available"}, status_code=503)
    # ONLY a POST is work. Reserving for every method meant a non-job-scoped
    # GET (an operator curl of /v1/videos, a dashboard) against a wedged
    # replica parked forever on the unbounded-read client holding a
    # reservation that nothing ever released — free_slots() dropped
    # permanently, and since admission delegates to it, the queue stopped
    # draining on an idle fleet with nothing counted anywhere.
    is_work = request.method == "POST"
    began = bal.reserve(target) if is_work else 0.0
    try:
        # A proxied read is also bounded: `client` has read=None, which only
        # a streaming download should ever get.
        resp = await _forward(target, full, request, headers, body, bounded=not is_work)
    except httpx.HTTPError as exc:
        if is_work:
            bal.release(target, failed=True)
        bal.mark(target, healthy=False)
        _publish("fleet", action="unhealthy", url=target)
        return JSONResponse({"detail": f"replica {target} unreachable: {exc}"}, status_code=502)
    except BaseException:
        # A disconnecting caller cancels this handler; the reserved slot
        # must not leak with it.
        if is_work:
            bal.release(target, failed=True)
        raise
    # A response carrying a job id means the replica is now busy for minutes,
    # not for the milliseconds this call took. Hand the held slot to the job
    # so occupancy and timing span the actual work.
    opened = False
    if resp.status_code < 400 and request.method == "POST":
        with contextlib.suppress(ValueError, KeyError, TypeError):
            payload = json.loads(resp.content)
            if payload.get("id") and payload.get("status") not in TERMINAL_STATUSES:
                bal.release(target)
                bal.begin_job(payload["id"], target, collection=collection_of(full))
                with contextlib.suppress(ValueError, TypeError):
                    _capture_meta(payload["id"], json.loads(body), via="direct", backend=target)
                opened = True
    if not opened and is_work:
        # Finished inside the call: a real, short job. (A proxied read holds
        # no reservation at all now, so there is nothing to release — and
        # its 4xx no longer inflates a replica's `failed` count, which
        # feeds every health and billing read.)
        bal.record_inline(target, began, ok=resp.status_code < 400)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=safe_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


async def _forward(
    target: str, path: str, request: Request, headers: dict, body: bytes, *, bounded: bool = False
):
    """Proxy one request. `bounded` picks the control-plane client, whose
    read timeout is finite — a submission may legitimately take a while,
    but a proxied read must never park a coroutine forever."""
    return await (ctl if bounded else client).request(
        request.method,
        f"{target}{path}",
        headers={**headers, **content_headers(request)},
        content=body or None,
        params=_upstream_params(request),
    )


async def _passthrough(
    target: str,
    path: str,
    request: Request,
    headers: dict,
    body: bytes,
    *,
    fetched_id: str = "",
    job_header: str = "",
) -> Response:
    """Stream a response through untouched — video downloads are large.

    `fetched_id` marks the artifact fetched ONLY when the stream ran to
    completion: the marker is set after the last byte is yielded, and a
    client disconnect closes the generator before that line runs.
    """
    req = client.build_request(
        request.method,
        f"{target}{path}",
        headers=headers,
        content=body or None,
        params=_upstream_params(request),
    )
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        bal.mark(target, healthy=False)
        _publish("fleet", action="unhealthy", url=target)
        return JSONResponse({"detail": f"replica {target} unreachable: {exc}"}, status_code=502)

    async def _body():
        # aiter_BYTES, not raw: _safe_headers strips content-encoding, so
        # the body must be decoded to match. Forwarding raw gzip with the
        # header removed handed the client binary garbage — invisibly, and
        # twice over on the two-hop path through the gateway.
        async for chunk in resp.aiter_bytes():
            yield chunk
        if fetched_id and resp.status_code == 200 and fetched_id not in _fetched:
            import time as _time

            _fetched[fetched_id] = _time.time()
            _publish("fetch", job_id=fetched_id)
            # DELIVERED, NOT YET DONE. The artifact is reclaimed when the
            # client ACKNOWLEDGES it has persisted the bytes (DELETE on this
            # same path), not when the last byte leaves here.
            #
            # Deleting on delivery made the transfer single-shot: a caller
            # that received 200 and then failed to write — no such directory,
            # disk full, killed mid-flush — lost the work permanently, with
            # no way to ask for it again. That cost a real 5.5-minute clip.
            # We control BOTH ends of this protocol, so "did the client keep
            # it?" is answerable rather than assumed.
            #
            # The property that mattered is preserved: nothing that has been
            # handed over can be handed over TWICE. `_fetched` is stamped
            # here, at delivery, so the retention window starts now and runs
            # whether or not an ack ever arrives — a client that dies mid-
            # write cannot strand a file indefinitely, and the file is still
            # gone long before any id could be reused. In the normal case the
            # ack lands milliseconds after the write and the file's life on
            # disk is shorter than it was under the previous rule's own
            # streaming window.
            # To DISK: a restart that forgot first-fetch times silently
            # reset every 1h window back to 6h.
            with contextlib.suppress(OSError):
                _persist_membership()

    out_headers = safe_headers(resp.headers)
    if job_header:
        # The EFFECTIVE job id rides every job-scoped response, so the
        # gateway can bind ids to fleets from an authoritative statement
        # instead of probing (see its _find_owner).
        out_headers["x-deepme-job-id"] = job_header
    return StreamingResponse(
        _body(),
        status_code=resp.status_code,
        headers=out_headers,
        background=_closer(resp),
    )


def _closer(resp):
    from starlette.background import BackgroundTask

    return BackgroundTask(resp.aclose)
