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
import json
import os
from pathlib import Path

import httpx

# NOT named queue.py: that shadows the stdlib `queue` module, and which one
# wins depends on sys.path order at import time.
from admission import (  # deployed next to this
    CANCELLED,
    PRIORITY_NORMAL,
    REJECTED,
    SUBMITTED,
    AdmissionQueue,
)
from balance import Balancer  # deployed next to this file
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

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
CTL_TIMEOUT = httpx.Timeout(30.0, connect=5.0)

DRAIN_S = 10.0

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
        # Boot: rebuild occupancy from replica truth before anything routes.
        with contextlib.suppress(Exception):
            await _recover_occupancy()
        beats = 0
        while True:
            await asyncio.sleep(DRAIN_S)
            beats += 1
            with contextlib.suppress(Exception):
                # Order matters: reconcile FIRST (frees slots the replicas
                # say are done), revive lapsed backends, then pump into
                # whatever is free, then prune. Together these enforce the
                # liveness invariant — with zero client traffic, finished
                # slots still free, failed backends still return, and the
                # queue still drains.
                await _reap_stale_jobs()
                await _revive_backends()
                await _pump()
                admission.prune()
            if beats % _RETENTION_EVERY_BEATS == 0:
                with contextlib.suppress(Exception):
                    _retention_pass()

    task = asyncio.create_task(drain())
    try:
        yield
    finally:
        task.cancel()


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
admission = AdmissionQueue(salt=f"{os.environ.get('BALANCER_NAME', 'b')}-{os.urandom(2).hex()}-")
client = httpx.AsyncClient(timeout=TIMEOUT)
ctl = httpx.AsyncClient(timeout=CTL_TIMEOUT)

# One vocabulary for "this job is over", shared by every code path. Two of
# the four paths previously listed only two of the four words, so inline
# cancellations opened occupancy that nothing would ever close.
TERMINAL_STATUSES = ("completed", "failed", "cancelled", "error")

# Job-scoped routes: "/v1/videos/{id}", ".../content". The id is the segment
# straight after the collection name.
_COLLECTIONS = ("videos", "images", "meshes")


def _auth(request: Request) -> dict[str, str]:
    """Forward the caller's credentials, or our own if it sent none."""
    incoming = request.headers.get("authorization")
    if incoming:
        return {"authorization": incoming}
    return {"authorization": f"Bearer {API_KEY}"} if API_KEY else {}


def _job_id(path: str) -> str | None:
    parts = [p for p in path.split("/") if p]
    for i, part in enumerate(parts):
        if part in _COLLECTIONS and i + 1 < len(parts):
            return parts[i + 1]
    return None


def _collection_of(path: str) -> str:
    """Which API family a request belongs to, for reconcile probes later."""
    for part in path.split("/"):
        if part in _COLLECTIONS:
            return part
    return "videos"


def _status_path(path: str, job_id: str) -> str:
    """The cheap status URL for a job, whatever sub-resource was asked for.

    Ownership must never be probed with the caller's own path: for
    ".../{id}/content" that downloads the whole MP4 from every replica —
    buffered, in full — just to find out which one has it.
    """
    head, _, _ = path.partition(f"/{job_id}")
    return f"{head}/{job_id}"


async def _find_owner(job_id: str, path: str, headers: dict) -> str | None:
    """Ask every replica which one holds this job.

    Only needed after the balancer lost its mapping. Sequential and cheap: a
    status GET is milliseconds, and a wrong guess would strand the result.

    Recovery restores OCCUPANCY too, not just routing: after a restart the
    fleet is full of jobs the balancer no longer counts, and healing only the
    route left pick() double-booking busy replicas while _all_busy() waved
    the whole former queue into engine-internal FIFOs.
    """
    path = _status_path(path, job_id)
    for st in bal.stats.values():
        if not st.healthy:
            continue
        with contextlib.suppress(httpx.HTTPError):
            r = await ctl.get(f"{st.url}{path}", headers=headers)
            if r.status_code < 400:
                bal.remember(job_id, st.url)
                with contextlib.suppress(ValueError, TypeError, KeyError):
                    if str(dict(r.json()).get("status") or "") not in TERMINAL_STATUSES:
                        bal.begin_job(job_id, st.url)
                return st.url
    return None


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
    headers = _auth(request)

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
                wait = admission.wait_estimate_s(ticket_id, mean, len(bal.stats) or 1)
                event = {
                    "job_id": ticket_id,
                    "status": "queued",
                    "queue_position": position,
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

            target = bal.owner(current) or await _find_owner(
                current, f"/v1/videos/{current}", headers
            )
            if target is None:
                yield _sse({"status": "unknown", "job_id": current})
                return
            status = "queued"
            with contextlib.suppress(httpx.HTTPError, ValueError):
                r = await client.get(f"{target}/v1/videos/{current}", headers=headers)
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
                bal.end_job(current, ok=status == "completed")
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
    live = [s for s in bal.stats.values() if s.accepting]
    return bool(live) and all(s.inflight > 0 for s in live)


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
    # Only "/v1/videos" is held. _COLLECTIONS already names videos, images and
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
    priority = int(payload.get("priority", PRIORITY_NORMAL))
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
    wait = admission.wait_estimate_s(ticket.ticket_id, mean, len(bal.stats) or 1)
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
        except httpx.HTTPError:
            bal.release(target, failed=True)
            bal.mark(target, healthy=False)
            # Transport error: the request never reached the replica, so the
            # ticket goes back to the queue for a different one.
            admission.release_claim(ticket.ticket_id)
            return
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
            admission.mark_rejected(ticket.ticket_id, detail)
            _publish("rejected", ticket=ticket.ticket_id, error=detail[:120])
            continue
        job_id, status = "", ""
        with contextlib.suppress(ValueError, KeyError, TypeError):
            payload = json.loads(resp.content)
            job_id = str(payload.get("id") or "")
            status = str(payload.get("status") or "")
        if job_id and status not in TERMINAL_STATUSES:
            # Same rule as the direct path: the slot becomes the job's.
            bal.release(target)
            bal.begin_job(job_id, target, collection=_collection_of(path))
            _capture_meta(job_id, ticket.body.get("body") or {}, via="queue")
            _publish("handover", ticket=ticket.ticket_id, job_id=job_id, backend=target)
        else:
            # Finished inside the call — a real, short job, counted as one.
            bal.record_inline(target, began, ok=True)
        admission.mark_submitted(ticket.ticket_id, job_id, target)


@app.get("/v1/queue")
async def queue_state() -> JSONResponse:
    """What is waiting, for operators and the service layer."""
    return JSONResponse(
        {
            **admission.snapshot(),
            "next": [
                {
                    "ticket": t.ticket_id,
                    "priority": t.priority,
                    "position": admission.position(t.ticket_id),
                }
                for t in admission.waiting()[:20]
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
        bal.end_job(job_id, ok=False)
        _note_terminal(job_id, "lost", closed_by="strikes")
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
    import time as _time

    now = _time.monotonic()
    auth = {"authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    for job_id, url, began, collection in bal.open_jobs():
        if now - began < RECONCILE_MIN_AGE_S:
            continue
        try:
            r = await ctl.get(f"{url}/v1/{collection}/{job_id}", headers=auth)
        except httpx.HTTPError:
            _strike(job_id, url, "replica unreachable")
            continue
        if r.status_code == 404:
            _strike(job_id, url, "unknown to its replica")
            continue
        if r.status_code >= 400:
            continue
        status = ""
        with contextlib.suppress(ValueError, TypeError, KeyError):
            status = str(dict(r.json()).get("status") or "")
        if status in TERMINAL_STATUSES:
            bal.end_job(job_id, ok=status == "completed")
            _note_terminal(job_id, status, closed_by="reconciler")
            _job_strikes.pop(job_id, None)
            # Closed by the platform, not by a client poll — the counter
            # a billing layer watches for walked-away work.
            bal.reaped += 1
            print(
                f"reconciled {job_id} ({status}) on {url}: closed without a client poll", flush=True
            )
        else:
            # Alive and known: a healthy answer clears past strikes.
            _job_strikes.pop(job_id, None)


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

EVENTS_RING_MAX = 500
_events_ring: list[dict] = []


def _seq_counter():
    n = 0
    while True:
        n += 1
        yield n


_event_seq = _seq_counter()
_subscribers: set = set()


def _publish(kind: str, **fields) -> None:
    import asyncio
    import time as _time

    evt = {"seq": next(_event_seq), "ts": round(_time.time(), 3), "type": kind, **fields}
    _events_ring.append(evt)
    del _events_ring[:-EVENTS_RING_MAX]
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
_recent: list[dict] = []
_RECENT_MAX = 50


def _capture_meta(job_id: str, payload: dict, *, via: str) -> None:
    if not job_id or not isinstance(payload, dict):
        return
    target = payload.get("target") or {}
    _job_meta[job_id] = {
        "task": payload.get("task"),
        "prompt": str(payload.get("prompt") or "")[:120],
        "seconds": target.get("duration_seconds") or payload.get("seconds"),
        "priority": payload.get("priority"),
        "via": via,
    }
    while len(_job_meta) > 2000:
        _job_meta.pop(next(iter(_job_meta)))
    _publish("started", job_id=job_id, **_job_meta[job_id])


def _note_terminal(job_id: str, status: str, *, closed_by: str) -> None:
    import time as _time

    entry = {
        "job_id": job_id,
        "status": status,
        "closed_by": closed_by,
        "finished_at": _time.time(),
        **_job_meta.pop(job_id, {}),
    }
    _recent.insert(0, entry)
    del _recent[_RECENT_MAX:]
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


def _retention_pass() -> None:
    """Reclaim expired artifacts, on the platform's clock."""
    import pathlib
    import time as _time

    global retention_deleted
    if not WORK_DIR:
        return
    entries: list[tuple[str, str, float]] = []
    outputs = pathlib.Path(WORK_DIR) / "outputs"
    inputs = pathlib.Path(WORK_DIR) / "inputs"
    with contextlib.suppress(OSError):
        entries.extend(
            (str(p), p.stem, p.stat().st_mtime) for p in outputs.iterdir() if p.is_file()
        )
    with contextlib.suppress(OSError):
        # Staged uploads have no artifact id: global TTL only.
        entries.extend((str(p), "", p.stat().st_mtime) for p in inputs.iterdir() if p.is_file())
    open_ids = {j for j, _u, _t, _c in bal.open_jobs()}
    doomed = _retention_candidates(
        entries,
        _fetched,
        open_ids,
        _time.time(),
        RETENTION_TTL_S,
        RETENTION_FETCHED_TTL_S,
    )
    reclaimed = 0
    for path in doomed:
        with contextlib.suppress(OSError):
            Path(path).unlink()
            retention_deleted += 1
            reclaimed += 1
            print(f"retention: reclaimed {path}", flush=True)
    if reclaimed:
        _publish("retention", reclaimed=reclaimed)
    # Fetch records for deleted artifacts have done their job.
    kept = {p.rsplit("/", 1)[-1].rsplit(".", 1)[0] for p in doomed}
    for art_id in list(_fetched):
        if art_id in kept:
            _fetched.pop(art_id, None)


async def _recover_occupancy() -> None:
    """Rebuild occupancy from replica truth at boot.

    A restart empties the job map while the replicas keep generating. Left
    unrecovered, pick() double-books busy replicas and _all_busy() reads
    false — waving the entire former queue into engine-internal FIFOs where
    nothing can reorder or cancel it. The replicas own the truth, so ask
    them: every non-terminal job they report re-opens its slot.

    Listing is /v1/videos only — the sole modality deployed; extend per
    collection when another ships (see the _admit latent-gap note).
    """
    auth = {"authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    for st in list(bal.stats.values()):
        with contextlib.suppress(httpx.HTTPError, ValueError, TypeError, KeyError):
            r = await ctl.get(f"{st.url}/v1/videos", headers=auth)
            if r.status_code >= 400:
                continue
            for job in dict(r.json()).get("data", []) or []:
                job_id = str(job.get("id") or "")
                status = str(job.get("status") or "")
                if job_id and status not in TERMINAL_STATUSES:
                    bal.begin_job(job_id, st.url)
                    print(f"recovered occupancy: {job_id} running on {st.url}", flush=True)


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


_restore_membership()


@app.post("/balancer/backends")
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


@app.post("/balancer/backends/drain")
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
    st = bal.stats[url]
    return JSONResponse({"url": url, "draining": True, "inflight": st.inflight})


@app.post("/balancer/backends/undrain")
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
    await _safe_pump()  # a slot just came back
    return JSONResponse({"url": url, "draining": False})


@app.delete("/balancer/backends")
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
    for t in admission.waiting():
        body = t.body.get("body") or {}
        target = body.get("target") or {}
        queued.append(
            {
                "ticket": t.ticket_id,
                "position": admission.position(t.ticket_id),
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


@app.get("/balancer/stats")
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
    headers = _auth(request)
    body = await request.body()

    # 1. A request about an EXISTING job goes to its owner, wherever it is.
    job_id = _job_id(full)

    # A ticket is not a job yet, but a 202 hands the caller an id shaped like
    # one and invites it to poll. Until a replica takes the work there is
    # nothing to forward, so answer from the queue; once there is, rewrite the
    # path to the real job so status AND content follow through untouched.
    if job_id and job_id.startswith("q_") and request.method in ("GET", "DELETE"):
        ticket = admission.get(job_id)
        if ticket is None:
            return JSONResponse({"detail": f"no such ticket {job_id}"}, status_code=404)
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
                    "billable": False,
                }
            )
        full = full.replace(job_id, ticket.job_id, 1)
        job_id = ticket.job_id

    if job_id and request.method in ("GET", "DELETE"):
        target = bal.owner(job_id) or await _find_owner(job_id, full, headers)
        if target is None:
            return JSONResponse({"detail": f"no replica holds job {job_id}"}, status_code=404)
        if request.method == "DELETE":
            # Forward FIRST. Freeing occupancy before the replica confirms
            # means a failed delete leaves a job still running on a replica
            # the balancer now believes is idle — it routes new work there,
            # which is exactly what job-scoped occupancy exists to prevent.
            resp = await _passthrough(target, full, request, headers, body)
            if resp.status_code < 400:
                bal.end_job(job_id, ok=False)
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
                if status in TERMINAL_STATUSES:
                    bal.end_job(job_id, ok=status == "completed")
                    _note_terminal(job_id, str(status), closed_by="client")
                    # A replica just freed — hand it the next waiting ticket.
                    await _safe_pump()
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=_safe_headers(resp.headers),
                media_type=resp.headers.get("content-type"),
            )
        # A sub-resource fetch. /content starts the artifact's re-fetch
        # retention window on its FIRST read — recorded at request time
        # (the download begins immediately) and only once, so repeat
        # fetches inside the window do not keep extending it.
        if full.rstrip("/").endswith("/content") and job_id not in _fetched:
            import time as _time

            _fetched[job_id] = _time.time()
        return await _passthrough(target, full, request, headers, body)

    # 2. Everything else is new work: give it to the least busy replica.
    # Reap first — routing on stale occupancy would treat an idle replica as
    # busy and send work to a genuinely busier one.
    if request.method == "POST":
        await _reap_stale_jobs()
        queued = await _admit(full, request, headers, body)
        if queued is not None:
            return queued
    target = bal.pick()
    if target is None:
        return JSONResponse({"detail": "no healthy replica available"}, status_code=503)
    began = bal.reserve(target)
    try:
        resp = await _forward(target, full, request, headers, body)
    except httpx.HTTPError as exc:
        bal.release(target, failed=True)
        bal.mark(target, healthy=False)
        return JSONResponse({"detail": f"replica {target} unreachable: {exc}"}, status_code=502)
    except BaseException:
        # A disconnecting caller cancels this handler; the reserved slot
        # must not leak with it.
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
                bal.begin_job(payload["id"], target, collection=_collection_of(full))
                with contextlib.suppress(ValueError, TypeError):
                    _capture_meta(payload["id"], json.loads(body), via="direct")
                opened = True
    if not opened:
        if request.method == "POST":
            # Finished inside the call: a real, short job.
            bal.record_inline(target, began, ok=resp.status_code < 400)
        else:
            # A proxied read is not work and must not reach the job counters.
            bal.release(target, failed=resp.status_code >= 400)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=_safe_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


def _safe_headers(headers) -> dict[str, str]:
    """Drop hop-by-hop headers; re-encoding here would contradict them."""
    drop = {"content-length", "content-encoding", "transfer-encoding", "connection"}
    return {k: v for k, v in headers.items() if k.lower() not in drop}


async def _forward(target: str, path: str, request: Request, headers: dict, body: bytes):
    return await client.request(
        request.method,
        f"{target}{path}",
        headers={**headers, **_content_headers(request)},
        content=body or None,
        params=_upstream_params(request),
    )


async def _passthrough(
    target: str, path: str, request: Request, headers: dict, body: bytes
) -> Response:
    """Stream a response through untouched — video downloads are large."""
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
        return JSONResponse({"detail": f"replica {target} unreachable: {exc}"}, status_code=502)
    return StreamingResponse(
        resp.aiter_raw(),
        status_code=resp.status_code,
        headers=_safe_headers(resp.headers),
        background=_closer(resp),
    )


def _closer(resp):
    from starlette.background import BackgroundTask

    return BackgroundTask(resp.aclose)


def _content_headers(request: Request) -> dict[str, str]:
    ct = request.headers.get("content-type")
    return {"content-type": ct} if ct else {}
