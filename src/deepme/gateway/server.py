"""Gateway: one address in front of every deployed fleet.

The internal-customer-API layer talks only to this. It names a model and a
task; which checkpoint, which balancer and which replica serve that request
are operational details that change whenever the fleet mix changes, and no
client should have to track them.

Two hops now sit between a caller and a GPU (gateway -> balancer -> replica),
so job ids — which only mean something to the replica that minted them — are
tracked here as well. A follow-up poll finds its way home without the caller
knowing there were hops at all.

Environment: GATEWAY_ROUTES (JSON list of {model, profile, tasks, url}),
GATEWAY_API_KEY, deployed to ~/deepme-runtime/gateway alongside route.py.
"""

from __future__ import annotations

import contextlib
import json
import os

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

# Shared with the balancer — see proxycore's docstring for why the owner
# probe must not be maintained twice. Deployed next to this file too.
from proxycore import (
    AmbiguousOwner,
    auth_headers,
    content_headers,
    find_owner,
    is_content,
    job_id_of,
    safe_headers,
)
from route import RoutingError, build  # deployed next to this file

ROUTES = json.loads(os.environ.get("GATEWAY_ROUTES", "[]"))
# Static price table derived at deploy time (see economics.py). The gateway
# never derives prices — it applies the live surge tier to this table.
PRICING = json.loads(os.environ.get("GATEWAY_PRICING", "{}"))
API_KEY = os.environ.get("GATEWAY_API_KEY", "")
# A generation runs for minutes; only connecting should be quick.
TIMEOUT = httpx.Timeout(connect=10.0, read=None, write=None, pool=None)
# Control-plane reads (capacity, jobs, queue, metrics, health, owner probes)
# are sub-second against a healthy balancer and MUST be bounded AND pooled
# apart from proxied traffic: a multi-hundred-MB /content download holding
# the shared pool made /v1/capacity time out, the gateway reported the fleet
# unreachable inside a 200, and clients read that as proof their (running,
# about-to-complete) jobs were gone — false stall verdicts, exactly while
# results were being fetched.
CTL_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


@contextlib.asynccontextmanager
async def _lifespan(_app):
    """Own the HTTP clients' lifetime.

    The gateway had no lifespan at all, so its pooled connections were
    never closed on shutdown — torn down abruptly instead of gracefully,
    and with no place to put any future startup invariant.
    """
    yield
    await client.aclose()
    await ctl.aclose()


app = FastAPI(lifespan=_lifespan)
table = build(ROUTES)
client = httpx.AsyncClient(timeout=TIMEOUT)
ctl = httpx.AsyncClient(timeout=CTL_TIMEOUT)


def _log(message: str) -> None:
    """One line to the journal, for things an operator will be asked about.

    This service used to print NOTHING — no routing refusal, no vanished
    job, no unreachable fleet left any trace, so "my job disappeared" was
    unanswerable without reproducing it. stdout, because systemd captures
    it and `deepme gateway logs` reads it back.
    """
    print(f"gateway: {message}", flush=True)


def require_key(request: Request) -> None:
    """Gate the CONTROL PLANE (/gateway/reload, /gateway/routes).

    `/gateway/reload` can repoint every model at an arbitrary URL — prompts
    and results included — and it was reachable by anyone who could talk to
    the port, which binds 0.0.0.0. `/v1/*` stays ungated on purpose: it
    forwards the CALLER's credentials onward, and gating it here would
    break the service layer's own auth. `/gateway/discover` likewise stays
    open — it is the documented call a service layer makes to find us.

    Off when no key is configured, so a local/dev deployment is unchanged.
    """
    if not API_KEY:
        return
    header = request.headers.get("authorization", "")
    if header.removeprefix("Bearer ").strip() != API_KEY:
        raise HTTPException(status_code=401, detail="control-plane routes require the platform key")


CATALOG = json.loads(os.environ.get("GATEWAY_CATALOG", "{}"))


@app.get("/gateway/discover")
async def discover() -> JSONResponse:
    """Service catalog for internal consumers.

    Everything an internal service needs to talk to this platform — models,
    their tasks and limits, and every endpoint with whether it is merely
    declared or actually running — so no other service maintains its own map.
    """
    return JSONResponse(CATALOG)


@app.post("/gateway/reload", dependencies=[Depends(require_key)])
async def reload_state(request: Request) -> JSONResponse:
    """Re-point routes/catalog/pricing IN PLACE — no restart.

    A restart severs every in-flight SSE stream and MP4 download, so a
    routine fleet resize was breaking unrelated customers' transfers. This
    swaps the frozen state while preserving the job-owner map, which is the
    only thing a restart cannot rebuild instantly.
    """
    global table, CATALOG, PRICING
    body: dict = {}
    with contextlib.suppress(ValueError, TypeError):
        body = json.loads(await request.body())
    routes = body.get("routes")
    if not isinstance(routes, list) or not routes:
        return JSONResponse(
            {"detail": "body must carry a non-empty routes list"},
            status_code=400,
        )
    old_owners = table.owners_snapshot()
    table = build(routes)
    table.adopt_owners(old_owners)  # jobs keep resolving across the swap
    if isinstance(body.get("catalog"), dict):
        CATALOG = body["catalog"]
    if isinstance(body.get("pricing"), dict):
        PRICING = body["pricing"]
    return JSONResponse(
        {
            "reloaded": True,
            "routes": len(routes),
            "owner_map_preserved": len(old_owners),
        }
    )


@app.get("/gateway/routes", dependencies=[Depends(require_key)])
async def routes() -> JSONResponse:
    """What is deployed and what each fleet can serve.

    A service layer reads this to know which capabilities are live right now,
    rather than hardcoding a topology that changes with the fleet mix.
    """
    return JSONResponse(table.snapshot())


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness of the PATH, not just this process.

    A gateway whose every balancer is down serves nothing; answering 200 on
    "routes are configured" reported exactly that state as healthy. Probes
    run concurrently with a short timeout so this stays monitor-friendly.
    """
    import asyncio

    if not table.routes:
        return JSONResponse({"status": "no fleets configured", "fleets": 0}, status_code=503)

    async def up(route) -> bool:
        try:
            r = await ctl.get(f"{route.url}/health", timeout=3)
            return bool(r.status_code < 500)
        except httpx.HTTPError:
            return False

    live = sum(await asyncio.gather(*(up(r) for r in table.routes)))
    return JSONResponse(
        {
            "status": "ok" if live else "no reachable fleet",
            "fleets": len(table.routes),
            "reachable_fleets": live,
        },
        status_code=200 if live else 503,
    )


@app.get("/v1/models")
async def models() -> JSONResponse:
    """List what is served, OpenAI-shaped.

    This is the standard "what do you have?" call, so it must answer without
    routing parameters. The catch-all below would otherwise try to resolve a
    fleet from an absent model name and refuse a plain GET /v1/models with a
    400 — discovery failing exactly when a client has nothing to go on yet.

    `tasks` is the extra a caller needs here: one model name can be backed by
    several fleets, and the task is what picks between them.
    """
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": model,
                    "object": "model",
                    "owned_by": "deepme",
                    "tasks": sorted(table.tasks_for(model)),
                }
                for model in table.models()
            ],
        }
    )


async def _fleet_capacity(route) -> dict:
    """One fleet's live capacity, with unreachability read safely.

    Shared by /v1/capacity (scheduling) and /v1/pricing (the surge tier is a
    function of the same queue signal) so the two can never disagree about
    what the fleet looks like.
    """
    entry = {
        "model": route.model,
        "profile": route.profile,
        "tasks": sorted(route.tasks),
        "url": route.url,
    }
    try:
        r = await ctl.get(f"{route.url}/v1/capacity")
        r.raise_for_status()
        entry.update(dict(r.json()), reachable=True)
    except (httpx.HTTPError, ValueError):
        # A fleet that cannot be asked has NO free capacity as far as a
        # scheduler is concerned — reporting 0 free is the safe reading,
        # and `reachable` says why rather than hiding it.
        entry.update(free=0, busy=0, replicas=0, healthy=0, accepting=False, reachable=False)
    return entry


@app.get("/v1/capacity")
async def capacity() -> JSONResponse:
    """One poll: where can work go right now, keyed by what you route on.

    For a caller that keeps its own priority queue and submits only into free
    capacity. `free` per entry is how many requests will start immediately;
    submit that many, then poll again.

    Fleets are asked CONCURRENTLY and none of them touches a replica to
    answer, so the cost of this call is one round-trip regardless of fleet
    count — it is meant to be polled every second.
    """
    import asyncio

    fleets = await asyncio.gather(*(_fleet_capacity(r) for r in table.routes))
    return JSONResponse(
        {
            "free": sum(int(f.get("free") or 0) for f in fleets),
            "accepting": [t for f in fleets if f.get("free") for t in f.get("tasks", [])],
            "fleets": list(fleets),
        }
    )


@app.get("/v1/pricing")
async def pricing() -> JSONResponse:
    """What a second of output costs right now, per task — for billing layers.

    The static table (cost, floor, market, surge ladder) is derived at deploy
    time from deploy/economics.toml and frozen in, like routes; this endpoint
    applies the LIVE tier from the same queue signal /v1/capacity reads. A
    customer-facing pricing service polls this as its baseline and adds its
    own product policy on top.

    Quotes are indicative: the price locks at the `submitted` event
    (billable=true), never at quote time — queue depth moves while a request
    waits, and a quote-time lock lets callers arbitrage the surge.
    """
    import asyncio

    if not PRICING:
        return JSONResponse(
            {
                "detail": "no pricing table deployed — "
                "restart the gateway after editing economics.toml"
            },
            status_code=503,
        )
    ladder = PRICING.get("surge", [])
    fleets = await asyncio.gather(*(_fleet_capacity(r) for r in table.routes))
    tasks: dict[str, dict[str, object]] = {}
    for f in fleets:
        # Depth is queue per replica that can actually DRAIN it. Dividing by
        # the raw replica count let dead and draining replicas dilute the
        # depth — underquoting the surge exactly when capacity was scarce
        # (every fleet switch drains replicas for up to an hour).
        accepting = f.get("accepting_replicas")
        replicas = int(accepting if accepting is not None else (f.get("replicas") or 0))
        depth = (int(f.get("queued") or 0) / replicas) if replicas else 0.0
        # The highest rung whose threshold the current depth clears.
        multiplier = 1.0
        for tier in ladder:
            if depth >= float(tier["queued_per_replica"]):
                multiplier = float(tier["multiplier"])
        spot = (
            bool(f.get("reachable"))
            and int(f.get("free") or 0) > 0
            and not int(f.get("queued") or 0)
        )
        for task in f.get("tasks", []):
            base = PRICING["tasks"].get(task)
            if base is None:
                continue
            if not f.get("reachable"):
                # No quote for what cannot be served: a price on an
                # unreachable fleet is an invitation to bill for nothing.
                # But never CLOBBER a live quote another fleet already gave
                # for this task — the day two fleets serve one task, routing
                # goes to the first resolvable fleet and pricing must agree.
                if not tasks.get(task, {}).get("available"):
                    tasks[task] = {"available": False}
                continue
            if tasks.get(task, {}).get("available"):
                continue  # first reachable fleet wins, matching resolve()
            price = base["spot_floor"] if spot else round(base["market"] * multiplier, 4)
            quote: dict[str, object] = {
                "available": True,
                "tier": "spot"
                if spot
                else ("market" if multiplier == 1.0 else f"surge-x{multiplier}"),
                "price_per_output_second": price,
                "spot_floor": base["spot_floor"],
                "market": base["market"],
                "queued_per_replica": round(depth, 2),
            }
            tasks[task] = quote
    return JSONResponse(
        {
            "currency": PRICING.get("currency", "USD"),
            "unit": PRICING.get("unit", "per_output_second"),
            "lock_at": PRICING.get("lock_at", "submitted"),
            # The curve is why flat per-second pricing undercharges long clips
            # (cost is superlinear in duration, measured) — an app layer pricing
            # by duration needs it in-band.
            "duration_curve": PRICING.get("duration_curve", []),
            # When this table was derived, and from what. The table freezes at
            # `gateway start`; a billing layer compares table_hash against
            # `deepme econ --json` to detect that economics.toml moved on.
            "derived_at": PRICING.get("derived_at"),
            "table_hash": PRICING.get("table_hash"),
            "tasks": tasks,
        }
    )


@app.get("/v1/events")
async def event_firehose(request: Request) -> Response:
    """One SSE stream for the whole platform: every fleet's firehose,
    multiplexed, each event tagged `fleet`. Per-fleet upstreams reconnect
    on their own (5s backoff) and their outages are themselves events
    (`fleet_down`) — a dashboard subscribes once and stays subscribed
    through resizes and restarts.
    """
    import asyncio

    limit = 0
    with contextlib.suppress(ValueError, TypeError):
        limit = int(request.query_params.get("limit") or 0)
    out: asyncio.Queue = asyncio.Queue(maxsize=512)

    def emit(evt: dict) -> None:
        """Never block the fleet pump on a slow subscriber.

        `await out.put(...)` on a bounded queue back-pressured this pump,
        which stopped draining the balancer's stream, which filled the
        balancer's own subscriber queue — the exact opposite of the
        platform's stated policy. The balancer drops oldest and marks a
        `gap`; the mux now does the same, so the two agree.
        """
        try:
            out.put_nowait(evt)
        except asyncio.QueueFull:
            with contextlib.suppress(Exception):
                out.get_nowait()
                out.put_nowait({"type": "gap", "fleet": evt.get("fleet")})

    async def pump_fleet(route) -> None:
        while True:
            try:
                async with client.stream("GET", f"{route.url}/v1/events") as r:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        with contextlib.suppress(ValueError, TypeError):
                            evt = json.loads(line[len("data: ") :])
                            evt["fleet"] = route.profile
                            emit(evt)
            except asyncio.CancelledError:
                raise
            except Exception:
                emit({"type": "fleet_down", "fleet": route.profile})
            # Backoff on EVERY iteration, not just the error path: a stream
            # that ends cleanly (a graceful balancer shutdown, a `limit`
            # tap finishing, a proxy closing an idle connection) sent this
            # loop straight back to reconnect, hammering the balancer with
            # one full HTTP request per iteration.
            await asyncio.sleep(5)

    async def stream():
        tasks = [asyncio.create_task(pump_fleet(r)) for r in table.routes]
        sent = 0
        try:
            while True:
                try:
                    evt = await asyncio.wait_for(out.get(), timeout=15)
                    yield f"data: {json.dumps(evt)}\n\n"
                    sent += 1
                    if limit and sent >= limit:
                        return
                except TimeoutError:
                    yield ": heartbeat\n\n"
                if await request.is_disconnected():
                    return
        finally:
            for t in tasks:
                t.cancel()

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/v1/jobs")
async def jobs() -> JSONResponse:
    """Every job on the platform, across every fleet — the operator view."""
    import asyncio

    async def one(route):
        entry = {
            "profile": route.profile,
            "tasks": sorted(route.tasks),
            "reachable": True,
            "running": [],
            "queued": [],
            "recent": [],
        }
        try:
            r = await ctl.get(f"{route.url}/v1/jobs")
            r.raise_for_status()
            entry.update(dict(r.json()))
        except (httpx.HTTPError, ValueError):
            entry["reachable"] = False
        return entry

    fleets = await asyncio.gather(*(one(r) for r in table.routes))
    return JSONResponse({"fleets": list(fleets)})


@app.get("/metrics")
async def gw_metrics() -> Response:
    """Aggregated Prometheus exposition: one scrape target for the platform,
    with per-fleet labels."""
    import asyncio

    async def one(route):
        try:
            r = await ctl.get(f"{route.url}/metrics")
            r.raise_for_status()
            # A POSITIVE signal, not only the negative one. Only
            # deepme_fleet_up=0 was ever emitted, so a scrape could never
            # distinguish "healthy" from "this fleet was never scraped".
            out = [f'deepme_fleet_up{{fleet="{route.profile}"}} 1']
            for line in r.text.splitlines():
                if line.startswith("#") or not line.strip():
                    out.append(line)
                else:
                    name, _, value = line.partition(" ")
                    out.append(f'{name}{{fleet="{route.profile}"}} {value}')
            return out
        except httpx.HTTPError:
            return [f'deepme_fleet_up{{fleet="{route.profile}"}} 0']

    parts = await asyncio.gather(*(one(r) for r in table.routes))
    return Response(
        "\n".join(line for p in parts for line in p) + "\n",
        media_type="text/plain",
    )


@app.get("/v1/queue")
async def queue() -> JSONResponse:
    """Everything waiting, across every fleet.

    Queues are per-fleet — each balancer holds its own — but the contract says
    a service layer talks only to the gateway. Without this the documented
    queue API is unreachable here: "/v1/queue" is not job-scoped, so it falls
    through to model routing and is refused for naming no model.
    """
    # REPORTS, DOES NOT SCHEDULE. Each balancer queues independently, so this
    # aggregates several queues rather than presenting one. That is correct
    # while fleets serve DIFFERENT tasks (h3-video and h3-ref cannot take each
    # other's work), but the moment two fleets serve the SAME task, work
    # should be admitted here and dealt out to whichever fleet frees first —
    # otherwise one fleet queues while its twin idles.
    # To extend: move admission up to this layer for tasks with more than one
    # capable fleet, keeping per-fleet queues for the rest; the gateway would
    # then need the drain loop and ticket routes the balancer already has, and
    # `resolve()` would return a set of candidate fleets instead of one.
    import asyncio

    async def one(route) -> dict:
        entry = {"profile": route.profile, "tasks": sorted(route.tasks)}
        try:
            r = await ctl.get(f"{route.url}/v1/queue")
            r.raise_for_status()
            entry.update(dict(r.json()), reachable=True)
        except (httpx.HTTPError, ValueError):
            # Report it as unreachable rather than omitting it: a fleet that
            # is down is not a fleet with an empty queue. `waiting: 0` is
            # still present — the documented shape must hold in the failure
            # case a client is being taught to handle — with `reachable`
            # saying why the zero cannot be trusted.
            entry.update(waiting=0, reachable=False)
        return entry

    # CONCURRENTLY, like every sibling endpoint here. Sequentially, each
    # unreachable fleet added a full CTL_TIMEOUT to the response, so this
    # one call degraded linearly where /v1/jobs stayed flat.
    fleets = list(await asyncio.gather(*(one(r) for r in table.routes)))
    waiting = sum(int(f.get("waiting") or 0) for f in fleets)
    return JSONResponse({"waiting": waiting, "fleets": fleets})


@app.delete("/v1/queue/{ticket_id}")
async def cancel(ticket_id: str, request: Request) -> Response:
    """Cancel a ticket without the caller knowing which fleet holds it."""
    headers = auth_headers(request, API_KEY)
    path = f"/v1/queue/{ticket_id}"
    # Normally remembered from the 202 that minted it. The fallback probes
    # "/v1/videos/{ticket}" — the path a balancer actually answers for a
    # ticket; "/v1/queue/{id}" is DELETE-only and would find nothing.
    target = table.owner(ticket_id) or await _find_owner(
        ticket_id, f"/v1/videos/{ticket_id}", headers
    )
    if target is None:
        return JSONResponse({"detail": f"no fleet holds ticket {ticket_id}"}, status_code=404)
    try:
        # A cancel is control plane: bounded, like every other one.
        r = await ctl.delete(f"{target}{path}", headers=headers)
    except httpx.HTTPError as exc:
        return JSONResponse({"detail": f"fleet unreachable: {exc}"}, status_code=502)
    if r.status_code < 400:
        table.forget(ticket_id)
    return Response(
        content=r.content,
        status_code=r.status_code,
        headers=safe_headers(r.headers),
        media_type=r.headers.get("content-type"),
    )


@app.api_route("/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy(path: str, request: Request) -> Response:
    full = "/" + path
    headers = auth_headers(request, API_KEY)
    body = await request.body()

    # An existing job goes back to the fleet that minted its id, always.
    job_id = job_id_of(full)
    if job_id and request.method in ("GET", "DELETE"):
        remembered = table.owner(job_id)
        try:
            target = remembered or await _find_owner(job_id, full, headers)
        except AmbiguousOwner:
            # CONFLICT, not absence: the id exists in more than one place,
            # or may. 404 here would tell the caller it is unknown and safe
            # to resubmit, doubling a billable generation.
            return JSONResponse(
                {"detail": f"job {job_id} is claimed by more than one fleet — refusing to guess"},
                status_code=409,
            )
        if target is None:
            # LOUD. "My job vanished" is the report an operator gets, and
            # this line is the only place that can explain it — the gateway
            # wrote nothing at all before, so a refusal here left no trace
            # anywhere in the system.
            _log(f"no fleet holds job {job_id} ({request.method} {full})")
            return JSONResponse({"detail": f"no fleet holds job {job_id}"}, status_code=404)
        streamed = await _stream(target, full, request, headers, body)
        if streamed.status_code == 502 and remembered:
            # The REMEMBERED fleet is unreachable — likely stopped by a
            # resize. A stale mapping must not condemn the caller to retry
            # 502s forever (the contract says 502 is retryable): forget it,
            # probe afresh, and if nobody holds the job say so honestly.
            table.forget(job_id)
            try:
                fresh = await _find_owner(job_id, full, headers)
            except AmbiguousOwner:
                return JSONResponse(
                    {"detail": f"job {job_id} is claimed by more than one fleet"},
                    status_code=409,
                )
            if fresh is None:
                _log(f"job {job_id} gone: remembered fleet {remembered} is unreachable")
                return JSONResponse(
                    {"detail": f"job {job_id} is gone — the fleet that held it no longer exists"},
                    status_code=404,
                )
            streamed = await _stream(fresh, full, request, headers, body)
        # Only drop the mapping once the fleet has actually accepted the
        # delete; otherwise a retry cannot find the job it failed to cancel.
        #
        # A DELETE on ".../content" is an ARTIFACT ACK, not a cancellation:
        # the job still exists and stays pollable, so forgetting its owner
        # here would force a fan-out re-probe on the next poll of a job we
        # already know the home of.
        if request.method == "DELETE" and streamed.status_code < 400 and not is_content(full):
            table.forget(job_id)
        return streamed

    # New work: choose the fleet by model and task.
    payload: dict = {}
    with contextlib.suppress(ValueError, TypeError):
        payload = json.loads(body) if body else {}
    model = str(payload.get("model") or request.query_params.get("model") or "")
    task = payload.get("task") or request.query_params.get("task")
    try:
        route = table.resolve(model, task or None)
    except RoutingError as exc:
        # Understood and refused — a client error, not something to retry.
        _log(f"routing refused: model={model!r} task={task!r}: {exc}")
        return JSONResponse({"detail": str(exc)}, status_code=400)

    try:
        resp = await client.request(
            request.method,
            f"{route.url}{full}",
            headers={**headers, **content_headers(request)},
            content=body or None,
            params=dict(request.query_params),
        )
    except httpx.HTTPError as exc:
        _log(f"fleet {route.profile} unreachable on {request.method} {full}: {exc!r}")
        return JSONResponse(
            {"detail": f"fleet {route.profile} unreachable: {exc}"}, status_code=502
        )
    if resp.status_code < 400 and request.method == "POST":
        with contextlib.suppress(ValueError, KeyError, TypeError):
            table.remember(json.loads(resp.content)["id"], route.url)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=safe_headers(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


async def _find_owner(job_id: str, path: str, headers: dict) -> str | None:
    """Which fleet holds this job? Needed after a gateway restart.

    Raises AmbiguousOwner on a CONFLICT — two fleets claiming one bare id,
    or one claiming it while another could not be reached. That used to
    return None here, which the caller renders as 404: "unknown, safe to
    resubmit", the opposite of the truth, billing the generation twice.
    """
    found = await find_owner(
        job_id,
        path,
        headers,
        candidates=[route.url for route in table.routes],
        client=ctl,
        log=_log,
    )
    if found is None:
        return None
    table.remember(job_id, found[0])
    return found[0]


async def _stream(target: str, path: str, request: Request, headers: dict, body: bytes) -> Response:
    """Pass a response through untouched — results and event streams alike."""
    req = client.build_request(
        request.method,
        f"{target}{path}",
        headers=headers,
        content=body or None,
        params=dict(request.query_params),
    )
    try:
        resp = await client.send(req, stream=True)
    except httpx.HTTPError as exc:
        return JSONResponse({"detail": f"fleet unreachable: {exc}"}, status_code=502)
    # The balancer stamps every job-scoped response with the EFFECTIVE job
    # id. Learning it here is what keeps real ids out of the owner probe:
    # POST responses only ever taught the gateway TICKET ids, so every
    # queued job's post-handover polls and final download used to resolve
    # by unverified fan-out.
    effective = resp.headers.get("x-deepme-job-id")
    if effective and resp.status_code < 400:
        # Only a SUCCESSFUL response is evidence of ownership. Remembering
        # on an error bound the id to a fleet that had just denied it.
        table.remember(effective, target)
    from starlette.background import BackgroundTask

    return StreamingResponse(
        # Decoded, because _safe_headers strips content-encoding — raw
        # bytes with the header gone are undecodable garbage downstream.
        resp.aiter_bytes(),
        status_code=resp.status_code,
        headers=safe_headers(resp.headers),
        background=BackgroundTask(resp.aclose),
    )
