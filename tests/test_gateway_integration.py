"""End-to-end tests for the gateway's HTTP layer, against fake fleets.

route.py's resolution rules are unit tested. What was not tested is the layer
that uses them: the proxy, job-owner tracking across two hops, and whether the
endpoints the published contract promises are actually reachable here. That
gap is why `GET /v1/queue` — documented in INTERNAL-API.md as the way a
service layer inspects the backlog — fell through to model routing and was
refused for naming no model.

Fleets are served in-process through an httpx transport: no sockets, no
balancers, no GPUs.
"""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src" / "deepme"
VIDEO, REF = "http://fleet-video", "http://fleet-ref"

ROUTES = [
    {"model": "minimax-h3", "profile": "h3-video", "tasks": ["t2va", "fl2va"], "url": VIDEO},
    {"model": "minimax-h3", "profile": "h3-ref", "tasks": ["ref2va"], "url": REF},
]


def _resp(status: int, payload=None, *, raw: bytes | None = None) -> httpx.Response:
    """A response the gateway can stream.

    The gateway passes job-scoped responses through with `stream=True`, and a
    body built from `json=`/`content=` is already consumed — so the fakes have
    to hand back a real byte stream, exactly as a live balancer does.
    """
    data = raw if raw is not None else json.dumps(payload).encode()

    async def body():
        yield data

    headers = {"content-type": "application/json"} if raw is None else {}
    return httpx.Response(status, content=body(), headers=headers)


class FakeFleets:
    """Two balancers, each minting its own job ids."""

    def __init__(self):
        self.jobs: dict[str, str] = {}  # job id -> fleet base
        self.seen: list[tuple[str, str]] = []  # (fleet, path)
        self.n = iter(range(1, 999))
        self.down: set[str] = set()
        self.queues = {VIDEO: 0, REF: 0}
        self.capacity = {
            VIDEO: {"free": 3, "busy": 0, "replicas": 3},
            REF: {"free": 1, "busy": 0, "replicas": 1},
        }

    def handler(self, request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}"
        path = request.url.path
        self.seen.append((base, path))
        if base in self.down:
            raise httpx.ConnectError("refused")
        if path == "/v1/capacity":
            entry = dict(self.capacity[base])
            entry.setdefault("accepting", entry["free"] > 0)
            return _resp(200, entry)
        if path == "/v1/queue":
            return _resp(200, {"waiting": self.queues[base], "tickets": self.queues[base]})
        if request.method == "POST" and path == "/v1/videos":
            job_id = f"{'v' if base == VIDEO else 'r'}id_{next(self.n)}"
            self.jobs[job_id] = base
            return _resp(200, {"id": job_id, "status": "queued"})
        if path.startswith("/v1/videos/"):
            job_id = path.split("/")[3]
            if self.jobs.get(job_id) != base:
                return _resp(404, {"detail": "not here"})
            if path.endswith("/content"):
                return _resp(200, raw=b"MP4" * 100)
            return _resp(200, {"id": job_id, "status": "queued"})
        if request.method == "DELETE" and path.startswith("/v1/queue/"):
            return _resp(200, {"cancelled": True})
        return _resp(404, {"detail": "no route"})


@pytest.fixture
def fleets():
    return FakeFleets()


@pytest.fixture
def gw(fleets):
    import os

    os.environ.update(GATEWAY_ROUTES=json.dumps(ROUTES), GATEWAY_API_KEY="")
    directory = SRC / "gateway"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("gw_integration", directory / "server.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gw_integration"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(directory))
    mod.client = httpx.AsyncClient(transport=httpx.MockTransport(fleets.handler))
    return mod


# --------------------------------------------------------------------------
# Routing by model and task


def test_task_chooses_the_fleet(gw, fleets):
    """One model, two checkpoints — the task is what distinguishes them."""
    with TestClient(gw.app) as client:
        client.post("/v1/videos", json={"model": "minimax-h3", "task": "t2va"})
        client.post("/v1/videos", json={"model": "minimax-h3", "task": "ref2va"})
    posted = [f for f, p in fleets.seen if p == "/v1/videos"]
    assert posted == [VIDEO, REF]


def test_an_ambiguous_model_is_refused_with_the_choices(gw):
    with TestClient(gw.app) as client:
        r = client.post("/v1/videos", json={"model": "minimax-h3"})
    assert r.status_code == 400
    assert "ref2va" in r.json()["detail"]


def test_an_unreachable_fleet_reports_502_not_a_routing_error(gw, fleets):
    """A down fleet is an infrastructure fault the caller may retry."""
    fleets.down.add(VIDEO)
    with TestClient(gw.app) as client:
        r = client.post("/v1/videos", json={"model": "minimax-h3", "task": "t2va"})
    assert r.status_code == 502


# --------------------------------------------------------------------------
# Job identity across two hops


def test_a_job_is_followed_back_to_the_fleet_that_minted_it(gw, fleets):
    with TestClient(gw.app) as client:
        job = client.post("/v1/videos", json={"model": "minimax-h3", "task": "ref2va"}).json()
        r = client.get(f"/v1/videos/{job['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == job["id"]


def test_owner_lookup_never_probes_with_the_content_path(gw, fleets):
    """Finding the owner must not download the video from every fleet.

    The probe runs against fleets that do NOT have the job, so using the
    caller's own path would pull a full MP4 body from each one — buffered in
    memory — purely to discover a 404.
    """
    with TestClient(gw.app) as client:
        job = client.post("/v1/videos", json={"model": "minimax-h3", "task": "ref2va"}).json()
        gw.table.forget(job["id"])  # as if the gateway had restarted
        fleets.seen.clear()
        r = client.get(f"/v1/videos/{job['id']}/content")

    assert r.status_code == 200
    probes = [p for f, p in fleets.seen if f == VIDEO]
    assert probes, "the wrong fleet was asked, which is the point of a probe"
    assert all(not p.endswith("/content") for p in probes)


# --------------------------------------------------------------------------
# The published contract is reachable here


def test_the_queue_is_visible_through_the_gateway(gw, fleets):
    """INTERNAL-API.md section 4 promises this endpoint at the gateway.

    Queues are per-fleet, so it has to aggregate; before this it fell through
    to model routing and was refused for naming no model.
    """
    fleets.queues[VIDEO], fleets.queues[REF] = 3, 1
    with TestClient(gw.app) as client:
        r = client.get("/v1/queue")
    assert r.status_code == 200
    assert r.json()["waiting"] == 4
    assert {f["profile"] for f in r.json()["fleets"]} == {"h3-video", "h3-ref"}


def test_a_down_fleet_is_reported_not_counted_as_empty(gw, fleets):
    """ "Unreachable" and "nothing queued" must not look the same."""
    fleets.down.add(REF)
    fleets.queues[VIDEO] = 2
    with TestClient(gw.app) as client:
        body = client.get("/v1/queue").json()
    by_profile = {f["profile"]: f for f in body["fleets"]}
    assert by_profile["h3-ref"]["reachable"] is False
    assert by_profile["h3-video"]["reachable"] is True
    assert body["waiting"] == 2


def test_a_ticket_is_cancelled_without_the_caller_naming_a_fleet(gw, fleets):
    """The caller has only the id from its 202; the gateway knows the rest."""
    with TestClient(gw.app) as client:
        job = client.post("/v1/videos", json={"model": "minimax-h3", "task": "ref2va"}).json()
        r = client.delete(f"/v1/queue/{job['id']}")
    assert r.status_code == 200
    assert (REF, f"/v1/queue/{job['id']}") in fleets.seen


def test_models_is_answerable_with_no_parameters(gw):
    """Discovery cannot require the thing the caller is trying to discover."""
    with TestClient(gw.app) as client:
        r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["tasks"] == ["fl2va", "ref2va", "t2va"]


# --------------------------------------------------------------------------
# The scheduling interface


def test_capacity_aggregates_free_slots_per_task(gw, fleets):
    """One poll answers "where can work go", keyed by what you route on."""
    fleets.capacity = {
        VIDEO: {"free": 2, "busy": 1, "replicas": 3},
        REF: {"free": 0, "busy": 1, "replicas": 1},
    }
    with TestClient(gw.app) as client:
        body = client.get("/v1/capacity").json()
    assert body["free"] == 2
    assert sorted(body["accepting"]) == ["fl2va", "t2va"], "ref2va is full"
    by_profile = {f["profile"]: f for f in body["fleets"]}
    assert by_profile["h3-video"]["free"] == 2
    assert by_profile["h3-ref"]["free"] == 0


def test_an_unreachable_fleet_never_looks_like_free_capacity(gw, fleets):
    """The dangerous failure: a scheduler submitting into a dead fleet.

    Omitting it, or defaulting `free` to anything but 0, sends work to a
    fleet that cannot take it. Reporting 0 free plus reachable=false is the
    safe reading and still says why.
    """
    fleets.capacity = {
        VIDEO: {"free": 3, "busy": 0, "replicas": 3},
        REF: {"free": 9, "busy": 0, "replicas": 1},
    }
    fleets.down.add(REF)
    with TestClient(gw.app) as client:
        body = client.get("/v1/capacity").json()
    by_profile = {f["profile"]: f for f in body["fleets"]}
    assert by_profile["h3-ref"]["free"] == 0
    assert by_profile["h3-ref"]["reachable"] is False
    assert by_profile["h3-ref"]["accepting"] is False
    assert body["free"] == 3, "only the reachable fleet counts"
    assert "ref2va" not in body["accepting"]


# --------------------------------------------------------------------------
# The pricing signal


PRICING = {
    "currency": "USD",
    "unit": "per_output_second",
    "lock_at": "submitted",
    "tasks": {
        "t2va": {"cost": 0.0356, "spot_floor": 0.05, "market": 0.13},
        "fl2va": {"cost": 0.0394, "spot_floor": 0.06, "market": 0.13},
        "ref2va": {"cost": 0.0557, "spot_floor": 0.08, "market": 0.26},
    },
    "surge": [
        {"queued_per_replica": 0.0, "multiplier": 1.0},
        {"queued_per_replica": 1.0, "multiplier": 1.8},
        {"queued_per_replica": 3.0, "multiplier": 2.3},
    ],
}


@pytest.fixture
def priced_gw(fleets, monkeypatch):
    monkeypatch.setenv("GATEWAY_PRICING", json.dumps(PRICING))
    import os

    os.environ.update(GATEWAY_ROUTES=json.dumps(ROUTES), GATEWAY_API_KEY="")
    directory = SRC / "gateway"
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("gw_priced", directory / "server.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["gw_priced"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(directory))
    mod.client = httpx.AsyncClient(transport=httpx.MockTransport(fleets.handler))
    return mod


def test_idle_capacity_quotes_the_spot_floor(priced_gw, fleets):
    """Free slots and an empty queue -> the floor, not the market rate."""
    fleets.capacity = {
        VIDEO: {"free": 2, "busy": 1, "replicas": 3, "queued": 0},
        REF: {"free": 1, "busy": 0, "replicas": 1, "queued": 0},
    }
    with TestClient(priced_gw.app) as client:
        body = client.get("/v1/pricing").json()
    assert body["tasks"]["t2va"]["tier"] == "spot"
    assert body["tasks"]["t2va"]["price_per_output_second"] == 0.05
    assert body["tasks"]["ref2va"]["price_per_output_second"] == 0.08


def test_a_full_fleet_quotes_market_and_a_queue_quotes_surge(priced_gw, fleets):
    """The tier is a function of queue depth per replica, per fleet."""
    fleets.capacity = {
        VIDEO: {"free": 0, "busy": 3, "replicas": 3, "queued": 4},  # 1.33/replica
        REF: {"free": 0, "busy": 1, "replicas": 1, "queued": 0},  # full, no queue
    }
    with TestClient(priced_gw.app) as client:
        body = client.get("/v1/pricing").json()
    assert body["tasks"]["ref2va"]["tier"] == "market"
    assert body["tasks"]["ref2va"]["price_per_output_second"] == 0.26
    assert body["tasks"]["t2va"]["tier"] == "surge-x1.8"
    assert body["tasks"]["t2va"]["price_per_output_second"] == pytest.approx(0.234)


def test_an_unreachable_fleet_is_not_quoted(priced_gw, fleets):
    """A price on a fleet that cannot serve is an invitation to bill for
    nothing — the task is marked unavailable instead."""
    fleets.down.add(REF)
    fleets.capacity = {
        VIDEO: {"free": 3, "busy": 0, "replicas": 3, "queued": 0},
        REF: {"free": 1, "busy": 0, "replicas": 1, "queued": 0},
    }
    with TestClient(priced_gw.app) as client:
        body = client.get("/v1/pricing").json()
    assert body["tasks"]["ref2va"] == {"available": False}
    assert body["tasks"]["t2va"]["available"] is True


def test_the_billing_contract_travels_with_every_quote(priced_gw, fleets):
    """lock_at=submitted is the rule that stops surge arbitrage; a billing
    layer needs it in-band, not in a doc it may not have read."""
    with TestClient(priced_gw.app) as client:
        body = client.get("/v1/pricing").json()
    assert body["lock_at"] == "submitted"
    assert body["unit"] == "per_output_second"


# --------------------------------------------------------------------------
# Audit fixes: surge depth, quote clobbering, health honesty, queue shape


def test_surge_depth_counts_only_replicas_that_can_drain_the_queue(priced_gw, fleets):
    """Dead and draining replicas diluted the depth, underquoting surge
    exactly when capacity was scarce (every resize drains replicas for up
    to an hour)."""
    fleets.capacity = {
        # 3 attached but only 1 accepting (2 draining): 4 queued over 1
        # accepting = depth 4.0 -> top rung. Diluted over all 3 it reads
        # 1.33 -> the x1.8 rung, one tier cheaper.
        VIDEO: {"free": 0, "busy": 1, "replicas": 3, "accepting_replicas": 1, "queued": 4},
        REF: {"free": 1, "busy": 0, "replicas": 1, "accepting_replicas": 1, "queued": 0},
    }
    with TestClient(priced_gw.app) as client:
        body = client.get("/v1/pricing").json()
    assert body["tasks"]["t2va"]["tier"] == "surge-x2.3", "depth 4.0, not the diluted 1.33"


def test_an_unreachable_fleet_cannot_clobber_a_live_quote(priced_gw, fleets):
    """The day two fleets serve one task, pricing must agree with routing
    (first resolvable fleet) — not report the last fleet's failure."""
    priced_gw.table = priced_gw.build(
        [
            {"model": "minimax-h3", "profile": "h3-video", "tasks": ["t2va"], "url": VIDEO},
            {
                "model": "minimax-h3",
                "profile": "h3-video-b",
                "tasks": ["t2va"],
                "url": REF,
            },  # second fleet, same task
        ]
    )
    fleets.down.add(REF)
    fleets.capacity = {VIDEO: {"free": 1, "busy": 0, "replicas": 1, "queued": 0}}
    with TestClient(priced_gw.app) as client:
        body = client.get("/v1/pricing").json()
    assert body["tasks"]["t2va"]["available"] is True


def test_pricing_carries_the_duration_curve_and_table_identity(priced_gw):
    """An app layer pricing by duration needs the curve in-band, and a
    billing layer needs table_hash to detect a stale frozen table."""
    with TestClient(priced_gw.app) as client:
        body = client.get("/v1/pricing").json()
    assert "duration_curve" in body
    assert "table_hash" in body


def test_gateway_health_is_the_health_of_the_path(gw, fleets):
    """200 with every balancer down reported a dead platform as healthy."""
    with TestClient(gw.app) as client:
        assert client.get("/health").status_code == 200
        fleets.down.update({VIDEO, REF})
        r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["reachable_fleets"] == 0


def test_queue_keeps_the_documented_shape_when_a_fleet_is_down(gw, fleets):
    """The example in INTERNAL-API shows `waiting` on every entry; a client
    reading fleets[i].waiting must not break precisely in the failure case
    the doc is teaching them to handle."""
    fleets.down.add(REF)
    with TestClient(gw.app) as client:
        body = client.get("/v1/queue").json()
    down = next(f for f in body["fleets"] if f["profile"] == "h3-ref")
    assert down["waiting"] == 0
    assert down["reachable"] is False


# --------------------------------------------------------------------------
# Reload in place, stale-owner healing


def test_reload_swaps_routes_without_losing_job_owners(gw, fleets):
    """A restart severs every in-flight stream; reload must not — and jobs
    submitted before the swap must keep resolving after it."""
    with TestClient(gw.app) as client:
        job = client.post("/v1/videos", json={"model": "minimax-h3", "task": "ref2va"}).json()
        r = client.post(
            "/gateway/reload",
            json={
                "routes": [
                    {
                        "model": "minimax-h3",
                        "profile": "h3-video",
                        "tasks": ["t2va", "fl2va", "ref2va"],
                        "url": REF,
                    }
                ],
                "pricing": {"tasks": {}, "surge": [], "lock_at": "submitted"},
            },
        )
        assert r.status_code == 200
        assert r.json()["owner_map_preserved"] >= 1
        followed = client.get(f"/v1/videos/{job['id']}")
        assert followed.status_code == 200, "old job resolves after the swap"


def test_a_stale_owner_heals_to_an_honest_404_not_eternal_502(gw, fleets):
    """After a resize removes a fleet, its remembered jobs answered 502 —
    which the contract calls retryable, so clients retried a dead id
    forever. The mapping is dropped, re-probed, and the truth is 404."""
    with TestClient(gw.app) as client:
        job = client.post("/v1/videos", json={"model": "minimax-h3", "task": "ref2va"}).json()
        fleets.down.add(REF)
        del fleets.jobs[job["id"]]  # the fleet is gone, job with it
        r = client.get(f"/v1/videos/{job['id']}")
    assert r.status_code == 404
    assert "no longer exists" in r.json()["detail"]


def test_jobs_and_metrics_aggregate_across_fleets(gw, fleets):
    """One operator endpoint for the platform; metrics gain fleet labels."""

    def with_obs(request):
        base = f"{request.url.scheme}://{request.url.host}"
        if request.url.path == "/v1/jobs":
            return _resp(
                200, {"running": [{"job_id": f"j-{base[-5:]}"}], "queued": [], "recent": []}
            )
        if request.url.path == "/metrics":
            return _resp(200, raw=b"deepme_slots_free 2\n")
        return fleets.handler(request)

    gw.client = httpx.AsyncClient(transport=httpx.MockTransport(with_obs))
    with TestClient(gw.app) as client:
        body = client.get("/v1/jobs").json()
        assert {f["profile"] for f in body["fleets"]} == {"h3-video", "h3-ref"}
        text = client.get("/metrics").text
    assert 'deepme_slots_free{fleet="h3-video"} 2' in text
    assert 'deepme_slots_free{fleet="h3-ref"} 2' in text


def test_the_gateway_firehose_tags_fleets_and_reports_outages(gw, fleets):
    """One subscription for the platform: events gain a fleet tag, and a
    fleet that cannot be streamed is itself an event, not silence."""

    async def sse_body():
        yield b'data: {"seq": 1, "type": "started", "job_id": "j1"}\n\n'
        import asyncio

        await asyncio.sleep(3600)  # hold the stream open like a real one

    def with_events(request):
        base = f"{request.url.scheme}://{request.url.host}"
        if request.url.path == "/v1/events":
            if base == REF:
                raise httpx.ConnectError("down")
            return httpx.Response(200, content=sse_body())
        return fleets.handler(request)

    gw.client = httpx.AsyncClient(transport=httpx.MockTransport(with_events))
    with TestClient(gw.app) as client, client.stream("GET", "/v1/events?limit=2") as r:
        seen = []
        for line in r.iter_lines():
            if line.startswith("data: "):
                seen.append(json.loads(line[len("data: ") :]))
                if len(seen) >= 2:
                    break
    kinds = {(e.get("type"), e.get("fleet")) for e in seen}
    assert ("started", "h3-video") in kinds
    assert ("fleet_down", "h3-ref") in kinds
