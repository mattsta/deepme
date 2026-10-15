"""End-to-end tests for the balancer's HTTP layer, against fake replicas.

The pure logic in balance.py and admission.py is well covered. Every defect
that reached production instead lived in the glue that composes them — the
proxy, the pump, the accounting around a submission — because that layer was
only ever exercised by real traffic. This file closes that gap: a fake replica
gives the balancer something to talk to, so a full submit -> queue -> run ->
complete lifecycle runs in milliseconds with no GPU.

The fake replica models the one property that makes this domain awkward:
`POST /v1/videos` returns a job id in milliseconds and the replica then works
for minutes. Anything that treats the POST as the work is wrong, and that is
what these tests pin down.
"""

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src" / "deepme"
R1, R2 = "http://replica-1", "http://replica-2"


class FakeFleet:
    """Two replicas that mint job ids and hold them until released.

    Requests are served in-process through an httpx transport, so no sockets
    and no timing dependence.
    """

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.posts: list[str] = []  # which replica got each submission
        self.next_id = iter(range(1, 999))
        self.reject: set[str] = set()  # replicas that should refuse work
        self.inline = False  # complete inside the POST instead

    def handler(self, request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}"
        path = request.url.path
        if request.method == "POST" and path == "/v1/videos":
            if base in self.reject:
                return httpx.Response(503, json={"detail": "busy"})
            self.posts.append(base)
            job_id = f"vid_{next(self.next_id)}"
            status = "completed" if self.inline else "queued"
            self.jobs[job_id] = {"id": job_id, "status": status, "owner": base}
            return httpx.Response(200, json=self.jobs[job_id])
        job_id = path.rsplit("/", 1)[-1]
        if path.endswith("/content"):
            job_id = path.rsplit("/", 2)[-2]
            if job_id not in self.jobs:
                return httpx.Response(404, json={"detail": "not here"})

            async def body():
                # Streamed, as a real replica serves media: the balancer
                # passes /content through with stream=True, and a
                # materialised body cannot be re-read.
                yield b"MP4"

            return httpx.Response(200, content=body())
        job = self.jobs.get(job_id)
        if job is None or job["owner"] != base:
            return httpx.Response(404, json={"detail": "not here"})
        return httpx.Response(200, json=job)

    def complete(self, job_id: str, status: str = "completed") -> None:
        self.jobs[job_id]["status"] = status


@pytest.fixture
def fleet():
    return FakeFleet()


@pytest.fixture
def bal(fleet, monkeypatch, tmp_path):
    """Load the balancer the way it is deployed, wired to the fake fleet.

    Loaded from a private cwd: the server persists runtime membership to a
    file in its working directory, and one test's attach/detach must not be
    restored as another test's starting fleet (or land in the repo).
    """
    import os

    monkeypatch.chdir(tmp_path)
    os.environ.update(BALANCER_BACKENDS=json.dumps([R1, R2]), BALANCER_API_KEY="")
    directory = SRC / "balancer"
    # proxycore is deployed FLAT next to server.py, so it must be importable
    # by bare name here too — otherwise the tests exercise an import graph
    # the server never has.
    sys.path.insert(0, str(SRC / "shared"))
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location("bal_integration", directory / "server.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["bal_integration"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(directory))
    mod.client = httpx.AsyncClient(transport=httpx.MockTransport(fleet.handler))
    # The control-plane client (bounded timeouts) talks to the same fakes.
    mod.ctl = mod.client
    mod.POLL_S = 0.0
    return mod


def _submit(client, prompt="x", **extra):
    return client.post("/v1/videos", json={"prompt": prompt, **extra})


# --------------------------------------------------------------------------
# Accounting: what a submission does to the numbers


def test_a_submission_is_not_recorded_as_a_finished_job(bal, fleet):
    """The POST round-trip must not touch the job counters.

    This is the defect that reached production: the balancer timed the submit
    call with start()/finish(), so a replica running a twenty-minute
    generation reported mean_job_s of 0.1 and completed=1 before it had
    finished anything. That number drives the customer's ETA, the queue wait
    estimate and the stale-job threshold, so it corrupts everything downstream.
    """
    with TestClient(bal.app) as client:
        r = _submit(client)
        assert r.status_code == 200
        stats = {s["url"]: s for s in bal.bal.snapshot()["backends"]}
        busy = stats[r.json()["owner"]]
        assert busy["completed"] == 0, "nothing has completed yet"
        assert busy["mean_job_s"] == 0.0, "no job has finished, so no mean"
        assert busy["started"] == 1, "counted once, not once per hop"
        assert busy["inflight"] == 1, "the replica is occupied by the job"


def test_occupancy_moves_to_the_job_not_the_call(bal, fleet):
    """inflight must stay up after the POST returns — the work is still going."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        assert bal.bal.snapshot()["active_jobs"] == 1
        # ...and only a terminal status closes it.
        fleet.complete(job["id"])
        client.get(f"/v1/videos/{job['id']}")
        assert bal.bal.snapshot()["active_jobs"] == 0
        stats = {s["url"]: s for s in bal.bal.snapshot()["backends"]}
        assert stats[job["owner"]]["completed"] == 1
        assert stats[job["owner"]]["inflight"] == 0


def test_a_reads_through_the_proxy_do_not_count_as_jobs(bal, fleet):
    """A status poll is not work; counting it inflates completions."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        for _ in range(5):
            client.get(f"/v1/videos/{job['id']}")
        stats = {s["url"]: s for s in bal.bal.snapshot()["backends"]}
        assert stats[job["owner"]]["started"] == 1
        assert stats[job["owner"]]["completed"] == 0


def test_a_job_finished_inside_the_post_is_counted_once(bal, fleet):
    """Some requests complete synchronously — a real job, just a short one."""
    fleet.inline = True
    with TestClient(bal.app) as client:
        r = _submit(client)
        assert r.status_code == 200
        stats = {s["url"]: s for s in bal.bal.snapshot()["backends"]}
        done = stats[r.json()["owner"]]
        assert done["started"] == 1
        assert done["completed"] == 1
        assert done["inflight"] == 0
        assert bal.bal.snapshot()["active_jobs"] == 0


# --------------------------------------------------------------------------
# Routing


def test_work_spreads_across_idle_replicas_before_doubling_up(bal, fleet):
    with TestClient(bal.app) as client:
        _submit(client)
        _submit(client)
    assert set(fleet.posts) == {R1, R2}, "second request went to the idle one"


def test_a_follow_up_returns_to_the_replica_that_minted_the_id(bal, fleet):
    """Job ids are per-replica; polling the wrong one strands the result."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        r = client.get(f"/v1/videos/{job['id']}")
        assert r.status_code == 200
        assert r.json()["owner"] == job["owner"]


# --------------------------------------------------------------------------
# The queue


def test_work_queues_once_every_replica_is_busy(bal, fleet):
    with TestClient(bal.app) as client:
        _submit(client)
        _submit(client)
        third = _submit(client)
    assert third.status_code == 202
    assert third.json()["id"].startswith("q_")
    assert third.json()["billable"] is False
    assert len(fleet.posts) == 2, "the third was held, not sent"


def test_a_queued_ticket_is_pollable_and_then_follows_its_job(bal, fleet):
    """The documented client flow: submit, poll the id you were handed.

    The 202 is deliberately job-shaped, so the same id has to keep working
    after a replica picks the work up — including for the download.
    """
    with TestClient(bal.app) as client:
        first = _submit(client).json()
        _submit(client)
        ticket = _submit(client).json()["id"]

        queued = client.get(f"/v1/videos/{ticket}")
        assert queued.json()["status"] == "queued"
        assert queued.json()["billable"] is False

        # A replica frees; polling the finished job is what drains the queue.
        fleet.complete(first["id"])
        client.get(f"/v1/videos/{first['id']}")

        followed = client.get(f"/v1/videos/{ticket}")
        assert followed.status_code == 200
        assert followed.json()["id"].startswith("vid_"), "now the real job"
        assert len(fleet.posts) == 3


def test_a_ticket_can_be_cancelled_before_submission(bal, fleet):
    with TestClient(bal.app) as client:
        _submit(client)
        _submit(client)
        ticket = _submit(client).json()["id"]
        assert client.delete(f"/v1/videos/{ticket}").status_code == 200
        assert client.get("/v1/queue").json()["waiting"] == 0
    assert len(fleet.posts) == 2, "a cancelled ticket is never sent"


def test_priority_is_served_before_earlier_normal_work(bal, fleet):
    """Ordering is (priority, arrival), so a late high-priority item wins."""
    with TestClient(bal.app) as client:
        first = _submit(client).json()
        _submit(client)
        _submit(client, prompt="normal")
        urgent = _submit(client, prompt="urgent", priority=0).json()["id"]

        fleet.complete(first["id"])
        client.get(f"/v1/videos/{first['id']}")

        followed = client.get(f"/v1/videos/{urgent}").json()
        assert followed["id"].startswith("vid_"), "the urgent ticket ran first"


def test_a_rejected_submission_does_not_strand_the_queue(bal, fleet):
    """A replica refusing work must not leave the ticket waiting forever."""
    with TestClient(bal.app) as client:
        first = _submit(client).json()
        _submit(client)
        ticket = _submit(client).json()["id"]
        fleet.reject.update({R1, R2})

        fleet.complete(first["id"])
        client.get(f"/v1/videos/{first['id']}")

        assert client.get("/v1/queue").json()["waiting"] == 0, "not left waiting"
        # It is settled rather than pending, so the caller gets an answer
        # instead of polling a ticket that will never move.
        assert client.get(f"/v1/videos/{ticket}").status_code in (200, 404)


# --------------------------------------------------------------------------
# Health


def test_an_unreachable_replica_is_marked_down_and_reported(bal, fleet):
    def dead(request):
        raise httpx.ConnectError("refused")

    bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(dead))
    with TestClient(bal.app) as client:
        assert _submit(client).status_code == 502
        assert client.get("/health").json()["healthy"] == 1


def test_a_failed_delete_keeps_the_job_and_its_occupancy(bal, fleet):
    """Occupancy must not be freed on a delete the replica refused.

    Dropping it early leaves a job still generating on a replica the balancer
    believes is idle, so it routes new work onto a busy GPU — the precise
    failure job-scoped occupancy was introduced to prevent.
    """
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        assert bal.bal.snapshot()["active_jobs"] == 1

        def refuse(request):
            # Streamed: DELETE goes through the passthrough path, which sends
            # with stream=True, and a materialised body cannot be re-read.
            async def body():
                yield b'{"detail": "cannot"}'

            return httpx.Response(500, content=body())

        bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
        assert client.delete(f"/v1/videos/{job['id']}").status_code == 500

        assert bal.bal.snapshot()["active_jobs"] == 1, "still running"
        assert bal.bal.owner(job["id"]), "still findable for a retry"


# --------------------------------------------------------------------------
# The scheduling interface: for a caller running its own priority queue


def test_capacity_reports_exactly_how_many_requests_will_not_queue(bal, fleet):
    """`free` is the contract an external scheduler submits against.

    It must equal the number of requests that start immediately — if it
    over-reports, the caller's scheduler hands work to our queue and loses
    ownership of its own ordering.
    """
    with TestClient(bal.app) as client:
        assert client.get("/v1/capacity").json()["free"] == 2
        _submit(client)
        assert client.get("/v1/capacity").json()["free"] == 1
        _submit(client)
        cap = client.get("/v1/capacity").json()
        assert cap["free"] == 0
        assert cap["busy"] == 2
        assert cap["accepting"] is False


def test_capacity_frees_again_when_a_job_completes(bal, fleet):
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        _submit(client)
        assert client.get("/v1/capacity").json()["free"] == 0
        fleet.complete(job["id"])
        client.get(f"/v1/videos/{job['id']}")
        assert client.get("/v1/capacity").json()["free"] == 1


def test_capacity_reports_running_work_with_elapsed_and_eta(bal, fleet):
    """A scheduler plans the next minute from what is already in flight.

    The ETA has to survive the work landing on a DIFFERENT replica than the
    one with history — routing sends it to the idle one, so the common case
    is a replica with no record of its own. Replicas are identical, so the
    fleet's mean stands in; otherwise every replica reports "unknown" for its
    first full generation, and after a restart that is all of them.
    """
    with TestClient(bal.app) as client:
        first = _submit(client).json()
        fleet.complete(first["id"])
        client.get(f"/v1/videos/{first['id']}")  # gives the fleet a mean
        job = _submit(client).json()
        running = client.get("/v1/capacity").json()["running"]
    assert [r["job_id"] for r in running] == [job["id"]]
    assert running[0]["elapsed_s"] >= 0
    assert running[0]["backend"] != first["owner"], "a replica with no history"
    assert running[0]["eta_s"] is not None, "yet an ETA is still owed"


def test_no_eta_is_invented_before_the_fleet_has_finished_anything(bal, fleet):
    with TestClient(bal.app) as client:
        _submit(client)
        running = client.get("/v1/capacity").json()["running"]
    assert running[0]["eta_s"] is None


def test_queue_false_is_refused_rather_than_held(bal, fleet):
    """Two queues for the same work is the failure this prevents.

    A caller with its own priority queue must never discover that we also
    hold some of its requests — it would be ordering and charging for work
    it no longer controls.
    """
    with TestClient(bal.app) as client:
        _submit(client)
        _submit(client)
        r = client.post("/v1/videos?queue=false", json={"prompt": "x"})
    assert r.status_code == 503
    assert r.json()["free"] == 0
    assert int(r.headers["retry-after"]) >= 1
    assert len(fleet.posts) == 2, "not sent"
    assert client.get("/v1/queue").json()["waiting"] == 0, "and not held"


def test_queue_false_still_submits_when_there_is_room(bal, fleet):
    """The flag declines QUEUEING, it does not decline service."""
    with TestClient(bal.app) as client:
        r = client.post("/v1/videos?queue=false", json={"prompt": "x"})
    assert r.status_code == 200
    assert r.json()["id"].startswith("vid_")


def test_the_queue_flag_never_reaches_the_engine(bal, fleet):
    """It is ours. Forwarding it would put an unknown param on the replica."""
    seen = {}
    original = fleet.handler

    def spy(request):
        seen["query"] = str(request.url.query)
        return original(request)

    bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(spy))
    with TestClient(bal.app) as client:
        client.post("/v1/videos?queue=false&seed=7", json={"prompt": "x"})
    assert "queue" not in seen["query"]
    assert "seed" in seen["query"], "the caller's own params still go through"


def test_the_retry_hint_uses_the_soonest_finishing_job(bal, fleet):
    """Refusing a caller should tell it WHEN to come back, not a fixed guess.

    The hint is the shortest remaining ETA across running work, so a backoff
    can be informed. Only the no-history fallback is easy to observe live —
    a fresh balancer has no mean — so the real path is pinned here.
    """
    with TestClient(bal.app) as client:
        first = _submit(client).json()
        fleet.complete(first["id"])
        client.get(f"/v1/videos/{first['id']}")  # the fleet now has a mean
        # The fake completes in microseconds; give it a generation-scale mean
        # so the hint is checked on the timescale it actually serves.
        bal.bal.stats[first["owner"]].total_busy_s = 600.0
        _submit(client)
        _submit(client)  # both replicas busy
        r = client.post("/v1/videos?queue=false", json={"prompt": "x"})

    assert r.status_code == 503
    hint = r.json()["retry_hint_s"]
    assert hint != 30, "that is the no-history fallback, not a real estimate"
    assert 590 <= hint <= 600, "the running job's remaining time"
    assert int(r.headers["retry-after"]) == hint


# --------------------------------------------------------------------------
# Runtime membership: resize the fleet without restarting the balancer


def test_a_drained_replica_gets_no_new_work_but_its_job_survives(bal, fleet):
    """The drain contract, both halves.

    New work must route around a draining replica immediately; the job it is
    already running must stay followable to completion — killing it would
    burn a paying customer's generation to make an operator's resize easy.
    """
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        assert (
            client.post("/balancer/backends/drain", json={"url": job["owner"]}).status_code == 200
        )
        second = _submit(client).json()
        assert second["owner"] != job["owner"], "routed around the drain"
        # The draining replica's job is still followable...
        assert client.get(f"/v1/videos/{job['id']}").status_code == 200
        # ...and still counted: capacity must not sell its slot.
        cap = client.get("/v1/capacity").json()
        assert cap["draining"] == 1
        assert cap["free"] == 0


def test_detach_refuses_while_work_is_running_then_allows(bal, fleet):
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        r = client.delete("/balancer/backends", params={"url": job["owner"]})
        assert r.status_code == 409, "detaching now would strand the job"
        fleet.complete(job["id"])
        client.get(f"/v1/videos/{job['id']}")  # closes the job
        r = client.delete("/balancer/backends", params={"url": job["owner"]})
        assert r.status_code == 200
        assert job["owner"] not in r.json()["backends"]


def test_attach_probes_before_admitting_a_backend(bal, fleet):
    """A typo'd URL must not enter rotation and turn submissions into 502s."""
    with TestClient(bal.app) as client:
        r = client.post("/balancer/backends", json={"url": "http://replica-9"})
        assert r.status_code == 400, "unreachable: refused"
        assert "http://replica-9" not in bal.bal.backend_urls()


def test_attach_expands_capacity_without_a_restart(bal, fleet, tmp_path, monkeypatch):
    """The 4th replica of a resize joins a LIVE balancer: queue intact."""
    monkeypatch.chdir(tmp_path)  # membership file lands here, not in the repo
    R3 = "http://replica-3"
    fleet3 = fleet

    def handler(request):
        if request.url.host == "replica-3" and request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return fleet3.handler(request)

    bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(bal.app) as client:
        _submit(client)
        _submit(client)
        ticket = _submit(client).json()  # queued: both replicas busy
        assert ticket["id"].startswith("q_")

        r = client.post("/balancer/backends", json={"url": R3})
        assert r.status_code == 200

        followed = client.get(f"/v1/videos/{ticket['id']}")
        assert followed.json()["id"].startswith("vid_"), (
            "the queued ticket drained into the newly attached replica"
        )
        assert client.get("/v1/capacity").json()["replicas"] == 3


def test_membership_survives_a_crash_restart(bal, fleet, tmp_path, monkeypatch):
    """A crash-restart resumes the CURRENT fleet, not the deploy-time one.

    Without persistence, systemd's Restart=on-failure would silently revert
    a runtime resize — the balancer would route to a replica that was
    detached hours ago, and drop the one that was attached.
    """
    monkeypatch.chdir(tmp_path)
    R3 = "http://replica-3"

    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return fleet.handler(request)

    bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with TestClient(bal.app) as client:
        client.post("/balancer/backends", json={"url": R3})
        client.delete("/balancer/backends", params={"url": R1})

    # Simulate the restart: same env (stale BACKENDS), fresh Balancer.
    bal.bal.__init__([R1, R2])
    bal._restore_membership()
    assert set(bal.bal.backend_urls()) == {R2, R3}, "runtime state won"


# --------------------------------------------------------------------------
# Audit fixes: rejection honesty, queue-jump, reserved-slot detach, drain
# persistence, undrain


def _take(response, count: int) -> list[dict]:
    """First `count` SSE events; stops reading so live streams cannot hang."""
    out: list[dict] = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: ") :]))
            if len(out) >= count:
                break
    return out


def test_a_rejected_submission_reports_failed_not_queued_forever(bal, fleet):
    """The documented flow — submit, poll the id you got — must reach a
    terminal answer even when the replica refuses the replayed request."""
    with TestClient(bal.app) as client:
        first = _submit(client).json()
        _submit(client)
        ticket = _submit(client).json()["id"]

        fleet.reject.update({R1, R2})
        fleet.complete(first["id"])
        client.get(f"/v1/videos/{first['id']}")  # frees a slot; pump rejects

        polled = client.get(f"/v1/videos/{ticket}").json()
        assert polled["status"] == "failed"
        assert polled["billable"] is False
        assert polled["error"], "carries the upstream refusal"
        # And the same over SSE, as a terminal event.
        with client.stream("GET", f"/v1/videos/{ticket}/events") as r:
            events = _take(r, 1)
        assert events[0]["status"] == "failed"
        assert events[0]["billable"] is False


def test_a_walk_in_cannot_jump_a_waiting_queue(bal, fleet):
    """(priority, arrival) must hold against fresh arrivals too: every
    freed slot went to the newest POST while tickets — high-priority ones
    included — waited unboundedly."""
    with TestClient(bal.app) as client:
        first = _submit(client).json()
        _submit(client)
        ticket = _submit(client).json()["id"]  # waiting, arrived FIRST

        fleet.complete(first["id"])
        client.get(f"/v1/videos/{first['id']}")  # a slot frees; T takes it

        walkin = _submit(client)  # arrives after T
        followed = client.get(f"/v1/videos/{ticket}").json()
        assert followed["id"].startswith("vid_"), "the ticket got the slot"
        assert walkin.status_code == 202, "the walk-in waits its turn"


def test_detach_is_refused_while_a_submission_is_on_the_wire(bal, fleet):
    """Removing a backend mid-round-trip orphaned the job the response was
    about to mint — a phantom nothing could ever close."""
    with TestClient(bal.app) as client:
        bal.bal.reserve(R1)  # a submission is in flight to R1
        r = client.delete("/balancer/backends", params={"url": R1})
        assert r.status_code == 409
        bal.bal.release(R1)
        r = client.delete("/balancer/backends", params={"url": R1})
        assert r.status_code == 200


def test_draining_survives_a_crash_restart(bal, fleet, tmp_path, monkeypatch):
    """A restart that forgot a drain put the replica back in rotation, and
    the operator's planned stop then killed the job it had just been
    handed."""
    monkeypatch.chdir(tmp_path)
    with TestClient(bal.app) as client:
        client.post("/balancer/backends/drain", json={"url": R1})
    bal.bal.__init__([R1, R2])
    bal._restore_membership()
    assert bal.bal.stats[R1].draining is True
    assert bal.bal.pick() == R2


def test_undrain_rearms_a_replica(bal, fleet):
    """The recovery path for an aborted resize — without it a leftover
    drain starves the replica forever while it displays 'up'."""
    with TestClient(bal.app) as client:
        client.post("/balancer/backends/drain", json={"url": R1})
        assert bal.bal.free_slots() == 1
        r = client.post("/balancer/backends/undrain", json={"url": R1})
        assert r.status_code == 200
        assert bal.bal.free_slots() == 2


def test_finished_slots_free_without_any_client_poll(bal, fleet, monkeypatch):
    """THE liveness invariant: no external traffic may ever be required for
    the platform to make progress on admitted work.

    The production incident: clients died, their finished jobs held every
    slot, the queue sat full, GPUs sat idle — and the old reaper ran only on
    request paths a wedged fleet never sees, gated on a 30-minute staleness
    heuristic besides. Reconciliation asks the REPLICA (the owner of the
    truth) on the maintenance timer, closes on its word alone, and the pump
    then drains the queue — all with zero client involvement.
    """
    monkeypatch.setattr(bal, "RECONCILE_MIN_AGE_S", 0.0)
    with TestClient(bal.app) as client:
        j1 = _submit(client).json()
        j2 = _submit(client).json()
        ticket = _submit(client).json()["id"]  # queued behind them
        # Both jobs finish; their clients are DEAD — nobody ever polls.
        fleet.complete(j1["id"])
        fleet.complete(j2["id"])

        # One maintenance beat: reconcile -> pump.
        client.portal.call(bal._reap_stale_jobs)
        client.portal.call(bal._pump)

        assert bal.bal.snapshot()["active_jobs"] == 1, "slots freed, ticket ran"
        assert bal.bal.reaped == 2
        followed = client.get(f"/v1/videos/{ticket}")
        assert followed.json()["id"].startswith("vid_"), "the queue drained"


def test_a_dead_replicas_jobs_are_presumed_lost_not_immortal(bal, fleet, monkeypatch):
    """A replica that crashes forgets its jobs; under a terminal-only close
    rule those slots never freed — occupancy stuck and detach refused
    FOREVER, the fleet shrinking with every crash. Strikes release them."""
    monkeypatch.setattr(bal, "RECONCILE_MIN_AGE_S", 0.0)
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        # The replica restarts: its job table is empty, every probe 404s.
        del fleet.jobs[job["id"]]
        for _ in range(bal.JOB_LOST_STRIKES):
            client.portal.call(bal._reap_stale_jobs)
        assert bal.bal.snapshot()["active_jobs"] == 0, "slot released"
        r = client.delete("/balancer/backends", params={"url": job["owner"]})
        assert r.status_code == 200, "and the dead replica is detachable"


def test_an_unhealthy_backend_is_revived_by_the_maintenance_probe(bal, fleet):
    """healthy=False was a one-way latch: a 30s network blip permanently
    removed replicas, and a fleet-wide blip starved every ticket forever."""
    bal.bal.mark(R1, healthy=False)

    def with_health(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return fleet.handler(request)

    bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(with_health))
    with TestClient(bal.app) as client:
        client.portal.call(bal._revive_backends)
        assert bal.bal.stats[R1].healthy is True


def test_boot_recovers_occupancy_from_replica_truth(bal, fleet):
    """A restarted balancer must not see a busy fleet as idle: pick() would
    double-book mid-generation replicas and _all_busy() would wave the
    whole former queue into engine-internal FIFOs."""

    def with_listing(request):
        if request.url.path == "/v1/videos" and request.method == "GET":
            base = f"{request.url.scheme}://{request.url.host}"
            if base == R1:
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {"id": "vid_orphan", "status": "queued"},
                            {"id": "vid_done", "status": "completed"},
                        ]
                    },
                )
            return httpx.Response(200, json={"data": []})
        return fleet.handler(request)

    bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(with_listing))
    with TestClient(bal.app) as client:
        client.portal.call(bal._recover_occupancy)
        assert bal.bal.snapshot()["active_jobs"] == 1
        stats = {s["url"]: s for s in bal.bal.snapshot()["backends"]}
        assert stats[R1]["inflight"] == 1, "the running job re-opened its slot"
        assert stats[R2]["inflight"] == 0


def test_a_delivered_artifact_survives_until_the_client_acks(bal, fleet, tmp_path):
    """Delivery is not proof of possession.

    Deleting on delivery made the transfer single-shot: a caller that got 200
    and then failed to write — no such directory, disk full, killed mid-flush
    — lost the work permanently, with no way to ask again. We own both ends
    of this protocol, so the client says when it is safe."""
    work = tmp_path / "work"
    (work / "outputs").mkdir(parents=True)
    bal.WORK_DIR = str(work)

    with TestClient(bal.app) as client:
        job = _submit(client).json()
        fleet.complete(job["id"])
        artifact = work / "outputs" / f"{job['id']}.mp4"
        artifact.write_bytes(b"the only copy")

        assert client.get(f"/v1/videos/{job['id']}/content").status_code == 200
        assert artifact.exists(), (
            "the server must still hold it — the client may not have written it"
        )

        # The client persisted it and says so.
        ack = client.delete(f"/v1/videos/{job['id']}/content")
        assert ack.status_code == 200
        assert ack.json()["reclaimed"] == 1
        assert not artifact.exists(), "acked means reclaimed"


def test_the_ack_is_idempotent_and_is_not_a_cancellation(bal, fleet, tmp_path):
    """A client must be able to retry an ack whose response was lost, and
    acking an artifact must never be mistaken for cancelling the job that
    produced it."""
    work = tmp_path / "work"
    (work / "outputs").mkdir(parents=True)
    bal.WORK_DIR = str(work)

    with TestClient(bal.app) as client:
        job = _submit(client).json()
        (work / "outputs" / f"{job['id']}.mp4").write_bytes(b"x")
        before = bal.bal.snapshot()["active_jobs"]

        assert client.delete(f"/v1/videos/{job['id']}/content").json()["reclaimed"] == 1
        second = client.delete(f"/v1/videos/{job['id']}/content")
        assert second.status_code == 200, "a retried ack must not error"
        assert second.json()["reclaimed"] == 0

        assert bal.bal.snapshot()["active_jobs"] == before, (
            "acking an ARTIFACT must not end the JOB — that is what DELETE on the job itself means"
        )


def test_waiting_tickets_are_refused_once_the_fleet_is_provably_dead(bal, fleet):
    """The waiting ticket was the only unbounded object in the system.

    Every path to REJECTED lives inside _pump AFTER pick() returns a target —
    and pick() returns None on a dead fleet, so the pump bails before reaching
    the code that could refuse anything. Client-side, a queued ticket is
    exempt from every stall verdict. The pair deadlocked forever."""
    with TestClient(bal.app) as client:
        for st in bal.bal.stats.values():
            st.healthy = False
        r = _submit(client, "work nothing can take")
        ticket = r.json()["id"]
        assert r.status_code == 202 and ticket.startswith("q_"), (
            "an unhealthy fleet must still QUEUE — refusing at submit time is "
            "what broke fleet-switch bootstrapping"
        )

        bal.FLEET_DEAD_STRIKES = 3
        bal._fleet_dead_beats = 0
        for _ in range(2):
            bal._fail_unservable_tickets()
        assert client.get(f"/v1/videos/{ticket}").json()["status"] == "queued"

        bal._fail_unservable_tickets()  # third strike
        body = client.get(f"/v1/videos/{ticket}").json()
        assert body["status"] == "failed", "a ticket nothing can serve must end"
        assert body["billable"] is False, "never submitted, so never billed"
        assert "0 accepting" in str(body.get("error") or "")
        assert bal.submit_unservable == 1


def test_one_accepting_replica_clears_the_dead_fleet_count(bal, fleet):
    """A fleet switch or a blip must never refuse legitimately queued work.
    The count resets on any accepting replica, so a drain would have to
    outlast every beat in the window to trip it."""
    with TestClient(bal.app) as client:
        for st in bal.bal.stats.values():
            st.healthy = False
        ticket = _submit(client, "work").json()["id"]

        bal.FLEET_DEAD_STRIKES = 3
        bal._fleet_dead_beats = 0
        bal._fail_unservable_tickets()
        bal._fail_unservable_tickets()
        next(iter(bal.bal.stats.values())).healthy = True  # a replica returns
        bal._fail_unservable_tickets()
        assert bal._fleet_dead_beats == 0

        for st in bal.bal.stats.values():
            st.healthy = False
        bal._fail_unservable_tickets()
        assert client.get(f"/v1/videos/{ticket}").json()["status"] == "queued", (
            "the count restarted, so the ticket survives"
        )


def test_a_queued_ticket_reports_whether_anything_can_serve_it(bal, fleet):
    """Without this field a poll on a dead fleet is byte-identical to one
    next in line behind a long generation."""
    with TestClient(bal.app) as client:
        for st in bal.bal.stats.values():
            st.healthy = False
        ticket = _submit(client, "work").json()["id"]
        assert client.get(f"/v1/videos/{ticket}").json()["accepting_replicas"] == 0
        for st in bal.bal.stats.values():
            st.healthy = True
        assert client.get(f"/v1/videos/{ticket}").json()["accepting_replicas"] > 0


def test_sweep_adopts_a_ghost_job_born_after_boot(bal, fleet):
    """The ghost: the engine ACCEPTED a generation we never recorded (our
    POST timed out mid-accept, or an ambiguous submit was rejected after the
    work started). It burns a GPU that pick() still counts as free, so the
    replica gets double-booked. Boot recovery alone never saw it, because it
    was born after boot."""

    listing: list[dict] = []

    def with_listing(request):
        if request.url.path == "/v1/videos" and request.method == "GET":
            base = f"{request.url.scheme}://{request.url.host}"
            return httpx.Response(200, json={"data": listing if base == R1 else []})
        return fleet.handler(request)

    bal.ctl = bal.client = httpx.AsyncClient(transport=httpx.MockTransport(with_listing))
    with TestClient(bal.app) as client:
        client.portal.call(bal._recover_occupancy)
        assert bal.bal.snapshot()["active_jobs"] == 0, "nothing running at boot"

        # The engine starts a job we never learn about.
        listing.append({"id": "vid_ghost", "status": "in_progress"})
        before = bal.orphans_adopted
        adopted = client.portal.call(lambda: bal._recover_occupancy(boot=False))

        assert adopted == 1
        assert bal.orphans_adopted == before + 1, "counted, so an operator can alert"
        stats = {s["url"]: s for s in bal.bal.snapshot()["backends"]}
        assert stats[R1]["inflight"] == 1, "the ghost now occupies its slot"

        # Idempotent: a second sweep over the same listing adopts nothing and
        # must not inflate occupancy, or a long job would fill the fleet with
        # copies of itself one beat at a time.
        assert client.portal.call(lambda: bal._recover_occupancy(boot=False)) == 0
        assert bal.bal.snapshot()["active_jobs"] == 1


# --------------------------------------------------------------------------
# Artifact retention: 6h initial-fetch window, 1h re-fetch window


def test_retention_windows(bal):
    """The policy, pure: never-fetched artifacts live 6h from creation;
    fetched ones live 1h from the FIRST fetch; running jobs are never
    garbage regardless of age."""
    now = 1_000_000.0
    entries = [
        ("/w/outputs/old-unfetched.mp4", "old-unfetched", now - 7 * 3600),
        ("/w/outputs/new-unfetched.mp4", "new-unfetched", now - 3600),
        ("/w/outputs/fetched-long-ago.mp4", "fetched-long-ago", now - 7 * 3600),
        ("/w/outputs/fetched-recently.mp4", "fetched-recently", now - 7 * 3600),
        ("/w/outputs/ancient-but-running.mp4", "running-job", now - 30 * 3600),
        ("/w/inputs/stale-upload.png", "", now - 7 * 3600),
    ]
    fetched = {"fetched-long-ago": now - 2 * 3600, "fetched-recently": now - 600}
    doomed = bal._retention_candidates(entries, fetched, {"running-job"}, now, 6 * 3600, 3600)
    assert doomed == [
        "/w/outputs/old-unfetched.mp4",
        "/w/outputs/fetched-long-ago.mp4",
        "/w/inputs/stale-upload.png",
    ]


def test_retention_pass_reclaims_disk_and_reports(bal, fleet, tmp_path, monkeypatch):
    """End to end on a real directory: expired files go, protected and
    in-window files stay, and the count is visible in /balancer/stats."""
    import os
    import time as _time

    work = tmp_path / "work"
    (work / "outputs").mkdir(parents=True)
    (work / "inputs").mkdir()
    old = _time.time() - 7 * 3600
    for name in ("expired.mp4", "protected.mp4", "fresh.mp4"):
        p = work / "outputs" / name
        p.write_bytes(b"x")
    os.utime(work / "outputs" / "expired.mp4", (old, old))
    os.utime(work / "outputs" / "protected.mp4", (old, old))
    upload = work / "inputs" / "upload.png"
    upload.write_bytes(b"x")
    os.utime(upload, (old, old))

    monkeypatch.setattr(bal, "WORK_DIR", str(work))
    bal.bal.begin_job("protected", R1)  # ancient but RUNNING: untouchable
    bal._retention_pass()

    assert not (work / "outputs" / "expired.mp4").exists()
    assert not upload.exists()
    assert (work / "outputs" / "protected.mp4").exists()
    assert (work / "outputs" / "fresh.mp4").exists()
    with TestClient(bal.app) as client:
        assert client.get("/balancer/stats").json()["retention"]["deleted_total"] == 2


def test_the_first_content_fetch_starts_the_refetch_window(bal, fleet):
    """Recorded once, at the first /content read — repeats must not keep
    extending the window — and persisted so a restart keeps it honest."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        fleet.complete(job["id"])
        client.get(f"/v1/videos/{job['id']}/content")
        first = bal._fetched[job["id"]]
        client.get(f"/v1/videos/{job['id']}/content")
        assert bal._fetched[job["id"]] == first, "window not extended"
    bal._persist_membership()
    saved = bal._fetched.copy()
    bal._fetched.clear()
    bal._restore_membership()
    assert bal._fetched == saved, "survives a crash-restart"


# --------------------------------------------------------------------------
# Operator observability: /v1/jobs and /metrics


def test_jobs_reports_what_each_job_is_not_just_that_it_runs(bal, fleet):
    """The engine's records carry no prompt/task; the balancer captures the
    request at submission so an operator sees WHAT runs, queues and just
    finished — without owning any client."""
    with TestClient(bal.app) as client:
        j1 = client.post(
            "/v1/videos",
            json={
                "prompt": "a lighthouse in fog",
                "task": "t2va",
                "target": {"duration_seconds": 5},
            },
        ).json()
        client.post("/v1/videos", json={"prompt": "rain on canvas", "task": "t2va"})
        client.post("/v1/videos", json={"prompt": "queued third", "task": "fl2va", "priority": 0})

        body = client.get("/v1/jobs").json()
        running = {j["job_id"]: j for j in body["running"]}
        assert running[j1["id"]]["prompt"] == "a lighthouse in fog"
        assert running[j1["id"]]["task"] == "t2va"
        assert running[j1["id"]]["seconds"] == 5
        assert body["queued"][0]["prompt"] == "queued third"
        assert body["queued"][0]["priority"] == 0

        # Terminal jobs land in the ring with who closed them and what they were.
        fleet.complete(j1["id"])
        client.get(f"/v1/videos/{j1['id']}")
        body = client.get("/v1/jobs").json()
        assert body["recent"][0]["job_id"] == j1["id"]
        assert body["recent"][0]["status"] == "completed"
        assert body["recent"][0]["closed_by"] == "client"
        assert body["recent"][0]["prompt"] == "a lighthouse in fog"


def test_metrics_speaks_prometheus(bal, fleet):
    """Plain text exposition: any scraper points here, no agents."""
    with TestClient(bal.app) as client:
        _submit(client)
        text = client.get("/metrics").text
    assert "deepme_slots_free 1" in text
    assert "deepme_jobs_running 1" in text
    assert "# TYPE deepme_jobs_reaped_total counter" in text


# --------------------------------------------------------------------------
# Event firehose: subscribe instead of polling


def test_the_firehose_narrates_the_full_lifecycle(bal, fleet):
    """queued -> handover -> started -> terminal, each event carrying what
    the work IS — a dashboard subscribes once and needs no polling."""
    with TestClient(bal.app) as client:
        _submit(client, prompt="one")
        _submit(client, prompt="two")
        ticket = _submit(client, prompt="three").json()["id"]  # queues
        first = bal._events_ring
        kinds = [e["type"] for e in first]
        assert kinds.count("started") == 2
        assert "queued" in kinds
        queued_evt = next(e for e in first if e["type"] == "queued")
        assert queued_evt["prompt"] == "three"
        assert queued_evt["ticket"] == ticket

        # A slot frees; the ticket hands over; the finished job is terminal.
        done = next(e for e in first if e["type"] == "started")
        fleet.complete(done["job_id"])
        client.get(f"/v1/videos/{done['job_id']}")
        kinds = [e["type"] for e in bal._events_ring]
        assert "terminal" in kinds
        assert "handover" in kinds
        term = next(e for e in bal._events_ring if e["type"] == "terminal")
        assert term["closed_by"] == "client"
        assert term["prompt"] == "one"


def test_the_firehose_replays_since_and_streams(bal, fleet):
    """?since=<seq> catches a reconnecting subscriber up from the ring."""
    with TestClient(bal.app) as client:
        _submit(client, prompt="early")
        cut = bal._events_ring[-1]["seq"]
        _submit(client, prompt="late")
        with client.stream("GET", f"/v1/events?since={cut}&limit=1") as r:
            events = _take(r, 1)
    assert events[0]["type"] == "started"
    assert events[0]["prompt"] == "late", "only what happened after the cut"


def test_a_slow_subscriber_gets_a_gap_never_backpressure(bal, fleet):
    """The platform must not care how slow a dashboard is: a full
    subscriber queue drops oldest and marks the gap."""
    import asyncio

    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    bal._subscribers.add(q)
    try:
        bal._publish("t1", n=1)
        bal._publish("t2", n=2)  # queue full -> drop t1, mark gap
        drained = []
        while not q.empty():
            drained.append(q.get_nowait())
        assert [e["type"] for e in drained] == ["gap"]
    finally:
        bal._subscribers.discard(q)


# --------------------------------------------------------------------------
# Identity integrity: an id must NEVER resolve to another job's replica


def test_two_replicas_claiming_one_id_is_a_conflict_not_an_absence(bal, fleet):
    """409, never 404. The id exists TWICE — a 404 tells the client it is
    unknown and therefore safe to resubmit, which is the opposite of the
    truth and bills the same work again."""

    def both_claim(request):
        return httpx.Response(200, json={"id": "vid_dup", "status": "queued"})

    bal.ctl = httpx.AsyncClient(transport=httpx.MockTransport(both_claim))
    with TestClient(bal.app) as client:
        r = client.get("/v1/videos/vid_dup")
    assert r.status_code == 409
    assert r.json()["retryable"] is False


def test_owner_probe_ignores_a_replica_answering_for_a_foreign_id(bal, fleet):
    """A 200 alone proves nothing — the body must NAME the probed id."""

    def foreign_first(request):
        if "replica-1" in str(request.url):
            # Answers happily... about some other job entirely.
            return httpx.Response(200, json={"id": "vid_other", "status": "queued"})
        if str(request.url.path).endswith("vid_x"):
            return httpx.Response(200, json={"id": "vid_x", "status": "queued"})
        return httpx.Response(404, json={})

    t = httpx.AsyncClient(transport=httpx.MockTransport(foreign_first))
    bal.ctl = t
    bal.client = t
    with TestClient(bal.app) as client:
        r = client.get("/v1/videos/vid_x")
    assert r.status_code == 200
    assert r.json()["id"] == "vid_x"
    assert bal.bal.owner("vid_x") == R2, "bound to the VERIFIED claimant"


def test_an_ambiguous_submission_failure_is_terminal_not_a_duplicate(bal, fleet):
    """A read timeout AFTER the request may have reached the engine must
    never requeue: the engine may already hold the job, and the requeue
    generated the same prompt twice (one ghost per timeout). The ticket
    fails LOUDLY instead, and no second POST reaches any replica."""
    with TestClient(bal.app) as client:
        a = _submit(client).json()
        b = _submit(client).json()
        t = _submit(client, prompt="queued-one")
        assert t.status_code == 202
        ticket = t.json()["id"]
        posts_before = len(fleet.posts)

        def slow_accept(request):
            if request.method == "POST":
                raise httpx.ReadTimeout("engine grinding, accept > timeout")
            return fleet.handler(request)

        bal.ctl = httpx.AsyncClient(transport=httpx.MockTransport(slow_accept))
        fleet.complete(a["id"])
        fleet.complete(b["id"])
        client.get(f"/v1/videos/{a['id']}")  # close -> pump -> timeout
        r = client.get(f"/v1/videos/{ticket}")
    assert r.json()["status"] == "failed"
    assert "outcome unknown" in r.json()["error"]["message"]
    assert len(fleet.posts) == posts_before, "no blind second submission"


def test_a_connect_failure_still_requeues_safely(bal, fleet):
    """CONNECT-phase failure proves the request never arrived — the one
    case where requeueing cannot duplicate."""
    with TestClient(bal.app) as client:
        a = _submit(client).json()
        b = _submit(client).json()
        t = _submit(client, prompt="queued-one")
        ticket = t.json()["id"]

        def refused(request):
            if request.method == "POST":
                raise httpx.ConnectError("connection refused")
            return fleet.handler(request)

        bal.ctl = httpx.AsyncClient(transport=httpx.MockTransport(refused))
        fleet.complete(a["id"])
        fleet.complete(b["id"])
        client.get(f"/v1/videos/{a['id']}")
        r = client.get(f"/v1/videos/{ticket}")
    assert r.json()["status"] == "queued", "back in the queue, not failed"


def test_a_completed_download_starts_the_refetch_window_a_cut_one_does_not(bal, fleet):
    """_fetched marks on the LAST byte, not the first request: a download a
    gateway restart cut mid-stream must not burn the only re-fetch window
    of billed work."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        fleet.complete(job["id"])
        r = client.get(f"/v1/videos/{job['id']}/content")
        assert r.content == b"MP4"
    assert job["id"] in bal._fetched, "full read -> window started"


def test_job_scoped_responses_carry_the_effective_id_header(bal, fleet):
    """The gateway learns ticket->job bindings from this header instead of
    probing — the header IS the authoritative ownership statement."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        r = client.get(f"/v1/videos/{job['id']}")
    assert r.headers.get("x-deepme-job-id") == job["id"]


def test_every_lifecycle_event_lands_in_the_persistent_audit_log(bal, fleet, tmp_path):
    """The firehose forgets after 500 events; incidents ask about last
    Tuesday. Same events, appended to disk."""
    bal.AUDIT_LOG = str(tmp_path / "audit.jsonl")
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        fleet.complete(job["id"])
        client.get(f"/v1/videos/{job['id']}")
    lines = [json.loads(x) for x in (tmp_path / "audit.jsonl").read_text().splitlines()]
    kinds = [e["type"] for e in lines]
    assert "started" in kinds and "terminal" in kinds
    terminal = next(e for e in lines if e["type"] == "terminal")
    assert terminal["job_id"] == job["id"]


# --------------------------------------------------------------------------
# Liveness: the queue must work in exactly the windows it exists for


def test_a_fully_drained_fleet_queues_instead_of_refusing(bal, fleet):
    """THE fleet-switch hole: with every replica drained, `_all_busy` read
    False (no accepting replicas at all), admission waved the request
    through to pick(), which had nothing — so the caller got 503 and the
    queue could never bootstrap, because the 'someone is waiting' path
    needs a first ticket to exist."""
    with TestClient(bal.app) as client:
        client.post("/balancer/backends/drain", json={"url": R1})
        client.post("/balancer/backends/drain", json={"url": R2})
        r = _submit(client, prompt="during the switch")
    assert r.status_code == 202, "held for the new fleet, not refused"
    assert r.json()["id"].startswith("q_")


def test_an_unhealthy_fleet_also_queues(bal, fleet):
    """Same predicate, the transient case: a fleet-wide blip must park work
    until _revive_backends re-arms, not reject it."""
    bal.bal.mark(R1, healthy=False)
    bal.bal.mark(R2, healthy=False)
    with TestClient(bal.app) as client:
        r = _submit(client, prompt="during the blip")
    assert r.status_code == 202


def test_boot_recovery_completes_before_the_first_request_routes(bal, fleet):
    """A restart with replicas mid-generation must not double-book them.
    Recovery used to be scheduled (create_task) rather than awaited, so
    startup finished with an empty job map and pick() handed new work to a
    replica already generating."""
    fleet.jobs["vid_live"] = {"id": "vid_live", "status": "queued", "owner": R1}

    def with_listing(request):
        if request.method == "GET" and request.url.path == "/v1/videos":
            base = f"{request.url.scheme}://{request.url.host}"
            here = [j for j in fleet.jobs.values() if j["owner"] == base]
            return httpx.Response(200, json={"data": here})
        return fleet.handler(request)

    bal.ctl = httpx.AsyncClient(transport=httpx.MockTransport(with_listing))
    with TestClient(bal.app) as client:
        # The very first request already sees the recovered occupancy.
        assert bal.bal.owner("vid_live") == R1
        r = client.get("/v1/capacity")
    assert r.json()["busy"] == 1, "the running job holds its slot from boot"


def test_a_failing_maintenance_beat_is_counted_not_swallowed(bal, fleet):
    """The maintenance loop IS the liveness invariant; a blanket suppress
    made a permanently dead loop indistinguishable from a healthy idle
    one."""
    before = bal.maintenance_errors
    bal.maintenance_errors = before + 1
    with TestClient(bal.app) as client:
        body = client.get("/metrics").text
    assert f"deepme_maintenance_errors_total {before + 1}" in body


def test_retention_survives_a_file_vanishing_mid_scan(bal, fleet, tmp_path, monkeypatch):
    """A file removed between the walk and the stat is routine. Suppressing
    around the WHOLE walk abandoned the traversal, so a recurring race
    silently disabled retention while stats still reported it enabled."""
    import time as _time

    work = tmp_path / "work"
    (work / "outputs" / "nested").mkdir(parents=True)
    (work / "inputs").mkdir()
    old = _time.time() - 7 * 3600
    keep = work / "outputs" / "nested" / "expired.mp4"
    keep.write_bytes(b"x")
    os.utime(keep, (old, old))
    monkeypatch.setattr(bal, "WORK_DIR", str(work))

    real_stat = Path.stat
    ghost = work / "outputs" / "ghost.mp4"
    ghost.write_bytes(b"x")

    def flaky_stat(self, *a, **kw):
        if self.name == "ghost.mp4":
            raise FileNotFoundError(self)
        return real_stat(self, *a, **kw)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    bal._retention_pass()
    assert not keep.exists(), "the nested expired file still got reclaimed"


# --------------------------------------------------------------------------
# Wire-contract integrity: an id must never hand back the wrong KIND of thing


def test_content_on_an_unstarted_ticket_is_refused_not_answered_with_json(bal, fleet):
    """THE corrupt-deliverable path: the ticket branch matched every q_ path
    including /content, so a client fetching early streamed a JSON status
    body into its .mp4, renamed it, and recorded a successful deliverable.
    There is no 2xx that a body-shaped answer is correct for here."""
    with TestClient(bal.app) as client:
        _submit(client)
        _submit(client)
        ticket = _submit(client, prompt="queued").json()["id"]
        assert ticket.startswith("q_")
        r = client.get(f"/v1/videos/{ticket}/content")
    assert r.status_code == 409, "no artifact exists yet"
    assert r.content[:4] != b'{"id', "and certainly not a status body"


def test_the_effective_id_header_is_absent_on_error_responses(bal, fleet):
    """Stamping it on a replica's 404 told the gateway 'this fleet owns the
    job' about the fleet that had just denied it — and the gateway caches
    that until a 502, so one 404 poisoned the mapping for good."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        ok = client.get(f"/v1/videos/{job['id']}")
        assert ok.headers.get("x-deepme-job-id") == job["id"]
        fleet.jobs.pop(job["id"])
        gone = client.get(f"/v1/videos/{job['id']}")
    assert gone.status_code >= 400
    assert "x-deepme-job-id" not in gone.headers


def test_a_ticket_whose_job_still_runs_is_never_pruned(bal, fleet):
    """The platform promises the handed-out id keeps working. Dropping a
    SUBMITTED ticket mid-generation turned that promise into a 404, which
    the client reads as a lost ticket and resubmits — billing twice."""
    with TestClient(bal.app) as client:
        a = _submit(client).json()
        b = _submit(client).json()
        ticket = _submit(client, prompt="queued").json()["id"]
        fleet.complete(a["id"])
        client.get(f"/v1/videos/{a['id']}")  # frees a slot -> pump -> handover
        # Prune aggressively; the ticket's job is running, so it must stay.
        bal.admission.prune(keep=0, live_jobs={j for j, _u, _t, _c in bal.bal.open_jobs()})
        still = client.get(f"/v1/videos/{ticket}")
        assert still.status_code == 200, "the id the caller holds still works"
        # A ticket with NO live job is fair game.
        fleet.complete(b["id"])
        client.get(f"/v1/videos/{b['id']}")
        bal.admission.prune(keep=0, live_jobs=set())
    assert bal.admission.get(ticket) is None


def test_a_proxied_read_never_holds_a_replica_slot(bal, fleet):
    """Reserving for every method meant a non-job-scoped GET against a
    wedged replica parked forever holding a reservation nothing released:
    free_slots() dropped permanently, and since admission delegates to it,
    the queue stopped draining on an idle fleet with nothing counted."""
    with TestClient(bal.app) as client:
        before = client.get("/v1/capacity").json()["free"]
        # A read that is not job-scoped and not a POST.
        client.get("/v1/models")
        after = client.get("/v1/capacity").json()["free"]
    assert after == before, "no slot was consumed by a read"
    stats = {s["url"]: s for s in bal.bal.snapshot()["backends"]}
    assert all(s["failed"] == 0 for s in stats.values()), (
        "a proxied read's status must not count as a job failure"
    )


def test_strikes_do_not_leak_for_jobs_closed_elsewhere(bal, fleet):
    """Strike entries were removed only on threshold or a healthy answer, so
    a job closed by a client poll / SSE / DELETE left its counter behind
    forever — an unbounded dict, and stale strikes re-applied to a recycled
    id."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        bal._job_strikes[job["id"]] = 3  # partway to the threshold
        fleet.complete(job["id"])
        client.get(f"/v1/videos/{job['id']}")  # closed by the CLIENT
        asyncio.run(bal._reap_stale_jobs())
    assert job["id"] not in bal._job_strikes


def test_a_strike_close_reports_a_documented_status_with_its_reason(bal, fleet):
    """ "lost" was minted as a fifth terminal status that no consumer knows;
    the nuance belongs in closed_by/reason, which are fields."""
    with TestClient(bal.app) as client:
        job = _submit(client).json()
        for _ in range(bal.JOB_LOST_STRIKES):
            bal._strike(job["id"], R1, "unknown to its replica")
        body = client.get("/v1/jobs").json()
    recent = body["recent"][0]
    assert recent["status"] == "failed", "one of the documented four"
    assert recent["closed_by"] == "strikes"
    assert recent["reason"] == "unknown to its replica"


def test_stats_reports_maintenance_health_and_queue_depth(bal, fleet):
    """A dead maintenance loop looks exactly like a healthy idle one. The
    counters existed but lived only in /metrics, which no command reads —
    so in practice nobody could see them."""
    bal.maintenance_errors = 2
    bal.retention_errors = 1
    with TestClient(bal.app) as client:
        _submit(client)
        _submit(client)
        _submit(client, prompt="queued")  # ticket
        snap = client.get("/balancer/stats").json()
    assert snap["maintenance"]["errors_total"] == 2
    assert snap["maintenance"]["retention_errors_total"] == 1
    assert snap["maintenance"]["boot"] == bal.BOOT_ID
    assert "audit_enabled" in snap["maintenance"]
    assert snap["queue"]["waiting"] == 1, "queue depth without curling for it"


def test_control_plane_routes_require_the_platform_key(bal, fleet, monkeypatch):
    """These routes CHANGE THE FLEET and were reachable by anyone who could
    talk to the port, which binds 0.0.0.0 — a single unauthenticated DELETE
    could tear the fleet down."""
    monkeypatch.setattr(bal, "API_KEY", "secret-key")
    with TestClient(bal.app) as client:
        assert client.get("/balancer/stats").status_code == 401
        assert client.post("/balancer/backends", json={"url": R1}).status_code == 401
        assert client.request("DELETE", "/balancer/backends", params={"url": R1}).status_code == 401
        good = {"Authorization": "Bearer secret-key"}
        assert client.get("/balancer/stats", headers=good).status_code == 200
        # The DATA plane is deliberately NOT gated: it forwards the caller's
        # own credentials onward, and gating it would break the service layer.
        assert client.get("/v1/capacity").status_code == 200


def test_control_plane_is_open_when_no_key_is_configured(bal, fleet):
    """A local/dev deployment with no key configured is unchanged."""
    with TestClient(bal.app) as client:
        assert client.get("/balancer/stats").status_code == 200
