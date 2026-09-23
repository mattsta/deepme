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

import importlib.util
import json
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
