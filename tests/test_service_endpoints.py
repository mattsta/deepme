"""Endpoint tests for the two service-layer HTTP surfaces.

Both servers are written to be deployed flat next to their helper modules
(`from route import ...`, `from admission import ...`), so they are loaded
here the same way they run on the box, with their environment supplied up
front. No network, no GPU, no engine.

What is covered is the behaviour a client depends on and that the routing
logic alone cannot show:

  * GET /v1/models answers without routing parameters — discovery has to work
    when the caller knows nothing yet.
  * A ticket id can be followed on the event stream from before it is a job,
    through the handover, without the caller resubscribing.
"""

import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parents[1] / "src" / "deepme"


def _load(name: str, directory: Path, env: dict) -> object:
    """Import a server module the way it runs deployed (flat imports)."""
    import os

    os.environ.update(env)
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(name, directory / "server.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(directory))


# --------------------------------------------------------------------------
# Gateway discovery


@pytest.fixture(scope="module")
def gateway():
    routes = [
        {
            "model": "minimax-h3",
            "profile": "h3-video",
            "tasks": ["t2va", "fl2va"],
            "url": "http://fake-balancer-video",
        },
        {
            "model": "minimax-h3",
            "profile": "h3-ref",
            "tasks": ["ref2va"],
            "url": "http://fake-balancer-ref",
        },
        {
            "model": "deepseek-v4-flash",
            "profile": "deepseek-v4-flash",
            "tasks": [],
            "url": "http://fake-balancer-dsv4",
        },
    ]
    mod = _load(
        "gw_server", SRC / "gateway", {"GATEWAY_ROUTES": json.dumps(routes), "GATEWAY_API_KEY": ""}
    )

    def refuse(request):
        raise httpx.ConnectError("tests are offline")

    mod.client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    return mod


def test_models_lists_without_routing_params(gateway):
    """The standard discovery call must not require a model or task.

    The catch-all proxy resolves a fleet from the request, so before this
    endpoint existed a plain GET /v1/models was answered with a 400 naming
    the empty model — discovery failing precisely when the client has
    nothing to go on.
    """
    r = TestClient(gateway.app).get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == ["deepseek-v4-flash", "minimax-h3"]


def test_models_merges_fleets_and_reports_tasks(gateway):
    """One model name, two fleets behind it, one entry — with the tasks.

    A caller picks a fleet by task, so the task list is the part of this
    response that is actually load-bearing: h3 is served by two separate
    checkpoints and nothing else in the response distinguishes them.
    """
    data = TestClient(gateway.app).get("/v1/models").json()["data"]
    h3 = next(m for m in data if m["id"] == "minimax-h3")
    assert h3["tasks"] == ["fl2va", "ref2va", "t2va"]
    assert h3["owned_by"] == "deepme"


# --------------------------------------------------------------------------
# Balancer ticket stream


@pytest.fixture
def balancer():
    mod = _load(
        "bal_server",
        SRC / "balancer",
        # A FAKE hostname, never 127.0.0.1: real loopback URLs let
        # the balancer's boot-recovery and pump open genuine local
        # sockets from inside pytest — and through any local
        # port-forward, reach production.
        {"BALANCER_BACKENDS": json.dumps(["http://fake-replica-1"]), "BALANCER_API_KEY": ""},
    )

    def refuse(request):
        raise httpx.ConnectError("tests are offline")

    # Both clients faked: these tests exercise queue/ticket state, so every
    # backend call is refused fast rather than dialing anything real.
    mod.client = httpx.AsyncClient(transport=httpx.MockTransport(refuse))
    mod.ctl = mod.client
    return mod


def _events(raw: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :]) for line in raw.splitlines() if line.startswith("data: ")
    ]


def _take(response, count: int) -> list[dict]:
    """Read the first `count` events, then stop reading.

    A stream that has handed over to a running job does not end until that
    job does, so anything that drains it to completion hangs. Only the
    handover itself is under test here.
    """
    out: list[dict] = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            out.append(json.loads(line[len("data: ") :]))
            if len(out) >= count:
                break
    return out


def test_ticket_stream_reports_queue_position_then_handover(balancer, monkeypatch):
    """One subscription spans queued -> submitted.

    The client subscribes with the ticket id it got back from a 202, before
    any replica has seen the work. When a replica takes it the stream must
    switch to the job id itself; otherwise the caller has to poll for the
    handover and resubscribe, which is the exact gap the ticket was meant to
    close.
    """
    balancer.POLL_S = 0.0
    ticket = balancer.admission.enqueue({"prompt": "x"}, priority=balancer.PRIORITY_NORMAL)

    async def fake_pump():
        # Stand in for a replica coming free on the second poll.
        balancer.admission.mark_submitted(ticket.ticket_id, "job-77", "http://127.0.0.1:8100")

    class _Done:
        status_code = 200

        @staticmethod
        def json():
            return {"status": "completed"}

    async def fake_get(url, **kw):
        # The replica reports the job finished, so the stream terminates
        # instead of following a job that never ends.
        return _Done()

    monkeypatch.setattr(balancer, "_pump", fake_pump)
    monkeypatch.setattr(balancer.client, "get", fake_get)
    balancer.bal.remember("job-77", "http://127.0.0.1:8100")

    with TestClient(balancer.app).stream("GET", f"/v1/videos/{ticket.ticket_id}/events") as r:
        events = _take(r, 3)

    assert events[0]["status"] == "queued"
    assert events[0]["job_id"] == ticket.ticket_id
    assert events[0]["billable"] is False
    assert events[1]["status"] == "submitted"
    assert events[1]["job_id"] == "job-77"  # switched to the real job
    assert events[1]["ticket"] == ticket.ticket_id
    assert events[1]["billable"] is True
    # ...and it keeps following that job through to its terminal state, on
    # the same subscription the caller opened with a ticket id.
    assert events[2]["status"] == "completed"
    assert events[2]["job_id"] == "job-77"


def test_ticket_stream_ends_on_cancel(balancer):
    """A cancelled ticket terminates the stream and says it was never billed.

    Cancellation is only offered before submission, so this is the one
    terminal state where the caller owes nothing — and the event has to say
    so, since the same stream otherwise ends in a completed job.
    """
    balancer.POLL_S = 0.0
    ticket = balancer.admission.enqueue({"prompt": "y"}, priority=balancer.PRIORITY_NORMAL)
    balancer.admission.cancel(ticket.ticket_id)

    with TestClient(balancer.app).stream("GET", f"/v1/videos/{ticket.ticket_id}/events") as r:
        events = _events("".join(r.iter_text()))

    assert events[-1]["status"] == "cancelled"
    assert events[-1]["billable"] is False


def test_ticket_is_pollable_like_a_job(balancer):
    """A 202 hands back an id and invites polling, so polling must work.

    The queued response is deliberately job-shaped, but a ticket id belongs to
    no replica — before this, following the documented flow (submit, poll the
    id you got) walked straight into a 404 from the owner lookup.
    """
    # The envelope _admit builds: the pump replays the whole request later.
    ticket = balancer.admission.enqueue(
        {"path": "/v1/videos", "body": {"prompt": "z"}, "headers": {}},
        priority=balancer.PRIORITY_NORMAL,
    )
    r = TestClient(balancer.app).get(f"/v1/videos/{ticket.ticket_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "queued"
    assert r.json()["billable"] is False


def test_ticket_delete_cancels_before_submission(balancer):
    """DELETE on a waiting ticket cancels it rather than 404ing.

    Cancellation is only meaningful on this side of the billing line, and the
    ticket id is the only handle the caller has at that point.
    """
    # The envelope _admit builds: the pump replays the whole request later.
    ticket = balancer.admission.enqueue(
        {"path": "/v1/videos", "body": {"prompt": "z"}, "headers": {}},
        priority=balancer.PRIORITY_NORMAL,
    )
    r = TestClient(balancer.app).delete(f"/v1/videos/{ticket.ticket_id}")
    assert r.status_code == 200
    assert r.json()["status"] == "cancelled"
    assert balancer.admission.get(ticket.ticket_id).state == "cancelled"


def test_unknown_ticket_is_reported_not_hung(balancer):
    """An id that never existed ends the stream instead of polling forever."""
    balancer.POLL_S = 0.0
    with TestClient(balancer.app).stream("GET", "/v1/videos/q_nope/events") as r:
        events = _events("".join(r.iter_text()))
    assert events[-1]["status"] == "unknown"
