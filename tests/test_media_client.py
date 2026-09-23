"""The generation client against a fake server, over real HTTP plumbing.

The existing media tests stub `_submit`, so they check what goes INTO a
request but never what the client does with the responses. That left the
customer-facing half of the contract untested — in particular the `202`, where
the server hands back a ticket instead of a job and the client is expected to
carry on unchanged.

Here the transport is faked instead of the function, so `_submit`, `_poll` and
`_download` all really run.
"""

from pathlib import Path
from typing import Any, ClassVar

import httpx
import pytest

from deepme import media


def _r(method: str, url: str, status: int, **kw) -> httpx.Response:
    """A response with its request attached, as a real transport returns.

    Without it `raise_for_status` cannot run, so a fake built the naive way
    tests a code path the client never takes.
    """
    return httpx.Response(status, request=httpx.Request(method, url), **kw)


class _Spec:
    profile = "h3"
    served_name = "h3-video"
    api_base = "http://h"
    unit = "deepme-llm-h3"
    min_steps = 2
    gen_defaults: ClassVar[dict[str, Any]] = {"task": "t2va", "conditions": []}


class FakeServer:
    """A balancer that queues the request, then hands it to a replica.

    Models the real sequence: a 202 with a `q_` id, that id answering as
    "queued" for a while, then the same id transparently answering as the
    real job (the balancer rewrites it) and finally serving the content.
    """

    def __init__(self, *, queue_polls: int = 2, accepted: bool = True):
        self.queue_polls = queue_polls
        self.accepted = accepted
        self.polls = 0
        self.paths: list[str] = []

    def handle(self, method: str, url: str) -> httpx.Response:
        path = httpx.URL(url).path
        self.paths.append(path)
        if method == "POST":
            code = 200 if self.accepted else 202
            body = (
                {"id": "vid_9", "status": "queued"}
                if self.accepted
                else {"id": "q_1", "status": "queued", "billable": False}
            )
            return _r(method, url, code, json=body)
        if path.endswith("/content"):
            return _r(method, url, 200, content=b"MP4DATA")
        self.polls += 1
        if self.polls <= self.queue_polls:
            return _r(method, url, 200, json={"id": "q_1", "status": "queued"})
        return _r(
            method,
            url,
            200,
            json={
                "id": "vid_9",
                "status": "completed",
                "inference_time_s": 812.0,
                "peak_memory_mb": 26700.0,
            },
        )


@pytest.fixture
def server(monkeypatch, tmp_path):
    fake = FakeServer()
    monkeypatch.setattr(media, "DEFAULT_OUT_DIR", tmp_path)
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(httpx, "post", lambda url, **kw: fake.handle("POST", url))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: fake.handle("GET", url))

    class _Client:
        def __init__(self, **kw): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            resp = fake.handle(method, url)

            class _Ctx:
                def __enter__(self_inner):
                    return resp

                def __exit__(self_inner, *a):
                    return False

            return _Ctx()

    monkeypatch.setattr(httpx, "Client", _Client)
    return fake


def test_a_queued_ticket_is_followed_to_a_finished_file(server, tmp_path):
    """A 202 must not need a different client path.

    The server hands back `q_1` instead of a job id; the client polls that id
    until it resolves to a finished job and downloads the result. If the
    client had to special-case tickets, every consumer would have to too.
    """
    server.accepted = False
    result = media.generate_video(_Spec(), "a lighthouse", out_dir=tmp_path)

    assert result.status == "completed"
    # Delivered under the server's DURABLE id, not the mortal ticket handle:
    # the poll bodies named vid_9 and the client harvests it (a ticket dies
    # with a balancer restart; the job id survives).
    assert result.media_id == "vid_9"
    assert result.path and result.path.read_bytes() == b"MP4DATA"
    assert server.polls > 1, "it really waited in the queue"
    assert result.inference_s == 812.0


def test_the_download_uses_the_harvested_job_id(server, tmp_path):
    """Once the poll bodies name the real job, everything — including the
    content fetch — keys on it, so the flow survives the ticket dying."""
    server.accepted = False
    media.generate_video(_Spec(), "a lighthouse", out_dir=tmp_path)
    assert "/v1/videos/vid_9/content" in server.paths


def test_a_job_accepted_immediately_skips_the_queue(server, tmp_path):
    result = media.generate_video(_Spec(), "a lighthouse", out_dir=tmp_path)
    assert result.media_id == "vid_9"
    assert result.status == "completed"


def test_a_rejected_submission_fails_loudly(monkeypatch, tmp_path):
    """A 4xx is a client error and must not be polled as if it were work."""
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: _r("POST", url, 400, json={"detail": "bad task"}),
    )
    with pytest.raises(SystemExit, match="bad task"):
        media.generate_video(_Spec(), "x", out_dir=tmp_path)


def test_progress_is_reported_while_waiting(server, tmp_path):
    """A caller must be able to show something during a long generation."""
    server.accepted = False
    ticks = []
    media.generate_video(_Spec(), "x", out_dir=tmp_path, on_tick=lambda t: ticks.append(t))
    assert ticks, "the client emitted progress"
    assert all(t.elapsed_s >= 0 for t in ticks)
    assert ticks[0].status == "queued"


def test_a_transient_transport_error_does_not_abandon_the_job(monkeypatch, server, tmp_path):
    """A long generation can outlive a proxy's idle timeout.

    Giving up on the first failed poll would strand a job that is still
    running and still billable.
    """
    calls = {"n": 0}
    real = server.handle

    def flaky(method, url):
        calls["n"] += 1
        if calls["n"] == 2:
            raise httpx.ReadTimeout("idle")
        return real(method, url)

    monkeypatch.setattr(httpx, "get", lambda url, **kw: flaky("GET", url))
    result = media.generate_video(_Spec(), "x", out_dir=tmp_path)
    assert result.status == "completed"


def test_an_unfinished_job_reports_where_to_look(monkeypatch, tmp_path):
    """A timeout must not imply the work stopped — it is still on a GPU."""
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: _r("POST", url, 200, json={"id": "vid_1", "status": "queued"}),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _r("GET", url, 200, json={"status": "queued"}),
    )
    with pytest.raises(SystemExit, match="may still"):
        media.generate_video(_Spec(), "x", out_dir=Path(tmp_path), timeout=0.01)


# --------------------------------------------------------------------------
# Progress attribution: whose journal holds this job's bar


def _cap(monkeypatch, payload):
    def fake_get(url, **kw):
        if url.endswith("/v1/capacity"):
            return _r("GET", url, 200, json=payload)
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(httpx, "get", fake_get)


@pytest.fixture(autouse=True)
def _fresh_owner_cache():
    media._OWNER_UNITS.clear()


@pytest.fixture
def registry(monkeypatch):
    from deepme import ports as ports_mod

    class _Reg:
        replicas: ClassVar[dict] = {"h3": [8100, 8110, 8120, 8130]}

    monkeypatch.setattr(ports_mod, "load", lambda: _Reg())


def test_progress_reads_the_journal_of_the_replica_running_the_job(monkeypatch, registry):
    """The regression that motivated this: behind a balancer the job runs on
    whichever replica was free, and each replica logs only its own bar.
    Reading replica 0's journal showed NOTHING when it was idle — and someone
    else's progress when it was busy."""
    _cap(
        monkeypatch,
        {
            "fleets": [
                {
                    "profile": "h3",
                    "running": [{"job_id": "vid_7", "backend": "http://127.0.0.1:8120"}],
                },
            ]
        },
    )
    seen = {}
    from deepme import serve

    monkeypatch.setattr(
        serve,
        "stage_progress",
        lambda spec, s, unit=None: seen.setdefault("unit", unit),
    )
    media.probe_stage(_Spec(), 60.0, "vid_7")
    assert seen["unit"] == "deepme-llm-h3-r2", "replica index 2, not replica 0"


def test_a_job_not_running_anywhere_shows_no_bar_rather_than_a_wrong_one(monkeypatch, registry):
    """Queued: no replica is denoising this job, so there is nothing to show.
    Falling back to replica 0 here is how the wrong-bar bug happened."""
    _cap(monkeypatch, {"fleets": [{"profile": "h3", "running": []}]})
    from deepme import serve

    monkeypatch.setattr(
        serve,
        "stage_progress",
        lambda *a, **k: pytest.fail("no journal should be read"),
    )
    assert media.probe_stage(_Spec(), 60.0, "vid_9") is None


def test_direct_runs_fall_back_to_the_replicas_own_journal(monkeypatch, registry):
    """--direct targets the replica itself, which has no capacity endpoint —
    and whose own journal genuinely is the job's journal."""

    def no_capacity(url, **kw):
        raise httpx.ConnectError("no such endpoint")

    monkeypatch.setattr(httpx, "get", no_capacity)
    seen = {}
    from deepme import serve

    monkeypatch.setattr(
        serve,
        "stage_progress",
        lambda spec, s, unit=None: seen.setdefault("unit", unit),
    )
    media.probe_stage(_Spec(), 60.0, "vid_1")
    assert seen["unit"] == "deepme-llm-h3"


def test_the_owner_is_resolved_once_then_cached(monkeypatch, registry):
    calls = {"n": 0}

    def counting_get(url, **kw):
        calls["n"] += 1
        return _r(
            "GET",
            url,
            200,
            json={
                "fleets": [
                    {
                        "profile": "h3",
                        "running": [{"job_id": "vid_5", "backend": "http://127.0.0.1:8110"}],
                    },
                ]
            },
        )

    monkeypatch.setattr(httpx, "get", counting_get)
    from deepme import serve

    monkeypatch.setattr(serve, "stage_progress", lambda *a, **k: None)
    for _ in range(5):
        media.probe_stage(_Spec(), 60.0, "vid_5")
    assert calls["n"] == 1, "an owner never changes; one lookup per job"


# --------------------------------------------------------------------------
# Ticket loss: the balancer-restart scenario that stalled overnight batches


class RestartingServer:
    """Queues the request, then 'restarts': the ticket id 404s forever.

    A resubmit reaches the reborn server, which accepts directly.
    """

    def __init__(self):
        self.posts = 0
        self.paths: list[str] = []

    def handle(self, method: str, url: str) -> httpx.Response:
        path = httpx.URL(url).path
        self.paths.append(path)
        if method == "POST":
            self.posts += 1
            if self.posts == 1:
                return _r(
                    method, url, 202, json={"id": "q_dead-1", "status": "queued", "billable": False}
                )
            return _r(method, url, 200, json={"id": "vid_2", "status": "queued"})
        if "q_dead-1" in path:
            return _r(method, url, 404, json={"detail": "no such ticket"})
        if path.endswith("/content"):
            return _r(method, url, 200, content=b"MP4DATA")
        return _r(method, url, 200, json={"id": "vid_2", "status": "completed"})


def test_a_lost_ticket_is_resubmitted_not_polled_until_timeout(monkeypatch, tmp_path):
    """The 2000-item stall: a balancer restart destroyed queued tickets, and
    their pollers swallowed the 404s as transient — items 'running' for
    hours against an idle fleet. A verifiably dead ticket must be
    resubmitted (nothing was billed) instead of waited out."""
    fake = RestartingServer()
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(httpx, "post", lambda url, **kw: fake.handle("POST", url))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: fake.handle("GET", url))

    class _Client:
        def __init__(self, **kw): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            resp = fake.handle(method, url)

            class _Ctx:
                def __enter__(self_inner):
                    return resp

                def __exit__(self_inner, *a):
                    return False

            return _Ctx()

    monkeypatch.setattr(httpx, "Client", _Client)

    events = []
    result = media.generate_video(_Spec(), "x", out_dir=tmp_path, on_event=events.append)
    assert result.status == "completed"
    assert fake.posts == 2, "exactly one resubmission"
    assert any("resubmitting" in e for e in events), "and it said so"


def test_resubmission_is_bounded_when_the_server_keeps_forgetting(monkeypatch, tmp_path):
    """A persistently-404ing id IS resubmitted — an unreachable result is
    worth nothing and completion is the point — but boundedly: a platform
    that forgets every id it mints must produce a loud failure naming where
    to look, not an infinite regenerate loop."""
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: _r("POST", url, 200, json={"id": "vid_1", "status": "queued"}),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _r("GET", url, 404, json={"detail": "gone"}),
    )
    with pytest.raises(SystemExit, match="keeps forgetting"):
        media.generate_video(_Spec(), "x", out_dir=tmp_path)


# --------------------------------------------------------------------------
# Reconnection: adopt instead of duplicate


def test_adopting_an_existing_job_never_resubmits(server, tmp_path):
    """The job outlives the client. A reconnecting caller that already holds
    an id must re-attach and download — resubmitting duplicates a
    generation that was already paid for."""
    server.queue_polls = 0
    posts_before = sum(1 for p in server.paths if p == "/v1/videos")
    result = media.generate_video(_Spec(), "x", out_dir=tmp_path, adopt="vid_9")
    posts_after = sum(1 for p in server.paths if p == "/v1/videos")
    assert result.status == "completed"
    assert posts_after == posts_before, "no POST was made"
    assert result.path and result.path.read_bytes() == b"MP4DATA"


def test_adopting_a_dead_id_falls_back_to_a_fresh_submission(monkeypatch, tmp_path):
    """If the server verifiably no longer knows the adopted id (replica or
    balancer restarted), nothing complete was lost — resubmit, and report
    the new id so the stored one is replaced."""
    fake = RestartingServer()
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(httpx, "post", lambda url, **kw: fake.handle("POST", url))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: fake.handle("GET", url))

    class _Client:
        def __init__(self, **kw): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            resp = fake.handle(method, url)

            class _Ctx:
                def __enter__(self_inner):
                    return resp

                def __exit__(self_inner, *a):
                    return False

            return _Ctx()

    monkeypatch.setattr(httpx, "Client", _Client)
    fake.posts = 1  # the dead id "q_dead-1" was minted by a past life

    ids = []
    result = media.generate_video(
        _Spec(),
        "x",
        out_dir=tmp_path,
        adopt="q_dead-1",
        on_submit=ids.append,
    )
    assert result.status == "completed"
    assert ids == ["vid_2"], "the replacement id was reported for persistence"


def test_cancel_event_interrupts_within_one_poll_cycle(monkeypatch, tmp_path):
    """Ctrl-C must not wait out a generation: the poll loop notices the
    event and raises InterruptedError, leaving the job running server-side
    for a later adopt."""
    import threading

    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: _r("POST", url, 200, json={"id": "vid_1", "status": "queued"}),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _r("GET", url, 200, json={"status": "queued"}),
    )
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(InterruptedError):
        media.generate_video(_Spec(), "x", out_dir=tmp_path, cancel=cancel)


# --------------------------------------------------------------------------
# One identity: the q_ -> vid_ split (audit F1/F4)


class HandoverServer:
    """202 ticket; polls of the ticket return bodies carrying the REAL id
    (the balancer's rewrite); then the balancer 'restarts' and the ticket
    404s while the job lives on."""

    def __init__(self):
        self.posts = 0
        self.ticket_dead = False
        self.job_done = False

    def handle(self, method: str, url: str) -> httpx.Response:
        path = httpx.URL(url).path
        if method == "POST":
            self.posts += 1
            return _r(method, url, 202, json={"id": "q_t1", "status": "queued", "billable": False})
        if path.endswith("/content"):
            return _r(method, url, 200, content=b"MP4DATA")
        if "q_t1" in path:
            if self.ticket_dead:
                return _r(method, url, 404, json={"detail": "no such ticket"})
            # The rewrite, FAITHFULLY: the balancer answers a ticket poll
            # with the JOB's body — id AND status. A fake that pinned the
            # ticket branch to "queued" forever hung the suite: the client
            # legitimately polls the ticket id it holds, and the job's
            # completion must be visible through it.
            status = "completed" if self.job_done else "queued"
            return _r(
                method,
                url,
                200,
                json={"id": "vid_real", "status": status, "created_at": 1786000000},
            )
        if "vid_real" in path:
            status = "completed" if self.job_done else "queued"
            return _r(
                method,
                url,
                200,
                json={"id": "vid_real", "status": status, "created_at": 1786000000},
            )
        return _r(method, url, 404, json={})


def _wire(monkeypatch, fake):
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    # NEVER the real probe: its --direct fallback reads journals over ssh.
    monkeypatch.setattr(media, "probe_stage", lambda *a, **k: None)
    monkeypatch.setattr(httpx, "post", lambda url, **kw: fake.handle("POST", url))
    monkeypatch.setattr(httpx, "get", lambda url, **kw: fake.handle("GET", url))

    class _Client:
        def __init__(self, **kw): ...
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, method, url, **kw):
            resp = fake.handle(method, url)

            class _Ctx:
                def __enter__(self_inner):
                    return resp

                def __exit__(self_inner, *a):
                    return False

            return _Ctx()

    monkeypatch.setattr(httpx, "Client", _Client)


def test_a_dead_ticket_with_a_known_job_is_followed_not_resubmitted(monkeypatch, tmp_path):
    """The double-billing defect: the balancer restarts, the ticket 404s,
    but its poll bodies had already named the real job — which is STILL
    RUNNING. The client must switch to the job, not regenerate it."""
    fake = HandoverServer()
    _wire(monkeypatch, fake)

    polls = {"n": 0}
    orig = fake.handle

    def scripted(method, url):
        polls["n"] += 1
        if polls["n"] == 3:
            fake.ticket_dead = True  # balancer restart
        if polls["n"] == 12:
            fake.job_done = True
        return orig(method, url)

    fake.handle = scripted
    ids = []
    result = media.generate_video(_Spec(), "x", out_dir=tmp_path, on_submit=ids.append)
    assert result.status == "completed"
    assert fake.posts == 1, "NEVER resubmitted — the job was alive"
    assert result.media_id == "vid_real", "delivered under the server's id"
    assert "vid_real" in ids, "the real id was persisted for adoption"


def test_progress_uses_the_servers_clock_after_reattach(monkeypatch, tmp_path):
    """'Counting up from zero' on every reconnect: elapsed must be the
    JOB's age (server created_at), not this process's polling time."""
    fake = HandoverServer()
    _wire(monkeypatch, fake)
    monkeypatch.setattr(media, "probe_stage", lambda *a, **k: None)
    monkeypatch.setattr(media.time, "time", lambda: 1786000500.0)  # job is 500s old

    ticks = []
    polls = {"n": 0}
    orig = fake.handle

    def scripted(method, url):
        polls["n"] += 1
        if polls["n"] >= 6:
            fake.job_done = True
        return orig(method, url)

    fake.handle = scripted
    media.generate_video(
        _Spec(),
        "x",
        out_dir=tmp_path,
        adopt="q_t1",
        on_tick=lambda t: ticks.append(t),
    )
    assert ticks, "progress was reported"
    assert ticks[-1].elapsed_s >= 490, "the job's true age, not seconds-since-attach"
    assert ticks[-1].job_id == "vid_real", "attributed to the real job"
