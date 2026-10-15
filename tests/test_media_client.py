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
            if method == "DELETE":  # the artifact ack
                return _r(method, url, 200, json={"id": "vid_9", "reclaimed": 1})
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
    monkeypatch.setattr(httpx, "delete", lambda url, **kw: fake.handle("DELETE", url))
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


def test_the_client_acks_only_after_the_file_is_durably_in_place(server, tmp_path):
    """The ack is a promise: "this survives a power loss, you may reclaim
    your copy". Sending it before the rename would let the server delete the
    only good copy of a file we might still fail to place."""
    seen: list[tuple[str, bool]] = []
    real_delete = httpx.delete

    def recording_delete(url, **kw):
        # Was the final file already there when the ack went out?
        seen.append((str(url), any(tmp_path.glob("*.mp4"))))
        return real_delete(url, **kw)

    import unittest.mock

    with unittest.mock.patch.object(httpx, "delete", recording_delete):
        result = media.generate_video(_Spec(), "a lighthouse", out_dir=tmp_path)

    assert len(seen) == 1, "exactly one ack per delivered artifact"
    url, file_existed = seen[0]
    assert url.endswith("/v1/videos/vid_9/content")
    assert file_existed, "acked before the bytes were in place"
    assert result.path and result.path.exists()


def test_a_failed_ack_does_not_fail_the_download(server, tmp_path, monkeypatch):
    """We already have the file. Cleanup is the server's housekeeping, and
    losing the ack costs one artifact one retention window — it must never
    turn a successful generation into a failure."""

    def dead_ack(url, **kw):
        raise httpx.ConnectError("ack went nowhere")

    monkeypatch.setattr(httpx, "delete", dead_ack)
    result = media.generate_video(_Spec(), "a lighthouse", out_dir=tmp_path)
    assert result.status == "completed"
    assert result.path and result.path.read_bytes() == b"MP4DATA"


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
    with pytest.raises(media.GenerationRejected, match="bad task"):
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
    with pytest.raises(media.GenerationTimeout, match="may still"):
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
    # Both caches, or a future test sharing a unit name inherits a stale
    # 20-second stage entry and fails by ordering.
    media._STAGE_CACHE.clear()


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
        # **kw, so the double keeps matching the real signature: probe_stage
        # swallows every exception, so a fake that is merely out of date fails
        # as "no progress available" rather than as a broken test.
        lambda spec, s, unit=None, **kw: seen.setdefault("unit", unit),
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


def test_a_failing_stage_probe_backs_off_instead_of_retrying_every_tick(monkeypatch, registry):
    """The throttle used to protect the healthy case and abandon the sick one.

    The cache was written only after a SUCCESSFUL probe, so an unresponsive
    host was never cached — and this runs inside the tick callback, on every
    poll, with the loop's cancel checks outside it. One dead host therefore
    re-probed and re-waited forever, freezing progress and swallowing ^C."""
    media._STAGE_CACHE = media._BoundedCache(64)
    calls = {"n": 0}

    from deepme import serve

    def always_fails(spec, s, unit=None, **kw):
        calls["n"] += 1
        raise RuntimeError("host is not answering")

    monkeypatch.setattr(serve, "stage_progress", always_fails)
    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: (_ for _ in ()).throw(httpx.ConnectError("x"))
    )

    for _ in range(5):
        assert media.probe_stage(_Spec(), 60.0, "vid_1") is None
    assert calls["n"] == 1, "a failing probe must back off, not retry every tick"


def test_the_stage_probe_is_bounded_well_under_the_poll_interval(monkeypatch, registry):
    """remote.run defaults to a 600s timeout — correct for an operator running
    a command once, catastrophic inside a 5s poll tick."""
    media._STAGE_CACHE = media._BoundedCache(64)
    seen = {}

    from deepme import serve

    monkeypatch.setattr(
        serve,
        "stage_progress",
        lambda spec, s, unit=None, **kw: seen.update(kw) or None,
    )
    monkeypatch.setattr(
        httpx, "get", lambda url, **kw: (_ for _ in ()).throw(httpx.ConnectError("x"))
    )
    media.probe_stage(_Spec(), 60.0, "vid_1")
    assert seen["timeout"] == media._STAGE_PROBE_S
    assert media._STAGE_PROBE_S < 20.0, "must not outlast the throttle window"


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
        # **kw, so the double keeps matching the real signature: probe_stage
        # swallows every exception, so a fake that is merely out of date fails
        # as "no progress available" rather than as a broken test.
        lambda spec, s, unit=None, **kw: seen.setdefault("unit", unit),
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
    with pytest.raises(media.TicketLostError, match="keeps forgetting"):
        media.generate_video(_Spec(), "x", out_dir=tmp_path)


def test_resubmit_disabled_turns_a_lost_job_into_a_clean_failure(monkeypatch, tmp_path):
    """Collect-only callers (batch --fetch-existing) exist to wind work DOWN:
    a lost job must become a loud per-item failure, and the fleet must see
    no new submission at all."""
    posts = []
    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: (
            posts.append(url) or _r("POST", url, 200, json={"id": "vid_new", "status": "queued"})
        ),
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, **kw: _r("GET", url, 404, json={"detail": "gone"}),
    )
    with pytest.raises(media.TicketLostError, match="resubmission is disabled"):
        media.generate_video(_Spec(), "x", out_dir=tmp_path, adopt="vid_dead", resubmit=False)
    assert not posts, "a collect-only caller must never submit"


# --------------------------------------------------------------------------
# Stall evidence: absence must be PROVEN, never inferred from a failed read


def _capacity_view(monkeypatch, payload):
    monkeypatch.setattr(httpx, "get", lambda url, **kw: _r("GET", url, 200, json=payload))


def test_an_unreachable_fleet_is_not_evidence_of_absence(monkeypatch, tmp_path):
    """The gateway answers 200 even when a fleet's balancer cannot be asked
    — `reachable: false`, NO running list. Counting that as absence killed
    healthy jobs exactly while a sibling's MP4 saturated the data path: the
    capacity probe timed out behind the download, and clients read their
    own about-to-complete jobs as dead."""
    _capacity_view(
        monkeypatch,
        {"fleets": [{"profile": "h3", "reachable": False, "free": 0}]},
    )
    running, _mean, evidence = media._server_view(_Spec(), "vid_1")
    assert not running
    assert evidence is False, "a failed read wearing a 200 proves nothing"


def test_a_reachable_fleet_reporting_its_running_set_is_evidence(monkeypatch, tmp_path):
    _capacity_view(
        monkeypatch,
        {
            "fleets": [
                {
                    "profile": "h3",
                    "reachable": True,
                    "mean_job_s": 800.0,
                    "running": [{"job_id": "vid_other"}],
                }
            ]
        },
    )
    running, mean, evidence = media._server_view(_Spec(), "vid_1")
    assert (running, mean, evidence) == (False, 800.0, True)
    running, _mean, _e = media._server_view(_Spec(), "vid_other")
    assert running is True


def test_a_flat_balancer_capacity_answer_counts_as_evidence(monkeypatch, tmp_path):
    """Direct-to-balancer callers get the flat shape with no `reachable`
    field at all — the balancer answered for itself, which IS evidence."""
    _capacity_view(monkeypatch, {"mean_job_s": 700.0, "running": [{"job_id": "vid_1"}]})
    running, mean, evidence = media._server_view(_Spec(), "vid_1")
    assert (running, mean, evidence) == (True, 700.0, True)


def test_a_reachable_fleet_with_no_running_key_is_not_evidence():
    """The gap that survived because the rule was written four times.

    `reachable: true` with no `running` list at all is not a report of an
    empty fleet — it is a fleet that did not answer the question. The stall
    verdict checked for the list; the batch-status cross-check checked only
    `reachable`, so the same payload meant "trustworthy" in one place and
    not the other."""
    view = media.capacity_view({"fleets": [{"profile": "h3", "reachable": True, "free": 0}]})
    assert view.complete is False
    assert view.absent("vid_1") is False, "absent() must never be `not holds()`"


def test_capacity_view_is_incomplete_if_any_matching_fleet_is_silent():
    """One good fleet does not redeem a silent one: the job may be on the
    fleet that could not be asked."""
    view = media.capacity_view(
        {
            "fleets": [
                {"profile": "h3", "reachable": True, "running": [{"job_id": "vid_1"}]},
                {"profile": "h3", "reachable": False},
            ]
        }
    )
    assert view.holds("vid_1") is True, "a positive sighting still counts"
    assert view.complete is False
    assert view.absent("vid_2") is False


def test_capacity_view_with_no_matching_fleet_is_not_evidence():
    """Asking about a profile nothing serves reads as an empty running set.
    Treating that as proof of absence would kill every job on a fleet whose
    profile name the caller got wrong."""
    view = media.capacity_view({"fleets": [{"profile": "other", "running": []}]}, "h3")
    assert view.complete is False


# --------------------------------------------------------------------------
# Stall verdicts under a controlled clock: evidence is per-IDENTITY and
# expectations are per-DURATION


class _Clock:
    """Deterministic time: sleeping advances it, nothing else does."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


def _run_poll(monkeypatch, *, handover_at, view, end=None, expected_s=None, deadline=1e9):
    """Drive media._poll against a scripted balancer.

    The status endpoint answers as a queued ticket until `handover_at`,
    then as the rewritten real-job body (the balancer's transparent ticket
    rewrite); `view` scripts _server_view; `end` optionally completes the
    job at a given clock time.
    """
    clock = _Clock()
    monkeypatch.setattr(media.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(media.time, "sleep", clock.sleep)
    monkeypatch.setattr(media, "probe_stage", lambda *a, **k: None)

    def status(url, **kw):
        if end is not None and clock.t >= end:
            return _r("GET", url, 200, json={"id": "vid_real", "status": "completed"})
        if clock.t >= handover_at:
            return _r("GET", url, 200, json={"id": "vid_real", "status": "queued"})
        return _r("GET", url, 200, json={"id": "q_t1", "status": "queued", "queue_position": 1})

    monkeypatch.setattr(httpx, "get", status)
    monkeypatch.setattr(media, "_server_view", lambda spec, mid: view(clock.t, mid))
    return media._poll(
        _Spec(),
        "q_t1",
        deadline=deadline,
        every=5.0,
        started=0.0,
        expected_s=expected_s,
    )


def _poll_against(monkeypatch, status, *, view=None, deadline=1e9):
    """Drive media._poll against an arbitrary scripted status endpoint.

    Every test below asserts the poll TERMINATES. Before these, each of these
    server states polled forever: the loop's only exits were terminal status,
    a 404 streak, or a verdict that required trusted capacity evidence about a
    non-ticket id — and none of these states can ever produce that.
    """
    clock = _Clock()
    monkeypatch.setattr(media.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(media.time, "sleep", clock.sleep)
    monkeypatch.setattr(media, "probe_stage", lambda *a, **k: None)
    monkeypatch.setattr(httpx, "get", lambda url, **kw: status(clock, url, **kw))
    monkeypatch.setattr(media, "_server_view", view or (lambda spec, mid: (False, None, False)))
    return media._poll(_Spec(), "q_t1", deadline=deadline, every=5.0, started=0.0)


def test_a_ticket_on_a_fleet_with_no_accepting_replicas_is_executed(monkeypatch):
    """THE forever-hang. A queued ticket is exempt from every other verdict —
    waiting is legitimate and unbounded — so when the fleet can never start it
    the client waits forever and is right to. `accepting_replicas: 0` is the
    fleet stating no replica can EVER take this, which is a positive fact, not
    a guess about elapsed time."""

    def dead_fleet(clock, url, **kw):
        return _r(
            "GET",
            url,
            200,
            json={
                "id": "q_t1",
                "status": "queued",
                "queue_position": 0,
                "accepting_replicas": 0,
            },
        )

    with pytest.raises(media.StalledError, match="NO replica accepting work"):
        _poll_against(monkeypatch, dead_fleet)


def test_a_ticket_merely_waiting_behind_work_is_never_executed(monkeypatch):
    """The other half: a fleet that CAN serve, with the ticket behind a long
    generation, must keep waiting however long that takes. Killing this is
    exactly the behaviour the platform refuses."""

    def busy_fleet(clock, url, **kw):
        if clock.t >= 100_000.0:
            return _r("GET", url, 200, json={"id": "vid_real", "status": "completed"})
        return _r(
            "GET",
            url,
            200,
            json={
                "id": "q_t1",
                "status": "queued",
                "queue_position": 3,
                "accepting_replicas": 4,
            },
        )

    job = _poll_against(monkeypatch, busy_fleet)
    assert job["status"] == "completed", "a servable queue must wait indefinitely"


def test_a_rejected_credential_is_reported_not_retried(monkeypatch):
    """401 is the same answer every time. With no else-branch it fell through
    every case — `lost` never incremented, no verdict armed — so the caller
    polled a 401 forever and was never told the key was wrong."""

    def unauthorized(clock, url, **kw):
        return _r("GET", url, 401, json={"detail": "nope"})

    with pytest.raises(media.GenerationRejected, match="credentials"):
        _poll_against(monkeypatch, unauthorized)


def test_a_persistently_unreadable_platform_is_executed(monkeypatch):
    """A crash-looping service answering 5xx forever. The 404 branch's comment
    promised 5xx 'must not keep a dead id alive forever'; falling through
    without counting achieved precisely that."""

    def broken(clock, url, **kw):
        return _r("GET", url, 502, json={"detail": "bad gateway"})

    with pytest.raises(media.StalledError, match="consecutive polls"):
        _poll_against(monkeypatch, broken)


def test_an_unreachable_platform_is_executed(monkeypatch):
    """The gateway goes away mid-run: no response at all, so the 404 branch is
    unreachable and the capacity read fails and is suppressed. Nothing counted
    transport failures, so the caller polled a dead host indefinitely."""

    def gone(clock, url, **kw):
        raise httpx.ConnectError("connection refused")

    with pytest.raises(media.StalledError, match="could not reach"):
        _poll_against(monkeypatch, gone)


def test_queue_wait_absence_does_not_execute_a_fresh_handover(monkeypatch, tmp_path):
    """THE 39x bug: a ticket is rightly absent from the running set for its
    whole queue wait, but the counter carried across the handover — verdict
    1 fired the instant the q_ exemption lifted, on evidence about an id
    the work no longer was. A 30-minute queue must not kill a job seconds
    after a replica finally takes it."""
    job = _run_poll(
        monkeypatch,
        handover_at=1800.0,  # 30 min queued: 30 evidence beats of "absence"
        view=lambda t, mid: (False, 800.0, True),  # never seen running at all
        end=2050.0,  # completes 250s after handover — under the fresh grace
    )
    assert job["status"] == "completed"


def test_a_handed_over_job_confirmed_absent_is_still_executed(monkeypatch, tmp_path):
    """The detector must still detect: absence PROVEN about the real id,
    past the grace window, counted from the handover."""
    with pytest.raises(media.StalledError, match="a replica took it"):
        _run_poll(
            monkeypatch,
            handover_at=600.0,
            view=lambda t, mid: (False, 800.0, True),
        )


def test_a_long_clip_is_not_executed_against_a_short_clip_mean(monkeypatch, tmp_path):
    """A 15s clip measures 5.7x a 4s clip (superlinear), but the fleet mean
    is dominated by short clips — 3x mean sits BELOW a 15s clip's honest
    wall time, and every long clip on a mixed fleet was killed
    mid-generation as 'wedged'. The measured duration curve is the
    per-request baseline."""
    running_view = lambda t, mid: (True, 843.0, True)  # noqa: E731
    # Old rule killed at 3x843=2529s; the 15s expectation (~3645s) lifts the
    # limit to ~10935s. Ends by deadline at 4500s: past the old limit, under
    # the honest one — must be a timeout, never a stall verdict.
    with pytest.raises(media.GenerationTimeout, match="did not finish"):
        _run_poll(
            monkeypatch,
            handover_at=0.0,
            view=running_view,
            expected_s=3644.7,
            deadline=4500.0,
        )
    # And a genuinely wedged engine is still caught past the lifted limit.
    with pytest.raises(media.StalledError, match="almost certainly wedged"):
        _run_poll(
            monkeypatch,
            handover_at=0.0,
            view=running_view,
            expected_s=3644.7,
        )


def test_expected_wall_time_interpolates_the_measured_curve(monkeypatch):
    class _Cfg:
        duration_points: ClassVar[list[dict]] = [
            {"seconds": 4.0, "gpu_seconds": 644.2},
            {"seconds": 10.0, "gpu_seconds": 1796.8},
            {"seconds": 15.0, "gpu_seconds": 3644.7},
        ]

    from deepme import economics

    monkeypatch.setattr(economics, "load", lambda path=None: _Cfg())
    assert media._expected_job_s(4.0) == pytest.approx(644.2)
    assert media._expected_job_s(15.0) == pytest.approx(3644.7)
    mid = media._expected_job_s(12.5)
    assert 1796.8 < mid < 3644.7, "between the measured points"
    assert media._expected_job_s(20.0) > 3644.7, "extrapolates past the last"
    assert media._expected_job_s(None) is None


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
