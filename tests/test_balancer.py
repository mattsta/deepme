"""Load-balancer decisions: which replica, and where a job already lives.

A media replica serves one job at a time and can hold it for tens of minutes,
so a wrong choice here is not a small latency cost — it queues a request
behind a 40-minute generation while a neighbour idles. And because job ids
are minted per replica, sending a poll to the wrong one strands the result.
"""

from deepme.balancer.balance import Balancer

URLS = ["http://a:8030", "http://b:8032", "http://c:8034"]


def test_work_spreads_before_any_replica_doubles_up():
    b = Balancer(list(URLS))
    picks = []
    for _ in range(3):
        u = b.pick()
        b.start(u)
        picks.append(u)
    assert sorted(picks) == sorted(URLS)


def test_the_idle_replica_wins_over_a_busy_one():
    """The whole point: never queue behind a long job when someone is free."""
    b = Balancer(list(URLS))
    b.start(URLS[0])
    b.start(URLS[1])
    assert b.pick() == URLS[2]


def test_a_finished_job_frees_its_replica():
    b = Balancer(list(URLS))
    began = b.start(URLS[0])
    b.start(URLS[1])
    b.start(URLS[2])
    b.finish(URLS[0], began)
    assert b.pick() == URLS[0]


def test_unhealthy_replicas_are_not_selected():
    b = Balancer(list(URLS))
    for u in URLS[1:]:
        b.mark(u, healthy=False)
    assert b.pick() == URLS[0]


def test_no_healthy_replica_returns_nothing_rather_than_guessing():
    b = Balancer(list(URLS))
    for u in URLS:
        b.mark(u, healthy=False)
    assert b.pick() is None


def test_a_replica_that_recovers_is_used_again():
    b = Balancer(list(URLS))
    b.mark(URLS[0], healthy=False)
    assert b.pick() != URLS[0]
    b.mark(URLS[0], healthy=True)
    assert b.pick() == URLS[0]  # fewest started


def test_a_job_is_routed_back_to_the_replica_that_created_it():
    b = Balancer(list(URLS))
    b.remember("vid_1", URLS[2])
    assert b.owner("vid_1") == URLS[2]


def test_an_unknown_job_reports_no_owner_so_the_caller_can_go_looking():
    assert Balancer(list(URLS)).owner("vid_missing") is None


def test_ownership_survives_the_replica_becoming_busy_again():
    b = Balancer(list(URLS))
    b.remember("vid_1", URLS[0])
    for _ in range(5):
        b.start(URLS[0])
    assert b.owner("vid_1") == URLS[0]


def test_job_tracking_is_bounded_so_a_long_lived_balancer_cannot_grow_forever():
    b = Balancer(list(URLS), max_jobs=3)
    for i in range(5):
        b.remember(f"vid_{i}", URLS[0])
    assert b.snapshot()["tracked_jobs"] == 3
    assert b.owner("vid_0") is None  # evicted: costs a probe, not correctness
    assert b.owner("vid_4") == URLS[0]


def test_forgetting_a_deleted_job_releases_its_mapping():
    b = Balancer(list(URLS))
    b.remember("vid_1", URLS[0])
    b.forget("vid_1")
    assert b.owner("vid_1") is None


def test_empty_job_ids_are_never_tracked():
    b = Balancer(list(URLS))
    b.remember("", URLS[0])
    assert b.snapshot()["tracked_jobs"] == 0


def test_failures_are_counted_separately_from_completions():
    b = Balancer(list(URLS))
    began = b.start(URLS[0])
    b.finish(URLS[0], began, ok=False)
    st = b.stats[URLS[0]]
    assert (st.completed, st.failed, st.inflight) == (0, 1, 0)


def test_inflight_never_goes_negative():
    b = Balancer(list(URLS))
    b.finish(URLS[0], 0.0)
    assert b.stats[URLS[0]].inflight == 0


def test_mean_job_time_is_reported_for_observing_load():
    b = Balancer(list(URLS))
    st = b.stats[URLS[0]]
    st.completed, st.total_busy_s = 2, 100.0
    assert st.mean_job_s == 50.0
    assert b.snapshot()["backends"][0]["mean_job_s"] == 50.0


def test_a_submitted_job_keeps_its_replica_busy_for_the_whole_job():
    """POST returns a job id in milliseconds while the replica works for
    minutes. Timing the HTTP call records ~0s AND frees the replica
    immediately, so the next request is routed onto a busy one."""
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    assert b.stats[URLS[0]].inflight == 1
    assert b.pick() != URLS[0], "a replica running a job must not be chosen"


def test_occupancy_ends_when_the_job_reaches_a_terminal_status():
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    b.end_job("vid_1")
    st = b.stats[URLS[0]]
    assert st.inflight == 0 and st.completed == 1
    # Eligible again — though an untouched replica is preferred, which is the
    # load spreading working, not a failure to release.
    b.begin_job("vid_2", URLS[1])
    b.begin_job("vid_3", URLS[2])
    assert b.pick() == URLS[0]


def test_job_duration_is_measured_not_reported_as_zero(monkeypatch):
    import itertools

    clock = itertools.count(start=100.0, step=846.0)
    monkeypatch.setattr("deepme.balancer.balance.time.monotonic", lambda: next(clock))
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    b.end_job("vid_1")
    assert b.stats[URLS[0]].mean_job_s == 846.0


def test_a_failed_job_is_counted_as_failed_and_still_frees_the_replica():
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    b.end_job("vid_1", ok=False)
    st = b.stats[URLS[0]]
    assert (st.failed, st.completed, st.inflight) == (1, 0, 0)


def test_a_duplicate_terminal_poll_does_not_double_count():
    """Clients poll repeatedly; only the first terminal reading closes a job."""
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    b.end_job("vid_1")
    b.end_job("vid_1")
    assert b.stats[URLS[0]].completed == 1


def test_resubmitting_a_known_job_id_does_not_double_book():
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    b.begin_job("vid_1", URLS[0])
    assert b.stats[URLS[0]].inflight == 1


def test_four_jobs_land_on_four_replicas_and_a_fifth_waits_for_none():
    """With job-scoped occupancy the fifth submission goes to whichever
    replica has the fewest open jobs, not to whoever answered last."""
    b = Balancer(list(URLS))
    for i, url in enumerate(URLS):
        b.begin_job(f"vid_{i}", url)
    assert all(s.inflight == 1 for s in b.stats.values())
    b.end_job("vid_1")
    assert b.pick() == URLS[1]


def test_active_jobs_are_reported_for_observability():
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    assert b.snapshot()["active_jobs"] == 1


def test_queue_position_tells_a_caller_if_it_is_running_or_waiting(monkeypatch):
    """The job API says "queued" from submission to completion either way, so
    without this a caller cannot tell generating from stuck behind others."""
    import itertools

    clock = itertools.count(start=100.0, step=1.0)
    monkeypatch.setattr("deepme.balancer.balance.time.monotonic", lambda: next(clock))
    b = Balancer(list(URLS))
    b.begin_job("first", URLS[0])
    b.begin_job("second", URLS[0])
    b.begin_job("elsewhere", URLS[1])
    assert b.queue_position("first") == 0  # running now
    assert b.queue_position("second") == 1  # one ahead of it
    assert b.queue_position("elsewhere") == 0  # different replica, not queued


def test_an_unknown_job_has_no_queue_position():
    assert Balancer(list(URLS)).queue_position("nope") == 0


def test_no_eta_is_offered_before_a_replica_has_finished_anything():
    """Better silence than an invented number on the first request."""
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    assert b.eta_hint_s("vid_1") is None


def test_eta_accounts_for_jobs_queued_ahead(monkeypatch):
    import itertools

    # Distinct start times, or neither job is "ahead" of the other.
    clock = itertools.count(start=0.0, step=1.0)
    monkeypatch.setattr("deepme.balancer.balance.time.monotonic", lambda: next(clock))
    b = Balancer(list(URLS))
    b.stats[URLS[0]].completed = 1
    b.stats[URLS[0]].total_busy_s = 600.0  # mean job 600s
    b.begin_job("running", URLS[0])
    b.begin_job("waiting", URLS[0])
    # The one behind must wait for the one in front, plus its own run.
    assert b.eta_hint_s("waiting") > b.eta_hint_s("running")


def test_a_job_open_far_longer_than_the_replica_takes_is_reapable(monkeypatch):
    """A client that disconnects never polls a terminal status, so the job
    stays open forever and the replica reads as permanently busy."""
    b = Balancer(list(URLS))
    b.stats[URLS[0]].completed = 2
    b.stats[URLS[0]].total_busy_s = 600.0  # mean 300s
    b.begin_job("vid_1", URLS[0])
    began = b._jobs["vid_1"][1]
    assert b.stale_jobs(now=began + 299) == []  # still plausible
    assert b.stale_jobs(now=began + 901) == [("vid_1", URLS[0])]  # 3x mean


def test_a_replica_with_no_history_uses_a_generous_floor():
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    began = b._jobs["vid_1"][1]
    assert b.stale_jobs(now=began + 1799) == []
    assert b.stale_jobs(now=began + 1801) == [("vid_1", URLS[0])]


def test_reaping_frees_the_replica_for_routing():
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    assert b.pick() != URLS[0]
    b.end_job("vid_1")
    b.begin_job("vid_2", URLS[1])
    b.begin_job("vid_3", URLS[2])
    assert b.pick() == URLS[0]


def test_a_job_that_is_genuinely_running_is_never_reaped():
    """Closing a live job would free a replica for work it cannot start."""
    b = Balancer(list(URLS))
    b.stats[URLS[0]].completed = 1
    b.stats[URLS[0]].total_busy_s = 900.0
    b.begin_job("vid_1", URLS[0])
    began = b._jobs["vid_1"][1]
    assert b.stale_jobs(now=began + 60) == []


def test_only_the_first_close_of_a_job_reports_true():
    """end_job's return is the terminal-accounting dedupe: client poll, SSE
    watcher and reconciler all see the same terminal status, and only the
    first closer may record it — double-noting put every job in the recent
    ring twice, the second entry stripped of its metadata."""
    b = Balancer(list(URLS))
    b.begin_job("vid_1", URLS[0])
    assert b.end_job("vid_1") is True
    assert b.end_job("vid_1") is False, "already closed: not yours to record"
    assert b.end_job("vid_never_started") is False
    assert b.stats[URLS[0]].completed == 1, "counted exactly once"


def test_owner_eviction_spares_open_jobs_and_refreshes_hot_ids():
    """An open job's owner is load-bearing routing — evicting it sent the
    next poll through the owner probe, the misattribution trigger. And
    recency, not first-seen order, decides who ages out."""
    b = Balancer(list(URLS))
    b.max_jobs = 2
    b.begin_job("vid_open", URLS[0])  # open: also remembered
    b.remember("vid_done", URLS[1])
    # Re-remembering refreshes: vid_done is now most recent.
    b.remember("vid_done", URLS[1])
    b.remember("vid_new", URLS[2])  # over cap: must evict a CLOSED id
    assert b.owner("vid_open") == URLS[0], "open jobs are never evicted"
    assert b.owner("vid_new") == URLS[2]
    assert b.owner("vid_done") is None, "the least-recent closed id went"
