"""Work held before it reaches a replica.

The billing line runs through here: a request inside an engine is consuming a
GPU and cannot be cancelled for free, so everything a customer may still
change has to happen while the request is still ours.
"""

import pytest

from deepme.balancer.admission import PRIORITY_HIGH, PRIORITY_LOW, AdmissionQueue


@pytest.fixture
def q():
    return AdmissionQueue()


def test_same_priority_is_served_in_arrival_order(q):
    """A burst of equal-priority work must be fair, not dict-ordered."""
    ids = [q.enqueue({"prompt": f"p{i}"}).ticket_id for i in range(4)]
    assert [t.ticket_id for t in q.waiting()] == ids


def test_higher_priority_jumps_the_line(q):
    normal = q.enqueue({"prompt": "a"})
    urgent = q.enqueue({"prompt": "b"}, priority=PRIORITY_HIGH)
    assert q.next_ready().ticket_id == urgent.ticket_id
    assert q.position(normal.ticket_id) == 1


def test_priority_does_not_starve_arrival_order_within_a_tier(q):
    first = q.enqueue({"prompt": "a"}, priority=PRIORITY_HIGH)
    second = q.enqueue({"prompt": "b"}, priority=PRIORITY_HIGH)
    q.enqueue({"prompt": "c"}, priority=PRIORITY_LOW)
    assert [t.ticket_id for t in q.waiting()][:2] == [first.ticket_id, second.ticket_id]


def test_position_zero_means_next_to_be_submitted(q):
    t = q.enqueue({"prompt": "a"})
    assert q.position(t.ticket_id) == 0


def test_cancelling_before_submission_succeeds_and_removes_it_from_the_line(q):
    t = q.enqueue({"prompt": "a"})
    ok, why = q.cancel(t.ticket_id)
    assert ok and "before submission" in why
    assert q.waiting() == []
    assert q.position(t.ticket_id) is None


def test_cancelling_after_submission_is_refused_with_the_reason(q):
    """Once a replica has it, stopping costs the same and recovers nothing."""
    t = q.enqueue({"prompt": "a"})
    q.mark_submitted(t.ticket_id, "vid_1", "http://a")
    ok, why = q.cancel(t.ticket_id)
    assert not ok
    assert "cannot be cancelled" in why


def test_cancelling_an_unknown_ticket_is_reported_not_crashed(q):
    ok, why = q.cancel("q_nope")
    assert not ok and "no such ticket" in why


def test_cancelling_twice_is_idempotent(q):
    t = q.enqueue({"prompt": "a"})
    q.cancel(t.ticket_id)
    ok, _ = q.cancel(t.ticket_id)
    assert ok


def test_a_cancelled_ticket_is_never_handed_to_a_replica(q):
    a = q.enqueue({"prompt": "a"})
    b = q.enqueue({"prompt": "b"})
    q.cancel(a.ticket_id)
    assert q.next_ready().ticket_id == b.ticket_id


def test_billable_only_once_a_replica_has_it(q):
    t = q.enqueue({"prompt": "a"})
    assert not t.billable
    q.mark_submitted(t.ticket_id, "vid_1", "http://a")
    assert q.get(t.ticket_id).billable


def test_submission_records_where_the_job_went(q):
    t = q.enqueue({"prompt": "a"})
    q.mark_submitted(t.ticket_id, "vid_9", "http://b")
    got = q.get(t.ticket_id)
    assert (got.job_id, got.backend) == ("vid_9", "http://b")


def test_a_cancelled_ticket_cannot_later_be_submitted(q):
    t = q.enqueue({"prompt": "a"})
    q.cancel(t.ticket_id)
    q.mark_submitted(t.ticket_id, "vid_1", "http://a")
    assert q.get(t.ticket_id).job_id is None


def test_reprioritising_reorders_waiting_work(q):
    a = q.enqueue({"prompt": "a"})
    b = q.enqueue({"prompt": "b"})
    assert q.reprioritise(b.ticket_id, PRIORITY_HIGH)
    assert q.next_ready().ticket_id == b.ticket_id
    assert q.position(a.ticket_id) == 1


def test_a_running_job_cannot_be_reordered(q):
    t = q.enqueue({"prompt": "a"})
    q.mark_submitted(t.ticket_id, "vid_1", "http://a")
    assert not q.reprioritise(t.ticket_id, PRIORITY_HIGH)


def test_wait_estimate_scales_with_position_and_fleet_width(q):
    for _ in range(8):
        q.enqueue({"prompt": "x"})
    last = q.waiting()[-1].ticket_id
    assert q.wait_estimate_s(last, 900.0, 4) == pytest.approx(7 / 4 * 900.0)


def test_no_wait_estimate_without_measured_history(q):
    t = q.enqueue({"prompt": "a"})
    assert q.wait_estimate_s(t.ticket_id, 0.0, 4) is None


def test_settled_tickets_are_pruned_but_waiting_ones_never_are(q):
    for i in range(10):
        t = q.enqueue({"prompt": f"p{i}"})
        if i < 6:
            q.mark_submitted(t.ticket_id, f"v{i}", "http://a")
    assert q.prune(keep=2) == 4
    assert q.snapshot()["waiting"] == 4


def test_snapshot_reports_the_queue_for_observability(q):
    a = q.enqueue({"prompt": "a"})
    q.enqueue({"prompt": "b"})
    q.mark_submitted(a.ticket_id, "v", "http://a")
    snap = q.snapshot()
    assert snap["waiting"] == 1
    assert snap["by_state"]["submitted"] == 1


# --------------------------------------------------------------------------
# The claim protocol: what makes concurrent pumps and mid-flight cancels safe


def test_a_claimed_ticket_is_invisible_to_the_next_pump():
    """The double-submit defect: submission is an HTTP round-trip (an await),
    and a second pump entering during it saw the same WAITING ticket and
    submitted it to a second replica — one generation, run and billed twice.
    Claiming is synchronous, so it is atomic between awaits."""
    q = AdmissionQueue()
    t = q.enqueue({"path": "/v1/videos"})
    assert q.claim(t.ticket_id) is True
    assert q.next_ready() is None, "a concurrent pump finds nothing to take"
    assert q.claim(t.ticket_id) is False, "and cannot claim it twice"


def test_cancel_is_refused_while_the_submission_is_on_the_wire():
    """A cancel that 'succeeded' during the round-trip told the customer
    'cancelled, not billable' while their job ran to completion anyway."""
    q = AdmissionQueue()
    t = q.enqueue({"path": "/v1/videos"})
    q.claim(t.ticket_id)
    ok, why = q.cancel(t.ticket_id)
    assert ok is False
    assert "handed to a replica" in why


def test_a_transport_failure_returns_the_claim_to_the_queue():
    """The request never reached a replica, so retrying elsewhere is safe."""
    q = AdmissionQueue()
    t = q.enqueue({"path": "/v1/videos"})
    q.claim(t.ticket_id)
    q.release_claim(t.ticket_id)
    assert q.next_ready() is t, "available again, nothing lost"


def test_a_rejected_ticket_is_terminal_and_says_why():
    """Retiring a refused ticket as 'submitted' left callers polling
    'queued' forever for work that never ran and never will."""
    q = AdmissionQueue()
    t = q.enqueue({"path": "/v1/videos"})
    q.claim(t.ticket_id)
    q.mark_rejected(t.ticket_id, "duration_seconds must be <= 15")
    assert t.state == "rejected"
    assert t.error == "duration_seconds must be <= 15"
    assert t.billable is False
    ok, _why = q.cancel(t.ticket_id)
    assert ok is True, "cancelling failed work is a no-op, not an error"
    assert q.next_ready() is None, "and it is never submitted"


def test_ticket_ids_carry_the_queue_salt():
    """Every balancer minted q_00000001 — and the gateway, which keys
    ownership by bare id, routed one customer's CANCEL to another
    customer's ticket on a different fleet."""
    a, b = AdmissionQueue(salt="video-ab12-"), AdmissionQueue(salt="ref-cd34-")
    ta, tb = a.enqueue({}), b.enqueue({})
    assert ta.ticket_id != tb.ticket_id
    assert ta.ticket_id.startswith("q_video-ab12-")
    assert tb.ticket_id.startswith("q_ref-cd34-")
