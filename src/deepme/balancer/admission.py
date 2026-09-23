"""Admission queue: work held BEFORE it reaches a replica.

Billing draws the line here. A request that has reached a replica is
consuming a GPU and is billable, and cancelling it recovers nothing — so
this service does not support cancel-in-flight. Everything a customer may
still change (cancel, reprioritise, see their place in line) has to happen
while the request is still ours, which means there must be a place for it to
wait that is not inside an engine.

Without this the balancer submits on arrival, so those states do not exist:
a caller is either running or rejected, with nothing in between.

Pure logic — no HTTP, no engine — so ordering, cancellation and position
reporting can be tested exhaustively without a fleet.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

WAITING, SUBMITTING, SUBMITTED, REJECTED, CANCELLED = (
    "waiting",
    "submitting",
    "submitted",
    "rejected",
    "cancelled",
)

# Lower number = served first. Named so a service layer maps its own tiers
# onto them without inventing magic integers.
PRIORITY_HIGH, PRIORITY_NORMAL, PRIORITY_LOW = 0, 5, 9


@dataclass
class Ticket:
    """One admitted request, before (and while) it becomes a job."""

    ticket_id: str
    body: dict
    priority: int = PRIORITY_NORMAL
    queued_at: float = field(default_factory=time.monotonic)
    seq: int = 0  # arrival order, to keep same-priority service fair
    state: str = WAITING
    job_id: str | None = None  # set once a replica has accepted it
    backend: str | None = None
    error: str | None = None  # why, when state == REJECTED

    @property
    def billable(self) -> bool:
        """True once a replica is working on it — the moment it costs us."""
        return self.state == SUBMITTED


@dataclass
class AdmissionQueue:
    """Holds requests until a replica can take them.

    Ordering is (priority, arrival). Arrival breaks ties so that a burst of
    same-priority work is served fairly instead of by dict iteration order.

    IN MEMORY ONLY, and that is an operational hazard worth knowing about:
    restarting a balancer silently discards every WAITING ticket, and the
    customers holding those ids will poll a 404 forever. Submitted work
    survives, because it lives on the replica and can be recovered by probing
    (see _find_owner) — it is precisely the not-yet-billable work that is
    lost. Deploy with an empty queue, or drain first: `GET /v1/queue`.

    To make it durable: tickets are already a plain dataclass with a stable
    id, so an append-only journal (one JSON line per enqueue/submit/cancel)
    written beside the unit and replayed on startup is enough — there is no
    need for a database. The replay must drop tickets whose job_id now reads
    terminal on its replica, or a restart would resubmit finished work and
    bill for it twice.
    """

    # Distinguishes this queue's tickets from every other queue's, forever.
    # Ids travel through the gateway, which keys ownership by bare id — two
    # balancers both minting q_00000001 had it routing one customer's cancel
    # to another customer's work.
    salt: str = ""
    _tickets: dict[str, Ticket] = field(default_factory=dict)
    _counter: itertools.count = field(default_factory=itertools.count)
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def enqueue(self, body: dict, *, priority: int = PRIORITY_NORMAL) -> Ticket:
        ticket = Ticket(
            ticket_id=f"q_{self.salt}{next(self._ids):08d}",
            body=dict(body),
            priority=priority,
            seq=next(self._counter),
        )
        self._tickets[ticket.ticket_id] = ticket
        return ticket

    def get(self, ticket_id: str) -> Ticket | None:
        return self._tickets.get(ticket_id)

    def waiting(self) -> list[Ticket]:
        """Everything still ours, in the order it will be served."""
        return sorted(
            (t for t in self._tickets.values() if t.state == WAITING),
            key=lambda t: (t.priority, t.seq),
        )

    def next_ready(self) -> Ticket | None:
        return next(iter(self.waiting()), None)

    def claim(self, ticket_id: str) -> bool:
        """Take exclusive ownership of a ticket for submission.

        The submission itself is an HTTP round-trip — an await, during which
        anything can interleave on the event loop. Claiming FIRST (this is
        synchronous, so it is atomic between awaits) is what makes the pump
        safe to enter concurrently: a second pump's next_ready() no longer
        sees this ticket, and a cancel during the round-trip is refused
        rather than silently racing the replica. Without this, one ticket
        was submitted to two replicas — the same 40-minute generation run
        and billed twice.
        """
        ticket = self._tickets.get(ticket_id)
        if ticket is None or ticket.state != WAITING:
            return False
        ticket.state = SUBMITTING
        return True

    def release_claim(self, ticket_id: str) -> None:
        """Return a claimed ticket to the queue — the submission never
        reached a replica (transport error), so retrying it elsewhere is
        safe and loses nothing."""
        ticket = self._tickets.get(ticket_id)
        if ticket is not None and ticket.state == SUBMITTING:
            ticket.state = WAITING

    def mark_rejected(self, ticket_id: str, detail: str) -> None:
        """The replica answered and said no. Terminal, and it must SAY so:
        retiring a rejected ticket as 'submitted' left callers polling
        'queued' forever for work that never ran and never will."""
        ticket = self._tickets.get(ticket_id)
        if ticket is None or ticket.state not in (WAITING, SUBMITTING):
            return
        ticket.state = REJECTED
        ticket.error = detail

    def position(self, ticket_id: str) -> int | None:
        """0 = next to be submitted. None once it is no longer waiting."""
        ticket = self._tickets.get(ticket_id)
        if ticket is None or ticket.state != WAITING:
            return None
        return self.waiting().index(ticket)

    def cancel(self, ticket_id: str) -> tuple[bool, str]:
        """Cancel while it is still waiting.

        Refuses once submitted, and says why: the work is already occupying a
        replica, so stopping it would cost the same and recover nothing.
        SUBMITTING is refused too — the request is on the wire to a replica
        right now, and a cancel that "succeeded" during that round-trip left
        the customer told 'cancelled, not billable' while their generation
        ran to completion anyway.
        """
        ticket = self._tickets.get(ticket_id)
        if ticket is None:
            return False, "no such ticket"
        if ticket.state == SUBMITTING:
            return False, (
                "being handed to a replica right now — retry in a moment if "
                "the handover fails, otherwise it will be running"
            )
        if ticket.state == SUBMITTED:
            return False, (
                "already submitted to a replica and is consuming GPU time — "
                "in-flight work cannot be cancelled"
            )
        if ticket.state == REJECTED:
            return True, "already failed — nothing to cancel"
        if ticket.state == CANCELLED:
            return True, "already cancelled"
        ticket.state = CANCELLED
        return True, "cancelled before submission"

    def mark_submitted(self, ticket_id: str, job_id: str, backend: str) -> None:
        ticket = self._tickets.get(ticket_id)
        if ticket is None or ticket.state not in (WAITING, SUBMITTING):
            return
        ticket.state = SUBMITTED
        ticket.job_id = job_id
        ticket.backend = backend

    def reprioritise(self, ticket_id: str, priority: int) -> bool:
        """Only meaningful while waiting; a running job cannot be reordered.

        NOT YET EXPOSED OVER HTTP. Priority can be set at submission but not
        changed afterwards, because no route calls this. The endpoint would
        sit next to cancel_ticket in server.py — `PATCH /v1/queue/{ticket}`
        with `{"priority": N}` — and the gateway would need the matching
        forwarder its DELETE already has. Left unwired rather than guessed at:
        a service layer that wants promotion may also want a fairness rule
        (a cap on promotions, or ageing so low-priority work cannot starve),
        and that policy belongs with whoever is charging for it.
        """
        ticket = self._tickets.get(ticket_id)
        if ticket is None or ticket.state != WAITING:
            return False
        ticket.priority = priority
        return True

    def wait_estimate_s(self, ticket_id: str, mean_job_s: float, replicas: int) -> float | None:
        """How long until this ticket is likely to start.

        Derived from how many are ahead and how fast the fleet drains them.
        None when there is no measured history — an invented number is worse
        than telling a customer we do not know yet.
        """
        position = self.position(ticket_id)
        if position is None or mean_job_s <= 0 or replicas <= 0:
            return None
        return position / replicas * mean_job_s

    def forget(self, ticket_id: str) -> None:
        self._tickets.pop(ticket_id, None)

    def prune(self, *, keep: int = 5000) -> int:
        """Drop the oldest settled tickets so a long-lived queue stays bounded.

        WAITING is never dropped, and neither is SUBMITTING: that ticket's
        POST is on the wire RIGHT NOW, and deleting it makes the completion
        callbacks (mark_submitted / release_claim) silent no-ops — the job
        runs but the handover is lost, or a transport-error requeue vanishes.
        """
        settled = sorted(
            (t for t in self._tickets.values() if t.state not in (WAITING, SUBMITTING)),
            key=lambda t: t.seq,
        )
        excess = max(len(settled) - keep, 0)
        for ticket in settled[:excess]:
            self._tickets.pop(ticket.ticket_id, None)
        return excess

    def snapshot(self) -> dict:
        states: dict[str, int] = {}
        for ticket in self._tickets.values():
            states[ticket.state] = states.get(ticket.state, 0) + 1
        return {"tickets": len(self._tickets), "by_state": states, "waiting": len(self.waiting())}
