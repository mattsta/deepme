"""Replica selection and job ownership for the load balancer.

Pure logic, no I/O, so the decisions can be tested without standing up
replicas. Two problems, and the second is the one that bites:

1. WHICH replica gets a new request. Media engines process one job at a time
   (sglang's diffusion server runs batching_max_size=1), and a job can hold a
   replica for tens of minutes, so "least outstanding work" is the only
   sensible rule — round-robin would queue a request behind a 40-minute
   generation while a neighbour sits idle.

2. WHERE a job already lives. Job ids are minted per replica and mean nothing
   to the others, so every follow-up poll and download for a job MUST return
   to the replica that created it. Losing that mapping strands the result.
"""

from __future__ import annotations

import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass
class BackendStats:
    url: str
    inflight: int = 0
    started: int = 0
    completed: int = 0
    failed: int = 0
    total_busy_s: float = 0.0
    healthy: bool = True
    # Draining is DISTINCT from unhealthy: a draining replica takes no new
    # work but its in-flight jobs are still followable and still counted.
    # Conflating the two would make _find_owner skip it, stranding exactly
    # the jobs a drain exists to protect.
    draining: bool = False
    # Slots held across a submission round-trip (see reserve): work that is
    # on the wire but not yet a job. Detach must count these too.
    reserved: int = 0

    @property
    def accepting(self) -> bool:
        return self.healthy and not self.draining

    @property
    def mean_job_s(self) -> float:
        return self.total_busy_s / self.completed if self.completed else 0.0


@dataclass
class Balancer:
    """Chooses replicas and remembers which one owns each job."""

    urls: list[str]
    # job id -> backend url. Bounded so a long-lived balancer cannot grow without
    # limit; evicting an old entry costs a fan-out probe, never correctness.
    max_jobs: int = 20_000
    # Jobs force-closed because no client ever read their terminal status —
    # a billing-relevant self-correction that must be countable, not silent.
    reaped: int = 0
    stats: dict[str, BackendStats] = field(default_factory=dict)
    _owner: OrderedDict[str, str] = field(default_factory=OrderedDict)
    # job id -> (backend, started at). Open jobs are what "busy" means here.
    _jobs: dict[str, tuple[str, float]] = field(default_factory=dict)
    # job id -> API collection ("videos"/"images"/...), for reconcile probes.
    _job_collections: dict[str, str] = field(default_factory=dict)
    _order: itertools.count = field(default_factory=itertools.count)

    def __post_init__(self) -> None:
        for u in self.urls:
            self.stats.setdefault(u, BackendStats(url=u))

    # -- selection ---------------------------------------------------------
    def pick(self) -> str | None:
        """The replica most able to start work now.

        Ordered by outstanding jobs, then by how much work it has been given,
        so idle replicas fill before any replica doubles up. Draining
        replicas are never picked — that is what draining means.
        """
        live = [s for s in self.stats.values() if s.accepting]
        if not live:
            return None
        best = min(live, key=lambda s: (s.inflight, s.started))
        return best.url

    # -- fleet membership at runtime ---------------------------------------
    # A mix change must not restart the balancer: a restart discards every
    # waiting ticket and the owner map, punishing customers for an operation
    # that is about replicas, not about them. Membership is therefore
    # mutable in place, and the server persists it so a crash-restart comes
    # back with the CURRENT fleet rather than the one from deploy time.
    def attach(self, url: str) -> bool:
        """Add a backend; False if it was already attached (re-arming it)."""
        existing = self.stats.get(url)
        if existing is not None:
            existing.healthy = True
            existing.draining = False
            return False
        self.stats[url] = BackendStats(url=url)
        return True

    def drain(self, url: str) -> bool:
        """Stop routing new work to a backend; its jobs remain followable."""
        st = self.stats.get(url)
        if st is None:
            return False
        st.draining = True
        return True

    def undrain(self, url: str) -> bool:
        st = self.stats.get(url)
        if st is None:
            return False
        st.draining = False
        return True

    def detach(self, url: str) -> tuple[bool, str]:
        """Remove a backend entirely. Refused while it still owns work —
        detaching it would strand those jobs mid-generation."""
        st = self.stats.get(url)
        if st is None:
            return False, "not attached"
        open_jobs = [j for j, (u, _) in self._jobs.items() if u == url]
        if open_jobs:
            return False, (
                f"{len(open_jobs)} job(s) still running there — drain and "
                "wait for them to finish first"
            )
        if st.reserved:
            # A submission is on the wire to this backend RIGHT NOW; removing
            # it mid-round-trip orphans the job the response is about to mint.
            return False, "a submission is in flight there — retry in a moment"
        del self.stats[url]
        # Ownership of FINISHED jobs is kept: a late poll for a result that
        # was already downloaded should 404 at the replica, not be re-probed
        # across the fleet.
        return True, "detached"

    def backend_urls(self) -> list[str]:
        return list(self.stats)

    # SUPERSEDED — no production caller. Occupancy spans the JOB, not the
    # HTTP call: use reserve/release for the submission window and
    # begin_job/end_job for the work itself. Timing a submission with this
    # pair is the defect that reported mean_job_s of 0.1s while a replica
    # was twenty minutes into a generation, corrupting every ETA, queue
    # estimate and staleness threshold downstream. Kept only because the
    # counter tests exercise the accounting through them.
    def start(self, url: str) -> float:
        self.stats[url].inflight += 1
        self.stats[url].started += 1
        return time.monotonic()

    def finish(self, url: str, began: float, *, ok: bool = True) -> None:
        st = self.stats[url]
        st.inflight = max(st.inflight - 1, 0)
        st.total_busy_s += max(time.monotonic() - began, 0.0)
        if ok:
            st.completed += 1
        else:
            st.failed += 1

    # -- the submission window ---------------------------------------------
    # Choosing a replica and learning whether it accepted the work are
    # separated by an HTTP round-trip. A slot has to be held across it, or
    # concurrent requests all pick the same idle replica. But that round-trip
    # is NOT the work: timing it as a finished job records ~0.1s against
    # generations that run for twenty minutes, and mean_job_s is what drives
    # every customer-facing ETA, the queue wait estimate, and the stale-job
    # threshold. Occupancy and work are therefore separate operations.
    def reserve(self, url: str) -> float:
        """Hold a slot while we find out whether the replica took the work."""
        st = self.stats[url]
        st.inflight += 1
        # Reservations are tracked separately from jobs so detach() can see
        # them: a backend removed during the submission round-trip left a
        # phantom job it could never close (KeyError swallowed downstream).
        st.reserved += 1
        return time.monotonic()

    def release(self, url: str, *, failed: bool = False) -> None:
        """Drop the held slot, recording no duration.

        `failed` counts a rejected submission — a real event worth seeing —
        but never touches total_busy_s, because no generation ran.
        """
        st = self.stats.get(url)
        if st is None:
            return
        st.inflight = max(st.inflight - 1, 0)
        st.reserved = max(st.reserved - 1, 0)
        if failed:
            st.failed += 1

    def record_inline(self, url: str, began: float, *, ok: bool = True) -> None:
        """A request the replica finished inside the HTTP call.

        That IS a complete job, just a short one, so it is counted and timed
        as one. Converts the held slot rather than adding to it.
        """
        st = self.stats.get(url)
        if st is None:
            return
        st.inflight = max(st.inflight - 1, 0)
        st.reserved = max(st.reserved - 1, 0)
        st.started += 1
        st.total_busy_s += max(time.monotonic() - began, 0.0)
        if ok:
            st.completed += 1
        else:
            st.failed += 1

    # -- job lifecycle -----------------------------------------------------
    # A media request is not one HTTP call. `POST /v1/videos` returns a job id
    # in milliseconds and the replica then works for many minutes. Timing the
    # HTTP call therefore records ~0s of work and, far worse, drops inflight
    # back to zero immediately — so the next request can be routed onto a
    # replica that is still busy. Occupancy has to span the JOB.
    def begin_job(self, job_id: str, url: str, collection: str = "videos") -> None:
        if not job_id or job_id in self._jobs:
            return
        self.remember(job_id, url)
        st = self.stats.get(url)
        if st is None:
            # Backend vanished between accept and here (detach now guards the
            # reservation window, so this is belt-and-braces). Ownership is
            # remembered so polls still route; occupancy has nothing to hold.
            return
        self._jobs[job_id] = (url, time.monotonic())
        # Which API family minted it — the reconciler probes the job's OWN
        # collection; hardcoding /v1/videos would make any other modality's
        # dead jobs unclosable.
        self._job_collections[job_id] = collection
        st.inflight += 1
        st.started += 1

    def end_job(self, job_id: str, *, ok: bool = True) -> bool:
        """Close a job's occupancy. True only if it was actually open.

        The return is the dedupe signal for terminal accounting: several
        paths (client poll, SSE watcher, reconciler, strikes) all observe
        the same terminal status, and whoever closes it FIRST owns the
        recent-ring entry — the rest must not re-record it, which double
        counted every job and stripped its metadata (popped on first note).
        """
        entry = self._jobs.pop(job_id, None)
        self._job_collections.pop(job_id, None)
        if entry is None:
            return False
        url, began = entry
        st = self.stats.get(url)
        if st is None:
            return True
        st.inflight = max(st.inflight - 1, 0)
        st.total_busy_s += max(time.monotonic() - began, 0.0)
        if ok:
            st.completed += 1
        else:
            st.failed += 1
        return True

    def active_jobs(self) -> int:
        return len(self._jobs)

    def open_jobs(self) -> list[tuple[str, str, float, str]]:
        """(job_id, backend, began, collection) for every slot-holding job.

        The reconciler's worklist: truth about completion lives on the
        replica, and the balancer must ask on its own clock. Relying on
        clients to poll made them load-bearing for OUR occupancy — a dead
        client left finished jobs blocking slots, which blocked the queue,
        which idled the GPUs while admitted work waited.
        """
        return [
            (j, u, t, self._job_collections.get(j, "videos")) for j, (u, t) in self._jobs.items()
        ]

    # A job closes when a status poll reads a terminal state. Nothing forces a
    # client to poll, so a disconnect leaves the job open forever: inflight
    # never returns to zero, routing cannot tell an idle replica from a busy
    # one, and mean_job_s inflates without bound — which is the figure a
    # billing layer would read. Observed live: four replicas stuck at
    # inflight=1 against an empty queue.
    STALE_MULTIPLE = 3.0  # of that replica's mean job time
    STALE_FLOOR_S = 1800.0  # when it has no history to compare against

    def stale_jobs(self, *, now: float | None = None) -> list[tuple[str, str]]:
        """(job_id, backend) for jobs open far longer than that replica takes.

        Deliberately generous: closing a job that is genuinely still running
        would free a replica for work it cannot start, which is worse than
        counting a finished one for a while longer.
        """
        current = time.monotonic() if now is None else now
        out = []
        for job_id, (url, began) in self._jobs.items():
            st = self.stats.get(url)
            mean = st.mean_job_s if st else 0.0
            limit = mean * self.STALE_MULTIPLE if mean > 0 else self.STALE_FLOOR_S
            if current - began > limit:
                out.append((job_id, url))
        return out

    def queue_position(self, job_id: str) -> int:
        """How many jobs are ahead of this one on ITS replica.

        0 means running now. A caller waiting ten minutes deserves to know
        whether it is generating or sitting behind someone else's clip, and
        that is not derivable from the job status, which says "queued" from
        submission until completion either way.
        """
        entry = self._jobs.get(job_id)
        if entry is None:
            return 0
        url, began = entry
        return sum(
            1 for jid, (u, t) in self._jobs.items() if u == url and t < began and jid != job_id
        )

    def eta_hint_s(self, job_id: str) -> float | None:
        """Rough completion estimate from this replica's own history.

        Only meaningful once a replica has finished something; before that we
        say nothing rather than invent a number.
        """
        entry = self._jobs.get(job_id)
        if entry is None:
            return None
        url, began = entry
        mean = self.stats[url].mean_job_s if url in self.stats else 0.0
        if mean <= 0:
            return None
        ahead = self.queue_position(job_id)
        elapsed = time.monotonic() - began
        return max(mean * (ahead + 1) - elapsed, 0.0)

    # -- scheduling signal --------------------------------------------------
    # For a caller that runs its OWN queue. Such a scheduler does not want
    # diagnostics, it wants one number — how many requests it may send right
    # now — plus enough about what is running to plan the next minute. It is
    # answered entirely from memory so it can be polled once a second.
    def free_slots(self) -> int:
        """Replicas that could start work this instant.

        One job per replica: these engines process a single request at a time
        (dynamic batching needs a T2I/T2V gate this model does not satisfy),
        so free replicas IS the number of requests that will not queue.
        Draining replicas are excluded — idle or not, they take nothing.
        """
        return sum(1 for s in self.stats.values() if s.accepting and not s.inflight)

    def fleet_mean_job_s(self) -> float:
        """Mean job time across every replica that has finished something.

        Replicas are identical copies of one model on identical GPUs, so a
        neighbour's history is a sound estimator for a replica that has none
        of its own. Without this, work routed to a fresh replica reports no
        ETA at all — and after a restart that is EVERY replica, so a caller
        would be told "unknown" for the first full generation on each one.
        """
        means = [s.mean_job_s for s in self.stats.values() if s.mean_job_s]
        return sum(means) / len(means) if means else 0.0

    def running_jobs(self) -> list[dict]:
        """What is in flight, with how long each has left."""
        now = time.monotonic()
        fleet_mean = self.fleet_mean_job_s()
        out = []
        for job_id, (url, began) in self._jobs.items():
            st = self.stats.get(url)
            mean = (st.mean_job_s if st else 0.0) or fleet_mean
            elapsed = now - began
            out.append(
                {
                    "job_id": job_id,
                    "backend": url,
                    "elapsed_s": round(elapsed, 1),
                    # None rather than a guess until this fleet has finished
                    # something to compare against.
                    "eta_s": round(max(mean - elapsed, 0.0), 1) if mean > 0 else None,
                }
            )
        return sorted(out, key=lambda j: j["elapsed_s"], reverse=True)

    def capacity(self) -> dict:
        """Everything a external scheduler needs, in one cheap read."""
        live = [s for s in self.stats.values() if s.healthy]
        means = [s.mean_job_s for s in live if s.mean_job_s]
        return {
            "replicas": len(self.stats),
            "healthy": len(live),
            # What a queue actually drains into. Surge pricing divides queue
            # depth by this — dividing by len(stats) counted dead and
            # draining replicas and underpriced congestion exactly when
            # capacity was scarce.
            "accepting_replicas": sum(1 for s in live if s.accepting),
            "draining": sum(1 for s in live if s.draining),
            "busy": sum(1 for s in live if s.inflight),
            "free": self.free_slots(),
            "mean_job_s": round(sum(means) / len(means), 1) if means else None,
            "running": self.running_jobs(),
        }

    def mark(self, url: str, *, healthy: bool) -> None:
        if url in self.stats:
            self.stats[url].healthy = healthy

    # -- job ownership -----------------------------------------------------
    def remember(self, job_id: str, url: str) -> None:
        """Bind a job to its replica, LRU-bounded, sparing open jobs.

        `move_to_end` is the OrderedDict operation for "this is the most
        recently used" — the pop/reinsert dance said the same thing by
        hand. Eviction still skips jobs that are currently open: an open
        job's owner is load-bearing routing, and dropping it sent the next
        poll through the owner probe.
        """
        if not job_id:
            return
        # An OPEN job's binding is authoritative and must never be
        # overwritten by a different replica. Engine ids collide across
        # replicas (they are minted per replica), so a stale sighting of the
        # same number elsewhere used to silently re-point a live job at the
        # replica holding an unrelated old artifact.
        held = self._jobs.get(job_id)
        if held is not None and held[0] != url:
            return
        if job_id in self._owner:
            self._owner[job_id] = url
            self._owner.move_to_end(job_id)
            return
        if len(self._owner) >= self.max_jobs:
            victim = next((old for old in self._owner if old not in self._jobs), None)
            if victim is not None:
                self._owner.pop(victim, None)
        self._owner[job_id] = url

    def owners_snapshot(self) -> dict[str, str]:
        """The whole owner map, for persistence across restarts."""
        return dict(self._owner)

    def owner(self, job_id: str) -> str | None:
        """Which replica holds this job, or None if we must go looking."""
        return self._owner.get(job_id)

    def forget(self, job_id: str) -> None:
        self._owner.pop(job_id, None)

    def snapshot(self) -> dict:
        return {
            "backends": [
                {
                    "url": s.url,
                    "healthy": s.healthy,
                    "draining": s.draining,
                    "inflight": s.inflight,
                    "started": s.started,
                    "completed": s.completed,
                    "failed": s.failed,
                    "mean_job_s": round(s.mean_job_s, 1),
                }
                for s in self.stats.values()
            ],
            "tracked_jobs": len(self._owner),
            "active_jobs": len(self._jobs),
            "reaped_jobs": self.reaped,
        }
