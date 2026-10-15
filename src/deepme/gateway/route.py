"""Routing: which fleet serves a request, decided by model AND task.

One model's capabilities can be split across separate deployments. MiniMax-H3
ships two checkpoints — FL2VA serves `t2va` and `fl2va`, Ref2VA serves
`ref2va` — so "which fleet" is not answerable from the model name alone.

Without this the service layer would have to know that ref2va lives on a
different port, which is exactly the coupling that makes a capability change
a client change. Here a caller names a model and a task; where that runs is
an operational detail behind one address.

Pure logic: no HTTP, no engine. Routing decisions and their failure messages
are the contract the internal-customer-API layer builds against, so they are
tested directly.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Route:
    """One deployed fleet and what it can serve."""

    model: str  # the served name a caller asks for
    profile: str  # which deployment backs it
    tasks: frozenset[str]  # tasks this fleet accepts
    url: str  # its balancer

    def serves(self, task: str) -> bool:
        return task in self.tasks


class RoutingError(Exception):
    """A request that cannot be served, with an actionable reason.

    Deliberately distinct from a transport failure: this means the request
    was understood and refused, so a service layer should surface it to the
    caller rather than retry it.
    """


@dataclass
class RoutingTable:
    """Where each (model, task) is served, plus which fleet owns each job."""

    routes: list[Route] = field(default_factory=list)
    # A job id only means something to the fleet that minted it, and now the
    # id travels through two hops, so the gateway keeps its own mapping.
    _owner: OrderedDict[str, str] = field(default_factory=OrderedDict)
    max_jobs: int = 50_000

    def models(self) -> list[str]:
        return sorted({r.model for r in self.routes})

    def tasks_for(self, model: str) -> set[str]:
        return {t for r in self.routes if r.model == model for t in r.tasks}

    def resolve(self, model: str, task: str | None) -> Route:
        """The fleet for this request, or a refusal that says what to do.

        A missing task is only ambiguous when several fleets back the model;
        with one it is obvious and demanding the field would be pedantry.
        """
        candidates = [r for r in self.routes if r.model == model]
        if not candidates:
            raise RoutingError(
                f"no deployment serves model {model!r} — available: "
                f"{self.models() or 'none (no fleet is running)'}"
            )
        if task is None:
            if len(candidates) == 1:
                return candidates[0]
            raise RoutingError(
                f"model {model!r} is served by several fleets, so a task is "
                f"required — one of {sorted(self.tasks_for(model))}"
            )
        for route in candidates:
            if route.serves(task):
                return route
        known = self.tasks_for(model)
        if task in _ALL_TASKS and task not in known:
            raise RoutingError(
                f"task {task!r} needs a deployment that is not running "
                f"(model {model!r} currently serves {sorted(known)}). Deploy a "
                f"fleet that includes it, e.g. `deepme fleet apply production`."
            )
        raise RoutingError(f"unknown task {task!r} for model {model!r} — serves {sorted(known)}")

    # -- job ownership -----------------------------------------------------
    def adopt_owners(self, owners: dict[str, str]) -> None:
        """Carry another table's job-owner map into this one.

        The reload path swaps the whole table while jobs are in flight;
        preserving their ownership is THIS class's invariant, not something
        callers should assemble from our privates.
        """
        for job_id, url in owners.items():
            self.remember(job_id, url)

    def owners_snapshot(self) -> dict[str, str]:
        return dict(self._owner)

    def remember(self, job_id: str, url: str) -> None:
        """Bind a job to its fleet, LRU-bounded.

        Re-remembering REFRESHES recency. Plain insertion-order eviction
        aged out ids purely by when they were first seen, so a job being
        actively polled could lose its mapping under churn — and the next
        poll then fell back to the cross-fleet owner probe, the path that
        served one customer another customer's video.
        """
        if not job_id:
            return
        if job_id in self._owner:
            self._owner.move_to_end(job_id)
            self._owner[job_id] = url
            return
        while len(self._owner) >= self.max_jobs:
            self._owner.popitem(last=False)
        self._owner[job_id] = url

    def owner(self, job_id: str) -> str | None:
        """The fleet holding this job — a HIT refreshes recency.

        `remember` refreshes on write, but reads did not, so an id that was
        polled every 5 seconds still aged out purely by insertion order and
        fell into the owner probe.
        """
        url = self._owner.get(job_id)
        if url is not None:
            self._owner.move_to_end(job_id)
        return url

    def forget(self, job_id: str) -> None:
        self._owner.pop(job_id, None)

    def snapshot(self) -> dict:
        return {
            "routes": [
                {"model": r.model, "profile": r.profile, "tasks": sorted(r.tasks), "url": r.url}
                for r in self.routes
            ],
            "tracked_jobs": len(self._owner),
        }


# Tasks the platform knows about, so an undeployed one can be distinguished
# from a misspelt one — the first is an operations problem, the second a
# client bug, and they need different messages.
_ALL_TASKS = frozenset({"t2va", "fl2va", "ref2va"})


def build(entries: list[dict]) -> RoutingTable:
    """Build a table from deployed-fleet descriptions."""
    routes = [
        Route(
            model=e["model"],
            profile=e.get("profile", e["model"]),
            tasks=frozenset(e.get("tasks", ())),
            url=e["url"],
        )
        for e in entries
    ]
    return RoutingTable(routes=routes)
