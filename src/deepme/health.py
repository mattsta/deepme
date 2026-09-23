"""Continuous health observation for deployed models.

`status` answers "how are things right now" when an operator asks. This module
answers the question nobody is present for: did a service go down at 3am, and
for how long. It samples every deployment, detects TRANSITIONS (up -> down,
down -> up, and restart-count jumps that reveal a crash loop nobody saw), and
writes each one to the runs ledger so the history survives the terminal.

Restart counts matter as much as reachability: a unit that crash-loops every
40 seconds can answer a health probe between crashes and look fine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import profiles, remote, runs, serve
from .profiles import RunSpec


@dataclass(frozen=True)
class UnitHealth:
    profile: str
    unit: str
    active: str  # systemd ActiveState
    api_ok: bool
    detail: str  # health probe result, or why it failed
    restarts: int  # NRestarts — jumps mean a crash loop

    @property
    def state(self) -> str:
        """The comparable state. Restart count is deliberately part of it so a
        crash loop registers as a change even while probes keep succeeding."""
        return f"{self.active}/{'up' if self.api_ok else 'down'}/{self.restarts}"

    @property
    def healthy(self) -> bool:
        return self.active == "active" and self.api_ok


def _unit_props(units: list[str]) -> dict[str, tuple[str, int]]:
    """ActiveState + NRestarts per unit, in ONE remote round trip."""
    if not units:
        return {}
    names = " ".join(f"{u}.service" for u in units)
    r = remote.run(f"systemctl show {names} -p Id -p ActiveState -p NRestarts", check=False)
    out: dict[str, tuple[str, int]] = {}
    unit = state = ""
    restarts = 0
    for line in r.stdout.splitlines():
        key, _, val = line.partition("=")
        if key == "Id":
            unit = val.removesuffix(".service")
        elif key == "ActiveState":
            state = val
        elif key == "NRestarts":
            restarts = int(val) if val.isdigit() else 0
            if unit:
                out[unit] = (state, restarts)
    return out


def snapshot(specs: list[RunSpec] | None = None) -> list[UnitHealth]:
    """Sample every profile that has a unit on the server."""
    if specs is None:
        specs = []
        for name in profiles.list_profiles():
            try:
                specs.append(profiles.load(name))
            except (Exception, SystemExit):
                # A gated repo or cold metadata cache must not blind the watch
                # for every OTHER model.
                continue
    props = _unit_props([s.unit for s in specs])
    out = []
    for spec in specs:
        active, restarts = props.get(spec.unit, ("absent", 0))
        # Probing a unit that is not running wastes a timeout per cycle.
        ok, detail = serve.health(spec) if active == "active" else (False, active)
        out.append(UnitHealth(spec.profile, spec.unit, active, ok, detail, restarts))
    return out


def diff(before: dict[str, str], now: list[UnitHealth]) -> list[tuple[UnitHealth, str]]:
    """Transitions since the last sample, as (health, previous state)."""
    return [h for h in ((h, before.get(h.profile, "")) for h in now) if h[1] and h[0].state != h[1]]


def watch(
    interval_s: float = 60.0,
    *,
    cycles: int = 0,
    on_event=None,
) -> list[UnitHealth]:
    """Sample until `cycles` is reached (0 = forever), recording transitions.

    The first sample establishes the baseline and is reported but not recorded
    as a transition — a service that was already down did not just fail.
    """
    seen: dict[str, str] = {}
    latest: list[UnitHealth] = []
    n = 0
    while cycles == 0 or n < cycles:
        latest = snapshot()
        changes = diff(seen, latest)
        for h, was in changes:
            runs.record(
                "health",
                None,
                outcome="healthy" if h.healthy else "unhealthy",
                detail={
                    "profile": h.profile,
                    "unit": h.unit,
                    "was": was,
                    "now": h.state,
                    "active": h.active,
                    "api_ok": h.api_ok,
                    "detail": h.detail,
                    "restarts": h.restarts,
                },
            )
        if on_event:
            on_event(latest, changes, n == 0)
        seen = {h.profile: h.state for h in latest}
        n += 1
        if cycles == 0 or n < cycles:
            time.sleep(interval_s)
    return latest
