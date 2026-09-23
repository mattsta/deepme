"""Deployment ledger: durable, observable artifacts of every operation.

Every state-changing action (serve start/stop, switch) and every measurement
(bench) appends a JSON record to .deepme/runs/runs.jsonl and writes a full
detail file next to it. This is the system of record for "what ran, when,
configured how, with what outcome" — inspectable via `deepme runs` without
touching the server. Live checks (`status`) show the present; the ledger
shows history.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any

from . import config
from .profiles import RunSpec

RUNS_DIR = config.STATE_DIR / "runs"
LEDGER = RUNS_DIR / "runs.jsonl"


def record(
    action: str,
    spec: RunSpec | None = None,
    *,
    outcome: str,
    detail: dict[str, Any] | None = None,
) -> str:
    """Append one ledger entry; returns the run id."""
    import secrets

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    # Random suffix so two ops in the same second never collide/overwrite.
    run_id = f"{ts}-{action}" + (f"-{spec.profile}" if spec else "") + f"-{secrets.token_hex(2)}"
    entry = {
        "id": run_id,
        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "action": action,
        "profile": spec.profile if spec else None,
        "outcome": outcome,
        "spec": asdict(spec) if spec else None,
        "detail": detail or {},
    }
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(entry, indent=2))
    summary = {k: entry[k] for k in ("id", "time", "action", "profile", "outcome")}
    with LEDGER.open("a") as f:
        f.write(json.dumps(summary) + "\n")
    return run_id


def tail(n: int = 20) -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().strip().splitlines()[-n:]:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # tolerate a torn/corrupt line (crash mid-append)
    return out


def load(run_id: str) -> dict:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        raise SystemExit(f"unknown run id {run_id!r} — see: uv run deepme runs")
    return json.loads(path.read_text())
