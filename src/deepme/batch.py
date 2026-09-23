"""Run a backlog of generations to completion, unattended.

Submitting a whole backlog at once looks simpler and is worse: the balancer
assigns each job to a replica the moment it arrives, and job ids are minted
per replica, so those assignments are permanent. A slow replica keeps its
share, a crashed one loses it, and nothing can be reordered or cancelled once
it is inside the engines.

So the queue lives HERE, and only as many items are in flight as there are
replicas. Work rebalances naturally (a fast replica simply gets the next
item sooner), a crash costs one item rather than a quarter of the backlog,
and the pending list stays editable.

State is written after every item, so an overnight run that dies at 3am
resumes where it stopped instead of regenerating six hours of work.
"""

from __future__ import annotations

import concurrent.futures as futures
import contextlib
import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config, media
from .profiles import RunSpec

BATCH_DIR = config.STATE_DIR / "batches"

PENDING, RUNNING, DONE, FAILED = "pending", "running", "done", "failed"


@dataclass
class Item:
    """One generation in the backlog."""

    index: int
    prompt: str
    params: dict[str, Any] = field(default_factory=dict)
    seconds: int | None = None
    seed: int | None = None
    steps: int | None = None
    status: str = PENDING
    file: str | None = None
    took_s: float = 0.0
    # Wall-clock start, so a status check from ANOTHER shell can say how long
    # an in-flight item has been going. The running process knows this; the
    # state file is the only way anyone else finds out.
    started_at: float | None = None
    # The server-side id, written the moment the server assigns it. THE
    # reconnection key: a job outlives the runner (it runs on a replica; the
    # runner is a poller), and without the id on disk a restarted runner
    # could only resubmit — duplicating work that was still generating, and
    # orphaning results that had finished. With it, resume ADOPTS.
    job_id: str | None = None
    # Server-reported peak GPU memory. Recorded per item because activations
    # scale with the REQUEST (duration x aspect ratio), so the capability
    # envelope can only be found by measuring the largest legal request —
    # everything else, residency included, is budgeted from what it leaves.
    peak_mb: float | None = None
    error: Any = None

    @property
    def name(self) -> str:
        return f"{self.index:04d}"


def load_items(path: Path) -> list[Item]:
    """Read a backlog.

    Two formats, because both are natural to produce: one prompt per line for
    something hand-written or piped from another tool, or JSONL when items
    need their own parameters.
    """
    text = path.read_text().strip()
    if not text:
        raise SystemExit(f"{path} is empty")
    items: list[Item] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            raw = json.loads(line)
            items.append(
                Item(
                    index=len(items),
                    prompt=raw["prompt"],
                    params=dict(raw.get("params", {})),
                    seconds=raw.get("seconds"),
                    seed=raw.get("seed"),
                    steps=raw.get("steps"),
                )
            )
        else:
            items.append(Item(index=len(items), prompt=line))
    if not items:
        raise SystemExit(f"{path} has no prompts (blank lines and # are skipped)")
    return items


@dataclass
class Batch:
    name: str
    profile: str
    items: list[Item]
    out_dir: str

    @property
    def state_path(self) -> Path:
        return BATCH_DIR / self.name / "state.json"

    def save(self) -> None:
        """Atomically: write-then-rename.

        This file is the ONLY copy of an overnight run's truth and is
        rewritten thousands of times; a kill -9 landing inside a plain
        write_text truncated it, and resume died on a JSONDecodeError with
        everything — done markers, job ids — gone.
        """
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "profile": self.profile,
                    "out_dir": self.out_dir,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "items": [asdict(i) for i in self.items],
                },
                indent=2,
            )
        )
        tmp.replace(self.state_path)

    @classmethod
    def load(cls, name: str) -> Batch | None:
        path = BATCH_DIR / name / "state.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return cls(
            name=raw["name"],
            profile=raw["profile"],
            out_dir=raw["out_dir"],
            items=[Item(**i) for i in raw["items"]],
        )

    def counts(self) -> dict[str, int]:
        out = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0}
        for item in self.items:
            out[item.status] = out.get(item.status, 0) + 1
        return out

    def todo(self) -> list[Item]:
        """Anything not finished. RUNNING counts as todo: if the process died
        mid-item, that item never completed and must be retried."""
        return [i for i in self.items if i.status in (PENDING, RUNNING)]


def estimate(pending: int, replicas: int, mean_job_s: float) -> dict[str, float]:
    """How long a backlog will take, and how much fits in a window.

    Answers the planning question directly rather than making the operator do
    the arithmetic: with N replicas each taking mean_job_s, throughput is
    replicas/mean_job_s.
    """
    replicas = max(replicas, 1)
    if mean_job_s <= 0:
        return {"hours": 0.0, "per_hour": 0.0, "overnight_10h": 0.0}
    per_hour = 3600.0 / mean_job_s * replicas
    return {
        "hours": pending / per_hour if per_hour else 0.0,
        "per_hour": per_hour,
        "overnight_10h": per_hour * 10,
    }


def _relay(on_tick, item: Item):
    """Bind an item to the per-generation tick callback.

    Several generations run at once, so each stream of progress has to carry
    which item it belongs to or the display cannot tell them apart.
    """
    if on_tick is None:
        return None

    def relay(tick) -> None:
        on_tick(item, tick)

    return relay


def _claim_batch(batch: Batch) -> None:
    """One runner per batch, enforced.

    Two concurrent runners share state.json last-writer-wins: both take the
    same todo list, every item generates twice (double the GPU spend), and
    half the outcomes are silently overwritten. The pid file makes the
    second runner fail loudly instead; a stale lock from a dead process is
    reclaimed automatically.
    """
    import os

    lock_path = batch.state_path.parent / "runner.pid"
    if lock_path.exists():
        alive = False
        with contextlib.suppress(ValueError, OSError):
            pid = int(lock_path.read_text().strip())
            try:
                os.kill(pid, 0)
                alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                # The pid EXISTS under another user — that is a live process,
                # not a stale lock. Treating EPERM as dead stole live locks.
                alive = True
        if alive:
            raise SystemExit(
                f"batch {batch.name!r} is already being run by pid {pid} — "
                f"watch it with `deepme batch status {batch.name}`, stop that "
                f"process, or delete {lock_path} if it is a recycled pid"
            )
        lock_path.unlink(missing_ok=True)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL: two runners racing this claim must not both pass an
    # exists-check — exactly one create succeeds.
    fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))


def run(
    spec: RunSpec,
    batch: Batch,
    *,
    concurrency: int,
    timeout: float = 7200,
    on_start=None,
    on_tick=None,
    on_done=None,
) -> Batch:
    """Work the backlog, keeping `concurrency` generations in flight.

    A bounded pool IS the queue: a worker takes the next pending item only
    when it has finished the last one, so the backlog stays here where it can
    still be changed, and a replica is never handed work it cannot start.
    """
    out = Path(batch.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _claim_batch(batch)
    lock = threading.Lock()
    # THE process-wide cancel signal, not a private one: the interrupt
    # protocol (interrupts.py) sets it on the first ^C, so workers begin
    # stopping the moment the signal lands — not after the main thread
    # unwinds to our except block.
    from . import interrupts

    cancel_event = interrupts.CANCEL

    def work(item: Item) -> Item:
        started = time.monotonic()
        # Record the start, not just the finish. An item takes ten-plus
        # minutes, so a state file that only changes on completion reports an
        # entire busy fleet as "pending" — `batch status` from another shell
        # (the whole point of an unattended run) then looks identical to a
        # batch that is stuck.
        with lock:
            item.status = RUNNING
            item.started_at = time.time()
            batch.save()
        if on_start:
            on_start(item)

        def record_id(job_id: str) -> None:
            # To DISK, immediately: this id is what lets a future runner
            # re-attach instead of duplicating (see Item.job_id).
            with lock:
                item.job_id = job_id
                batch.save()

        try:
            result = media.generate_video(
                spec,
                item.prompt,
                seconds=item.seconds,
                seed=item.seed,
                steps=item.steps,
                params=item.params,
                out_dir=out,
                timeout=timeout,
                # Without this a batch prints nothing for the ten-plus minutes
                # an item takes, which is indistinguishable from being hung.
                on_tick=_relay(on_tick, item),
                on_submit=record_id,
                # A restart re-attaches to the job this item already owns.
                adopt=item.job_id,
                cancel=cancel_event,
            )
            if result.ok and result.path:
                named = result.path.with_name(f"{item.name}{result.path.suffix}")
                result.path.replace(named)
                item.file, item.status = named.name, DONE
            else:
                item.status, item.error = FAILED, result.error
            item.took_s = result.total_s
            item.peak_mb = result.peak_memory_mb
        except InterruptedError:
            # Ctrl-C: the JOB is still running server-side. The item stays
            # RUNNING with its job_id on disk, so the next run re-attaches
            # to it — nothing is lost and nothing is duplicated.
            with lock:
                batch.save()
            return item
        except (Exception, SystemExit) as exc:
            # One bad item must not end the night. SystemExit EXPLICITLY:
            # the client raises it for rejected submissions and exhausted
            # resubmissions, and as a BaseException it sailed through a bare
            # `except Exception`, aborted pool.map in the main thread, and
            # left the executor's remaining futures generating headless for
            # hours with the display torn down.
            item.status, item.error = FAILED, str(exc)
            item.took_s = time.monotonic() - started
        with lock:
            batch.save()
            if on_done:
                on_done(item, batch)
        return item

    # Completeness reconciliation before anything runs:
    #  * FAILED with a job id is ADOPTABLE — a download cut mid-stream (a
    #    gateway restart does this routinely) marked billed work failed
    #    forever, though the finished file sat on the server;
    #  * DONE whose local file vanished is a hole in the deliverable set the
    #    operator believes complete — re-fetch if the id survives, else redo.
    for item in batch.items:
        if item.status == FAILED and item.job_id:
            item.status, item.error = RUNNING, None
        elif item.status == DONE and item.file and not (out / item.file).exists():
            item.status = RUNNING if item.job_id else PENDING
    todo = batch.todo()
    for item in todo:
        if item.status != RUNNING or not item.job_id:
            item.status = PENDING
    batch.save()
    # FROZEN POOL — the worker count is fixed for the whole (possibly
    # overnight) run, so a `fleet switch` mid-batch breaks the one-item-per-
    # replica invariant: scale-down queues the surplus INSIDE engines,
    # scale-up idles the new replicas. To make it adaptive: replace the
    # executor with a hand-rolled loop that, before dispatching each item,
    # re-reads `free` from /v1/capacity and sizes in-flight work to
    # min(capacity, remaining) — the admission queue makes over-submission
    # safe meanwhile, so this is an efficiency gap, not a correctness one.
    pool = futures.ThreadPoolExecutor(max_workers=max(concurrency, 1))
    lock_path = batch.state_path.parent / "runner.pid"
    try:
        list(pool.map(work, todo))
    except KeyboardInterrupt:
        # Clean stop, not a stack trace: signal every worker (their poll
        # loops notice within one cycle), let them save state, and hand the
        # caller an accurate picture. Deliberately NOT the executor context
        # manager — its __exit__ blocks until generations finish, which is
        # tens of minutes of an unresponsive ^C.
        cancel_event.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    finally:
        # The lock dies with the run, both exits. Leaving it made every
        # resume's fate rest on pid liveness checks — and a recycled pid
        # blocked a legitimate resume with a lie.
        with contextlib.suppress(OSError):
            lock_path.unlink()
    pool.shutdown(wait=True)
    return batch
