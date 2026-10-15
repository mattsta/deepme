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
import hashlib
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
# Deliberately not generated: a duplicate request whose output exists (or
# will exist) elsewhere. Skipped items are never picked up by a run — the
# state file says WHY, so there is no invisible backlog.
SKIPPED = "skipped"

# Load-dependent engine failures (a CUDA OOM while the fleet is busy) are
# worth retrying; request errors are not. Bounded so a genuinely broken
# item cannot consume the night, and backed off so the fleet drains before
# the next attempt rather than racing the same contention.
TRANSIENT_RETRIES = 2
TRANSIENT_BACKOFF_S = 30.0


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
    # WHICH REQUEST that job id was recorded for. An id alone is not proof
    # of ownership: the client accepts whatever id a poll body reports and
    # persists it, so one bad mapping anywhere on the platform permanently
    # binds a foreign job to this item — and adoption SKIPS generation and
    # downloads that job's bytes. Binding the id to the prompt it was
    # created for makes a mismatch detectable locally, with no server help.
    job_prompt: str | None = None
    # Server-reported peak GPU memory. Recorded per item because activations
    # scale with the REQUEST (duration x aspect ratio), so the capability
    # envelope can only be found by measuring the largest legal request —
    # everything else, residency included, is budgeted from what it leaves.
    peak_mb: float | None = None
    error: Any = None

    @property
    def name(self) -> str:
        return f"{self.index:04d}"

    @property
    def request_fingerprint(self) -> str:
        """What this item is ASKING for, hashed. Identity of the request,
        not of the job."""

        payload = json.dumps(
            [self.prompt, self.seed, self.seconds, self.steps, self.params], sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def owns(self, job_id: str | None) -> bool:
        """May this item adopt that job?

        A fingerprint that CONTRADICTS this request means the id belongs to
        different work — an edited prompts file, a harvested id from
        someone else's job, a slot reused across runs. Adopting it skips
        generation and delivers that other request's video here, so it is
        refused.

        A MISSING fingerprint is a record written before ids were bound to
        requests. It is unprovable, not wrong, and every item on an
        existing installation looks like this — disowning them would
        regenerate an entire archive to fix a subset. They are adopted and
        flagged by `batch audit` instead; everything written from now on
        carries its fingerprint and is checkable.
        """
        if not job_id:
            return False
        return self.job_prompt is None or self.job_prompt == self.request_fingerprint


def skip_existing(batch: Batch) -> int:
    """Refuse to generate what this platform already has. Returns how many.

    THE LOOP THIS ENDS: generation is deterministic, so the same request
    always yields the same bytes. Duplicate prompts across prompt files
    therefore produced byte-identical clips; `dedupe --apply` deleted the
    copies and reset those items to PENDING; the next run regenerated the
    identical bytes; the next dedupe deleted them again. Forever, burning a
    GPU slot every lap. Deleting the output could never fix a duplicate
    REQUEST — only refusing to make it again can.

    Checked against every batch on disk INCLUDING THIS ONE. The duplicates
    are across files (the same prompt in four backlogs) and also within a
    single one — 64 pending items in one batch were twins of clips that
    batch had already made — so only the item ITSELF is exempt, identified
    by (batch, index). An item whose request already has a file is marked
    SKIPPED (the status that exists for exactly this) with its source
    recorded, so the state file says why and there is no invisible backlog.

    Deliberately not a content hash: this runs BEFORE generating, so the
    only thing knowable is the request. That is enough — identical request
    means identical bytes on this engine, which is the fact the whole
    duplicate problem rests on.
    """
    from . import reconcile

    index = reconcile.existing_by_spec()
    skipped = 0
    for item in batch.items:
        if item.status not in (PENDING, FAILED):
            continue
        found = index.get(reconcile.census_key(item))
        if not found:
            continue
        owner, owner_index, path = found
        if owner == batch.name and owner_index == item.index:
            continue  # its own output: a re-run must still be able to redo it
        where = f"{owner}[{owner_index:04d}]"
        item.status = SKIPPED
        item.error = f"identical request already generated by {where} — {path}"
        item.job_id, item.job_prompt = None, None
        journal(batch, "skipped", item=item.index, reason="duplicate-request", source=where)
        skipped += 1
    return skipped


def _is_delivery_failure(error: Any) -> bool:
    """Did the WORK succeed and only its delivery fail?

    Only those are safe to re-adopt: the artifact exists server-side and
    re-fetching it is the whole point. A generation failure re-adopted just
    re-reads the failure — and once the platform's id map has drifted, it
    downloads someone else's video instead.
    """
    text = (error.get("message", "") if isinstance(error, dict) else str(error or "")).lower()
    return any(
        marker in text
        for marker in ("content fetch failed", "download", "connection", "timed out", "timeout")
    )


class PromptsChanged(RuntimeError):
    """The prompts file no longer matches the batch state it names."""


def merge_items(existing: list[Item], from_file: list[Item]) -> list[Item]:
    """Reconcile a resumed batch against its prompts file.

    A resume used to IGNORE the file entirely and re-run whatever the state
    held. That silently:
      * re-ran old prompts when a file was edited (the edits never applied);
      * never ran prompts appended to a file;
      * and — worst — kept each item's recorded job_id, so `adopt=` skipped
        generation and DOWNLOADED THE OLD JOB'S VIDEO as the new prompt's
        output. New prompts in, old content out.

    Items are matched by INDEX, which is what the runner has always used, so
    this is the only place that can notice the index no longer means the
    same request. Appending is safe and silent; a CHANGED prompt at an
    existing index is not — that item's recorded job belongs to different
    work, so it is refused loudly rather than reinterpreted.
    """
    drift = [
        i.index
        for i, was in zip(from_file, existing, strict=False)
        if (i.prompt, i.seed, i.seconds, i.steps, i.params)
        != (was.prompt, was.seed, was.seconds, was.steps, was.params)
    ]
    if drift:
        shown = ", ".join(f"{d:04d}" for d in drift[:10])
        raise PromptsChanged(
            f"{len(drift)} item(s) differ from the saved batch at the same "
            f"index ({shown}{'…' if len(drift) > 10 else ''}). Those indexes "
            f"already carry job ids for the OLD requests, so resuming would "
            f"deliver the old videos for the new prompts. Use --restart to "
            f"regenerate from the file, or --name to keep them separate."
        )
    if len(from_file) < len(existing):
        raise PromptsChanged(
            f"the file has {len(from_file)} item(s) but the saved batch has "
            f"{len(existing)} — resuming would silently drop the tail. Use "
            f"--restart, or --name for a new batch."
        )
    # A pure APPEND is the common, safe case: keep every existing item (and
    # its identity), take the new tail as fresh work.
    return list(existing) + list(from_file[len(existing) :])


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


# NO AUTOMATIC SEEDS, deliberately. Seedless generation is deterministic on
# this engine — the same prompt with no seed gives a byte-identical clip on
# any replica (verified live: two runs, two replicas, one md5). That makes
# "same prompt" an EXACT predictor of "same bytes", which is what lets the
# census refuse duplicate work BEFORE the GPU time is spent rather than
# detecting it afterwards by hashing what you already paid for. Assigning a
# per-item seed would make every repeated prompt look like new work and
# demote that prediction to after-the-fact detection.
#
# Want a different take? Change the prompt. A different request is honestly a
# different request, and it needs no second axis of identity for the dedupe
# layer to track. An explicit per-item `seed` in the JSONL still works for
# anyone who really means it.


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
        config.write_text_atomic(
            self.state_path,
            json.dumps(
                {
                    "name": self.name,
                    "profile": self.profile,
                    "out_dir": self.out_dir,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "items": [asdict(i) for i in self.items],
                },
                indent=2,
            ),
        )

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
        out = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0, SKIPPED: 0}
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


def journal(batch: Batch, event: str, **fields: Any) -> None:
    """Append one line to the batch's forensic trail. Never overwritten.

    state.json is CURRENT state: a resubmit overwrites item.job_id, which is
    exactly the history an incident needs ("item 60 held job X, then job
    Y"). This file is the other half of the truth — every identity an item
    ever held, and the MD5 of every file at the moment it was downloaded —
    so duplicate/misattribution questions are answerable from logs instead
    of from content archaeology.

    Appended with ONE `os.write` to an O_APPEND descriptor. Buffered text
    writes are atomic only while a record stays under the 8KiB buffer —
    true today because every unbounded field is truncated, but nothing
    enforces it, and a single long prompt or engine error body would split
    the write and let two workers interleave fragments into exactly the
    file a duplicate/misattribution incident is reconstructed from. One
    append syscall is atomic regardless of size.
    """
    import os

    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    path = batch.state_path.parent / "journal.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(entry) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


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


class BatchBusy(RuntimeError):
    """Someone else holds this batch. Not an exit — the caller decides."""


@contextlib.contextmanager
def claim(batch: Batch):
    """Exclusive access to one batch's state, for the duration of a block.

    ANY writer of state.json takes this — the runner and the repair tools
    alike. Two writers share the file last-writer-wins: a runner holds the
    batch in memory for a whole overnight run and saves after every item,
    so a repair that edits the file underneath has its resets silently
    reinstated by the runner's next save — and resume then re-adopts the
    very job the repair existed to escape.

    `fcntl.flock`, not a pid file: the kernel drops the lock when the
    holder dies, however it dies. The pid-file version had to guess
    liveness from `kill(pid, 0)` and got it wrong in both directions — a
    stale lock after `kill -9` blocked legitimate resumes, and a recycled
    pid blocked them with a lie. The pid is still written INSIDE the file,
    for humans reading it; nothing depends on it.
    """
    import fcntl
    import os

    lock_path = batch.state_path.parent / "runner.pid"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.seek(0)
            holder = handle.read().strip() or "another process"
            raise BatchBusy(
                f"batch {batch.name!r} is already held by pid {holder} — "
                f"watch it with `deepme batch status {batch.name}`, or stop "
                f"that process"
            ) from None
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        # Closing releases the lock; the file may linger and that is fine —
        # an unlocked runner.pid claims nothing.
        handle.close()


def run(
    spec: RunSpec,
    batch: Batch,
    *,
    concurrency: int,
    # OFF by default, matching the CLI and generate_video: a blanket
    # deadline wasted hours on dead items and hid the absence of real
    # detection. A programmatic caller silently inherited 2 hours.
    timeout: float = 0,
    on_start=None,
    on_tick=None,
    on_done=None,
    fetch_only: bool = False,
) -> Batch:
    """Work the backlog, keeping `concurrency` generations in flight.

    A bounded pool IS the queue: a worker takes the next pending item only
    when it has finished the last one, so the backlog stays here where it can
    still be changed, and a replica is never handed work it cannot start.

    `fetch_only` collects without submitting: only items that already own a
    server job are adopted and downloaded; pending items are left untouched
    and NOTHING new reaches the fleet — a lost job is marked failed, never
    resubmitted. This is how a batch is wound down: the in-flight tail is
    harvested while the backlog stays parked for a later real run.
    """
    out = Path(batch.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with claim(batch):
        return _run_claimed(
            spec,
            batch,
            out=out,
            concurrency=concurrency,
            timeout=timeout,
            on_start=on_start,
            on_tick=on_tick,
            on_done=on_done,
            fetch_only=fetch_only,
        )


def _run_claimed(
    spec: RunSpec,
    batch: Batch,
    *,
    out: Path,
    concurrency: int,
    timeout: float,
    on_start,
    on_tick,
    on_done,
    fetch_only: bool,
) -> Batch:
    """`run` with the batch already claimed — the lock spans the whole run,
    including the reconciliation pass and the pool teardown."""
    lock = threading.Lock()
    # THE process-wide cancel signal, not a private one: the interrupt
    # protocol (interrupts.py) sets it on the first ^C, so workers begin
    # stopping the moment the signal lands — not after the main thread
    # unwinds to our except block.
    from . import interrupts
    from .digests import DigestCache

    cancel_event = interrupts.CANCEL
    # Hashing at download time primes the shared digest cache, so a later
    # dedupe scan re-hashes nothing it already saw.
    digest_cache = DigestCache.load()

    def work(item: Item) -> Item:
        started = time.monotonic()
        # BEFORE anything reaches the fleet. `cancel_futures` cannot cancel
        # a future that has already started, and the cancel event was only
        # ever checked inside the poll loop — so a worker picking up an
        # item mid-cancel submitted a brand-new billable job and only THEN
        # noticed it was cancelled.
        if cancel_event.is_set():
            raise InterruptedError(item.job_id or item.name)
        # Record the start, not just the finish. An item takes ten-plus
        # minutes, so a state file that only changes on completion reports an
        # entire busy fleet as "pending" — `batch status` from another shell
        # (the whole point of an unattended run) then looks identical to a
        # batch that is stuck.
        with lock:
            item.status = RUNNING
            item.started_at = time.time()
            batch.save()
            # The PROMPT goes in the record. Items are matched by index, so
            # the journal is the only place that can later prove which
            # request a job id was actually submitted for — the drift
            # detector reads exactly this field, and without it the check
            # silently passed on every batch.
            journal(
                batch,
                "started",
                item=item.index,
                adopt=item.job_id,
                prompt=item.prompt,
                seed=item.seed,
            )
        if on_start:
            on_start(item)

        def record_id(job_id: str) -> None:
            # To DISK, immediately: this id is what lets a future runner
            # re-attach instead of duplicating (see Item.job_id). The
            # journal keeps EVERY id this item ever held — state.json only
            # remembers the last, which is exactly the history an incident
            # needs.
            with lock:
                item.job_id = job_id
                # Bind the id to the request it belongs to, in the same
                # write. Without this an id is just a string that a later
                # run will adopt on faith.
                item.job_prompt = item.request_fingerprint
                batch.save()
                journal(
                    batch,
                    "job-id",
                    item=item.index,
                    job_id=job_id,
                    fingerprint=item.job_prompt,
                )

        try:
            for attempt in range(TRANSIENT_RETRIES + 1):
                result = media.generate_video(
                    spec,
                    item.prompt,
                    seconds=item.seconds,
                    seed=item.seed,
                    steps=item.steps,
                    params=item.params,
                    out_dir=out,
                    timeout=timeout,
                    # Without this a batch prints nothing for the ten-plus
                    # minutes an item takes, which is indistinguishable from
                    # being hung.
                    on_tick=_relay(on_tick, item),
                    on_submit=record_id,
                    # A restart re-attaches to the job this item already
                    # OWNS — verified, not assumed. An id recorded for a
                    # different request is not adopted; it regenerates.
                    adopt=item.job_id if item.owns(item.job_id) else None,
                    cancel=cancel_event,
                    resubmit=not fetch_only,
                )
                # A load-dependent engine failure (a CUDA OOM while the
                # fleet is busy) is not a property of the request — the
                # same body succeeds on a less-loaded replica. Recording it
                # as permanent threw away work the platform could finish.
                # Bounded, backed off so the fleet drains first, and never
                # applied to request errors, which fail identically forever.
                if result.ok or not media.is_transient(result.error):
                    break
                if fetch_only or attempt >= TRANSIENT_RETRIES or cancel_event.is_set():
                    break
                delay = TRANSIENT_BACKOFF_S * (attempt + 1)
                with lock:
                    journal(
                        batch,
                        "retry",
                        item=item.index,
                        job_id=item.job_id,
                        attempt=attempt + 1,
                        after_s=delay,
                        error=str(result.error)[:200],
                    )
                    # A fresh attempt needs a FRESH identity: the recorded
                    # job failed, and adopting it would just re-read that
                    # failure instead of generating.
                    item.job_id = None
                    batch.save()
                if cancel_event.wait(delay):
                    raise InterruptedError(item.name)
            if result.ok and result.path:
                # The slot name carries the BATCH and the JOB, not just the
                # index. `NNNN.mp4` is batch-agnostic, and `Path.replace` is
                # a silent overwrite — so two batches pointed at one --out
                # destroyed each other's clips while BOTH state files
                # recorded `file: "0007.mp4", status: done`, and a rerun
                # into a dirty directory inherited the previous run's video
                # in the new run's slot.
                stem = f"{item.name}-{(result.media_id or 'nojob')[:8]}"
                named = result.path.with_name(f"{stem}{result.path.suffix}")
                result.path.replace(named)
                # Content identity at the moment of delivery: which bytes,
                # from which job, into which slot — the line that makes
                # duplicate hunts a log query instead of archaeology.
                # Hashed BEFORE the lock: it reads a multi-hundred-MB file.
                md5 = None
                with contextlib.suppress(OSError):
                    md5 = digest_cache.digest(named)
                # UNDER THE LOCK, as one unit. `item.file, item.status = ...`
                # is two separate stores, and another worker holding the
                # lock could serialise the item between them — persisting
                # `{"status": "done", "file": null}`. Resume's repair rule
                # (`status == DONE and item.file and not exists()`) then
                # short-circuits on the null, so that item is never
                # re-fetched and never regenerated: a silent hole in a
                # deliverable set the operator believes complete.
                with lock:
                    item.file, item.status = named.name, DONE
                    item.took_s = result.total_s
                    item.peak_mb = result.peak_memory_mb
                    journal(
                        batch,
                        "done",
                        item=item.index,
                        job_id=item.job_id,
                        file=named.name,
                        md5=md5,
                        took_s=round(result.total_s, 1),
                    )
            else:
                with lock:
                    item.status, item.error = FAILED, result.error
                    item.took_s = result.total_s
                    item.peak_mb = result.peak_memory_mb
                    journal(
                        batch,
                        "failed",
                        item=item.index,
                        job_id=item.job_id,
                        error=str(result.error)[:200],
                    )
        except InterruptedError:
            # Ctrl-C: the JOB is still running server-side. The item stays
            # RUNNING with its job_id on disk, so the next run re-attaches
            # to it — nothing is lost and nothing is duplicated.
            with lock:
                batch.save()
                journal(batch, "interrupted", item=item.index, job_id=item.job_id)
            return item
        except Exception as exc:
            # One bad item must not end the night. This used to have to name
            # SystemExit too, because the client signalled ordinary failures
            # with a BaseException that sailed through `except Exception`,
            # aborted pool.map from a worker thread, and left the remaining
            # futures generating headless. The client now raises MediaError,
            # so a plain Exception clause is both correct and sufficient.
            with lock:
                item.status, item.error = FAILED, str(exc)
                item.took_s = time.monotonic() - started
                journal(batch, "failed", item=item.index, job_id=item.job_id, error=str(exc)[:200])
        with lock:
            batch.save()
        # OUTSIDE the lock: on_done is caller code that renders to a
        # terminal, and a blocked write (piped to a stalled mount, a
        # flow-controlled tty) would hold every other worker out of
        # `record_id` — the one path whose whole purpose is getting job ids
        # to disk before anything can go wrong.
        if on_done:
            on_done(item, batch)
        return item

    # Completeness reconciliation before anything runs:
    #  * a DELIVERY failure with a job id is ADOPTABLE — a download cut
    #    mid-stream (a gateway restart does this routinely) marked billed
    #    work failed forever though the finished file sat on the server;
    #  * a GENERATION failure is NOT: re-adopting it just re-reads the same
    #    failure, or worse, downloads whatever that id now resolves to. This
    #    promotion used to fire on every FAILED item with an id, which is how
    #    old content kept being re-delivered run after run.
    #  * DONE whose local file vanished is a hole in the deliverable set the
    #    operator believes complete — re-fetch if the id survives, else redo.
    for item in batch.items:
        # A job id that does not belong to THIS request is dropped before
        # anything can act on it. This is the single check that stops a
        # foreign job's video being delivered as this item's output.
        if item.job_id and not item.owns(item.job_id):
            journal(
                batch,
                "disowned",
                item=item.index,
                job_id=item.job_id,
                reason="recorded for a different request",
            )
            item.job_id, item.job_prompt = None, None
            if item.status in (RUNNING, FAILED):
                item.status, item.error = PENDING, None
        if item.status == FAILED and item.job_id and _is_delivery_failure(item.error):
            item.status, item.error = RUNNING, None
        elif item.status == FAILED and not fetch_only:
            # A RUN THAT MAY GENERATE RETRIES ITS FAILURES. `todo()` returns
            # only PENDING/RUNNING, so a FAILED item was inert: nothing ever
            # picked it up again, while the CLI told the operator to "retry
            # them" by re-running — advice that quietly did nothing. Items
            # failed by a collect-only pass ("no longer exists on the server
            # and resubmission is disabled") are the common case and are
            # entirely generatable; they were simply work a --fetch-existing
            # run is forbidden to create.
            #
            # PENDING, and the id is DROPPED: this regenerates from the
            # prompt rather than adopting. Promoting a generation failure to
            # RUNNING would re-adopt the old id and download whatever it now
            # resolves to — which is how old content kept coming back run
            # after run. Regeneration cannot deliver someone else's video.
            item.status, item.error = PENDING, None
            item.job_id, item.job_prompt = None, None
        elif item.status == DONE and item.file and not (out / item.file).exists():
            item.status = RUNNING if item.job_id else PENDING
    todo = batch.todo()
    for item in todo:
        if item.status != RUNNING or not item.job_id:
            item.status = PENDING
    if not fetch_only:
        skip_existing(batch)
        todo = batch.todo()
    batch.save()
    if fetch_only:
        # Only items whose work is ALREADY ON A GPU. A `q_` id is a QUEUE
        # TICKET — work that has not started — so adopting one makes the
        # collector sit waiting while the balancer's pump hands it to a
        # replica. That is commissioning a new generation, which is exactly
        # what this mode exists to avoid: it reported "submitting nothing"
        # and then showed a job climbing through denoise. Tickets stay
        # untouched for a later real run.
        todo = [
            i for i in todo if i.status == RUNNING and i.job_id and not i.job_id.startswith("q_")
        ]
        # The cap exists to limit NEW work per replica; adopted jobs are
        # already running server-side, so poll them all at once.
        concurrency = len(todo) or 1
    # FROZEN POOL — the worker count is fixed for the whole (possibly
    # overnight) run, so a `fleet switch` mid-batch breaks the one-item-per-
    # replica invariant: scale-down queues the surplus INSIDE engines,
    # scale-up idles the new replicas. To make it adaptive: replace the
    # executor with a hand-rolled loop that, before dispatching each item,
    # re-reads `free` from /v1/capacity and sizes in-flight work to
    # min(capacity, remaining) — the admission queue makes over-submission
    # safe meanwhile, so this is an efficiency gap, not a correctness one.
    pool = futures.ThreadPoolExecutor(max_workers=max(concurrency, 1))
    try:
        # BaseException, not KeyboardInterrupt: the interrupt protocol
        # escalates with SystemExit, and `map` has ALREADY submitted every
        # item as a future. Catching only KeyboardInterrupt left the whole
        # backlog queued in a live executor, which the interpreter's own
        # atexit hook then drained — thousands of real generations
        # submitted during "teardown", with the display already gone.
        list(pool.map(work, todo))
    except BaseException:
        # wait=False on the ESCALATION path: workers are already stopping
        # (they watch CANCEL, and every wait including the download is now
        # cancellable), and blocking the unwind on the join is what made a
        # second ^C do nothing until a third. The `finally` below still
        # joins for the normal exit.
        cancel_event.set()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        # Unconditional: every exit path shuts the pool down, including the
        # ones no `except` clause names.
        pool.shutdown(wait=True, cancel_futures=True)
        with contextlib.suppress(OSError):
            digest_cache.save()
    return batch
