# Improvement plan

Known work, prioritized. Each item states the DEFECT (what is wrong today),
the IMPACT (what it costs, concretely), and the CHANGE (smallest fix that
closes it). Ordered by expected cost of leaving it alone, not by effort.

Status values: `open`, `in progress`, `done`, `deferred (reason)`.

Items marked **[deploy]** change code that runs on the server and therefore
land in a production recycle window, not immediately.

---

## P0 — can lose money, data, or serve wrong content

### P0.1 Control-plane routes are unauthenticated **[deploy]**

- **Defect.** `API_KEY` is only ever _sent_ (`_auth` forwards or injects it);
  no route validates an inbound credential. `Depends`/`HTTPBearer` appear
  nowhere in `src/`. Gateway and balancers bind `0.0.0.0`.
- **Impact.** Anything that can reach :8080 can `POST /gateway/reload` and
  point every model at an attacker's URL — prompts and results included —
  or `DELETE /balancer/backends` to tear the fleet down. On :9100 the same
  applies to drain/detach.
- **Change.** One `_require_key` dependency applied to `/balancer/*` and
  `/gateway/*` routes (NOT to `/v1/*`, which forwards caller credentials).
  Operator commands already hold the key via `config.api_key()`; verify
  every `balancerctl`/`gatewayctl` call sends it before enabling.
- **Risk.** Breaks operator commands if any path omits the header. Needs a
  full call-site sweep first.
- Status: open

### P0.2 Ghost jobs from ambiguous submissions are undetectable **[deploy]**

- **Defect.** A read timeout during the pump's submission POST is now a
  terminal rejection (correct — a blind requeue duplicated work), but if the
  engine _did_ accept, that job runs with no ticket, no occupancy, no owner
  and no meta. Nothing ever reconciles it.
- **Impact.** Silent GPU spend; the job appears only as unexplained
  occupancy after the next restart's recovery pass.
- **Change.** Either an idempotency key echoed by the engine, or a
  post-timeout reconcile probe: list the replica's jobs and adopt any whose
  submission fingerprint matches the ticket. Prefer the probe (no engine
  change required).
- Status: open

### P0.3 `proxycore`: balancer and gateway have diverged copies **[deploy]**

- **Defect.** `_find_owner`, `_status_path`/inline equivalent, `_auth`,
  `_job_id`, `_safe_headers`, `_content_headers` and `_COLLECTIONS` exist
  twice. They have already drifted: the balancer filters unhealthy replicas
  out of the owner probe (so a transient error hides the only claimant and
  returns 404); the gateway logs nothing on ambiguity.
- **Impact.** Adding a modality (images/meshes) requires editing two
  tuples; missing one makes those jobs unroutable after a restart with no
  error. Ownership-probe behavior differs between hops during incidents.
- **Change.** New `src/deepme/proxycore.py` added to BOTH deploy push lists
  (`balancerctl` and `gatewayctl`), imported flat as `from proxycore import
...` exactly like `balance.py` is today. Move the seven shared pieces;
  parameterize the owner probe with a candidate list and an `on_claim` hook.
- Status: open

---

## P1 — degrades correctness or hides failure

### P1.1 Connection pooling in the client poll loop

- **Defect.** Every client HTTP call is a module-level `httpx.get`/`post`,
  each building and discarding a connection pool.
- **Impact.** A 2000-item run polling every 5s per in-flight item opens
  sockets by the tens of thousands against one host; the resulting
  intermittent `ConnectError` is swallowed as transient and surfaces only
  as unexplained stall verdicts.
- **Change.** One thread-safe `httpx.Client` per endpoint. Requires
  replacing the test suite's function-level `httpx.get/post` monkeypatching
  with `MockTransport` seams first — do that in the same change or the
  fakes stop intercepting.
- Status: open

### P1.2 Balancer module-global state has no owner **[deploy]**

- **Defect.** `_job_meta`, `_recent`, `_job_strikes`, `_fetched`,
  `retention_deleted`, `maintenance_errors`, `_events_ring`, `_subscribers`
  are module globals mutated from many paths, with `bal.reaped` mutated
  from `server.py` rather than by `Balancer`.
- **Impact.** Two sources of truth: `_job_meta` vs `bal._jobs` (recovery
  and owner-probe paths open occupancy without capturing meta, so `/v1/jobs`
  shows bare ids after a restart); `_job_strikes` entries leak for jobs
  closed by any other path.
- **Change.** A `FleetState` object built in the lifespan and stored on
  `app.state`, with `end_job` as the single place per-job entries are
  cleared. Mechanical but wide.
- Status: open

### P1.3 Owner map is persisted only incidentally **[deploy]**

- **Defect.** `_persist_membership` snapshots `owners`, but `bal.remember`
  (its only writer) never triggers a persist; only membership edits and a
  content fetch do.
- **Impact.** After hours without either, a crash-restart restores a stale
  owner map, so stranded polls fall through to the verified probe, which
  refuses on 0-or-2 claimants — 404s for jobs that are running fine.
- **Change.** Debounced write-through on `remember`/`forget` (they are
  frequent; a dirty flag flushed on the maintenance beat is enough).
- Status: open

### P1.4 Blocking filesystem I/O on the event loop **[deploy]**

- **Defect.** `_persist_membership` (called from a streaming response
  generator and four handlers) and `_audit_write` (a `stat` + append on
  EVERY published event) run synchronously in async context. The retention
  pass is now threaded; these are not.
- **Impact.** With a large outputs tree or a slow disk, SSE streams and
  proxied downloads stall, `/health` times out, and the gateway marks a
  healthy fleet down.
- **Change.** `anyio.to_thread.run_sync` for persistence; batch audit
  appends (queue + periodic flush) rather than one write per event.
- Status: open

### P1.5 Admission queue is a full sort per read **[deploy]**

- **Defect.** `waiting()` sorts every ticket; `next_ready()`, `position()`
  and `snapshot()` each call it. `/v1/jobs` calls `position()` per waiting
  ticket — O(n² log n) with no `await` in the loop.
- **Impact.** At the `prune(keep=5000)` bound this pins the event loop for
  seconds: heartbeats stop, `/health` times out, the fleet reads as down.
- **Change.** Sort once and `enumerate` in `/v1/jobs`; back `next_ready`
  with `heapq` keyed `(priority, seq)`.
- Status: open

### P1.6 `_reap_stale_jobs` runs unbounded sequential probes inside handlers **[deploy]**

- **Defect.** Called from every POST and from `/balancer/stats`: one `ctl`
  probe per open job, sequentially, 30s ceiling each, no overall deadline.
- **Impact.** One wedged replica adds minutes to a client's submission
  before admission is even reached; the client times out and retries,
  starting another pass.
- **Change.** `asyncio.gather` the probes; drop the call from `proxy` (the
  maintenance beat already covers it) or gate it behind a max-age so one
  pass serves all callers.
- Status: open

---

## P2 — duplication and drift that will cause the next incident

### P2.1 Six capacity/jobs JSON parsers

- **Defect.** `media._owner_unit`, `media._server_view`, `cli` capacity
  render, `cli` top render, `cli` batch-status cross-check, and the gateway
  aggregator each parse the same shape independently. Only `_server_view`
  handles the unreachable-fleet trap correctly (two dangerous consumers
  were fixed this session; the parsers remain separate).
- **Change.** One `capacity_view(spec) -> FleetView` consumed by all
  client-side callers. Pure client code, no deploy constraint.
- Status: open

### P2.2 Three JSON-lines writers with three timestamp formats

- **Defect.** `batch.journal`, balancer `_audit_write`, `runs.record` —
  different field names (`ts` ISO / `ts` epoch / `time` spaced), only one
  rotates, only one tolerates torn lines on read. `batch.journal` is the
  incident-reconstruction file and it is the unbounded one.
- **Change.** `config.append_jsonl(path, record, *, max_bytes=None)` with
  one timestamp convention; the balancer keeps a `proxycore` twin.
- Status: open

### P2.3 Dead code that misdescribes the live system

- **Defect.** `balance.stale_jobs`/`STALE_MULTIPLE`/`STALE_FLOOR_S` and
  `Balancer.start`/`finish` are superseded by the strike/reconcile design
  but still read as the rule. `sharedstore.py`'s entire acquisition half
  (`buffer_key`, `acquire`, `publish`, `.ready`/`.lock` protocol) describes
  a scheme the DEPLOYED hook does not implement (different key inputs,
  opposite decision on rank, no ready marker).
- **Impact.** The next reader trusts the wrong contract. `sharedstore`'s
  live half (`usage`/`orphans`/`clear`) is maintained against a layout the
  other file defines.
- **Change.** Delete the dead halves; state the on-disk contract in exactly
  one docstring. NOTE: `test_shared_conventions` asserts the stale-job
  constants match the client's stall heuristic — that test encodes the
  false "client and server agree" claim and should be retired with them.
- Status: open

### P2.4 Three near-identical service controllers

- **Defect.** `guardctl`, `balancerctl`, `gatewayctl` each implement
  `setup()`, unit templates, `stop/status/api_base` with an identical
  4-line sync script. `guardctl.status` drops stderr; `gatewayctl` has no
  `logs()` at all, so the gateway is the one tier with no log command.
- **Change.** `unitctl.py` with `sync_project/install_unit/stop/status/
logs`; the three keep only their templates and `api_base`. Add
  `deepme gateway logs`.
- Status: open

### P2.5 Status vocabularies compared as literals in the CLI

- **Defect.** `batch.py` defines PENDING/RUNNING/DONE/FAILED/SKIPPED;
  `story.py` defines a smaller second set; `cli.py` compares raw strings in
  ~9 places.
- **Impact.** A rename leaves `--fetch-existing`'s adoption filter matching
  nothing while reporting success.
- **Change.** Import and compare the symbols.
- Status: open

### P2.6 Replica base-URL built by hand in seven places

- **Defect.** `f"http://127.0.0.1:{port}"` in `balancerctl`, `gatewayctl`,
  `switch` (×4), `cli`. `cli` uses it as a lookup KEY against balancer
  membership strings.
- **Impact.** Any normalization change makes `fleet status` report every
  replica NOT ATTACHED.
- **Change.** `balancerctl.replica_url(port)`.
- Status: open

---

## P3 — smaller correctness and hygiene

- **`economics.record_duration_batch` substitutes the first `points = []`
  in the file**, unanchored — `[measured.steps]` also has one. Anchor to the
  section and write atomically.
- **`config` freezes `SSH_HOST`/`SSH_USER`/`REMOTE_HOME` at import** while
  `cluster_*` reads live; a `config init` in the same process leaves later
  commands pointed at the old server. Make them accessors.
- **`remote.run` converts `TimeoutExpired` to `Result(124)`** — a
  client-side timeout is indistinguishable from a remote exit 124. Also
  `stream=True` inherits stdin.
- **`runs.tail` silently skips corrupt lines** — count and report them.
- **`_fetched` records leak** for artifacts deleted mid-fetch; prune against
  ids observed on disk, not against the current `doomed` list.
- **`_find_owner` mutates occupancy from a lookup**, re-opening struck-out
  jobs so `reaped` counts the same job repeatedly.
- **Gateway `/gateway/reload` rebinds a module global `table`** while
  in-flight handlers hold the old object; mutate in place instead.
- **Story seam trim uses `-ss` before `-i` with `-c copy`**, which seeks to
  the nearest keyframe — the documented one-frame trim is not what happens.

---

## P0 (round 2) — found by the four-lane audit, still open

### P0.4 Ghost jobs have no reconciler, and the client resubmits into them **[deploy]**

- **Defect.** Two paths create a running job the balancer does not know
  about: the direct POST path (client HTTP timeout at 60s while the engine
  accepts and generates) and the ambiguous-submit rejection. Nothing goes
  engine→balancer at runtime — `_reap_stale_jobs` only iterates jobs the
  balancer already tracks, and `_recover_occupancy` runs once, at boot.
- **Impact.** Silent GPU spend; the slot reads free so `pick()` double-books
  a one-job-at-a-time engine; the result expires unfetched.
- **Change.** The right primitive is an **idempotency key**: the balancer
  already mints a stable salted `ticket_id` before the POST — stamp it as
  `request_id` and have the engine return the existing job for a repeat.
  That closes the ghost, the requeue, and the cancelled-mid-POST case at
  once. If the engine cannot change: a per-backend orphan sweep (list
  `/v1/videos`, adopt non-terminal ids absent from `_jobs`) on the
  maintenance beat, which downgrades "leaks until restart" to "≤10s".
- Status: open

### P0.5 A wedged-but-answering engine holds its slot forever **[deploy]**

- **Defect.** Closure is terminal-only; strikes fire only on 404/unreachable.
  An engine that keeps answering `{"status":"queued"}` for work it will
  never finish is invisible to both. `balance.stale_jobs()` exists for
  exactly this and is dead code.
- **Impact.** The client gives up at its own 1800s floor and fails the item;
  the server's slot never returns. Permanent per-wedge capacity loss.
- **Change.** Wire `stale_jobs()` into the maintenance beat: close with
  `ok=False` and DELETE to the replica. Then delete the duplicate constants.
- Status: open

### P0.6 Reserved slots leak on non-POST proxy paths **[deploy]**

- **Defect.** The fallback reserves a slot for ANY method, then forwards on
  the unbounded-read client. A non-job-scoped GET against a wedged replica
  parks forever holding the reservation.
- **Impact.** `free_slots()` permanently drops; since `_all_busy()` now
  delegates to it, the queue stops draining on an idle fleet, permanently,
  with nothing counted.
- **Change.** Reserve only for POST; bound the non-POST leg.
- Status: open

## P1 (round 2)

- **Ambiguous owner should be 409, not 404** — the resource exists _twice_;
  404 tells a client "safe to resubmit", which is the opposite of the truth.
- **`reaped` double-counts** — both increments sit outside the `end_job`
  dedupe guard, and `_job_strikes` entries leak for jobs closed elsewhere.
- **`CTL_TIMEOUT` (30s) × serial probes swamps `DRAIN_S` (10s)** — one
  wedged replica turns a "10-second" beat into minutes, so `JOB_LOST_STRIKES`
  becomes 5×(beat duration) and the whole invariant stalls together.
  Drop to 10s and `gather` the probes.
- **`journal()` is atomic only by accident** — one un-truncated field pushes
  a record past the 8 KiB buffer and worker lines interleave in the exact
  file an incident is reconstructed from. Use one `os.write` or a lock.
- **`/v1/models` omits `created`** — third-party OpenAI clients reject the
  model list.
- **SSE has no `[DONE]` terminator** — OpenAI-style consumers read the close
  as truncation. Either emit it or state plainly that this is not an
  OpenAI-compatible stream.
- **Unreachable capacity entries drop six documented fields** — fill the
  zeroed shape (keeping `running` ABSENT, which the client's trust bit
  depends on) and document that one exception.
- **`cancel()` reports a REJECTED ticket as `cancelled: true`** — a billing
  layer records a cancellation for work that may be running.
- **Internal queue states leak as wire `status`** (`submitting`, `rejected`).

## P2 (round 2) — observability

- **`audit.jsonl` is written and never read.** No command, no doc. Add
  `deepme events --audit` reading it (and `.1`) over `remote.run`.
- **No CLI reads `/metrics`**, so `maintenance_errors`/`retention_errors` —
  added precisely because a dead loop looks like an idle one — are invisible.
  Promote them into `/balancer/stats` and print in `balancer status`.
- **Missing counters**, ranked: submission rejections by cause; per-replica
  error rate; ticket cancellations/`by_state`; owner-probe refusals; content
  fetch count/failures; queue wait time; disk free bytes.
- **No publish on fleet LOSS** — drain/undrain/detach and every
  `mark(healthy=False)` are silent, so a fleet that shrank overnight is
  invisible while joins and heals are recorded.
- **Commands report success without verifying** — `gateway start`,
  `balancer start`, `guard start`, `serve stop/restart`, `store clear`,
  `patches apply`, `download start`, `artifacts sweep --delete` (which
  claims reclaimed GB it never measured). Each has a cheap verifier already
  in its module.
- **`batch status` reports a dead ticket as "queued"** — it cross-checks the
  running set but never the queue.
- **`batch run` exits 0 with failed items**; so does `showcase run`.
- **No command reads `journal.jsonl`** — add `batch journal <name>`.
- **`capacity`/`discover` stack-trace where `top` explains itself.**
- **`--json` missing** where a night-watch script needs it most:
  `batch status`, `batch list`, `balancer status`.

## Completed (tracked in the task list)

Closed this round, each with a regression test and the full gate green:

- **Auth on the control plane** (P0.1) — `/balancer/*` and `/gateway/reload`
  - `/gateway/routes` now require the platform key; `/v1/*` and
    `/gateway/discover` stay open by design (they forward caller credentials
    / are the documented service-discovery call). Every operator call site was
    swept first — 16 of them, none of which sent a key — so enforcement could
    not lock the operator out. **[deploy]**
- **Slot reservation scoped to POST** (P0.6) — a proxied read no longer
  holds a replica slot forever against a wedged replica, which used to stop
  the queue draining permanently with nothing counted. **[deploy]**
- **Maintenance beat bounded and parallel** (P1.6) — reconcile probes run
  concurrently, `CTL_TIMEOUT` 30s→10s, and the redundant in-request reap is
  gone from the POST path. **[deploy]**
- **Reap counting and strike leak** (P1) — `reaped` counted inside the
  `end_job` dedupe; `_job_strikes` pruned against open jobs; strike closes
  report a documented status with a `reason` field instead of minting a
  fifth status. **[deploy]**
- **Admission queue listing** (P1.5) — one sorted pass enumerated instead of
  a re-sort per ticket (was O(n² log n) with no await, pinning the loop).
  **[deploy]**
- **Operator health surfaced** — `maintenance.errors_total`,
  `retention_errors_total`, `audit_enabled`, `boot` and queue depth now in
  `/balancer/stats` and printed by `balancer status`. **[deploy]**
- **Transient OOM retry** — a CUDA OOM is load-dependent, not a property of
  the request; bounded retries with backoff and a fresh identity, never
  applied to request errors.
- **Journal atomicity** — one `os.write` to an `O_APPEND` fd, so the
  forensic trail cannot interleave regardless of record size.
- **Exit codes and ticket checks** — `batch run`/`showcase run` exit
  non-zero on failures; `batch status` cross-checks the QUEUE and reports
  `TICKET GONE`; `--json` on `batch status`/`balancer status`; `capacity`
  explains itself instead of stack-tracing.

## Verification debt

- The fast suite is the only gate; several fixes this session are asserted
  by construction rather than by test. Specifically untested: the executor
  flood fix under a real `SystemExit`, `flock` contention between a runner
  and a repair tool, `.part` cleanup on a failed download, `aiter_bytes`
  streaming behavior, and the gateway's `x-deepme-job-id` learning path.
- No test exercises the free-threaded contention the audits flagged
  (shared caches, journal appends). A stress test with N threads would
  cover a class the unit suite structurally cannot.
