# Internal infrastructure API

The contract between **your service layer** and **this inference platform**.

Customers never reach these endpoints. Your service layer owns identity,
pricing, quotas and presentation; this platform owns GPUs, placement,
scheduling and generation. This document is the seam.

Everything here is derived from live state — nothing below is a mapping you
have to maintain by hand.

---

## 1. Start here: service discovery

**`GET http://<host>:8080/gateway/discover`**

The one call your service layer makes to learn what exists. Do not hardcode
ports, model names, or which checkpoint serves which task — all of it moves
when the fleet mix changes.

```json
{
  "host": "…",
  "models": [
    {
      "model": "minimax-h3",
      "tasks": ["fl2va", "ref2va", "t2va"],
      "profiles": ["h3-video", "h3-ref"],
      "replicas": 4,
      "balancers": ["http://…:9100", "http://…:9200"],
      "limits": {"min_steps": 2},
      "generate_defaults": {"task": "t2va", "target": {...}}
    }
  ],
  "endpoints": [
    {"name": "h3-video/replica-0", "kind": "replica",
     "url": "http://…:8100", "declared": true, "running": true,
     "state": "running"},
    {"name": "h3-video/replica-3", "kind": "replica",
     "url": "http://…:8130", "declared": true, "running": false,
     "state": "declared"}
  ],
  "summary": {"models": 1, "running_endpoints": 6, "declared_endpoints": 17}
}
```

**`declared` is not `running`.** Declared means a port is reserved in the
registry; running means a unit currently claims it. Treating configuration as
reality is the expensive mistake here, so the distinction is preserved.

**`tasks` is what is deployed right now**, not what the model could do. Switch
the fleet from `production` (3+1) to `max-base` (4+0) and `ref2va` disappears
from the catalog rather than being advertised and then failing.

Operators see the same thing with `deepme discover`.

**`GET /v1/models`** is the smaller, OpenAI-shaped version, for clients that
only need names and tasks:

```json
{
  "object": "list",
  "data": [
    {
      "id": "minimax-h3",
      "object": "model",
      "owned_by": "deepme",
      "tasks": ["fl2va", "ref2va", "t2va"]
    }
  ]
}
```

It takes no parameters — discovery has to answer when the caller knows nothing
yet. Note that partitions merge: one model name, with `tasks` as the thing you
route on.

---

## 2. Submitting work

**`POST http://<host>:8080/v1/videos`**

```json
{
  "model": "minimax-h3",
  "task": "t2va",
  "prompt": "a lighthouse beam sweeping through heavy fog",
  "conditions": [],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5
  },
  "seed": 1101,
  "priority": 5
}
```

Routing is by **model + task**. `task` may be omitted only when one fleet
backs the model; with several it is required, and the error says which are
available.

Two possible responses, and your client must handle both:

| status | meaning                                                | body                                                                                                  |
| ------ | ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------- |
| `200`  | submitted to a replica — **billable from this moment** | `{"id": "vid_…", "status": "queued", …}`                                                              |
| `202`  | held in the admission queue — **not billable yet**     | `{"id": "q_…", "status": "queued", "queue_position": 2, "estimated_wait_s": 1642, "billable": false}` |

Add `?queue=false` and the `202` becomes a `503` instead — see §4b if you run
your own queue.

**The admission rule (stable):** a submission is held (or, with
`?queue=false`, refused) whenever the fleet is full **or any ticket is
already waiting** — a freed slot always goes to the queue head, so walk-ins
cannot starve queued work. Consequence for own-queue schedulers: `free > 0`
does NOT guarantee acceptance while others queue; the 503 body's `waiting`
count is your backoff signal:

```json
{"detail": "all replicas busy or queued work is ahead of you — queueing was
            declined", "free": 1, "waiting": 3, "retry_hint_s": 118}
```

Do not mix own-queue (`?queue=false`) and platform-queue callers against one
fleet unless the own-queue side backs off on `waiting > 0`.

### Ticket ids and their lifetime (read this before billing against them)

- The id format is `q_` + an **opaque** remainder (balancer name + per-boot
  nonce + sequence). Only the `q_` prefix is stable and parseable; never
  compare ids across balancer restarts by structure.
- **Tickets are volatile.** They live in balancer memory: a balancer restart
  destroys every WAITING ticket, and the boot nonce guarantees the old ids
  are never re-served. A persistent 404 on a `q_` id means the ticket is
  lost — treat it as failed-unbilled and resubmit (the bundled client does
  this automatically, and follows the real job instead when the poll bodies
  already named it).

### The full wire-status vocabulary (stable)

What a poll or SSE stream can say about an id, and what each means:

| status                                         | where | meaning                                                                                                                                               |
| ---------------------------------------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `queued`                                       | both  | waiting, or generating (the engine cannot tell you which — use `queue_position`/capacity). A ticket mid-handover briefly shows `queue_position: null` |
| `submitted`                                    | SSE   | the billing line: `job_id` switches to the real id, `billable: true`                                                                                  |
| `running`                                      | SSE   | position 0 and generating                                                                                                                             |
| `completed` / `failed` / `cancelled` / `error` | both  | terminal. Rejected tickets report `failed` with `error{message}` and `billable: false`                                                                |
| `unknown`                                      | SSE   | the id no longer exists anywhere; the stream closes — resubmit if unfetched                                                                           |

Both carry an `id` you poll the same way, so the shape of your client does not
change with load. The `q_` prefix tells you which side of the billing line it
is on, and `billable` states it explicitly.

### Generation modes

| mode                        | `task`   | what you send                              |
| --------------------------- | -------- | ------------------------------------------ |
| text to video               | `t2va`   | `prompt`, `conditions: []`                 |
| continue from a first frame | `fl2va`  | one condition, `frame_index: 0`            |
| generate up to a last frame | `fl2va`  | one condition, `frame_index: -1`           |
| between two fixed frames    | `fl2va`  | two conditions, `frame_index` 0 and -1     |
| reference-guided            | `ref2va` | one or more `role: "reference"` conditions |

A condition:

```json
{
  "type": "image",
  "uri": "data:image/png;base64,…",
  "role": "keyframe",
  "frame_index": 0
}
```

- `type`: `image` \| `video` \| `video_audio` \| `audio`
- `role`: `keyframe` (pins an exact frame) \| `reference` (guides only)
- `uri`: `https://…`, `file://…`, or a `data:` URI. The platform cannot read
  your filesystem, so inline anything local.
- Only `frame_index` values `0`, `-1`, or both are accepted for `fl2va` —
  a keyframe in the middle is not a supported mode.

### Enforced limits

Rejected at submission with `400`, never mid-generation:

| field                     | accepted                                            |
| ------------------------- | --------------------------------------------------- |
| `target.short_edge`       | exactly `768`                                       |
| `target.aspect_ratio`     | `auto`, `21:9`, `16:9`, `4:3`, `1:1`, `3:4`, `9:16` |
| `target.duration_seconds` | `4.0`–`15.0` (fixed 24 fps)                         |
| `num_inference_steps`     | `>= min_steps` from discovery (2 for H3)            |

Fields outside `{task, prompt, conditions, target, flow_shift,
audio_flow_shift, seed, num_inference_steps}` are **discarded silently by
the engine** — notably `num_frames`, `fps` and `size`. Do not rely on them.
(`priority` is consumed by the BALANCER's queue, never forwarded — it is
real, just not an engine field.)

Measured envelope: the largest legal request (15s at 21:9) peaks at 26.7GiB of
32.6GiB VRAM, so every valid combination fits.

---

## 3. Following a job

### Streaming (preferred)

**`GET /v1/videos/{id}/events`** — server-sent events, emitted on change only.

```
data: {"job_id":"…","status":"queued","queue_position":2,"eta_s":1642}
data: {"job_id":"…","status":"running","queue_position":0,"eta_s":538}
data: {"job_id":"…","status":"completed","queue_position":0}
```

`eta_s` is `null` until a replica has finished something comparable — no
estimate is offered rather than an invented one. Reconnect with the same id;
the stream is stateless.

**Subscribing with a `q_` ticket works too, and is the point of the ticket.**
One subscription spans the whole life of the request, across the handover to a
replica, so you never have to poll for the moment a ticket becomes a job:

```
data: {"job_id":"q_00000001","status":"queued","queue_position":1,"billable":false}
data: {"job_id":"vid_…","ticket":"q_00000001","status":"submitted","billable":true}
data: {"job_id":"vid_…","status":"running","queue_position":0,"eta_s":538}
data: {"job_id":"vid_…","status":"completed","queue_position":0}
```

The `submitted` event is the billing line: `job_id` switches to the real job
and `billable` flips to `true`. Everything before it costs nothing.

### Polling

**`GET /v1/videos/{id}`** → `{"status": "queued|completed|failed", …}`

The upstream engine reports `queued` from submission until `completed` with
nothing between, which is why `queue_position` exists — it is the only way to
distinguish _generating_ from _waiting behind other work_. It appears in
ticket-poll bodies and on the SSE stream; once a real job id takes over, the
poll body is the engine's own (proxied untouched) and does not carry it —
watch `/v1/events` or `/v1/jobs` for position after handover.

`{id}` may be a `q_` ticket. While it waits you get the queued shape; once a
replica takes it, the same id transparently follows the real job — including
`/content`. So a client that submits and polls the id it was handed works
unchanged whether it got a `200` or a `202`.

### Result

**`GET /v1/videos/{id}/content`** → the MP4
(H.264 1344×768 @ 24fps + AAC stereo 32kHz).

---

## 4. Queue control

Work can only be changed while it is still **ours** — before a replica has it.

| call                               | effect                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------ |
| `GET /v1/queue`                    | what is waiting, in service order                                        |
| `GET /v1/queue` _(at the gateway)_ | the same, aggregated across fleets — see below                           |
| `DELETE /v1/queue/{ticket}`        | `200` cancelled, or `409` if already submitted                           |
| `DELETE /v1/videos/{ticket}`       | the same cancellation, for clients that only kept the id from submission |

Each fleet queues independently, so at the gateway the response is a rollup
rather than one ordered list:

```json
{
  "waiting": 4,
  "fleets": [
    {
      "profile": "h3-video",
      "tasks": ["fl2va", "t2va"],
      "waiting": 4,
      "reachable": true
    },
    {
      "profile": "h3-ref",
      "tasks": ["ref2va"],
      "waiting": 0,
      "reachable": false
    }
  ]
}
```

`reachable: false` means that fleet could not be asked — **not** that its queue
is empty, and its work is missing from `waiting`. Check it before treating the
total as authoritative.

**Priority**: `priority` on submission — `0` high, `5` normal, `9` low.
Ordering is `(priority, arrival)`, so a burst at one tier stays fair. Priority
cannot be changed after submission — see `AdmissionQueue.reprioritise` for what
exposing that would take.

**There is no cancel-in-flight, by design.** Once a request reaches a replica
it is consuming a GPU; stopping it recovers nothing and it is billable either
way. If your product offers cancellation, offer it against `q_` tickets.

---

## 4b. Running your own queue (recommended for paid tiers)

Everything above assumes we hold the backlog. If your app already has a
priority queue — paid tiers, per-user fairness, requests users add and pull
back — **keep it there** and use this platform as a pool of slots. Ordering
policy is a product concern; it does not belong behind the router, where it
cannot see who is paying.

Two pieces make that safe.

### `GET /v1/capacity` — the scheduling signal

```json
{
  "free": 2,
  "accepting": ["fl2va", "t2va"],
  "fleets": [
    {
      "model": "minimax-h3",
      "profile": "h3-video",
      "tasks": ["fl2va", "t2va"],
      "url": "http://127.0.0.1:9100",
      "replicas": 3,
      "healthy": 3,
      "accepting_replicas": 3,
      "draining": 0,
      "busy": 1,
      "free": 2,
      "queued": 0,
      "accepting": true,
      "reachable": true,
      "mean_job_s": 842.9,
      "disk": {
        "free_gb": 812.4,
        "used_pct": 57.1,
        "watermark_exceeded": false
      },
      "running": [
        {
          "job_id": "vid_…",
          "backend": "http://…:8100",
          "elapsed_s": 120.4,
          "eta_s": 722.5
        }
      ]
    },
    {
      "model": "minimax-h3",
      "profile": "h3-ref",
      "tasks": ["ref2va"],
      "replicas": 1,
      "healthy": 1,
      "busy": 1,
      "free": 0,
      "queued": 0,
      "accepting": false,
      "reachable": true,
      "mean_job_s": 1248.5,
      "running": [{ "job_id": "rid_…", "elapsed_s": 60.0, "eta_s": 1188.5 }]
    }
  ]
}
```

- **`free` is the contract**: the number of requests that will start
  immediately. One job per replica — these engines process a single request
  at a time — so `free` is healthy, non-draining replicas with nothing in
  flight (a replica finishing its last job before leaving is never `free`).
- **`reachable: false` always reports `free: 0`.** A fleet that cannot be
  asked is not a fleet with spare capacity; the safe reading is built in so a
  scheduler cannot submit into a dead fleet.
- **`eta_s` is `null` only when the whole fleet has never finished anything.**
  A replica with no history of its own borrows the fleet mean — replicas are
  identical, and otherwise every replica reports "unknown" for its first
  generation after a restart.
- **Cost**: answered from memory, no replica is contacted, fleets are polled
  concurrently. Designed for a 1 Hz loop.
- Fleet entries also carry `accepting_replicas` (the surge-depth divisor —
  replicas that can actually drain a queue), an always-present `draining`
  count, `queued`, an `accepting` boolean, `url`, and
  `disk {free_gb, used_pct, watermark_exceeded}` — the watermark (0.85,
  operator-tunable later; treat the threshold as unstable, the boolean as
  stable) means "stop selling spot and let retention catch up".
- **The shape can change under you, live.** Operators resize the fleet at
  runtime (`fleet switch`): `replicas`/`free` move, a `draining` count may
  appear (replicas finishing their last job before leaving), and a task can
  vanish from `accepting` entirely when its fleet is removed — or appear
  minutes later. Treat every poll as the current truth rather than caching
  the topology; `/v1/models` and `/v1/pricing` move with it — a removed
  task disappears from the quote entirely, while `available: false` marks a
  task whose fleet is still routed but currently unreachable.

### `?queue=false` — refuse rather than hold

```
POST /v1/videos?queue=false
```

Returns `503` with `Retry-After` instead of a `202` ticket whenever the
fleet is full **or any platform-queue ticket is waiting** (§2's admission
rule — a freed slot always serves the queue head first). **Use this if you
run your own queue.** Two queues for the same work is the failure mode:
ours would hold requests your scheduler still believes are its own to
order, cancel and bill for. Refusing keeps one authority.

```json
{
  "detail": "all replicas busy or queued work is ahead of you — queueing was declined",
  "free": 1,
  "waiting": 3,
  "retry_hint_s": 118
}
```

`retry_hint_s` is the soonest-finishing job's ETA, so a backoff can be
informed rather than fixed. `waiting` is the platform queue's depth — while
it is non-zero, `free > 0` does NOT mean your submission will be accepted.

### The loop

```js
while (true) {
  const cap = await get("/v1/capacity"); // one round-trip
  for (const fleet of cap.fleets) {
    if (!fleet.reachable) continue;
    if (fleet.queued > 0) continue; // platform queue drains first — back off
    for (let i = 0; i < fleet.free; i++) {
      const job = myQueue.popHighestPriorityFor(fleet.tasks); // YOUR policy
      if (!job) break;
      const r = await post("/v1/videos?queue=false", job.body);
      if (r.status === 503) {
        myQueue.push(job);
        break;
      } // lost the race, or queued work got ahead — the body says which
      myTracking.set(r.body.id, job);
    }
  }
  await sleep(1000);
}
```

Handle the `503` even though `free` said otherwise: between your poll and your
submit, another caller may take the slot. `free` is a snapshot, not a
reservation — there is no slot-holding API, deliberately, because a crashed
client holding reservations would idle the fleet.

Track completion however you prefer — SSE (§3) if you want push, or poll
`/v1/videos/{id}`. `running[].eta_s` in the capacity response is usually
enough to know when to look again.

---

## 4c. Pricing signal

**`GET /v1/pricing`** _(gateway)_ — what a second of output costs right now,
per task. A customer-facing pricing service polls this as its **baseline** and
adds product policy on top; nothing customer-visible needs to know how the
numbers were derived.

```json
{
  "currency": "USD",
  "unit": "per_output_second",
  "lock_at": "submitted",
  "tasks": {
    "t2va": {
      "available": true,
      "tier": "spot",
      "price_per_output_second": 0.08,
      "spot_floor": 0.08,
      "market": 0.13,
      "queued_per_replica": 0.0
    },
    "ref2va": {
      "available": true,
      "tier": "surge-x1.8",
      "price_per_output_second": 0.468,
      "spot_floor": 0.08,
      "market": 0.26,
      "queued_per_replica": 1.0
    }
  }
}
```

- **Tier is live, table is frozen.** Cost, floor, market and the surge ladder
  are derived from `deploy/economics.toml` (server $/hr × measured
  replica-seconds per task) at `gateway start`. The tier applied to them comes
  from the same queue signal `/v1/capacity` reads, at request time:
  `spot` (free slot, empty queue) → `market` (full, no queue) →
  `surge-xN` (by queue depth per replica).
- **`lock_at: "submitted"` is the billing contract.** Quotes are indicative;
  the price locks at the `submitted` event (`billable: true`), never at quote
  time. Queue depth moves while a request waits — a quote-time lock lets
  callers arbitrage the surge, and a completion-time price bills people for
  a surge they never saw.
- **`available: false` means do not sell it.** The fleet serving that task is
  unreachable; a quote would be an invitation to bill for nothing.
- Floors are per task, derived (cost × worst-case duration ratio × margin,
  rounded up) — a flat spot price under-waters the expensive cases twice
  over: $0.05 flat is below ref2va's $0.0557/s cost, and below even t2va's
  cost at 15s ($0.054/s — duration scaling is superlinear, measured). The
  response's `duration_curve` carries the measured points (`vs_linear` per
  clip length) for app layers that price by duration.
- `derived_at`/`table_hash` identify the frozen table. `deepme econ` warns
  when the gateway serves an older table than the config would derive.

Operators see the same derivation with `uv run deepme econ` (costs,
break-even utilization, fleet-mix ranking, surge ladder). Changing a price
means editing `deploy/economics.toml` and restarting the gateway — prices are
never typed into code.

---

---

## 4d. Artifact reclamation

A file is reclaimed when its owner says so, with time as the backstop —
never the other way round:

| state                   | reclaimed                              |
| ----------------------- | -------------------------------------- |
| **acked** by the client | **immediately** (`DELETE .../content`) |
| delivered, not acked    | **1 hour** from the FIRST delivery     |
| never fetched           | **6 hours** from creation              |
| job still running       | untouchable, regardless of age         |

`DELETE /v1/videos/{id}/content` is the ACK: "I have persisted these bytes,
reclaim your copy." It returns `{id, reclaimed: <files>}`, is idempotent (a
retried ack answers 200 with `reclaimed: 0`), and is **not** a cancellation —
`DELETE` on the job cancels work, `DELETE` on `/content` only frees the file.
It is never forwarded to a replica.

Deleting on DELIVERY instead — which this replaced — made every transfer
single-shot: a caller that received 200 and then failed to write lost the
work permanently, with no way to ask again. We own both ends of this
protocol, so possession is confirmed rather than assumed. The guarantee that
motivated delete-on-delivery is unchanged, because the retention clock starts
at first delivery whether or not an ack arrives: a client that dies mid-write
cannot strand a file, and nothing lingers long enough for an id to be reused.

The window starts at the first `/content` read and is **not** extended by
repeat fetches. After expiry, `GET .../content` returns 404 — re-generate if
you still need it. Staged condition uploads (`inputs/`) follow the 6-hour
rule. Enforcement runs in the balancer's maintenance loop (~1min cadence);
`/balancer/stats` reports `retention {deleted_total, tracked_fetches, ttl_s,
fetched_ttl_s, enabled}`. First-fetch times survive balancer restarts.
`deepme artifacts sweep` remains as a manual override; `deepme artifacts
audit` shows what is on disk right now.

**Design your client to download promptly and treat the local copy as the
copy.** The server is a buffer, not a library.

## 4e. Operations endpoints

Diagnostics rather than scheduling. Do not poll `/balancer/stats` in a control
loop — it reaps stale jobs before answering, which contacts replicas.

| call                  | where             | for                                                                                                                                                                                                                                                                                   |
| --------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /health`         | gateway, balancer | liveness; `503` when nothing healthy is behind it                                                                                                                                                                                                                                     |
| `GET /balancer/stats` | balancer          | per-replica `inflight`/`started`/`completed`/`failed`/`mean_job_s`/`healthy`/`draining`, plus `tracked_jobs`, `active_jobs`, **`reaped_jobs`** (stable — jobs the platform closed because no client ever read their terminal status; a billing layer watches it) and `retention{...}` |
| `GET /gateway/routes` | gateway           | the live routing table and how many jobs it is tracking                                                                                                                                                                                                                               |

Operator observability (independent of any client):

| call                                                | for                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET /v1/jobs` (gateway: per fleet; balancer: flat) | every job WITH what it is — running `{job_id, backend, elapsed_s, eta_s, task, prompt, seconds, via}`, queued `{ticket, position, priority, task, prompt}`, and the last 50 terminal `{status, closed_by, finished_at, ...}`. Request metadata is captured by the balancer at submission — the engine records none                                                                                                                                                                                                                                                                                                |
| `GET /metrics` (both)                               | Prometheus text exposition; the gateway aggregates with `fleet="..."` labels. Point scrapers here                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `GET /v1/events` (both; SSE)                        | the FIREHOSE: every lifecycle transition pushed live — `queued`/`started`/`handover`/`terminal` (with `closed_by`)/`cancelled`/`rejected`, plus `fleet` (attach/revive), `retention`, `fleet_down`, `gap`. Gateway multiplexes all fleets with a `fleet` tag and self-healing reconnects, and honors `?limit=N` (a bounded tap) only. On the **balancer** endpoint, `?since=<seq>` additionally replays the 500-event ring for reconnect catch-up, and a slow subscriber gets a `gap` marker, never back-pressure (seq numbers are per-balancer, so replay cannot exist at the mux). Heartbeat comments every 15s |

`deepme top [--watch] [--json]` renders `/v1/jobs` with live denoise stages
per running job — the operator's one-screen answer to "what is the platform
doing right now".

Runtime fleet membership (what `fleet switch` drives; usable for surgery):

| call                                                | codes                                                                                         |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `POST /balancer/backends {url}`                     | `200`; `400` bad url or failed health probe (a typo cannot join rotation)                     |
| `POST /balancer/backends/drain {url}`               | `200` (jobs keep running, no new work); `404` not attached                                    |
| `POST /balancer/backends/undrain {url}`             | `200` re-armed; `404` not attached                                                            |
| `DELETE /balancer/backends?url=`                    | `200`; `404` not attached; `409` jobs running or a submission on the wire — retry after drain |
| `POST /gateway/reload {routes, catalog?, pricing?}` | `200` (in-place swap, job-owner map preserved — no severed streams); `400` empty routes       |

Error-taxonomy addenda: `503 "no pricing table deployed"` from `/v1/pricing`
is an OPERATOR condition (restart/reload the gateway), not a backoff; a
retried `502` on a job id may legitimately resolve to a **terminal 404**
("the fleet that held it no longer exists") after a resize — stop retrying
on that. `DELETE` of an unknown ticket is `404` on every route.

`mean_job_s` is the figure behind every ETA and the stale-job threshold. If it
reads in fractions of a second for a model whose generations take minutes, the
occupancy accounting is broken — not the model.

---

## 5. Errors

| status              | class                                                        | retry?                                  |
| ------------------- | ------------------------------------------------------------ | --------------------------------------- |
| `400`               | client error — unknown task, bad limits, malformed condition | no, fix the request                     |
| `400` "not running" | capability not deployed (e.g. `ref2va` under `max-base`)     | not until an operator changes the fleet |
| `404`               | unknown job id                                               | no                                      |
| `409`               | cancel refused — already submitted                           | no                                      |
| `502`               | a fleet was unreachable                                      | yes, with backoff                       |
| `503`               | no healthy replica                                           | yes, with backoff                       |

The two `400` shapes read differently on purpose: an **undeployed capability**
is an operations problem your service layer should surface to an operator; an
**unknown task** is a bug in the caller.

---

## 6. Architecture

```
your service layer  (identity, pricing, quotas, customer queueing)
        │
        ▼
   gateway :8080          model + task -> fleet; discovery; job affinity
        ├──────────────► balancer :9100 ──► h3-video ×3   (t2va, fl2va)
        └──────────────► balancer :9200 ──► h3-ref  ×1    (ref2va)
```

- **Gateway** — one address. Routes by model and task, serves discovery,
  tracks which fleet owns each job id.
- **Balancer** (one per model) — admission queue, least-outstanding-work
  routing, job affinity, occupancy and timing, the event stream.
- **Replica** — one generation at a time. Never addressed directly by your
  service layer.

Job ids are minted per replica and mean nothing elsewhere, so both hops keep
their own ownership map. A poll always finds its way home; if a map is lost to
a restart, the hop probes rather than 404s.

---

## 7. Capacity and cost

GPUs are the hard constraint: **8 cards ÷ 2 per replica = 4 replicas**, in any
mix. The split between capability tiers is therefore a pricing decision, and
lives in `deploy/fleets/*.toml`:

| fleet        | composition             | trade                                |
| ------------ | ----------------------- | ------------------------------------ |
| `max-base`   | h3-video ×4             | maximum base throughput, no `ref2va` |
| `production` | h3-video ×3 + h3-ref ×1 | −25% base, premium tier available    |
| `even`       | ×2 + ×2                 | even split                           |

Measured: ~800s for a 5s clip at the default schedule (50 sigma points →
49 denoise iterations; both numbers appear in logs) (superlinear in
duration — 15s costs 1.51× linear); four concurrent clips in 18m28s
against ~62m serial. Roughly **17 clips/hour, ~170 overnight** at four
replicas. Cost is linear in `num_inference_steps`. Per-length timing and
throughput tables are in [VIDEO-ARCHITECTURE.md](VIDEO-ARCHITECTURE.md)
§4–5.

Operators change the mix with `deepme fleet switch <name>` — delta-aware:
unchanged replicas keep serving, leavers drain (running jobs finish),
arrivals attach to live balancers so queues survive. `fleet apply` is the
blunt full-teardown path for a cold or broken machine, and needs
`deepme gateway start` to refresh routing and discovery.

---

## 8. Operating it

```sh
uv run deepme discover                 # catalog: models, tasks, endpoints
uv run deepme ports                    # declared allocations + conflicts
uv run deepme fleet list|plan|status|switch|apply   # composition
uv run deepme capacity [--watch]       # the scheduling signal, as the app sees it
uv run deepme econ [--json]            # costs, floors, mixes, surge — with drift checks
uv run deepme balancer status -m <p>   # per-replica load for one model
uv run deepme gateway status           # what is routed where
uv run deepme store status|clear       # shared weight store
uv run deepme watch                    # health transitions to the ledger
```

See `RUNBOOK.md` §7.4–7.6 for fleet operations, §7.7 for the vendored-patch
workflow, §7.8 for economics/capacity/artifacts,
and `README.md` for the measured performance and memory characteristics.
