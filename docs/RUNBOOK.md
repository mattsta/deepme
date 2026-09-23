# Operator Runbook

Procedures for running the deepme inference platform. All commands run from
this repo (`uv sync` once). Every operation is idempotent unless noted.

## 1. Fresh server bring-up

```sh
uv run deepme config init --host <ip> --user <ssh-user>
uv run deepme gpu-fix          # driver stack; RTX 5090 REQUIRES -open module
uv run deepme setup            # ninja/g++ + engine uv projects
uv run deepme guard setup      # (optional) guard proxy dependencies
uv run deepme status           # verify: GPUs visible, no units yet
```

Different driver series (future hardware): set `[server] driver_series = "..."`
in `.deepme/config.toml` before `gpu-fix`. Different cluster shape: set
`[cluster] gpus / vram_gb` (drives browse fit-checks and perf estimates).

## 2. Deploy a model

```sh
uv run deepme browse                    # or: model add org/Repo
uv run deepme model show -m <p>         # resolved spec + fit check
uv run deepme model variants -m <p>     # (GGUF) choose a quantization
uv run deepme download start -m <p>     # resumable; watch: download status
uv run deepme serve start -m <p>        # waits for health, diagnoses failures
uv run deepme test -m <p> "hello"
uv run deepme bench -m <p> -c 1 -c 4    # record baseline in the ledger
```

## 3. Switch models (one command)

```sh
uv run deepme switch <profile>
```

Fit-checks, auto-downloads missing weights, stops conflicting units, WAITS
for GPU memory release, deploys, waits for health, proves with a completion,
re-points the guard if it's running, persists the default. Every step lands
in the ledger (`deepme runs`).

## 4. Incident response

```sh
uv run deepme watch               # sample every model; log up/down transitions
uv run deepme status              # fleet: GPUs, units, guard, API health
uv run deepme serve diag -m <p>   # restart count + log window before crash
uv run deepme serve logs -m <p> -n 200   # raw journal
uv run deepme doctor -e <engine>  # engine-critical imports w/ tracebacks
uv run deepme jit-heal            # finish raced flashinfer JIT builds
uv run deepme gpus                # every GPU process with owner/command
uv run deepme reclaim             # kill non-deepme GPU processes
uv run deepme gpu-fix             # driver userland/module realignment
uv run deepme reboot              # LAST resort (drivers reload live normally)
```

Diagnosis order for a crash-looping service: `serve diag` (real traceback) →
`doctor` (import layer) → `jit-heal` (JIT caches) → check the toolchain
facts in README ("Hard-won toolchain facts") before changing anything.

**Fleet / gateway incidents** (the video-platform failure modes):

- Before ANY balancer restart: `curl :91xx/v1/queue` — a restart destroys
  every waiting ticket. `deepme balancer start` refuses over live state;
  `--force` overrides deliberately and destroys the queue. (§7.6)
- Replicas wedged / phantom occupancy: the maintenance loop self-heals in
  ≤60s (reconcile → revive → pump); `deepme top` shows what every job is,
  `deepme job-progress <id>` one job's stage. Stuck beyond that: §7.5's
  reconcile — re-running the same `fleet switch` converges from any state.
- `/v1/pricing` returning `503 "no pricing table"`: operator condition —
  `deepme gateway start` (or a reload) republishes it. (§7.8)
- Never `serve stop`/`scale` a profile with jobs or queue: the guard
  refuses; drain via `deepme fleet switch` instead. `--force` kills paying
  work.

`watch` samples every deployment on an interval and writes each transition to
the runs ledger (`uv run deepme runs`), so an outage that happened while
nobody was looking is still on the record. It tracks systemd restart counts
alongside reachability, because a service crash-looping on a 40-second timer
answers health probes between crashes and otherwise reads as healthy.

`Failed with result 'timeout'` together with `status=9/KILL` is not a model
crash: systemd asked the unit to stop and SIGKILLed it when it did not. On
Ubuntu the usual cause is **needrestart** — any `apt-get install` that pulls
in a shared library restarts dependent services, killing an in-flight
generation. Provisioning pins `NEEDRESTART_MODE=l` (list, never restart) for
exactly this reason; keep that on any package command you run by hand.

A launch that fails on MEMORY rather than a traceback shows up as
`Failed with result 'oom-kill'` in the journal, with the peak in the
`Consumed ... memory peak` line. Set `ram_gb` under `[resources]` in the
profile so the kill lands on that unit instead of the machine.

## 5. Guard policy change

```sh
$EDITOR deploy/guard/policy.toml
uv run deepme guard start         # push + restart (inference NOT touched)
uv run deepme guard test          # live adversarial verification
```

## 6. Version pinning & engine upgrades

Engine dependencies are **frozen to committed lockfiles**
(`deploy/engines/<engine>/uv.lock`) — `deepme setup` pushes the lock and syncs
`--frozen`, so a fresh system gets the exact validated stack (vllm 0.26.0,
torch 2.11.0, sglang 0.5.17, cuda-runtime 13.3.29, flashinfer pinned). No
silent drift. This is what makes bring-up reproducible.

**Check first, for free:** `uv run deepme upgrade-preview -e <engine>`
resolves what a fresh upgrade _would_ pull (versions of vllm/sglang/torch/
flashinfer/CUDA) in a throwaway staging dir — no install, no touch to the
live venv, safe while production serves. If nothing meaningful moves, skip
the upgrade entirely.

To upgrade (e.g. retry DSpark spec decode, or pick up a vLLM with a GGUF
loader) — a deliberate, staged operation because it drags torch/flashinfer/
CUDA, the fragile part of the stack:

```sh
$EDITOR deploy/engines/<engine>/pyproject.toml   # widen/bump the version range
uv run deepme setup -e <engine> --upgrade        # re-resolves + refreshes the
                                                 # committed lock (commit it!)
uv run deepme doctor -e <engine>                 # imports still clean?
uv run deepme browse --refresh-archs             # newly supported archs/formats
# schedule a restart window, then:
uv run deepme serve start -m <p> [--spec]
uv run deepme bench -m <p> -c 1                  # compare against the ledger
```

If the upgrade regresses, `git checkout deploy/engines/<engine>/uv.lock` and
`deepme setup -e <engine>` restores the validated versions exactly.

## 7.1. CPU colocation (GGUF on cores, alongside GPU models)

```sh
uv run deepme model add <org/Repo-GGUF> --name <p> --port 8010
$EDITOR deploy/models/<p>.toml     # add [resources] mode="cpu", cpus, ram_gb
uv run deepme model variants -m <p>   # pick a quant (policy: >= 6-bit)
uv run deepme resources               # confirm free cores/RAM
uv run deepme download start -m <p>   # fetches ONLY the selected variant
uv run deepme serve start -m <p>      # colocates; does not touch GPU models
```

Requires the llama.cpp engine: `uv run deepme setup -e llamacpp` (CUDA source
build; also usable for GPU-offloaded GGUF). Verify with `deepme resources`
(claims table) and `systemctl show <unit> -p AllowedCPUs -p MemoryMax`.

## 7.2. Multi-model (NxY) operation

Give each profile a disjoint `gpus = [...]` list and distinct `port`.
Non-overlapping profiles coexist; `serve start`/`switch` only displaces
units whose GPU sets actually conflict (read from unit files on the server).

How many GPUs a model needs is a property of its _recipe_, not its size. A
62GiB video model configured for layerwise offload occupies 2 GPUs and a
large slice of host RAM; the same model configured for resident weights
would not fit on the whole machine. Check `deepme model show -m <p>`
(`fit_check`) before assuming a placement, and `deepme resources` for what is
actually free — RAM is claimed too, so two offload-heavy models can conflict
without sharing a single GPU.

## 7.3. Diffusion / media models

Serve them exactly as the engine vendor publishes for your hardware —
sglang's cookbook has a per-GPU-model recipe, and its auto-tuner assumes
datacenter cards. Put the recipe in `[serve.engine_args]` with a link.
Verify with `deepme generate` (produces a real file), not just health.

## 7.4. Replica fleets (max concurrency for one model)

Media engines serve ONE request at a time, so throughput comes from running
several copies. Measured on 8x32GiB: four H3 replicas in 172GiB of host RAM,
four concurrent clips in 18m28s against ~62m serial (3.4x).

```sh
uv run deepme plan                       # what fits in the free space now
uv run deepme serve scale -m <p> -n 4    # lay N copies across free GPUs
uv run deepme serve wait  -m <p>         # block until every replica answers
uv run deepme balancer start -m <p>      # ONE api per profile (:9100, :9200)
uv run deepme balancer status            # per-replica inflight / done / mean job
uv run deepme showcase run -m <p>        # concurrent proof + demo assets
```

`balancer start` MUST run after a scale: it rebuilds its backend list from the
live units, so before that it still points at the previous ports.

**Verify by HTTP, never by unit state.** `ActiveState=running` means the
process exists, not that it serves — a replica can be healthy on a port you
are not watching. `serve wait` probes each assigned port and reports drift
explicitly. Every port comes from `deploy/ports.toml`; nothing is derived from
anything else, because a diffusion server binds its API port AND the next one
and derived assignment handed replica 1 a port replica 0 already owned. Check
overlaps with `uv run deepme ports`.

**Host RAM is uncapped, deliberately.** GPUs bind here (8 cards / 2 per
replica = 4 replicas) and 4 replicas rest at ~163GiB of 503GiB. A cgroup cap
has to cover the LOAD PEAK, which is several times the resting size, so every
cap was a guess that cost a restart to discover. Set `ram_gb` again only if a
model genuinely threatens the machine.

## 7.5. Changing the fleet mix at runtime

```sh
uv run deepme fleet list                 # available mixes
uv run deepme fleet switch max-base     # 3+1 -> 4+0 by DELTA, live
uv run deepme fleet switch production   # and back
```

`fleet switch` is the resize path; `fleet apply` remains the blunt one for a
cold machine or a broken state. Switch auto-detects the running fleet from
live unit claims (never from what was applied last), computes the delta, and:

- **keeps** replicas whose unit, GPUs and port already match the target —
  they are not restarted, not even touched;
- **drains** the ones that leave: out of rotation immediately, running
  jobs finish, then the unit stops and the balancer detaches it;
- **starts** the arrivals and attaches them to their balancer AT RUNTIME —
  the balancer is not restarted, so waiting tickets and job tracking
  survive the resize;
- **re-points the gateway** on any shape change — the frozen catalog and
  replica counts in `/gateway/discover` go stale on any resize, so only a
  true no-op switch skips the restart.

It refuses to remove a profile that still has queued tickets no remaining
fleet can serve (`--force` abandons them, explicitly — the abandoned ticket
ids are printed, since their holders are about to poll 404s). Every switch
lands in the runs ledger (`deepme runs`).

**Re-running is always safe.** The switch ends with a reconcile pass that
diffs each balancer's live membership against the target: arrivals whose
attach failed last time are attached, stale backends detached, leftover
drains un-drained, orphan balancers stopped. Whatever intermediate state a
crash or Ctrl-C left, running the same switch again converges.
`deepme fleet status` shows the current shape — including per-replica
balancer membership and draining flags — and names which declared mix it
matches.

Behind it are balancer admin calls, usable directly for one-off surgery:

```
POST   :91xx/balancer/backends         {"url": "http://127.0.0.1:8130"}
POST   :91xx/balancer/backends/drain   {"url": ...}   # jobs keep running
POST   :91xx/balancer/backends/undrain {"url": ...}   # re-arm after an abort
DELETE :91xx/balancer/backends?url=... # refused (409) while jobs run there
```

Attach probes the replica's /health first, so a typo cannot enter rotation.
Runtime membership is persisted next to the balancer (`backends-<profile>.json`)
and wins over the deploy-time list on a crash-restart; `balancer start` deletes
it — an explicit start is a fresh freeze.

## 7.6. Service layer (gateway, queue, events)

One address in front of every fleet. Start it AFTER the balancers — it reads
their addresses from the registry and records routes at setup time.

```sh
uv run deepme gateway start --setup   # routes model+task -> balancer (:8080)
curl :8080/v1/models                  # what is served, no parameters needed
curl :8080/gateway/discover           # full catalog: tasks, limits, endpoints
curl :8080/v1/queue                   # backlog across every fleet
```

`deepme generate` now prefers the gateway, then the balancer, then the
replica, using whichever is actually up (`cli.front_end`).

**Draining before a balancer restart.** The admission queue is in memory, so
restarting a balancer discards every WAITING ticket while submitted work
survives on the replicas. Check `GET /v1/queue` first; a restart with tickets
waiting strands those callers on ids that will 404 forever.

**Reading the numbers.** `mean_job_s` in `balancer/stats` is the figure behind
every customer ETA, the queue wait estimate and the stale-job threshold. If it
reads in fractions of a second for a model whose generations take minutes, the
occupancy accounting is wrong, not the model — see `Balancer.reserve`.

## 7.7. Shared weight store (vendored engine fork)

Layerwise offload keeps weights in host RAM as pinned per-process buffers, so
N replicas held N identical copies (96GiB per rank) and the box capped at 2.
A runtime hook backs those buffers with /dev/shm and re-pins each mapping with
`cudaHostRegister`, so replicas share one physical copy at full DMA speed.

Measured: private memory per rank **95.9GiB -> 2.8GiB**; store 124GiB paid
once; output **byte-identical** (same prompt+seed produced the same sha256 as
the pre-fork baseline); step time unchanged.

```sh
uv run deepme patches install-hooks   # install into the engine venv
uv run deepme patches status          # applied / clean / stale per patch
uv run deepme store status            # what the shared store holds
uv run deepme store clear             # reclaim segments nothing is mapping
uv run deepme patches remove-hooks    # back to stock engine (incident escape)
```

Enabled per deployment by `DEEPME_SHARED_WEIGHTS = "1"` in the profile's
`[serve.env]`. Unset means stock behaviour — the hook is inert.

**After an sglang upgrade**, in order:

1. `uv run deepme upgrade-preview -e sglang` — see what a resolve would pull.
2. Upgrade, then `uv run deepme patches status`. **stale** means the engine
   moved under a patch; re-base it and refresh checksums with
   `deepme patches record`.
3. `deepme patches install-hooks` (idempotent; `deepme setup` also does it).
4. Re-run the correctness gate: generate with a fixed seed and compare the
   sha256 against the recorded baseline. The hook names
   `LayerwiseOffloadManager._initialize`; if that method is renamed the hook
   logs a warning and no-ops, so the engine keeps working with stock memory
   behaviour and you lose only the sharing.

**Two costs to know.** Teardown is slower — un-registering page-locked memory
scales with store size, so a 4-replica cycle is minutes, not seconds
(`GPU_RELEASE_TIMEOUT_S` allows for it). And segments outlive the processes
that made them: a model you will not serve again holds its store until
`deepme store clear` runs.

## 7.8. Economics, capacity and artifacts (daily operations)

```sh
uv run deepme capacity --watch     # free slots, queues, running, disk — live
uv run deepme econ                 # costs/floors/mixes/surge + drift warnings
uv run deepme job-progress <id> -m <p>   # one job's stage, same chain as displays
uv run deepme artifacts audit -m <p>     # what's on the work volume
uv run deepme artifacts sweep -m <p> --older-than-hours 48   # manual override
```

Artifact RETENTION is automatic (balancer maintenance loop): 6h for
never-fetched outputs, 1h after the first fetch, running jobs untouchable;
`inputs/` uploads follow the 6h rule. `/balancer/stats` carries the
`retention` counters; `capacity` carries `disk` with an 0.85 watermark.
The sweep command is the manual override, dry-run by default.

Prices derive from `deploy/economics.toml` × measured occupancy and FREEZE
into the gateway at start/reload — `deepme econ` warns when the live table
is older than the config would derive, or when the running replica count
diverges from the cost basis. Re-measure durations with the duration-curve
batch and `econ --record-duration`.

`docs/INVARIANTS.md` is the release's self-contract: every self-management
guarantee with its enforcement and test locations. Breaking a row is
breaking the release.

## 8. History & audit

```sh
uv run deepme runs                # deployments, switches, benches, guard ops
uv run deepme runs show <id>      # full spec + outcome for any operation
```
