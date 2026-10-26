# deepme

Config-driven serving platform for large models on a dedicated multi-GPU
server (reference: 8x RTX 5090 32GB, PCIe-only, 500GB RAM). Today it runs a
production **video-generation platform** — replica fleets behind per-model
load balancers and a routing gateway, with an admission queue, derived
pricing/economics, artifact retention, and full operator observability
(`top`, `capacity`, `events`, Prometheus `/metrics`) — alongside
OpenAI-compatible text-LLM serving. One CLI drives all of it.

**See it in action:** the clips in the gallery at
<https://matt.sh/apps/facetious/propaganda/> were generated with this
repository and the MiniMax-H3 video profiles in `deploy/models/`
(`h3-video`, `h3-ref`).

**A model = one profile file** (`deploy/models/<name>.toml`) declaring the HF
repo and serving intent (engine, GPUs, port, context, memory). Everything
knowable from an API is auto-fetched, never hand-written: weight sizes and
shard counts come from the Hugging Face API (cached in `.deepme/hf/`). Each
profile runs as its own systemd unit pinned to its GPU set, so multiple
models coexist when their GPU sets don't overlap; conflicting units are
stopped automatically on launch.

All operations run from this repo via `uv run deepme <command>` over SSH.
Services expose OpenAI-compatible APIs behind a bearer key. No server
identities live in the repo.

**Docs**: [docs/RUNBOOK.md](docs/RUNBOOK.md) (operator procedures: bring-up,
deploy, fleet resizes, incidents, economics, engine upgrades) ·
[docs/INTERNAL-API.md](docs/INTERNAL-API.md) (the service-layer contract:
submit/follow/queue/pricing/observability) ·
[docs/INVARIANTS.md](docs/INVARIANTS.md) (every self-management guarantee,
with where it is enforced and tested) ·
[docs/USER-GUIDE.md](docs/USER-GUIDE.md) (client API usage: chat,
streaming, tools, guard controls, and video generation) ·
[docs/VIDEO-ARCHITECTURE.md](docs/VIDEO-ARCHITECTURE.md) (how the video
model is placed across PCIe-only GPUs, the shared weight-memory design, and
measured timing/concurrency by clip length).

## Cold start → serving (fresh system runbook)

```sh
uv sync                                                  # local CLI
uv run deepme config init --host <ip> --user <ssh-user>
uv run deepme gpu-fix        # NVIDIA driver stack: -open kernel module (REQUIRED
                             # for RTX 5090/Blackwell), userland/module alignment
uv run deepme setup          # build tools + engine uv projects (~deepme-runtime)
uv run deepme model list     # pick or add a profile
uv run deepme download start -m <profile>   # resumable, tmux, hf_transfer
uv run deepme serve start -m <profile>      # fit-check, deploy, wait healthy
uv run deepme test -m <profile> "hello"     # real completion + timing
uv run deepme bench -m <profile> -c 1 -c 4  # benchmark
```

`deepme model use <name>` persists a default profile so `-m` can be omitted
everywhere. Every step is idempotent.

## Restart / redeploy the running platform

After changing a profile's `[serve.env]` or `engine_args`, the
balancer/gateway source, or the port registry. **Bottom-up, always:**

```sh
uv run deepme capacity                                # what is running now
uv run deepme serve scale -m h3-video -n 4 --force    # 1. replicas (~6 min)
uv run deepme balancer start -m h3-video --force      # 2. balancer
uv run deepme gateway start                           # 3. gateway
uv run deepme capacity && uv run deepme top           # verify
```

**`serve restart` does not apply env or engine-arg changes** — it is a bare
`systemctl restart` and reuses the unit file those values are frozen into.
Only `serve scale`/`serve start` re-render the unit, so the wrong command
looks successful and changes nothing.

`--force` overrides the drain guard that refuses to stop replicas with work
in flight; omit it unless you mean to kill running generations. Restarting a
balancer discards **waiting queue tickets** (in-memory) — running jobs
survive on their replicas and are re-adopted, and clients holding a ticket
resubmit themselves. Full detail, including narrower operations and the
fleet-mix path that avoids a restart entirely, in
[docs/RUNBOOK.md](docs/RUNBOOK.md) §0.

## Browsing & picking models interactively

```sh
uv run deepme browse                       # top 100 trending, filtered to what fits
uv run deepme browse -s H3                 # SEARCH the whole HF catalog by name
uv run deepme browse -s qwen --sort downloads --limit 200
uv run deepme browse --author unsloth --max-size 60   # org + size ceiling
uv run deepme browse --engine llamacpp --tag text-generation
uv run deepme browse --all                 # include excluded rows with reasons
uv run deepme browse --json                # machine-readable
```

`--search`/`--author`/`--tag` filter **server-side**, so they reach models far
outside the trending window; `--contains` filters the fetched page locally.
Cheap filters run _before_ the per-repo deep fetch, so scanning a large
catalog costs one request rather than one per model. Sort by
`trending|downloads|likes|modified`.

`browse` intersects three live data sources — HF's trending models across
**all modalities and formats** (safetensors sized per-dtype; **GGUF
quantizations sized per-variant** — each quant is an independently deployable
implementation, fit-checked against its smallest variant; thin list-API
entries trigger a one-time cached deep fetch of the repo inventory), the
architectures the _installed_ vllm/sglang registries actually support
(fetched from the server, cached; `--refresh-archs` after engine upgrades),
and the cluster weight budget (`[cluster] gpus/vram_gb` in
`.deepme/config.toml`, default 8x32GB) — and shows only deployable candidates.
Picking a row scaffolds its profile and offers deployment (explicit
confirmation; nothing restarts without a yes).

## Adding a new model

```sh
uv run deepme model add org/Some-Model      # fetches facts, writes skeleton
$EDITOR deploy/models/some-model.toml       # engine/gpus/port/context intent
uv run deepme model show -m some-model      # resolved spec + VRAM fit check
uv run deepme download start -m some-model && uv run deepme serve start -m some-model
```

The fit check fails fast (arithmetic, not a 6-minute launch) when weights +
overhead exceed `gpus × 32GB × gpu_mem`. NxY splits are just profiles with
disjoint `gpus` lists and distinct ports — e.g. two 4-GPU models with `tp = 4`.

**Switching models is one command**: `uv run deepme switch <profile>` observes
current state, auto-downloads missing weights, safely stops whatever holds the
GPUs, deploys, waits for health, and proves the API with a real completion
before returning (also persists the new default).

New model families usually fail on _kernels_, not configs (see engine status
below) — expect the first bring-up of a new architecture to need an engine
version bump in `deploy/engines/*/pyproject.toml`.

## Production profile: deepseek-v4-flash

vLLM 0.26.0, TP=8, FP8 weights + FP8 KV, **full 1M-token context** (1.29M-token
GPU KV pool), 200GB RAM KV tier (OffloadingConnector), prefix caching,
DeepSeek-V4 reasoning + tool-call parsers, detailed usage reporting
(`cached_tokens`, forced streaming usage, X-Request-Id), `--max-num-seqs 8`
(low-batch internal usage).

### Benchmarks (512-token generations, 2026-08-08)

| concurrency | TTFT median | per-stream tok/s | aggregate tok/s |
| ----------- | ----------- | ---------------- | --------------- |
| 1           | 0.27s       | 87.6             | 84              |
| 2           | 0.38s       | 77.5             | 146             |
| 4           | 0.39s       | 72.6             | 275             |
| 16          | 5.6s        | 43.3             | 470             |

Long context: 100K-token prompt prefills in 71.5s cold (~1,400 tok/s,
PCIe-TP bound); repeated prompts hit the prefix cache → **TTFT 1.2s**.
`bench` sends a unique nonce per stream by default so TTFT reflects real
prefill (not a warmed prefix cache); `--no-nonce` measures the cached path.
Results are recorded to the runs ledger; add `--out file.json` to also write
JSON.

### Video generation (MiniMax-H3, 2026-08-08)

| what                                                           | measured                                           |
| -------------------------------------------------------------- | -------------------------------------------------- |
| output                                                         | 1344x768 H.264 @ 24fps + AAC stereo 32kHz, one MP4 |
| 4s clip, default schedule (50 sigma points = 49 denoise iters) | 659s wall, 577s of it denoise @ 11.7s/step         |
| 4-second clip, 20 steps                                        | 283s wall — cost is linear in `--steps`            |
| peak GPU                                                       | 21.9GiB per card, 2 of 8 GPUs                      |
| host RAM                                                       | ~314GB peak (the DiT lives there)                  |

`--steps` is the only speed lever available here. `quality="high"` (Cache-DiT
step skipping, ~1.4x) **fails closed** on this hardware: the engine accepts it
only on the audited 4xH200 resident deployment and rejects the request rather
than silently changing output — verified live, not inferred.

That step time is the price of layerwise offload: a 62GiB DiT on 32GiB cards
streams blocks from host RAM every step. Cards with enough VRAM to hold the
model run the same recipe resident and are far faster — this configuration
buys the ability to run the model at all, on 2 GPUs, leaving 6 free.

## Using the API

OpenAI-compatible at `http://<host>:<port>/v1`, bearer key in
`.deepme/api_key` (auto-generated, gitignored). Model name = the profile's
`served_name`. Responses include `usage.prompt_tokens_details.cached_tokens`;
`/metrics` exposes Prometheus counters (cache hit rate, KV usage, latency
histograms). Plain HTTP — for anything beyond trusted networks use an SSH
tunnel or firewall the port.

```python
from openai import OpenAI

client = OpenAI(base_url="http://<host>:8000/v1", api_key=open(".deepme/api_key").read().strip())
```

### Media models (video / image)

Diffusion profiles serve a different OpenAI surface on their own port:
`POST /v1/videos` submits a job, `GET /v1/videos/{id}` reports status, and
`GET /v1/videos/{id}/content` returns the finished MP4. Generation takes
minutes, so it is a job API, not a streaming one.

```bash
uv run deepme generate -m <profile> "a corgi surfing at sunset" --seconds 4
```

`generate` submits, follows the job to a terminal status, saves the file
under `.deepme/generated/`, and records prompt, parameters, timing and peak
GPU memory to the runs ledger. Send only the parameters you mean to override
— each release ships tuned defaults for resolution and step count.

Repos that publish several task partitions (MiniMax-H3 ships `FL2VA/` for
text and first/last-frame conditioning and `Ref2VA/` for reference images)
set `subfolder` in the profile. Engines declaring `MODEL_SUBFOLDER` receive
the repo root plus the partition name; others get the joined path. Pointing
sglang at the partition directory instead trips its completeness pre-check
(it globs component dirs non-recursively and misses H3's
`video_vae/source/model.safetensors`), after which it tries to re-download
the local path as a repo id and crash-loops.

**Follow the vendor's hardware recipe.** sglang's diffusion cookbook
publishes a per-hardware configuration per model, and the engine's own
auto-tuner assumes datacenter cards. For MiniMax-H3 on 32GiB consumer GPUs
the published recipe is layerwise offload of `dit,text_encoder,vae` on 2 GPUs
with `--performance-mode memory`; the 4-GPU recipes target 80–141GiB cards.
Configuring by intuition instead cost three crash-loops here: replicated
weights (62GiB DiT per 32GiB card), then 490GB of host RAM from four ranks
each staging the whole checkpoint. Profiles carry the recipe as
`[serve.engine_args]` with a link to the source.

### Where a model's weights actually live

`fit_check` is only meaningful if it knows this, so engines declare it
(`engines.WeightPlacement`) instead of the platform assuming division by GPU
count:

| placement       | VRAM per GPU               | set by                                                                  |
| --------------- | -------------------------- | ----------------------------------------------------------------------- |
| tensor-parallel | size / tp                  | default for vllm + sglang LLMs                                          |
| layer split     | size / GPUs, derated       | llama.cpp                                                               |
| replicated      | **the whole model**        | sglang diffusion default (Ulysses splits the sequence, not the weights) |
| FSDP-sharded    | size / ranks               | `--use-fsdp-inference`                                                  |
| offloaded       | ~none; bounded by HOST RAM | `--layerwise-offload-components dit,…`                                  |

The last two rows are why a 62GiB model runs on 32GiB cards at all, and why
the offloaded case is checked against host RAM instead of VRAM.

## Throughput estimates (`~tok/s` column)

`src/deepme/perfmodel.py`: single-stream decode is memory-bandwidth-bound, so
`est ~= effective_bandwidth / (weight_bytes x active_fraction)`. Active
fraction comes from each repo's config.json (expert count, top-k, layer dims —
dense = 1.0). Bandwidth tiers: single-GPU ~1.25TB/s; multi-GPU MoE (expert
parallel) **calibrated to our production measurement** (within ~10%);
multi-GPU dense over PCIe is estimated pending a dense-TP bench. Heuristic
for ranking (~2x confidence), not a benchmark — `deepme bench` is ground
truth and auto-records to the runs ledger.

**Weight sizing** takes the _minimum_ of two independent estimates, because
each overestimates in a different way: per-dtype param arithmetic inflates
sub-byte packings (FP4 experts — DeepSeek-V4 would read as 283GiB instead of
155GiB), while summing repo file bytes inflates multi-component repos
(MiniMax-H3's 104 files across transformer/vae/text_encoder/audio_vae read as
464GiB instead of ~62GiB for one instance). One deployment can be no larger
than either bound.

## Quantization policy

`src/deepme/policy.py` is the single allow/deny gate for weight
representations. Defaults: full-width formats and GGUF quants **>= 6 bits**
(Q6*K, Q8_0, F16/BF16) are servable; 4-bit and below are excluded;
\_unclassifiable quants count as unsupported* (unknown is never silently
allowed). An arithmetic plausibility floor (~0.5 bits/weight vs the repo's
parameter count) prevents auxiliary GGUFs (draft/speculator heads,
projectors) from ever being selected as "the model". Override per machine:

```toml
# .deepme/config.toml
[policy]
min_gguf_bits = 6
deny = ["IQ*"]          # extra fnmatch deny patterns (checked first)
allow = ["*MXFP6*"]     # extra allow patterns
```

Explicit `variant = "..."` in a profile is operator intent and is honored
as written; only auto-selection is policy-filtered.

## Output guard (forbidden-pattern firewall)

### Architecture

```
                :8001 (guarded, opt-out per request)
 client ──────► deepme-guard proxy ──────► engine API :8000 (unguarded, direct)
                │                          (vllm / sglang / ... — profile port)
                │  upstream is ALWAYS streamed (SSE), regardless of what
                │  the client asked for:
                │
                │   adapter.stream ─► typed events (content / reasoning /
                │                     tool_call deltas / finish / usage)
                │        │
                │   StreamGate: incremental release of provably-clean text
                │        │            (hold-back window + shadow scan)
                │   violation? ─► abort upstream mid-stream ─► continuation
                │                 (continue_final_message + bad_words +
                │                  logit_bias); exhaustion TRUNCATES
                │
                ├─ client stream=true  ─► true SSE piping (deltas as released)
                └─ client stream=false ─► assembled single JSON response
```

**Defaults**: the engine port (e.g. :8000) is untouched and unguarded — the
guard is a separate opt-in endpoint (:8001). On :8001, enforcement is ON by
default with `default_sets` and `scope = ["content"]`; requests opt out or
reshape per call via the `guard` body object.

| channel             | default        | enforcement                                                                                                                             |
| ------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| assistant content   | **guarded**    | full 3 layers (bad_words + gate + continuation)                                                                                         |
| reasoning_content   | passed through | opt-in `scope: ["reasoning"]` → buffered, full-request regeneration                                                                     |
| tool_call arguments | passed through | opt-in `scope: ["tool_arguments"]` → buffered, full-request regeneration (agents editing forbidden identifiers keep working by default) |

Note: layer-1 sampler bans (`bad_words`) are **request-global** — the engine
applies them to every generated token including tool arguments; only the
gate/regeneration layers are channel-scoped. Agents that must emit forbidden
strings verbatim (e.g. editing an existing identifier) should send
`{"guard": {"enabled": false}}` or a narrowed `sets` selection.
| /v1/completions | guarded | prompt-concat continuation, same three layers, violations audited |
| other endpoints | passed through | transparent proxy |

Endpoint/mode matrix (live-validated by `deepme guard test`): chat
non-stream ✓, chat stream ✓ (true incremental SSE — 100+ deltas, not
segment blobs), tool calls ✓ (delta-aggregated, JSON-intact), reasoning
fields preserved ✓, per-request bypass ✓. Violations are reported per
response in `guard.violations[]` with `channel` and `action`
(`regenerated` / `replaced ...`).

A proxy (`deepme guard`, port 8001, systemd `deepme-guard`) in front of the
model API that makes configured output patterns impossible — words, graphemes/
code points, multi-word phrases, or full regexes. Three enforcement layers:

1. **Sampler-level `bad_words`** injected into every request (multi-token
   sequence bans incl. case/space variants) — the model picks alternatives
   naturally, so most rules never even trigger downstream layers.
2. **Streaming gate** (`src/deepme/guard/gate.py`, pure logic, 28 unit tests):
   releases only provably-clean text (hold-back window = longest pattern +
   slack + current word); scans a formatting-stripped shadow too, so
   markup-fragmented evasions like `*tha*t` are caught; on violation the
   output truncates at the match start.
3. **Continuation regeneration**: resumes via `continue_final_message` with
   accumulating `logit_bias` (space/case token variants) against the rejected
   fragment. The guard NEVER authors text: after `max_retries` the output
   generation always terminates.

Policy is `deploy/guard/policy.toml`: **named rule sets** (rule kinds
`word|substring|phrase|regex`, per-rule `case_sensitive`,
plus `strip_chars`, `max_retries`, `default_sets`) — edit and
`deepme guard start` re-pushes. **Requests control enforcement per-call**
via a `guard` body object:

```jsonc
{"guard": {"enabled": false}}                       // full bypass (e.g. coding
                                                    // agents editing existing
                                                    // identifiers)
{"guard": {"sets": ["dashes", "style"]}}            // union of named sets
{"guard": {"rules": [{"kind": "word", "pattern": "moreover"}]}}  // inline,
                                                    // live for this request
```

Seam whitespace is normalized (no doubled spaces leak from the
truncate/replace/continue machinery). `deepme guard test` runs live
adversarial prompts — including a bypass case — and verifies with the local
gate that nothing leaked; responses carry a `guard.violations` audit field.
The guard is additive: deploying/stopping it never touches the inference
service.

## Serving it to other services

A service layer talks to ONE address and discovers everything else:

```sh
curl http://<host>:8080/gateway/discover     # models, tasks, limits, endpoints
curl -X POST http://<host>:8080/v1/videos    # submit (routed by model + task)
curl -N  http://<host>:8080/v1/videos/<id>/events   # progress + queue position
```

Nothing downstream hardcodes a port, a model name, or which checkpoint serves
which task — all of that moves when the fleet mix changes, and the catalog is
derived from live state rather than maintained by hand. The catalog keeps
**declared** (reserved in the port registry) separate from **running** (a unit
claims it), because treating configuration as reality is the expensive
mistake in a service map.

Full contract — request shapes for every generation mode, enforced limits,
the event stream, queue control, the error taxonomy and where the billing
line sits — is in **[docs/INTERNAL-API.md](docs/INTERNAL-API.md)**.

```
service layer -> gateway :8080 ─┬─ balancer :9100 -> h3-video x3  (t2va, fl2va)
                                └─ balancer :9200 -> h3-ref  x1   (ref2va)
```

Ports for scaled replicas, balancers and services are **declared, never
derived** (`deploy/ports.toml`, checked by `deepme ports`; a plain
single-replica `serve start` still uses the profile's own `[serve] port`).
Derivation produced two live collisions: a diffusion server binds its API
port AND the next one, and the torch.distributed rendezvous defaulted to one
fixed port for every profile.

## Replica fleets & load balancing

Media engines serve one request at a time, so throughput comes from running
several copies of a model and steering each request to whichever is free.

```sh
uv run deepme serve scale -m h3-video -n 4    # N copies across free GPUs
uv run deepme serve wait  -m h3-video         # block until all answer HTTP
uv run deepme balancer start -m h3-video      # one API per profile (:9100)
uv run deepme balancer status                 # inflight / done / mean job
uv run deepme gateway start --setup           # one address for everything (:8080)
```

Measured on 8x32GiB: **four replicas at ~163GiB of host RAM at rest, 172GiB
during concurrent generation** (two needed
402GiB before the shared weight store), four concurrent clips in **18m28s**
against ~62m serial — 3.4x, with the shortfall from clip-length variance
rather than contention. Bring-up is ~5min: the first replica is staggered
because it builds the shared store, the rest come up in parallel.

`balancer start` must run AFTER a scale — it rebuilds its backend list from
the live units. Two rules do the routing:

- **new request** -> the replica with the least outstanding _job_ work
  (occupancy spans the job, not the HTTP call that submits it: `POST
/v1/videos` returns in milliseconds while the replica works for minutes)
- **existing job** -> always the replica that minted its id, since ids mean
  nothing to the others. A lost mapping triggers a probe, never a 404.

`deepme generate` and `deepme showcase` use the most capable front end that is
actually up — gateway, then balancer, then the replica itself; `--direct` pins
to a single replica.

Ports are never derived, only declared: `deploy/ports.toml` assigns every
scaled replica, balancer and service, because a diffusion server binds its
API port AND the next one, and derived assignment handed replica 1 a port
replica 0 already owned. `uv run deepme ports` fails on any overlap.

### Progress for callers

The upstream job API reports `queued` from submission until `completed`,
which is indistinguishable from stuck. The balancer adds a live stream:

```sh
curl -N http://<host>:8080/v1/videos/<id>/events
```

```json
{"job_id":"…","status":"queued","queue_position":2,"eta_s":1642}
{"job_id":"…","status":"running","queue_position":0,"eta_s":538}
{"job_id":"…","status":"completed","queue_position":0}
```

No ETA is offered until a replica has finished something — silence beats an
invented number.

## Batch: unattended backlogs

```sh
uv run deepme batch run prompts.txt -m h3-video
uv run deepme batch status prompts
```

**Plain text** — one prompt per line; `#` and blank lines ignored:

```
A lighthouse beam sweeping through heavy fog, waves booming below
# vertical cut for social
A night market wok throwing flames, vendor calling over the crowd
```

**JSONL** — when items need their own parameters:

```jsonl
{"prompt":"A harbour at dawn, gulls wheeling","seconds":6,"seed":42}
{"prompt":"Rain on a window, thunder rolling in","seconds":5,"steps":30}
{"prompt":"A market stall at night","seconds":5,"params":{"target.aspect_ratio":"9:16"}}
```

Fields: `prompt` (required), `seconds`, `seed`, `steps`, and `params` for
anything else — dotted keys reach nested request objects.

It keeps **one item in flight per healthy replica**, deliberately. Submitting
the whole backlog at once would let the balancer assign every job to a
replica immediately, and job ids are minted per replica, so those assignments
become permanent: a slow replica keeps its share, a crashed one loses it, and
nothing can be reordered or cancelled. Holding the queue client-side means
work rebalances naturally and a crash costs one item.

State is written after every item, so an overnight run that dies at 3am
resumes rather than regenerating. It prints capacity up front from measured
history — _"~5.8h at 17.2/hour (~172 in a 10h night)"_. At ~850s per 5s clip
on four replicas that is roughly **17 clips/hour, ~170 overnight**.

Each item records the server-reported peak VRAM, which is how the request
envelope below was measured.

`--fetch-existing` winds a batch down instead of continuing it: items that
already own a server job are adopted and downloaded, nothing new is
submitted (a lost job is marked failed, never regenerated), and the pending
backlog stays parked for a later run without the flag.

## Story: chained continuation

H3's `fl2va` task pins an image to a chosen frame, so one clip's final frame
can seed the next and the pair reads as continuous motion instead of a cut.

```sh
uv run deepme story run scenes.txt -m h3-video --seconds 5
uv run deepme story join mystory --keep-seam   # re-join, no regeneration
```

Same input formats as `batch` — one prompt per segment. Segment 0 is plain
text-to-video; each later one extracts its predecessor's last frame
(`ffmpeg -sseof`, seeking from the end rather than decoding the whole file)
and conditions on it. Segments are joined with a **stream copy, not a
re-encode**: every clip comes from the same model at the same geometry, so
their codecs already match.

**The seam:** a continuation's first frame IS the frame it was seeded from,
so a naive join shows it twice. One frame (0.0417s at the fixed 24fps) is
trimmed from each continuation; `--keep-seam` disables that.

Two consequences worth planning around:

- **Strictly serial** — segment N+1 needs a frame that does not exist until N
  finishes, so a fleet cannot accelerate ONE story. Run several stories
  concurrently instead. A 10-segment 50-second story is ~2.4h on one replica.
- **Drift** — each segment sees only a single frame of history, so colour,
  lighting and subject identity can wander over many hops. Short chains hold
  together; long ones want a fixed seed and consistent style language in
  every prompt.

A failed segment stops the chain rather than continuing from a clip that was
never produced. Re-running resumes at the failure.

## Request envelope (what a caller may ask for)

Weights are fixed; **activations scale with the request**. H3 accepts 4–15s
at a fixed 24fps and any of six aspect ratios at a fixed 768 short edge, so
the largest legal request is 15s at 21:9. Measured peak VRAM
(`deploy/prompts/envelope.jsonl`, ~5 minutes to re-run):

| request      | frames | peak VRAM   | headroom of 32.6GiB |
| ------------ | ------ | ----------- | ------------------- |
| 6s 16:9      | 144    | 24.5GiB     | 8.1GiB              |
| 10s 16:9     | 240    | 26.8GiB     | 5.8GiB              |
| 15s 16:9     | 360    | 26.7GiB     | 5.9GiB              |
| **15s 21:9** | 360    | **26.7GiB** | **5.9GiB**          |

Peak **plateaus past ~10s** and barely moves with aspect ratio — layerwise
offload bounds the footprint, so it does not scale with latent volume the way
naive arithmetic predicts. The whole legal range fits; no extra clamp needed.

Measure with `steps=2`: peak is set by allocation (frames x pixels), not by
how many times the loop runs, so an envelope sweep costs minutes instead of
hours. Validated — 6s at 2 steps gave 24.5GiB against 21.9GiB for a 5s clip
at 50 steps.

**This is the budget every other knob spends from.** Resident DiT layers cost
~1.16GiB each (measured: 21.9GiB at 20 layers, 31.2GiB at 28), so 20 leaves
5.9GiB on the worst case, 24 would leave ~1.3GiB, and 28 would not fit it at
all. Do not raise residency without re-running the envelope.

## Resource allocation & colocation

The server is a pool (GPUs, NUMA-grouped CPUs, RAM) and every running unit
records its claim in its own systemd unit file (`X-DEEPME-GPUS/CPUS/RAM/PORT`),
so `deepme resources` reads the live allocation from the server's own state:

```
Capacity gpus=8 cpus=256 ram=504GiB  numa={0:128cpu, 1:128cpu}
  deepme-llm-deepseek-v4-flash  gpus=0-7  cpus=(unpinned)  port=8000
  deepme-llm-lfm-cpu            gpus=-    cpus=8-39  ram=24GiB  port=8010
Free gpus=[]  cpus=216  ram=448GiB   (reserved 8 cpus / 32GiB for the system)
```

A profile declares **demand**, and CPU-mode models colocate alongside GPU
models — they claim no GPUs, only cores and RAM:

```toml
[resources]
mode = "cpu"      # gpu (default) | cpu
cpus = 32         # logical CPUs; pinned via systemd AllowedCPUs
ram_gb = 24       # hard cap via systemd MemoryMax
```

Placement is NUMA-aware (cores are packed within one node — this box has
GPUs 0-3 + CPUs 0-63,128-191 on node 0, the rest on node 1; cross-socket
traffic dominates CPU inference). `serve start` validates the placement
against live capacity before launching, and **only genuinely conflicting
units are displaced**: GPU overlap or port collision. A CPU model never
evicts a GPU model; insufficient CPU/RAM fails the placement with a clear
message rather than silently evicting something.

## Engine registry

Engines are **declared capabilities**, not hard-coded paths:
`src/deepme/engines.py` defines each engine's weight formats, a **`Feature`
flag matrix** (continuation, bad_words, logit_bias, tokenize, kv_ram_tier,
spec_decode, ...) — view it with `deepme engines` — plus its dependency sync
environment and launch-command builder. Compatibility is the attribute
cross-product `model(format, archs, sizes) x engine(formats, archs)` at any
cardinality. Adding engine N+1 = one `EngineDef` + one
`deploy/engines/<name>/pyproject.toml`.

Subsystems consume the matrix instead of branching on engine names: the
output guard receives the upstream engine's feature set at deploy time and
runs through an **adapter hierarchy** (`src/deepme/guard/adapters.py`) —
API mechanics per backend live only there, and each guard operation is
capability-gated, so behavior degrades cleanly (no continuation support ->
truncate-only enforcement; no bad_words -> gate-only) with zero engine
conditionals in the guard loop. Responses record the action taken per
violation (`regenerated` / `replaced ...`) for audit.

Format support is **probed from the installed binaries**, never assumed:
`deepme engines` shows declared vs probed vs effective (= declared ∩ probed)
formats. Only effective formats make a model deployable, so an engine build
that lacks a loader (e.g. this vLLM 0.26 has no GGUF loader) is never
mis-advertised; `deepme setup` re-probes and invalidates the cache. When the
server is unreachable the catalog degrades to declared formats rather than
hiding everything. GGUF weight formats route to whichever installed engine
proves support (here: sglang via `--load-format gguf`, or `llamacpp` once
installed via `deepme setup -e llamacpp`).

## Engine status on SM120 (RTX 5090)

- **vLLM 0.26.0 — production.** Works with spec decode off. Its DSpark
  speculative decoding has an SM120 kernel-dispatch bug.
- **sglang 0.5.17 — blocked.** DSpark draft autotune hits a missing SM120
  sparse-MLA kernel shape. Not a config issue.
- Spec decode is worth ~2-3x single-stream decode; retry on engine upgrades
  (bump `deploy/engines/*/pyproject.toml`, `deepme setup`, `serve start`).

## Hard-won toolchain facts (encoded in provisioning — do not undo)

1. **RTX 5090 requires the NVIDIA `-open` kernel module**; the proprietary
   module fails `RmInitAdapter (0x22)`. `gpu-fix` enforces this.
2. **No system CUDA toolkit.** The engine venv's pip CUDA toolkit is the only
   toolchain; a second one poisons JIT auto-detection (deep_gemm/tilelang/
   tvm_ffi each detect CUDA differently).
3. **The pip `nvidia/cu13` dir is merged from wheels at different versions**
   (nvcc 13.3 + cuda-runtime 13.0 as shipped) → every JIT compiler dies on
   CCCL's "compiler and toolkit headers are incompatible". The engine
   pyprojects override `nvidia-cuda-runtime`/`-nvrtc` to nvcc's minor.
4. **flashinfer never JIT-compiles at runtime**: `flashinfer-jit-cache`
   (pinned to flashinfer's version, index `flashinfer.ai/whl/cu130`).
5. The cu13 wheel needs `lib64 → lib` + unversioned `.so` symlinks for JIT
   linking; post-sync fixups create them.
6. This OS's glibc is too new for CUDA < 13.3 — don't downgrade the toolkit.

## Configuration

- Server: `DEEPME_HOST`/`DEEPME_USER` env → `.deepme/config.toml`
  (`deepme config init`, gitignored; merges into existing sections — never
  clobbers `[cluster]`/`[policy]`/`[model]`).
- Cluster shape: `[cluster] gpus`/`vram_gb` — the single source for all VRAM
  budget math (fit checks, catalog capacity, perf estimates).
- Driver/guard: `[server] driver_series` (default 595), `guard_port` (8001).
- Default model profile: `deepme model use <name>` (same file).
- Engine dependencies: uv projects in `deploy/engines/<engine>/` with
  **committed `uv.lock` files**. `deepme setup` syncs `--frozen` to the lock
  (exact validated versions — reproducible on any fresh system, no drift).
  `deepme setup --upgrade` re-resolves within the pyproject ranges and
  refreshes the lock (deliberate; bench + validate after). All dependency
  changes go through these files — never ad-hoc installs.
- API key: `.deepme/api_key`, auto-generated.

## Commands

| command                                                       | what it does                                                                                                         |
| ------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `model list\|show\|use\|add\|refresh\|variants`               | manage profiles (facts from HF API); `variants` picks a GGUF quant                                                   |
| `browse [--all] [--limit N]`                                  | interactive picker: HF trending x engine support x cluster fit                                                       |
| `runs` / `runs show <id>`                                     | deployment/bench ledger (history, full specs, outcomes)                                                              |
| `check [--fix] [--types]`                                     | quality gate: ruff + pytest in ~2s; `--types` adds mypy (slow on 3.14t)                                              |
| `resources [--refresh]`                                       | capacity, live claims per unit, free GPU/CPU/RAM                                                                     |
| `config init\|show`                                           | target server (local, gitignored)                                                                                    |
| `status`                                                      | fleet view: GPUs, disk, all units, per-profile API health                                                            |
| `gpu-fix`                                                     | driver stack: -open module, version alignment, reload                                                                |
| `setup [-e ...]`                                              | build tools + engine uv projects (idempotent)                                                                        |
| `versions`                                                    | engine/toolchain versions per project                                                                                |
| `gpus` / `reclaim [--kill]`                                   | GPU processes / kill non-deepme GPU processes (lists by default)                                                     |
| `switch <profile>`                                            | one-command model switch: fit-check → auto-download → stop conflicts → deploy → health → completion proof            |
| `download start\|restart\|status [-m P]`                      | resumable weight download                                                                                            |
| `serve start\|stop\|restart\|status\|wait\|diag\|logs [-m P]` | service lifecycle                                                                                                    |
| `doctor [-e engine]`                                          | probe engine-critical imports, full tracebacks                                                                       |
| `jit-heal`                                                    | serially finish raced flashinfer JIT builds                                                                          |
| `test [-m P] [--pad-tokens N]`                                | one completion + timing + cache usage                                                                                |
| `test-tools [-m P]`                                           | verify structured tool_calls/reasoning output                                                                        |
| `bench [-m P] -c N... [--out f.json]`                         | TTFT + throughput sweep                                                                                              |
| `serve scale -m P -n N`                                       | run N replicas across free GPUs (first staggered, rest parallel)                                                     |
| `balancer start\|stop\|status [-m P]`                         | one API in front of N replicas; per-replica load                                                                     |
| `generate [-m P] "prompt" [--direct]`                         | one video via the best front end (gateway → balancer → replica); live step/ETA progress                              |
| `batch run <file> [-m P]` / `batch status <name>`             | unattended backlog, resumable, one in flight per replica; `--fetch-existing` collects owned jobs and submits nothing |
| `story run <file> [-m P]` / `story join <name>`               | chained clips, each seeded by the previous last frame                                                                |
| `showcase list\|run [-m P]`                                   | demo reel: concurrent proof that leaves assets behind                                                                |
| `plan [--repo R]`                                             | what fits in the FREE space right now, and where                                                                     |
| `memprofile [-m P]`                                           | host RAM per component vs measured RSS/PSS                                                                           |
| `store status\|clear`                                         | shared weight store (/dev/shm) usage and cleanup                                                                     |
| `patches status\|apply\|revert\|install-hooks\|record`        | vendored engine changes + upgrade staleness detection                                                                |
| `watch [-i N]`                                                | sample every deployment; log up/down transitions to the ledger                                                       |
| `fleet list\|plan\|status\|switch\|apply`                     | declared mixes; `switch` resizes live by delta (drains, re-attaches, converges on re-run)                            |
| `gateway start\|stop\|status`                                 | one address in front of every fleet: model+task routing, capacity, pricing                                           |
| `capacity [--watch] [--json]`                                 | the scheduling signal: free slots, queues, running jobs, disk — what an app-layer scheduler polls                    |
| `top [--watch] [--json]`                                      | operator view of EVERY job: prompt/task/stage/step-countdown, queue positions, recent outcomes                       |
| `events [--raw]`                                              | subscribe to the platform firehose: every job/fleet transition pushed live (SSE), fleet-tagged                       |
| `discover [--json]`                                           | service catalog: models, tasks, every endpoint declared-vs-running                                                   |
| `econ [--json] [--record-duration B]`                         | costs, floors, mixes, surge — derived from economics.toml + measurements, with live drift warnings                   |
| `job-progress <id> [-m P]`                                    | one job's live stage via the exact chain every progress display uses                                                 |
| `artifacts audit\|sweep [-m P]`                               | server-side generated files: usage view; manual TTL sweep (dry-run default) over the automatic retention             |
| `batch list`                                                  | every batch on disk with its counts                                                                                  |
| `ports [--check]`                                             | the declared port registry; exits non-zero on any overlap                                                            |
| `engines` / `upgrade-preview -e E`                            | engine capability matrix / what an upgrade would pull, resolved offline                                              |
| `guard setup\|start\|stop\|status\|test`                      | output-guard proxy lifecycle and live adversarial verification                                                       |
| `reboot [--force]`                                            | reboot server (refuses while generations run; confirmed; ledgered)                                                   |
| `ssh "cmd"`                                                   | escape hatch (one-time inspection only)                                                                              |

## Layout

- `deploy/models/*.toml` — model profiles (operator intent only)
- `deploy/engines/*/pyproject.toml` — engine dependency projects (uv-managed)
- `src/deepme/profiles.py` — profile loading → resolved RunSpec + fit check
- `src/deepme/hfmeta.py` — Hugging Face API metadata (cached, never hand-written)
- `src/deepme/config.py` — server/global config only
- `src/deepme/remote.py` — SSH exec layer (ControlMaster, stdin scripts)
- `src/deepme/provision.py` — driver stack + engine uv-project sync
- `src/deepme/serve.py` — engine launch builders + per-profile systemd units
- `src/deepme/download.py` / `gpus.py` / `doctor.py` — weights, GPUs, diagnostics
- `src/deepme/client.py` / `bench.py` — API client + load benchmark (RunSpec-driven)
- `src/deepme/cli.py` — typer CLI wiring
