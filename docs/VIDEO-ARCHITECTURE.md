# Video Platform Architecture & Performance Review

How MiniMax-H3 (text/image → video with synchronised stereo audio) is laid
out on a PCIe-only 8x RTX 5090 machine, how its weights are shared in memory,
and what the measured timing, memory and concurrency look like. All figures
were measured on the reference deployment in August 2026. Sources and method
are in [§8](#8-measurement-provenance--method).

Operator procedures live in [RUNBOOK.md](RUNBOOK.md) (§7.4 fleets, §7.7
shared weight store). The service contract is in
[INTERNAL-API.md](INTERNAL-API.md).

## Summary

- **NVLink is not the constraint; 32 GB per card is.** The model (~62 GiB per
  deployment) cannot live in VRAM, so the transformer streams from host RAM
  over PCIe every step. That host→GPU traffic is something NVLink would not
  carry anyway. Note that the RTX 5090 has no NVLink: GeForce cards dropped it
  starting with the 40 series.
- **The 8 GPUs run as 4 independent 2-GPU replicas,** not one model spanning 8
  cards. The only GPU↔GPU traffic is the tensor-parallel exchange inside each
  pair, and both cards of a pair sit on the same socket.
- **There is one physical copy of every weight across replicas.** It is loaded
  by mmap, stored in a content-addressed `/dev/shm` store, and pinned per
  process. Private memory per GPU process went from 95.9 GiB to 2.8 GiB, and
  the binding constraint moved from RAM (2 replicas) to GPUs (4 replicas).
- **Concurrency costs nothing.** Per-clip wall time is flat from 1 to 4
  concurrent clips at every clip length, so throughput scales with replica
  count: 17.6 five-second clips/hour, or 4.0 fifteen-second clips/hour.
- **Cost is superlinear in clip length.** The model denoises the whole clip as
  one space-time volume, so step time grows faster than frame count: 5 s → 15 s
  is 3x the frames but about 4.5x the time per step. VRAM, by contrast,
  plateaus past ~10 s.

---

## 1. Hardware and constraints

| resource    | reference machine                                                |
| ----------- | ---------------------------------------------------------------- |
| GPUs        | 8x RTX 5090, 32 GB each (SM120), **PCIe only, no NVLink**        |
| CPUs        | 2x AMD EPYC 7702 (PCIe 4.0), 2 NUMA nodes, 256 logical CPUs      |
| NUMA layout | GPUs 0–3 on node 0, GPUs 4–7 on node 1                           |
| host RAM    | 503 GiB                                                          |
| engine      | sglang ≥ 0.5.17 ([pin](../deploy/engines/sglang/pyproject.toml)) |

MiniMax-H3 is a diffusers pipeline: a DiT transformer, text encoder, video
VAE and audio VAE. It ships two task partitions in one repo. `FL2VA` serves
`t2va`/`fl2va` (profile [`h3-video`](../deploy/models/h3-video.toml)), and
`Ref2VA` serves `ref2va` (profile [`h3-ref`](../deploy/models/h3-ref.toml)).
One deployment is ~62 GiB, about twice a card's VRAM.

## 2. Placement: how one model spans GPUs

There are two ways to run a model larger than one card:

1. **Shard it across GPUs that exchange data constantly** (Ulysses sequence
   parallelism, FSDP, wide tensor parallelism). This is where NVLink matters,
   and it needs the weights resident in VRAM.
2. **Keep the transformer in host RAM and stream blocks to the GPU as each
   denoising step needs them** (layerwise offload).

The profiles use (2). This is the vendor's published 2x RTX 5090 recipe, linked
from the profile and carried verbatim in `[serve.engine_args]`:

| flag                                                  | effect                                                    |
| ----------------------------------------------------- | --------------------------------------------------------- |
| `--tp-size 2`                                         | each replica = one GPU pair; layers split across the pair |
| `--ulysses-degree 1`                                  | no sequence parallelism (it requires resident weights)    |
| `--performance-mode memory`                           | memory-first tuning                                       |
| `--layerwise-offload-components dit,text_encoder,vae` | weights live in host RAM, streamed per layer              |
| `--dit-offload-prefetch-size 1`                       | next block is fetched while the current one computes      |
| `--dit-layerwise-resident-layers 20`                  | 20 DiT blocks (~1.16 GiB each) stay on the card           |
| `--enable-torch-compile false`                        | vendor notes compile changes numerical output             |

How the fit check reasons about this placement (resident vs streamed, bounded
by host RAM instead of VRAM) is in `_sglang_weight_placement` in
[`src/deepme/engines.py`](../src/deepme/engines.py). The README's
"Where a model's weights actually live" table covers the other placements.

### Fleet topology

```
        NUMA node 0 (socket 0)                    NUMA node 1 (socket 1)
 ┌─────────────────────────────────┐      ┌─────────────────────────────────┐
 │  replica  GPU0 ══TP2══ GPU1     │      │  replica  GPU4 ══TP2══ GPU5     │
 │  replica  GPU2 ══TP2══ GPU3     │      │  replica  GPU6 ══TP2══ GPU7     │
 └────────────────┬────────────────┘      └────────────────┬────────────────┘
                  │  PCIe 4.0: streamed DiT / encoder / VAE blocks, every step
 ┌────────────────┴────────────────────────────────────────┴────────────────┐
 │  host RAM: /dev/shm shared weight store — ONE copy, pinned per process   │
 └──────────────────────────────────────────────────────────────────────────┘
        ══TP2══  per-layer tensor-parallel exchange: the ONLY GPU↔GPU traffic
```

- **Pairs never straddle sockets.** `pick_gpus` in
  [`src/deepme/resources.py`](../src/deepme/resources.py) packs a placement
  inside one NUMA node, because tensor parallelism exchanges data every layer.
- **Replicas are independent.** No GPU traffic passes between replicas. Each
  serves one generation at a time, except that the `h3-video` profile allows
  dynamic batching of up to 2 requests with identical sampling signatures
  (`--batching-max-size 2`, enabled for the final days of measurement).
- **Routing.** Each profile has a balancer that sends each new request to the
  replica with the least outstanding _job_ work, and the gateway routes by
  model+task. See [INTERNAL-API.md](INTERNAL-API.md) and RUNBOOK §7.4–7.6.
- **Mix.** 8 GPUs / 2 per replica = 4 replicas in any arrangement. The split
  between `h3-video` and `h3-ref` is a pricing decision
  ([`deploy/fleets/`](../deploy/fleets/),
  [`deploy/economics.toml`](../deploy/economics.toml)).

## 3. Memory architecture: one copy of every weight

Because the weights live in host RAM rather than VRAM, RAM, not GPUs, first
decided how many replicas fit. The memory layout therefore aims to keep
exactly one physical copy of every weight on the machine.

### Load path: mmap into the shared page cache

The engine's default loader (the Run:ai model streamer) stages every
safetensors file into **private** CPU memory per rank before handing tensors
over. That costs ~230 GiB transiently per replica, so two replicas loading at
once would run the box out of memory. The profiles set
`SGLANG_USE_RUNAI_MODEL_STREAMER=0`, which makes the loader mmap each shard and
read one tensor at a time. The pages then land in the kernel's **shared** page
cache, which every replica reads from.

### Offload buffers: a content-addressed shared store

Layerwise offload packs each `(layer, dtype)` group into one flat pinned CPU
buffer. Pinned memory belongs to one process, so N replicas would hold N
identical copies (measured at 96 GiB per rank). A runtime hook,
[`src/deepme/hooks/deepme_hooks.py`](../src/deepme/hooks/deepme_hooks.py),
changes that while `LayerwiseOffloadManager._initialize` runs:

- **Shared backing.** It swaps the allocator for one that returns tensors
  backed by files in `/dev/shm/deepme-weights`, which is tmpfs, so a "file"
  only ever costs RAM. Every process maps the same physical pages.
- **Pinned speed is kept.** `cudaHostRegister` page-locks each process's own
  view of an existing mapping, so host→GPU copies stay full-speed pinned DMA.
  This is what makes "shared" and "pinned" compatible.
- **Keyed by content, not path or rank.** A component fingerprint (structure
  plus sampled values) keys each buffer, which has three consequences:
  - Replicated components, such as the text encoder needed by both ranks of a
    pair, are stored once rather than once per rank.
  - Tensor-parallel shards differ by content, so they correctly stay separate.
  - FL2VA and Ref2VA ship a byte-identical 62.1 GiB text encoder and audio VAE
    and differ only in the transformer, so a mixed `h3-video` + `h3-ref` fleet
    maps one copy of the shared parts plus each partition's transformer. Both
    profiles opt in; the `h3-ref` side is not yet proven live (see
    [§7](#7-open-questions-not-measured)).
- **Fails safe.** The fill copy always re-runs, so the hook cannot corrupt
  weights by misjudging whether a buffer is ready. Any failure falls back to
  the engine's stock allocation.
- **Opt-in.** The hook runs only when `DEEPME_SHARED_WEIGHTS=1` is set in the
  profile's `[serve.env]`. Both H3 profiles set it, together with
  `SGLANG_USE_RUNAI_MODEL_STREAMER=0` and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It is installed with
  `deepme patches install-hooks`. Store admin
  (`deepme store status|clear`) is in
  [`src/deepme/sharedstore.py`](../src/deepme/sharedstore.py).

| measured                        | before sharing | with shared store                   |
| ------------------------------- | -------------- | ----------------------------------- |
| private memory per GPU process  | 95.9 GiB       | **2.8 GiB**                         |
| shared store                    | —              | 124 GiB, paid once                  |
| host RAM, 2 replicas            | 402 GiB        | —                                   |
| host RAM, 4 replicas at rest    | (did not fit)  | **~163 GiB** of 503                 |
| host RAM, 4 replicas generating | —              | ~172 GiB                            |
| replica ceiling                 | 2 (RAM-bound)  | **4 (GPU-bound)**                   |
| step time / output              | baseline       | unchanged / byte-identical (sha256) |

Host RAM is therefore deliberately left uncapped (`ram_gb = 0`); see RUNBOOK
§7.4.

Costs:

- **Slower teardown.** Un-registering tens of GiB of page-locked memory takes
  minutes per 4-replica cycle (`GPU_RELEASE_TIMEOUT_S` in
  [`src/deepme/serve.py`](../src/deepme/serve.py)).
- **Segments outlive their processes.** They stay in `/dev/shm` until
  `deepme store clear` runs.

Bring-up is ~5 min: the first replica builds the store and the others start in
parallel and map it.

### GPU-side budget

20 resident DiT blocks keep every step from being a full round trip to RAM. The
residency count was set by measuring peak VRAM across the legal request
envelope ([`deploy/prompts/envelope.jsonl`](../deploy/prompts/envelope.jsonl)):

| request   | frames | peak VRAM per card                   |
| --------- | ------ | ------------------------------------ |
| 4 s 16:9  | 96     | 21.9 GiB                             |
| 5 s 16:9  | 120    | 22.8 GiB (production median)         |
| 6 s 16:9  | 144    | 24.5 GiB                             |
| 10 s 16:9 | 240    | 26.8 GiB                             |
| 15 s 21:9 | 360    | **26.7 GiB** (largest legal request) |

- **Headroom.** 20 resident blocks leave ~5.9 GiB on the worst case. 24 would
  leave ~1.3 GiB, and 28 would not fit.
- **VRAM plateaus past ~10 s,** because offload bounds the footprint.
- **Allocator.** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a
  measured need, not a precaution. Without it, a 15 s clip failed allocating
  2.92 GiB while 6.02 GiB sat reserved-but-unusable in fragmented allocator
  blocks.

## 4. Performance by clip length

Output is 1344x768 (at 16:9) H.264 at a fixed 24 fps, plus 32 kHz stereo AAC,
in one MP4. Wall time runs from submission to the file on disk. It therefore
includes every stage (text encode, all denoising steps, video and audio VAE
decode, mux, download). Where the engine's own compute time was recorded, it
is within 4–9 s of wall time.

At the default 49 denoising steps (50 sigma points):

| clip | frames | wall time per clip                             | time per step | peak VRAM |
| ---- | ------ | ---------------------------------------------- | ------------- | --------- |
| 4 s  | 96     | ~645–715 s (~11–12 min)                        | ~12.5 s       | 21.9 GiB  |
| 5 s  | 120    | **818 s** median (p5 790, p95 920; 1,253 runs) | ~16 s         | 22.8 GiB  |
| 10 s | 240    | ~1,800–2,000 s (~30–33 min)                    | ~37 s         | 26.8 GiB  |
| 15 s | 360    | **~3,600 s** (every complete run 57–66 min)    | ~72 s         | ~26.7 GiB |

**Why cost is superlinear in length.** H3 denoises the clip as one space-time
volume, so every step's attention spans all frames jointly. 5 s → 15 s is 3x
the frames for ~4.5x the time per step. Per second of output, a 15 s clip costs
~1.45x what a 5 s clip does at full load.
[`deploy/economics.toml`](../deploy/economics.toml) (`[measured.duration]`)
carries the curve that pricing uses for this.

Other levers:

| lever                               | measured                                                                                                    |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| steps                               | ~linear plus ~30 s fixed overhead: 4 s clip at 20 steps 283 s, at 50 steps 659 s; 5 s at 16 steps 250–335 s |
| task `ref2va`                       | 1,254 s for 5 s (24.1 s/step vs ~16.3 for `t2va`), i.e. 1.57x                                               |
| task `fl2va`                        | 886 s for 5 s (one keyframe-conditioned clip)                                                               |
| `quality="high"` (Cache-DiT, ~1.4x) | **refused** by the engine except on the audited 4x H200 resident deployment                                 |

## 5. Concurrency and throughput

Per-clip wall time at 49 steps, by the number of clips in flight on the
machine during that clip (N, which includes the clip itself):

| clip | N=1          | N=2          | N=3         | N=4 (full machine) |
| ---- | ------------ | ------------ | ----------- | ------------------ |
| 4 s  | ~700 s (2)   | 659 s (1)    | 644 s (1)   | not measured       |
| 5 s  | 825 s (6)    | 816 s (18)   | 816 s (67)  | **819 s (1,136)**  |
| 10 s | ~1,860 s (3) | 1,797 s (1)  | —           | not measured       |
| 15 s | —            | ~3,450 s (3) | 3,615 s (1) | **3,579 s (30)**   |

<sub>Run counts in parentheses.</sub>

Per-clip time is flat in N at every length. Four replicas streaming weights
from one shared store at once do not slow each other. In a direct test, four
concurrent clips finished in 18m28s against ~62 min serially (3.4x; the gap
from 4x came from clip-length variance, not contention).

Full-machine throughput (4 replicas busy):

| clip | clips / hour | output seconds / hour |                        |
| ---- | ------------ | --------------------- | ---------------------- |
| 4 s  | ~22          | ~87                   | projected from lower N |
| 5 s  | **17.6**     | 88                    | measured               |
| 10 s | ~8           | ~79                   | projected from lower N |
| 15 s | **4.0**      | 60                    | measured               |

**Realised in production batches:** unattended runs of 100–200 five-second
clips delivered **14.9–17.3 clips/hour** against the 17.6 ideal. The gap is
idle slot time between one item finishing and the next being submitted.

## 6. Interconnect review: what does PCIe cost?

| traffic                             | crosses                    | would NVLink help? | evidence                                                                                                            |
| ----------------------------------- | -------------------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------- |
| per-layer TP exchange inside a pair | GPU↔GPU, same socket      | yes, this only     | not isolated; grows linearly with length while total cost grows superlinearly, so its share shrinks on longer clips |
| streamed DiT/encoder/VAE blocks     | host RAM → GPU, every step | no                 | no cross-replica contention: per-clip time is flat in N (§5)                                                        |
| between replicas                    | nothing                    | n/a                | replicas are independent                                                                                            |

**What actually limits this machine is memory size, not link speed.** The
fastest published H3 configurations keep all weights resident and use Ulysses
sequence parallelism across 4 GPUs, plus Cache-DiT step skipping. They need
80 GB+ cards; on 32 GB cards they are not configurable (resident) or the engine
refuses them (`quality="high"`).

**Contrast with the LLM path.** Where a single model genuinely spans all 8
cards with a collective every layer (DeepSeek-V4-Flash, TP=8 + EP), PCIe is
the bottleneck. Long-prompt prefill is TP-bound; see the calibration notes in
[`src/deepme/perfmodel.py`](../src/deepme/perfmodel.py). The video placement
deliberately avoids that pattern.

## 7. Open questions (not measured)

- **Shared store on Ref2VA is configured, not yet proven.** In production,
  `h3-ref` ran with an empty `[serve.env]`, so it loaded privately and kept
  private offload buffers; the runs ledger records exactly that. The
  production env from `h3-video` is now committed to
  [`h3-ref.toml`](../deploy/models/h3-ref.toml), but no deployment has run
  with it. On the first one, check that `deepme store status` shows the text
  encoder and audio VAE segments shared across both profiles, and compare a
  fixed-seed output hash against a stock run, as RUNBOOK §7.7 does for FL2VA.
  The RAM figures in §3 were measured on `h3-video` replicas.

- **In-pair TP cost.** The share of step time spent on the GPU↔GPU exchange
  has never been isolated. A 1-GPU vs 2-GPU replica comparison, or a profiler
  trace, would settle it. The batching comment in
  [`h3-video.toml`](../deploy/models/h3-video.toml) calls the deployment
  "PCIe-bound, not compute-bound"; that is an inference, not a measurement.
  The superlinear duration curve says compute dominates long clips, and short
  clips may carry a fixed per-step transfer component.
- **Residency speed benefit.** The VRAM cost per resident block was measured;
  the speed benefit of raising the count was not.
- **NUMA placement of the shared store.** `/dev/shm` pages land on the node of
  the process that first touches them. Replicas on the other socket may
  therefore stream across the inter-socket link. No per-replica timing was
  recorded to show whether that matters; the fleet-wide flatness in §5 suggests
  it is small.
- **Batched pairs at 15 s.** `--batching-max-size 2` was never verified on two
  largest-legal requests (2 x 26.7 GiB cannot fit unless most of the peak is
  shared).
- **Aspect ratio vs wall time.** VRAM is flat across ratios; wall time by ratio
  was never measured.
- **Stage-level timing.** Per-step and per-stage timings were logged only to
  the replicas' systemd journals (parsed live by `serve.stage_progress`). They
  were never persisted, and batch records keep wall time and peak VRAM, not
  the engine-reported `inference_time_s` the client receives.

## 8. Measurement provenance & method

| source                                                                                                                                | what it holds                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| batch records (`.deepme/batches/*/state.json`, `journal.jsonl`; gitignored, operator-local)                                           | per-clip wall time (`took_s`), start time, peak VRAM, adoption events |
| runs ledger (`.deepme/runs/`; gitignored)                                                                                             | single generations incl. engine-reported compute time                 |
| [`deploy/economics.toml`](../deploy/economics.toml) `[measured.*]`                                                                    | task, duration and steps curves used for pricing                      |
| [`deploy/prompts/envelope.jsonl`](../deploy/prompts/envelope.jsonl), [`duration-curve.jsonl`](../deploy/prompts/duration-curve.jsonl) | re-runnable measurement batches                                       |
| RUNBOOK §7.4 / §7.7, README                                                                                                           | replica RAM and shared-store measurements                             |

The corpus is ~1,300 completed generations, ~340 replica-hours. For the
concurrency tables:

- **Concurrency level.** Each clip's N is the time-weighted mean number of
  clips in flight during its lifetime, rounded.
- **Exclusions.** The tables leave out:
  - items whose last start was an **adoption** (re-attached after a restart,
    so wall time is a remainder);
  - runs under 60 s (deduplicated or adopted);
  - runs outside 0.6–1.6x the median for their length (queue wait or
    interruptions, since wall time keeps counting while a request waits).
- **Rows with 1–3 runs** come from the `duration-curve` batch and one-off
  generations. Full-machine rows for 4 s and 10 s are projections that rely on
  the flat-in-N result.

## Code map

| concern                              | where                                                                                                                                               |
| ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| H3 recipes, env, envelope notes      | [`deploy/models/h3-video.toml`](../deploy/models/h3-video.toml), [`h3-ref.toml`](../deploy/models/h3-ref.toml)                                      |
| weight placement / fit check         | [`src/deepme/engines.py`](../src/deepme/engines.py) `_sglang_weight_placement`, [`src/deepme/profiles.py`](../src/deepme/profiles.py)               |
| NUMA-packed GPU/CPU placement        | [`src/deepme/resources.py`](../src/deepme/resources.py) `pick_gpus`, `pick_cpus`                                                                    |
| shared weight store (deployed hook)  | [`src/deepme/hooks/deepme_hooks.py`](../src/deepme/hooks/deepme_hooks.py); admin: [`src/deepme/sharedstore.py`](../src/deepme/sharedstore.py)       |
| replica ports / rendezvous           | [`deploy/ports.toml`](../deploy/ports.toml)                                                                                                         |
| balancer / admission queue / gateway | [`src/deepme/balancer/`](../src/deepme/balancer/), [`src/deepme/gateway/`](../src/deepme/gateway/)                                                  |
| fleet mixes and pricing              | [`deploy/fleets/`](../deploy/fleets/), [`deploy/economics.toml`](../deploy/economics.toml), [`src/deepme/economics.py`](../src/deepme/economics.py) |
| batch runner and timing records      | [`src/deepme/batch.py`](../src/deepme/batch.py), [`src/deepme/media.py`](../src/deepme/media.py)                                                    |
