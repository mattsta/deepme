# deepme

Config-driven infrastructure CLI for serving large models on a dedicated
multi-GPU server (reference: 8x RTX 5090 32GB, PCIe-only, 500GB RAM).

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

## Browsing & picking models interactively

```sh
uv run deepme browse                # top 100 trending HF models, filtered
uv run deepme browse --all          # include excluded rows with reasons
```

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
JSON results in `.deepme/bench-*.json`.

## Using the API

OpenAI-compatible at `http://<host>:<port>/v1`, bearer key in
`.deepme/api_key` (auto-generated, gitignored). Model name = the profile's
`served_name`. Responses include `usage.prompt_tokens_details.cached_tokens`;
`/metrics` exposes Prometheus counters (cache hit rate, KV usage, latency
histograms). Plain HTTP — for anything beyond trusted networks use an SSH
tunnel or firewall the port.

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://<host>:8000/v1", api_key=open(".deepme/api_key").read().strip()
)
```

## Throughput estimates (`~tok/s` column)

`src/deepme/perfmodel.py`: single-stream decode is memory-bandwidth-bound, so
`est ~= effective_bandwidth / (weight_bytes x active_fraction)`. Active
fraction comes from each repo's config.json (expert count, top-k, layer dims —
dense = 1.0). Bandwidth tiers: single-GPU ~1.25TB/s; multi-GPU MoE (expert
parallel) **calibrated to our production measurement** (within ~10%);
multi-GPU dense over PCIe is estimated pending a dense-TP bench. Heuristic
for ranking (~2x confidence), not a benchmark — `deepme bench` is ground
truth and auto-records to the runs ledger. Weight sizes always prefer actual
file bytes over dtype arithmetic (sub-byte packings like FP4 experts break
param-count math).

## Quantization policy

`src/deepme/policy.py` is the single allow/deny gate for weight
representations. Defaults: full-width formats and GGUF quants **>= 6 bits**
(Q6_K, Q8_0, F16/BF16) are servable; 4-bit and below are excluded;
_unclassifiable quants count as unsupported_ (unknown is never silently
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
| /v1/completions & other endpoints | passed through | transparent proxy (guard covers chat completions) |

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
2. **Streaming gate** (`src/deepme/guard/gate.py`, pure logic, 21 unit tests):
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

Registered: `vllm` (safetensors), `sglang` (safetensors),
`llamacpp` (gguf, **experimental** — CUDA source build via
`deepme setup -e llamacpp`; unlocks the GGUF quantization universe).

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
  (`deepme config init`, gitignored).
- Default model profile: `deepme model use <name>` (same file).
- Engine dependencies: uv projects in `deploy/engines/<engine>/pyproject.toml`,
  synced + `uv sync`ed by `deepme setup`. All dependency changes go through
  those files — never ad-hoc installs.
- API key: `.deepme/api_key`, auto-generated.

## Commands

| command                                                       | what it does                                                                                              |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `model list\|show\|use\|add\|refresh`                         | manage profiles (facts from HF API)                                                                       |
| `browse [--all] [--limit N]`                                  | interactive picker: HF trending x engine support x cluster fit                                            |
| `runs` / `runs show <id>`                                     | deployment/bench ledger (history, full specs, outcomes)                                                   |
| `config init\|show`                                           | target server (local, gitignored)                                                                         |
| `status`                                                      | fleet view: GPUs, disk, all units, per-profile API health                                                 |
| `gpu-fix`                                                     | driver stack: -open module, version alignment, reload                                                     |
| `setup [-e ...]`                                              | build tools + engine uv projects (idempotent)                                                             |
| `versions`                                                    | engine/toolchain versions per project                                                                     |
| `gpus` / `reclaim`                                            | GPU processes / kill non-deepme GPU processes                                                             |
| `switch <profile>`                                            | one-command model switch: fit-check → auto-download → stop conflicts → deploy → health → completion proof |
| `download start\|restart\|status [-m P]`                      | resumable weight download                                                                                 |
| `serve start\|stop\|restart\|status\|wait\|diag\|logs [-m P]` | service lifecycle                                                                                         |
| `doctor [-e engine]`                                          | probe engine-critical imports, full tracebacks                                                            |
| `jit-heal`                                                    | serially finish raced flashinfer JIT builds                                                               |
| `test [-m P] [--pad-tokens N]`                                | one completion + timing + cache usage                                                                     |
| `test-tools [-m P]`                                           | verify structured tool_calls/reasoning output                                                             |
| `bench [-m P] -c N... [--out f.json]`                         | TTFT + throughput sweep                                                                                   |
| `reboot`                                                      | reboot server, wait for SSH (last resort)                                                                 |
| `ssh "cmd"`                                                   | escape hatch (one-time inspection only)                                                                   |

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
