# deepme

Infrastructure CLI for serving **DeepSeek-V4-Flash-0731** (304B-class MoE, FP8, 1M
context, DSpark speculative decoding) on a dedicated multi-GPU server
(reference: 8x RTX 5090 32GB, PCIe-only, 2 NUMA nodes, 500GB RAM).

All operations run from this repo via `uv run deepme <command>` over SSH. The
inference service is a systemd unit (`deepme-llm`) exposing an OpenAI-compatible
API. No server identities live in the repo — the target machine is configured
locally (gitignored) or via environment variables.

## Cold start → serving (fresh system runbook)

```sh
uv sync                                                  # local CLI
uv run deepme config init --host <ip> --user <ssh-user> --port 8000
uv run deepme gpu-fix        # NVIDIA driver stack: -open kernel module (REQUIRED
                             # for RTX 5090/Blackwell), userland/module version
                             # alignment, live module reload when possible
uv run deepme setup          # ninja/g++ (for JIT kernel builds) + engine uv
                             # projects synced to ~/deepme-runtime/{vllm,sglang}
uv run deepme download start # 167GB FP8 weights, tmux + hf_transfer, resumable
uv run deepme download status            # ...until 48/48 shards
uv run deepme serve start                # production config (see below)
uv run deepme test "hello"               # real completion + TTFT/tok-s
uv run deepme bench -c 1 -c 2 -c 4      # benchmark
```

Every step is idempotent — rerun freely. `status` shows the whole picture at
any time; `serve diag` prints restart counts + the log window before the last
crash; `doctor` probes engine-critical imports with full tracebacks; `jit-heal`
serially finishes flashinfer JIT builds if ranks ever race.

## Production configuration

```sh
uv run deepme serve start --engine vllm --no-spec --max-len 1048576 \
  --gpu-mem 0.94 --kv-ram 200 --extra=--max-num-seqs --extra=8
```

vLLM 0.26.0, TP=8 + expert parallel, FP8 weights + FP8 KV cache, **full
1M-token context** (GPU KV pool: 1,287,511 tokens), automatic prefix caching,
KV tiered into 200GB system RAM (OffloadingConnector: evicted prefixes spill
to RAM and restore on cache hits). Tuned for low-batch single-user usage
(`--max-num-seqs 8`); costs nothing vs the 524K config at c=1-4.

### `serve start` options

| flag          | default | meaning                                                                     |
| ------------- | ------- | --------------------------------------------------------------------------- |
| `--engine`    | vllm    | `vllm` or `sglang`                                                          |
| `--tp`        | 8       | tensor-parallel size                                                        |
| `--max-len`   | 262144  | per-request context ceiling (model supports 1M)                             |
| `--gpu-mem`   | 0.90    | GPU memory fraction (higher → more KV, less headroom)                       |
| `--kv-ram N`  | 0       | tier KV into N GB of system RAM (vllm OffloadingConnector / sglang HiCache) |
| `--no-spec`   | off     | disable DSpark speculative decoding                                         |
| `--extra ARG` | —       | raw engine args, repeatable                                                 |

## Benchmarks (512-token generations, 2026-08-08)

vLLM production config (524K ctx, RAM tier, no spec decode):

| concurrency | TTFT median | per-stream tok/s | aggregate tok/s |
| ----------- | ----------- | ---------------- | --------------- |
| 1           | 0.27s       | 87.6             | 84              |
| 2           | 0.38s       | 77.5             | 146             |
| 4           | 0.39s       | 72.6             | 275             |
| 16          | 5.6s        | 43.3             | 470             |

Long-context (1M config): 100K-token prompt prefills in 71.5s (~1,400 tok/s,
PCIe-TP bound); an identical repeated prompt hits the prefix cache and answers
with **TTFT 1.2s**. JSON results in `.deepme/bench-*.json`.

## Engine status on SM120 (RTX 5090)

- **vLLM 0.26.0 — production.** Works with `--no-spec`. Its DSpark speculative
  decoding has an SM120 kernel-dispatch bug ("num_tokens > 64 (7 vs. 64)").
- **sglang 0.5.17 — blocked.** Starts, loads, compiles, then the DSpark draft
  autotune hits a missing SM120 sparse-MLA kernel shape ("Unsupported
  sparse-MLA prefill configuration num_heads=64 topk=192"). Not a config issue.
- Spec decode is worth ~2-3x single-stream decode; retry both engines on
  upgrades: bump versions in `deploy/engines/*/pyproject.toml`, `deepme setup`,
  `deepme serve start`.

## Hard-won toolchain facts (encoded in provisioning — do not undo)

1. **RTX 5090 requires the NVIDIA `-open` kernel module**; the proprietary
   module fails `RmInitAdapter (0x22)`. `gpu-fix` enforces this.
2. **No system CUDA toolkit.** The engine venv's pip CUDA toolkit is the only
   toolchain; a second one of a different version poisons JIT auto-detection
   (deep_gemm/tilelang/tvm_ffi each detect CUDA differently).
3. **The pip `nvidia/cu13` dir is merged from wheels at different versions**
   (nvcc 13.3 + cuda-runtime 13.0 as shipped) → every JIT compiler dies on
   CCCL's "compiler and toolkit headers are incompatible". The engine
   pyprojects override `nvidia-cuda-runtime`/`nvidia-cuda-nvrtc` to nvcc's minor.
4. **flashinfer never JIT-compiles at runtime**: `flashinfer-jit-cache` (pinned
   to the engine's flashinfer version, index `flashinfer.ai/whl/cu130`) ships
   prebuilt kernels.
5. The cu13 wheel needs `lib64 → lib` and unversioned `.so` symlinks for JIT
   linking (`-lcudart`); post-sync fixups create them.
6. This OS's glibc is too new for CUDA < 13.3 (`rsqrt noexcept` clash) — don't
   downgrade the toolkit.

## Configuration

Precedence: `DEEPME_HOST`/`DEEPME_USER`/`DEEPME_PORT` env vars →
`.deepme/config.toml` (`deepme config init`, gitignored) → defaults. The API
bearer key auto-generates into `.deepme/api_key` (gitignored).

Engine dependencies are uv projects in `deploy/engines/<engine>/pyproject.toml`,
synced to the server and materialized with `uv sync` by `deepme setup`. All
dependency changes go through those files — never ad-hoc installs.

## Commands

| command                                                     | what it does                                          |
| ----------------------------------------------------------- | ----------------------------------------------------- |
| `config init\|show`                                         | set/inspect target server (local, gitignored)         |
| `status`                                                    | GPUs, disk, model files, systemd state, API health    |
| `gpu-fix`                                                   | driver stack: -open module, version alignment, reload |
| `setup [-e ...]`                                            | build tools + engine uv projects (idempotent)         |
| `versions`                                                  | engine/toolchain versions per project                 |
| `gpus` / `reclaim`                                          | GPU processes / kill non-service GPU processes        |
| `download start\|restart\|status`                           | resumable weight download                             |
| `serve start\|stop\|restart\|status\|wait\|diag\|logs [-f]` | service lifecycle                                     |
| `doctor [-e engine]`                                        | probe engine-critical imports, full tracebacks        |
| `jit-heal`                                                  | serially finish raced flashinfer JIT builds           |
| `test "prompt"`                                             | one streaming completion + timing                     |
| `bench -c N... [--out f.json]`                              | TTFT + throughput sweep                               |
| `reboot`                                                    | reboot server, wait for SSH (last resort)             |
| `ssh "cmd"`                                                 | escape hatch (one-time inspection only)               |

## Layout

- `src/deepme/config.py` — single source of truth (server config, model, paths)
- `src/deepme/remote.py` — SSH exec layer (ControlMaster, stdin scripts, timeouts)
- `src/deepme/provision.py` — driver stack + engine uv-project sync
- `src/deepme/download.py` / `gpus.py` / `doctor.py` — weights, GPUs, diagnostics
- `src/deepme/serve.py` — engine CLI builders + systemd unit management
- `src/deepme/client.py` / `bench.py` — API client and load benchmark
- `src/deepme/cli.py` — typer CLI wiring
- `deploy/engines/*/pyproject.toml` — engine dependency projects (uv-managed)
