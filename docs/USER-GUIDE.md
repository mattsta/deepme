# API User Guide

Two OpenAI-compatible endpoints, same bearer key (`.deepme/api_key`):

- `http://<host>:8000/v1` — direct engine API, unguarded
- `http://<host>:8001/v1` — output-guarded proxy (forbidden-pattern firewall)

Model name = the profile's `served_name` (`deepme model list`). Sampling
defaults per model card: see the profile's `[sampling]`.

## Basic chat

```python
from openai import OpenAI

api = OpenAI(base_url="http://<host>:8000/v1", api_key=open(".deepme/api_key").read().strip())
r = api.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "hello"}],
    temperature=1.0,
    top_p=0.95,
)
print(r.choices[0].message.content)
print(r.usage.prompt_tokens_details.cached_tokens)  # prefix-cache hits
```

## Streaming

```python
for chunk in api.chat.completions.create(
    model="deepseek-v4-flash", messages=[...], stream=True, stream_options={"include_usage": True}
):
    delta = chunk.choices[0].delta if chunk.choices else None
    if delta and delta.content:
        print(delta.content, end="", flush=True)
```

Works identically through the guard (true incremental SSE; text is released
as soon as it's provably policy-clean).

## Tool calling

```python
r = api.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Weather in Tokyo?"}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ],
)
r.choices[0].message.tool_calls  # structured, JSON arguments
```

Reasoning content (when the model emits it) arrives in
`message.reasoning_content` / streaming `delta.reasoning_content`.

## Raw completions

`POST /v1/completions` with `prompt` — supported on both endpoints
(guarded via prompt-continuation on :8001).

## Video generation

Diffusion profiles serve a job API on their own port, not a chat API. A clip
takes minutes, so the request returns an id and you poll it.

```python
import httpx, time

key = open(".deepme/api_key").read().strip()
h = {"Authorization": f"Bearer {key}"}
base = "http://<host>:8030/v1/videos"

job = httpx.post(
    base,
    headers=h,
    timeout=3600,
    json={
        "model": "h3-video",
        "prompt": "a corgi surfing a wave at sunset, cinematic",
        "seconds": 4,
    },
).json()

while job["status"] not in ("completed", "failed"):
    time.sleep(5)
    job = httpx.get(f"{base}/{job['id']}", headers=h).json()

mp4 = httpx.get(f"{base}/{job['id']}/content", headers=h, timeout=600).content
open("out.mp4", "wb").write(mp4)
```

Send only the parameters you mean to override — each release ships tuned
defaults for resolution and step count. Accepted fields include `seconds`,
`size` (`1280x720`), `num_frames`, `fps`, `num_inference_steps`, `seed`, and
`negative_prompt`. `multipart/form-data` to the same route adds image
conditioning via `input_reference`.

The `model` field is informational: a diffusion server hosts exactly one
pipeline and dispatches every request to it.

MiniMax-H3 additionally accepts the fields from its model card — `task`
(`t2va` for text-only, `fl2va` with `conditions` for frame conditioning), a
`target` object (`short_edge`, `aspect_ratio`, `duration_seconds`),
`flow_shift` / `audio_flow_shift`, and `num_outputs_per_prompt` (1–10):

```jsonc
{
  "prompt": "three cats play tiny brass instruments, then exit",
  "task": "t2va",
  "conditions": [],
  "target": {
    "short_edge": 768,
    "aspect_ratio": "16:9",
    "duration_seconds": 5,
  },
  "num_inference_steps": 50,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 1101,
}
```

Output is H.264 video at 24fps with AAC stereo audio at 32kHz, in one MP4.

The CLI does the whole loop with a live progress bar (step count, rate and
ETA scraped from the engine — the job API itself only reports 0% then 100%),
saves the file, and records the run:

```bash
uv run deepme generate -m h3-video "a corgi surfing at sunset" --seconds 4
```

### Conditioning modes

H3 has three modes, and they are **not** all one deployment — the tasks are
split across two weight partitions:

| you want                            | task     | flags                                 | profile    |
| ----------------------------------- | -------- | ------------------------------------- | ---------- |
| text only                           | `t2va`   | _(none)_                              | `h3-video` |
| continue from an exact frame        | `fl2va`  | `--first-frame` and/or `--last-frame` | `h3-video` |
| guide by style/subject/motion/audio | `ref2va` | `--ref` (repeatable)                  | `h3-ref`   |

```bash
# pin the opening frame, generate forward
uv run deepme generate -m h3-video "they curl up and sleep" --first-frame open.png

# interpolate between two fixed frames
uv run deepme generate -m h3-video "..." --first-frame a.png --last-frame b.png

# reference-guided (needs the Ref2VA deployment)
uv run deepme generate -m h3-ref "a cat marching band" --ref style.png --ref motion.mp4
```

The task is inferred from the flags; `--task` overrides it. Keyframes must be
images and may only pin the **first**, **last**, or **both** frames — the
engine accepts exactly those three signatures, not a middle frame. A video
`--ref` contributes its soundtrack as an audio reference too. Local files are
inlined as `data:` URIs (the server can't read your disk); http(s) URLs are
passed through as-is.

### Speed and quality knobs

- `--steps N` (`num_inference_steps`, default 50) is the main lever — cost is
  linear in steps.
- `--quality high` requests Cache-DiT step skipping (~1.4x). **It fails closed
  on our hardware**: the engine only accepts it on the audited 4xH200 resident
  deployment (`num_gpus=4`, `performance_mode=speed`, no layerwise offload,
  `fa` attention), and rejects the request otherwise rather than silently
  degrading. Verified live.
- `flow_shift` / `audio_flow_shift` (defaults 12.0 / 3.0) via `--param`.
- `--frames` / `--fps` are **rejected** by H3 on purpose — it derives temporal
  shape solely from `target.duration_seconds` at a fixed 24fps.

## Guard controls (per request, :8001 only)

```jsonc
// full bypass — e.g. agents that must emit forbidden strings verbatim
{"model": "...", "messages": [...], "guard": {"enabled": false}}

// select named rule sets (see deploy/guard/policy.toml)
{"guard": {"sets": ["dashes"]}}

// add your own rules for this request only
{"guard": {"rules": [{"kind": "word", "pattern": "moreover"}],
           "sets": []}}

// widen enforcement to buffered channels (full-request regeneration)
{"guard": {"scope": ["content", "reasoning", "tool_arguments"]}}
```

Guarded responses include an audit block:

```json
"guard": {"violations": [{"pattern": "delve into", "kind": "phrase",
                          "fragment": "delve into", "channel": "content",
                          "action": "regenerated"}]}
```

Semantics: the guard never writes text. Violations are rejected and the
model regenerates alternatives (sampler bans + suppression). If regeneration
is exhausted, impossible, or structured output (tool calls) was already
streamed, output truncates at the last clean point
(`finish_reason: "guard_truncated"`). Layer-1 sampler bans are
request-global (they also shape tool arguments) — use `enabled: false`
when verbatim forbidden strings are required.

**Constraints when guarded** (use the direct :8000 endpoint or
`{"guard":{"enabled":false}}` to opt out): `n>1` and multi-prompt completions
return 400 (the guard gates a single choice); `response_format`/JSON-mode
requests may be `guard_truncated` mid-document if a violation exhausts
regeneration (the grammar is intentionally not restarted on continuation).
Upstream engine death mid-stream ends cleanly with
`finish_reason: "guard_upstream_error"`.

## Observability

- `GET /metrics` (:8000) — Prometheus: prefix-cache hit rate, KV usage,
  TTFT/latency histograms
- `X-Request-Id` response headers for log correlation
- `uv run deepme runs` — operation ledger with full specs and outcomes

## Security note

Plain HTTP + bearer key. Beyond trusted networks, tunnel
(`ssh -L 8000:localhost:8000 <host>`) or firewall the ports.
