# API User Guide

Three OpenAI-compatible endpoints:

- `http://<host>:8080/v1` — the **gateway**: the platform entry point.
  Routes by model + task, queues when the fleet is busy, survives fleet
  resizes. All video/service traffic goes here.
- `http://<host>:8000/v1` — direct text-LLM engine API, unguarded (bearer
  key `.deepme/api_key`)
- `http://<host>:8001/v1` — output-guarded text proxy (forbidden-pattern
  firewall, same bearer key)

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

Video is a job API, not a chat API. A clip takes minutes, so the request
returns an id and you poll it. Talk to the **gateway (:8080)** — it routes
on `model` + `task`, load-balances across replicas, queues when everything
is busy, and keeps your job reachable across fleet resizes. (Hitting a
replica port directly bypasses all of that.)

```python
import httpx, time

key = open(".deepme/api_key").read().strip()
h = {"Authorization": f"Bearer {key}"}
base = "http://<host>:8080/v1/videos"

r = httpx.post(
    base,
    headers=h,
    timeout=120,
    json={
        "model": "h3-video",
        "prompt": "a corgi surfing a wave at sunset, cinematic",
        "seconds": 4,
    },
)
job = r.json()

while job["status"] not in ("completed", "failed", "cancelled", "error"):
    time.sleep(5)
    job = httpx.get(f"{base}/{job['id']}", headers=h).json()

mp4 = httpx.get(f"{base}/{job['id']}/content", headers=h, timeout=600).content
open("out.mp4", "wb").write(mp4)
```

**When the fleet is busy you get `202` instead of `200`**, and the id is a
`q_` queue ticket rather than a `vid_` job id. Poll it exactly the same way
— while it waits the body carries `queue_position`, `accepting_replicas`
and `billable: false`;
once a replica takes it, the same id transparently follows the real job
(including `/content`), and the body's `id` switches to the real `vid_` id.
`DELETE /v1/queue/{ticket}` abandons an unstarted ticket free of charge;
`?queue=false` on the POST makes a busy fleet answer `503` instead of
queueing. Full protocol: [INTERNAL-API.md](INTERNAL-API.md) §4.

**Queueing is unbounded on purpose — `accepting_replicas` is how you tell
waiting from stranded.** A busy fleet and a dead fleet both queue you, and
both answer `200 {"status": "queued"}`, so elapsed time alone can never
distinguish "next in line behind a 15-minute clip" from "nothing is running
and nothing will be". The two are distinguished by one field:

| `queue_position` | `accepting_replicas` | meaning                                        |
| ---------------- | -------------------- | ---------------------------------------------- |
| any              | `> 0`                | genuinely queued — wait, however long it takes |
| any              | `0`                  | **stranded**: no replica can start this, ever  |

Do not put a timeout on a queued ticket; time out on sustained
`accepting_replicas: 0` instead. The balancer applies the same rule from its
side and fails waiting tickets after ~5 minutes of zero accepting replicas
(`FLEET_DEAD_STRIKES`, reset by any replica returning), so a stranded ticket
ends as `status: "failed"`, `billable: false` with an error naming the fleet
state. It was never submitted, so resubmitting when capacity returns costs
nothing. The CLI does all of this for you and additionally **refuses to
submit at all** when the fleet cannot serve the request.

**Fetch, then ACK.** The server keeps your artifact until you confirm you
have it:

```python
mp4 = httpx.get(f"{base}/{job['id']}/content", headers=h, timeout=600).content
open("out.mp4", "wb").write(mp4)  # persist it FIRST
httpx.delete(f"{base}/{job['id']}/content", headers=h)  # then ack
```

`DELETE .../content` means "I have these bytes, reclaim your copy". It is
idempotent (retry a lost ack freely) and is **not** a cancellation — `DELETE`
on the job itself cancels work; `DELETE` on `/content` only frees the file.

Until you ack, `/content` is re-readable, so a failed write is recoverable:
retry the GET. The retention clock starts at FIRST DELIVERY regardless, so a
client that dies mid-write cannot strand a file — it is reclaimed 1 hour
after first delivery, or 6 hours if never fetched at all. Ack promptly
anyway: an unacked artifact is disk you are still holding.

Do not skip the ack "because retention will get it": the window exists for
crashes, not as a substitute. The CLI does this for you, and only after the
bytes are `fsync`ed and renamed into place.

Send only the parameters you mean to override — each release ships tuned
defaults for resolution and step count. The accepted field set is per model
card; for the deployed MiniMax-H3 profiles the validator accepts exactly
`{task, prompt, conditions, target, flow_shift, audio_flow_shift, seed,
num_inference_steps}` (plus the CLI-level `seconds`, which sets
`target.duration_seconds`) and **silently discards everything else** —
including `size`, `fps` and `num_frames`. Image conditioning is a JSON
`conditions` array of `data:` URIs or http(s) URLs (see Conditioning modes
below); there is no multipart upload.

The `model` field matters at the gateway: it routes on `model` + `task` and
answers `400` when it cannot resolve them. (Only on a direct single-pipeline
replica hit is it informational.)

MiniMax-H3's card fields — `task` (`t2va` for text-only, `fl2va` with
`conditions` for frame conditioning), a `target` object (`short_edge`,
`aspect_ratio`, `duration_seconds`), and `flow_shift` / `audio_flow_shift`:

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
  "num_inference_steps": 49,
  "flow_shift": 12.0,
  "audio_flow_shift": 3.0,
  "seed": 1101,
}
```

Output is H.264 video at 24fps with AAC stereo audio at 32kHz, in one MP4.

The CLI does the whole loop with a live progress bar, saves the file, and
records the run. API callers get the same stage detail from
`GET /v1/videos/{id}/events` (SSE per-job progress) or the platform-wide
`/v1/jobs` — the bare poll body itself only reports 0% then 100%:

```bash
uv run deepme generate -m h3-video "a corgi surfing at sunset" --seconds 4
```

**Always pass `-m` for video.** `deepme model use` sets ONE default profile
shared by the text commands and the video commands, and `switch` moves it —
so whichever was deployed last wins, and a bare `deepme generate` inherits an
LLM profile and refuses. `generate`, `batch`, `story` and `showcase` now fail
immediately and name the video profiles instead of failing per item.

Before submitting, those commands verify against `/v1/capacity` that a fleet
serves the profile, is reachable, has at least one replica **accepting**, and
serves the requested task — and refuse otherwise, because a submission a dead
fleet accepts becomes a queue ticket that waits indefinitely (above).
`--direct` skips the check: it deliberately bypasses the balancer, so fleet
capacity says nothing about it.

Cost control: `--steps` is the main lever and dominates wall time — the
default 49 is roughly 15 minutes for a 5-second clip on this hardware, while
`--steps 16` lands in a few minutes. Drop steps first when iterating.

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

- `--steps N` (`num_inference_steps`, default 49: 50 sigma points → 49
  denoise iterations) is the main lever — cost is roughly linear in steps.
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

Platform (gateway :8080 — see [INTERNAL-API.md](INTERNAL-API.md) §7 for
shapes):

- `GET /v1/capacity` — free slots, queue depth and ETA per task, live
- `GET /v1/jobs` — every running/queued/recent job with its parameters
- `GET /v1/events` — SSE firehose of every lifecycle transition
- `GET /v1/videos/{id}/events` — per-job progress SSE (stages, position)
- `DELETE /v1/videos/{id}/content` — artifact ack: reclaim a delivered file
- `GET /v1/pricing` — the deployed price table with derivation
- `GET /metrics` — Prometheus text, aggregated across fleets with
  `fleet="..."` labels

Text engine (:8000):

- `GET /metrics` — Prometheus: prefix-cache hit rate, KV usage,
  TTFT/latency histograms
- `X-Request-Id` response headers for log correlation

Operator: `uv run deepme top` (one-screen live view),
`uv run deepme runs` — operation ledger with full specs and outcomes.

## Security note

Plain HTTP. The text endpoints (:8000/:8001) check the bearer key; the
gateway and balancers (:8080, :9100/:9200) bind `0.0.0.0` and **inject the
platform's key when a caller sends none** — the video path is effectively
unauthenticated at those layers. Beyond trusted networks, tunnel
(`ssh -L 8080:localhost:8080 <host>`, same for 8000/8001) or firewall all
of 8000, 8001, 8080 and 9100/9200.
