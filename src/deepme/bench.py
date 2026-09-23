"""Load benchmark against the deployed OpenAI-compatible endpoint.

Measures TTFT and decode throughput per stream plus aggregate output
tokens/sec at several concurrency levels — enough to compare engine
configurations (spec-decode on/off, sglang vs vllm) apples-to-apples.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx

from . import config
from .profiles import RunSpec

# A prompt that forces long, non-degenerate output without a huge input.
_BENCH_PROMPT = (
    "Write a detailed technical essay on the history of instruction set "
    "architectures, covering CISC vs RISC, SIMD evolution, and modern "
    "accelerator ISAs. Be thorough and keep going until told to stop."
)


@dataclass
class StreamStats:
    ttft_s: float
    tokens: int
    duration_s: float


async def _one_stream(
    spec: RunSpec, client: httpx.AsyncClient, max_tokens: int, nonce: str
) -> StreamStats | None:
    # A unique nonce per stream defeats the server's prefix cache so every
    # benchmark measures real prefill (identical prompts would share KV and
    # report fictitiously fast TTFT). --no-nonce (nonce="") measures the
    # cached path deliberately.
    content = (f"[{nonce}] " if nonce else "") + _BENCH_PROMPT
    payload = {
        "model": spec.served_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.monotonic()
    ttft = None
    tokens = 0
    try:
        async with client.stream(
            "POST",
            f"{spec.api_base}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {config.api_key()}"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    tokens = usage["completion_tokens"]
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    # First token of ANY channel marks time-to-first-token
                    # (reasoning models emit reasoning_content before content).
                    if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                        ttft = time.monotonic() - start
    except Exception as exc:
        print(f"  stream failed: {type(exc).__name__}: {exc}")
        return None
    if ttft is None:
        return StreamStats(ttft_s=-1.0, tokens=tokens, duration_s=time.monotonic() - start)
    return StreamStats(ttft_s=ttft, tokens=tokens, duration_s=time.monotonic() - start)


async def _run_level(spec: RunSpec, concurrency: int, max_tokens: int, nonce_prefix: str) -> dict:
    async with httpx.AsyncClient(timeout=1200) as client:
        wall_start = time.monotonic()
        results = await asyncio.gather(
            *[
                _one_stream(spec, client, max_tokens, f"{nonce_prefix}-{i}" if nonce_prefix else "")
                for i in range(concurrency)
            ]
        )
        wall = time.monotonic() - wall_start
    good = [r for r in results if r is not None and r.tokens > 0]
    if not good:
        return {"concurrency": concurrency, "error": "all streams failed"}
    total_tokens = sum(r.tokens for r in good)
    # TTFT only from streams that produced a first token (ttft>=0); count
    # excluded ones honestly rather than folding a 0.0 sentinel into the median.
    ttfts = [r.ttft_s for r in good if r.ttft_s >= 0]
    per_stream_tps = [
        (r.tokens - 1) / (r.duration_s - r.ttft_s)
        for r in good
        if r.ttft_s >= 0 and r.duration_s > r.ttft_s and r.tokens > 1
    ]
    return {
        "concurrency": concurrency,
        "completed": len(good),
        "no_ttft": len(good) - len(ttfts),
        "wall_s": round(wall, 1),
        "ttft_s_median": round(statistics.median(ttfts), 3) if ttfts else None,
        "per_stream_tps_median": round(statistics.median(per_stream_tps), 1)
        if per_stream_tps
        else 0,
        "aggregate_output_tps": round(total_tokens / wall, 1),
        "total_output_tokens": total_tokens,
    }


def run(
    spec: RunSpec, concurrencies: list[int], max_tokens: int = 1024, nonce: bool = True
) -> tuple[list[dict], bool]:
    """Returns (rows, all_ok). all_ok is False if any level had zero completed
    streams — the caller sets exit code / ledger outcome from it."""
    import secrets

    rows, all_ok = [], True
    for c in concurrencies:
        print(f"benchmarking concurrency={c} max_tokens={max_tokens} nonce={nonce} ...")
        prefix = secrets.token_hex(4) if nonce else ""
        row = asyncio.run(_run_level(spec, c, max_tokens, prefix))
        print(f"  {row}")
        if "error" in row:
            all_ok = False
        rows.append(row)
    return rows, all_ok
