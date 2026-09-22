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
    spec: RunSpec, client: httpx.AsyncClient, max_tokens: int
) -> StreamStats | None:
    payload = {
        "model": spec.served_name,
        "messages": [{"role": "user", "content": _BENCH_PROMPT}],
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
                    if (choice.get("delta") or {}).get("content") and ttft is None:
                        ttft = time.monotonic() - start
    except Exception as exc:  # noqa: BLE001 - a failed stream is a data point
        print(f"  stream failed: {type(exc).__name__}: {exc}")
        return None
    return StreamStats(
        ttft_s=ttft or 0.0, tokens=tokens, duration_s=time.monotonic() - start
    )


async def _run_level(spec: RunSpec, concurrency: int, max_tokens: int) -> dict:
    async with httpx.AsyncClient(timeout=1200) as client:
        wall_start = time.monotonic()
        results = await asyncio.gather(
            *[_one_stream(spec, client, max_tokens) for _ in range(concurrency)]
        )
        wall = time.monotonic() - wall_start
    good = [r for r in results if r is not None and r.tokens > 0]
    if not good:
        return {"concurrency": concurrency, "error": "all streams failed"}
    total_tokens = sum(r.tokens for r in good)
    per_stream_tps = [
        (r.tokens - 1) / (r.duration_s - r.ttft_s)
        for r in good
        if r.duration_s > r.ttft_s and r.tokens > 1
    ]
    return {
        "concurrency": concurrency,
        "completed": len(good),
        "wall_s": round(wall, 1),
        "ttft_s_median": round(statistics.median(r.ttft_s for r in good), 3),
        "per_stream_tps_median": round(statistics.median(per_stream_tps), 1)
        if per_stream_tps
        else 0,
        "aggregate_output_tps": round(total_tokens / wall, 1),
        "total_output_tokens": total_tokens,
    }


def run(spec: RunSpec, concurrencies: list[int], max_tokens: int = 1024) -> list[dict]:
    rows = []
    for c in concurrencies:
        print(f"benchmarking concurrency={c} max_tokens={max_tokens} ...")
        row = asyncio.run(_run_level(spec, c, max_tokens))
        print(f"  {row}")
        rows.append(row)
    return rows
