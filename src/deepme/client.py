"""OpenAI-compatible API client helpers, driven by a RunSpec."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from . import config
from .profiles import RunSpec


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key()}"}


@dataclass
class ChatResult:
    text: str
    ttft_s: float
    total_s: float
    completion_tokens: int
    prompt_tokens: int
    cached_tokens: int = 0

    @property
    def decode_tps(self) -> float:
        decode_time = self.total_s - self.ttft_s
        if decode_time <= 0 or self.completion_tokens <= 1:
            return 0.0
        return (self.completion_tokens - 1) / decode_time


def chat(
    spec: RunSpec,
    prompt: str,
    *,
    max_tokens: int = 512,
    timeout: float = 600,
) -> ChatResult:
    """Streaming chat completion with client-side TTFT / tok/s measurement.

    Sampling defaults come from the model profile.
    """
    payload = {
        "model": spec.served_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text_parts: list[str] = []
    ttft = None
    completion_tokens = prompt_tokens = cached_tokens = 0
    start = time.monotonic()
    with (
        httpx.Client(timeout=timeout) as client,
        client.stream(
            "POST",
            f"{spec.api_base}/v1/chat/completions",
            json=payload,
            headers=_headers(),
        ) as resp,
    ):
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                details = usage.get("prompt_tokens_details") or {}
                cached_tokens = details.get("cached_tokens", cached_tokens)
            for choice in chunk.get("choices", []):
                delta = (choice.get("delta") or {}).get("content")
                if delta:
                    if ttft is None:
                        ttft = time.monotonic() - start
                    text_parts.append(delta)
    total = time.monotonic() - start
    return ChatResult(
        text="".join(text_parts),
        ttft_s=ttft if ttft is not None else total,
        total_s=total,
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
    )


def tool_call_test(spec: RunSpec) -> dict:
    """Send a canned tool-enabled request; return the parsed tool_calls and
    reasoning fields so parser wiring can be verified end-to-end."""
    payload = {
        "model": spec.served_name,
        "messages": [
            {
                "role": "user",
                "content": "What's the weather in Tokyo right now? Use the tool.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "max_tokens": 1024,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
    }
    r = httpx.post(
        f"{spec.api_base}/v1/chat/completions",
        json=payload,
        headers=_headers(),
        timeout=300,
    )
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    return {
        "finish_reason": r.json()["choices"][0].get("finish_reason"),
        "tool_calls": msg.get("tool_calls"),
        "reasoning_present": bool(msg.get("reasoning_content") or msg.get("reasoning")),
        "content": (msg.get("content") or "")[:200],
    }


def models(spec: RunSpec) -> list[str]:
    r = httpx.get(f"{spec.api_base}/v1/models", headers=_headers(), timeout=10)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]
