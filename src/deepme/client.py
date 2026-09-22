"""OpenAI-compatible API client helpers for testing the deployed model."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from . import config


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key()}"}


@dataclass
class ChatResult:
    text: str
    ttft_s: float
    total_s: float
    completion_tokens: int
    prompt_tokens: int

    @property
    def decode_tps(self) -> float:
        decode_time = self.total_s - self.ttft_s
        if decode_time <= 0 or self.completion_tokens <= 1:
            return 0.0
        return (self.completion_tokens - 1) / decode_time


def chat(
    prompt: str,
    *,
    max_tokens: int = 512,
    temperature: float = config.DEFAULT_TEMPERATURE,
    top_p: float = config.DEFAULT_TOP_P,
    timeout: float = 600,
) -> ChatResult:
    """Streaming chat completion with client-side TTFT / tok/s measurement."""
    payload = {
        "model": config.SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    text_parts: list[str] = []
    ttft = None
    completion_tokens = 0
    prompt_tokens = 0
    start = time.monotonic()
    with httpx.Client(timeout=timeout) as client, client.stream(
        "POST",
        f"{config.API_BASE}/v1/chat/completions",
        json=payload,
        headers=_headers(),
    ) as resp:
        resp.raise_for_status()
        import json as _json

        for line in resp.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = _json.loads(data)
            usage = chunk.get("usage")
            if usage:
                completion_tokens = usage.get(
                    "completion_tokens", completion_tokens
                )
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
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
    )


def models() -> list[str]:
    r = httpx.get(f"{config.API_BASE}/v1/models", headers=_headers(), timeout=10)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", [])]
