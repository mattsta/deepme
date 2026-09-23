"""Engine adapters for the output guard: API mechanics per backend.

Single point of responsibility: HOW to express a continuation, a sampler
ban, or token suppression for a given engine lives here and nowhere else.
WHICH capabilities an engine has is declared centrally (deepme engines
registry) and passed in as GUARD_FEATURES — an adapter method only runs
when its feature is enabled, so the guard loop degrades per-engine without
a single engine-name conditional in server.py:

  continuation missing  -> violations truncate at the clean boundary
  bad_words missing     -> no preemptive bans (gate still guarantees output)
  tokenize missing      -> no logit-bias suppression on retries

Deployed next to server.py; stdlib + httpx only.
"""

from __future__ import annotations


class EngineAdapter:
    """Base adapter: OpenAI-compatible generic backend (conservative)."""

    def __init__(self, client, features: set[str]):
        self.client = client
        self.features = features

    # -- capability-gated operations ------------------------------------

    # Structured-output / grammar params must NOT be re-applied to a
    # continuation: they restart the grammar mid-document (e.g. JSON mode
    # forces a fresh {...}, producing invalid output). Stripped from every
    # continuation request.
    _CONT_STRIP = (
        "response_format",
        "guided_json",
        "guided_regex",
        "guided_grammar",
        "guided_choice",
        "guided_decoding",
        "tool_choice",
        "n",
    )

    def _strip_structured(self, d: dict) -> dict:
        for k in self._CONT_STRIP:
            d.pop(k, None)
        return d

    def continuation_body(self, body: dict, accepted: str) -> dict | None:
        """Request that resumes the accepted partial output; None if the
        backend can't continue (guard then truncates at the clean boundary)."""
        if "continuation" not in self.features:
            return None
        cont = self._strip_structured(dict(body))
        cont["messages"] = [*list(body["messages"]), {"role": "assistant", "content": accepted}]
        cont["continue_final_message"] = True
        cont["add_generation_prompt"] = False
        # Shrink the remaining budget by what's already accepted (~4 chars/tok)
        # so retries can't multiply the client's token budget.
        if isinstance(body.get("max_tokens"), int):
            cont["max_tokens"] = max(16, body["max_tokens"] - len(accepted) // 4)
        return cont

    def apply_bans(self, body: dict, patterns: list[str]) -> dict:
        """Attach sampler-level sequence bans if the backend supports them."""
        if "bad_words" not in self.features or not patterns:
            return body
        out = dict(body)
        out["bad_words"] = list(body.get("bad_words") or []) + patterns
        return out

    async def suppression_ids(self, model: str, fragment: str, auth: str) -> list[int]:
        """Token ids to bias away on retry (space/case surface forms)."""
        if not {"tokenize", "logit_bias"} <= self.features:
            return []
        ids = []
        variants = {
            fragment,
            " " + fragment.lstrip(),
            fragment.capitalize(),
            " " + fragment.lstrip().capitalize(),
        }
        for v in variants:
            try:
                r = await self.client.post(
                    "/tokenize",
                    json={"model": model, "prompt": v, "add_special_tokens": False},
                    headers={"Authorization": auth},
                )
                toks = r.json().get("tokens") or []
                if toks:
                    ids.append(toks[0])
            except Exception:
                continue
        return ids

    def completions_continuation(self, body: dict, accepted: str) -> dict:
        """Raw completions continuation is universal: same request, prompt
        extended by the accepted output. No special flags, no capability."""
        cont = self._strip_structured(dict(body))
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        cont["prompt"] = prompt + accepted
        if isinstance(body.get("max_tokens"), int):
            # Shrink the remaining budget by what's already accepted (~4
            # chars/token) so retries can't multiply the client's budget.
            used_est = len(accepted) // 4
            cont["max_tokens"] = max(16, body["max_tokens"] - used_est)
        return cont

    async def stream_completions(self, body: dict, auth: str):
        """Streamed /v1/completions: yields same typed events as stream()."""
        import json as _json

        b = dict(body)
        b["stream"] = True
        b["stream_options"] = {"include_usage": True}
        async with self.client.stream(
            "POST",
            "/v1/completions",
            json=b,
            headers={"Authorization": auth},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                chunk = _json.loads(data)
                if chunk.get("model"):
                    yield ("model", chunk["model"])
                if chunk.get("usage"):
                    yield ("usage", chunk["usage"])
                for choice in chunk.get("choices", []):
                    if choice.get("text"):
                        yield ("content", choice["text"])
                    if choice.get("finish_reason"):
                        yield ("finish", choice["finish_reason"])

    async def stream(self, body: dict, auth: str):
        """ALWAYS stream from the backend (never block on a full generation);
        yield typed events parsed from upstream SSE:

            ("content", str)         assistant text delta
            ("reasoning", str)       reasoning_content delta
            ("tool_calls", list)     raw tool_call delta fragments
            ("model", str)
            ("finish", str)          finish_reason
            ("usage", dict)

        The generator stops cleanly when the caller breaks (violation abort:
        the underlying connection is closed, upstream request cancelled)."""
        import json as _json

        b = dict(body)
        b["stream"] = True
        b["stream_options"] = {"include_usage": True}
        async with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json=b,
            headers={"Authorization": auth},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                chunk = _json.loads(data)
                if chunk.get("model"):
                    yield ("model", chunk["model"])
                if chunk.get("usage"):
                    yield ("usage", chunk["usage"])
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield ("content", delta["content"])
                    if delta.get("reasoning_content"):
                        yield ("reasoning", delta["reasoning_content"])
                    if delta.get("tool_calls"):
                        yield ("tool_calls", delta["tool_calls"])
                    if choice.get("finish_reason"):
                        yield ("finish", choice["finish_reason"])


class VllmAdapter(EngineAdapter):
    """vLLM implements every guard-relevant operation via the base shapes."""


class SglangAdapter(EngineAdapter):
    """sglang: continuation + logit_bias; no bad_words, no /tokenize (until a
    tokenizer adapter exists, suppression_ids returns [] via feature gate)."""


class LlamaCppAdapter(EngineAdapter):
    """llama.cpp server: no chat continuation -> replacement-only guard."""


ADAPTERS: dict[str, type[EngineAdapter]] = {
    "vllm": VllmAdapter,
    "sglang": SglangAdapter,
    "llamacpp": LlamaCppAdapter,
}


def make_adapter(engine: str, client, features: set[str]) -> EngineAdapter:
    return ADAPTERS.get(engine, EngineAdapter)(client, features)
