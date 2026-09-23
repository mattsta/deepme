"""Output-guard proxy: OpenAI-compatible front for an inference engine.

Sits in front of the upstream (vLLM) API. /v1/chat/completions responses are
gated through GuardPolicy (gate.py): provably-clean text streams to the
client; on a violation the generation is rejected at the violation start
and REGENERATED via continuation (chat: continue_final_message;
completions: prompt concat) with sampler bans + accumulating logit_bias.
The guard never authors text: after max_retries the output truncates at
the clean boundary (finish_reason=guard_truncated); buffered channels
(reasoning/tool args in scope) are withheld rather than edited.

Environment: GUARD_UPSTREAM (e.g. http://127.0.0.1:8000),
GUARD_POLICY (path to policy.toml), GUARD_PORT.

Runs on the inference server from ~/deepme-runtime/guard (fastapi+uvicorn+
httpx); gate.py sits alongside this file.
"""

from __future__ import annotations

import contextlib
import json
import os
import tomllib
from typing import Any, TypedDict

import httpx
from adapters import make_adapter  # deployed next to this file
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from gate import GuardPolicy, StreamGate
from gate import join_seam as _join_seam


class GenMeta(TypedDict, total=False):
    """Per-generation metadata accumulated by the guard loop."""

    violations: list[dict[str, Any]]
    finish_reason: str | None
    usage: dict[str, Any] | None
    model: str | None
    tool_deltas: list[dict[str, Any]]
    regenerations: int
    upstream_error: str


def _is_deterministic(body: dict) -> bool:
    """A request whose regeneration would repeat identically unless the guard
    changes the sampling constraints (bans/bias)."""
    return body.get("seed") is not None or (body.get("temperature") == 0)


def _bans_effective(bans: list[str]) -> bool:
    """Whether sampler bans actually reach the engine — asked of the adapter
    itself (capability), not of a config string that may not match reality."""
    if not bans:
        return False
    probe = {"probe": True}
    return ADAPTER.apply_bans(probe, bans) != probe


def _retry_is_futile(deterministic: bool, progressed: bool, new_constraint: bool) -> bool:
    """A deterministic retry that neither advanced the accepted text nor added
    a sampling constraint will reproduce the identical violation — burning a
    full generation for nothing. (Non-deterministic sampling can differ on its
    own, so only deterministic requests are cut short.)"""
    return deterministic and not progressed and not new_constraint


def _merge_usage(acc: dict | None, new: dict | None) -> dict | None:
    """Accumulate usage across regeneration attempts: sum completion tokens,
    keep the first attempt's prompt tokens (continuations re-send accepted
    text as prompt, which would double-count)."""
    if new is None:
        return acc
    if acc is None:
        return dict(new)
    out = dict(acc)
    out["completion_tokens"] = (acc.get("completion_tokens") or 0) + (
        new.get("completion_tokens") or 0
    )
    out["total_tokens"] = (out.get("prompt_tokens") or 0) + out["completion_tokens"]
    return out


UPSTREAM = os.environ["GUARD_UPSTREAM"].rstrip("/")
POLICY_PATH = os.environ["GUARD_POLICY"]
ENGINE = os.environ.get("GUARD_ENGINE", "vllm")
FEATURES = set(filter(None, os.environ.get("GUARD_FEATURES", "").split(",")))

with open(POLICY_PATH, "rb") as f:
    POLICY = GuardPolicy.from_toml_dict(tomllib.load(f))

app = FastAPI()
client = httpx.AsyncClient(base_url=UPSTREAM, timeout=1800)
ADAPTER = make_adapter(ENGINE, client, FEATURES)


def _preemptive_bad_words(policy: GuardPolicy) -> list[str]:
    """Sampler-level multi-token bans for word/phrase rules, applied to every
    request up front — the model picks alternatives naturally instead of our
    having to truncate-and-retry. (substring/regex rules can't be expressed
    as token sequences; the gate still guarantees those.)"""
    out = []
    for rule in policy.rules:
        if rule.kind in ("word", "phrase"):
            base = rule.pattern
            out += [
                base,
                " " + base,
                base.capitalize(),
                " " + base.capitalize(),
                base.upper(),
                " " + base.upper(),
            ]
        elif rule.kind == "substring":
            out += [rule.pattern, " " + rule.pattern]
    return out


async def guarded_events(body: dict, auth: str, policy: GuardPolicy, *, mode: str = "chat"):
    """Guarded generation as a typed event stream. Upstream is ALWAYS
    streamed; content deltas release incrementally through the gate.

    Channel semantics (reject-and-regenerate everywhere; the guard never
    authors or edits text — exhaustion truncates/withholds):

      content     -> violation aborts upstream, resumes via continuation
                     (chat: continue_final_message; completions: prompt
                     concat — universal, no flags needed)
      reasoning / tool args (when in scope) -> BUFFERED until clean-complete
                     (they precede/accompany content, nothing emitted early);
                     violation regenerates the FULL request with suppression

    Events: ("text", str) | ("reasoning", str) | ("tool_calls", list)
          | ("meta", dict)  -- exactly once, at the end
    """
    accepted = ""
    biases: dict[str, int] = dict(body.get("logit_bias") or {})
    retries_left = policy.max_retries
    full_retries_left = policy.max_retries
    bans = _preemptive_bad_words(policy)
    guard_reasoning = "reasoning" in policy.scope and mode == "chat"
    guard_tool_args = "tool_arguments" in policy.scope and mode == "chat"
    meta: GenMeta = {
        "violations": [],
        "finish_reason": None,
        "usage": None,
        "model": None,
    }
    request_body = ADAPTER.apply_bans(dict(body), bans)
    continuing = False
    deterministic = _is_deterministic(body)
    emitted_structured = False  # unguarded reasoning/tool_calls already sent
    upstream_error = None

    def _stream(rb):
        return ADAPTER.stream(rb, auth) if mode == "chat" else ADAPTER.stream_completions(rb, auth)

    def _continuation(acc):
        if mode == "completions":
            return ADAPTER.completions_continuation(body, acc)
        return ADAPTER.continuation_body(body, acc)

    while True:
        gate = StreamGate(policy)
        r_gate = StreamGate(policy) if guard_reasoning else None
        violation = None  # content-channel violation
        aux_violation = None  # buffered-channel violation -> full regen
        aux_channel = None
        first_release = True
        accepted_at_start = accepted  # forward-progress tracking
        biases_at_start = len(biases)
        reasoning_buf: list[str] = []
        reasoning_clean: list[str] = []
        tool_deltas: list = []
        emitted_reasoning: list[str] = []

        # Upstream is always streamed; aclosing() guarantees the upstream
        # request is cancelled when we break on a violation, and try/except
        # turns a mid-stream engine death into a clean truncation.
        try:
            async with contextlib.aclosing(_stream(request_body)) as stream:
                async for kind, payload in stream:
                    if kind == "content":
                        out, v = gate.feed(payload)
                        if out:
                            if continuing and first_release:
                                out = _join_seam(accepted, out)
                            first_release = False
                            accepted += out
                            yield ("text", out)
                        if v is not None:
                            violation = v
                            break
                    elif kind == "reasoning":
                        if guard_reasoning and r_gate is not None:
                            reasoning_buf.append(payload)
                            r_out, rv = r_gate.feed(payload)
                            if r_out:
                                reasoning_clean.append(r_out)
                            if rv is not None:
                                aux_violation, aux_channel = rv, "reasoning"
                                break
                        else:
                            emitted_reasoning.append(payload)
                            emitted_structured = True
                            yield ("reasoning", payload)
                    elif kind == "tool_calls":
                        tool_deltas.extend(payload)
                        if not guard_tool_args:
                            emitted_structured = True
                            yield ("tool_calls", payload)
                    elif kind == "finish":
                        meta["finish_reason"] = payload
                    elif kind == "usage":
                        meta["usage"] = _merge_usage(meta["usage"], payload)
                    elif kind == "model":
                        meta["model"] = payload
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            upstream_error = f"{type(exc).__name__}: {exc}"

        # Upstream died mid-stream: flush whatever is provably clean, end.
        if upstream_error is not None and violation is None and aux_violation is None:
            tail, violation = gate.flush()
            if tail and violation is None:
                accepted += tail
                yield ("text", tail)
            if violation is None:
                meta["finish_reason"] = "guard_upstream_error"
                meta["upstream_error"] = upstream_error
                meta["tool_deltas"] = tool_deltas
                yield ("meta", meta)
                return

        # ---- buffered channels: finalize + scan (only when no abort yet) ----
        if violation is None and aux_violation is None:
            tail, violation = gate.flush()
            if tail:
                if continuing and first_release:
                    tail = _join_seam(accepted, tail)
                accepted += tail
                yield ("text", tail)
            if violation is None and guard_reasoning and r_gate is not None:
                r_tail, rv = r_gate.flush()
                if rv is not None:
                    aux_violation, aux_channel = rv, "reasoning"
                else:
                    reasoning_clean.append(r_tail)
            if violation is None and aux_violation is None and guard_tool_args:
                calls = _assemble_tool_calls(tool_deltas) or []
                for c in calls:
                    v = policy.scan(c["function"]["arguments"] + " " + c["function"]["name"])
                    if v is not None:
                        aux_violation, aux_channel = v, "tool_arguments"
                        break

        if violation is None and aux_violation is None:
            if guard_reasoning and reasoning_clean:
                yield ("reasoning", "".join(reasoning_clean))
            if guard_tool_args and tool_deltas:
                yield ("tool_calls", tool_deltas)
            meta["tool_deltas"] = tool_deltas
            yield ("meta", meta)
            return

        # ---- content violation: prefix continuation ----
        if violation is not None:
            meta["violations"].append(
                {
                    "pattern": violation.rule.pattern,
                    "kind": violation.rule.kind,
                    "fragment": violation.fragment,
                    "channel": "content",
                }
            )
            continuing = True
            cont = _continuation(accepted)
            for tok in await ADAPTER.suppression_ids(
                body.get("model", ""), violation.fragment, auth
            ):
                biases[str(tok)] = -100
            # Reasons we cannot usefully continue -> truncate at clean boundary
            # (the guard never authors text; output is model-generated up to
            # here and simply ends):
            #   - backend can't continue
            #   - retries exhausted
            #   - structured output (tool_calls/reasoning) already streamed:
            #     a fresh continuation would duplicate/garble it
            #   - deterministic request with no new suppression to change it
            next_body = ADAPTER.apply_bans(cont, bans) if cont is not None else None
            if next_body is not None and biases:
                next_body = {**next_body, "logit_bias": biases}
            reason = None
            if cont is None:
                reason = "truncated (no continuation support)"
            elif retries_left <= 0:
                reason = "truncated (retries exhausted)"
            elif emitted_structured:
                reason = "truncated (structured output already emitted)"
            elif _retry_is_futile(
                deterministic,
                progressed=len(accepted) > len(accepted_at_start),
                new_constraint=(len(biases) > biases_at_start or _bans_effective(bans)),
            ):
                reason = "truncated (deterministic; no new constraint)"
            if reason is not None:
                meta["violations"][-1]["action"] = reason
                meta["finish_reason"] = "guard_truncated"
                meta["tool_deltas"] = tool_deltas
                yield ("meta", meta)
                return
            retries_left -= 1
            meta["violations"][-1]["action"] = "regenerated"
            meta["regenerations"] = meta.get("regenerations", 0) + 1
            request_body = next_body
            continue

        # ---- buffered-channel violation: regenerate the FULL request ----
        assert aux_violation is not None  # the only remaining branch
        meta["violations"].append(
            {
                "pattern": aux_violation.rule.pattern,
                "kind": aux_violation.rule.kind,
                "fragment": aux_violation.fragment,
                "channel": aux_channel,
            }
        )
        can_full_regen = (
            full_retries_left > 0
            and not accepted
            and not emitted_reasoning
            and not emitted_structured
        )
        if can_full_regen:
            full_retries_left -= 1
            for tok in await ADAPTER.suppression_ids(
                body.get("model", ""), aux_violation.fragment, auth
            ):
                biases[str(tok)] = -100
            if _retry_is_futile(
                deterministic,
                progressed=False,  # full regen restarts; no accepted prefix
                new_constraint=(len(biases) > biases_at_start or _bans_effective(bans)),
            ):
                # Identical regen would repeat — fall through to withhold.
                pass
            else:
                meta["violations"][-1]["action"] = "regenerated (full request)"
                request_body = ADAPTER.apply_bans(dict(body), bans)
                if biases:
                    request_body["logit_bias"] = biases
                continuing = False
                continue
        # Regeneration exhausted or impossible (output already emitted):
        # WITHHOLD the violating channel entirely. The guard never edits —
        # a channel is either model-generated-and-clean or absent.
        meta["violations"][-1]["action"] = f"withheld ({aux_channel})"
        meta["finish_reason"] = meta.get("finish_reason") or "guard_truncated"
        if aux_channel != "tool_arguments":
            meta["tool_deltas"] = tool_deltas
        yield ("meta", meta)
        return


def _assemble_tool_calls(deltas: list) -> list | None:
    """Aggregate streamed tool_call delta fragments into full OpenAI
    tool_calls."""
    calls: dict[int, dict] = {}
    for d in deltas:
        idx = d.get("index", 0)
        c = calls.setdefault(
            idx,
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if d.get("id"):
            c["id"] = d["id"]
        fn = d.get("function") or {}
        if fn.get("name"):
            c["function"]["name"] += fn["name"]
        if fn.get("arguments"):
            c["function"]["arguments"] += fn["arguments"]
    if not calls:
        return None
    out = [calls[i] for i in sorted(calls)]
    return out


def _resolve_policy(body: dict):
    return POLICY.resolve(body.pop("guard", None))


def _guard_unsupported(body: dict) -> str | None:
    """Request shapes the guard cannot correctly gate (single-choice only —
    the guard tracks one accepted text). Returns an error message or None."""
    if int(body.get("n") or 1) > 1:
        return (
            "guard does not support n>1 (it gates a single choice); "
            'set n=1, or disable the guard with {"guard":{"enabled":false}}'
        )
    prompt = body.get("prompt")
    if isinstance(prompt, list) and len(prompt) > 1:
        return "guard does not support multi-prompt completions; send one prompt per request"
    return None


async def _passthrough_chat(body: dict, auth: str, stream: bool, path: str):
    if stream:

        async def raw_sse():
            b = dict(body)
            b["stream"] = True
            async with client.stream(
                "POST",
                path,
                json=b,
                headers={"Authorization": auth},
            ) as resp:
                async for line in resp.aiter_lines():
                    yield line + "\n"

        return StreamingResponse(raw_sse(), media_type="text/event-stream")
    r = await client.post(path, json=body, headers={"Authorization": auth})
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type"),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    auth = request.headers.get("authorization", "")
    body = await request.json()
    try:
        policy = _resolve_policy(body)
    except ValueError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    stream = bool(body.get("stream"))
    if policy is None or not policy.rules:
        return await _passthrough_chat(body, auth, stream, "/v1/chat/completions")
    unsupported = _guard_unsupported(body)
    if unsupported:
        return JSONResponse({"error": {"message": unsupported}}, status_code=400)

    if not stream:
        text_parts, reasoning_parts, meta = [], [], {}
        async for kind, payload in guarded_events(body, auth, policy):
            if kind == "text":
                text_parts.append(payload)
            elif kind == "reasoning":
                reasoning_parts.append(payload)
            elif kind == "meta":
                meta = payload
        message: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_parts),
        }
        if reasoning_parts:
            message["reasoning_content"] = "".join(reasoning_parts)
        tool_calls = _assemble_tool_calls(meta.get("tool_deltas") or [])
        if tool_calls:
            message["tool_calls"] = tool_calls
        return JSONResponse(
            {
                "id": "guard-chatcmpl",
                "object": "chat.completion",
                "model": meta.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": meta.get("finish_reason"),
                    }
                ],
                "usage": meta.get("usage"),
                "guard": {"violations": meta.get("violations", [])},
            }
        )

    async def sse():
        meta = {}
        async for kind, payload in guarded_events(body, auth, policy):
            if kind == "meta":
                meta = payload
                continue
            delta = {}
            if kind == "text":
                delta = {"content": payload}
            elif kind == "reasoning":
                delta = {"reasoning_content": payload}
            elif kind == "tool_calls":
                delta = {"tool_calls": payload}
            evt = {
                "id": "guard-chatcmpl",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(evt)}\n\n"
        final = {
            "id": "guard-chatcmpl",
            "object": "chat.completion.chunk",
            "model": meta.get("model"),
            "choices": [{"index": 0, "delta": {}, "finish_reason": meta.get("finish_reason")}],
            "usage": meta.get("usage"),
            "guard": {"violations": meta.get("violations", [])},
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/v1/completions")
async def completions(request: Request):
    auth = request.headers.get("authorization", "")
    body = await request.json()
    try:
        policy = _resolve_policy(body)
    except ValueError as exc:
        return JSONResponse({"error": {"message": str(exc)}}, status_code=400)
    stream = bool(body.get("stream"))
    if policy is None or not policy.rules:
        return await _passthrough_chat(body, auth, stream, "/v1/completions")
    unsupported = _guard_unsupported(body)
    if unsupported:
        return JSONResponse({"error": {"message": unsupported}}, status_code=400)

    if not stream:
        text_parts, meta = [], {}
        async for kind, payload in guarded_events(body, auth, policy, mode="completions"):
            if kind == "text":
                text_parts.append(payload)
            elif kind == "meta":
                meta = payload
        return JSONResponse(
            {
                "id": "guard-cmpl",
                "object": "text_completion",
                "model": meta.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "text": "".join(text_parts),
                        "finish_reason": meta.get("finish_reason"),
                    }
                ],
                "usage": meta.get("usage"),
                "guard": {"violations": meta.get("violations", [])},
            }
        )

    async def sse():
        meta = {}
        async for kind, payload in guarded_events(body, auth, policy, mode="completions"):
            if kind == "meta":
                meta = payload
                continue
            if kind == "text":
                evt = {
                    "id": "guard-cmpl",
                    "object": "text_completion",
                    "choices": [{"index": 0, "text": payload, "finish_reason": None}],
                }
                yield f"data: {json.dumps(evt)}\n\n"
        final = {
            "id": "guard-cmpl",
            "object": "text_completion",
            "model": meta.get("model"),
            "choices": [{"index": 0, "text": "", "finish_reason": meta.get("finish_reason")}],
            "usage": meta.get("usage"),
            "guard": {"violations": meta.get("violations", [])},
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def passthrough(request: Request, path: str):
    r = await client.request(
        request.method,
        f"/{path}",
        content=await request.body(),
        headers={
            k: v
            for k, v in request.headers.items()
            if k.lower() in ("authorization", "content-type")
        },
        params=dict(request.query_params),
    )
    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type=r.headers.get("content-type"),
    )
