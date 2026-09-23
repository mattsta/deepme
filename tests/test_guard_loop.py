"""Unit tests for the guard generation loop (guarded_events).

The loop is the most intricate logic in the system — abort-on-violation,
continuation, retry accounting, scope handling, truncation semantics — and
was previously covered only by live smoke tests. Here it runs against a FAKE
adapter (scripted upstream event streams), so every branch is exercised with
no network, no GPU, no engine.

The guard modules are written to be deployed flat next to each other on the
server (`from gate import ...`), so we load server.py with that import path
and a stub environment.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

GUARD_DIR = Path(__file__).resolve().parents[1] / "src" / "deepme" / "guard"


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    """Import guard/server.py the way it runs on the server (flat imports)."""
    policy_file = tmp_path_factory.mktemp("guard") / "policy.toml"
    policy_file.write_text(
        '[guard]\nmax_retries = 2\nhold_slack = 8\ndefault_sets = ["w"]\n\n'
        '[sets.w]\nrules = [{ kind = "word", pattern = "that" }]\n'
    )
    import os

    os.environ.update(
        GUARD_UPSTREAM="http://127.0.0.1:65535",
        GUARD_POLICY=str(policy_file),
        GUARD_ENGINE="vllm",
        GUARD_FEATURES="continuation,bad_words,logit_bias,tokenize",
    )
    sys.path.insert(0, str(GUARD_DIR))
    spec = importlib.util.spec_from_file_location("guard_server", GUARD_DIR / "server.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeAdapter:
    """Scripted upstream: each attempt yields one prepared event list."""

    def __init__(self, attempts, features=("continuation", "bad_words", "logit_bias", "tokenize")):
        self.attempts = list(attempts)
        self.features = set(features)
        self.calls = []  # request bodies seen, in order

    async def stream(self, body, auth):
        self.calls.append(body)
        events = self.attempts.pop(0) if self.attempts else [("finish", "stop")]
        for e in events:
            yield e

    stream_completions = stream

    def apply_bans(self, body, patterns):
        if "bad_words" not in self.features or not patterns:
            return body
        out = dict(body)
        out["bad_words"] = list(body.get("bad_words") or []) + patterns
        return out

    def continuation_body(self, body, accepted):
        if "continuation" not in self.features:
            return None
        cont = dict(body)
        cont["messages"] = [*list(body["messages"]), {"role": "assistant", "content": accepted}]
        cont["continue_final_message"] = True
        return cont

    def completions_continuation(self, body, accepted):
        cont = dict(body)
        cont["prompt"] = body.get("prompt", "") + accepted
        return cont

    async def suppression_ids(self, model, fragment, auth):
        return [1234] if {"tokenize", "logit_bias"} <= self.features else []


async def drain(server, body, policy, mode="chat"):
    text, reasoning, tools, meta = "", "", [], {}
    async for kind, payload in server.guarded_events(body, "", policy, mode=mode):
        if kind == "text":
            text += payload
        elif kind == "reasoning":
            reasoning += payload
        elif kind == "tool_calls":
            tools.extend(payload)
        elif kind == "meta":
            meta = payload
    return text, reasoning, tools, meta


def body(**kw):
    b = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    b.update(kw)
    return b


@pytest.mark.asyncio
async def test_clean_stream_passes_through(server):
    server.ADAPTER = FakeAdapter(
        [
            [
                ("content", "all clean here"),
                ("finish", "stop"),
                ("usage", {"completion_tokens": 3, "prompt_tokens": 5}),
            ]
        ]
    )
    text, _, _, meta = await drain(server, body(), server.POLICY)
    assert text == "all clean here"
    assert meta["violations"] == []
    assert meta["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_violation_regenerates_and_keeps_clean_prefix(server):
    server.ADAPTER = FakeAdapter(
        [
            [("content", "keep this that bad"), ("finish", "stop")],
            [("content", "and a clean ending"), ("finish", "stop")],
        ]
    )
    text, _, _, meta = await drain(server, body(), server.POLICY)
    assert "that" not in text.lower()
    assert text.startswith("keep this")
    assert meta["violations"][0]["action"] == "regenerated"
    assert meta["violations"][0]["channel"] == "content"
    # second call was a continuation carrying the accepted prefix
    assert server.ADAPTER.calls[1].get("continue_final_message") is True


@pytest.mark.asyncio
async def test_retries_exhausted_truncates(server):
    bad = [("content", "nope that here"), ("finish", "stop")]
    server.ADAPTER = FakeAdapter([bad, bad, bad, bad])
    text, _, _, meta = await drain(server, body(), server.POLICY)
    assert "that" not in text.lower()
    assert meta["finish_reason"] == "guard_truncated"
    assert "truncated" in meta["violations"][-1]["action"]


@pytest.mark.asyncio
async def test_no_continuation_support_truncates(server):
    server.ADAPTER = FakeAdapter(
        [[("content", "start that end"), ("finish", "stop")]], features=("bad_words",)
    )
    text, _, _, meta = await drain(server, body(), server.POLICY)
    assert "that" not in text.lower()
    assert meta["violations"][-1]["action"] == "truncated (no continuation support)"
    assert meta["finish_reason"] == "guard_truncated"


@pytest.mark.asyncio
async def test_upstream_death_ends_cleanly(server):
    import httpx

    class DyingAdapter(FakeAdapter):
        async def stream(self, body, auth):
            yield ("content", "partial output ")
            raise httpx.ReadError("connection lost")

    server.ADAPTER = DyingAdapter([])
    text, _, _, meta = await drain(server, body(), server.POLICY)
    assert meta["finish_reason"] == "guard_upstream_error"
    assert "ReadError" in meta["upstream_error"]
    assert "partial" in text


@pytest.mark.asyncio
async def test_emitted_tool_calls_prevent_content_continuation(server):
    # A tool call already streamed to the client; a later content violation
    # must truncate (continuing would duplicate/garble the tool call).
    server.ADAPTER = FakeAdapter(
        [
            [
                (
                    "tool_calls",
                    [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}],
                ),
                ("content", "then that word"),
                ("finish", "tool_calls"),
            ],
        ]
    )
    _, _, tools, meta = await drain(server, body(), server.POLICY)
    assert len(tools) == 1
    assert "structured output already emitted" in meta["violations"][-1]["action"]


@pytest.mark.asyncio
async def test_deterministic_request_skips_futile_retry(server):
    # temperature=0 on an engine that supports neither bad_words nor token
    # suppression: the continuation request would be byte-identical, so
    # regenerating is provably futile — truncate instead of burning GPU.
    pol = server.POLICY.resolve({"rules": [{"kind": "substring", "pattern": "@@"}], "sets": []})
    # Violation at offset 0: no clean prefix accepted, so a retry makes no
    # forward progress and (with no bans/bias support) adds no constraint.
    server.ADAPTER = FakeAdapter(
        [[("content", "@@ bad start"), ("finish", "stop")]] * 4,
        features=("continuation",),  # no bad_words, no tokenize/logit_bias
    )
    _, _, _, meta = await drain(server, body(temperature=0), pol)
    assert "deterministic" in meta["violations"][-1]["action"]
    assert len(server.ADAPTER.calls) == 1  # no wasted second generation


@pytest.mark.asyncio
async def test_deterministic_still_retries_when_constraint_changes(server):
    # Same deterministic request, but the engine DOES accept suppression —
    # the retry is meaningful and must happen.
    server.ADAPTER = FakeAdapter(
        [
            [("content", "x that y"), ("finish", "stop")],
            [("content", "x clean y"), ("finish", "stop")],
        ]
    )
    _, _, _, meta = await drain(server, body(temperature=0), server.POLICY)
    assert meta["violations"][-1]["action"] == "regenerated"
    assert len(server.ADAPTER.calls) == 2


@pytest.mark.asyncio
async def test_usage_accumulates_across_attempts(server):
    server.ADAPTER = FakeAdapter(
        [
            [
                ("content", "aa that"),
                ("usage", {"completion_tokens": 10, "prompt_tokens": 7}),
                ("finish", "stop"),
            ],
            [
                ("content", "bb done"),
                ("usage", {"completion_tokens": 5, "prompt_tokens": 20}),
                ("finish", "stop"),
            ],
        ]
    )
    _, _, _, meta = await drain(server, body(), server.POLICY)
    # An ABORTED attempt never delivers its usage frame (we cut the stream at
    # the violation), so reported usage covers completed attempts only. The
    # regenerations counter is what tells the operator extra GPU work happened.
    assert meta["usage"]["completion_tokens"] == 5
    assert meta["regenerations"] == 1


@pytest.mark.asyncio
async def test_usage_sums_when_attempts_complete(server):
    # Both attempts deliver usage (violation found at flush, after the frame).
    server.ADAPTER = FakeAdapter(
        [
            [
                ("usage", {"completion_tokens": 10, "prompt_tokens": 7}),
                ("content", "aa that"),
                ("finish", "stop"),
            ],
            [
                ("usage", {"completion_tokens": 5, "prompt_tokens": 20}),
                ("content", "bb done"),
                ("finish", "stop"),
            ],
        ]
    )
    _, _, _, meta = await drain(server, body(), server.POLICY)
    assert meta["usage"]["completion_tokens"] == 15
    assert meta["usage"]["prompt_tokens"] == 7  # first attempt's real input


@pytest.mark.asyncio
async def test_completions_mode_uses_prompt_concat(server):
    server.ADAPTER = FakeAdapter(
        [
            [("content", "one that two"), ("finish", "stop")],
            [("content", " clean tail"), ("finish", "stop")],
        ]
    )
    text, _, _, _meta = await drain(
        server, {"model": "m", "prompt": "seed "}, server.POLICY, mode="completions"
    )
    assert "that" not in text.lower()
    assert server.ADAPTER.calls[1]["prompt"].startswith("seed ")


@pytest.mark.asyncio
async def test_guard_unsupported_shapes(server):
    assert server._guard_unsupported({"n": 2}) is not None
    assert server._guard_unsupported({"prompt": ["a", "b"]}) is not None
    assert server._guard_unsupported({"n": 1}) is None


def test_assemble_tool_calls_merges_fragments(server):
    deltas = [
        {"index": 0, "id": "a", "function": {"name": "get_", "arguments": '{"x'}},
        {"index": 0, "function": {"name": "weather", "arguments": '": 1}'}},
    ]
    calls = server._assemble_tool_calls(deltas)
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}
