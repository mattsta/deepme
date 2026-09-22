"""Unit tests for the output-guard gate (pure logic, no network/GPU)."""

import pytest

from deepme.guard.gate import GuardPolicy, Rule, StreamGate, join_seam

DASHES = ["—", "–", "‒", "―", "‐", "‑"]


def policy() -> GuardPolicy:
    rules = [Rule(pattern="that", kind="word")]
    rules += [Rule(pattern=d, kind="substring") for d in DASHES]
    rules += [Rule(pattern="delve into", kind="phrase")]
    return GuardPolicy(rules=rules, hold_slack=8)


def run_stream(p: GuardPolicy, chunks: list[str]):
    gate = StreamGate(p)
    released, violations = "", []
    for c in chunks:
        out, v = gate.feed(c)
        released += out
        if v:
            violations.append(v)
    out, v = gate.flush()
    released += out
    if v:
        violations.append(v)
    return released, violations


def test_clean_text_passes_through_completely():
    text = "The quick brown fox jumps over the lazy dog. This is fine."
    released, violations = run_stream(policy(), list(text))
    assert released == text
    assert violations == []


def test_isolated_word_is_caught():
    released, violations = run_stream(policy(), ["I think that ", "it works"])
    assert len(violations) == 1
    assert violations[0].fragment.lower() == "that"
    assert "that" not in released.lower()


def test_word_inside_other_words_is_allowed():
    text = "thatched roofs and thermostats matter."
    released, violations = run_stream(policy(), [text])
    assert violations == []
    assert released == text


def test_case_insensitive_word():
    _, violations = run_stream(policy(), ["Note That here"])
    assert len(violations) == 1


@pytest.mark.parametrize("dash", DASHES)
def test_dash_codepoints_are_caught(dash):
    released, violations = run_stream(policy(), [f"one {dash} two"])
    assert len(violations) == 1
    assert dash not in released


def test_ascii_hyphen_is_allowed():
    text = "well-known state-of-the-art hy-phen"
    released, violations = run_stream(policy(), [text])
    assert violations == []
    assert released == text


def test_phrase_across_chunk_boundaries():
    released, violations = run_stream(policy(), ["let us delve ", "into the topic"])
    assert len(violations) == 1
    assert "delve" not in released


def test_word_split_across_many_tiny_chunks():
    # sub-fragmented output (one letter at a time) must still be caught
    released, violations = run_stream(policy(), list("so t h a t".replace(" ", "")))
    # "sothat" contains no isolated 'that' (it's inside a word) — allowed
    assert violations == []
    released, violations = run_stream(
        policy(), list("so ") + list("that") + list(" ok")
    )
    assert len(violations) == 1
    assert "that" not in released.lower()


def test_violation_returns_clean_prefix_only():
    gate = StreamGate(policy())
    out1, v1 = gate.feed("alpha beta that gamma")
    assert v1 is not None
    assert out1 == "alpha beta "


def test_hold_back_never_releases_partial_forbidden_word():
    gate = StreamGate(policy())
    out, v = gate.feed("tha")  # could be forming "that"
    assert v is None
    assert out == ""  # held back


def test_multiple_violations_sequential():
    p = policy()
    gate = StreamGate(p)
    released, violations = "", []
    for chunk in ["that — that"]:
        out, v = gate.feed(chunk)
        released += out
        while v:
            out, v = gate.feed("")
            released += out
            if v:
                violations.append(v)
    # first violation reported on feed; buffer reset drops the rest of chunk
    # (caller regenerates) — just assert nothing forbidden was released
    assert "that" not in released.lower()
    assert "—" not in released


def test_regex_rule():
    p = GuardPolicy(
        rules=[Rule(pattern=r"\b\d{3}-\d{2}-\d{4}\b", kind="regex")],
        hold_slack=16,
    )
    released, violations = run_stream(p, ["ssn is 123-45-6789 ok"])
    assert len(violations) == 1
    assert "123-45-6789" not in released


def test_case_sensitive_rule():
    p = GuardPolicy(rules=[Rule(pattern="Moreover", kind="word", case_sensitive=True)])
    _, v1 = run_stream(p, ["moreover it works"])
    assert v1 == []
    _, v2 = run_stream(p, ["Moreover it works"])
    assert len(v2) == 1


def test_markup_fragmented_evasion_is_caught():
    # "*tha*t" — markdown-split spelling of a forbidden word
    released, violations = run_stream(policy(), ["a moody shift *tha*t chills"])
    assert len(violations) == 1
    assert "tha" not in released.lower().replace("thatched", "")


def test_strip_chars_does_not_false_positive_inside_words():
    released, violations = run_stream(policy(), ["use snake_case_that_works here"])
    # underscore is stripped in shadow: "snakecasethatworks" — 'that' is
    # inside a word there, so still allowed
    assert violations == []


def _sets_policy() -> GuardPolicy:
    return GuardPolicy.from_toml_dict(
        {
            "guard": {"default_sets": ["words"]},
            "sets": {
                "words": {"rules": [{"pattern": "that", "kind": "word"}]},
                "dashes": {"rules": [{"pattern": "—", "kind": "substring"}]},
            },
        }
    )


def test_named_sets_default_selection():
    p = _sets_policy()
    assert p.scan("so that works") is not None
    assert p.scan("a — b") is None  # dashes not in default_sets


def test_request_selects_sets_union():
    p = _sets_policy().resolve({"sets": ["words", "dashes"]})
    assert p.scan("a — b") is not None
    assert p.scan("so that works") is not None


def test_request_disables_guard():
    assert _sets_policy().resolve({"enabled": False}) is None


def test_request_inline_rules():
    p = _sets_policy().resolve(
        {"sets": [], "rules": [{"pattern": "foo", "kind": "word"}]}
    )
    assert p.scan("so that works") is None  # named sets deselected
    assert p.scan("a foo b") is not None  # inline rule active


def test_unknown_set_raises():
    with pytest.raises(ValueError):
        _sets_policy().resolve({"sets": ["nope"]})


def test_scope_default_and_override():
    p = _sets_policy()
    assert p.scope == ("content",)
    p2 = p.resolve({"scope": ["content", "tool_arguments"]})
    assert "tool_arguments" in p2.scope


def test_join_seam_collapses_doubled_space():
    assert join_seam("word ", " next") == "next"
    assert join_seam("word ", "   next") == "next"


def test_join_seam_preserves_single_boundary():
    assert join_seam("word ", "next") == "next"
    assert join_seam("word", " next") == " next"


def test_join_seam_preserves_newline_indentation():
    assert join_seam("line\n", "    indented") == "    indented"
    assert join_seam("word ", "\n    block") == "\n    block"


def test_join_seam_preserves_model_punctuation():
    assert join_seam("word ", ", next") == ", next"


def test_policy_from_toml_dict():
    d = {
        "guard": {"max_retries": 3, "hold_slack": 4},
        "rule": [
            {"pattern": "that", "kind": "word"},
            {"pattern": "—", "kind": "substring"},
        ],
    }
    p = GuardPolicy.from_toml_dict(d)
    assert p.max_retries == 3
    assert len(p.rules) == 2
    assert p.scan("x — y").fragment == "—"
