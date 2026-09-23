"""Output-guard gate: streaming forbidden-pattern detection with hold-back.

Pure stdlib, no project imports — this exact file also runs inside the guard
proxy on the inference server, and is unit-tested locally (tests/test_gate.py).

Rule kinds:
  word      — forbidden as an isolated word (letter/digit boundaries)
  substring — forbidden anywhere (graphemes, code points, e.g. em dashes)
  phrase    — forbidden multi-word sequence (any whitespace between words)
  regex     — full regex, compiled as written (operator-owned power tool)

Every rule takes case_sensitive (default false). The guard NEVER writes
text of its own: violations are rejected and regenerated; when regeneration
is exhausted or impossible, output is truncated/withheld at the clean
boundary. regex rules should stay bounded in match length — the hold-back
window uses max pattern length, so an unbounded regex needs a generous
[guard] hold_slack.

Streaming semantics: feed() returns text that is PROVABLY clean — a tail
window (longest pattern + slack + the current unfinished word) is held back
so a pattern can never straddle the release boundary. On violation, feed()
returns the clean prefix and a Violation; the caller truncates there and
regenerates the continuation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Whitespace between phrase words is bounded so a phrase's maximum raw match
# span is finite and the streaming hold window can guarantee catching it.
PHRASE_GAP_MAX = 8
# Regex matches have unknown length; assume up to this raw span for hold sizing.
REGEX_SPAN_MAX = 256


@dataclass(frozen=True)
class Rule:
    pattern: str
    kind: str = "word"  # word | substring | phrase | regex
    case_sensitive: bool = False

    def compile(self) -> re.Pattern:
        flags = 0 if self.case_sensitive else re.IGNORECASE
        if self.kind == "word":
            return re.compile(rf"(?<![\w]){re.escape(self.pattern)}(?![\w])", flags)
        if self.kind == "substring":
            return re.compile(re.escape(self.pattern), flags)
        if self.kind == "phrase":
            words = self.pattern.split()
            body = rf"\s{{1,{PHRASE_GAP_MAX}}}".join(re.escape(w) for w in words)
            return re.compile(rf"(?<![\w]){body}(?![\w])", flags)
        if self.kind == "regex":
            return re.compile(self.pattern, flags)
        raise ValueError(f"unknown rule kind {self.kind!r}")

    def base_span(self) -> int:
        """Upper bound on the raw characters a match can span, BEFORE shadow
        (strip-char) inflation. Drives the streaming hold window."""
        if self.kind == "phrase":
            words = self.pattern.split()
            return sum(len(w) for w in words) + PHRASE_GAP_MAX * max(0, len(words) - 1)
        if self.kind == "regex":
            return REGEX_SPAN_MAX
        return len(self.pattern)


@dataclass(frozen=True)
class Violation:
    rule: Rule
    fragment: str  # the matched forbidden text
    offset: int  # char offset of the match within the fed stream segment


def _parse_rule(r: dict) -> Rule:
    return Rule(
        pattern=r["pattern"],
        kind=r.get("kind", "word"),
        case_sensitive=bool(r.get("case_sensitive", False)),
    )


@dataclass
class GuardPolicy:
    rules: list[Rule] = field(default_factory=list)
    max_retries: int = 6
    hold_slack: int = 16
    # Formatting characters stripped in a shadow copy before scanning, so
    # markup-fragmented evasions (e.g. "*tha*t" for "that") are still caught.
    # Matches are mapped back to raw-text offsets.
    strip_chars: str = "*_~`"
    # Named rule sets (referenced by requests); default_sets are active when
    # a request doesn't specify.
    sets: dict = field(default_factory=dict)  # name -> list[Rule]
    default_sets: tuple = ()
    # Which output channels are enforced. "content" regenerates via prefix
    # continuation; "reasoning"/"tool_arguments" (opt-in) are buffered and
    # regenerate the full request on violation.
    scope: tuple = ("content",)

    @classmethod
    def from_toml_dict(cls, d: dict) -> GuardPolicy:
        g = d.get("guard", {})
        sets = {
            name: [_parse_rule(r) for r in body.get("rules", [])]
            for name, body in d.get("sets", {}).items()
        }
        legacy = [_parse_rule(r) for r in d.get("rule", [])]
        if legacy:
            sets.setdefault("default", []).extend(legacy)
        default_sets = tuple(g.get("default_sets", list(sets)))
        return cls(
            rules=[r for s in default_sets for r in sets.get(s, [])],
            max_retries=int(g.get("max_retries", 6)),
            hold_slack=int(g.get("hold_slack", 16)),
            strip_chars=g.get("strip_chars", "*_~`"),
            sets=sets,
            default_sets=default_sets,
            scope=tuple(g.get("scope", ["content"])),
        )

    def resolve(self, request_guard: dict | None) -> GuardPolicy | None:
        """Effective policy for one request.

        request_guard (the request body's "guard" object):
          {"enabled": false}                    -> None (bypass entirely)
          {"sets": ["dashes", "style"]}         -> union of those named sets
          {"rules": [{...}, ...]}               -> inline ad-hoc rules (added)
        Omitted -> default sets. Unknown set names raise ValueError.
        """
        if request_guard is None:
            return self
        if request_guard.get("enabled") is False:
            return None
        names = request_guard.get("sets")
        if names is None:
            rules = list(self.rules)
        else:
            unknown = [n for n in names if n not in self.sets]
            if unknown:
                raise ValueError(f"unknown guard sets: {unknown}; available: {sorted(self.sets)}")
            rules = [r for n in names for r in self.sets[n]]
        rules += [_parse_rule(r) for r in request_guard.get("rules", [])]
        return GuardPolicy(
            rules=rules,
            max_retries=int(request_guard.get("max_retries", self.max_retries)),
            hold_slack=self.hold_slack,
            strip_chars=self.strip_chars,
            sets=self.sets,
            default_sets=self.default_sets,
            scope=tuple(request_guard.get("scope", self.scope)),
        )

    def compiled(self) -> list[tuple[Rule, re.Pattern]]:
        return [(r, r.compile()) for r in self.rules]

    @property
    def max_pattern_len(self) -> int:
        return max((len(r.pattern) for r in self.rules), default=0)

    @property
    def hold_window(self) -> int:
        """Raw characters to hold back so a forming match can never straddle
        the release boundary. Sized from each rule's max RAW span: base span
        (phrase gaps bounded, regex bounded) plus a strip-char inflation
        budget (hold_slack) for markup-fragmented evasions like '*t*h*a*t'."""
        base = max((r.base_span() for r in self.rules), default=0)
        return base + max(self.hold_slack, 1)

    def scan(self, text: str) -> Violation | None:
        """Earliest violation in text — scanned both raw and with formatting
        characters stripped (shadow), offsets mapped back to raw text."""
        views: list[tuple[str, list[int] | None]] = [(text, None)]
        if self.strip_chars and any(c in text for c in self.strip_chars):
            shadow_chars: list[str] = []
            offset_map: list[int] = []
            for i, ch in enumerate(text):
                if ch not in self.strip_chars:
                    shadow_chars.append(ch)
                    offset_map.append(i)
            views.append(("".join(shadow_chars), offset_map))
        best: tuple[int, Rule, str] | None = None
        for view_text, view_map in views:
            for rule, rx in self.compiled():
                m = rx.search(view_text)
                if m:
                    raw_start = view_map[m.start()] if view_map else m.start()
                    if best is None or raw_start < best[0]:
                        best = (raw_start, rule, m.group(0))
        if best is None:
            return None
        start, rule, fragment = best
        return Violation(rule=rule, fragment=fragment, offset=start)


def join_seam(emitted_prefix: str, continuation: str) -> str:
    """Whitespace management at a regeneration fracture point.

    The emitted prefix is already streamed to the client (irrevocable), so
    only the continuation side can be adjusted. The single failure mode
    handled: a space/tab-suffixed fracture meeting a space/tab-prefixed
    regenerated token would leak doubled spaces that neither the model nor
    the operator produced. Newline boundaries are preserved untouched
    (indentation after newlines is intentional model output)."""
    if emitted_prefix.endswith((" ", "\t")) and continuation.startswith((" ", "\t")):
        return continuation.lstrip(" \t")
    return continuation


class StreamGate:
    """Incremental gate over a token/text stream for one generation."""

    def __init__(self, policy: GuardPolicy):
        self.policy = policy
        self.buf = ""
        self.hold = policy.hold_window
        # Backoff cap: never withhold more than this (pathological all-space or
        # all-markup streams still flush). 4x the hold window is generous.
        self._backoff_cap = self.hold * 4
        self._strip = set(policy.strip_chars)

    def feed(self, text: str) -> tuple[str, Violation | None]:
        """Add text; return (provably-clean releasable text, violation or None).

        On violation, the returned text is the clean prefix BEFORE the match;
        the internal buffer resets (caller regenerates from there).
        """
        self.buf += text
        v = self.policy.scan(self.buf)
        if v is not None:
            clean = self.buf[: v.offset]
            self.buf = ""
            return clean, v
        # No violation: release everything except the hold window, and never
        # cut inside a run that could still be forming a match — word chars,
        # whitespace (phrase gaps), or strip chars (markup fragmentation) all
        # extend a potential match, so back off over all three (bounded).
        release_end = len(self.buf) - self.hold
        if release_end <= 0:
            return "", None
        floor = max(0, len(self.buf) - self._backoff_cap)
        while release_end > floor:
            ch = self.buf[release_end - 1]
            if ch.isalnum() or ch == "_" or ch.isspace() or ch in self._strip:
                release_end -= 1
            else:
                break
        out, self.buf = self.buf[:release_end], self.buf[release_end:]
        return out, None

    def flush(self) -> tuple[str, Violation | None]:
        """End of stream: scan and release whatever remains."""
        v = self.policy.scan(self.buf)
        if v is not None:
            clean = self.buf[: v.offset]
            self.buf = ""
            return clean, v
        out, self.buf = self.buf, ""
        return out, None
