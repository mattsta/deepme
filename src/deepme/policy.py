"""Quantization policy: which weight representations we consider servable.

One place to edit the allow/deny gates. Defaults: full-width formats and
GGUF quantizations of >= 6 bits per weight (Q6_K, Q8_0, F16/BF16/F32...)
are allowed; 4-bit and below are excluded globally (quality floor for our
usage). Variants whose bit-width can't be classified are DENIED — unknown
counts as unsupported, never silently allowed.

Override per machine in .deepme/config.toml:

    [policy]
    min_gguf_bits = 6
    deny = ["IQ*"]        # extra fnmatch-style deny patterns (checked first)
    allow = ["*MXFP6*"]   # extra allow patterns (checked before bit parsing)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fnmatch import fnmatch

from . import config

# Bit-width classification for GGUF variant filenames. Patterns are matched
# against the filename uppercased; first hit wins.
_BITS_PATTERNS: tuple[tuple[str, float], ...] = (
    (r"F32", 32),
    (r"BF16", 16),
    (r"F16", 16),
    (r"Q8_", 8),
    (r"Q6_", 6),
    (r"Q5_", 5),
    (r"Q4_", 4),
    (r"Q3_", 3),
    (r"Q2_", 2),
    (r"IQ4", 4),
    (r"IQ3", 3),
    (r"IQ2", 2),
    (r"IQ1", 1),
    (r"TQ1", 1),
    (r"TQ2", 2),
    (r"MXFP4", 4),
    (r"NVFP4", 4),
    (r"FP8", 8),
    (r"INT8", 8),
    (r"INT4", 4),
)


def gguf_bits(variant_name: str) -> float | None:
    """Best-effort bits-per-weight of a GGUF variant filename; None=unknown."""
    up = variant_name.upper()
    for pattern, bits in _BITS_PATTERNS:
        if re.search(pattern, up):
            return bits
    return None


@dataclass(frozen=True)
class QuantPolicy:
    min_gguf_bits: float = 6.0
    deny: tuple[str, ...] = ()  # fnmatch patterns, checked first
    allow: tuple[str, ...] = ()  # fnmatch patterns, checked before bit parse

    def variant_allowed(self, variant_name: str) -> tuple[bool, str]:
        """(allowed, reason). Unknown bit-widths are denied, with the reason."""
        up = variant_name.upper()
        for pat in self.deny:
            if fnmatch(up, pat.upper()):
                return False, f"denied by pattern {pat!r}"
        for pat in self.allow:
            if fnmatch(up, pat.upper()):
                return True, f"allowed by pattern {pat!r}"
        bits = gguf_bits(variant_name)
        if bits is None:
            return False, "unclassifiable quantization (unknown counts as unsupported)"
        if bits < self.min_gguf_bits:
            return False, f"{bits:g}-bit < policy minimum {self.min_gguf_bits:g}-bit"
        return True, f"{bits:g}-bit >= {self.min_gguf_bits:g}-bit"

    def filter_variants(self, variants: dict[str, int]) -> dict[str, int]:
        return {v: b for v, b in variants.items() if self.variant_allowed(v)[0]}


# Below ~0.5 bits/weight a file cannot contain the full model — repos bundle
# auxiliary GGUFs (draft/speculator heads, projectors) that must never be
# selected as "the model". Arithmetic gate, no name lists.
MIN_PLAUSIBLE_BPW = 0.5


def full_model_variants(variants: dict[str, int], total_params: int) -> dict[str, int]:
    """Drop variants too small to plausibly hold total_params weights."""
    if not total_params:
        return variants
    floor_bytes = total_params * MIN_PLAUSIBLE_BPW / 8
    return {v: b for v, b in variants.items() if b >= floor_bytes}


def current() -> QuantPolicy:
    """Active policy: defaults overridden by [policy] in .deepme/config.toml."""
    p = config._file_config().get("policy", {})
    return QuantPolicy(
        min_gguf_bits=float(p.get("min_gguf_bits", 6.0)),
        deny=tuple(p.get("deny", ())),
        allow=tuple(p.get("allow", ())),
    )
