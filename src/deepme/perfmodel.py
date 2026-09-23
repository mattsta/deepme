"""First-order decode-throughput estimates from model architecture.

Single-stream decode is memory-bandwidth-bound: every generated token must
read the model's ACTIVE weights once. So

    est_tok_s = effective_bandwidth / bytes_read_per_token
    bytes_read_per_token ~= weight_bytes x active_fraction

For dense models active_fraction = 1. For MoE, each token reads attention +
shared parts plus only top-k of E routed experts; the expert share of total
parameters is computed from config.json dims (hidden_size,
moe_intermediate_size, expert counts, layers).

Effective bandwidth is CALIBRATED, not theoretical: our production
measurement (DeepSeek-V4-Flash, TP=8 over PCIe, 87 tok/s at ~8.6GB active
bytes/token) implies ~750GB/s effective for PCIe tensor-parallel — 5% of
the 14.3TB/s aggregate HBM, i.e. allreduce-bound. Single-GPU models avoid
that entirely (~70% of one 5090's 1.79TB/s).

These are ranking heuristics (good to ~2x), not benchmarks — `deepme bench`
against a deployment is ground truth, and results land in the runs ledger.
"""

from __future__ import annotations

from . import config

GPU_HBM_GBS = 1790  # RTX 5090 memory bandwidth, GB/s
BW_SINGLE_GPU = GPU_HBM_GBS * 0.70  # typical achievable single-GPU efficiency
# MoE with expert parallelism: experts are read in parallel across GPUs, so
# multi-GPU decode is NOT allreduce-bound. CALIBRATED from production:
# DeepSeek-V4-Flash measured 87 tok/s at ~17.7GB active bytes/token
# (155GiB FP8 weights x ~0.11 active fraction) -> ~1540 GB/s effective.
BW_MULTI_GPU_MOE = 1540
# Dense tensor-parallel over PCIe pays the full allreduce tax every layer.
# ESTIMATED (no production dense-TP measurement yet — bench one to calibrate).
BW_MULTI_GPU_DENSE = 600


def per_gpu_budget_gib() -> float:
    """Usable VRAM per GPU for weights — single source is config (cluster
    shape x usable fraction - overhead)."""
    return config.per_gpu_weight_budget_gib()


# config.json field aliases across model families
_EXPERT_COUNT_KEYS = (
    "n_routed_experts",
    "num_local_experts",
    "num_experts",
    "moe_num_experts",
)
_TOPK_KEYS = ("num_experts_per_tok", "moe_top_k", "top_k", "num_selected_experts")
_SHARED_KEYS = ("n_shared_experts", "moe_num_shared_experts", "num_shared_experts")


def _first(cfg: dict, keys: tuple[str, ...], default=0) -> int:
    for k in keys:
        v = cfg.get(k)
        if isinstance(v, (int, float)) and v:
            return int(v)
    return default


def active_fraction(cfg: dict, total_params: int) -> float:
    """Fraction of weights read per token; 1.0 for dense models."""
    experts = _first(cfg, _EXPERT_COUNT_KEYS)
    topk = _first(cfg, _TOPK_KEYS)
    if not experts or not topk or not total_params:
        return 1.0
    hidden = int(cfg.get("hidden_size") or 0)
    moe_inter = int(cfg.get("moe_intermediate_size") or cfg.get("intermediate_size") or 0)
    layers = int(cfg.get("num_hidden_layers") or 0)
    if not (hidden and moe_inter and layers):
        # No dims to compute the expert share; assume experts dominate.
        return min(1.0, topk / experts + 0.05)
    # ROUTED-expert params only (shared experts + attention are in the dense
    # always-on remainder). Of `experts` routed experts, `topk` fire per token.
    expert_params_total = experts * 3 * hidden * moe_inter * layers
    expert_share = min(0.99, expert_params_total / total_params)
    active_expert_share = expert_share * (topk / experts)
    # The (1 - expert_share) dense remainder (attention + shared experts) is
    # always read; shared experts are NOT added to the routed numerator (that
    # would double-count them).
    return (1.0 - expert_share) + active_expert_share


# Decode-throughput estimates only make sense for autoregressive generators.
# Non-generative tasks (embeddings, similarity, classification, ...) produce
# vectors/labels, not token streams — estimating "tok/s" for them is
# misleading and is suppressed. Parameterized here; extend as needed.
NON_GENERATIVE_TASKS = frozenset(
    {
        "sentence-similarity",
        "feature-extraction",
        "text-classification",
        "token-classification",
        "zero-shot-classification",
        "fill-mask",
        "image-classification",
        "zero-shot-image-classification",
        "image-feature-extraction",
        "audio-classification",
        "text-ranking",
    }
)
_NON_GENERATIVE_ARCH_SUFFIXES = (
    "Model",
    "ForMaskedLM",
    "ForSequenceClassification",
    "ForTokenClassification",
    "ForQuestionAnswering",
    "ForReranking",
)


def is_generative(task: str, architectures: tuple[str, ...]) -> bool:
    if task in NON_GENERATIVE_TASKS:
        return False
    return not (
        architectures and all(a.endswith(_NON_GENERATIVE_ARCH_SUFFIXES) for a in architectures)
    )


def estimate_decode_tps(
    cfg: dict, total_params: int, weight_bytes: float
) -> tuple[float, int] | None:
    """(estimated single-stream tok/s, tp_needed) or None if unsizeable."""
    if not weight_bytes:
        return None
    import math

    tp = max(1, math.ceil(weight_bytes / (per_gpu_budget_gib() * 1024**3)))
    frac = active_fraction(cfg or {}, total_params)
    bytes_per_token = weight_bytes * frac
    is_moe = frac < 1.0
    bw = BW_SINGLE_GPU if tp == 1 else BW_MULTI_GPU_MOE if is_moe else BW_MULTI_GPU_DENSE
    return (bw * 1e9) / bytes_per_token, tp
