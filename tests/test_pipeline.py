"""Pure-function tests for metadata/policy/perf/config logic (no network/GPU)."""

import pytest

from deepme import hfmeta, perfmodel
from deepme.policy import QuantPolicy, full_model_variants, gguf_bits

# ---- GGUF variant parsing (hfmeta._gguf_variants) --------------------------


def test_gguf_variants_merge_parts_and_drop_aux():
    siblings = [
        {"rfilename": "UD-Q8/model-00001-of-00002.gguf", "size": 10},
        {"rfilename": "UD-Q8/model-00002-of-00002.gguf", "size": 12},
        {"rfilename": "Q4_K_M.gguf", "size": 5},
        {"rfilename": "mmproj-model.gguf", "size": 1},  # aux: excluded
        {"rfilename": "model.imatrix.gguf", "size": 1},  # aux: excluded
    ]
    v = hfmeta._gguf_variants(siblings)
    assert v["UD-Q8/model.gguf"] == 22  # parts summed
    assert v["Q4_K_M.gguf"] == 5
    assert not any("mmproj" in k or "imatrix" in k for k in v)


# ---- quant-bit classification ---------------------------------------------


@pytest.mark.parametrize(
    "name,bits",
    [
        ("model.Q4_K_M.gguf", 4),
        ("UD-Q8_K_XL/model.gguf", 8),
        ("model.IQ4_XS.gguf", 4),
        ("model.Q6_K.gguf", 6),
        ("model.F16.gguf", 16),
        ("model.BF16.gguf", 16),
        ("model.IQ1_S.gguf", 1),
        ("model.MXFP4.gguf", 4),
    ],
)
def test_gguf_bits(name, bits):
    assert gguf_bits(name) == bits


def test_gguf_bits_unknown():
    assert gguf_bits("model.WEIRDQUANT.gguf") is None


# ---- policy allow/deny + min bits -----------------------------------------


def test_policy_denies_below_min_and_unknown():
    p = QuantPolicy(min_gguf_bits=6)
    assert p.variant_allowed("m.Q8_0.gguf")[0] is True
    assert p.variant_allowed("m.Q4_K_M.gguf")[0] is False
    assert p.variant_allowed("m.WHAT.gguf")[0] is False  # unknown -> denied


def test_policy_deny_pattern_matches_subdir_path():
    # regression: bare "IQ*" must match a variant with a subdir prefix
    p = QuantPolicy(min_gguf_bits=4, deny=("IQ*",))
    assert p.variant_allowed("UD-IQ4_XS/model.gguf")[0] is False
    assert p.variant_allowed("UD-Q4_K_M/model.gguf")[0] is True


def test_policy_allow_pattern_overrides_bits():
    p = QuantPolicy(min_gguf_bits=8, allow=("*MXFP6*",))
    assert p.variant_allowed("model.MXFP6.gguf")[0] is True


# ---- plausibility floor ----------------------------------------------------


def test_full_model_variants_drops_tiny_aux():
    variants = {"draft.gguf": 10 * 1024**3, "full.gguf": 150 * 1024**3}
    kept = full_model_variants(variants, total_params=300_000_000_000)
    # draft head (~0.27 bits/weight) dropped; full model kept
    assert "full.gguf" in kept
    assert "draft.gguf" not in kept


# ---- perf: active fraction -------------------------------------------------


def test_active_fraction_dense_is_one():
    assert perfmodel.active_fraction({}, 7_000_000_000) == 1.0


def test_active_fraction_moe_deepseek_like():
    cfg = {
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "n_shared_experts": 1,
        "hidden_size": 4096,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 43,
    }
    frac = perfmodel.active_fraction(cfg, 304_000_000_000)
    assert 0.03 < frac < 0.25  # small active fraction for a big sparse MoE


def test_estimate_decode_tps_none_when_unsizeable():
    assert perfmodel.estimate_decode_tps({}, 0, 0) is None
