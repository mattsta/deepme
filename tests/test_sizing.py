"""Weight-size accounting: each estimate has a different overestimation mode.

Regression tests from the MiniMax-H3 audit: a 33B diffusers pipeline was
reported as 464GiB (the whole repo: transformer + transformer_ref + vae +
text_encoder + audio_vae + FL2VA + Ref2VA, 104 files) instead of ~62GiB for
one instance of the weights.
"""

from deepme import hfmeta


def test_dtype_key_prefers_longest_prefix():
    # BF16 must not be read as "B"+F16 or fall through to F16's 2 bytes by
    # accident; F8/F4 must not collide.
    assert hfmeta._dtype_key("BF16") == "BF16"
    assert hfmeta._dtype_key("F16") == "F16"
    assert hfmeta._dtype_key("F8_E4M3") == "F8"
    assert hfmeta._dtype_key("F4") == "F4"
    assert hfmeta._dtype_key("F32") == "F32"


def test_param_bytes_matches_hand_arithmetic():
    counts = {"BF16": 33_105_770_752, "F32": 17_222_144}
    expected = 33_105_770_752 * 2 + 17_222_144 * 4
    got = sum(n * hfmeta._DTYPE_BYTES_PER_PARAM[hfmeta._dtype_key(dt)] for dt, n in counts.items())
    assert got == expected
    assert 60 < got / 1024**3 < 63  # ~62GiB for a 33B BF16 model


def min_estimate(file_gib: float, param_gib: float) -> float:
    """The rule catalog uses: one deployment is no larger than either bound."""
    candidates = [g for g in (file_gib, param_gib) if g > 0]
    return min(candidates) if candidates else 0.0


def test_multi_component_repo_uses_param_math():
    # MiniMax-H3: file sum covers many components; param math is the instance.
    assert min_estimate(464.1, 61.7) == 61.7


def test_sub_byte_packed_repo_uses_file_bytes():
    # DeepSeek-V4-Flash: param math over-counts FP4-packed experts; the files
    # are the truth.
    assert min_estimate(155.4, 283.5) == 155.4


def test_missing_one_estimate_falls_back_to_the_other():
    assert min_estimate(0.0, 61.7) == 61.7
    assert min_estimate(155.4, 0.0) == 155.4
    assert min_estimate(0.0, 0.0) == 0.0
