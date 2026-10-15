"""Partitioned repos: who resolves the subfolder, us or the engine.

Some repos ship several task partitions (MiniMax-H3: FL2VA/, Ref2VA/), each a
complete pipeline. Engines that select a partition by name must be handed the
repo ROOT plus the name — pointing them at the partition directory trips their
own model-completeness pre-checks. Engines without that capability can only act
on a path, so we join it for them.
"""

import dataclasses
import re

import pytest

from deepme import engines
from deepme.profiles import RunSpec

BASE = {
    # A real profile name: ports are declared per profile in
    # deploy/ports.toml, and an undeclared one must fail rather than guess.
    "profile": "h3-video",
    "repo": "Org/Repo-X",
    "served_name": "p",
    "weight_gib": 10.0,
    "shards": 4,
    "engine": "sglang",
    "format": "safetensors",
    "variant": "",
    "model_file": "",
    "tp": 4,
    "gpus": (0, 1, 2, 3),
    "port": 8030,
    "max_len": 131072,
    "gpu_mem": 0.9,
    "kv_ram_gb": 0,
    "spec_decode": False,
    "engine_args": (),
    "extra_args": (),
    "temperature": 1.0,
    "top_p": 1.0,
    "kind": "diffusion",
    "subfolder": "FL2VA",
}


def spec(**over) -> RunSpec:
    return RunSpec(**{**BASE, **over})


def test_download_path_is_always_the_repo_root():
    """Weights are fetched whole; a partition is a view of them, not a target."""
    assert spec().download_path.endswith("/Repo-X")
    assert spec(subfolder="").download_path == spec().download_path


def test_subfolder_capable_engine_gets_the_root_and_the_name():
    s = spec()
    assert s.model_path == s.download_path
    cmd = engines.get("sglang").build_command(s, "key")
    assert "--model-subfolder" in cmd
    assert cmd[cmd.index("--model-subfolder") + 1] == "FL2VA"
    assert cmd[cmd.index("--model-path") + 1] == s.download_path


def test_engine_without_the_capability_gets_a_joined_path():
    edef = engines.get("vllm")
    assert not edef.supports(engines.Feature.MODEL_SUBFOLDER)
    s = spec(engine="vllm", kind="llm")
    assert s.model_path == f"{s.download_path}/FL2VA"


def test_no_subfolder_leaves_paths_and_command_untouched():
    s = spec(subfolder="")
    assert s.model_path == s.download_path
    assert "--model-subfolder" not in engines.get("sglang").build_command(s, "key")


@pytest.mark.parametrize("bad", ["/etc", "../secrets", "FL2VA/../..", "a b"])
def test_traversal_and_injection_are_rejected_at_load(bad, tmp_path, monkeypatch):
    from deepme import config, hfmeta, profiles

    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(
        hfmeta,
        "get",
        lambda repo, **kw: hfmeta.RepoMeta(
            repo=repo, weight_bytes=1, weight_shards=1, total_bytes=1
        ),
    )
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    (tmp_path / "m.toml").write_text(f'[model]\nrepo = "Org/Repo-X"\nsubfolder = "{bad}"\n')
    with pytest.raises(profiles.ProfileError, match="invalid subfolder"):
        profiles.load("m")


def test_runspec_stays_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec().subfolder = "Ref2VA"  # type: ignore[misc]


def test_diffusion_without_sharding_needs_the_whole_model_per_gpu():
    """Ulysses sequence parallelism splits the SEQUENCE, not the weights."""
    plan = engines.get("sglang").placement_for(spec())
    assert plan.divisor == 1 and plan.resident
    assert "replicated" in plan.how


def test_fsdp_flag_makes_the_weights_divide_across_ranks():
    plan = engines.get("sglang").placement_for(spec(engine_args=("--use-fsdp-inference", "true")))
    assert plan.divisor == 4 and plan.resident
    assert "fsdp" in plan.how


def test_layerwise_dit_offload_moves_the_budget_to_host_ram():
    """The vendor's 5090 recipe: the DiT never becomes GPU-resident."""
    plan = engines.get("sglang").placement_for(
        spec(gpus=(6, 7), engine_args=("--layerwise-offload-components", "dit,text_encoder,vae"))
    )
    assert not plan.resident
    assert plan.host_copies == 2


def test_offloading_only_the_encoder_leaves_the_dit_on_the_gpu():
    plan = engines.get("sglang").placement_for(
        spec(engine_args=("--layerwise-offload-components", "text_encoder,vae"))
    )
    assert plan.resident


def test_fit_check_rejects_an_unshardable_diffusion_model():
    """62GiB of DiT replicated onto 32GiB cards must fail BEFORE launch."""
    assert spec(weight_gib=62.0).fit_check() is not None
    assert spec(weight_gib=62.0, engine_args=("--use-fsdp-inference", "true")).fit_check() is None


def test_offloaded_fit_check_is_against_host_ram():
    offload = ("--layerwise-offload-components", "dit,text_encoder,vae")
    # Two ranks x 62GiB fits a 200GiB budget; a 100GiB budget does not.
    assert (
        spec(weight_gib=62.0, gpus=(6, 7), tp=2, engine_args=offload, ram_gib=200.0).fit_check()
        is None
    )
    err = spec(weight_gib=62.0, gpus=(6, 7), tp=2, engine_args=offload, ram_gib=100.0).fit_check()
    assert err is not None and "host RAM" in err


def test_llm_path_still_divides_by_tp():
    plan = engines.get("sglang").placement_for(spec(kind="llm"))
    assert (plan.divisor, plan.how) == (4, "tp=4")


def test_profile_env_reaches_the_unit(tmp_path, monkeypatch):
    """Per-deployment env must land in the unit and override engine defaults."""
    from deepme import serve

    s = spec(env=(("SGLANG_USE_RUNAI_MODEL_STREAMER", "0"),), ram_gib=420.0)
    monkeypatch.setattr(serve, "build_command", lambda _s: ["/bin/true"])
    unit = serve.render_unit(s)
    assert 'Environment="SGLANG_USE_RUNAI_MODEL_STREAMER=0"' in unit
    # Profile env must come after the engine/platform defaults to win.
    assert unit.index("SGLANG_USE_RUNAI") > unit.index("CUDA_VISIBLE_DEVICES")
    assert "MemoryMax=420G" in unit
    assert "X-DEEPME-GPUS=0,1,2,3" in unit


@pytest.mark.parametrize("bad_key", ["BAD KEY", "9LEADING", "WITH-DASH"])
def test_malformed_env_keys_are_rejected(bad_key, tmp_path, monkeypatch):
    from deepme import config, hfmeta, profiles

    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(
        hfmeta,
        "get",
        lambda repo, **kw: hfmeta.RepoMeta(
            repo=repo, weight_bytes=1, weight_shards=1, total_bytes=1
        ),
    )
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    (tmp_path / "m.toml").write_text(
        f'[model]\nrepo = "Org/Repo-X"\n\n[serve.env]\n"{bad_key}" = "1"\n'
    )
    with pytest.raises(profiles.ProfileError, match=re.escape("invalid [serve.env] entry")):
        profiles.load("m")


def test_tp_must_divide_the_attention_heads():
    """An uneven head split is only rejected by the engine AFTER loading."""
    err = spec(kind="llm", attn_heads=64, tp=6, gpus=(0, 1, 2, 3, 4, 5)).fit_check()
    assert err is not None and "attention heads" in err
    assert "[1, 2, 4]" in err  # usable tp values within 6 GPUs
    assert spec(kind="llm", attn_heads=64, tp=4).fit_check() is None


def test_unknown_head_count_does_not_block_a_launch():
    """Repos without a config.json must stay deployable."""
    assert spec(kind="llm", attn_heads=0, tp=3, gpus=(0, 1, 2)).fit_check() is None


def test_generate_defaults_come_from_the_profile(tmp_path, monkeypatch):
    """Model-required request fields belong to the deployment, not the code."""
    from deepme import config, hfmeta, profiles

    monkeypatch.setattr(profiles, "PROFILES_DIR", tmp_path)
    monkeypatch.setattr(
        hfmeta,
        "get",
        lambda repo, **kw: hfmeta.RepoMeta(
            repo=repo, weight_bytes=1, weight_shards=1, total_bytes=1
        ),
    )
    monkeypatch.setattr(config, "STATE_DIR", tmp_path)
    (tmp_path / "m.toml").write_text(
        '[model]\nrepo = "Org/Repo-X"\n\n[generate]\ntask = "t2va"\nconditions = []\n'
    )
    assert profiles.load("m").gen_defaults == {"task": "t2va", "conditions": []}


def test_no_generate_section_yields_no_defaults():
    assert spec().gen_defaults == {}
