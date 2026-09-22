"""Model profiles: config-driven multi-model deployment.

A profile is one TOML file in deploy/models/ describing a deployable model:
where its weights come from, which engine/GPUs/port serve it, its per-engine
launch args, and its sampling defaults. Commands resolve a profile (plus any
CLI overrides) into an immutable RunSpec that the serve/download/client
layers consume — nothing below this module knows model names.

The active default profile is persisted in .deepme/config.toml
([model] default = "...") via `deepme model use`.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass

from . import config, hfmeta

PROFILES_DIR = config.REPO_ROOT / "deploy" / "models"

GPU_VRAM_GB = 32  # reference hardware; used only for the pre-launch fit check


@dataclass(frozen=True)
class RunSpec:
    """Fully-resolved launch description for one model deployment."""

    profile: str  # profile name (file stem)
    repo: str
    served_name: str
    weight_gib: float  # from the HF API (cached), never hand-written
    shards: int  # safetensors count, from the HF API
    engine: str
    format: str  # "safetensors" | "gguf"
    variant: str  # gguf variant filename ("" = auto/only one)
    model_file: str  # resolved concrete file (gguf), "" for dirs
    tp: int
    gpus: tuple[int, ...]
    port: int
    max_len: int
    gpu_mem: float
    kv_ram_gb: int
    spec_decode: bool
    engine_args: tuple[str, ...]  # profile's args for the chosen engine
    extra_args: tuple[str, ...]  # one-off CLI additions
    temperature: float
    top_p: float

    @property
    def model_path(self) -> str:
        return f"{config.REMOTE_MODELS_DIR}/{self.repo.rsplit('/', 1)[-1]}"

    @property
    def unit(self) -> str:
        return f"deepme-llm-{self.profile}"

    @property
    def api_base(self) -> str:
        return f"http://{config.SSH_HOST}:{self.port}"

    def fit_check(self) -> str | None:
        """Cheap arithmetic sanity check before a 6-minute launch attempt.

        Weights shard across TP ranks; each rank also needs runtime overhead
        (activations, graphs, spec head). Returns a human-readable error, or
        None if the configuration plausibly fits.
        """
        budget_per_gpu = GPU_VRAM_GB * self.gpu_mem
        weights_per_gpu = self.weight_gib / max(self.tp, 1)
        overhead = 3.0  # GB: activations + CUDA graphs + fragmentation
        from . import engines as engines_mod

        err = engines_mod.get(self.engine).validate_spec(self)
        if err:
            return err
        if self.tp > len(self.gpus):
            return f"tp={self.tp} exceeds gpus={list(self.gpus)}"
        if weights_per_gpu + overhead > budget_per_gpu:
            return (
                f"~{weights_per_gpu:.1f}GB weights/GPU + ~{overhead:.0f}GB overhead "
                f"exceeds {budget_per_gpu:.1f}GB budget "
                f"({GPU_VRAM_GB}GB x {self.gpu_mem}) at tp={self.tp}"
            )
        return None


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.toml"))


def default_profile() -> str | None:
    cfg = config._file_config()
    return cfg.get("model", {}).get("default") or (
        list_profiles()[0] if len(list_profiles()) == 1 else None
    )


def set_default_profile(name: str) -> None:
    if name not in list_profiles():
        raise SystemExit(f"unknown profile {name!r}; available: {list_profiles()}")
    # rewrite config.toml preserving the [server] section
    lines = []
    if config.CONFIG_FILE.exists():
        current = tomllib.loads(config.CONFIG_FILE.read_text())
    else:
        current = {}
    server = current.get("server", {})
    if server:
        lines.append("[server]")
        for k, v in server.items():
            lines.append(f'{k} = "{v}"' if isinstance(v, str) else f"{k} = {v}")
        lines.append("")
    lines += ["[model]", f'default = "{name}"', ""]
    config.STATE_DIR.mkdir(exist_ok=True)
    config.CONFIG_FILE.write_text("\n".join(lines))


def load(name: str | None = None, **overrides) -> RunSpec:
    """Load a profile by name (or the persisted default) and apply overrides.

    Overrides use RunSpec field names; None values are ignored, so CLI
    options can be passed through unconditionally.
    """
    name = name or default_profile()
    if not name:
        raise SystemExit(
            f"no model profile selected — pass -m/--model or run: "
            f"uv run deepme model use <name>  (available: {list_profiles()})"
        )
    path = PROFILES_DIR / f"{name}.toml"
    if not path.exists():
        raise SystemExit(f"unknown profile {name!r}; available: {list_profiles()}")
    raw = tomllib.loads(path.read_text())
    model, serve_c = raw.get("model", {}), raw.get("serve", {})
    sampling = raw.get("sampling", {})
    ov = {k: v for k, v in overrides.items() if v is not None}
    engine = ov.get("engine", serve_c.get("engine", "vllm"))
    engine_args = serve_c.get("engine_args", {}).get(engine, [])
    meta = hfmeta.get(model["repo"])
    variant = str(model.get("variant", ""))
    # Format: explicit in profile, else derived from what the repo publishes
    # and what the chosen engine can serve.
    from . import engines as engines_mod

    fmt = model.get("format") or (
        "gguf"
        if (
            "gguf" in meta.formats
            and "safetensors" not in engines_mod.get(engine).formats
        )
        else (meta.formats[0] if meta.formats else "safetensors")
    )
    if fmt == "gguf" and meta.gguf_variants:
        if not variant:
            # Auto-selection respects the quant policy (explicit `variant` in
            # the profile is operator intent and is honored as written).
            from . import policy

            allowed = policy.current().filter_variants(
                policy.full_model_variants(meta.gguf_variants, meta.total_params)
            )
            if not allowed:
                raise SystemExit(
                    f"{meta.repo}: no gguf variant satisfies the quant policy "
                    f"(min {policy.current().min_gguf_bits:g}-bit) — set an "
                    "explicit `variant` in the profile to override"
                )
            variant = min(allowed, key=allowed.get)
        weight_gib = meta.gguf_variants.get(variant, 0) / 1024**3
        shards = 1
    else:
        weight_gib = meta.weight_gib
        shards = meta.weight_shards
    return RunSpec(
        profile=name,
        repo=model["repo"],
        served_name=model.get("served_name", name),
        weight_gib=weight_gib,
        shards=shards,
        engine=engine,
        format=fmt,
        variant=variant,
        model_file="",  # resolved at deploy time for gguf (serve.start)
        tp=int(ov.get("tp", serve_c.get("tp", 8))),
        gpus=tuple(ov.get("gpus", serve_c.get("gpus", list(range(8))))),
        port=int(ov.get("port", serve_c.get("port", 8000))),
        max_len=int(ov.get("max_len", serve_c.get("max_len", 131072))),
        gpu_mem=float(ov.get("gpu_mem", serve_c.get("gpu_mem", 0.90))),
        kv_ram_gb=int(ov.get("kv_ram_gb", serve_c.get("kv_ram_gb", 0))),
        spec_decode=bool(ov.get("spec_decode", serve_c.get("spec_decode", False))),
        engine_args=tuple(str(a) for a in engine_args),
        extra_args=tuple(ov.get("extra_args", ())),
        temperature=float(sampling.get("temperature", 1.0)),
        top_p=float(sampling.get("top_p", 1.0)),
    )
