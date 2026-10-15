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

import json
import re
import tomllib
from dataclasses import dataclass

from . import config, hfmeta

PROFILES_DIR = config.REPO_ROOT / "deploy" / "models"

# Profile names become systemd unit names, tmux sessions, and remote paths —
# constrain them so nothing derived can inject shell or break unit syntax.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# GGUF variant paths come from HF filenames (repo-relative, may have a subdir).
_SAFE_VARIANT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*\.gguf$")
# Repo-relative partition directory: must stay inside the repo (no absolute
# paths, no traversal) since it reaches both a shell command and a filesystem
# path.
_SAFE_SUBFOLDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")
# systemd Environment= keys.
_SAFE_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ProfileError(Exception):
    """A profile cannot be selected, found or parsed.

    An ordinary exception: this is a library, and a caller surveying many
    profiles must be able to skip a bad one without catching
    BaseException. The CLI turns it into an exit code at its boundary.
    """


def _validate_name(name: str) -> str:
    if not _SAFE_NAME.match(name):
        raise ProfileError(
            f"invalid profile name {name!r}: allowed chars are letters, "
            "digits, dot, underscore, hyphen (must start alphanumeric)"
        )
    return name


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
    # Resource placement (see resources.py). CPU mode colocates alongside GPU
    # models: it claims no GPUs, only cores + RAM.
    mode: str = "gpu"  # gpu | cpu
    cpus: int = 0  # logical CPU reservation (0 = unpinned)
    ram_gib: float = 0.0  # RAM reservation/cap in GiB (0 = uncapped)
    # RAM a SUBSEQUENT replica needs, when weights are shared between them
    # (see hooks/deepme_hooks.py). The first replica pays for the weight
    # store; the rest map it and pay only private overhead. 0 = no sharing,
    # so every replica is charged the full ram_gib.
    extra_replica_ram_gib: float = 0.0
    # What the FIRST replica occupies at rest, when that differs from its cap.
    # Loads are staggered, so a load peak is transient and never overlaps
    # another profile's; charging it permanently reserves memory that is only
    # briefly used and refuses placements the machine can hold. 0 = charge the
    # cap, which is the safe default for a profile nobody has measured.
    resting_ram_gib: float = 0.0
    # What the ALLOCATOR charges this deployment, when that differs from its
    # cgroup cap. A replica sharing weights rests at a few GiB but still
    # materialises each component before offloading it, so its LOAD PEAK is
    # as large as ever. One number cannot do both jobs: the cap must cover
    # the peak or systemd OOM-kills it mid-load, while the allocator must
    # count the resting cost or it refuses replicas that comfortably fit.
    claim_ram_gib: float = 0.0

    @property
    def accounted_ram_gib(self) -> float:
        """What the allocator charges — resting size, never the load peak."""
        if self.claim_ram_gib:
            return self.claim_ram_gib
        return self.resting_ram_gib or self.ram_gib

    # What KIND of model this is (llm | diffusion) — decides the engine's
    # serving path and the API surface. Detected from repo facts, overridable.
    kind: str = "llm"
    # Repo subfolder holding the servable pipeline. Some repos ship several
    # task partitions (MiniMax-H3: FL2VA/, Ref2VA/), each a complete pipeline
    # with its own model_index.json; the root is only a container.
    subfolder: str = ""
    # Environment for THIS deployment's unit. Engine behaviour that is only
    # reachable by env var (loader strategies, kernel toggles) belongs here
    # rather than in the engine's global runtime_env, which every deployment
    # of that engine would inherit.
    env: tuple[tuple[str, str], ...] = ()
    # Attention heads from the repo's config.json (0 = unknown). Tensor
    # parallelism splits heads across ranks, so tp must divide this.
    attn_heads: int = 0
    # Smallest usable denoise step count (0 = no declared floor). Below a
    # model's floor the schedule degenerates and the engine fails with an
    # error that names the symptom, not the cause.
    min_steps: int = 0
    # Which copy of this model this is. Replicas are how throughput is bought
    # for engines that serve one request at a time (see balancer/).
    replica: int = 0
    # Generation tasks this deployment accepts. A model's capabilities can be
    # split across checkpoints (H3: FL2VA serves t2va+fl2va, Ref2VA serves
    # ref2va), so the profile — which knows which partition it loaded — is the
    # only place that can state this. The gateway builds its routing table
    # from it rather than hardcoding a model's topology.
    tasks: tuple[str, ...] = ()
    # Request-body defaults for generation, as JSON (see `gen_defaults`).
    # Media models take model-specific fields the platform should not have to
    # enumerate — MiniMax-H3 rejects any request without a `task`. Held as a
    # string so the spec stays immutable and serializes into the ledger.
    gen_defaults_json: str = "{}"

    @property
    def download_path(self) -> str:
        """Where weights land — always the repo root, never a partition."""
        return f"{config.REMOTE_MODELS_DIR}/{self.repo.rsplit('/', 1)[-1]}"

    @property
    def model_path(self) -> str:
        """The path argument for this engine.

        Engines that select a partition by name (Feature.MODEL_SUBFOLDER) must
        receive the repo ROOT — they resolve the subfolder themselves and their
        loaders depend on that framing. Engines without it get the partition
        path joined on, which is the only thing they can act on.
        """
        from . import engines as engines_mod

        if not self.subfolder:
            return self.download_path
        edef = engines_mod.get(self.engine)
        if edef.supports(engines_mod.Feature.MODEL_SUBFOLDER):
            return self.download_path
        return f"{self.download_path}/{self.subfolder}"

    @property
    def gen_defaults(self) -> dict:
        """Profile-declared fields merged into every generation request."""
        return dict(json.loads(self.gen_defaults_json))

    @property
    def unit(self) -> str:
        """Replica 0 keeps the plain unit name so single deployments — and
        every unit already on a server — are unaffected by scaling existing."""
        base = f"deepme-llm-{self.profile}"
        return base if self.replica == 0 else f"{base}-r{self.replica}"

    @property
    def api_base(self) -> str:
        return f"http://{config.SSH_HOST}:{self.port}"

    def fit_check(self) -> str | None:
        """Cheap arithmetic sanity check before a launch attempt.

        Weight memory is divided by the number of GPUs the engine actually
        spreads across: tp ranks for tensor-parallel engines (vllm/sglang),
        or len(gpus) for layer-split engines (llama.cpp — tp is inert there).
        Returns a human-readable error, or None if it plausibly fits.
        """
        from . import engines as engines_mod

        edef = engines_mod.get(self.engine)
        err = edef.validate_spec(self)
        if err:
            return err
        if self.mode == "cpu":
            # CPU inference is bounded by system RAM, not VRAM.
            budget = self.ram_gib or (config.cluster_ram_gib() - 32)
            if self.weight_gib > budget:
                return (
                    f"~{self.weight_gib:.0f}GiB weights exceed the "
                    f"{budget:.0f}GiB CPU RAM budget for this profile"
                )
            return None
        if self.tp > len(self.gpus):
            return f"tp={self.tp} exceeds gpus={list(self.gpus)}"
        # Tensor parallelism splits attention heads across ranks; an uneven
        # split is rejected by the engine only after the weights are loaded.
        if (
            self.attn_heads
            and self.tp > 1
            and self.attn_heads % self.tp
            and edef.supports(engines_mod.Feature.TENSOR_PARALLEL)
        ):
            return (
                f"tp={self.tp} does not divide {self.attn_heads} attention "
                f"heads — usable tp values: "
                f"{[n for n in range(1, len(self.gpus) + 1) if self.attn_heads % n == 0]}"
            )
        # The engine decides where its weights end up — replicated parallelism
        # needs the whole model per GPU even with 8 of them, and offloaded
        # weights are bounded by HOST memory instead (see WeightPlacement).
        plan = edef.placement_for(self)
        if not plan.resident:
            # A later replica with a shared weight store maps the copy that
            # replica 0 already built, so it pays private overhead only.
            # Charging it the full weight cost is what refuses replicas 2..N
            # when the memory is demonstrably there.
            if self.replica and self.extra_replica_ram_gib:
                return None
            need = self.weight_gib * max(plan.host_copies, 1)
            budget = self.ram_gib or (config.cluster_ram_gib() - 32)
            if need > budget:
                return (
                    f"~{need:.0f}GiB of host RAM needed ({plan.how}, "
                    f"{plan.host_copies} ranks x {self.weight_gib:.0f}GiB) "
                    f"exceeds the {budget:.0f}GiB budget"
                )
            return None
        divisor, how = plan.divisor, plan.how
        # Layer split is uneven (discrete layers + KV buffers) — derate.
        derate = 1.0 if edef.supports(engines_mod.Feature.TENSOR_PARALLEL) else 0.85
        budget_per_gpu = config.per_gpu_weight_budget_gib(self.gpu_mem) * derate
        weights_per_gpu = self.weight_gib / divisor
        if weights_per_gpu > budget_per_gpu:
            return (
                f"~{weights_per_gpu:.1f}GiB weights/GPU exceeds "
                f"~{budget_per_gpu:.1f}GiB budget ({how}, "
                f"{config.cluster_vram_gib():.0f}GiB x {self.gpu_mem})"
            )
        return None


def list_profiles() -> list[str]:
    return sorted(p.stem for p in PROFILES_DIR.glob("*.toml"))


def diffusion_profiles() -> list[str]:
    """Profiles that can generate media, read from DISK ONLY.

    Deliberately does not go through `load()`: that resolves Hugging Face
    metadata over the network (30s uncached) to determine `kind`, which is
    absurd cost for "which profiles could the user have meant?" — a question
    asked precisely when the command is already known to be wrong. A
    diffusion profile is one that declares generation `tasks` or a
    `[generate]` block; a text profile declares neither.
    """
    out = []
    for path in sorted(PROFILES_DIR.glob("*.toml")):
        try:
            data = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if data.get("tasks") or data.get("generate"):
            out.append(path.stem)
    return out


def default_profile() -> str | None:
    return config.section("model").get("default") or (
        list_profiles()[0] if len(list_profiles()) == 1 else None
    )


def set_default_profile(name: str) -> None:
    if name not in list_profiles():
        raise ProfileError(f"unknown profile {name!r}; available: {list_profiles()}")
    config.update_config("model", default=name)


def load(name: str | None = None, **overrides) -> RunSpec:
    """Load a profile by name (or the persisted default) and apply overrides.

    Overrides use RunSpec field names; None values are ignored, so CLI
    options can be passed through unconditionally.

    Raises ProfileError — never SystemExit. Callers that survey MANY
    profiles (fleet status, top) have to tolerate one unloadable profile,
    and they were doing it with `suppress(Exception, SystemExit)`, which
    also swallowed the interrupt protocol's own escalation: a ^C landing
    inside a watch loop's per-profile load was silently discarded.
    """
    name = name or default_profile()
    if not name:
        raise ProfileError(
            f"no model profile selected — pass -m/--model or run: "
            f"uv run deepme model use <name>  (available: {list_profiles()})"
        )
    _validate_name(name)
    path = PROFILES_DIR / f"{name}.toml"
    if not path.exists():
        raise ProfileError(f"unknown profile {name!r}; available: {list_profiles()}")
    raw = tomllib.loads(path.read_text())
    model, serve_c = raw.get("model", {}), raw.get("serve", {})
    sampling = raw.get("sampling", {})
    res_c = raw.get("resources", {})  # mode/cpus/ram_gb — see resources.py
    ov = {k: v for k, v in overrides.items() if v is not None}
    engine = ov.get("engine", serve_c.get("engine", "vllm"))
    engine_args = serve_c.get("engine_args", {}).get(engine, [])
    meta = hfmeta.get(model["repo"])
    variant = str(model.get("variant", ""))
    if variant and not _SAFE_VARIANT.match(variant):
        raise ProfileError(f"invalid gguf variant path {variant!r} in profile {name!r}")
    # Unit environment: systemd takes one KEY=VALUE per line, so a newline in
    # either half would inject directives into the generated unit.
    env_raw = serve_c.get("env", {})
    for k, v in env_raw.items():
        if not _SAFE_ENV_KEY.match(k) or "\n" in str(v):
            raise ProfileError(
                f"invalid [serve.env] entry {k!r} in profile {name!r}: keys must "
                "be A-Z/0-9/underscore and values single-line"
            )
    env = tuple((str(k), str(v)) for k, v in sorted(env_raw.items()))
    subfolder = str(ov.get("subfolder", model.get("subfolder", "")))
    if subfolder and (not _SAFE_SUBFOLDER.match(subfolder) or ".." in subfolder.split("/")):
        raise ProfileError(
            f"invalid subfolder {subfolder!r} in profile {name!r}: must be a "
            "repo-relative directory (no leading slash, no '..')"
        )
    # Format: explicit in profile, else derived from what the repo publishes
    # and what the chosen engine can serve.
    from . import engines as engines_mod

    fmt = model.get("format") or (
        "gguf"
        if ("gguf" in meta.formats and "safetensors" not in engines_mod.get(engine).formats)
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
                raise ProfileError(
                    f"{meta.repo}: no gguf variant satisfies the quant policy "
                    f"(min {policy.current().min_gguf_bits:g}-bit) — set an "
                    "explicit `variant` in the profile to override"
                )
            variant = min(allowed, key=lambda k: allowed[k])
        weight_gib = meta.gguf_variants.get(variant, 0) / 1024**3
        shards = 1
    else:
        # ONE deployment's size, not the whole repo (see RepoMeta.deployable_gib)
        weight_gib = meta.deployable_gib
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
        mode=str(ov.get("mode", res_c.get("mode", "gpu"))),
        cpus=int(ov.get("cpus", res_c.get("cpus", 0))),
        ram_gib=float(ov.get("ram_gib", res_c.get("ram_gb", 0))),
        extra_replica_ram_gib=float(res_c.get("ram_gb_extra_replica", 0)),
        resting_ram_gib=float(res_c.get("ram_gb_resting", 0)),
        # Detected from repo facts; a profile may override deliberately.
        kind=str(ov.get("kind", model.get("kind", meta.kind))),
        subfolder=subfolder,
        env=env,
        attn_heads=int((meta.config_dims or {}).get("num_attention_heads", 0)),
        tasks=tuple(str(t) for t in serve_c.get("tasks", ())),
        gen_defaults_json=json.dumps(raw.get("generate", {}), sort_keys=True),
        min_steps=int(raw.get("limits", {}).get("min_steps", 0)),
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
