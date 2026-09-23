"""Engine registry: inference engines as declared capabilities, not code paths.

Every engine is an EngineDef describing what it can serve (weight formats,
whether it has an introspectable architecture registry), how its dependencies
are provisioned (a uv project under deploy/engines/<name>/ plus optional sync
env), and how to build its launch command from a RunSpec. The rest of the
platform — catalog compatibility, provisioning, serving — consults the
registry; adding engine N+1 is one EngineDef + one pyproject, no platform
edits.

Compatibility is a cross-product decided from attributes:
    model(format, architectures, per-variant sizes) x engine(formats, archs)
Any cardinality on either side.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .profiles import RunSpec


class Feature(enum.Flag):
    """Capabilities an engine's serving API provides. Declared ONCE here;
    every subsystem (guard adapters, serve builders, catalog) consults this
    matrix instead of hard-coding engine names."""

    NONE = 0
    CONTINUATION = enum.auto()  # resume a partial assistant message
    BAD_WORDS = enum.auto()  # sampler-level multi-token sequence bans
    LOGIT_BIAS = enum.auto()  # per-token-id bias
    TOKENIZE = enum.auto()  # tokenizer endpoint (needed for logit bias)
    KV_RAM_TIER = enum.auto()  # KV cache tiering into system RAM
    SPEC_DECODE = enum.auto()  # speculative decoding
    EXPERT_PARALLEL = enum.auto()
    REASONING_PARSER = enum.auto()
    TOOL_PARSER = enum.auto()
    TENSOR_PARALLEL = enum.auto()  # shards each tensor across tp ranks
    #   (absent: engine splits by layer across visible GPUs, e.g. llama.cpp)
    MODEL_SUBFOLDER = enum.auto()  # selects a repo partition by name, so the
    #   path argument stays the repo ROOT (engines without this get the
    #   subfolder joined onto the path instead — see RunSpec.model_path)

    def names(self) -> tuple[str, ...]:
        return tuple(
            f.name.lower()
            for f in Feature
            if f is not Feature.NONE and f in self and f.name is not None
        )


@dataclass(frozen=True)
class WeightPlacement:
    """Where one copy of the weights lives once the engine is running.

    A fit check is only meaningful against this: dividing model size by GPU
    count assumes sharding, which several parallelism modes do not do.
    """

    divisor: int  # GPUs one resident copy is divided across
    how: str  # operator-readable explanation, shown in fit-check errors
    resident: bool = True  # False: weights stream from host RAM per layer
    host_copies: int = 0  # copies held in host RAM when not resident


# Coarse classes of model the platform deploys. An engine declares which it
# can serve; a repo's facts determine which it IS (see profiles/catalog).
KIND_LLM = "llm"  # autoregressive text/multimodal chat
KIND_DIFFUSION = "diffusion"  # image/video generation pipelines
MODEL_KINDS = (KIND_LLM, KIND_DIFFUSION)


@dataclass(frozen=True)
class EngineDef:
    name: str
    formats: tuple[str, ...]  # weight formats it can serve
    has_arch_registry: bool  # True: arch support is introspectable
    # Model kinds this engine can serve. Probed per-install where the engine
    # exposes a registry (sglang's multimodal_gen) — see list_kinds.py.
    kinds: tuple[str, ...] = (KIND_LLM,)
    features: Feature = Feature.NONE  # declared capability matrix
    experimental: bool = False  # not yet benchmarked/production-proven
    sync_env: dict[str, str] = field(default_factory=dict)  # env for `uv sync`
    # Engines that COMPILE against CUDA at install time need the toolkit
    # present before their own build runs — but the toolkit arrives as a
    # dependency of the same project (chicken-and-egg). Name a dependency
    # GROUP here (declared in the engine's pyproject): it is synced first,
    # then the main sync builds against it. No ad-hoc pip installs.
    sync_prepare_group: str = ""
    # Extra environment for the SERVING unit ({cuda} -> venv CUDA toolkit).
    # Engines that link CUDA libs from the venv (rather than a system install)
    # need them on the loader path at runtime.
    runtime_env: dict[str, str] = field(default_factory=dict)
    # (spec, api_key) -> argv. Every registered engine defines one.
    build_command: Callable[[RunSpec, str], list[str]] | None = None
    # (spec) -> WeightPlacement. Only for engines whose placement depends on
    # more than the feature flags — see placement_for() for the default.
    weight_split: Callable[[RunSpec], WeightPlacement] | None = None

    def supports(self, feature: Feature) -> bool:
        return feature in self.features

    def placement_for(self, spec: RunSpec) -> WeightPlacement:
        """Where this deployment's weights end up.

        Default: tensor-parallel engines divide by tp; the rest split by
        layer across the visible GPUs (llama.cpp).
        """
        if self.weight_split is not None:
            return self.weight_split(spec)
        if self.supports(Feature.TENSOR_PARALLEL):
            return WeightPlacement(max(spec.tp, 1), f"tp={spec.tp}")
        n = max(len(spec.gpus), 1)
        return WeightPlacement(n, f"layer-split across {n} GPUs")

    @property
    def venv(self) -> str:
        return config.engine_venv(self.name)

    @property
    def python(self) -> str:
        return f"{self.venv}/bin/python"

    def validate_spec(self, spec: RunSpec) -> str | None:
        """Attribute-level validation before any launch attempt."""
        if spec.format not in self.formats:
            return f"engine {self.name} serves {self.formats}, but model format is {spec.format!r}"
        if spec.kind not in self.kinds:
            return (
                f"engine {self.name} serves {self.kinds} models, "
                f"but this model is kind {spec.kind!r}"
            )
        return None


# ---------------------------------------------------------------------------
# launch command builders (the ONLY engine-specific code in the platform)
# ---------------------------------------------------------------------------


def _vllm_cmd(spec: RunSpec, key: str) -> list[str]:
    cmd = [
        f"{config.engine_venv('vllm')}/bin/vllm",
        "serve",
        spec.model_path,
        "--served-model-name",
        spec.served_name,
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(spec.tp),
        "--enable-expert-parallel",
        "--max-model-len",
        str(spec.max_len),
        "--gpu-memory-utilization",
        str(spec.gpu_mem),
        "--kv-cache-dtype",
        "fp8",
        "--host",
        "0.0.0.0",
        "--port",
        str(spec.port),
        "--api-key",
        key,
    ]
    if spec.spec_decode:
        cmd += [
            "--speculative-config",
            json.dumps(
                {
                    "method": "dspark",
                    "num_speculative_tokens": 7,
                    "draft_sample_method": "greedy",
                }
            ),
        ]
    if spec.kv_ram_gb:
        cmd += [
            "--kv-transfer-config",
            json.dumps(
                {
                    "kv_connector": "OffloadingConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {"cpu_bytes_to_use": spec.kv_ram_gb * 1024**3},
                }
            ),
        ]
    return cmd


def _sglang_diffusion_cmd(spec: RunSpec, key: str) -> list[str]:
    """Diffusion (image/video) serving via sglang.multimodal_gen.

    `sglang serve` auto-detects diffusion from the model path; we pass
    --model-type explicitly so a detection miss fails loudly instead of
    silently starting an LLM server.

    Partitioned repos are selected with --model-subfolder, NOT by pointing
    --model-path at the partition directory: the loader's completeness
    pre-check globs component dirs non-recursively, so a component that keeps
    its weights one level down (MiniMax-H3's video_vae/source/) reads as
    "missing" and the server tries to re-download the local path as if it
    were a repo id. Passing the root + subfolder skips that check entirely.
    """
    return [
        f"{config.engine_venv('sglang')}/bin/sglang",
        "serve",
        "--model-path",
        spec.model_path,
        *(["--model-subfolder", spec.subfolder] if spec.subfolder else []),
        "--model-type",
        "diffusion",
        "--num-gpus",
        str(len(spec.gpus) or spec.tp),
        "--host",
        "0.0.0.0",
        "--port",
        str(spec.port),
        # Without this the server AUTO-SELECTS another port when the requested
        # one is busy and reports success. Everything here addresses replicas
        # by their assigned port — health checks, the balancer's backends, the
        # allocator's accounting — so a drifted replica serves happily on a
        # port nobody knows and reads as permanently unhealthy. Fail loudly.
        "--strict-ports",
        "true",
        # Every replica runs its own torch.distributed rendezvous and internal
        # scheduler. These default to fixed ports, so without distinct ones a
        # second replica dies with EADDRINUSE. Declared per profile AND per
        # replica in deploy/ports.toml: deriving them from the replica index
        # alone collided across profiles (h3-video r0 and h3-ref r0 both
        # wanted 5644), which strict-ports turned into a loud failure.
        "--master-port",
        str(_registry().master_port(spec.profile, spec.replica)),
        "--scheduler-port",
        str(_registry().scheduler_port(spec.profile, spec.replica)),
    ]


def _sglang_cmd(spec: RunSpec, key: str) -> list[str]:
    if spec.kind == KIND_DIFFUSION:
        return _sglang_diffusion_cmd(spec, key)
    # gguf: point at the resolved variant FILE and tell the loader explicitly
    # (support PROBED on the installed build: --load-format gguf choice).
    model_arg = spec.model_file if spec.format == "gguf" else spec.model_path
    cmd = [
        config.venv_python("sglang"),
        "-m",
        "sglang.launch_server",
        "--model-path",
        model_arg,
        *(["--load-format", "gguf"] if spec.format == "gguf" else []),
        "--served-model-name",
        spec.served_name,
        "--trust-remote-code",
        "--tp",
        str(spec.tp),
        "--context-length",
        str(spec.max_len),
        "--mem-fraction-static",
        str(spec.gpu_mem),
        "--host",
        "0.0.0.0",
        "--port",
        str(spec.port),
        "--api-key",
        key,
    ]
    if spec.spec_decode:
        cmd += ["--speculative-algorithm", "DSPARK"]
    if spec.kv_ram_gb:
        cmd += ["--enable-hierarchical-cache", "--hicache-size", str(spec.kv_ram_gb)]
    return cmd


def _registry():
    """The declared port registry (deploy/ports.toml)."""
    from . import ports

    return ports.load()


def _flag_value(args: tuple[str, ...], flag: str) -> str:
    """Value following `flag` in an argv list ('' if absent or trailing)."""
    for i, a in enumerate(args):
        if a == flag and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(f"{flag}="):
            return a.split("=", 1)[1]
    return ""


def _sglang_weight_placement(spec: RunSpec) -> WeightPlacement:
    """sglang shards LLM weights by tp; diffusion depends on the recipe.

    Diffusion defaults to Ulysses sequence parallelism, which splits the
    SEQUENCE while every rank keeps a full copy of the DiT. Two flags change
    that: --use-fsdp-inference shards the weights across ranks, and
    layerwise offload of `dit` moves them to host RAM entirely, streaming
    blocks in per step. The offloaded case is bounded by HOST memory, not
    VRAM — which is how a 62GiB model runs on 32GiB cards at all.
    """
    if spec.kind != KIND_DIFFUSION:
        return WeightPlacement(max(spec.tp, 1), f"tp={spec.tp}")
    args = tuple(a.lower() for a in (*spec.engine_args, *spec.extra_args))
    ranks = max(len(spec.gpus) or spec.tp, 1)
    offloaded = _flag_value(args, "--layerwise-offload-components").split(",")
    if "dit" in (c.strip() for c in offloaded):
        return WeightPlacement(
            1,
            "DiT streamed per layer from host RAM",
            resident=False,
            host_copies=ranks,
        )
    if "--use-fsdp-inference" in args and "false" not in args:
        return WeightPlacement(ranks, f"fsdp-sharded across {ranks} GPUs")
    return WeightPlacement(1, "replicated per rank (sequence parallel)")


def _llamacpp_cmd(spec: RunSpec, key: str) -> list[str]:
    # CPU mode: no GPU offload, thread count from the CPU reservation.
    if spec.mode == "cpu":
        threads = spec.cpus or 16
        return [
            config.venv_python("llamacpp"),
            "-m",
            "llama_cpp.server",
            "--model",
            spec.model_file or f"{spec.model_path}/MISSING.gguf",
            "--host",
            "0.0.0.0",
            "--port",
            str(spec.port),
            "--api_key",
            key,
            "--n_gpu_layers",
            "0",  # pure CPU
            "--n_threads",
            str(threads),
            "--n_threads_batch",
            str(threads),
            "--n_ctx",
            str(min(spec.max_len, 131072)),
        ]
    # GGUF: model_path is a directory; the profile's `variant` (or the only
    # .gguf present) selects which quantization to serve. Resolution of the
    # concrete file happens at deploy time in serve.start (remote glob).
    return [
        config.venv_python("llamacpp"),
        "-m",
        "llama_cpp.server",
        "--model",
        spec.model_file or f"{spec.model_path}/MISSING.gguf",
        "--host",
        "0.0.0.0",
        "--port",
        str(spec.port),
        "--api_key",
        key,
        "--n_gpu_layers",
        "-1",
        "--split_mode",
        "1",  # layer split across visible GPUs
        "--n_ctx",
        str(min(spec.max_len, 131072)),
        "--flash_attn",
        "true",
    ]


REGISTRY: dict[str, EngineDef] = {
    "vllm": EngineDef(
        name="vllm",
        formats=("safetensors",),
        has_arch_registry=True,
        features=(
            Feature.CONTINUATION
            | Feature.BAD_WORDS
            | Feature.LOGIT_BIAS
            | Feature.TOKENIZE
            | Feature.KV_RAM_TIER
            | Feature.SPEC_DECODE
            | Feature.EXPERT_PARALLEL
            | Feature.REASONING_PARSER
            | Feature.TOOL_PARSER
            | Feature.TENSOR_PARALLEL
        ),
        build_command=_vllm_cmd,
    ),
    "sglang": EngineDef(
        name="sglang",
        # gguf: launchable by our builder; actual support verified per-install
        # by the format probe (list_formats.py) — effective = declared ∩ probed
        formats=("safetensors", "gguf"),
        has_arch_registry=True,
        # sglang.multimodal_gen serves diffusion pipelines (Wan, FLUX,
        # MiniMax-H3, ...) alongside LLMs — verified by the kinds probe.
        kinds=(KIND_LLM, KIND_DIFFUSION),
        # continue_final_message + logit_bias supported; no bad_words param,
        # no OpenAI-style /tokenize (so logit-bias suppression is inert until
        # a tokenize adapter is written).
        features=(
            Feature.CONTINUATION
            | Feature.LOGIT_BIAS
            | Feature.KV_RAM_TIER
            | Feature.SPEC_DECODE
            | Feature.REASONING_PARSER
            | Feature.TOOL_PARSER
            | Feature.TENSOR_PARALLEL
            | Feature.MODEL_SUBFOLDER
        ),
        build_command=_sglang_cmd,
        weight_split=_sglang_weight_placement,
    ),
    "llamacpp": EngineDef(
        name="llamacpp",
        formats=("gguf",),
        has_arch_registry=False,
        features=Feature.LOGIT_BIAS,  # guard degrades to truncate-only
        experimental=True,
        # llama-cpp-python compiles CUDA kernels at install time; CMake needs
        # nvcc + the toolkit, which only exist after the pip CUDA wheels are
        # installed — hence the prepare pass. CUDAToolkit_ROOT/PATH point
        # CMake at the venv's own toolkit (never a system CUDA — see README
        # "hard-won toolchain facts").
        sync_prepare_group="cudabuild",
        # libllama.so links libcudart.so.13 from the venv's pip toolkit, which
        # is not on the system loader path.
        runtime_env={"LD_LIBRARY_PATH": "{cuda}/lib"},
        sync_env={
            "CMAKE_ARGS": (
                "-DGGML_CUDA=on "
                "-DCUDAToolkit_ROOT={cuda} "
                "-DCMAKE_CUDA_COMPILER={cuda}/bin/nvcc "
                "-DCMAKE_CUDA_ARCHITECTURES=120"
            ),
            "FORCE_CMAKE": "1",
            "CUDAToolkit_ROOT": "{cuda}",
            # {uvbin} keeps uv itself reachable — an env PATH REPLACES the
            # inherited one, so omitting it makes the sync command vanish.
            "PATH": "{cuda}/bin:{uvbin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        },
        build_command=_llamacpp_cmd,
    ),
}


def get(name: str) -> EngineDef:
    if name not in REGISTRY:
        raise SystemExit(f"unknown engine {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names(*, include_experimental: bool = True) -> tuple[str, ...]:
    return tuple(n for n, e in REGISTRY.items() if include_experimental or not e.experimental)
