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

    def names(self) -> tuple[str, ...]:
        return tuple(
            f.name.lower() for f in Feature if f is not Feature.NONE and f in self
        )


@dataclass(frozen=True)
class EngineDef:
    name: str
    formats: tuple[str, ...]  # weight formats it can serve
    has_arch_registry: bool  # True: arch support is introspectable
    features: Feature = Feature.NONE  # declared capability matrix
    experimental: bool = False  # not yet benchmarked/production-proven
    sync_env: dict[str, str] = field(default_factory=dict)  # env for `uv sync`
    build_command: Callable[[RunSpec, str], list[str]] = None  # (spec, api_key)

    def supports(self, feature: Feature) -> bool:
        return feature in self.features

    @property
    def venv(self) -> str:
        return config.engine_venv(self.name)

    @property
    def python(self) -> str:
        return f"{self.venv}/bin/python"

    def validate_spec(self, spec: RunSpec) -> str | None:
        """Attribute-level validation before any launch attempt."""
        if spec.format not in self.formats:
            return (
                f"engine {self.name} serves {self.formats}, "
                f"but model format is {spec.format!r}"
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
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use": spec.kv_ram_gb * 1024**3
                    },
                }
            ),
        ]
    return cmd


def _sglang_cmd(spec: RunSpec, key: str) -> list[str]:
    cmd = [
        config.venv_python("sglang"),
        "-m",
        "sglang.launch_server",
        "--model-path",
        spec.model_path,
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


def _llamacpp_cmd(spec: RunSpec, key: str) -> list[str]:
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
        ),
        build_command=_vllm_cmd,
    ),
    "sglang": EngineDef(
        name="sglang",
        formats=("safetensors",),
        has_arch_registry=True,
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
        ),
        build_command=_sglang_cmd,
    ),
    "llamacpp": EngineDef(
        name="llamacpp",
        formats=("gguf",),
        has_arch_registry=False,
        features=Feature.LOGIT_BIAS,  # guard degrades to replacement-only
        experimental=True,
        sync_env={"CMAKE_ARGS": "-DGGML_CUDA=on", "FORCE_CMAKE": "1"},
        build_command=_llamacpp_cmd,
    ),
}


def get(name: str) -> EngineDef:
    if name not in REGISTRY:
        raise SystemExit(f"unknown engine {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name]


def names(*, include_experimental: bool = True) -> tuple[str, ...]:
    return tuple(
        n for n, e in REGISTRY.items() if include_experimental or not e.experimental
    )
