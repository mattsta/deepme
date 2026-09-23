"""Model metadata fetched from the Hugging Face API — never hand-written.

Weight size and shard count are facts the HF API already knows; operators
only declare intent (which repo, how to serve it). Results are cached in
.deepme/hf/ so offline runs and repeat commands don't touch the network;
`refresh=True` (deepme model refresh) re-fetches.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import httpx

from . import config

CACHE_DIR = config.STATE_DIR / "hf"

# Bytes per parameter by safetensors dtype prefix (shared with catalog sizing).
_DTYPE_BYTES_PER_PARAM = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F4": 0.5,
    "I4": 0.5,
    "U4": 0.5,
}


def _dtype_key(dtype: str) -> str:
    """Longest matching dtype prefix (BF16 before F16, F8 before F4...)."""
    up = dtype.upper()
    for prefix in sorted(_DTYPE_BYTES_PER_PARAM, key=len, reverse=True):
        if up.startswith(prefix):
            return prefix
    return "BF16"


@dataclass(frozen=True)
class RepoMeta:
    repo: str
    weight_bytes: int  # total size of *.safetensors
    weight_shards: int  # count of *.safetensors files
    total_bytes: int  # everything in the repo
    # deep facts: per-variant GGUF sizes (quantizations are first-class
    # implementations, each independently deployable) + config.json contents
    gguf_variants: dict | None = None  # variant filename -> bytes (parts summed)
    architectures: tuple = ()
    model_type: str = ""
    total_params: int = 0  # from safetensors or gguf metadata blocks
    pipeline_tag: str = ""
    config_dims: dict | None = None  # architecture dims for perf estimation
    library: str = ""  # HF library_name (transformers, diffusers, ...)
    is_pipeline: bool = False  # diffusers-style multi-component pipeline
    components: tuple = ()  # component subfolders holding weights
    param_bytes: int = 0  # ONE instance's weights, from per-dtype counts

    @property
    def weight_gib(self) -> float:
        return self.weight_bytes / 1024**3

    @property
    def formats(self) -> tuple[str, ...]:
        out = []
        if self.weight_shards:
            out.append("safetensors")
        if self.gguf_variants:
            out.append("gguf")
        return tuple(out)

    @property
    def deployable_gib(self) -> float:
        """Size of ONE deployment, the minimum of two upper bounds.

        Per-dtype param math overestimates sub-byte packing (FP4 experts);
        summing repo file bytes overestimates repos holding several
        components or duplicate copies (diffusers pipelines). A deployment
        can be no larger than either.
        """
        bounds = [b for b in (self.weight_bytes, self.param_bytes) if b > 0]
        return (min(bounds) / 1024**3) if bounds else 0.0

    @property
    def kind(self) -> str:
        """Model class: 'diffusion' for image/video pipelines, else 'llm'.

        Decided from what the repo publishes: a diffusers pipeline index
        (model_index.json), library_name, or a generative-media pipeline tag.
        """
        if self.is_pipeline or self.library == "diffusers":
            return "diffusion"
        media = (
            "text-to-video",
            "image-to-video",
            "image-text-to-video",
            "text-to-image",
            "image-to-image",
        )
        if self.pipeline_tag in media:
            return "diffusion"
        return "llm"

    def smallest_gguf_gib(self) -> float | None:
        if not self.gguf_variants:
            return None
        return min(self.gguf_variants.values()) / 1024**3


_GGUF_PART = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)")


def _gguf_variants(siblings: list[dict]) -> dict[str, int]:
    """Group *.gguf files into logical variants (multi-part files summed).

    Repos commonly publish MANY quantization variants (Q4_K_M, Q8_0, ...),
    each a complete model — sizes must be per-variant, never summed across
    variants.
    """
    variants: dict[str, int] = {}
    for s in siblings:
        name = s.get("rfilename", "")
        if not name.endswith(".gguf"):
            continue
        low = name.lower()
        # Auxiliary gguf files are NOT standalone model variants: mmproj
        # (vision projectors) and imatrix (calibration data) must never be
        # size-checked as if they were deployable models.
        if "mmproj" in low or "imatrix" in low:
            continue
        variant = _GGUF_PART.sub("", name)
        variants[variant] = variants.get(variant, 0) + (s.get("size") or 0)
    return variants


def get(repo: str, *, refresh: bool = False) -> RepoMeta:
    """Full repo facts: file inventory (safetensors + GGUF variants, exact
    bytes) plus config.json architectures. One network round per repo, ever
    — cached indefinitely (releases are immutable); refresh re-fetches."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / (repo.replace("/", "--") + ".v5.json")
    if cache_file.exists() and not refresh:
        d = json.loads(cache_file.read_text())
        d["architectures"] = tuple(d.get("architectures") or ())
        d["components"] = tuple(d.get("components") or ())
        return RepoMeta(**d)
    r = httpx.get(
        f"https://huggingface.co/api/models/{repo}",
        params={"blobs": "true"},
        follow_redirects=True,
        timeout=30,
    )
    if r.status_code in (401, 403):
        raise SystemExit(
            f"{repo}: gated/private on Hugging Face — accept the license and "
            "set HF_TOKEN on the server before downloading"
        )
    if r.status_code == 404:
        raise SystemExit(f"{repo}: not found on Hugging Face — check the repo name")
    r.raise_for_status()
    info = r.json()
    siblings = info.get("siblings", [])
    weights = [s for s in siblings if s.get("rfilename", "").endswith(".safetensors")]
    # Parameter counts from whichever metadata block the repo provides
    st_block = info.get("safetensors") or {}
    gguf_block = info.get("gguf") or {}
    total_params = int(st_block.get("total") or gguf_block.get("total") or 0)
    # Bytes for ONE instance of the weights, from per-dtype parameter counts.
    # (File-byte sums cover the whole repo, which may hold several components
    # or duplicate copies — see `is_pipeline`.)
    param_counts = st_block.get("parameters") or {}
    param_bytes = int(
        sum(n * _DTYPE_BYTES_PER_PARAM.get(_dtype_key(dt), 2.0) for dt, n in param_counts.items())
    )
    library = info.get("library_name") or ""
    # diffusers-style pipelines declare model_index.json and split weights
    # across component subfolders (transformer/, vae/, text_encoder/, ...).
    names = [s.get("rfilename", "") for s in siblings]
    is_pipeline = any(n in ("model_index.json", "modular_model_index.json") for n in names)
    components = tuple(
        sorted(
            {n.split("/", 1)[0] for n in names if "/" in n and n.endswith((".safetensors", ".bin"))}
        )
    )
    archs: tuple[str, ...] = ()
    model_type = ""
    config_dims: dict | None = None
    if any(s.get("rfilename") == "config.json" for s in siblings):
        rc = httpx.get(
            f"https://huggingface.co/{repo}/raw/main/config.json",
            follow_redirects=True,
            timeout=30,
        )
        if rc.status_code == 200:
            try:
                cfg = rc.json()
                archs = tuple(cfg.get("architectures") or ())
                model_type = cfg.get("model_type") or ""
                # numeric architecture dims only (for perf estimation)
                config_dims = {
                    k: v
                    for k, v in cfg.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
            except ValueError:
                pass
    meta = RepoMeta(
        repo=repo,
        weight_bytes=sum(s.get("size") or 0 for s in weights),
        weight_shards=len(weights),
        total_bytes=sum(s.get("size") or 0 for s in siblings),
        gguf_variants=_gguf_variants(siblings) or None,
        architectures=archs,
        model_type=model_type,
        total_params=total_params,
        pipeline_tag=info.get("pipeline_tag") or "",
        config_dims=config_dims,
        library=library,
        is_pipeline=is_pipeline,
        components=components,
        param_bytes=param_bytes,
    )
    d = dict(meta.__dict__)
    d["architectures"] = list(meta.architectures)
    d["components"] = list(meta.components)
    cache_file.write_text(json.dumps(d, indent=2))
    return meta
