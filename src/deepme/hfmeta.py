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
    cache_file = CACHE_DIR / (repo.replace("/", "--") + ".v4.json")
    if cache_file.exists() and not refresh:
        d = json.loads(cache_file.read_text())
        d["architectures"] = tuple(d.get("architectures") or ())
        return RepoMeta(**d)
    r = httpx.get(
        f"https://huggingface.co/api/models/{repo}",
        params={"blobs": "true"},
        follow_redirects=True,
        timeout=30,
    )
    r.raise_for_status()
    info = r.json()
    siblings = info.get("siblings", [])
    weights = [s for s in siblings if s.get("rfilename", "").endswith(".safetensors")]
    # Parameter counts from whichever metadata block the repo provides
    st_block = info.get("safetensors") or {}
    gguf_block = info.get("gguf") or {}
    total_params = int(st_block.get("total") or gguf_block.get("total") or 0)
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
    )
    d = dict(meta.__dict__)
    d["architectures"] = list(meta.architectures)
    cache_file.write_text(json.dumps(d, indent=2))
    return meta
