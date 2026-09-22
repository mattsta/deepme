"""Model catalog: HF top models, annotated with engine support and fit.

Discovery is API-driven end to end: the Hugging Face list API supplies
per-dtype parameter counts (exact weight bytes — no hand-entered sizes) and
architectures; the *installed* engines' registries (fetched once from the
server, cached) supply what we can actually serve; the cluster shape from
config supplies what fits. The picker only has to render the intersection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

from . import config

ARCH_CACHE = config.STATE_DIR / "engine-archs.json"
CATALOG_CACHE_DIR = config.STATE_DIR / "hf" / "catalog"
CATALOG_TTL_S = 3600  # HF trending moves slowly and the API rate-limits;
# reuse list responses for an hour (--refresh bypasses)

# bytes per parameter by safetensors dtype prefix
_DTYPE_BYTES = {
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


def _dtype_bytes(dtype: str) -> float:
    for prefix, size in _DTYPE_BYTES.items():
        if dtype.upper().startswith(prefix):
            return size
    return 2.0  # conservative default


@dataclass(frozen=True)
class CatalogEntry:
    repo: str
    trending_rank: int
    task: str  # HF pipeline tag (modality hint, not a filter)
    downloads: int
    likes: int
    params_b: float  # total parameters, billions
    weight_gib: float  # exact, from per-dtype parameter counts
    dtypes: tuple[str, ...]
    architectures: tuple[str, ...]
    engines: tuple[str, ...]  # installed engines that support the arch
    fits: bool
    unfit_reason: str
    est_tps: float = 0.0  # heuristic single-stream decode tok/s (see perfmodel)
    tp_needed: int = 0  # GPUs needed for the weights at ~28GiB each
    # Machine-auditable rejection reasons; empty tuple == deployable.
    # Kinds: no-metadata | gated | too-big | unsupported-arch
    rejections: tuple[str, ...] = ()


def cluster_capacity_gib() -> float:
    """Deployable weight budget: gpus x vram x usable fraction - overhead."""
    cluster = config._file_config().get("cluster", {})
    n = int(cluster.get("gpus", 8))
    vram = float(cluster.get("vram_gb", 32))
    # 0.90 usable fraction, ~3GB/GPU runtime overhead (activations/graphs/KV floor)
    return n * (vram * 0.90 - 3.0)


def engine_archs(*, refresh: bool = False) -> dict[str, list[str]]:
    """Supported architectures per installed engine, cached locally."""
    if ARCH_CACHE.exists() and not refresh:
        return json.loads(ARCH_CACHE.read_text())
    from . import doctor, remote

    result: dict[str, list[str]] = {}
    path = doctor.push_script("list_archs.py")
    for eng in config.ENGINES:
        r = remote.run(
            f"{config.venv_python(eng)} {path} {eng}", check=False, timeout=300
        )
        try:
            result[eng] = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001 - engine may be broken/missing
            result[eng] = []
    config.STATE_DIR.mkdir(exist_ok=True)
    ARCH_CACHE.write_text(json.dumps(result))
    return result


def fetch_top(
    limit: int = 100, tag: str | None = None, *, refresh: bool = False
) -> list[dict]:
    """Top trending models (any modality) with size + config metadata.

    No task prefilter: anything representable as weights is a candidate —
    the installed engines' registries are the only real serveability gate.
    An optional pipeline tag narrows by modality when the operator asks.

    Responses are TTL-cached locally (CATALOG_TTL_S) keyed by (limit, tag)
    so repeated browsing doesn't hammer the rate-limited HF API.
    """
    CATALOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CATALOG_CACHE_DIR / f"top-{tag or 'all'}-{limit}.json"
    if not refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CATALOG_TTL_S:
            return json.loads(cache_file.read_text())
    r = httpx.get(
        "https://huggingface.co/api/models",
        params=[
            *([("pipeline_tag", tag)] if tag else []),
            ("sort", "trendingScore"),
            ("limit", str(limit)),
            ("expand[]", "safetensors"),
            ("expand[]", "config"),
            ("expand[]", "downloads"),
            ("expand[]", "likes"),
            ("expand[]", "gated"),
            ("expand[]", "pipeline_tag"),
        ],
        follow_redirects=True,
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    cache_file.write_text(json.dumps(data))
    return data


def build(
    limit: int = 100,
    *,
    tag: str | None = None,
    refresh: bool = False,
    refresh_archs: bool = False,
) -> list[CatalogEntry]:
    from . import engines as engines_mod
    from . import hfmeta

    archs_by_engine = engine_archs(refresh=refresh_archs)
    capacity = cluster_capacity_gib()
    entries: list[CatalogEntry] = []
    for rank, m in enumerate(fetch_top(limit, tag, refresh=refresh), start=1):
        st = m.get("safetensors") or {}
        param_counts: dict[str, int] = st.get("parameters") or {}
        total_params = st.get("total") or sum(param_counts.values())
        model_archs = tuple((m.get("config") or {}).get("architectures") or ())
        gguf_gib: float | None = None
        gguf_count = 0
        formats: list[str] = ["safetensors"] if total_params else []
        # The list API is shallow; when it can't size or classify a repo,
        # fetch the repo's full inventory ourselves (one round-trip ever —
        # cached forever). GGUF quantizations are first-class deployable
        # implementations, sized per-variant.
        task = m.get("pipeline_tag") or ""
        deep_st_gib = 0.0
        gguf_policy_note = ""
        config_dims: dict = {}
        # Deep repo facts (one fetch per repo ever, cached): file inventory,
        # config dims for perf estimation, params, formats.
        try:
            deep = hfmeta.get(m["id"])
        except Exception:  # noqa: BLE001 - fetch failure -> stated below
            deep = None
        if deep:
            model_archs = model_archs or deep.architectures
            formats = list(deep.formats)
            total_params = total_params or deep.total_params
            task = task or deep.pipeline_tag
            deep_st_gib = deep.weight_gib
            config_dims = deep.config_dims or {}
        if deep and deep.gguf_variants:
            # Quant policy gate: only policy-allowed, plausibly-full-model
            # variants count as deployable implementations (see policy.py).
            from . import policy

            full = policy.full_model_variants(deep.gguf_variants, deep.total_params)
            allowed = policy.current().filter_variants(full)
            gguf_count = len(allowed)
            denied_count = len(deep.gguf_variants) - gguf_count
            if allowed:
                gguf_gib = min(allowed.values()) / 1024**3
            else:
                formats = [f for f in formats if f != "gguf"]
                if denied_count:
                    gguf_policy_note = (
                        f"quant-policy: all {denied_count} gguf variants "
                        f"below {policy.current().min_gguf_bits:g}-bit "
                        "minimum or unclassifiable"
                    )
        # Weight size, most-authoritative first: ACTUAL file bytes ->
        # per-dtype param arithmetic -> smallest deployable gguf variant.
        # (File bytes outrank dtype math: sub-byte packings like FP4 experts
        # make param-count arithmetic overestimate, e.g. DeepSeek-V4.)
        param_bytes = sum(
            count * _dtype_bytes(dt) for dt, count in param_counts.items()
        )
        if "safetensors" in formats and deep_st_gib:
            weight_gib = deep_st_gib
        elif param_bytes:
            weight_gib = param_bytes / 1024**3
        else:
            weight_gib = gguf_gib or 0.0
        # Engine support: declared cross-product of engine formats x
        # (arch registry when the engine has one).
        engines = tuple(
            e.name
            for e in engines_mod.REGISTRY.values()
            if set(e.formats) & set(formats)
            and (
                not e.has_arch_registry
                or any(a in archs_by_engine.get(e.name, []) for a in model_archs)
            )
        )
        # Every exclusion is recorded with its specific, auditable reason —
        # nothing is silently dropped.
        rejections: list[str] = []
        if gguf_policy_note:
            rejections.append(gguf_policy_note)
        if not formats:
            rejections.append(
                "no-weights: repo inventory has neither safetensors nor gguf "
                "files (adapter/dataset/unfetchable)"
                if not gguf_policy_note
                else "no-weights: no policy-allowed weight representation remains"
            )
        if m.get("gated"):
            rejections.append(
                "gated: requires HF auth/license acceptance before download"
            )
        if formats and weight_gib > capacity:
            rejections.append(
                f"too-big: {weight_gib:.0f}GiB ({'smallest gguf variant' if gguf_gib else 'safetensors'}) "
                f"> {capacity:.0f}GiB cluster budget"
            )
        if formats and not engines:
            what = "/".join(model_archs) or "no architectures declared"
            rejections.append(
                f"unsupported: formats={formats} archs={what} — no registered "
                "engine covers this format+arch combination"
            )
        fits = not any(
            r.startswith(("too-big", "gated", "no-weights", "quant-policy"))
            for r in rejections
        )
        reason = "; ".join(rejections)
        if gguf_count:
            formats = [f"gguf x{gguf_count}"] + [f for f in formats if f != "gguf"]
        # Heuristic decode throughput on this cluster (see perfmodel.py) —
        # only for autoregressive generators; embeddings/classifiers emit
        # vectors, not token streams.
        from . import perfmodel

        est_tps, tp_needed = 0.0, 0
        if perfmodel.is_generative(task, model_archs):
            est = perfmodel.estimate_decode_tps(
                config_dims, total_params, weight_gib * 1024**3
            )
            if est:
                est_tps, tp_needed = est
        elif weight_gib:
            import math

            tp_needed = max(1, math.ceil(weight_gib / perfmodel.PER_GPU_BUDGET_GIB))
        entries.append(
            CatalogEntry(
                repo=m["id"],
                trending_rank=rank,
                task=task or "?",
                downloads=int(m.get("downloads") or 0),
                likes=int(m.get("likes") or 0),
                params_b=total_params / 1e9,
                weight_gib=weight_gib,
                dtypes=tuple(sorted(param_counts)) or tuple(formats),
                architectures=model_archs,
                engines=engines,
                fits=fits,
                unfit_reason=reason,
                rejections=tuple(rejections),
                est_tps=round(est_tps, 1),
                tp_needed=tp_needed,
            )
        )
    return entries


def rejection_summary(entries: list[CatalogEntry]) -> dict[str, int]:
    """Count of excluded models per rejection kind, for the audit line."""
    counts: dict[str, int] = {}
    for e in entries:
        for r in e.rejections:
            kind = r.split(":", 1)[0]
            counts[kind] = counts.get(kind, 0) + 1
    return counts
