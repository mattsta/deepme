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
FORMATS_CACHE = config.STATE_DIR / "engine-formats.json"
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


def capacity_for_gpus(n: int) -> float:
    """Deployable weight budget across `n` GPUs."""
    cluster = config._file_config().get("cluster", {})
    vram = float(cluster.get("vram_gb", 32))
    # 0.90 usable fraction, ~3GB/GPU runtime overhead (activations/graphs/KV floor)
    return max(n, 0) * (vram * 0.90 - 3.0)


def cluster_capacity_gib() -> float:
    """Budget on an EMPTY machine — what the hardware could ever run."""
    cluster = config._file_config().get("cluster", {})
    return capacity_for_gpus(int(cluster.get("gpus", 8)))


def free_capacity_gib() -> float:
    """Budget in the GPUs that are free RIGHT NOW, given running units.

    The difference matters operationally: `cluster_capacity_gib` answers
    "could this box ever run it", this answers "can I start it now".
    """
    from . import resources

    cap = resources.probe_capacity()
    avail = resources.availability(cap, resources.current_claims())
    return capacity_for_gpus(len(avail.free_gpus))


def engine_archs(*, refresh: bool = False) -> dict[str, list[str]]:
    """Supported architectures per INSTALLED engine (probe evidence), cached.

    A key present with a non-empty list means "probed, supports these". A key
    ABSENT means "unknown" (engine not installed, or server was unreachable) —
    callers must not treat unknown as unsupported. Failed/unreachable probes
    are never cached (so a transient outage can't poison the catalog)."""
    if ARCH_CACHE.exists() and not refresh:
        return json.loads(ARCH_CACHE.read_text())
    probed = _probe_engines("list_archs.py")
    if probed is None:
        return {}  # server unreachable: everything unknown, don't cache
    config.STATE_DIR.mkdir(exist_ok=True)
    ARCH_CACHE.write_text(json.dumps(probed))
    return probed


def _probe_engines(script: str) -> dict[str, list[str]] | None:
    """Run a per-engine probe on the server. Returns None if the server is
    unreachable (caller degrades); otherwise a dict with an entry ONLY for
    engines whose venv exists and probed successfully (installed engines)."""
    from . import doctor, remote
    from . import engines as engines_mod

    if not remote.run("echo ok", check=False, timeout=10).ok:
        return None
    path = doctor.push_script(script)
    result: dict[str, list[str]] = {}
    for eng in engines_mod.names():
        installed = remote.run(
            f"test -x {config.venv_python(eng)} && echo Y || echo N", check=False
        )
        if "Y" not in installed.stdout:
            continue  # not installed -> unknown, omit
        r = remote.run(
            f"{config.venv_python(eng)} {path} {eng} 2>/dev/null",
            check=False,
            timeout=300,
        )
        try:
            data = json.loads(r.stdout.strip().splitlines()[-1])
            result[eng] = data if isinstance(data, list) else []
        except Exception:
            result[eng] = []
    return result


def engine_formats(*, refresh: bool = False) -> dict[str, list[str]]:
    """PROBED weight-format support per INSTALLED engine (never assumed).
    Same absence semantics as engine_archs; unreachable probes not cached."""
    if FORMATS_CACHE.exists() and not refresh:
        return json.loads(FORMATS_CACHE.read_text())
    probed = _probe_engines("list_formats.py")
    if probed is None:
        return {}
    config.STATE_DIR.mkdir(exist_ok=True)
    FORMATS_CACHE.write_text(json.dumps(probed))
    return probed


def effective_formats(engine: str, probed: dict[str, list[str]]) -> set[str]:
    """Formats we can BOTH launch (declared) AND that the INSTALLED engine
    proves it loads. Engine absent from `probed` = not installed / unknown ->
    no effective formats (not a deployment candidate). When the whole probe
    was unavailable (server down), `probed` is empty and callers degrade to
    declared via `probed_or_declared`."""
    from . import engines as engines_mod

    declared = set(engines_mod.get(engine).formats)
    if engine not in probed:
        return set()
    return declared & set(probed[engine])


def probed_or_declared(probed: dict) -> dict:
    """When the probe was unavailable (server down -> empty dict), degrade to
    each engine's DECLARED formats so the catalog still works offline rather
    than hiding everything. When probe data exists, use it as-is."""
    from . import engines as engines_mod

    if probed:
        return probed
    return {e.name: list(e.formats) for e in engines_mod.REGISTRY.values()}


def candidate_engines(
    formats: set,
    archs: tuple,
    probed_formats: dict,
    archs_by_engine: dict,
    kind: str = "llm",
) -> list[str]:
    """Proof-chain engine selection: model KIND x weight FORMAT x ARCH.

    An engine qualifies when it serves this kind of model, can load one of
    its formats (probed on the installed build), and — for LLMs on engines
    with an introspectable registry — supports the architecture. Unknown
    never rejects; only positive evidence of absence does. Production
    engines are preferred over experimental ones.
    """
    from . import engines as engines_mod

    pf = probed_or_declared(probed_formats)
    out = []
    for e in engines_mod.REGISTRY.values():
        if kind not in e.kinds:
            continue
        if not (set(formats) & effective_formats(e.name, pf)):
            continue
        # Arch registries list LLM architectures; diffusion pipelines are
        # registered separately, so the arch gate applies to LLMs only.
        known = archs_by_engine.get(e.name)
        if (
            kind == engines_mod.KIND_LLM
            and e.has_arch_registry
            and known
            and not any(a in known for a in archs)
        ):
            continue
        out.append(e.name)
    out.sort(key=lambda n: engines_mod.get(n).experimental)  # production first
    return out


SORTS = {
    "trending": "trendingScore",
    "downloads": "downloads",
    "likes": "likes",
    "modified": "lastModified",
}


def fetch_top(
    limit: int = 100,
    tag: str | None = None,
    *,
    refresh: bool = False,
    sort: str = "trending",
    search: str | None = None,
    author: str | None = None,
) -> list[dict]:
    """Model listing (any modality) with size + config metadata.

    No task prefilter: anything representable as weights is a candidate —
    the installed engines' registries are the only real serveability gate.
    `search`/`author`/`tag` narrow server-side when the operator asks (the
    HF API does the filtering, so a search reaches models far outside the
    top-N window).

    Responses are TTL-cached locally (CATALOG_TTL_S) keyed by every query
    parameter, so repeated browsing doesn't hammer the rate-limited API.
    """
    if sort not in SORTS:
        raise SystemExit(f"unknown sort {sort!r}; choose from {sorted(SORTS)}")
    CATALOG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = f"{sort}-{tag or 'all'}-{author or 'any'}-{(search or 'none').replace('/', '_')}-{limit}"
    cache_file = CATALOG_CACHE_DIR / f"q-{key}.json"
    if not refresh and cache_file.exists():
        age = time.time() - cache_file.stat().st_mtime
        if age < CATALOG_TTL_S:
            return json.loads(cache_file.read_text())
    r = httpx.get(
        "https://huggingface.co/api/models",
        params=[
            *([("pipeline_tag", tag)] if tag else []),
            *([("search", search)] if search else []),
            *([("author", author)] if author else []),
            ("sort", SORTS[sort]),
            ("direction", "-1"),
            ("limit", str(limit)),
            ("expand[]", "safetensors"),
            ("expand[]", "config"),
            ("expand[]", "downloads"),
            ("expand[]", "likes"),
            ("expand[]", "gated"),
            ("expand[]", "pipeline_tag"),
        ],
        follow_redirects=True,
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    cache_file.write_text(json.dumps(data))
    return data


def prefilter(
    m: dict, *, name_contains: str | None = None, min_downloads: int = 0, min_likes: int = 0
) -> bool:
    """Cheap match on LIST-API data only — no network.

    Applied BEFORE the per-repo deep fetch so scanning a 1000-model catalog
    costs one request, not a thousand. `name_contains` matches the repo id
    (org and model name), case-insensitively.
    """
    if name_contains and name_contains.lower() not in m.get("id", "").lower():
        return False
    if min_downloads and int(m.get("downloads") or 0) < min_downloads:
        return False
    return not (min_likes and int(m.get("likes") or 0) < min_likes)


def build(
    limit: int = 100,
    *,
    tag: str | None = None,
    refresh: bool = False,
    refresh_archs: bool = False,
    sort: str = "trending",
    search: str | None = None,
    author: str | None = None,
    name_contains: str | None = None,
    min_downloads: int = 0,
    min_likes: int = 0,
    max_entries: int = 0,
    capacity: float | None = None,
) -> list[CatalogEntry]:
    from . import hfmeta

    archs_by_engine = engine_archs(refresh=refresh_archs)
    probed_formats = engine_formats(refresh=refresh_archs)
    # Default budget is the whole machine; callers pass the FREE budget to ask
    # "what could I start right now" instead of "what could this box ever run".
    if capacity is None:
        capacity = cluster_capacity_gib()
    entries: list[CatalogEntry] = []
    listing = fetch_top(limit, tag, refresh=refresh, sort=sort, search=search, author=author)
    # Cheap filters first: only survivors pay for a deep repo fetch.
    survivors = [
        (rank, m)
        for rank, m in enumerate(listing, start=1)
        if prefilter(
            m, name_contains=name_contains, min_downloads=min_downloads, min_likes=min_likes
        )
    ]
    if max_entries:
        survivors = survivors[:max_entries]
    for rank, m in survivors:
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
            deep = hfmeta.get(m["id"], refresh=refresh)
        except Exception:
            # (gated/missing/unreachable) must not abort the whole catalog
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
        param_bytes = sum(count * _dtype_bytes(dt) for dt, count in param_counts.items())
        if not param_bytes and deep is not None:
            param_bytes = deep.param_bytes
        # Per-format sizes; a repo is deployable if the MIN representation
        # fits (a huge BF16 with a small allowed gguf variant is deployable
        # via the gguf path). weight_gib/label track the representation we'd
        # actually deploy: smallest that fits, else smallest overall.
        st_gib = 0.0
        if "safetensors" in formats:
            # Two independent estimates, each with its own overestimation mode:
            #   param math  — overestimates for sub-byte packing (FP4 experts)
            #   file bytes  — overestimates for repos holding several
            #                 components/copies (diffusers pipelines, refs)
            # One deployment can be no larger than either, so take the min.
            candidates = [g for g in (deep_st_gib, param_bytes / 1024**3) if g > 0]
            st_gib = min(candidates) if candidates else 0.0
        sizes: dict[str, float] = {}
        if st_gib:
            sizes["safetensors"] = st_gib
        if gguf_gib:
            sizes["gguf"] = gguf_gib
        fitting = {f: s for f, s in sizes.items() if s <= capacity}
        if fitting:
            weight_fmt, weight_gib = min(fitting.items(), key=lambda kv: kv[1])
            size_fits = True
        elif sizes:
            weight_fmt, weight_gib = min(sizes.items(), key=lambda kv: kv[1])
            size_fits = False
        else:
            weight_fmt, weight_gib, size_fits = None, 0.0, True
        # Engine support: proof chain — probed format evidence from the
        # installed binaries x arch registries (see candidate_engines).
        kind = deep.kind if deep is not None else "llm"
        engines = tuple(
            candidate_engines(set(formats), model_archs, probed_formats, archs_by_engine, kind=kind)
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
            rejections.append("gated: requires HF auth/license acceptance before download")
        if formats and not size_fits:
            label = "smallest gguf variant" if weight_fmt == "gguf" else weight_fmt
            rejections.append(
                f"too-big: {weight_gib:.0f}GiB ({label}) > {capacity:.0f}GiB cluster budget"
            )
        if formats and not engines:
            if kind == "diffusion":
                # A diffusion pipeline with no capable engine installed —
                # name the kind so the fix (install/enable one) is obvious.
                comp = ", ".join(deep.components[:5]) if deep else "components"
                rejections.append(
                    f"no-engine-for-kind: diffusion pipeline ({comp}) — no "
                    "installed engine serves diffusion models"
                )
            else:
                what = "/".join(model_archs) or "no architectures declared"
                rejections.append(
                    f"unsupported: formats={formats} archs={what} — no registered "
                    "engine covers this format+arch combination"
                )
        fits = not any(
            r.startswith(("too-big", "gated", "no-weights", "quant-policy")) for r in rejections
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
            est = perfmodel.estimate_decode_tps(config_dims, total_params, weight_gib * 1024**3)
            if est:
                est_tps, tp_needed = est
        elif weight_gib:
            import math

            tp_needed = max(1, math.ceil(weight_gib / perfmodel.per_gpu_budget_gib()))
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
