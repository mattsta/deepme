"""Runtime hook: back sglang's offloaded weight buffers with shared memory.

Installed into the engine's site-packages with a .pth file that imports it at
interpreter start, so it applies without editing engine source. That matters
for maintenance: a source diff has to be re-based every time upstream
reformats the file, whereas this only breaks if the internals it names
actually change — and it checks for that and no-ops instead of exploding.

WHAT IT DOES
LayerwiseOffloadManager._initialize packs each (layer, dtype) group of weights
into one flat CPU buffer allocated with `torch.empty(n, dtype=..,
pin_memory=True)`. Pinned memory is page-locked into ONE process, so N
replicas of a model hold N identical private copies (measured: 96GiB per
rank, which caps this box at 2 replicas long before the GPUs do).

While that method runs, this hook substitutes an allocator that returns a
tensor backed by a file in /dev/shm, keyed by content. The first process to
need a buffer creates it; every later process maps the same physical pages.
`cudaHostRegister` then page-locks each process's own mapping, so transfers
stay pinned-fast — that is what makes "shared" and "pinned" compatible.

The copy that fills the buffer always runs, even when another process already
filled it. Rewriting identical bytes is a no-op semantically and costs only
load time, and it means the hook cannot corrupt weights by mis-detecting
readiness.

Off unless DEEPME_SHARED_WEIGHTS is set. Every failure path falls back to the
upstream allocation.
"""

from __future__ import annotations

import contextlib
import logging
import os

logger = logging.getLogger("deepme.hooks")

ENABLED_ENV = "DEEPME_SHARED_WEIGHTS"
STORE_DIR = os.environ.get("DEEPME_SHARED_STORE_DIR", "/dev/shm/deepme-weights")
REGISTER_ENV = "DEEPME_SHARED_WEIGHTS_PIN"  # set to 0 to skip cudaHostRegister


def _enabled() -> bool:
    return os.environ.get(ENABLED_ENV, "").lower() in ("1", "true", "yes")


def _key(model_id: str, ordinal: int, dtype: str, numel: int) -> str:
    """Address a buffer by what is in it.

    Rank is deliberately NOT part of this. With a content fingerprint it is
    redundant where it would be safe and harmful where it would not: a
    tensor-parallel component gives each rank DIFFERENT weights, so their
    fingerprints already differ, while a replicated component (H3's text
    encoder) gives every rank the SAME weights — and including rank there
    would force a second copy of 62GiB for no reason.
    """
    import hashlib

    raw = f"{model_id}|{ordinal}|{dtype}|{numel}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


# How many parameters to sample when fingerprinting a component. Enough to
# separate two differently-trained models with certainty; small enough that
# fingerprinting is free next to loading tens of GiB.
_FINGERPRINT_PARAMS = 48


def component_fingerprint(model) -> str:
    """Identify a component by its WEIGHTS, not by where it was loaded from.

    Path-based identity misses that partitions of one repo often ship the same
    component twice: MiniMax-H3's FL2VA and Ref2VA hold a byte-identical
    62.1GiB text encoder and audio VAE, and differ only in the transformer.
    Keyed by path they get two physical copies; keyed by content they share
    one, which is what makes a mixed fleet affordable.

    Structure alone is not enough — the two transformers have identical shapes
    and differ only in values — so a few sampled elements are folded in. That
    is O(1) per tensor and decisive: differently-trained weights cannot agree
    on sampled values.
    """
    import hashlib

    digest = hashlib.sha256()
    try:
        params = sorted(model.named_parameters(), key=lambda kv: kv[0])
    except Exception:
        return "unknown"
    step = max(len(params) // _FINGERPRINT_PARAMS, 1)
    for name, tensor in params[::step][:_FINGERPRINT_PARAMS]:
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        try:
            flat = tensor.detach().flatten()
            if flat.numel():
                picks = (0, flat.numel() // 2, flat.numel() - 1)
                for i in picks:
                    digest.update(f"{float(flat[i].float()):.6e}".encode())
        except Exception:
            # A meta/sharded tensor cannot be sampled; structure still counts.
            digest.update(b"?")
    return digest.hexdigest()[:32]


def _shared_tensor(torch, path: str, nbytes: int, numel: int, dtype):
    """A tensor whose storage is a shared mapping of `path`."""
    storage = torch.UntypedStorage.from_file(path, shared=True, nbytes=nbytes)
    tensor = torch.empty(0, dtype=dtype)
    tensor.set_(storage, 0, (numel,))
    return tensor


def _register_pinned(torch, tensor, nbytes: int) -> bool:
    """Page-lock this process's mapping so H2D copies stay fast.

    cudaHostRegister locks an EXISTING mapping rather than allocating, which
    is the whole trick: the physical pages stay shared while each process
    gets pinned semantics over them.
    """
    if os.environ.get(REGISTER_ENV, "1").lower() in ("0", "false", "no"):
        return False
    try:
        err = torch.cuda.cudart().cudaHostRegister(tensor.data_ptr(), nbytes, 0)
        # cudaErrorHostMemoryAlreadyRegistered is success for our purposes.
        return int(err) in (0, 712)
    except Exception as exc:
        logger.warning("deepme: cudaHostRegister failed (%s); unpinned shared", exc)
        return False


@contextlib.contextmanager
def _shared_allocator(model_id: str):
    """Swap torch.empty for a shared-memory allocator, for pinned flat buffers.

    Scoped to the offload manager's initialise call, and narrowed to exactly
    the allocation shape it uses for weight buffers: a 1-D pinned tensor.
    Anything else falls through to the real torch.empty untouched.
    """
    import torch

    original = torch.empty
    state = {"ordinal": 0, "shared": 0, "bytes": 0}
    os.makedirs(STORE_DIR, exist_ok=True)

    def patched(*args, **kwargs):
        pinned = kwargs.get("pin_memory")
        one_dim = len(args) == 1 and isinstance(args[0], int)
        if not (pinned and one_dim):
            return original(*args, **kwargs)
        numel = args[0]
        dtype = kwargs.get("dtype")
        try:
            nbytes = numel * torch.empty(0, dtype=dtype).element_size()
            state["ordinal"] += 1
            key = _key(model_id, state["ordinal"], str(dtype), numel)
            path = os.path.join(STORE_DIR, f"{key}.bin")
            tensor = _shared_tensor(torch, path, nbytes, numel, dtype)
            _register_pinned(torch, tensor, nbytes)
            state["shared"] += 1
            state["bytes"] += nbytes
            return tensor
        except Exception as exc:
            logger.warning("deepme: shared buffer failed (%s); using private", exc)
            return original(*args, **kwargs)

    torch.empty = patched
    try:
        yield state
    finally:
        torch.empty = original
        if state["shared"]:
            logger.info(
                "deepme: %d offload buffers (%.1f GiB) backed by shared memory in %s",
                state["shared"],
                state["bytes"] / 1024**3,
                STORE_DIR,
            )


def install() -> bool:
    """Wrap LayerwiseOffloadManager._initialize. Safe to call more than once."""
    if not _enabled():
        return False
    try:
        from sglang.multimodal_gen.runtime.managers.memory_managers import (
            layerwise_offload as lo,
        )
    except Exception:
        return False

    manager = getattr(lo, "LayerwiseOffloadManager", None)
    target = getattr(manager, "_initialize", None)
    if manager is None or target is None:
        logger.warning(
            "deepme: LayerwiseOffloadManager._initialize not found — sglang "
            "internals moved; shared weights disabled (engine unaffected)"
        )
        return False
    if getattr(target, "_deepme_wrapped", False):
        return True

    def _initialize(self, *args, **kwargs):
        # Content identity, so byte-identical components share one copy no
        # matter which partition or replica loaded them. Falls back to the
        # path only if the module cannot be fingerprinted.
        model_id = (
            component_fingerprint(getattr(self, "model", None))
            if getattr(self, "model", None) is not None
            else _guess_model_id()
        )
        if model_id == "unknown":
            model_id = _guess_model_id()
        with _shared_allocator(model_id):
            return target(self, *args, **kwargs)

    # Marker so a second install() is a no-op rather than a double wrap.
    _initialize._deepme_wrapped = True  # type: ignore[attr-defined]
    manager._initialize = _initialize
    logger.info("deepme: shared offload weight store enabled (%s)", STORE_DIR)
    return True


def _guess_model_id() -> str:
    """Identify the checkpoint from the command line when the manager does not
    carry it — the key must not collide across different models."""
    import sys

    argv = sys.argv
    for flag in ("--model-path", "--model"):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                sub = ""
                if "--model-subfolder" in argv:
                    j = argv.index("--model-subfolder")
                    sub = argv[j + 1] if j + 1 < len(argv) else ""
                return f"{argv[i + 1]}::{sub}"
    return "unknown"


# Rank is intentionally absent from the key — see _key(). Sharded components
# already differ by content per rank; replicated ones are identical and must
# share. There is nothing left for a rank probe to decide.


with contextlib.suppress(Exception):
    install()
