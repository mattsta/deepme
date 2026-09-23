"""One physical copy of a model's offloaded weights, shared by every replica.

Layerwise offload keeps weights in host RAM as one flat pinned buffer per
(layer, dtype). Pinned memory is page-locked into ONE process, so N replicas
of the same model hold N identical private copies — measured at 96GiB per
rank, which is what limits replica count long before the GPUs do.

This backs those buffers with a file in /dev/shm instead. The first process to
need a buffer builds it under an exclusive lock; every later process maps the
same pages. Pinned-DMA throughput is preserved separately by registering the
mapping with CUDA (cudaHostRegister page-locks an EXISTING mapping, which is
what makes shared and pinned compatible at all).

Deployed to the engine's site-packages as `deepme_sharedstore.py` and called
from a small patch to layerwise_offload.py — the logic lives here, where we
can test it, rather than inside a diff.

Safety properties:
  * A buffer is only readable by others after its `.ready` marker exists, so a
    half-filled buffer is never mistaken for a complete one.
  * The key covers every byte of layout (shapes, offsets, dtype, order), so
    two different tensors can never collide on one file.
  * Any failure falls back to private allocation. Sharing is an optimisation;
    it must never be the reason a launch fails.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Files live here: tmpfs, so "writing" a buffer only ever costs RAM.
STORE_DIR = os.environ.get("DEEPME_SHARED_STORE_DIR", "/dev/shm/deepme-weights")
ENABLED_ENV = "DEEPME_SHARED_WEIGHTS"
VERIFY_ENV = "DEEPME_SHARED_WEIGHTS_VERIFY"


def enabled() -> bool:
    """Opt-in. Unset means the engine behaves exactly as upstream."""
    return os.environ.get(ENABLED_ENV, "").lower() in ("1", "true", "yes")


def verifying() -> bool:
    """Build privately as well and compare, to prove the shared bytes match."""
    return os.environ.get(VERIFY_ENV, "").lower() in ("1", "true", "yes")


def buffer_key(
    *,
    model_id: str,
    component: str,
    layer_idx: int,
    dtype: str,
    total_numel: int,
    tp_rank: int,
    layout: list[tuple[str, int, int]],
) -> str:
    """Content address for one flat buffer.

    Everything that determines the bytes goes in: which checkpoint, which
    component and layer, the dtype, the total length, this rank's shard, and
    the (name, offset, numel) of every tensor packed inside. Two replicas of
    the same deployment produce the same key; anything else cannot collide.
    """
    payload = json.dumps(
        {
            "model": model_id,
            "component": component,
            "layer": layer_idx,
            "dtype": dtype,
            "numel": total_numel,
            "rank": tp_rank,
            "layout": [list(x) for x in layout],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


def paths(key: str) -> tuple[str, str, str]:
    """(data file, ready marker, lock file) for a key.

    Strings, not Path objects: callers hand these to mmap/CUDA plumbing.
    """
    base = Path(STORE_DIR) / key
    return f"{base}.bin", f"{base}.ready", f"{base}.lock"


@dataclass
class Acquired:
    """A buffer, and whether this process must still fill it."""

    tensor: Any
    prefilled: bool  # True: another process already wrote these bytes
    shared: bool  # False: fell back to a private allocation
    key: str = ""


def _fresh(data_path: str, ready_path: str, nbytes: int) -> bool:
    """Is there a complete buffer of the right size already?"""
    try:
        return Path(ready_path).exists() and Path(data_path).stat().st_size == nbytes
    except OSError:
        return False


def acquire(
    key: str,
    nbytes: int,
    *,
    map_shared: Callable[[str, int], Any],
    allocate_private: Callable[[], Any],
    lock: Callable[[str], Any] | None = None,
) -> Acquired:
    """Map the shared buffer for `key`, building it if we are first.

    `map_shared(path, nbytes)` returns a tensor backed by that file;
    `allocate_private()` is the upstream allocation, used as the fallback.
    The caller fills the tensor when `prefilled` is False and MUST call
    `publish(key)` afterwards so other processes may use it.
    """
    if not enabled():
        return Acquired(allocate_private(), prefilled=False, shared=False)
    data_path, ready_path, lock_path = paths(key)
    try:
        Path(STORE_DIR).mkdir(parents=True, exist_ok=True)
        guard = (lock or _flock)(lock_path)
        with guard:
            already = _fresh(data_path, ready_path, nbytes)
            tensor = map_shared(data_path, nbytes)
            return Acquired(tensor, prefilled=already, shared=True, key=key)
    except Exception:
        # Never let an optimisation stop a model from loading.
        return Acquired(allocate_private(), prefilled=False, shared=False)


def publish(key: str) -> None:
    """Mark a buffer complete so other processes may map it."""
    _, ready_path, _ = paths(key)
    with contextlib.suppress(OSError):
        Path(ready_path).write_text("ok")


def _flock(path: str):
    """Exclusive lock held for the build, so exactly one process fills."""
    import fcntl

    class _Lock:
        def __enter__(self):
            self.fh = Path(path).open("w")
            fcntl.flock(self.fh, fcntl.LOCK_EX)
            return self

        def __exit__(self, *exc):
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            return False

    return _Lock()


def orphans() -> list[str]:
    """Store files no live process is mapping.

    These are what a key-scheme change leaves behind: the previous keys are
    still on tmpfs holding RAM, and nothing will ever ask for them again.
    Observed: changing to content addressing grew the store from 123.5GiB to
    202GiB because the old path-keyed buffers were never released.
    """
    out: list[str] = []
    try:
        entries = list(Path(STORE_DIR).iterdir())
    except OSError:
        return out
    out.extend(str(p) for p in entries if not _in_use(str(p)))
    return out


def usage() -> dict:
    """What the store currently holds — for observability and leak hunting."""
    total = 0
    files = 0
    try:
        for p in Path(STORE_DIR).iterdir():
            if p.name.endswith(".bin"):
                total += p.stat().st_size
                files += 1
    except OSError:
        pass
    stale = orphans()
    stale_bytes = 0
    for path in stale:
        with contextlib.suppress(OSError):
            stale_bytes += Path(path).stat().st_size
    return {
        "dir": STORE_DIR,
        "buffers": files,
        "bytes": total,
        "orphans": len(stale),
        "orphan_bytes": stale_bytes,
    }


def clear(*, only_unused: bool = True) -> int:
    """Remove store files. Segments outlive the processes that made them, so
    a model that will not be served again leaks RAM until this runs."""
    removed = 0
    try:
        entries = list(Path(STORE_DIR).iterdir())
    except OSError:
        return 0
    for p in entries:
        if only_unused and _in_use(str(p)):
            continue
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def main(argv: list[str] | None = None) -> int:
    """Run on the server: `python sharedstore.py status|clear [--all]`.

    Segments outlive the processes that created them, so a model that will
    not be served again holds RAM until this runs.
    """
    import sys

    args = argv if argv is not None else sys.argv[1:]
    action = args[0] if args else "status"
    if action == "clear":
        removed = clear(only_unused="--all" not in args)
        print(json.dumps({"removed": removed, **usage()}))
    else:
        print(json.dumps(usage()))
    return 0


def _in_use(path: str) -> bool:
    """Is any live process mapping this file?"""
    try:
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if path in (proc / "maps").read_text():
                    return True
            except OSError:
                continue
    except OSError:
        return True  # cannot tell: assume in use rather than free live memory
    return False


if __name__ == "__main__":  # pragma: no cover - server-side entry point
    raise SystemExit(main())
