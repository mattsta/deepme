"""Report weight bytes per component and dtype, straight from safetensors headers.

Runs on the server against the downloaded checkpoint. Reads only the header of
each shard (an 8-byte little-endian length followed by JSON), so a 62GiB model
is measured in milliseconds without loading anything.

Output: one JSON object on stdout ->
    {"components": {"transformer": {"bytes": N, "tensors": N,
                                    "dtypes": {"BF16": N}}, ...}}

Why this exists: the layerwise offload store allocates one flat CPU buffer per
(layer, dtype) and copies the weights in, so per-rank host RAM should equal
each component's bytes (divided by the TP degree for sharded components).
Comparing that prediction against measured RSS says whether the memory is
weights — which can be shared between replicas — or something else.
"""

from __future__ import annotations

import json
import os
import struct
import sys

# Bytes per element, by safetensors dtype name.
WIDTH = {
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
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F4": 0.5,
    "I4": 0.5,
    "U4": 0.5,
}


def shard_header(path: str) -> dict:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            return {}
        (n,) = struct.unpack("<Q", raw)
        return json.loads(f.read(n) or b"{}")


def measure(root: str) -> dict:
    """Walk every component subdirectory, summing tensor bytes by dtype."""
    out: dict[str, dict] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        shards = [f for f in filenames if f.endswith(".safetensors")]
        if not shards:
            continue
        # Name the component by its path relative to the partition root, so
        # nested weight dirs (video_vae/source/) are reported where they live.
        rel = os.path.relpath(dirpath, root)
        comp = out.setdefault(rel, {"bytes": 0, "tensors": 0, "dtypes": {}})
        for name in shards:
            for tname, meta in shard_header(os.path.join(dirpath, name)).items():
                if tname == "__metadata__" or not isinstance(meta, dict):
                    continue
                dtype = meta.get("dtype", "BF16")
                shape = meta.get("shape") or []
                numel = 1
                for dim in shape:
                    numel *= int(dim)
                nbytes = int(numel * WIDTH.get(dtype, 2))
                comp["bytes"] += nbytes
                comp["tensors"] += 1
                comp["dtypes"][dtype] = comp["dtypes"].get(dtype, 0) + nbytes
    return out


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    print(json.dumps({"root": root, "components": measure(root)}))
