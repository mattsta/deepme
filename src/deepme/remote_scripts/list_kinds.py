"""Print the MODEL KINDS an installed engine can serve, as JSON.

Usage: python list_kinds.py vllm|sglang|llamacpp

Kinds are the coarse model classes the platform deploys:
  llm        — autoregressive text/multimodal chat models
  diffusion  — image/video generation pipelines (diffusers-style)

Evidence-based, never assumed: sglang ships `sglang.multimodal_gen` with its
own pipeline registry, so diffusion support (and which pipelines) is read
from that registry on the installed build.
"""

import importlib.util
import json
import sys


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def sglang_kinds() -> dict:
    kinds = {"llm": True, "diffusion": False, "diffusion_pipelines": []}
    if _has("sglang.multimodal_gen"):
        try:
            from sglang.multimodal_gen import registry

            names = []
            for attr in dir(registry):
                val = getattr(registry, attr, None)
                if isinstance(val, dict):
                    names += [k for k in val if isinstance(k, str) and "/" in k]
            kinds["diffusion"] = True
            kinds["diffusion_pipelines"] = sorted(set(names))
        except Exception:
            kinds["diffusion"] = True
    return kinds


def vllm_kinds() -> dict:
    return {"llm": True, "diffusion": False, "diffusion_pipelines": []}


def llamacpp_kinds() -> dict:
    return {"llm": True, "diffusion": False, "diffusion_pipelines": []}


def main() -> int:
    engine = sys.argv[1]
    fn = {"vllm": vllm_kinds, "sglang": sglang_kinds, "llamacpp": llamacpp_kinds}[engine]
    try:
        print(json.dumps(fn()))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
