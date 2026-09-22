"""Print the model architectures supported by an installed engine, as JSON.

Usage: python list_archs.py vllm|sglang
Run inside the engine's venv; deepme uses this to filter the model catalog
to what the installed engines can actually serve (real data, not guesses).
"""

import json
import sys


def vllm_archs() -> list[str]:
    from vllm.model_executor.models.registry import ModelRegistry

    return sorted(ModelRegistry.get_supported_archs())


def sglang_archs() -> list[str]:
    try:
        from sglang.srt.models.registry import ModelRegistry

        return sorted(ModelRegistry.get_model_archs())
    except Exception:
        # fallback: scan model modules for entry classes
        import importlib
        import pkgutil

        import sglang.srt.models as m

        archs = set()
        for info in pkgutil.iter_modules(m.__path__):
            try:
                mod = importlib.import_module(f"sglang.srt.models.{info.name}")
                for cls in getattr(mod, "EntryClass", []) or []:
                    archs.add(cls.__name__)
            except Exception:
                continue
        return sorted(archs)


def main() -> int:
    engine = sys.argv[1]
    archs = vllm_archs() if engine == "vllm" else sglang_archs()
    print(json.dumps(archs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
