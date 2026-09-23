"""Print the weight formats an installed engine can actually load, as JSON.

Usage: python list_formats.py vllm|sglang|llamacpp
Evidence-based, never assumed:
  vllm    — presence of a gguf loader module in the installed package
  sglang  — 'gguf' among the server's --load-format argparse choices
  llamacpp— gguf by definition of the engine

deepme caches this per-engine (invalidated by `deepme setup`) and uses
declared-launchable ∩ probed as the effective format support.
"""

import importlib
import importlib.util
import json
import sys


def vllm_formats() -> list[str]:
    fmts = ["safetensors"]
    for mod in (
        "vllm.model_executor.model_loader.gguf_loader",
        "vllm.model_executor.model_loader.gguf",
    ):
        if importlib.util.find_spec(mod) is not None:
            fmts.append("gguf")
            break
    return fmts


def sglang_formats() -> list[str]:
    fmts = ["safetensors"]
    import argparse

    from sglang.srt.server_args import ServerArgs

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    if any(
        "--load-format" in (action.option_strings or [])
        and action.choices
        and "gguf" in action.choices
        for action in parser._actions
    ):
        fmts.append("gguf")
    return fmts


def llamacpp_formats() -> list[str]:
    return ["gguf"]


def main() -> int:
    engine = sys.argv[1]
    fn = {"vllm": vllm_formats, "sglang": sglang_formats, "llamacpp": llamacpp_formats}[engine]
    try:
        print(json.dumps(sorted(set(fn()))))
    except Exception as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
