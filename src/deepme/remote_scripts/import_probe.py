"""Probe module imports inside an engine venv and print full tracebacks.

Usage: python import_probe.py module.a module.b ...
Prints "OK <mod>" or "FAIL <mod>" followed by the traceback, so silent
"Ignore import error" fallbacks in engines can be diagnosed.
"""

import importlib
import sys
import traceback


def main() -> int:
    failures = 0
    for mod in sys.argv[1:]:
        try:
            importlib.import_module(mod)
            print(f"OK {mod}", flush=True)
        except BaseException:
            failures += 1
            print(f"FAIL {mod}", flush=True)
            traceback.print_exc()
            print("---", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
