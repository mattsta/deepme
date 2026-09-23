"""Conventions two deployments must AGREE on, pinned by import.

The balancer/gateway deploy standalone and cannot import deepme, so their
copies of shared vocabulary cannot be one definition — but they CAN be one
test. The balancer's own history shows what happens otherwise: a terminal-
status list went two-words-partial in one path and inline cancellations
opened occupancy nothing would ever close.
"""

import importlib.util
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "deepme"


def _load_flat(name: str, directory: Path):
    import os

    os.environ.setdefault("BALANCER_BACKENDS", json.dumps([]))
    os.environ.setdefault("GATEWAY_ROUTES", json.dumps([]))
    sys.path.insert(0, str(directory))
    try:
        spec = importlib.util.spec_from_file_location(name, directory / "server.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(directory))


def test_terminal_status_vocabulary_is_one_vocabulary():
    from deepme import media

    bal = _load_flat("conv_bal", SRC / "balancer")
    assert set(bal.TERMINAL_STATUSES) == set(media.TERMINAL), (
        "client and balancer disagree about when a job is OVER — the exact "
        "drift that once left occupancy open forever"
    )


def test_client_and_reaper_share_the_staleness_rule():
    """'Identical, so client and server agree on what stuck means' was an
    agreement with no enforcement. Now it has one."""
    import balance as bal_logic  # left on sys.path is not guaranteed; import fresh

    from deepme import media

    sys.path.insert(0, str(SRC / "balancer"))
    try:
        import importlib

        bal_logic = importlib.import_module("balance")
    finally:
        sys.path.remove(str(SRC / "balancer"))
    assert media._STALL_RUNNING_MULTIPLE == bal_logic.Balancer.STALE_MULTIPLE
    assert media._STALL_RUNNING_FLOOR_S == bal_logic.Balancer.STALE_FLOOR_S


def test_the_ticket_prefix_is_the_agreed_stable_marker():
    from deepme import media  # noqa: F401 - the client hardcodes q_
