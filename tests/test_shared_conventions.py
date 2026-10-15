"""Conventions two deployments must AGREE on, pinned by import.

The balancer/gateway deploy standalone and cannot import deepme. What they
CAN share is a flat module pushed into both directories — proxycore — and
everything that lives there is now one definition rather than one agreement.
What remains here is the vocabulary those services share with the CLIENT
(deepme.media), which genuinely cannot be one definition, plus the checks
that keep proxycore deployable at all.

The balancer's own history shows what happens without this: a terminal-
status list went two-words-partial in one path, inline cancellations opened
occupancy nothing would ever close, and the owner probe's misattribution
guard was fixed in one file and not the other.
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
    sys.path.insert(0, str(SRC / "shared"))  # proxycore deploys flat alongside
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


def test_both_services_deploy_proxycore():
    """A shared flat module that only ONE ctl pushes is worse than no shared
    module: the service that misses it dies at import on the next restart,
    and nothing local can tell — ruff, mypy and this suite all pass, because
    they read the source tree rather than what was copied to the server."""
    from deepme import balancerctl, gatewayctl

    for mod in (balancerctl, gatewayctl):
        source = Path(mod.__file__).read_text()
        assert "proxycore.py" in source, (
            f"{mod.__name__} does not push proxycore.py — its service will "
            f"fail at import on the server after the next restart"
        )


def test_proxycore_imports_nothing_from_deepme():
    """On the server there is no `deepme` package — the services are flat
    files. An import that resolves locally and not there is invisible until
    a restart, which is the worst possible moment to discover it."""
    import ast

    # Parsed, not grepped: this module's own docstring explains the rule and
    # so contains the words a text scan would flag.
    tree = ast.parse((SRC / "shared" / "proxycore.py").read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    offenders = [n for n in names if n.split(".")[0] == "deepme"]
    assert not offenders, f"proxycore cannot import from deepme: {offenders}"


def test_the_ticket_prefix_is_the_agreed_stable_marker():
    from deepme import media  # noqa: F401 - the client hardcodes q_
