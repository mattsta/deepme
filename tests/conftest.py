"""Hard isolation: tests can NEVER reach live infrastructure.

This is enforcement, not convention. A progress-probe fallback once SSHed to
the production server from inside a pytest run — journal reads on every poll
tick, minutes of wall time, against hardware running paying work — because
nothing made "tests are offline" true by construction. Now it is:

  * every SSH goes through `remote.run`/`run_script`/`write_remote_file`,
    and those raise here unless a test explicitly patches its own seam;
  * the configured server host is poisoned to an unresolvable name, so any
    live HTTP attempt (balancer/gateway api_base) fails instantly instead
    of touching production.

A test that needs remote behaviour patches the exact function it exercises;
the default is a loud failure naming the attempt.
"""

import pytest


@pytest.fixture(autouse=True)
def _no_live_infrastructure(monkeypatch):
    from deepme import config, remote

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "test attempted to reach the LIVE server through deepme.remote — "
            "patch the specific seam (remote.run / stage_progress / "
            "balancerctl.stats ...) instead of letting code dial out: "
            f"args={args[:1]}"
        )

    monkeypatch.setattr(remote, "run", forbidden)
    monkeypatch.setattr(remote, "run_script", forbidden)
    monkeypatch.setattr(remote, "write_remote_file", forbidden, raising=False)
    monkeypatch.setattr(config, "SSH_HOST", "tests-must-not-dial-out.invalid", raising=False)
