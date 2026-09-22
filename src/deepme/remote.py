"""Thin SSH execution layer used by every deepme command.

Uses OpenSSH with a shared ControlMaster socket so repeated commands are
fast, and pipes multi-line provisioning scripts over stdin ("bash -s") to
avoid shell-quoting hazards.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from . import config

_SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPath=~/.ssh/deepme-%r@%h:%p",
    "-o",
    "ControlPersist=15m",
]


@dataclass
class Result:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run(
    command: str,
    *,
    check: bool = True,
    timeout: float | None = 600,
    stream: bool = False,
) -> Result:
    """Run a shell command on the server. With stream=True, output goes
    straight to the terminal (for logs/watch commands)."""
    config.require_host()
    argv = ["ssh", *_SSH_OPTS, config.SSH_TARGET, command]
    try:
        if stream:
            proc = subprocess.run(argv, timeout=timeout)
            result = Result(proc.returncode, "", "")
        else:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
            result = Result(proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired:
        result = Result(124, "", f"ssh timed out after {timeout}s")
    if check and not result.ok:
        raise RuntimeError(
            f"remote command failed ({result.returncode}): {command}\n"
            f"stdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
        )
    return result


def run_script(
    script: str, *, check: bool = True, timeout: float | None = 1800
) -> Result:
    """Run a multi-line bash script on the server via stdin."""
    config.require_host()
    argv = ["ssh", *_SSH_OPTS, config.SSH_TARGET, "bash -s"]
    proc = subprocess.run(
        argv, input=script, capture_output=True, text=True, timeout=timeout
    )
    result = Result(proc.returncode, proc.stdout, proc.stderr)
    if check and not result.ok:
        raise RuntimeError(
            f"remote script failed ({result.returncode}):\n"
            f"stdout: {result.stdout[-3000:]}\nstderr: {result.stderr[-3000:]}"
        )
    return result


def write_remote_file(path: str, content: str, *, sudo: bool = False) -> None:
    """Write a file on the server (content piped over stdin, no quoting issues)."""
    config.require_host()
    tee = f"sudo tee {path}" if sudo else f"tee {path}"
    argv = ["ssh", *_SSH_OPTS, config.SSH_TARGET, f"{tee} > /dev/null"]
    proc = subprocess.run(
        argv, input=content, capture_output=True, text=True, timeout=60
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed writing {path}: {proc.stderr[-1000:]}")
