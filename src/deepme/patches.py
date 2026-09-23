"""Vendored patches against engine source, and how to carry them forward.

We patch installed engine code when a capability we need is not reachable
from configuration. That is a maintenance liability unless three things are
true, which this module enforces:

  1. Every patch records the upstream version it was written against AND the
     sha256 of each pristine target file. If an upgrade moves that code, the
     patch is reported STALE and refuses to apply — silently mis-applying to
     shifted code is the failure mode that turns a fork into an outage.
  2. Applying is idempotent, so provisioning can run it unconditionally and a
     fresh machine ends up identical to a long-lived one.
  3. Applied/absent is DETECTED from the file on disk (a reverse dry-run),
     never tracked in a state file that can drift from reality.

Layout: deploy/patches/<engine>/manifest.toml + NNN-*.patch, where each patch
is a unified diff rooted at the engine venv's site-packages directory.
"""

from __future__ import annotations

import shlex
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import config, remote

PATCH_ROOT = config.REPO_ROOT / "deploy" / "patches"

# What a target file can be, from our point of view.
CLEAN = "clean"  # pristine upstream, patch not applied
APPLIED = "applied"  # our patch is in place
STALE = "stale"  # upstream changed under us — re-base required
MISSING = "missing"  # target file absent entirely


@dataclass(frozen=True)
class Patch:
    engine: str
    name: str  # patch filename
    description: str
    upstream_version: str
    targets: tuple[str, ...]  # paths relative to site-packages
    sha256: dict[str, str]  # target -> pristine sha256

    @property
    def path(self) -> Path:
        return PATCH_ROOT / self.engine / self.name

    def text(self) -> str:
        return self.path.read_text()


@dataclass(frozen=True)
class PatchState:
    patch: Patch
    state: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.state in (CLEAN, APPLIED)


def patch_dir(engine: str) -> Path:
    return PATCH_ROOT / engine


def load(engine: str = "sglang") -> list[Patch]:
    """Patches declared for an engine, in application order."""
    manifest = patch_dir(engine) / "manifest.toml"
    if not manifest.exists():
        return []
    raw = tomllib.loads(manifest.read_text())
    out = [
        Patch(
            engine=engine,
            name=entry["file"],
            description=entry.get("description", ""),
            upstream_version=entry.get("upstream_version", ""),
            targets=tuple(entry.get("targets", [])),
            sha256=dict(entry.get("sha256", {})),
        )
        for entry in raw.get("patch", [])
    ]
    return sorted(out, key=lambda p: p.name)


def site_packages(engine: str) -> str:
    """The engine venv's site-packages on the server (patch root)."""
    venv = config.engine_venv(engine)
    return f"{venv}/lib/python{config.ENGINE_PYTHON}/site-packages"


def _remote_shas(engine: str, targets: list[str]) -> dict[str, str]:
    if not targets:
        return {}
    root = site_packages(engine)
    listed = " ".join(shlex.quote(f"{root}/{t}") for t in targets)
    r = remote.run(f"sha256sum {listed} 2>/dev/null || true", check=False)
    out = {}
    for line in r.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[1].removeprefix(f"{root}/")] = parts[0]
    return out


def _push(patch: Patch) -> str:
    """Copy the patch to the server; return its remote path."""
    dest = f"{config.REMOTE_HOME}/deepme-patches/{patch.engine}"
    remote.run(f"mkdir -p {shlex.quote(dest)}")
    path = f"{dest}/{patch.name}"
    remote.write_remote_file(path, patch.text())
    return path


def _dry_run(patch: Patch, *, reverse: bool) -> bool:
    """Would this patch apply (or un-apply) cleanly right now?"""
    path = _push(patch)
    flag = "-R " if reverse else ""
    r = remote.run(
        f"cd {shlex.quote(site_packages(patch.engine))} && "
        f"patch -p1 --dry-run --force {flag}< {shlex.quote(path)}",
        check=False,
    )
    return r.returncode == 0


def inspect(patch: Patch) -> PatchState:
    """Where this patch stands against the code actually installed."""
    shas = _remote_shas(patch.engine, list(patch.targets))
    missing = [t for t in patch.targets if t not in shas]
    if missing:
        return PatchState(patch, MISSING, f"target(s) not installed: {missing}")
    # Reverse-applying cleanly is proof our changes are present.
    if _dry_run(patch, reverse=True):
        return PatchState(patch, APPLIED, "verified by reverse dry-run")
    # Not applied: the pristine file must match what we were written against.
    drifted = [t for t, want in patch.sha256.items() if want and shas.get(t) and shas[t] != want]
    if drifted:
        return PatchState(
            patch,
            STALE,
            f"upstream changed under us: {drifted} — re-base against "
            f"{patch.upstream_version or 'the new version'} before applying",
        )
    if _dry_run(patch, reverse=False):
        return PatchState(patch, CLEAN, "applies cleanly")
    return PatchState(patch, STALE, "does not apply and is not applied — re-base required")


def status(engine: str = "sglang") -> list[PatchState]:
    return [inspect(p) for p in load(engine)]


def apply(engine: str = "sglang", *, force: bool = False) -> list[str]:
    """Apply every patch that is not already applied. Idempotent.

    Refuses on STALE unless forced: applying a diff to code that moved is how
    a fork silently corrupts an engine.
    """
    log = []
    for st in status(engine):
        if st.state == APPLIED:
            log.append(f"[=] {st.patch.name}: already applied")
            continue
        if st.state in (STALE, MISSING) and not force:
            log.append(f"[!] {st.patch.name}: {st.state} — {st.detail}")
            continue
        path = _push(st.patch)
        r = remote.run(
            f"cd {shlex.quote(site_packages(engine))} && patch -p1 --forward < {shlex.quote(path)}",
            check=False,
        )
        ok = r.returncode == 0
        log.append(
            f"[{'+' if ok else 'x'}] {st.patch.name}: "
            f"{'applied' if ok else r.stdout.strip() or r.stderr.strip()}"
        )
    return log


def revert(engine: str = "sglang") -> list[str]:
    """Return the engine to pristine upstream — the escape hatch when a patch
    is suspected in an incident."""
    log = []
    for st in status(engine):
        if st.state != APPLIED:
            log.append(f"[=] {st.patch.name}: not applied")
            continue
        path = _push(st.patch)
        r = remote.run(
            f"cd {shlex.quote(site_packages(engine))} && patch -p1 -R < {shlex.quote(path)}",
            check=False,
        )
        log.append(f"[{'-' if r.returncode == 0 else 'x'}] {st.patch.name}: reverted")
    return log


HOOKS_DIR = config.REPO_ROOT / "src" / "deepme" / "hooks"
# A .pth file whose line starts with `import` is executed at interpreter
# startup — the supported way to install a runtime hook without editing any
# engine source, so there is no diff to re-base when upstream reformats.
PTH_NAME = "deepme_hooks.pth"


def install_hooks(engine: str = "sglang") -> list[str]:
    """Install runtime hooks into an engine venv. Idempotent."""
    root = site_packages(engine)
    log = []
    for path in sorted(HOOKS_DIR.glob("*.py")):
        remote.write_remote_file(f"{root}/{path.name}", path.read_text())
        log.append(f"[+] {path.name}")
    remote.write_remote_file(f"{root}/{PTH_NAME}", "import deepme_hooks\n")
    log.append(f"[+] {PTH_NAME} (imports hooks at interpreter start)")
    return log


def remove_hooks(engine: str = "sglang") -> list[str]:
    """Remove the hooks — the escape hatch if one is suspected in an incident."""
    root = shlex.quote(site_packages(engine))
    names = " ".join(shlex.quote(p.name) for p in HOOKS_DIR.glob("*.py"))
    remote.run(f"cd {root} && rm -f {PTH_NAME} {names}", check=False)
    return [f"[-] removed hooks and {PTH_NAME}"]


def hooks_installed(engine: str = "sglang") -> bool:
    root = shlex.quote(site_packages(engine))
    r = remote.run(f"test -f {root}/{PTH_NAME} && echo yes || echo no", check=False)
    return "yes" in r.stdout


def record_pristine(engine: str, targets: list[str]) -> dict[str, str]:
    """Checksums to paste into a manifest when authoring or re-basing a patch.

    Run against an UNPATCHED engine; these are what later upgrades are
    compared to.
    """
    return _remote_shas(engine, targets)
