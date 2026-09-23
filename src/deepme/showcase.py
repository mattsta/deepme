"""Declarative demo reels: validation that leaves something behind.

Proving concurrency needs several generations running at once, and a
generation costs real GPU minutes. Running four identical smoke tests spends
that and produces nothing; running four DIFFERENT clips produces the same
proof plus a reel that can be shown to anyone.

A reel is deploy/showcase/<profile>.toml — one entry per clip, each
demonstrating a distinct capability. `run()` submits them concurrently
through whatever front end is given (the balancer, normally), so they spread
across replicas, and writes a manifest recording exactly how each was made.
"""

from __future__ import annotations

import concurrent.futures as futures
import dataclasses
import json
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config, media
from .profiles import RunSpec

SHOWCASE_DIR = config.REPO_ROOT / "deploy" / "showcase"
DEFAULT_GALLERY = config.STATE_DIR / "showcase"


@dataclass(frozen=True)
class Clip:
    name: str
    caption: str
    prompt: str
    seconds: int | None = None
    seed: int | None = None
    steps: int | None = None
    params: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass
class ClipResult:
    clip: Clip
    ok: bool
    path: Path | None
    seconds_taken: float
    served_by: str
    error: Any = None


def reel_path(profile: str) -> Path:
    return SHOWCASE_DIR / f"{profile}.toml"


def load(profile: str) -> list[Clip]:
    path = reel_path(profile)
    if not path.exists():
        raise SystemExit(f"no showcase reel for {profile!r} — create {path} with [[clip]] entries")
    raw = tomllib.loads(path.read_text())
    clips = []
    for entry in raw.get("clip", []):
        clips.append(
            Clip(
                name=entry["name"],
                caption=entry.get("caption", ""),
                prompt=entry["prompt"].strip(),
                seconds=entry.get("seconds"),
                seed=entry.get("seed"),
                steps=entry.get("steps"),
                params=dict(entry.get("params", {})),
            )
        )
    if not clips:
        raise SystemExit(f"{path} declares no clips")
    return clips


def _expand(params: dict[str, Any]) -> dict[str, Any]:
    """Dotted keys ("target.aspect_ratio") become nested request objects."""
    body: dict[str, Any] = {}
    for key, value in params.items():
        media.set_path(body, key, value)
    return body


def generate_one(spec: RunSpec, clip: Clip, out_dir: Path, timeout: float) -> ClipResult:
    started = time.monotonic()
    try:
        result = media.generate_video(
            spec,
            clip.prompt,
            seconds=clip.seconds,
            seed=clip.seed,
            steps=clip.steps,
            params=_expand(clip.params),
            out_dir=out_dir,
            timeout=timeout,
        )
    except SystemExit as exc:
        return ClipResult(clip, False, None, time.monotonic() - started, "", str(exc))
    dest = result.path
    if dest and dest.exists():
        # Name the file after the clip so the gallery is readable, not by job id.
        named = dest.with_name(f"{clip.name}{dest.suffix}")
        dest.replace(named)
        dest = named
    return ClipResult(
        clip=clip,
        ok=result.ok,
        path=dest,
        seconds_taken=result.total_s,
        served_by=result.media_id,
        error=result.error,
    )


def run(
    spec: RunSpec,
    clips: list[Clip],
    *,
    out_dir: Path | None = None,
    concurrency: int = 0,
    timeout: float = 5400,
    on_done=None,
) -> list[ClipResult]:
    """Generate every clip, concurrently.

    Concurrency defaults to one request per clip: the front end decides which
    replica each lands on, which is exactly the behaviour under test.
    """
    out = out_dir or DEFAULT_GALLERY
    out.mkdir(parents=True, exist_ok=True)
    workers = concurrency or len(clips)
    results: list[ClipResult] = []
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(generate_one, spec, clip, out, timeout): clip for clip in clips}
        for future in futures.as_completed(pending):
            result = future.result()
            results.append(result)
            if on_done:
                on_done(result)
    results.sort(key=lambda r: [c.name for c in clips].index(r.clip.name))
    write_manifest(out, spec, results)
    return results


def write_manifest(out_dir: Path, spec: RunSpec, results: list[ClipResult]) -> Path:
    """Record how each clip was made, so the reel can be reproduced or cited."""
    path = out_dir / "manifest.json"
    payload = {
        "profile": spec.profile,
        "repo": spec.repo,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "clips": [
            {
                "name": r.clip.name,
                "caption": r.clip.caption,
                "prompt": r.clip.prompt,
                "seconds": r.clip.seconds,
                "seed": r.clip.seed,
                "params": r.clip.params,
                "ok": r.ok,
                "file": r.path.name if r.path else None,
                "took_s": round(r.seconds_taken, 1),
                "job_id": r.served_by,
                "error": r.error,
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path
