"""Chained generation: extend a clip from the last frame of the one before.

H3's fl2va task pins an image to a chosen frame, so the final frame of one
clip can seed the next and the pair reads as continuous motion rather than a
cut. Repeat N times and concatenate, and a 5-second model produces a minutes
-long sequence.

This is unavoidably SERIAL — segment N+1 cannot start until N has produced
the frame it begins from — so unlike a batch it cannot be spread across
replicas. What it can do is survive interruption: every segment is recorded
as it completes, so a long story resumes instead of regenerating.

The seam matters. The first frame of a segment IS the last frame of its
predecessor, so joining them naively shows that frame twice; the join drops
the duplicate.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import config, media
from .profiles import RunSpec

STORY_DIR = config.STATE_DIR / "stories"

PENDING, DONE, FAILED = "pending", "done", "failed"


@dataclass
class Segment:
    index: int
    prompt: str
    seconds: int | None = None
    seed: int | None = None
    steps: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    status: str = PENDING
    file: str | None = None
    took_s: float = 0.0
    error: Any = None

    @property
    def name(self) -> str:
        return f"seg{self.index:03d}"


@dataclass
class Story:
    name: str
    profile: str
    out_dir: str
    segments: list[Segment]

    @property
    def dir(self) -> Path:
        return Path(self.out_dir)

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def final_path(self) -> Path:
        return self.dir / f"{self.name}.mp4"

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "profile": self.profile,
                    "out_dir": self.out_dir,
                    "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    "segments": [asdict(s) for s in self.segments],
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, name: str, out_dir: Path | None = None) -> Story | None:
        path = (out_dir or STORY_DIR / name) / "state.json"
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return cls(
            name=raw["name"],
            profile=raw["profile"],
            out_dir=raw["out_dir"],
            segments=[Segment(**s) for s in raw["segments"]],
        )

    def done_clips(self) -> list[Path]:
        return [self.dir / s.file for s in self.segments if s.status == DONE and s.file]


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit(
            "ffmpeg is required to chain and join segments — install it "
            "(brew install ffmpeg) or run the segments individually"
        )
    return exe


def last_frame(video: Path, dest: Path) -> Path:
    """Extract the final frame as a PNG, to seed the next segment.

    Seeks from the END rather than decoding the whole file, and takes the
    last frame the decoder yields.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _ffmpeg(),
        "-v",
        "error",
        "-y",
        "-sseof",
        "-1",  # last second only
        "-i",
        str(video),
        "-update",
        "1",  # keep overwriting, so the last wins
        "-q:v",
        "2",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    if not dest.exists():
        raise SystemExit(f"could not extract a final frame from {video}")
    return dest


def concat(clips: list[Path], dest: Path, *, drop_seam_frame: bool = True) -> Path:
    """Join segments into one file.

    Streams are copied, not re-encoded: every segment comes from the same
    model at the same geometry, so their codecs already match and a copy
    keeps it lossless and fast.

    `drop_seam_frame` trims the opening frame of each continuation, which is
    a duplicate of the frame it was seeded from.
    """
    if not clips:
        raise SystemExit("no finished segments to join")
    dest.parent.mkdir(parents=True, exist_ok=True)
    listing = dest.parent / f"{dest.stem}-inputs.txt"
    parts = []
    for i, clip in enumerate(clips):
        trimmed = clip
        if drop_seam_frame and i:
            trimmed = clip.with_name(f"{clip.stem}-trimmed{clip.suffix}")
            subprocess.run(
                [
                    _ffmpeg(),
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(clip),
                    # 24fps fixed: one frame is ~0.0417s.
                    "-ss",
                    "0.0417",
                    "-c",
                    "copy",
                    str(trimmed),
                ],
                check=True,
                capture_output=True,
            )
        parts.append(trimmed)
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    subprocess.run(
        [
            _ffmpeg(),
            "-v",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            str(dest),
        ],
        check=True,
        capture_output=True,
    )
    return dest


def run(
    spec: RunSpec,
    story: Story,
    *,
    timeout: float = 7200,
    on_done=None,
) -> Story:
    """Generate every pending segment in order, each seeded by the last.

    Segment 0 is plain text-to-video; every later one is fl2va conditioned on
    its predecessor's final frame. A failure stops the chain rather than
    continuing from a frame that does not exist.
    """
    story.dir.mkdir(parents=True, exist_ok=True)
    frames = story.dir / "frames"

    for i, seg in enumerate(story.segments):
        if seg.status == DONE and seg.file and (story.dir / seg.file).exists():
            continue
        params = dict(seg.params)
        if i:
            previous = story.segments[i - 1]
            if not previous.file:
                raise SystemExit(f"{seg.name} continues {previous.name}, which has no clip")
            seed_frame = last_frame(story.dir / previous.file, frames / f"{previous.name}.png")
            params.setdefault("task", "fl2va")
            params.setdefault("conditions", media.keyframe_conditions(str(seed_frame), None))
        started = time.monotonic()
        try:
            result = media.generate_video(
                spec,
                seg.prompt,
                seconds=seg.seconds,
                seed=seg.seed,
                steps=seg.steps,
                params=params,
                out_dir=story.dir,
                timeout=timeout,
            )
            if result.ok and result.path:
                named = result.path.with_name(f"{seg.name}{result.path.suffix}")
                result.path.replace(named)
                seg.file, seg.status = named.name, DONE
            else:
                seg.status, seg.error = FAILED, result.error
            seg.took_s = result.total_s
        except Exception as exc:
            seg.status, seg.error = FAILED, str(exc)
            seg.took_s = time.monotonic() - started
        story.save()
        if on_done:
            on_done(seg, story)
        if seg.status == FAILED:
            raise SystemExit(
                f"{seg.name} failed ({seg.error}) — the chain stops here; fix "
                f"and re-run to resume from this segment"
            )
    return story
