"""Content digests, and a cache so a file is hashed at most once per version.

Dedupe scans and batch journals both need digests of multi-hundred-MB clips.
Recomputing them on every run turns a 20th dedupe pass into 20 full reads of
the archive. Identity here is (size, mtime): a rename keeps both, so a clip
renamed to its item name stays a cache hit; a re-download changes mtime and
is honestly re-hashed.

Deliberately dependency-free (stdlib only): the offline repair tooling that
uses this must not drag in the HTTP generation client to hash a file.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from . import config

CACHE_PATH = config.STATE_DIR / "digest-cache.json"


def file_digest(path: Path) -> str:
    """MD5 of a media file's bytes — content identity, not security.

    stdlib's implementation, not a hand-rolled read loop: it releases the
    GIL around the read/update and picks the platform's buffer size, which
    matters directly here — this runs on pool threads over huge clips on a
    free-threaded build.
    """
    with path.open("rb") as f:
        return hashlib.file_digest(f, "md5").hexdigest()


class DigestCache:
    """(size, mtime) -> digest, shared by every consumer via one file.

    Concurrency-safe across PROCESSES as well as threads: a batch run and a
    dedupe pass legitimately run at once, so `save()` merges with whatever
    is on disk instead of publishing its own stale snapshot, and each writer
    uses its own temp file. Publishing a snapshot wholesale meant a batch
    that started hours earlier erased every entry a dedupe had just
    computed — re-reading the whole archive, which is the exact cost this
    class exists to avoid.
    """

    def __init__(self, path: Path | None = None, entries: dict | None = None):
        self.path = path or CACHE_PATH
        self._entries: dict[str, list] = entries or {}
        self._lock = threading.Lock()
        self._added: dict[str, list] = {}

    @classmethod
    def load(cls, path: Path | None = None) -> DigestCache:
        p = path or CACHE_PATH
        return cls(path=p, entries=_read(p))

    def digest(self, path: Path) -> str:
        """The file's MD5 — computed only if its (size, mtime) changed."""
        key = str(path.resolve())
        st = path.stat()
        with self._lock:
            hit = self._entries.get(key)
            if hit and hit[0] == st.st_size and hit[1] == st.st_mtime:
                return str(hit[2])
        md5 = file_digest(path)
        with self._lock:
            entry = [st.st_size, st.st_mtime, md5]
            self._entries[key] = entry
            self._added[key] = entry
        return md5

    def save(self) -> None:
        """Merge this run's entries into the file. Atomic, and only when
        something changed."""
        with self._lock:
            if not self._added:
                return
            added, self._added = self._added, {}
        merged = {**_read(self.path), **added}
        config.write_text_atomic(self.path, json.dumps(merged))
        with self._lock:
            self._entries = merged


def _read(path: Path) -> dict[str, list]:
    """The cache file, or an empty one. A corrupt cache re-hashes; it must
    never take down the tool that reads it."""
    try:
        return dict(json.loads(path.read_text()))
    except (OSError, ValueError):
        return {}
