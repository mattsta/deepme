"""The shared digest cache: hash a file at most once per version.

Twenty dedupe passes over a terabyte of clips must cost one read, not
twenty — and a changed file must never serve a stale hash.
"""

from deepme import digests


def test_an_unchanged_file_is_hashed_exactly_once(tmp_path, monkeypatch):
    calls = []
    real = digests.file_digest
    monkeypatch.setattr(digests, "file_digest", lambda p: calls.append(p) or real(p))
    f = tmp_path / "a.mp4"
    f.write_bytes(b"CONTENT")
    cache = digests.DigestCache(path=tmp_path / "cache.json")
    first = cache.digest(f)
    again = cache.digest(f)
    assert first == again
    assert len(calls) == 1, "second read served from the cache"


def test_a_changed_file_is_rehashed(tmp_path):
    import os

    f = tmp_path / "a.mp4"
    f.write_bytes(b"OLD")
    cache = digests.DigestCache(path=tmp_path / "cache.json")
    old = cache.digest(f)
    f.write_bytes(b"NEWBYTES")
    os.utime(f, (9999999999, 9999999999))
    assert cache.digest(f) != old


def test_the_cache_survives_a_round_trip_and_tolerates_corruption(tmp_path):
    f = tmp_path / "a.mp4"
    f.write_bytes(b"CONTENT")
    path = tmp_path / "cache.json"
    c1 = digests.DigestCache(path=path)
    md5 = c1.digest(f)
    c1.save()
    c2 = digests.DigestCache.load(path)
    assert c2.digest(f) == md5
    path.write_text("{corrupt")
    c3 = digests.DigestCache.load(path)
    assert c3.digest(f) == md5, "a corrupt cache re-hashes, never crashes"
