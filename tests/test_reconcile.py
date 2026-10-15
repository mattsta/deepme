"""The duplicate-content repair: keep the oldest, delete the rest, and
reset the losing items so they REGENERATE — never re-adopt the wrong job.

The stakes are the incident itself: prompts already hold wrong videos, so a
repair that deletes files but leaves job ids behind re-downloads the very
misattribution being repaired.
"""

import json
import os
from pathlib import Path

import pytest

from deepme import batch as batch_mod
from deepme import digests, reconcile


@pytest.fixture(autouse=True)
def _isolated_digest_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(digests, "CACHE_PATH", tmp_path / "digest-cache.json")


def _clip(path: Path, content: bytes, mtime: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))
    return path


def _state(tmp_path, name, items):
    d = tmp_path / "batches" / name
    d.mkdir(parents=True)
    (d / "state.json").write_text(
        json.dumps(
            {
                "name": name,
                "profile": "h3-video",
                "out_dir": str(tmp_path / name / "clips"),
                "items": items,
            }
        )
    )


def _item(index, file=None, job_id=None, status="done", prompt=None, seed=None):
    return {
        "index": index,
        "prompt": prompt if prompt is not None else f"p{index}",
        "status": status,
        "file": file,
        "job_id": job_id,
        "seed": seed,
    }


def test_duplicates_keep_the_oldest_and_drop_every_newer_copy(tmp_path):
    a = _clip(tmp_path / "a" / "0001.mp4", b"SAME", 100.0)
    b = _clip(tmp_path / "a" / "0007.mp4", b"SAME", 200.0)
    c = _clip(tmp_path / "b" / "0002.mp4", b"SAME", 300.0)
    unique = _clip(tmp_path / "a" / "0003.mp4", b"OTHER", 50.0)
    plan = reconcile.build_plan(reconcile.scan_dirs([tmp_path / "a", tmp_path / "b"]), {})
    assert plan.scanned == 4
    (group,) = plan.groups
    assert group.keep == a
    assert set(group.drop) == {b, c}, "newer copies go, across directories"
    assert unique.exists()


def test_non_media_bookkeeping_is_never_scanned(tmp_path):
    _clip(tmp_path / "a" / "state.json", b"SAME", 100.0)
    _clip(tmp_path / "a" / "state2.json", b"SAME", 200.0)
    assert reconcile.build_plan(reconcile.scan_dirs([tmp_path / "a"]), {}).groups == []


def test_apply_deletes_and_resets_items_with_job_ids_cleared(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    old = _clip(tmp_path / "one" / "clips" / "0001.mp4", b"SAME", 100.0)
    dup = _clip(tmp_path / "two" / "clips" / "0060.mp4", b"SAME", 900.0)
    _state(
        tmp_path,
        "one",
        [_item(1, file="0001.mp4", job_id="vid_good")],
    )
    _state(
        tmp_path,
        "two",
        [_item(60, file="0060.mp4", job_id="vid_wrong"), _item(61, file=None, status="pending")],
    )
    dirs, owners = reconcile.batch_clip_index()
    plan = reconcile.build_plan(reconcile.scan_dirs(dirs), owners)
    assert [(n, i) for n, i, _ in plan.resets] == [("two", 60)]

    deleted, reset = reconcile.apply_plan(plan)
    assert (deleted, reset) == (1, 1)
    assert old.exists() and not dup.exists()

    two = batch_mod.Batch.load("two")
    repaired = two.items[0]
    assert repaired.status == "pending"
    assert repaired.file is None
    assert repaired.job_id is None, "keeping the id would re-adopt the wrong job"
    untouched = batch_mod.Batch.load("one").items[0]
    assert untouched.status == "done"
    assert untouched.job_id == "vid_good"


def test_every_newer_copy_is_deleted_and_its_item_told_not_to_regenerate(tmp_path, monkeypatch):
    """MD5 DECIDES DELETION: identical bytes are a duplicate file, always.

    An earlier version SPARED copies that shared the keeper's request,
    because deleting one reset its item to PENDING and the next run remade
    it — the delete/regenerate loop. Sparing was a workaround for that loop,
    not a judgement about the file, and it left `dedupe --apply` reporting
    duplicates and deleting nothing. The loop is now closed by marking the
    item SKIPPED instead of PENDING, so the file can go.

    The spec decides only what happens to the ITEM: same request as the
    keeper -> SKIPPED (the keeper's file IS its output); different request
    -> PENDING (a deterministic engine cannot make one video from two
    requests, so the file was never that item's honest output)."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"SAME", 100.0)
    _clip(tmp_path / "two" / "clips" / "0002.mp4", b"SAME", 200.0)  # twin
    _clip(tmp_path / "two" / "clips" / "0003.mp4", b"SAME", 300.0)  # imposter
    _state(tmp_path, "one", [_item(1, file="0001.mp4", prompt="corgi", seed=7)])
    _state(
        tmp_path,
        "two",
        [
            _item(2, file="0002.mp4", prompt="corgi", seed=7),
            _item(3, file="0003.mp4", prompt="mecha", seed=9, job_id="vid_w"),
        ],
    )
    dirs, owners = reconcile.batch_clip_index()
    plan = reconcile.build_plan(reconcile.scan_dirs(dirs), owners)
    (group,) = plan.groups
    assert group.keep.name == "0001.mp4", "the OLDEST survives"
    assert sorted(p.name for p in group.drop) == ["0002.mp4", "0003.mp4"], (
        "every newer copy of identical bytes is deleted"
    )
    assert [(n, i) for n, i, _p, _k in plan.skips] == [("two", 2)], "same request: skip"
    assert [(n, i) for n, i, _ in plan.resets] == [("two", 3)], "different request: reset"


def test_a_seedless_duplicate_is_deleted_and_never_regenerated(tmp_path, monkeypatch):
    """End to end, on the shape the operator actually has: the same prompt
    in two backlogs, no explicit seed, identical bytes. The newer file goes,
    the item is SKIPPED, and a later run does not remake it."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"SAME", 100.0)
    _clip(tmp_path / "two" / "clips" / "0002.mp4", b"SAME", 200.0)
    _state(tmp_path, "one", [_item(1, file="0001.mp4", prompt="corgi")])
    _state(tmp_path, "two", [_item(2, file="0002.mp4", prompt="corgi")])
    dirs, owners = reconcile.batch_clip_index()
    plan = reconcile.build_plan(reconcile.scan_dirs(dirs), owners)

    deleted, changed = reconcile.apply_plan(plan)
    assert (deleted, changed) == (1, 1)
    assert (tmp_path / "one" / "clips" / "0001.mp4").exists(), "the oldest survives"
    assert not (tmp_path / "two" / "clips" / "0002.mp4").exists(), "the newer copy goes"

    two = batch_mod.Batch.load("two")
    assert two.items[0].status == batch_mod.SKIPPED, "NEVER pending: that is the loop"
    assert two.items[0].job_id is None
    assert "0001.mp4" in two.items[0].error, "it names the copy that survived"
    assert two.todo() == [], "a re-run generates nothing"


def test_identical_bytes_from_DIFFERENT_prompts_are_still_repaired(tmp_path, monkeypatch):
    """The real misattribution signal, seedless or not: a deterministic
    engine cannot produce one video from two different requests."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"SAME", 100.0)
    _clip(tmp_path / "two" / "clips" / "0002.mp4", b"SAME", 200.0)
    _state(tmp_path, "one", [_item(1, file="0001.mp4", prompt="corgi")])
    _state(tmp_path, "two", [_item(2, file="0002.mp4", prompt="a mecha")])
    dirs, owners = reconcile.batch_clip_index()
    plan = reconcile.build_plan(reconcile.scan_dirs(dirs), owners)
    (group,) = plan.groups
    assert [p.name for p in group.drop] == ["0002.mp4"]


def test_an_omitted_duration_is_the_same_request_as_the_declared_default(tmp_path, monkeypatch):
    """The classifier bug that kept four copies of one clip in the loop.

    The same prompt sits in four backlogs; one writes "seconds": 5 and the
    rest omit it — and 5 IS the default. Hashing them apart made four copies
    of ONE request look like four different requests that had somehow
    produced identical bytes, i.e. misattribution, so they were deleted and
    regenerated on every pass."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"SAME", 100.0)
    _clip(tmp_path / "two" / "clips" / "0002.mp4", b"SAME", 200.0)
    _state(tmp_path, "one", [_item(1, file="0001.mp4", prompt="puppet keynote", seed=820000011)])
    _state(tmp_path, "two", [_item(2, file="0002.mp4", prompt="puppet keynote", seed=820000011)])
    for b in reconcile.load_batches():
        if b.name == "two":
            b.items[0].seconds = reconcile.DEFAULT_SECONDS
            b.save()
    dirs, owners = reconcile.batch_clip_index()
    plan = reconcile.build_plan(reconcile.scan_dirs(dirs), owners)
    assert [(n, i) for n, i, _p, _k in plan.skips] == [("two", 2)], (
        "declared default vs omitted is ONE request: skip, never regenerate"
    )
    assert plan.resets == [], "not a misattribution — nothing to remake"


def test_a_request_whose_output_already_exists_is_skipped_not_regenerated(tmp_path, monkeypatch):
    """THE LOOP, ended. Deleting output can never fix a duplicate REQUEST:
    the prompt is still in a backlog, so the next run remakes it, and the
    next dedupe deletes it again. Only refusing to generate converges."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"CLIP", 100.0)
    _state(tmp_path, "one", [_item(1, file="0001.mp4", prompt="puppet keynote", seed=42)])
    _state(tmp_path, "two", [_item(2, prompt="puppet keynote", seed=42, status="pending")])

    two = next(b for b in reconcile.load_batches() if b.name == "two")
    assert batch_mod.skip_existing(two) == 1
    assert two.items[0].status == batch_mod.SKIPPED
    assert "one[0001]" in str(two.items[0].error)
    assert two.todo() == [], "nothing left to generate"


def test_duplicates_within_one_batch_are_skipped_too(tmp_path, monkeypatch):
    """Duplicates live inside a single backlog as often as across several —
    64 pending items in one batch were twins of clips that same batch had
    already produced. Excluding the whole batch would miss exactly that; only
    the ITEM itself is exempt, so a re-run can still redo its own work."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"CLIP", 100.0)
    _state(
        tmp_path,
        "one",
        [
            _item(1, file="0001.mp4", prompt="puppet keynote", seed=42),
            _item(2, prompt="puppet keynote", seed=42, status="pending"),
        ],
    )
    one = next(b for b in reconcile.load_batches() if b.name == "one")
    assert batch_mod.skip_existing(one) == 1, "the sibling twin is skipped"
    assert one.items[1].status == batch_mod.SKIPPED
    assert one.items[0].status == batch_mod.DONE, "the item that MADE it is untouched"


def test_an_item_is_never_a_duplicate_of_its_own_output(tmp_path, monkeypatch):
    """Re-running a batch whose file vanished must still redo that item —
    matching its own recorded output would deadlock it as SKIPPED forever."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"CLIP", 100.0)
    _state(tmp_path, "one", [_item(1, file="0001.mp4", prompt="puppet keynote", seed=42)])
    one = next(b for b in reconcile.load_batches() if b.name == "one")
    one.items[0].status = batch_mod.PENDING
    assert batch_mod.skip_existing(one) == 0
    assert one.items[0].status == batch_mod.PENDING


def test_fixup_skips_a_duplicate_in_a_live_batch_without_renumbering(tmp_path, monkeypatch):
    """Duplicates are removed from the STATE, never from the prompts file.

    Editing a backlog file renumbers every item after the deleted line, and
    an index carries a job id — so resume would pair recorded jobs with the
    wrong prompts. Every batch here is mid-run, so a tool that can only edit
    prompt files is a tool that can never be used. Marking the ITEM skipped
    leaves indices, job ids and the file untouched, and works on a live
    batch."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _state(
        tmp_path,
        "live",
        [
            _item(1, file="0001.mp4", prompt="keynote", seed=1),
            _item(2, prompt="keynote", seed=1, status="pending"),
            _item(3, prompt="solo", seed=2, status="pending"),
        ],
    )
    summary = reconcile.summarize_census(reconcile.spec_census(reconcile.load_batches()))
    assert reconcile.apply_fixup(summary) == {"live": 1}

    after = batch_mod.Batch.load("live")
    assert [i.index for i in after.items] == [1, 2, 3], "indices are NEVER renumbered"
    assert after.items[1].status == batch_mod.SKIPPED
    assert "live[0001]" in after.items[1].error, "it says which twin made it redundant"
    assert after.items[2].status == batch_mod.PENDING, "unique work is untouched"
    assert [i.index for i in after.todo()] == [3], "only real work remains"


def test_the_spec_census_names_identical_requests_even_without_files(tmp_path, monkeypatch):
    """The census is pure state analysis: it names the duplicates that WILL
    exist once pending items generate, so the operator knows which content
    matches to expect and ignore."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _state(tmp_path, "one", [_item(1, prompt="corgi", seed=7, status="pending")])
    _state(
        tmp_path,
        "two",
        [
            _item(2, prompt="corgi", seed=7, status="done", file="0002.mp4"),
            _item(3, prompt="corgi", status="pending"),  # seedless: own group? no — unique
            _item(4, prompt="solo", seed=1, status="pending"),
        ],
    )
    _state(tmp_path, "three", [_item(5, prompt="corgi", status="pending")])
    census = reconcile.spec_census(reconcile.load_batches())
    assert len(census) == 2
    seeded = next(g for g in census if g.deterministic)
    assert seeded.seed == 7
    assert {(m.batch, m.index) for m in seeded.members} == {("one", 1), ("two", 2)}
    unseeded = next(g for g in census if g.seed is None)
    assert {(m.batch, m.index) for m in unseeded.members} == {("two", 3), ("three", 5)}
    # Seedless is still DETERMINISTIC: the engine defaults the seed, so the
    # same prompt reproduces the same bytes and re-running both is waste.
    assert unseeded.deterministic


def test_the_census_summary_is_a_run_or_never_run_verdict(tmp_path, monkeypatch):
    """The operator question is 'what burns GPU for nothing': pending
    seeded twins of DONE work must never run; all-pending seeded sets need
    exactly one run; seedless repeats are intent, not waste."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _state(
        tmp_path,
        "one",
        [
            _item(1, prompt="corgi", seed=7, status="done", file="0001.mp4"),
            _item(2, prompt="mecha", seed=9, status="pending"),
            _item(3, prompt="solo-free", status="pending"),
        ],
    )
    _state(
        tmp_path,
        "two",
        [
            _item(4, prompt="corgi", seed=7, status="pending"),  # done twin exists
            _item(5, prompt="mecha", seed=9, status="pending"),  # both pending
            _item(6, prompt="solo-free", status="pending"),  # seedless repeat
        ],
    )
    census = reconcile.spec_census(reconcile.load_batches())
    s = reconcile.summarize_census(census)
    assert [(m.batch, m.index) for m in s.already_generated] == [("two", 4)]
    # The mecha pair AND the seedless pair are each one-run-only now: both
    # reproduce identical bytes, so the second member is pure waste.
    assert len(s.redundant) == 2
    assert {m.index for m in s.redundant} <= {2, 3, 5, 6}
    assert s.seedless == [], "no seedless escape hatch: seedless is deterministic"
    assert s.by_batch(s.already_generated) == {"two": [4]}


def test_fixup_skips_duplicates_so_only_unique_work_remains(tmp_path, monkeypatch):
    """The executable verdict: duplicates become SKIPPED in their own state
    files — visible in `batch list`, self-explaining, and never selected by
    a run. What remains pending IS the unique work."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _state(
        tmp_path,
        "one",
        [
            _item(1, prompt="corgi", seed=7, status="done", file="0001.mp4"),
            _item(2, prompt="mecha", seed=9, status="pending"),
        ],
    )
    _state(
        tmp_path,
        "two",
        [
            _item(4, prompt="corgi", seed=7, status="pending"),
            _item(5, prompt="mecha", seed=9, status="pending"),
            _item(6, prompt="unique", seed=1, status="pending"),
            _item(7, prompt="free", status="pending"),
        ],
    )
    _state(tmp_path, "three", [_item(8, prompt="free", status="pending")])
    summary = reconcile.summarize_census(reconcile.spec_census(reconcile.load_batches()))
    skipped = reconcile.apply_fixup(summary)

    # corgi (twin of done), mecha (twin of pending), free (twin of pending
    # in batch three) — seedless is deterministic too, so it counts.
    assert skipped == {"two": 3}
    two = batch_mod.Batch.load("two")
    by_index = {i.index: i for i in two.items}
    assert by_index[4].status == "skipped"
    assert "twin of generated one[0001]" in by_index[4].error
    assert by_index[5].status == "skipped"
    assert "twin of pending" in by_index[5].error
    assert by_index[6].status == "pending", "unique work untouched"
    assert by_index[7].status == "skipped", "a seedless twin reproduces the same bytes"
    assert [i.index for i in two.todo()] == [6], "runs will only see unique work"


def test_dry_run_is_a_plan_not_an_action(tmp_path, monkeypatch):
    """build_plan must not delete or rewrite anything — the operator reads
    the plan BEFORE trusting it with --apply."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    _clip(tmp_path / "one" / "clips" / "0001.mp4", b"SAME", 100.0)
    dup = _clip(tmp_path / "one" / "clips" / "0002.mp4", b"SAME", 200.0)
    _state(
        tmp_path,
        "one",
        [_item(1, file="0001.mp4"), _item(2, file="0002.mp4", job_id="vid_x")],
    )
    dirs, owners = reconcile.batch_clip_index()
    before = (tmp_path / "batches" / "one" / "state.json").read_text()
    plan = reconcile.build_plan(reconcile.scan_dirs(dirs), owners)
    assert plan.drops == [dup]
    assert dup.exists()
    assert (tmp_path / "batches" / "one" / "state.json").read_text() == before
