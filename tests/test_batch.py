"""Working a backlog unattended.

The failure that matters is losing hours of overnight work, so state must
survive a crash, and one bad item must not end the run.
"""

import contextlib
import json
from pathlib import Path

import pytest

from deepme import batch as batch_mod


def write(tmp_path, text, name="p.txt"):
    p = tmp_path / name
    p.write_text(text)
    return p


def test_plain_text_is_one_prompt_per_line(tmp_path):
    items = batch_mod.load_items(write(tmp_path, "a cat\na dog\n"))
    assert [i.prompt for i in items] == ["a cat", "a dog"]


def test_blank_lines_and_comments_are_skipped(tmp_path):
    items = batch_mod.load_items(write(tmp_path, "# note\n\na cat\n\n"))
    assert [i.prompt for i in items] == ["a cat"]


def test_jsonl_carries_per_item_parameters(tmp_path):
    line = json.dumps({"prompt": "a cat", "seconds": 15, "params": {"target.aspect_ratio": "21:9"}})
    (item,) = batch_mod.load_items(write(tmp_path, line, "p.jsonl"))
    assert item.seconds == 15
    assert item.params == {"target.aspect_ratio": "21:9"}


def test_an_empty_backlog_is_rejected_clearly(tmp_path):
    with pytest.raises(SystemExit, match="empty"):
        batch_mod.load_items(write(tmp_path, "   "))


def test_a_file_of_only_comments_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="no prompts"):
        batch_mod.load_items(write(tmp_path, "# just a note\n"))


def _batch(tmp_path, statuses):
    items = [batch_mod.Item(index=i, prompt=f"p{i}", status=s) for i, s in enumerate(statuses)]
    return batch_mod.Batch(name="b", profile="h3", items=items, out_dir=str(tmp_path))


def test_an_interrupted_item_is_retried_not_abandoned(tmp_path):
    """A process killed mid-item leaves it RUNNING; that work never finished."""
    b = _batch(tmp_path, ["done", "running", "pending", "failed"])
    assert [i.index for i in b.todo()] == [1, 2]


def test_state_round_trips_so_an_overnight_run_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["done", "pending"])
    b.items[0].file, b.items[0].peak_mb = "0000.mp4", 22400.0
    b.save()
    again = batch_mod.Batch.load("b")
    assert again is not None
    assert again.counts() == {"pending": 1, "running": 0, "done": 1, "failed": 0}
    assert again.items[0].file == "0000.mp4"
    assert again.items[0].peak_mb == 22400.0


def test_an_unknown_batch_loads_as_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path)
    assert batch_mod.Batch.load("absent") is None


def test_one_failing_item_does_not_stop_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["pending", "pending", "pending"])

    def fake(spec, prompt, **kw):
        if prompt == "p1":
            raise RuntimeError("boom")
        out = Path(kw["out_dir"]) / f"{prompt}.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 1000.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=2)
    assert b.counts()["done"] == 2
    assert b.counts()["failed"] == 1


def test_peak_memory_is_recorded_per_item(tmp_path, monkeypatch):
    """Activations scale with the request, so the envelope is per-item data."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["pending"])

    def fake(spec, prompt, **kw):
        out = Path(kw["out_dir"]) / "x.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 31167.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert b.items[0].peak_mb == 31167.0


@pytest.mark.parametrize(
    "pending,replicas,mean,hours",
    [(100, 4, 900.0, 6.25), (10, 1, 3600.0, 10.0), (0, 4, 900.0, 0.0)],
)
def test_capacity_estimate_answers_the_planning_question(pending, replicas, mean, hours):
    est = batch_mod.estimate(pending, replicas, mean)
    assert est["hours"] == pytest.approx(hours)


def test_no_estimate_without_timing_history():
    assert batch_mod.estimate(100, 4, 0.0)["per_hour"] == 0.0


def test_an_item_is_marked_running_before_it_starts(tmp_path, monkeypatch):
    """State must change when work STARTS, not only when it finishes.

    An item takes ten-plus minutes. A state file that is only written on
    completion reports a fully busy fleet as "pending", so `deepme batch
    status` from a second shell — the whole point of an unattended run — is
    indistinguishable from a batch that is wedged.
    """
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["pending"])
    seen = {}

    def fake(spec, prompt, **kw):
        # Read the state from DISK, as another process would.
        onlooker = batch_mod.Batch.load(b.name)
        seen["counts"] = onlooker.counts()
        seen["started_at"] = onlooker.items[0].started_at
        out = Path(kw["out_dir"]) / f"{prompt}.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 1.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)

    assert seen["counts"]["running"] == 1, "visible as running while it runs"
    assert seen["counts"]["pending"] == 0
    assert seen["started_at"], "and carries a start time, so elapsed is knowable"
    assert b.counts()["done"] == 1


def test_the_job_id_is_persisted_the_moment_the_server_assigns_it(tmp_path, monkeypatch):
    """The reconnection key. The job outlives the runner; without the id on
    DISK, a restarted runner can only resubmit — duplicating work still
    generating and orphaning finished results."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["pending"])

    def fake(spec, prompt, **kw):
        kw["on_submit"]("vid_77")
        # Another process reads the state file NOW, mid-generation:
        onlooker = batch_mod.Batch.load(b.name)
        assert onlooker.items[0].job_id == "vid_77", "already on disk"
        out = Path(kw["out_dir"]) / f"{prompt}.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 1.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert b.items[0].job_id == "vid_77"


def test_resume_adopts_the_recorded_job_instead_of_resubmitting(tmp_path, monkeypatch):
    """An interrupted item with a job_id re-attaches; one without (never
    submitted) starts fresh."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["running", "running"])
    b.items[0].job_id = "vid_5"  # was mid-generation at the kill
    b.items[1].job_id = None  # never reached the server

    seen = {}

    def fake(spec, prompt, **kw):
        seen[prompt] = kw.get("adopt")
        out = Path(kw["out_dir"]) / f"{prompt}.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 1.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert seen["p0"] == "vid_5", "re-attached, not resubmitted"
    assert seen["p1"] is None, "fresh submission for the unsubmitted one"


def test_failed_items_with_a_job_id_are_adopted_on_resume(tmp_path, monkeypatch):
    """Billed work must never be unrecoverable: a download cut mid-stream
    (gateway restarts do this) marked the item FAILED forever while the
    finished file sat on the server."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["failed"])
    b.items[0].job_id = "vid_9"
    b.items[0].error = "content fetch failed (502)"
    seen = {}

    def fake(spec, prompt, **kw):
        seen["adopt"] = kw.get("adopt")
        out = Path(kw["out_dir"]) / f"{prompt}.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 1.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert seen["adopt"] == "vid_9", "re-fetched, not regenerated"
    assert b.items[0].status == "done"


def test_a_done_item_whose_file_vanished_is_not_silently_complete(tmp_path, monkeypatch):
    """state says done, disk says nothing: a hole in a deliverable set the
    operator believes complete. Re-fetch via the surviving id."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["done"])
    b.items[0].file = "0000.mp4"  # never actually written
    b.items[0].job_id = "vid_5"
    seen = {}

    def fake(spec, prompt, **kw):
        seen["adopt"] = kw.get("adopt")
        out = Path(kw["out_dir"]) / f"{prompt}.mp4"
        out.write_bytes(b"x")

        class R:
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 1.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert seen["adopt"] == "vid_5"
    assert b.items[0].status == "done"


def test_the_interrupt_protocol_is_two_stage():
    """First ^C: cooperative (CANCEL set, KeyboardInterrupt raised so the
    command prints its summary). Second ^C: immediate hard exit — the
    atexit thread-join a second interrupt used to detonate never runs."""
    from deepme import interrupts

    exits = []
    orig_exit, orig_hits = interrupts.os._exit, dict(interrupts._hits)
    interrupts.os._exit = lambda code: exits.append(code)
    interrupts._hits["n"] = 0
    interrupts.CANCEL.clear()
    try:
        import signal as _sig

        prev = _sig.getsignal(_sig.SIGINT)
        interrupts.install()
        handler = _sig.getsignal(_sig.SIGINT)
        with contextlib.suppress(KeyboardInterrupt):
            handler(_sig.SIGINT, None)
        assert interrupts.CANCEL.is_set(), "workers start stopping at once"
        assert exits == []
        # The real os._exit never returns; the fake does, so the handler
        # falls through to its first-^C tail — swallow that artifact.
        with contextlib.suppress(KeyboardInterrupt):
            handler(_sig.SIGINT, None)
        assert exits == [130], "second ^C exits immediately, no joins"
        _sig.signal(_sig.SIGINT, prev)
    finally:
        interrupts.os._exit = orig_exit
        interrupts._hits.update(orig_hits)
        interrupts.CANCEL.clear()


def test_every_generation_inherits_the_cancel_signal(monkeypatch, tmp_path):
    """No per-caller wiring: with the process-wide CANCEL set, any
    generate_video — story, showcase, plain generate — stops within one
    poll cycle instead of running to completion."""
    import httpx

    from deepme import interrupts, media

    monkeypatch.setattr(media.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        httpx,
        "post",
        lambda url, **kw: httpx.Response(
            200, request=httpx.Request("POST", url), json={"id": "vid_1", "status": "queued"}
        ),
    )
    interrupts.CANCEL.set()
    try:
        with pytest.raises(InterruptedError):
            media.generate_video(
                type(
                    "S",
                    (),
                    {
                        "profile": "h3",
                        "served_name": "x",
                        "api_base": "http://h",
                        "unit": "u",
                        "min_steps": 2,
                        "gen_defaults": {},
                    },
                )(),
                "x",
                out_dir=tmp_path,
            )
    finally:
        interrupts.CANCEL.clear()
