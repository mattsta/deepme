"""Working a backlog unattended.

The failure that matters is losing hours of overnight work, so state must
survive a crash, and one bad item must not end the run.
"""

import json
from pathlib import Path
from typing import ClassVar

import pytest

from deepme import batch as batch_mod
from deepme import media as media_mod


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
    assert again.counts() == {"pending": 1, "running": 0, "done": 1, "failed": 0, "skipped": 0}
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
            media_id = "vid_test"
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
            media_id = "vid_test"
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
            media_id = "vid_test"
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
            media_id = "vid_test"
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
            media_id = "vid_test"
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
            media_id = "vid_test"
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
            media_id = "vid_test"
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, 1.0, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert seen["adopt"] == "vid_5"
    assert b.items[0].status == "done"


def test_the_interrupt_protocol_is_two_stage():
    """First ^C: cooperative (CANCEL set, KeyboardInterrupt raised so the
    command prints its summary). Second ^C: escalated but still CLEAN —
    SystemExit(130) through finally blocks, atexit and bounded joins. No
    hard exit anywhere: workers watch CANCEL, so teardown is seconds."""
    from deepme import interrupts

    orig_hits = dict(interrupts._hits)
    interrupts._hits["n"] = 0
    interrupts.CANCEL.clear()
    try:
        import signal as _sig

        prev = _sig.getsignal(_sig.SIGINT)
        interrupts.install()
        handler = _sig.getsignal(_sig.SIGINT)
        with pytest.raises(KeyboardInterrupt):
            handler(_sig.SIGINT, None)
        assert interrupts.CANCEL.is_set(), "workers start stopping at once"
        with pytest.raises(SystemExit) as exc:
            handler(_sig.SIGINT, None)
        assert exc.value.code == 130, "second ^C: clean escalation, not os._exit"
        _sig.signal(_sig.SIGINT, prev)
    finally:
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


# --------------------------------------------------------------------------
# fetch-only: wind a batch down without ever creating new server work


def test_fetch_only_adopts_owned_jobs_and_submits_nothing(tmp_path, monkeypatch):
    """--fetch-existing harvests the in-flight tail: items with a server job
    are adopted and downloaded; pending items stay parked exactly as a later
    submitting run expects them."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["pending", "running", "pending"])
    b.items[1].job_id = "vid_live"
    calls = []

    def fake(spec, prompt, **kw):
        calls.append(kw)
        out = Path(kw["out_dir"]) / "x.mp4"
        out.write_bytes(b"x")

        class R:
            media_id = "vid_test"
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, None, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=4, fetch_only=True)
    assert len(calls) == 1, "only the item that owns a job is worked"
    assert calls[0]["adopt"] == "vid_live"
    assert calls[0]["resubmit"] is False, "a lost job must not be redone"
    assert b.counts() == {"pending": 2, "running": 0, "done": 1, "failed": 0, "skipped": 0}


def test_only_a_delivery_failure_is_readopted(tmp_path, monkeypatch):
    """A cut DOWNLOAD over billed work is re-fetched — the artifact exists
    server-side. A GENERATION failure is not: re-adopting it re-reads the
    failure, or once the platform's id map has drifted, downloads someone
    else's video. Promoting every FAILED item with an id is how old content
    kept being re-delivered run after run."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["failed", "failed"])
    b.items[0].job_id = "vid_billed"
    b.items[0].error = {"message": "content fetch failed (502): gateway restarted"}
    b.items[1].job_id = "vid_broken"
    b.items[1].error = {"message": "short_edge must be exactly 768"}
    fetched = []

    def fake(spec, prompt, **kw):
        fetched.append(kw["adopt"])
        out = Path(kw["out_dir"]) / "y.mp4"
        out.write_bytes(b"x")

        class R:
            media_id = "vid_test"
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, None, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1, fetch_only=True)
    assert fetched == ["vid_billed"], "only the delivery failure is re-adopted"
    assert b.items[0].status == "done"
    assert b.items[1].status == "failed", "a generation failure is never re-adopted"


def test_a_run_that_may_generate_retries_its_failures(tmp_path, monkeypatch):
    """`todo()` returns only PENDING/RUNNING, so a FAILED item was INERT —
    nothing ever picked it up again — while the CLI told the operator to
    "retry them" by re-running, advice that quietly did nothing. Items failed
    by a collect-only pass ("no longer exists on the server and resubmission
    is disabled") are the common case and are perfectly generatable: they
    were simply work that a --fetch-existing run is forbidden to create."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["failed"])
    b.items[0].job_id = "vid_gone"
    b.items[0].error = "vid_gone no longer exists on the server and resubmission is disabled"
    seen = []

    def fake(spec, prompt, **kw):
        seen.append(kw.get("adopt"))
        out = Path(kw["out_dir"]) / "y.mp4"
        out.write_bytes(b"x")

        class R:
            media_id = "vid_new"
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, None, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert seen == [None], "REGENERATED from the prompt, never re-adopted"
    assert b.items[0].status == "done"


def test_a_collect_only_run_never_revives_a_failure(tmp_path, monkeypatch):
    """--fetch-existing may not create work, so it must not turn a failed
    item back into something submittable."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["failed"])
    b.items[0].error = "no longer exists on the server"

    def must_not_run(spec, prompt, **kw):
        raise AssertionError("collect-only must never generate")

    monkeypatch.setattr(batch_mod.media, "generate_video", must_not_run)
    batch_mod.run(object(), b, concurrency=1, fetch_only=True)
    assert b.items[0].status == "failed"


def test_the_journal_records_every_identity_and_the_content_hash(tmp_path, monkeypatch):
    """state.json holds only the LAST job id; the journal is append-only —
    every id an item ever held plus the MD5 of what was delivered, so a
    duplicate hunt is a log query instead of content archaeology."""
    from deepme import digests

    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    monkeypatch.setattr(digests, "CACHE_PATH", tmp_path / "digest-cache.json")
    b = _batch(tmp_path, ["pending"])

    def fake(spec, prompt, **kw):
        kw["on_submit"]("vid_first")
        kw["on_submit"]("vid_second")  # e.g. a harvest after handover
        out = Path(kw["out_dir"]) / "x.mp4"
        out.write_bytes(b"BYTES")

        class R:
            media_id = "vid_test"
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, None, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    lines = [
        json.loads(x)
        for x in (batch_mod.BATCH_DIR / "b" / "journal.jsonl").read_text().splitlines()
    ]
    kinds = [e["event"] for e in lines]
    assert kinds == ["started", "job-id", "job-id", "done"]
    assert [e.get("job_id") for e in lines if e["event"] == "job-id"] == [
        "vid_first",
        "vid_second",
    ], "BOTH identities kept — state.json only remembers the last"
    done = lines[-1]
    # The slot name carries the job id, so two batches sharing one --out
    # can no longer silently overwrite each other's clips.
    assert done["file"] == "0000-vid_test.mp4"
    import hashlib

    assert done["md5"] == hashlib.md5(b"BYTES").hexdigest()


def test_fetch_only_marks_a_lost_job_failed_without_resubmitting(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["running", "pending"])
    b.items[0].job_id = "vid_gone"

    def fake(spec, prompt, **kw):
        assert kw["resubmit"] is False
        raise media_mod.TicketLostError(f"{kw['adopt']} no longer exists on the server")

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1, fetch_only=True)
    assert b.items[0].status == "failed"
    assert "no longer exists" in b.items[0].error
    assert b.items[1].status == "pending", "the backlog stays parked"


# --------------------------------------------------------------------------
# Transient failures: retry load-dependent errors, never request errors


def _oom(msg="CUDA out of memory. Tried to allocate 2.92 GiB"):
    return {"message": f"Model generation returned no output. Error from scheduler: {msg}"}


def test_a_cuda_oom_is_retried_with_a_fresh_identity(tmp_path, monkeypatch):
    """A 15s clip OOM'd at allocation while four others ran — nothing about
    the REQUEST was wrong, it needed a less-loaded replica. Recording that
    as permanent discarded work the platform could finish."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    monkeypatch.setattr(batch_mod, "TRANSIENT_BACKOFF_S", 0.0)
    b = _batch(tmp_path, ["pending"])
    attempts = []

    def fake(spec, prompt, **kw):
        attempts.append(kw.get("adopt"))
        if len(attempts) == 1:

            class Bad:
                media_id = "vid_test"
                ok, path, total_s, peak_memory_mb = False, None, 0.1, None
                error = _oom()

            return Bad()
        out = Path(kw["out_dir"]) / "x.mp4"
        out.write_bytes(b"x")

        class R:
            media_id = "vid_test"
            ok, path, total_s, peak_memory_mb, error = True, out, 5.0, None, None

        return R()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert len(attempts) == 2, "retried once"
    assert attempts[1] is None, "a fresh attempt must not adopt the failed job"
    assert b.items[0].status == "done"


def test_a_request_error_is_never_retried(tmp_path, monkeypatch):
    """Every attempt costs a slot and fails identically — retrying a
    validation error burns the fleet for nothing."""
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    monkeypatch.setattr(batch_mod, "TRANSIENT_BACKOFF_S", 0.0)
    b = _batch(tmp_path, ["pending"])
    calls = []

    def fake(spec, prompt, **kw):
        calls.append(1)

        class Bad:
            media_id = "vid_test"
            ok, path, total_s, peak_memory_mb = False, None, 0.1, None
            error: ClassVar[dict] = {"message": "short_edge must be exactly 768"}

        return Bad()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert len(calls) == 1
    assert b.items[0].status == "failed"


def test_retries_are_bounded_and_journaled(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    monkeypatch.setattr(batch_mod, "TRANSIENT_BACKOFF_S", 0.0)
    b = _batch(tmp_path, ["pending"])
    calls = []

    def fake(spec, prompt, **kw):
        calls.append(1)

        class Bad:
            media_id = "vid_test"
            ok, path, total_s, peak_memory_mb = False, None, 0.1, None
            error = _oom()

        return Bad()

    monkeypatch.setattr(batch_mod.media, "generate_video", fake)
    batch_mod.run(object(), b, concurrency=1)
    assert len(calls) == batch_mod.TRANSIENT_RETRIES + 1, "bounded"
    assert b.items[0].status == "failed"
    lines = [
        json.loads(x)
        for x in (batch_mod.BATCH_DIR / "b" / "journal.jsonl").read_text().splitlines()
    ]
    assert [e["event"] for e in lines].count("retry") == batch_mod.TRANSIENT_RETRIES


def test_transient_classification_reads_both_error_shapes():
    assert batch_mod.media.is_transient(_oom())
    assert batch_mod.media.is_transient("CUDA out of memory")
    assert not batch_mod.media.is_transient({"message": "invalid aspect_ratio"})
    assert not batch_mod.media.is_transient(None)


def test_journal_lines_never_interleave_under_threads(tmp_path, monkeypatch):
    """The forensic trail must survive concurrent writers with records
    LARGER than a text buffer — one os.write to an O_APPEND fd is atomic
    regardless of size, a buffered write is not."""
    import concurrent.futures as futures

    monkeypatch.setattr(batch_mod, "BATCH_DIR", tmp_path / "batches")
    b = _batch(tmp_path, ["pending"])
    huge = "x" * 40000  # far past any text-buffer boundary

    def writer(n):
        batch_mod.journal(b, "stress", item=n, blob=huge)

    with futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(writer, range(40)))

    text = (batch_mod.BATCH_DIR / "b" / "journal.jsonl").read_text()
    lines = text.splitlines()
    assert len(lines) == 40
    for line in lines:
        parsed = json.loads(line)  # every line whole and parseable
        assert parsed["blob"] == huge
