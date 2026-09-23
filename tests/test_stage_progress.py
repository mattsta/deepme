"""Reading a running stage's progress out of the engine's log.

The video job API reports 0% until it reports 100%, so an eleven-minute
generation has no usable HTTP progress signal. The engine's own tqdm bar is
the only real one, which makes parsing it — and bounding its staleness —
worth testing.
"""

import pytest

from deepme import media, serve

LINE = "minimax_h3 denoise:  96%|#########6| 47/49 [09:02<00:23, 11.75s/it]"


def _probe(monkeypatch, stdout):
    calls = []

    class R:
        pass

    def fake_run(cmd, **kw):
        calls.append(cmd)
        r = R()
        r.stdout = stdout
        return r

    monkeypatch.setattr(serve.remote, "run", fake_run)
    return calls


def test_parses_step_total_and_rate(monkeypatch):
    _probe(monkeypatch, LINE)
    p = serve.stage_progress(_Spec(), 60)
    assert (p.label, p.step, p.total) == ("minimax_h3 denoise", 47, 49)
    assert p.rate_s == pytest.approx(11.75)


def test_eta_uses_remaining_steps(monkeypatch):
    _probe(monkeypatch, LINE)
    p = serve.stage_progress(_Spec(), 60)
    assert p.eta_s == pytest.approx(2 * 11.75)
    assert p.fraction == pytest.approx(47 / 49)


def test_it_per_second_is_inverted_to_seconds_per_step(monkeypatch):
    _probe(monkeypatch, "vae decode:  50%|##   | 5/10 [00:01<00:01, 4.00it/s]")
    assert serve.stage_progress(_Spec(), 60).rate_s == pytest.approx(0.25)


def test_the_last_bar_wins(monkeypatch):
    """Several stages run in sequence; the newest line is the current one."""
    _probe(monkeypatch, LINE + "\n" + "decode:  10%|# | 1/10 [00:01<00:09, 1.00s/it]")
    assert serve.stage_progress(_Spec(), 60).label == "decode"


def test_no_bar_yields_no_progress(monkeypatch):
    _probe(monkeypatch, "nothing interesting here\n")
    assert serve.stage_progress(_Spec(), 60) is None


def test_the_journal_window_is_bounded_by_elapsed_time(monkeypatch):
    """A bar from a PREVIOUS generation must never be shown as this one's."""
    calls = _probe(monkeypatch, LINE)
    serve.stage_progress(_Spec(), 600)
    assert "--since '-605s'" in calls[0]


def test_short_runs_still_get_a_sane_window(monkeypatch):
    calls = _probe(monkeypatch, LINE)
    serve.stage_progress(_Spec(), 1)
    assert "--since '-10s'" in calls[0]


def test_probe_failures_never_break_a_generation(monkeypatch):
    def boom(*a, **kw):
        raise OSError("ssh died")

    monkeypatch.setattr(serve, "stage_progress", boom)
    assert media.probe_stage(_Spec(), 60) is None


class _Spec:
    unit = "deepme-llm-h3-video"
    profile = "h3-video"


STAGE_LINE = "[08-08 20:39] [MiniMaxH3TextEncodingStage] started..."


def test_a_named_stage_is_reported_when_no_bar_is_running(monkeypatch):
    """The job API says "queued" the whole time; the stage name is the truth."""
    _probe(monkeypatch, STAGE_LINE)
    p = serve.stage_progress(_Spec(), 60)
    assert p.label == "MiniMaxH3TextEncodingStage"
    assert not p.counted


def test_a_bar_after_a_stage_line_wins(monkeypatch):
    _probe(monkeypatch, STAGE_LINE + "\n" + LINE)
    p = serve.stage_progress(_Spec(), 60)
    assert p.counted and p.label == "minimax_h3 denoise"


def test_a_stage_line_after_a_bar_wins(monkeypatch):
    """Denoising finished and decoding began — stop showing the stale bar."""
    _probe(monkeypatch, LINE + "\n[08-08 20:56] [MiniMaxH3DecodingStage] started...")
    p = serve.stage_progress(_Spec(), 60)
    assert not p.counted and p.label == "MiniMaxH3DecodingStage"


def test_the_journal_scan_covers_both_patterns(monkeypatch):
    calls = _probe(monkeypatch, "")
    serve.stage_progress(_Spec(), 60)
    assert "Stage\\] started" in calls[0] and "it\\]" in calls[0]
