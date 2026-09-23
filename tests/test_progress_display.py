"""Rendering progress for a job that runs for ten minutes.

These run offline on purpose: the first version of this display crashed on
its very first tick (formatting a None total), and finding that out cost a
full generation. Formatting is a pure function so it can be tested without a
terminal; the context manager is smoke-tested for the same crash.
"""

import pytest
from rich.console import Console

from deepme import cli
from deepme.media import Tick
from deepme.serve import StageProgress

STAGE = StageProgress(label="minimax_h3 denoise", step=47, total=49, rate_s=11.75)
QUEUED = Tick(elapsed_s=3.0, status="queued", stage=None)
RUNNING = Tick(elapsed_s=545.0, status="running", stage=STAGE)


def test_before_any_stage_the_total_is_unknown_not_zero():
    """A 0 total would render a full bar; None renders an indeterminate one."""
    f = cli._tick_fields(QUEUED)
    assert f["total"] is None
    assert f["steps"] == ""
    assert f["description"] == "starting"
    assert "0:03 elapsed" in f["note"]


def test_a_running_stage_reports_steps_rate_and_eta():
    f = cli._tick_fields(RUNNING)
    assert f["description"] == "minimax_h3 denoise"
    assert (f["steps"], f["completed"], f["total"]) == ("47/49", 47, 49)
    assert "0:23 left" in f["note"]
    assert "11.8s/step" in f["note"]


def _drive(console, ticks):
    with cli._generation_progress(console) as tick_cb:
        for t in ticks:
            tick_cb(t)


def test_driving_the_live_display_never_raises():
    """Covers the None-total crash and a stage vanishing between stages."""
    con = Console(force_terminal=True, force_interactive=False, width=200)
    _drive(con, [QUEUED, RUNNING, Tick(elapsed_s=600.0, status="running", stage=None)])


def test_piped_output_prints_timestamped_lines_not_a_bar():
    con = Console(force_terminal=False, width=200, record=True)
    _drive(con, [QUEUED, Tick(elapsed_s=45.0, status="running", stage=STAGE)])
    out = con.export_text()
    assert "minimax_h3 denoise 47/49" in out
    assert "0:45 elapsed" in out


def test_piped_output_stays_quiet_between_intervals():
    con = Console(force_terminal=False, width=200, record=True)
    _drive(con, [Tick(elapsed_s=float(i), status="running", stage=STAGE) for i in range(20)])
    assert con.export_text().count("elapsed") <= 1


@pytest.mark.parametrize(
    "secs,text",
    [(0, "0:00"), (23, "0:23"), (90, "1:30"), (659, "10:59"), (3661, "1:01:01")],
)
def test_durations_read_as_clock_time(secs, text):
    assert cli._hms(secs) == text


def test_negative_durations_do_not_render_garbage():
    assert cli._hms(-5) == "0:00"


UNCOUNTED = Tick(
    elapsed_s=42.0,
    status="queued",
    stage=StageProgress(label="MiniMaxH3TextEncodingStage", step=0, total=0, rate_s=0.0),
)


def test_the_servers_queued_status_is_never_shown():
    """It reads as "waiting behind other work" when nothing is waiting: the
    API has no running state, only queued -> completed."""
    assert cli._tick_fields(QUEUED)["description"] == "starting"
    assert "queued" not in str(cli._tick_fields(UNCOUNTED))


def test_an_uncounted_stage_shows_its_name_and_the_clock():
    f = cli._tick_fields(UNCOUNTED)
    assert f["description"] == "MiniMaxH3TextEncodingStage"
    assert f["steps"] == "" and f["total"] is None
    assert "0:42 elapsed" in f["note"]


def test_a_waiting_ticket_shows_its_queue_position_not_starting():
    """ "starting" for an item 3-deep in the queue hid waiting-vs-wedged —
    the position is the one honest pre-stage signal, shared by batch and
    single-generate through this one formatter."""
    from deepme.media import Tick

    f = cli._tick_fields(Tick(elapsed_s=42.0, status="queued", queue_position=3, job_id="q_x"))
    assert f["description"] == "queued #3"
    assert "0:42" in f["note"]


def test_an_unqueued_prestage_tick_still_reads_starting():
    from deepme.media import Tick

    f = cli._tick_fields(Tick(elapsed_s=5.0, status="queued"))
    assert f["description"] == "starting"
