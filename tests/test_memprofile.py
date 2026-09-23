"""Attributing a deployment's host RAM to components.

This decides whether the memory can be shared between replicas at all, so a
silent zero here would send the whole optimisation in the wrong direction.
"""

from deepme.memprofile import Component, ProcMemory, Profile


def comp(name, total, sharded=False, tp=2):
    return Component(
        name=name,
        total_gib=total,
        per_rank_gib=total / tp if sharded else total,
        sharded=sharded,
        dtypes={"BF16": 1},
    )


def proc(label, rss, pss=None, private=None):
    return ProcMemory(
        pid=1,
        label=label,
        rss_gib=rss,
        pss_gib=pss if pss is not None else rss,
        private_gib=private if private is not None else rss,
    )


WORKERS = (proc("sgl_diffusion::", 96.4), proc("sgl_diffusion::", 96.2))


def test_truncated_process_names_are_still_matched():
    """/proc comm caps at 15 chars: "sgl_diffusion::scheduler_TP0" arrives as
    "sgl_diffusion::", so matching the suffix finds nothing and reports 0."""
    p = Profile(components=(), procs=WORKERS, tp=2)
    assert p.measured_per_rank_gib == 96.4


def test_supervisor_processes_do_not_count_as_ranks():
    p = Profile(components=(), procs=(*WORKERS, proc("sglang", 0.9)), tp=2)
    assert p.measured_per_rank_gib == 96.4


def test_a_stopped_deployment_measures_zero_without_crashing():
    assert Profile(components=(), procs=(), tp=2).measured_per_rank_gib == 0.0


def test_only_sharded_components_are_divided_by_tp():
    """The text encoder is replicated per rank; the transformer is split."""
    p = Profile(
        components=(comp("transformer", 61.7, sharded=True), comp("text_encoder", 62.1)),
        procs=WORKERS,
        tp=2,
    )
    assert p.predicted_per_rank_gib == 61.7 / 2 + 62.1


def test_the_unexplained_remainder_is_what_sharing_cannot_fix():
    p = Profile(components=(comp("text_encoder", 90.0),), procs=WORKERS, tp=2)
    assert p.unexplained_gib == 96.4 - 90.0


def test_shared_pages_are_derived_from_rss_minus_private():
    assert proc("x", 96.0, pss=50.0, private=10.0).shared_gib == 86.0


def test_fully_private_memory_reports_no_sharing():
    assert proc("x", 96.4, pss=96.1, private=95.9).shared_gib < 1.0


def test_replica_units_are_not_confused_with_their_prefix(monkeypatch):
    """ "deepme-llm-h3-video" is a prefix of "deepme-llm-h3-video-r1": matching
    on the bare name mixes a replica's processes into its neighbour's."""
    from deepme import memprofile

    captured = {}

    class R:
        stdout = ""

    def fake(script, **kw):
        captured["script"] = script
        return R()

    monkeypatch.setattr(memprofile.remote, "run_script", fake)
    spec = type("S", (), {"unit": "deepme-llm-h3-video"})()
    memprofile._process_memory(spec)
    assert "deepme-llm-h3-video.service" in captured["script"]
