"""Catalog search/filter predicates (pure — no network)."""

import pytest

from deepme import catalog


def m(repo: str, downloads: int = 0, likes: int = 0) -> dict:
    return {"id": repo, "downloads": downloads, "likes": likes}


def test_name_contains_matches_case_insensitively():
    assert catalog.prefilter(m("MiniMaxAI/MiniMax-H3"), name_contains="h3")
    assert catalog.prefilter(m("MiniMaxAI/MiniMax-H3"), name_contains="MiniMax")
    assert not catalog.prefilter(m("Qwen/Qwen3-8B"), name_contains="h3")


def test_name_contains_matches_org_too():
    assert catalog.prefilter(m("unsloth/Model-X"), name_contains="unsloth")


def test_no_filters_passes_everything():
    assert catalog.prefilter(m("any/thing"))


def test_min_downloads_and_likes():
    entry = m("a/b", downloads=100, likes=5)
    assert catalog.prefilter(entry, min_downloads=100)
    assert not catalog.prefilter(entry, min_downloads=101)
    assert catalog.prefilter(entry, min_likes=5)
    assert not catalog.prefilter(entry, min_likes=6)


def test_missing_counts_treated_as_zero():
    assert not catalog.prefilter({"id": "a/b"}, min_downloads=1)


def test_unknown_sort_is_rejected_with_options():
    with pytest.raises(SystemExit) as e:
        catalog.fetch_top(sort="popularity")
    assert "trending" in str(e.value)


def test_known_sorts_map_to_api_fields():
    assert catalog.SORTS["trending"] == "trendingScore"
    assert catalog.SORTS["downloads"] == "downloads"
    assert set(catalog.SORTS) == {"trending", "downloads", "likes", "modified"}
