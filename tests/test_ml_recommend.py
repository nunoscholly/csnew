"""Tests for ml.recommend_candidates."""
import pandas as pd
import pytest

from ml import SKILL_COLUMNS, recommend_candidates


def _make_team(**reqs):
    """Helper: build a fake team row (Series) with all req_* columns defaulted to 0."""
    base = {f"req_{s}": 0 for s in SKILL_COLUMNS}
    base.update({f"req_{k}": v for k, v in reqs.items()})
    base["id"] = "team-1"
    base["name"] = "Team 1"
    return pd.Series(base)


def _make_candidate(pid, name, **skills):
    """Helper: candidate row dict with all 9 skills defaulted to 3."""
    row = {s: 3 for s in SKILL_COLUMNS}
    row.update(skills)
    row["id"] = pid
    row["name"] = name
    return row


def test_returns_empty_when_no_candidates():
    team = _make_team()
    out = recommend_candidates(team, pd.DataFrame())
    assert out.empty


def test_threshold_filters_out_low_skill_candidates():
    team = _make_team(strength=4)  # need strength >= 4
    candidates = pd.DataFrame([
        _make_candidate("p1", "Weak",   strength=2),
        _make_candidate("p2", "Strong", strength=5),
    ])
    out = recommend_candidates(team, candidates)
    assert list(out["name"]) == ["Strong"]


def test_zero_threshold_lets_everyone_through():
    team = _make_team()  # all reqs = 0
    candidates = pd.DataFrame([
        _make_candidate("p1", "A", strength=1),
        _make_candidate("p2", "B", strength=5),
    ])
    out = recommend_candidates(team, candidates)
    assert set(out["name"]) == {"A", "B"}


def test_ranks_by_distance_to_requirement_vector():
    # Team wants strength=5, leadership=5, others 0 (don't care).
    team = _make_team(strength=5, leadership=5)
    candidates = pd.DataFrame([
        _make_candidate("p1", "Far",   strength=5, leadership=5),  # distance 0 from req
        _make_candidate("p2", "Mid",   strength=4, leadership=4),  # closer to req than p3
        _make_candidate("p3", "Close", strength=5, leadership=5, design=5),
    ])
    out = recommend_candidates(team, candidates)
    # p1 and p3 both match strength/leadership exactly; p1 has design=3, p3 design=5.
    # Requirement design=0, so p1 (design=3) is closer to req than p3 (design=5).
    assert out.iloc[0]["name"] == "Far"
    assert list(out["name"])[:2] == ["Far", "Mid"] or list(out["name"])[:2] == ["Far", "Close"]
    # Distances must be non-decreasing.
    distances = list(out["distance"])
    assert distances == sorted(distances)


def test_caps_results_at_k():
    team = _make_team()
    candidates = pd.DataFrame([
        _make_candidate(f"p{i}", f"P{i}", strength=3) for i in range(10)
    ])
    out = recommend_candidates(team, candidates, k=5)
    assert len(out) == 5


def test_returns_fewer_than_k_when_few_survivors():
    team = _make_team(strength=5)
    candidates = pd.DataFrame([
        _make_candidate("p1", "A", strength=5),
        _make_candidate("p2", "B", strength=5),
        _make_candidate("p3", "C", strength=2),  # filtered out
    ])
    out = recommend_candidates(team, candidates, k=5)
    assert len(out) == 2


def test_output_columns():
    team = _make_team()
    candidates = pd.DataFrame([_make_candidate("p1", "A")])
    out = recommend_candidates(team, candidates)
    expected = {"id", "name", "distance", *SKILL_COLUMNS}
    assert expected.issubset(set(out.columns))
