# Unit tests for the gap-weighted kNN recommender (ml.team_gap_vector, ml.recommend_complementary)
import numpy as np
import pandas as pd

from ml import SKILL_COLUMNS, team_gap_vector


def _make_member(**skills):
    # Build fake participant row; defaults each skill to 3
    row = {s: 3 for s in SKILL_COLUMNS}
    row.update(skills)
    return row


def test_gap_vector_empty_team_is_all_fives():
    g = team_gap_vector(pd.DataFrame(columns=SKILL_COLUMNS))
    assert g.shape == (len(SKILL_COLUMNS),)
    assert np.all(g == 5.0)


def test_gap_vector_uses_max_per_skill():
    members = pd.DataFrame([
        _make_member(strength=5, leadership=2),
        _make_member(strength=3, leadership=4),
    ])
    g = team_gap_vector(members)
    # strength max=5 -> gap 0; leadership max=4 -> gap 1; others default max=3 -> gap 2
    assert g[SKILL_COLUMNS.index("strength")] == 0
    assert g[SKILL_COLUMNS.index("leadership")] == 1
    assert g[SKILL_COLUMNS.index("design")] == 2


def test_gap_vector_fully_covered_team_is_all_zeros():
    members = pd.DataFrame([_make_member(**{s: 5 for s in SKILL_COLUMNS})])
    g = team_gap_vector(members)
    assert np.all(g == 0.0)


def test_gap_vector_clamps_above_max_rating():
    # Skills above 5 (e.g. admin entered 6) must not produce negative gaps
    members = pd.DataFrame([_make_member(**{s: 6 for s in SKILL_COLUMNS})])
    g = team_gap_vector(members)
    assert np.all(g == 0.0)


from ml import recommend_complementary


def _make_candidate(pid, name, **skills):
    # Build candidate row; defaults each skill to 3, plus id/name
    row = {s: 3 for s in SKILL_COLUMNS}
    row.update(skills)
    row["id"] = pid
    row["name"] = name
    return row


def test_recommend_empty_candidates_returns_empty_with_columns():
    members = pd.DataFrame([_make_member(strength=2)])
    out = recommend_complementary(members, pd.DataFrame())
    assert out.empty
    expected_cols = {"id", "name", "distance", "gap_score", *SKILL_COLUMNS}
    assert expected_cols.issubset(set(out.columns))


def test_recommend_fully_covered_team_returns_empty():
    # Team has max=5 in every skill -> no gaps -> nothing to complement
    members = pd.DataFrame([_make_member(**{s: 5 for s in SKILL_COLUMNS})])
    candidates = pd.DataFrame([_make_candidate("p1", "Alice")])
    out = recommend_complementary(members, candidates)
    assert out.empty


def test_recommend_ranks_gap_filling_candidate_first():
    # Team is weak in 'construction' (max=1). Other skills covered (max=5).
    all_max = {s: 5 for s in SKILL_COLUMNS}
    all_max["construction"] = 1
    members = pd.DataFrame([
        _make_member(**all_max),
    ])
    candidates = pd.DataFrame([
        _make_candidate("p1", "WeakConstruction", construction=1),
        _make_candidate("p2", "StrongConstruction", construction=5),
    ])
    out = recommend_complementary(members, candidates, k=2)
    assert list(out["name"]) == ["StrongConstruction", "WeakConstruction"]


def test_recommend_ignores_covered_skills_in_ranking():
    # Team weak in 'design' only. Candidate A strong in design but weak in
    # covered skills; Candidate B weak in design but strong elsewhere.
    # A must rank above B because covered skills have weight zero.
    all_max = {s: 5 for s in SKILL_COLUMNS}
    all_max["design"] = 1
    members = pd.DataFrame([
        _make_member(**all_max),
    ])
    candidates = pd.DataFrame([
        _make_candidate("p1", "FillsGap",   design=5, strength=1, leadership=1),
        _make_candidate("p2", "ShinyButOff", design=1, strength=5, leadership=5),
    ])
    out = recommend_complementary(members, candidates, k=2)
    assert list(out["name"]) == ["FillsGap", "ShinyButOff"]


def test_recommend_caps_results_at_k():
    members = pd.DataFrame([_make_member(strength=1)])
    candidates = pd.DataFrame([
        _make_candidate(f"p{i}", f"P{i}", strength=3) for i in range(10)
    ])
    out = recommend_complementary(members, candidates, k=5)
    assert len(out) == 5


def test_recommend_returns_fewer_than_k_when_few_candidates():
    members = pd.DataFrame([_make_member(strength=1)])
    candidates = pd.DataFrame([
        _make_candidate("p1", "A", strength=5),
        _make_candidate("p2", "B", strength=4),
    ])
    out = recommend_complementary(members, candidates, k=5)
    assert len(out) == 2


def test_recommend_gap_score_is_dot_product_with_gap():
    # Team weak only in 'construction' (gap=4). Candidate has construction=5.
    # gap_score = candidate_skills . gap = 5 * 4 = 20 (other terms zero).
    all_max = {s: 5 for s in SKILL_COLUMNS}
    all_max["construction"] = 1
    members = pd.DataFrame([
        _make_member(**all_max),
    ])
    candidates = pd.DataFrame([
        _make_candidate("p1", "Builder", construction=5, strength=2),
    ])
    out = recommend_complementary(members, candidates, k=1)
    assert out.iloc[0]["gap_score"] == 20.0
