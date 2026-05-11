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
