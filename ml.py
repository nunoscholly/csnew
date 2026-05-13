# Gap-weighted kNN recommender: for each team, suggest participants whose skills fill the team's skill gaps

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

SKILL_COLUMNS = [
    "strength",
    "driving",
    "design",
    "social",
    "construction",
    "english",
    "german",
    "photography",
    "leadership",
    "communication",
    "experience",
    "problem_solving",
]


def skill_label(skill: str) -> str:
    # Human-readable label, e.g. "problem_solving" -> "Problem Solving"
    return skill.replace("_", " ").title()


def _backfill_missing_skills(df: pd.DataFrame, default: int) -> pd.DataFrame:
    # If a skill column declared in SKILL_COLUMNS is missing from the DataFrame
    # (e.g. its DB migration hasn't been applied yet), inject it with `default`
    # so downstream operations don't KeyError. Use default=5 for team data
    # (treats the skill as fully covered, gap=0) and default=3 for candidates
    # (neutral midpoint). With gap=0 the weight is zero, so the value of the
    # candidate's filled column doesn't affect ranking anyway.
    missing = [c for c in SKILL_COLUMNS if c not in df.columns]
    if not missing:
        return df
    df = df.copy()
    for c in missing:
        df[c] = default
    return df


def team_gap_vector(team_participants: pd.DataFrame) -> np.ndarray:
    # Returns a len(SKILL_COLUMNS)-dim vector of max(5 - team_max_skill, 0);
    # empty team -> all 5s. Skills missing from the DataFrame are treated as
    # already covered (gap=0) so they don't drive recommendations.
    if team_participants.empty:
        return np.full(len(SKILL_COLUMNS), 5.0)
    df = _backfill_missing_skills(team_participants, default=5)
    team_max = df[list(SKILL_COLUMNS)].max(axis=0).to_numpy(dtype=float)
    return np.maximum(5.0 - team_max, 0.0)


def recommend_complementary(
    team_participants: pd.DataFrame,
    candidates_df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    # Rank candidates by gap-weighted Euclidean kNN; covered skills contribute zero to distance
    out_cols = ["id", "name", "distance", "gap_score", *SKILL_COLUMNS]
    if candidates_df.empty:
        return pd.DataFrame(columns=out_cols)

    g = team_gap_vector(team_participants)
    if not np.any(g):
        return pd.DataFrame(columns=out_cols)

    candidates_df = _backfill_missing_skills(candidates_df, default=3)
    skill_matrix = candidates_df[list(SKILL_COLUMNS)].to_numpy(dtype=float)
    target = np.where(g > 0, 5.0, 0.0)  # ideal complement: skill 5 in every gap dim, 0 elsewhere

    # Pre-scale by sqrt(g): plain Euclidean on scaled vectors equals weighted Euclidean with weights g
    weights = np.sqrt(g)
    scaled_matrix = skill_matrix * weights
    scaled_target = target * weights

    n_neighbors = min(k, len(candidates_df))
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(scaled_matrix)
    distances, indices = knn.kneighbors([scaled_target])

    ordered = candidates_df.iloc[indices[0]].copy()
    ordered["distance"] = distances[0]
    ordered["gap_score"] = skill_matrix[indices[0]] @ g
    return ordered[out_cols].reset_index(drop=True)
