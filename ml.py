"""
ml.py
-----
Team balance classifier for the Event Team Manager app.

The model is intentionally simple: a DecisionTreeClassifier predicts
whether a team is "balanced" or "unbalanced" based on three features
engineered from the participants table.

Features per team:
    - team_size: number of participants on the team
    - num_skills: number of distinct skill values on the team
    - confirmed_ratio: confirmed members / team_size

Label (target):
    - "balanced" if the team has at least 3 members, at least 2 distinct
      skills, and a confirmed_ratio >= 0.5; "unbalanced" otherwise.

A small hardcoded seed dataset is included so the classifier can train
even before any real participants exist in the database.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

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
]


# ---------------------------------------------------------------------------
# Seed dataset
# ---------------------------------------------------------------------------
# Each row is a hypothetical team. We use this to bootstrap training so the
# UI works on day one. Columns: team_size, num_skills, confirmed_ratio, label.
SEED_TEAMS = pd.DataFrame(
    [
        {"team_size": 4, "num_skills": 3, "confirmed_ratio": 1.00, "label": "balanced"},
        {"team_size": 5, "num_skills": 3, "confirmed_ratio": 0.80, "label": "balanced"},
        {"team_size": 3, "num_skills": 2, "confirmed_ratio": 0.67, "label": "balanced"},
        {"team_size": 4, "num_skills": 2, "confirmed_ratio": 0.50, "label": "balanced"},
        {"team_size": 6, "num_skills": 4, "confirmed_ratio": 0.83, "label": "balanced"},
        {"team_size": 5, "num_skills": 3, "confirmed_ratio": 0.60, "label": "balanced"},
        {"team_size": 1, "num_skills": 1, "confirmed_ratio": 1.00, "label": "unbalanced"},
        {"team_size": 2, "num_skills": 1, "confirmed_ratio": 0.50, "label": "unbalanced"},
        {"team_size": 3, "num_skills": 1, "confirmed_ratio": 0.33, "label": "unbalanced"},
        {"team_size": 4, "num_skills": 1, "confirmed_ratio": 0.25, "label": "unbalanced"},
        {"team_size": 5, "num_skills": 2, "confirmed_ratio": 0.20, "label": "unbalanced"},
        {"team_size": 2, "num_skills": 2, "confirmed_ratio": 0.00, "label": "unbalanced"},
        {"team_size": 6, "num_skills": 2, "confirmed_ratio": 0.33, "label": "unbalanced"},
        {"team_size": 3, "num_skills": 3, "confirmed_ratio": 0.33, "label": "unbalanced"},
    ]
)

FEATURE_COLUMNS = ["team_size", "num_skills", "confirmed_ratio"]


def build_training_data(
    participants_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Engineer team-level features from a participants DataFrame.

    Args:
        participants_df: rows of participants. Expected columns are
            "team_id", "skill", and "status". May be empty — in which
            case we fall back to the seed dataset only.

    Returns:
        A tuple (X, y) where:
            X is a DataFrame with FEATURE_COLUMNS, one row per team.
            y is a Series of "balanced" / "unbalanced" labels.

    Notes:
        We always concatenate the seed dataset to whatever real teams
        exist. That way the classifier has enough rows to train on
        even when the database is sparse.
    """
    real_rows = []
    if not participants_df.empty:
        # Group by team and compute the three features.
        grouped = participants_df.groupby("team_id")
        for _, team_df in grouped:
            team_size = len(team_df)
            num_skills = team_df["skill"].dropna().nunique()
            confirmed = (team_df["status"] == "confirmed").sum()
            confirmed_ratio = confirmed / team_size if team_size else 0.0

            # Apply the labelling rule.
            label = (
                "balanced"
                if team_size >= 3 and num_skills >= 2 and confirmed_ratio >= 0.5
                else "unbalanced"
            )
            real_rows.append(
                {
                    "team_size": team_size,
                    "num_skills": num_skills,
                    "confirmed_ratio": confirmed_ratio,
                    "label": label,
                }
            )

    real_df = pd.DataFrame(real_rows)
    combined = pd.concat([SEED_TEAMS, real_df], ignore_index=True)
    return combined[FEATURE_COLUMNS], combined["label"]


def train_model(X: pd.DataFrame, y: pd.Series) -> DecisionTreeClassifier:
    """
    Train a DecisionTreeClassifier on the engineered features.

    Args:
        X: feature matrix (see FEATURE_COLUMNS).
        y: target labels ("balanced" / "unbalanced").

    Returns:
        A fitted DecisionTreeClassifier.
    """
    # max_depth keeps the tree small and easy to reason about — good
    # for a beginner-friendly demo and helps avoid overfitting on the
    # tiny seed dataset.
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)
    return model


def train_model_with_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series]:
    """
    Split data, train a DecisionTreeClassifier, and return the test split.

    Args:
        X: feature matrix (see FEATURE_COLUMNS).
        y: target labels ("balanced" / "unbalanced").

    Returns:
        (model, X_test, y_test) — fitted model plus the held-out test split.
    """
    # Course requirement: evaluate on unseen data, not training data, so the
    # reported metrics reflect generalisation rather than memorisation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    model = train_model(X_train, y_train)
    return model, X_test, y_test


def predict_team_balance(
    model: DecisionTreeClassifier,
    team_features: dict,
) -> str:
    """
    Predict whether a single team is "balanced" or "unbalanced".

    Args:
        model: a fitted DecisionTreeClassifier.
        team_features: dict with keys team_size, num_skills, confirmed_ratio.

    Returns:
        The predicted label as a string.
    """
    # Wrap the dict in a one-row DataFrame so column order matches training.
    row = pd.DataFrame([team_features])[FEATURE_COLUMNS]
    return str(model.predict(row)[0])


def evaluation_report(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> str:
    """
    Produce a sklearn classification report as a plain text string.

    Args:
        model: a fitted DecisionTreeClassifier.
        X: feature matrix to evaluate on (use the held-out test split).
        y: true labels matching X.

    Returns:
        A multi-line string with precision, recall, f1-score per class.
    """
    predictions = model.predict(X)
    return classification_report(y, predictions, zero_division=0)


def features_for_team(team_participants: pd.DataFrame) -> dict:
    """
    Compute the three balance features for one team's participants.

    Args:
        team_participants: rows of participants belonging to a single team.

    Returns:
        A dict with keys team_size, num_skills, confirmed_ratio.
    """
    team_size = len(team_participants)
    if team_size == 0:
        return {"team_size": 0, "num_skills": 0, "confirmed_ratio": 0.0}

    num_skills = team_participants["skill"].dropna().nunique()
    confirmed = (team_participants["status"] == "confirmed").sum()
    return {
        "team_size": int(team_size),
        "num_skills": int(num_skills),
        "confirmed_ratio": float(confirmed / team_size),
    }


def recommend_candidates(
    team_row: pd.Series,
    candidates_df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """
    Recommend up to k candidates whose skills best match a team's requirements.

    Steps:
      1. Filter: keep candidates where every skill >= the team's matching
         req_<skill> threshold. Thresholds of 0 are no-ops.
      2. Rank: fit sklearn's NearestNeighbors on the survivors' 9-D skill
         vectors and query with the team's 9-D requirement vector. Lower
         distance = closer to the team's stated requirements.

    Args:
        team_row: a pandas Series with req_<skill> columns for each of
            the 9 skills (and any other team columns; we ignore them).
        candidates_df: a DataFrame of candidate participants. Must contain
            the 9 skill columns plus 'id' and 'name'.
        k: max number of recommendations to return.

    Returns:
        A DataFrame with columns id, name, distance, and the 9 skill
        columns, sorted ascending by distance. Empty if no candidates
        survive the threshold filter (or candidates_df is empty).
    """
    if candidates_df.empty:
        return pd.DataFrame(columns=["id", "name", "distance", *SKILL_COLUMNS])

    # 1. Threshold filter: every skill must meet team's req_<skill>.
    mask = pd.Series(True, index=candidates_df.index)
    for skill in SKILL_COLUMNS:
        threshold = int(team_row.get(f"req_{skill}", 0) or 0)
        if threshold > 0:
            mask &= candidates_df[skill] >= threshold
    survivors = candidates_df[mask].copy()

    if survivors.empty:
        return pd.DataFrame(columns=["id", "name", "distance", *SKILL_COLUMNS])

    # 2. kNN rank against the team's requirement vector.
    n = len(survivors)
    n_neighbors = min(k, n)
    skill_matrix = survivors[SKILL_COLUMNS].to_numpy()
    requirement_vector = [
        int(team_row.get(f"req_{s}", 0) or 0) for s in SKILL_COLUMNS
    ]

    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    knn.fit(skill_matrix)
    distances, indices = knn.kneighbors([requirement_vector])

    ordered = survivors.iloc[indices[0]].copy()
    ordered["distance"] = distances[0]
    cols = ["id", "name", "distance", *SKILL_COLUMNS]
    return ordered[cols].reset_index(drop=True)
