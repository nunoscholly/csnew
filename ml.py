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
from sklearn.tree import DecisionTreeClassifier


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
        X: feature matrix to evaluate on (typically the training data
           plus seed rows — fine for a small classroom demo).
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
