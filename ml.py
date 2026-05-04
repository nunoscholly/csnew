# Team balance classifier: DecisionTree model predicts "balanced"/"unbalanced" from team size, skill diversity, and confirmation ratio

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
    # Engineer team features and apply balance label; combine with seed dataset for initial training
    real_rows = []
    if not participants_df.empty:
        # Extract team size, skill diversity, and confirmation ratio
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
    # Train shallow DecisionTree to balance interpretability and generalization on small seed dataset
    # Limit depth to avoid overfitting and keep model understandable
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X, y)
    return model


def train_model_with_split(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[DecisionTreeClassifier, pd.DataFrame, pd.Series]:
    # Train on split data and return model + held-out test set for evaluation of generalization
    # Evaluate on unseen data, not training data, to reflect real-world performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    model = train_model(X_train, y_train)
    return model, X_test, y_test


def predict_team_balance(
    model: DecisionTreeClassifier,
    team_features: dict,
) -> str:
    # Predict team balance label for a single team
    # Wrap dict in DataFrame to enforce FEATURE_COLUMNS order
    row = pd.DataFrame([team_features])[FEATURE_COLUMNS]
    return str(model.predict(row)[0])


def evaluation_report(
    model: DecisionTreeClassifier,
    X: pd.DataFrame,
    y: pd.Series,
) -> str:
    # Generate sklearn classification report on test split
    predictions = model.predict(X)
    return classification_report(y, predictions, zero_division=0)


def features_for_team(team_participants: pd.DataFrame) -> dict:
    # Extract team size, skill diversity, and confirmation ratio from participants
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
    # Recommend k candidates by filtering thresholds then ranking via kNN on skill vectors
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

    # Rank survivors by Euclidean distance to team's requirement vector
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
