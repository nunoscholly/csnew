# ML module:
#   1. Gap-weighted kNN recommender (unsupervised): suggests participants who fill a team's skill gaps.
#   2. Skill-imputation kNN classifier (supervised): predicts whether a participant is "high" (>= threshold)
#      on a target skill from the other 8 skill ratings, with train/test split and classification_report.

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors

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

HIGH_SKILL_THRESHOLD = 4


def team_gap_vector(team_participants: pd.DataFrame) -> np.ndarray:
    # Returns 9-dim vector of max(5 - team_max_skill, 0) per skill; empty team -> all 5s
    if team_participants.empty:
        return np.full(len(SKILL_COLUMNS), 5.0)
    team_max = team_participants[list(SKILL_COLUMNS)].max(axis=0).to_numpy(dtype=float)
    gap = np.maximum(5.0 - team_max, 0.0)
    return gap


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


# ---------------------------------------------------------------------------
# Supervised: skill-imputation kNN classifier
# ---------------------------------------------------------------------------
# Problem: given a participant's ratings on 8 skills, predict whether their 9th
# skill is "high" (>= HIGH_SKILL_THRESHOLD). Labels are real values pulled from
# the database, so this is genuine supervised learning, not a heuristic loopback.

def build_skill_imputation_dataset(
    participants_df: pd.DataFrame,
    target_skill: str,
    threshold: int = HIGH_SKILL_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # X = the other 8 skill ratings; y = 1 if target_skill >= threshold else 0
    if target_skill not in SKILL_COLUMNS:
        raise ValueError(f"Unknown skill: {target_skill}")
    feature_cols = [s for s in SKILL_COLUMNS if s != target_skill]
    X = participants_df[feature_cols].to_numpy(dtype=float)
    y = (participants_df[target_skill].to_numpy(dtype=float) >= threshold).astype(int)
    return X, y, feature_cols


def train_skill_classifier(
    participants_df: pd.DataFrame,
    target_skill: str,
    threshold: int = HIGH_SKILL_THRESHOLD,
    k: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    # Train a kNN classifier predicting whether target_skill is >= threshold from the other 8 skills.
    # Returns a dict with the fitted model, classification_report (dict form), and split metadata.
    X, y, feature_cols = build_skill_imputation_dataset(participants_df, target_skill, threshold)

    if len(X) < 10:
        return {"target_skill": target_skill, "error": "need at least 10 participants to train"}
    if len(np.unique(y)) < 2:
        return {"target_skill": target_skill, "error": "only one class present in labels"}

    # Stratify only if every class has at least 2 examples (otherwise train_test_split errors out)
    stratify = y if np.min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    n_neighbors = min(k, len(X_train))
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["low", "high"],
        output_dict=True, zero_division=0,
    )
    return {
        "target_skill": target_skill,
        "feature_cols": feature_cols,
        "threshold": threshold,
        "k": n_neighbors,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "classifier": clf,
        "y_test": y_test,
        "y_pred": y_pred,
        "report": report,
    }


def evaluate_all_skills(
    participants_df: pd.DataFrame,
    threshold: int = HIGH_SKILL_THRESHOLD,
    k: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    # Run the imputation classifier for each of the 9 skills; return a summary DataFrame.
    rows = []
    for skill in SKILL_COLUMNS:
        result = train_skill_classifier(
            participants_df, skill, threshold=threshold, k=k,
            test_size=test_size, random_state=random_state,
        )
        if "error" in result:
            rows.append({
                "skill": skill, "accuracy": np.nan,
                "precision_high": np.nan, "recall_high": np.nan, "f1_high": np.nan,
                "support_high": 0, "n_train": 0, "n_test": 0, "note": result["error"],
            })
            continue
        r = result["report"]
        rows.append({
            "skill": skill,
            "accuracy": r["accuracy"],
            "precision_high": r["high"]["precision"],
            "recall_high": r["high"]["recall"],
            "f1_high": r["high"]["f1-score"],
            "support_high": int(r["high"]["support"]),
            "n_train": result["n_train"],
            "n_test": result["n_test"],
            "note": "",
        })
    return pd.DataFrame(rows)
