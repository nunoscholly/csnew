# Unit tests for the supervised skill-imputation classifier (ml.train_skill_classifier, ml.evaluate_all_skills)
import numpy as np
import pandas as pd
import pytest

from ml import (
    HIGH_SKILL_THRESHOLD,
    SKILL_COLUMNS,
    build_skill_imputation_dataset,
    evaluate_all_skills,
    train_skill_classifier,
)


def _make_participants(n: int, seed: int = 0) -> pd.DataFrame:
    # Build a synthetic participant DataFrame with skills correlated so the classifier has signal:
    # leadership tracks social + english; design tracks photography; the rest are noisier.
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        row = {s: int(rng.integers(1, 6)) for s in SKILL_COLUMNS}
        # Inject correlations
        row["leadership"] = int(np.clip(round((row["social"] + row["english"]) / 2 + rng.normal(0, 0.5)), 1, 5))
        row["design"] = int(np.clip(round(row["photography"] * 0.8 + rng.normal(0, 0.5) + 1), 1, 5))
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_dataset_shapes_and_label_encoding():
    df = _make_participants(20)
    X, y, features = build_skill_imputation_dataset(df, "leadership", threshold=4)
    assert X.shape == (20, 8)
    assert y.shape == (20,)
    assert "leadership" not in features
    assert len(features) == 8
    # y should be 0/1 only
    assert set(np.unique(y)).issubset({0, 1})
    # Hand-check: y_i == 1 iff df['leadership'][i] >= 4
    expected = (df["leadership"].to_numpy() >= 4).astype(int)
    assert np.array_equal(y, expected)


def test_build_dataset_unknown_skill_raises():
    df = _make_participants(10)
    with pytest.raises(ValueError):
        build_skill_imputation_dataset(df, "telepathy")


def test_train_classifier_returns_report_with_expected_keys():
    df = _make_participants(60, seed=1)
    result = train_skill_classifier(df, "leadership", threshold=4, k=5)
    assert "error" not in result
    assert result["target_skill"] == "leadership"
    assert result["n_train"] + result["n_test"] == 60
    report = result["report"]
    # classification_report dict form contains the class names plus accuracy + averages
    for key in ("low", "high", "accuracy", "macro avg", "weighted avg"):
        assert key in report
    for metric in ("precision", "recall", "f1-score", "support"):
        assert metric in report["high"]


def test_train_classifier_too_few_rows_returns_error():
    df = _make_participants(5)
    result = train_skill_classifier(df, "leadership")
    assert "error" in result


def test_train_classifier_single_class_returns_error():
    # All participants have leadership = 1 -> only class 0 present -> cannot train
    df = _make_participants(30)
    df["leadership"] = 1
    result = train_skill_classifier(df, "leadership", threshold=4)
    assert "error" in result


def test_evaluate_all_skills_returns_row_per_skill():
    df = _make_participants(80, seed=2)
    summary = evaluate_all_skills(df, threshold=4, k=5)
    assert len(summary) == len(SKILL_COLUMNS)
    assert set(summary["skill"]) == set(SKILL_COLUMNS)
    for col in ("accuracy", "precision_high", "recall_high", "f1_high"):
        assert col in summary.columns


def test_classifier_recovers_signal_on_correlated_skills():
    # leadership is engineered to track (social + english) / 2 with small noise.
    # A kNN classifier on the other 8 skills should beat the naive majority baseline.
    df = _make_participants(200, seed=3)
    result = train_skill_classifier(df, "leadership", threshold=4, k=5, test_size=0.25, random_state=0)
    assert "error" not in result
    y_test = result["y_test"]
    majority = max(np.mean(y_test == 0), np.mean(y_test == 1))
    assert result["report"]["accuracy"] >= majority - 0.05  # at least competitive with majority class


def test_threshold_argument_changes_labels():
    df = _make_participants(40, seed=4)
    _, y_low, _ = build_skill_imputation_dataset(df, "leadership", threshold=2)
    _, y_high, _ = build_skill_imputation_dataset(df, "leadership", threshold=5)
    # Higher threshold -> fewer positives
    assert y_low.sum() >= y_high.sum()


def test_default_threshold_matches_constant():
    df = _make_participants(20, seed=5)
    _, y_default, _ = build_skill_imputation_dataset(df, "leadership")
    _, y_explicit, _ = build_skill_imputation_dataset(df, "leadership", threshold=HIGH_SKILL_THRESHOLD)
    assert np.array_equal(y_default, y_explicit)
