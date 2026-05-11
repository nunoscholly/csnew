# kNN Complementary-Fit Recommender Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the supervised DecisionTree "team balance" classifier in `ml.py` and `app.py` with an unsupervised gap-weighted kNN recommender (scikit-learn `NearestNeighbors`) that suggests participants whose skills fill each team's weakest skill dimensions.

**Architecture:** A single new function `recommend_complementary(team_participants, candidates_df, k)` replaces the existing `recommend_candidates` (Teams page) and the DecisionTree machinery used by ML Insights. The function computes a per-skill gap vector `g = max(5 − team_max, 0)`, pre-scales candidate skill vectors and a target vector (which equals `g`) by `√g`, then runs plain-Euclidean `NearestNeighbors` — mathematically equivalent to weighted-Euclidean kNN with weights `g`. Both the Teams page and ML Insights page call this same function.

**Tech Stack:** Python 3, scikit-learn (`NearestNeighbors`), pandas, numpy, Streamlit, pytest.

**Reference spec:** `docs/superpowers/specs/2026-05-11-knn-complementary-fit-recommender-design.md`

---

## File Structure

**Files modified:**
- `ml.py` — remove all DecisionTree code; add `team_gap_vector` and `recommend_complementary`; remove `recommend_candidates`.
- `app.py` — refactor Teams page recommender call (~line 203); rewrite `page_ml_insights` (~lines 432–475).
- `tests/test_ml_recommend.py` — rename/replace with `tests/test_ml_complementary.py` (or rewrite in place); old tests targeting `recommend_candidates` are obsolete.

**Files NOT touched:**
- `database.py`, `auth.py`, supabase schema, `requirements.txt`.

---

## Task 1: Write failing test for `team_gap_vector`

**Files:**
- Test: `tests/test_ml_complementary.py` (new file)

- [ ] **Step 1: Create the new test file with failing tests for `team_gap_vector`**

Create `tests/test_ml_complementary.py`:

```python
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


def test_gap_vector_never_negative():
    # If somehow team_max exceeds 5, gap is clamped at 0 (defensive)
    members = pd.DataFrame([_make_member(**{s: 5 for s in SKILL_COLUMNS})])
    g = team_gap_vector(members)
    assert np.all(g >= 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml_complementary.py -v`
Expected: ImportError or AttributeError — `team_gap_vector` does not exist in `ml.py` yet.

---

## Task 2: Implement `team_gap_vector` in `ml.py`

**Files:**
- Modify: `ml.py`

- [ ] **Step 1: Add `team_gap_vector` to `ml.py`**

Append this function to `ml.py` (just below the existing `SKILL_COLUMNS` constant; place near the top of the module since later functions will call it):

```python
def team_gap_vector(team_participants: pd.DataFrame) -> np.ndarray:
    # Returns 9-dim vector of max(5 - team_max_skill, 0) per skill; empty team -> all 5s
    if team_participants.empty:
        return np.full(len(SKILL_COLUMNS), 5.0)
    team_max = team_participants[list(SKILL_COLUMNS)].max(axis=0).to_numpy(dtype=float)
    gap = np.maximum(5.0 - team_max, 0.0)
    return gap
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_ml_complementary.py -v`
Expected: 4 passed (all `test_gap_vector_*` cases).

- [ ] **Step 3: Commit**

```bash
git add ml.py tests/test_ml_complementary.py
git commit -m "feat(ml): add team_gap_vector for skill-gap analysis

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Write failing tests for `recommend_complementary`

**Files:**
- Test: `tests/test_ml_complementary.py`

- [ ] **Step 1: Append `recommend_complementary` tests**

Append to `tests/test_ml_complementary.py`:

```python
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
    members = pd.DataFrame([
        _make_member(**{s: 5 for s in SKILL_COLUMNS}, **{"construction": 1}),
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
    members = pd.DataFrame([
        _make_member(**{s: 5 for s in SKILL_COLUMNS}, **{"design": 1}),
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
    # gap_score = candidate_skills · gap = 5 * 4 = 20 (other terms zero).
    members = pd.DataFrame([
        _make_member(**{s: 5 for s in SKILL_COLUMNS}, **{"construction": 1}),
    ])
    candidates = pd.DataFrame([
        _make_candidate("p1", "Builder", construction=5, strength=2),
    ])
    out = recommend_complementary(members, candidates, k=1)
    assert out.iloc[0]["gap_score"] == 20.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ml_complementary.py -v`
Expected: ImportError on `from ml import recommend_complementary`.

---

## Task 4: Implement `recommend_complementary` in `ml.py`

**Files:**
- Modify: `ml.py`

- [ ] **Step 1: Add `recommend_complementary` to `ml.py`**

Append this function to `ml.py` (after `team_gap_vector`):

```python
def recommend_complementary(
    team_participants: pd.DataFrame,
    candidates_df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    # Rank candidates by gap-weighted Euclidean kNN; covered skills contribute zero to distance
    empty_cols = ["id", "name", "distance", "gap_score", *SKILL_COLUMNS]
    if candidates_df.empty:
        return pd.DataFrame(columns=empty_cols)

    g = team_gap_vector(team_participants)
    if g.sum() == 0:
        return pd.DataFrame(columns=empty_cols)

    skill_matrix = candidates_df[list(SKILL_COLUMNS)].to_numpy(dtype=float)
    target = g  # ideal-complement direction equals the gap itself

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
    cols = ["id", "name", "distance", "gap_score", *SKILL_COLUMNS]
    return ordered[cols].reset_index(drop=True)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_ml_complementary.py -v`
Expected: 11 passed (4 gap-vector + 7 recommend-complementary).

- [ ] **Step 3: Commit**

```bash
git add ml.py tests/test_ml_complementary.py
git commit -m "feat(ml): add recommend_complementary gap-weighted kNN recommender

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Refactor Teams page to call `recommend_complementary`

**Files:**
- Modify: `app.py` (lines ~184–217 in `page_teams`)

- [ ] **Step 1: Update the Teams page recommender call**

In `app.py` `page_teams`, replace the block that currently reads (around line 203):

```python
                recs = ml.recommend_candidates(team_row, unassigned, k=5)
                if recs.empty:
                    st.info("No candidates match the thresholds for this team.")
                else:
                    show_cols = ["name", "distance", *ml.SKILL_COLUMNS]
                    st.dataframe(
                        recs[show_cols].assign(
                            distance=recs["distance"].round(2)
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "Lower distance = closer to the team's stated "
                        "requirements. Use the Participants page to assign."
                    )
```

with:

```python
                team_members = all_in_event[all_in_event["team_id"] == team_row["id"]]
                recs = ml.recommend_complementary(team_members, unassigned, k=5)
                if recs.empty:
                    st.info("Team is fully covered — no skill gaps left to fill.")
                else:
                    show_cols = ["name", "distance", *ml.SKILL_COLUMNS]
                    st.dataframe(
                        recs[show_cols].assign(
                            distance=recs["distance"].round(2)
                        ),
                        use_container_width=True,
                    )
                    st.caption(
                        "Lower distance = better complement to the team's current skill gaps. "
                        "Use the Participants page to assign."
                    )
```

Notes:
- `all_in_event` is already in scope (line 195 — `all_in_event = db.get_event_participants(supabase, event_id)`).
- The `show_cols` filter intentionally drops `gap_score` so the Teams page UI looks unchanged.

- [ ] **Step 2: Syntax-check the changed file**

Run: `python -m py_compile app.py && python -c "import ml"`
Expected: both succeed silently. (We can't run `import app` directly because `app.py` calls `st.set_page_config()` at module level, which requires Streamlit's runtime context.)

- [ ] **Step 3: Run the test suite to confirm nothing else broke**

Run: `pytest tests/ -v`
Expected: all `tests/test_ml_complementary.py` tests pass; `tests/test_ml_recommend.py` still passes (it tests the still-existing `recommend_candidates`).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "refactor(app): Teams page uses recommend_complementary

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Rewrite `page_ml_insights` around the new recommender

**Files:**
- Modify: `app.py` (function `page_ml_insights`, ~lines 432–475)

- [ ] **Step 1: Replace `page_ml_insights` body**

In `app.py`, replace the entire `page_ml_insights` function (starts around line 435) with:

```python
def page_ml_insights() -> None:
    # ML Insights: per-team gap analysis and gap-weighted kNN recommendations for unassigned participants
    st.header("ML Insights — Complementary-Fit Recommender (kNN)")
    st.caption(
        "For each team in the selected event, we identify the skills the team is weakest in "
        "(the gap) and use scikit-learn's k-Nearest-Neighbors to suggest unassigned participants "
        "whose strengths best fill that gap."
    )

    events = db.get_events(supabase)
    if events.empty:
        st.warning("No events exist yet.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Select event", list(event_options.keys()))
    event_id = event_options[event_label]

    teams = db.get_teams(supabase, event_id)
    if teams.empty:
        st.info("No teams in this event yet.")
        return

    participants = db.get_all_participants_for_event(supabase, event_id)
    if participants.empty:
        st.info("No participants in this event yet — add some on the Participants page.")
        return

    unassigned = participants[participants["team_id"].isna()]
    if unassigned.empty:
        st.info("All participants in this event are already assigned to teams. Showing gap analysis only.")

    for _, team_row in teams.iterrows():
        team_id = team_row["id"]
        team_name = team_row["name"]
        team_members = participants[participants["team_id"] == team_id]

        st.subheader(team_name)

        gap = ml.team_gap_vector(team_members)
        if gap.sum() == 0:
            st.success("Team is fully covered — no skill gaps.")
            continue

        # Top-3 weakest skills (largest gap values), broken ties by SKILL_COLUMNS order
        gap_pairs = sorted(
            zip(ml.SKILL_COLUMNS, gap), key=lambda kv: kv[1], reverse=True
        )
        top_gaps = [f"{name} (gap={int(g)})" for name, g in gap_pairs[:3] if g > 0]
        st.markdown("**Weakest skills:** " + ", ".join(top_gaps))

        if unassigned.empty:
            continue

        recs = ml.recommend_complementary(team_members, unassigned, k=5)
        if recs.empty:
            st.info("No suitable unassigned candidates.")
            continue

        show_cols = ["name", "distance", "gap_score", *ml.SKILL_COLUMNS]
        st.dataframe(
            recs[show_cols].assign(
                distance=recs["distance"].round(2),
                gap_score=recs["gap_score"].round(2),
            ),
            use_container_width=True,
        )

    st.divider()
    st.caption(
        "Method: gap-weighted Euclidean kNN over 9-dimensional skill vectors. "
        "Lower distance = better complement; gap_score is the dot product of "
        "the candidate's skills with the team's gap vector (higher = covers more gap)."
    )
```

- [ ] **Step 2: Syntax-check `app.py`**

Run: `python -m py_compile app.py && python -c "import ml"`
Expected: both succeed silently. (DecisionTree references in `ml.py` still exist but are now unused; the page no longer calls them.)

- [ ] **Step 3: Run the test suite**

Run: `pytest tests/ -v`
Expected: `test_ml_complementary.py` tests all pass. `test_ml_recommend.py` still passes for now (it tests the obsolete `recommend_candidates`, which still exists at this point).

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(app): rewrite ML Insights page around gap-weighted kNN

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Remove obsolete DecisionTree code and old recommender from `ml.py`

**Files:**
- Modify: `ml.py`
- Delete: `tests/test_ml_recommend.py`

- [ ] **Step 1: Strip obsolete code from `ml.py`**

Open `ml.py` and remove the following (now-unused) symbols:

- `from sklearn.metrics import classification_report`
- `from sklearn.model_selection import train_test_split`
- `from sklearn.tree import DecisionTreeClassifier`
- The `from typing import Tuple` import (no longer needed after deletions below)
- `SEED_TEAMS` (the seed DataFrame, ~17 lines)
- `FEATURE_COLUMNS`
- `build_training_data`
- `train_model`
- `train_model_with_split`
- `predict_team_balance`
- `evaluation_report`
- `features_for_team`
- `recommend_candidates`

Update the module's top comment to reflect the new purpose. The new top of `ml.py` should look like:

```python
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
]
```

…followed only by `team_gap_vector` and `recommend_complementary`.

- [ ] **Step 2: Delete the obsolete test file**

Run: `git rm tests/test_ml_recommend.py`

(This file tested `recommend_candidates`, which no longer exists.)

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: only `tests/test_ml_complementary.py` runs, all 11 tests pass.

- [ ] **Step 4: Verify the app still compiles and `ml` imports**

Run: `python -m py_compile app.py && python -c "import ml"`
Expected: both succeed silently.

- [ ] **Step 5: Grep for any leftover references**

Run: `grep -RnE "recommend_candidates|build_training_data|train_model|predict_team_balance|features_for_team|evaluation_report|SEED_TEAMS|FEATURE_COLUMNS" app.py ml.py tests/ || echo "clean"`
Expected: prints `clean` (no matches).

- [ ] **Step 6: Commit**

```bash
git add ml.py tests/test_ml_recommend.py
git commit -m "refactor(ml): drop DecisionTree classifier and threshold recommender

The supervised balance classifier and the threshold-filter
recommender are both superseded by recommend_complementary.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Manual UI smoke test

**Files:** none — runtime check.

- [ ] **Step 1: Start the Streamlit app**

Run: `streamlit run app.py`
Expected: app starts; browser opens to the app.

- [ ] **Step 2: Verify ML Insights renders**

1. Log in.
2. Pick an event with ≥ 1 team and several unassigned participants.
3. Navigate to "ML Insights".
4. Confirm: page title is "ML Insights — Complementary-Fit Recommender (kNN)"; for each team, "Weakest skills:" line appears; recommendation tables render with columns `name, distance, gap_score, <skill columns>`.

- [ ] **Step 3: Spot-check recommendation correctness**

For a team whose "Weakest skills:" line names skill X, scroll the recommendation table and confirm the top row has a higher value in skill X than the bottom row (or at least equal — ties allowed).

- [ ] **Step 4: Verify Teams page recommender**

1. Navigate to "Teams".
2. Select an event and pick a team under "Recommend candidates".
3. Confirm a recommendation table renders. If the team is fully covered, the new info message "Team is fully covered — no skill gaps left to fill." should appear.

- [ ] **Step 5: Verify edge cases**

1. Event with no teams → "No teams in this event yet."
2. Event with all participants assigned → "All participants in this event are already assigned to teams. Showing gap analysis only."

- [ ] **Step 6: Stop Streamlit**

Ctrl-C in the terminal where it's running.

---

## Task 9: Final verification and push

**Files:** none.

- [ ] **Step 1: Run the test suite one final time**

Run: `pytest tests/ -v`
Expected: 11 passed.

- [ ] **Step 2: Show recent commits for sanity check**

Run: `git log --oneline -10`
Expected: see the 4 commits from Tasks 2, 4, 5, 6, 7 in order (5 implementation commits total).

- [ ] **Step 3: Check working tree is clean**

Run: `git status`
Expected: "nothing to commit, working tree clean".

- [ ] **Step 4: (Optional) Push to remote**

Only if the user has asked for it. Run: `git push`.
