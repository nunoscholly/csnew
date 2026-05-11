# kNN Complementary-Fit Recommender — Design

**Date:** 2026-05-11
**Status:** Approved (pending written-spec review)

## Goal

Replace the supervised DecisionTree "team balance" classifier with an unsupervised kNN recommender (scikit-learn `NearestNeighbors`) that, for each existing team, suggests unassigned participants whose skills best **complement the team's gaps**. Unify both in-app recommenders (Teams page and ML Insights page) on a single gap-weighted kNN implementation.

## Motivation

The current ML Insights page trains a `DecisionTreeClassifier` to predict whether a team is "balanced" or "unbalanced." That model is supervised, but the label is rule-derived (no ground truth), so the classifier mostly memorises the rule. A kNN-based unsupervised recommender is a better fit for the actual product question: "who should we add to this team next?"

## Scope

- **In scope:** rewrite `ml.py`'s public surface; rewrite `page_ml_insights` in `app.py`; refactor the Teams page recommender to share the same kNN implementation.
- **Out of scope:** auto team-formation (creating whole teams from scratch); ML Insights for new teams that don't exist yet; changes to database schema.

## Definitions

- **Skill columns** (unchanged, 9 dimensions): `strength, driving, design, social, construction, english, german, photography, leadership`. Each rated 1–5.
- **Team profile** (per skill): the **max rating among current team members**. A team "covers" a skill if any member is strong in it. Empty team → profile is all zeros.
- **Gap vector** `g` (per skill): `max(5 − team_max, 0)`. A 9-dim non-negative vector. Empty team → `g = [5, 5, 5, 5, 5, 5, 5, 5, 5]` (everything is a gap).
- **Complementary fit:** a candidate is a good fit for a team if their skills are strong precisely in the dimensions where the team has gaps.

## Architecture

### `ml.py` — public surface after the change

**Remove:**
- `SEED_TEAMS`, `FEATURE_COLUMNS`
- `build_training_data`, `train_model`, `train_model_with_split`, `predict_team_balance`, `evaluation_report`, `features_for_team`
- Imports: `DecisionTreeClassifier`, `classification_report`, `train_test_split`

**Keep:**
- `SKILL_COLUMNS` (used by `app.py` for column ordering)
- `NearestNeighbors` import

**Add:**

```python
def team_gap_vector(team_participants: pd.DataFrame) -> np.ndarray:
    """For each skill in SKILL_COLUMNS, returns max(5 - team_max, 0).
    Empty team -> all-5s. Returns shape (9,) float array."""
```

```python
def recommend_complementary(
    team_participants: pd.DataFrame,
    candidates_df: pd.DataFrame,
    k: int = 5,
) -> pd.DataFrame:
    """Rank candidates by gap-weighted kNN distance to the team's gap vector.
    Returns columns: id, name, distance, gap_score, *SKILL_COLUMNS.
    Empty candidates_df or all-zero gap -> empty DataFrame with those columns."""
```

**Refactor:**
- `recommend_candidates` is removed. Its single caller (`app.py` Teams page) is updated to call `recommend_complementary(team_members, unassigned, k=5)` instead. The `req_<skill>` columns are no longer used by ml.py.

### Algorithm: gap-weighted kNN

1. Compute `g = team_gap_vector(team_participants)`.
2. If `g.sum() == 0`, return empty result (team is fully covered).
3. If `candidates_df` is empty, return empty result.
4. Build the candidate skill matrix `X` (shape `n × 9`) from `candidates_df[SKILL_COLUMNS]`.
5. Build the target vector `t` so that `t_i = 5` wherever `g_i > 0` and `t_i = 0` elsewhere. The "ideal candidate" is maximum strength in every gap dimension; covered dimensions don't contribute to distance anyway because their weight is zero.
6. **Pre-scale** both `X` and `t` element-wise by `√g`:
   - `X_scaled = X * np.sqrt(g)`
   - `t_scaled = t * np.sqrt(g)`
   This is mathematically equivalent to weighted-Euclidean kNN with weights `g`, and avoids sklearn's deprecated `wminkowski` metric. Skills with `g_i = 0` contribute zero to the distance, so they're effectively ignored.
7. Fit `NearestNeighbors(n_neighbors=min(k, n), metric="euclidean")` on `X_scaled`.
8. Query with `t_scaled` → `(distances, indices)`.
9. For each returned candidate, also compute a human-readable `gap_score = candidate_skill_vector · g` (dot product). Higher = covers more of the gap.
10. Return a DataFrame with `id, name, distance, gap_score, *SKILL_COLUMNS`, ordered by ascending distance (best fit first).

### `app.py` — `page_ml_insights` rewrite

Replace the function body with:

1. Header: `"ML Insights — Complementary-Fit Recommender (kNN)"`.
2. Short caption explaining the method: "For each team, we identify which skills are weakest (the team's gap) and use a k-Nearest-Neighbors search to suggest unassigned participants whose strengths fill that gap."
3. Event selector (unchanged logic).
4. Load `teams = db.get_teams_with_requirements(supabase, event_id)` and `participants = db.get_all_participants_for_event(supabase, event_id)`.
5. Split participants: `team_members_by_team` (groupby `team_id`, excluding nulls) and `unassigned = participants[participants.team_id.isna()]`.
6. Early exits:
   - Event has no teams → info message, return.
   - Event has no unassigned participants → warning, then still show gap analysis but skip recommendations.
7. For each team:
   - Compute `g = ml.team_gap_vector(team_members)`.
   - Render team header (team name).
   - **Gap summary:** show top-3 weakest skills (highest `g` values), e.g., `"Weakest skills: construction (gap=3), english (gap=2), photography (gap=1)"`. If `g.sum() == 0`, show `"Team is fully covered — no gaps."` and skip recommendation table.
   - **Recommendations table:** call `ml.recommend_complementary(team_members, unassigned, k=5)`. Render with `st.dataframe`. Columns shown: `name, distance, gap_score, *SKILL_COLUMNS`. Empty result → small info note.

### `app.py` — Teams page change

Replace the existing call at line ~203:

```python
recs = ml.recommend_candidates(team_row, unassigned, k=5)
```

with:

```python
team_members = participants[participants["team_id"] == team_row["id"]]
recs = ml.recommend_complementary(team_members, unassigned, k=5)
```

The existing display-column filter on the Teams page (`show_cols = ["name", "distance", *ml.SKILL_COLUMNS]`) is left as-is, so `gap_score` is silently dropped there. ML Insights is the only page that surfaces `gap_score`. This keeps the Teams page UI visually unchanged.

## Data flow

```
event_id
  └─ db.get_teams_with_requirements(supabase, event_id)            -> teams
  └─ db.get_all_participants_for_event(supabase, event_id)         -> participants
       ├─ unassigned = participants[participants.team_id.isna()]
       └─ for each team_id, members:
            g = ml.team_gap_vector(members)
            recs = ml.recommend_complementary(members, unassigned, 5)
            render team header, gap summary, recs table
```

No new database queries are introduced.

## Error handling

| Case | Behavior |
|------|----------|
| No events in DB | Existing warning ("No events exist yet"), return. |
| Event has no teams | `st.info("No teams in this event yet.")`, return. |
| Event has no unassigned participants | `st.info("All participants are already assigned.")`, still show per-team gap analysis but skip rec tables. |
| Team has zero members (empty team) | Gap = all 5s, recommendations behave like "find generally strong candidates." Show normal recommendation table. |
| Team is fully covered (`g.sum() == 0`) | Show `"Team is fully covered — no gaps."`, skip rec table for that team. |
| Fewer than `k` unassigned candidates | `min(k, n)` neighbours returned, no error. |

## Edge cases for the kNN math

- `g` with one non-zero entry (team only weak in one skill) → distance collapses to a 1-D distance in that skill. Behaves correctly.
- All candidates have identical skill vectors → kNN returns them in arbitrary index order; that's acceptable (tie-breaking not specified).
- Candidate ratings outside 1–5 are not validated; we trust the schema's CHECK constraints.

## Testing strategy

The repo has a pytest suite at `tests/`. The existing `tests/test_ml_recommend.py` tests `recommend_candidates`, which is being removed; it must be replaced with tests for the new API.

**Automated (pytest):**
- `team_gap_vector` — empty team → all-5s; partial team → correct per-skill `max(5 − team_max, 0)`; team with a 5 in every skill → all zeros.
- `recommend_complementary` —
  - empty candidates → empty DataFrame with correct columns.
  - all-zero gap (fully covered team) → empty DataFrame.
  - candidate strong in the gap skill ranks above candidate weak in it.
  - skills already covered by the team are ignored in ranking (a candidate weak in a covered skill is *not* penalized).
  - `k` caps result count; fewer-than-k candidates returns all of them.
  - output schema includes `id, name, distance, gap_score, *SKILL_COLUMNS`.

**Manual (Streamlit UI):**
1. `streamlit run app.py`.
2. Select an event with at least one team and several unassigned participants.
3. ML Insights renders without error; gap summary lists weakest skills; recommendations rank candidates strong in those skills highest.
4. Teams page still renders recommendations after the refactor.
5. Edge cases: event with no teams; event where all participants are assigned; team fully covered (no gaps).

## Migration / rollout

This is a self-contained code change in `ml.py` + `app.py`. No data migration, no schema change, no environment changes. A single commit can ship it.

## Open questions

None at design time. Both design questions raised in brainstorming were resolved:
- **Unify both recommenders:** yes (Teams page + ML Insights both use `recommend_complementary`).
- **Empty team handling:** treat all skills as gaps (gap vector = all 5s).
