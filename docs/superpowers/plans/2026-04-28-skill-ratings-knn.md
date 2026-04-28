# Skill Ratings & kNN Team Recommendations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 9 fixed skill ratings (1–5) to participants and matching minimum-skill thresholds to teams, with a sklearn kNN recommender that suggests unassigned candidates for a team.

**Architecture:** New Supabase migration adds skill columns to `participants` and threshold columns to `teams`, makes `team_id` nullable so participants form an event-level pool. A pure `recommend_candidates` function in `ml.py` filters by thresholds then ranks survivors with `sklearn.neighbors.NearestNeighbors`. `app.py` is refactored: Participants page becomes event-scoped with skill sliders; Teams page gains threshold sliders and a recommendation panel.

**Tech Stack:** Streamlit, Supabase (PostgreSQL + supabase-py), scikit-learn, pandas, pytest.

**Spec:** See `docs/superpowers/specs/2026-04-28-skill-ratings-knn-design.md`.

---

## Task 1: Schema migration

**Files:**
- Create: `supabase/migrations/20260428180000_skill_ratings.sql`

- [ ] **Step 1: Create the migration file**

```sql
-- Skill ratings on participants + minimum-skill thresholds on teams.
-- Also makes participants.team_id nullable and adds participants.event_id
-- so participants can exist as an event-level pool independent of any team.

alter table public.participants
  add column if not exists strength     int default 3,
  add column if not exists driving      int default 3,
  add column if not exists design       int default 3,
  add column if not exists social       int default 3,
  add column if not exists construction int default 3,
  add column if not exists english      int default 3,
  add column if not exists german       int default 3,
  add column if not exists photography  int default 3,
  add column if not exists leadership   int default 3,
  add column if not exists event_id     uuid references public.events(id) on delete cascade;

alter table public.participants
  alter column team_id drop not null;

-- Backfill event_id for any pre-existing rows from their team.
update public.participants p
   set event_id = t.event_id
  from public.teams t
 where p.team_id = t.id and p.event_id is null;

alter table public.teams
  add column if not exists req_strength     int default 0,
  add column if not exists req_driving      int default 0,
  add column if not exists req_design       int default 0,
  add column if not exists req_social       int default 0,
  add column if not exists req_construction int default 0,
  add column if not exists req_english      int default 0,
  add column if not exists req_german       int default 0,
  add column if not exists req_photography  int default 0,
  add column if not exists req_leadership   int default 0;
```

- [ ] **Step 2: Push the migration to Supabase**

Run: `supabase db push`
Expected: output ends with "Finished supabase db push." and lists the new migration.

- [ ] **Step 3: Verify the columns exist**

Run: `supabase db remote commit` is NOT needed. Instead inspect the table by running a quick check from a python REPL:

```python
import os
from dotenv import load_dotenv
from supabase import create_client
load_dotenv()
sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
print(sb.table("participants").select("strength,driving,event_id").limit(1).execute())
print(sb.table("teams").select("req_strength,req_leadership").limit(1).execute())
```

Expected: both queries succeed (empty data list is fine — the point is no "column does not exist" error).

- [ ] **Step 4: Commit**

```bash
git add supabase/migrations/20260428180000_skill_ratings.sql
git commit -m "feat: add skill rating + threshold columns, event-level participant pool"
```

---

## Task 2: kNN recommender (TDD)

**Files:**
- Modify: `requirements.txt` (add pytest)
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_ml_recommend.py`
- Modify: `ml.py` (add `recommend_candidates`)

The 9 skill column names are referenced repeatedly. Define them once in `ml.py` as a module-level constant and reuse.

- [ ] **Step 1: Add pytest to requirements.txt**

Append to `requirements.txt`:

```
pytest
```

Then install: `python3 -m pip install pytest`
Expected: pytest installs (or "already satisfied").

- [ ] **Step 2: Create empty tests package**

Create `tests/__init__.py` with no content (just the file).

- [ ] **Step 3: Write the failing tests**

Create `tests/test_ml_recommend.py`:

```python
"""Tests for ml.recommend_candidates."""
import pandas as pd
import pytest

from ml import SKILL_COLUMNS, recommend_candidates


def _make_team(**reqs):
    """Helper: build a fake team row (Series) with all req_* columns defaulted to 0."""
    base = {f"req_{s}": 0 for s in SKILL_COLUMNS}
    base.update({f"req_{k}": v for k, v in reqs.items()})
    base["id"] = "team-1"
    base["name"] = "Team 1"
    return pd.Series(base)


def _make_candidate(pid, name, **skills):
    """Helper: candidate row dict with all 9 skills defaulted to 3."""
    row = {s: 3 for s in SKILL_COLUMNS}
    row.update(skills)
    row["id"] = pid
    row["name"] = name
    return row


def test_returns_empty_when_no_candidates():
    team = _make_team()
    out = recommend_candidates(team, pd.DataFrame())
    assert out.empty


def test_threshold_filters_out_low_skill_candidates():
    team = _make_team(strength=4)  # need strength >= 4
    candidates = pd.DataFrame([
        _make_candidate("p1", "Weak",   strength=2),
        _make_candidate("p2", "Strong", strength=5),
    ])
    out = recommend_candidates(team, candidates)
    assert list(out["name"]) == ["Strong"]


def test_zero_threshold_lets_everyone_through():
    team = _make_team()  # all reqs = 0
    candidates = pd.DataFrame([
        _make_candidate("p1", "A", strength=1),
        _make_candidate("p2", "B", strength=5),
    ])
    out = recommend_candidates(team, candidates)
    assert set(out["name"]) == {"A", "B"}


def test_ranks_by_distance_to_requirement_vector():
    # Team wants strength=5, leadership=5, others 0 (don't care).
    team = _make_team(strength=5, leadership=5)
    candidates = pd.DataFrame([
        _make_candidate("p1", "Far",   strength=5, leadership=5),  # distance 0 from req
        _make_candidate("p2", "Mid",   strength=4, leadership=4),  # closer to req than p3
        _make_candidate("p3", "Close", strength=5, leadership=5, design=5),
    ])
    out = recommend_candidates(team, candidates)
    # p1 and p3 both match strength/leadership exactly; p1 has design=3, p3 design=5.
    # Requirement design=0, so p1 (design=3) is closer to req than p3 (design=5).
    assert out.iloc[0]["name"] == "Far"
    assert list(out["name"])[:2] == ["Far", "Mid"] or list(out["name"])[:2] == ["Far", "Close"]
    # Distances must be non-decreasing.
    distances = list(out["distance"])
    assert distances == sorted(distances)


def test_caps_results_at_k():
    team = _make_team()
    candidates = pd.DataFrame([
        _make_candidate(f"p{i}", f"P{i}", strength=3) for i in range(10)
    ])
    out = recommend_candidates(team, candidates, k=5)
    assert len(out) == 5


def test_returns_fewer_than_k_when_few_survivors():
    team = _make_team(strength=5)
    candidates = pd.DataFrame([
        _make_candidate("p1", "A", strength=5),
        _make_candidate("p2", "B", strength=5),
        _make_candidate("p3", "C", strength=2),  # filtered out
    ])
    out = recommend_candidates(team, candidates, k=5)
    assert len(out) == 2


def test_output_columns():
    team = _make_team()
    candidates = pd.DataFrame([_make_candidate("p1", "A")])
    out = recommend_candidates(team, candidates)
    expected = {"id", "name", "distance", *SKILL_COLUMNS}
    assert expected.issubset(set(out.columns))
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_ml_recommend.py -v`
Expected: ImportError or AttributeError — `SKILL_COLUMNS` and `recommend_candidates` don't exist yet.

- [ ] **Step 5: Implement `recommend_candidates` in `ml.py`**

Add at the top of `ml.py` (near the existing imports):

```python
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

Add at the bottom of `ml.py`:

```python
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
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_ml_recommend.py -v`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
git add requirements.txt tests/__init__.py tests/test_ml_recommend.py ml.py
git commit -m "feat: add kNN recommend_candidates with threshold prefilter"
```

---

## Task 3: database.py CRUD updates

**Files:**
- Modify: `database.py`

We update `add_participant` to accept skills and event_id, add assign/unassign helpers, add `get_event_participants`, and update `create_team` to accept thresholds.

- [ ] **Step 1: Replace `add_participant` in `database.py`**

Find the existing `add_participant` function and replace it with:

```python
def add_participant(
    supabase: Client,
    event_id: str,
    name: str,
    skills: dict,
    skill: Optional[str] = None,
    status: str = "pending",
    team_id: Optional[str] = None,
) -> dict:
    """
    Insert a new participant into the event-level pool.

    Args:
        supabase: an initialised supabase-py client.
        event_id: the UUID of the event this participant belongs to.
        name: the participant's full name.
        skills: dict of the 9 skill ratings (1-5). Keys must match the
            column names: strength, driving, design, social, construction,
            english, german, photography, leadership.
        skill: legacy text label (design/engineering/business/other) used
            by the existing balance classifier. Optional.
        status: "pending" or "confirmed". Defaults to "pending".
        team_id: optional team to assign to immediately. None = unassigned.

    Returns:
        The inserted row as a dict.
    """
    payload = {
        "event_id": event_id,
        "team_id": team_id,
        "name": name,
        "skill": skill,
        "status": status,
        **skills,
    }
    response = supabase.table("participants").insert(payload).execute()
    return response.data[0]
```

- [ ] **Step 2: Add `assign_participant_to_team` and `unassign_participant`**

Add immediately after `add_participant`:

```python
def assign_participant_to_team(
    supabase: Client,
    participant_id: str,
    team_id: str,
) -> None:
    """Assign an existing participant to a team."""
    supabase.table("participants").update({"team_id": team_id}).eq(
        "id", participant_id
    ).execute()


def unassign_participant(supabase: Client, participant_id: str) -> None:
    """Remove a participant's team assignment (sets team_id = NULL)."""
    supabase.table("participants").update({"team_id": None}).eq(
        "id", participant_id
    ).execute()
```

- [ ] **Step 3: Add `get_event_participants`**

Add after `get_participants`:

```python
def get_event_participants(supabase: Client, event_id: str) -> pd.DataFrame:
    """
    Fetch every participant in an event, with team_name attached when
    the participant is assigned to a team.

    Args:
        supabase: an initialised supabase-py client.
        event_id: the UUID of the event.

    Returns:
        A DataFrame of participants with all columns plus 'team_name'
        (NaN for unassigned participants). Empty if no participants.
    """
    response = (
        supabase.table("participants")
        .select("*")
        .eq("event_id", event_id)
        .order("created_at")
        .execute()
    )
    participants = pd.DataFrame(response.data)
    if participants.empty:
        return participants

    teams_response = (
        supabase.table("teams")
        .select("id,name")
        .eq("event_id", event_id)
        .execute()
    )
    teams = pd.DataFrame(teams_response.data)
    if teams.empty:
        participants["team_name"] = pd.NA
        return participants

    teams = teams.rename(columns={"id": "team_id", "name": "team_name"})
    return participants.merge(teams, on="team_id", how="left")
```

- [ ] **Step 4: Replace `create_team` to accept thresholds**

Find `create_team` and replace with:

```python
def create_team(
    supabase: Client,
    event_id: str,
    name: str,
    thresholds: Optional[dict] = None,
) -> dict:
    """
    Insert a new team for the given event.

    Args:
        supabase: an initialised supabase-py client.
        event_id: the UUID of the event the team belongs to.
        name: the team name.
        thresholds: optional dict of req_<skill> minimums (0-5). Missing
            keys default to 0 ("don't care").

    Returns:
        The inserted row as a dict.
    """
    payload = {"event_id": event_id, "name": name}
    if thresholds:
        payload.update(thresholds)
    response = supabase.table("teams").insert(payload).execute()
    return response.data[0]
```

- [ ] **Step 5: Smoke-check the imports compile**

Run: `python3 -c "import database; print('ok')"`
Expected: `ok`.

- [ ] **Step 6: Commit**

```bash
git add database.py
git commit -m "feat: event-scoped participant pool, assign/unassign, team thresholds"
```

---

## Task 4: Refactor Participants page (event-scoped, skill sliders, assign/unassign)

**Files:**
- Modify: `app.py` — replace the entire `page_participants` function.

The existing page is team-scoped. The new page is event-scoped: pick an event → see the pool → add new participants with 9 skill sliders → optionally assign on creation → assign/unassign existing participants from a small panel.

- [ ] **Step 1: Replace `page_participants`**

Find the existing `page_participants` function (and its existing delete-participant block — replace ALL of that). The new function:

```python
def page_participants() -> None:
    """Event-scoped participant pool: list, create, assign, unassign, delete."""
    st.header("➕ Participants")

    events = db.get_events(supabase)
    if events.empty:
        st.warning("No events exist yet. Create one on the Events page first.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Select event", list(event_options.keys()))
    event_id = event_options[event_label]

    teams = db.get_teams(supabase, event_id)
    team_options = {row["name"]: row["id"] for _, row in teams.iterrows()}

    participants = db.get_event_participants(supabase, event_id)
    if participants.empty:
        st.info("No participants in this event yet.")
    else:
        display_cols = ["name", "team_name", "status", *ml.SKILL_COLUMNS]
        st.dataframe(
            participants[display_cols].fillna({"team_name": "— unassigned —"}),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Add a participant")
    with st.form("new_participant_form"):
        name = st.text_input("Name")
        skill = st.selectbox(
            "Legacy skill label (used by balance classifier)",
            ["design", "engineering", "business", "other"],
        )
        status = st.selectbox("Status", ["pending", "confirmed"])
        team_choice = st.selectbox(
            "Assign to team",
            ["— Unassigned —", *team_options.keys()],
        )
        st.markdown("**Skill ratings (1 = weak, 5 = strong)**")
        skills_input = {}
        cols = st.columns(3)
        for i, s in enumerate(ml.SKILL_COLUMNS):
            with cols[i % 3]:
                skills_input[s] = st.slider(s.capitalize(), 1, 5, 3, key=f"new_{s}")
        submitted = st.form_submit_button("Add participant")

    if submitted:
        if not name:
            st.warning("Participant name is required.")
        else:
            chosen_team_id = (
                team_options[team_choice]
                if team_choice != "— Unassigned —"
                else None
            )
            db.add_participant(
                supabase,
                event_id=event_id,
                name=name,
                skills=skills_input,
                skill=skill,
                status=status,
                team_id=chosen_team_id,
            )
            st.success(f"Added {name} to event '{event_label}'.")
            st.rerun()

    if not participants.empty:
        st.divider()
        st.subheader("Assign / unassign")
        participant_options = {
            f"{row['name']} ({row.get('team_name') or 'unassigned'})": row["id"]
            for _, row in participants.iterrows()
        }
        chosen = st.selectbox(
            "Pick a participant",
            list(participant_options.keys()),
            key="assign_pick",
        )
        chosen_id = participant_options[chosen]

        col_a, col_b = st.columns(2)
        with col_a:
            if team_options:
                target_team_label = st.selectbox(
                    "Assign to team",
                    list(team_options.keys()),
                    key="assign_team",
                )
                if st.button("➡️ Assign", key="assign_btn"):
                    db.assign_participant_to_team(
                        supabase, chosen_id, team_options[target_team_label]
                    )
                    st.success(f"Assigned to '{target_team_label}'.")
                    st.rerun()
            else:
                st.caption("Create a team first to enable assignment.")
        with col_b:
            if st.button("⬅️ Unassign", key="unassign_btn"):
                db.unassign_participant(supabase, chosen_id)
                st.success("Participant is now unassigned.")
                st.rerun()

        st.divider()
        st.subheader("Remove a participant")
        delete_options = {
            f"{row['name']} ({row.get('team_name') or 'unassigned'})": row["id"]
            for _, row in participants.iterrows()
        }
        to_delete = st.selectbox(
            "Select participant to remove",
            list(delete_options.keys()),
            key="delete_participant_select",
        )
        if st.button("🗑️ Remove participant", key="delete_participant_btn"):
            db.delete_participant(supabase, delete_options[to_delete])
            st.success(f"Removed {to_delete}.")
            st.rerun()
```

- [ ] **Step 2: Smoke-check the file compiles**

Run: `python3 -m py_compile app.py`
Expected: no output (success).

- [ ] **Step 3: Manually test in the browser**

Run: `python3 -m streamlit run app.py`
Then in the browser:
1. Log in.
2. Go to Events → create an event "Test Event 1".
3. Go to Teams → create a team "Alpha" for that event.
4. Go to Participants → select "Test Event 1".
5. Add a participant "Anna" with `strength=5, leadership=4`, leave others at 3, assign to team "Alpha". Confirm she appears in the list with her team.
6. Add a second participant "Ben" left unassigned. Confirm `team_name` shows `— unassigned —`.
7. In the assign/unassign panel: pick "Ben (unassigned)", assign him to Alpha — confirm the list updates.
8. Unassign Ben — confirm the list updates again.

Expected: every step works without errors.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: event-scoped Participants page with skill sliders + assign/unassign"
```

---

## Task 5: Teams page — threshold sliders + recommendation panel

**Files:**
- Modify: `app.py` — replace `page_teams`.

The new Teams page: pick event → list teams → create form gains 9 threshold sliders (default 0) → for each existing team, show a "Recommend candidates" panel powered by `ml.recommend_candidates`.

- [ ] **Step 1: Replace `page_teams`**

Find `page_teams` (and its delete-team block — replace ALL of that):

```python
def page_teams() -> None:
    """List teams for an event, create them with skill thresholds, recommend candidates."""
    st.header("👥 Teams")

    events = db.get_events(supabase)
    if events.empty:
        st.warning("No events exist yet. Create one on the Events page first.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Select event", list(event_options.keys()))
    event_id = event_options[event_label]

    teams = db.get_teams(supabase, event_id)
    if teams.empty:
        st.info("No teams yet for this event.")
    else:
        threshold_cols = [f"req_{s}" for s in ml.SKILL_COLUMNS]
        display_cols = ["name", *threshold_cols]
        st.dataframe(teams[display_cols], use_container_width=True)

    st.divider()
    st.subheader("Create a new team")
    with st.form("new_team_form"):
        team_name = st.text_input("Team name")
        st.markdown(
            "**Minimum skill thresholds** (0 = don't care, candidates "
            "with the skill below the threshold are filtered out)"
        )
        thresholds_input = {}
        cols = st.columns(3)
        for i, s in enumerate(ml.SKILL_COLUMNS):
            with cols[i % 3]:
                thresholds_input[f"req_{s}"] = st.slider(
                    s.capitalize(), 0, 5, 0, key=f"req_{s}"
                )
        submitted = st.form_submit_button("Create team")

    if submitted:
        if not team_name:
            st.warning("Team name is required.")
        else:
            db.create_team(supabase, event_id, team_name, thresholds_input)
            st.success(f"Team '{team_name}' created.")
            st.rerun()

    if not teams.empty:
        st.divider()
        st.subheader("Recommend candidates")
        team_label = st.selectbox(
            "Pick a team", list(teams["name"]), key="recommend_team_select"
        )
        team_row = teams[teams["name"] == team_label].iloc[0]

        all_in_event = db.get_event_participants(supabase, event_id)
        if all_in_event.empty:
            st.info("No participants in this event yet.")
        else:
            unassigned = all_in_event[all_in_event["team_id"].isna()]
            if unassigned.empty:
                st.info("No unassigned participants to recommend from.")
            else:
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

        st.divider()
        st.subheader("Delete a team")
        st.caption("Deleting a team also removes all its participants.")
        team_delete_options = {row["name"]: row["id"] for _, row in teams.iterrows()}
        to_delete = st.selectbox(
            "Select team to delete",
            list(team_delete_options.keys()),
            key="delete_team_select",
        )
        if st.button("🗑️ Delete team", key="delete_team_btn"):
            db.delete_team(supabase, team_delete_options[to_delete])
            st.success(f"Team '{to_delete}' deleted.")
            st.rerun()
```

- [ ] **Step 2: Smoke-check the file compiles**

Run: `python3 -m py_compile app.py`
Expected: no output.

- [ ] **Step 3: Manually test in the browser**

If Streamlit isn't already running, run: `python3 -m streamlit run app.py`. Then:
1. Go to Teams → select "Test Event 1".
2. Create team "Beta" with `req_strength=4`, `req_leadership=3`, others at 0. Confirm it appears with the threshold columns set.
3. Go to Participants → ensure event has at least one unassigned participant who meets the thresholds (e.g. Anna from Task 4) and one who doesn't (e.g. add "Charlie" with `strength=2`).
4. Back on Teams → "Recommend candidates" → pick "Beta". Confirm only Anna shows up (Charlie filtered by strength threshold). Distance column is present.
5. Set Beta's hypothetical thresholds to `req_strength=5, req_leadership=5` (do this by deleting the team and recreating, since editing thresholds is out of scope). Confirm "no candidates match the thresholds" message appears if no one meets it.

Expected: each step works.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: team thresholds + kNN candidate recommendation panel"
```

---

## Task 6: README + final integration check

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README**

Find the "Features" or top-level overview section in `README.md` and add a bullet (preserve existing style):

```markdown
- **Skill ratings & kNN recommendations** — rate participants on 9 skills (strength, driving, design, social, construction, english, german, photography, leadership) on a 1–5 scale, set per-team minimum thresholds, and let scikit-learn's NearestNeighbors recommend the best-fitting unassigned candidates for each team.
```

If there's a "Pages" section, also note the Participants page is now event-scoped.

- [ ] **Step 2: Final integration walkthrough**

Run: `python3 -m streamlit run app.py`. Confirm the full happy path end-to-end:
1. Create an event.
2. Create two teams with different thresholds.
3. Add five participants with varied skills, some assigned and some not.
4. Open each team's recommendation panel; confirm rankings make intuitive sense (highest-skill candidates near the top of the team that asked for those skills).
5. Delete a participant, delete a team, delete an event — confirm cascades still work.

Expected: every flow works without errors.

- [ ] **Step 3: Run the test suite**

Run: `python3 -m pytest tests/ -v`
Expected: 7 passed.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document skill ratings + kNN recommendations"
```

- [ ] **Step 5: Push**

```bash
git push
```

Expected: push succeeds, branch is up to date with origin/main.

---

## Done

All spec sections are now implemented:
- 9-skill schema migration ✓ (Task 1)
- kNN recommender with TDD ✓ (Task 2)
- database.py CRUD updates ✓ (Task 3)
- Event-scoped Participants page ✓ (Task 4)
- Teams page thresholds + recommendation panel ✓ (Task 5)
- README + integration verification ✓ (Task 6)

Out of scope (per spec): editing skills/thresholds after creation, dashboard radar chart.
