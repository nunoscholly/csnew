# Skill Ratings & kNN Team Recommendations — Design

**Date:** 2026-04-28
**Project:** Event Team Manager (HSG CS)
**Status:** Approved for planning

## Goal

Let organisers rate each participant on 9 fixed skills, define minimum skill requirements per team, and use scikit-learn's `NearestNeighbors` to recommend the best-fitting unassigned participants for a team.

## Skills (fixed list, 1–5 scale)

`strength`, `driving`, `design`, `social`, `construction`, `english`, `german`, `photography`, `leadership`.

Default rating for new participants: **3** (neutral).
Default team threshold: **0** ("don't care").

## Schema changes

A new migration adds skill columns to `participants`, threshold columns to `teams`, makes `participants.team_id` nullable, and adds `participants.event_id` so participants can exist as an event-level pool independent of any team.

```sql
-- participants: 9 skill ratings (1-5), nullable team, required event
alter table participants
  add column strength     int default 3,
  add column driving      int default 3,
  add column design       int default 3,
  add column social       int default 3,
  add column construction int default 3,
  add column english      int default 3,
  add column german       int default 3,
  add column photography  int default 3,
  add column leadership   int default 3,
  add column event_id uuid references events(id) on delete cascade,
  alter column team_id drop not null;

-- backfill event_id for any existing rows from their team
update participants p
   set event_id = t.event_id
  from teams t
 where p.team_id = t.id and p.event_id is null;

-- teams: 9 minimum-skill thresholds (0-5, 0 = "don't care")
alter table teams
  add column req_strength     int default 0,
  add column req_driving      int default 0,
  add column req_design       int default 0,
  add column req_social       int default 0,
  add column req_construction int default 0,
  add column req_english      int default 0,
  add column req_german       int default 0,
  add column req_photography  int default 0,
  add column req_leadership   int default 0;
```

Existing RLS policies still apply — no policy changes required.

## UI changes

### Participants page (refactor)
- Becomes **event-scoped**, not team-scoped.
- User selects an event → table shows every participant in that event with their skills and current team (or "unassigned").
- Add-form: name, optional skill (legacy `skill` column kept for now), 9 skill sliders (1–5, default 3), optional team dropdown ("Unassigned" default).
- Each participant row gets an "Assign to team" / "Unassign" control.

### Teams page (extend)
- Team creation form adds 9 threshold sliders (0–5, default 0).
- Team detail panel shows current members and a **"Recommend candidates"** section that calls the kNN recommender against unassigned participants in that event.

### Dashboard
- Out of scope for v1. (Radar chart noted as a possible follow-up.)

### ML Insights
- Unchanged. The existing balance classifier still runs.

## ML logic

`ml.recommend_candidates(team_row, candidates_df, k=5) -> pd.DataFrame`:

1. **Filter:** keep candidates where every skill ≥ the matching `req_*` threshold on the team. Thresholds of 0 are no-ops.
2. **Rank:** if any survivors, fit `sklearn.neighbors.NearestNeighbors(n_neighbors=min(k, n_survivors), metric="euclidean")` on the survivors' 9-D skill vectors, query with the team's 9-D requirement vector, take the nearest.
3. **Return:** DataFrame with `id, name, distance, <9 skill columns>` sorted ascending by `distance`. Lower distance = closer to the team's stated requirements.

### Edge cases
- 0 survivors → return empty DataFrame; UI shows "no candidates match the thresholds".
- < 5 survivors → return what's available (k clamps to `n_survivors`).
- 0 unassigned participants in the event → UI shows "no unassigned participants to recommend from".

## Files to touch

- **New:** `supabase/migrations/<new-timestamp>_skill_ratings.sql` (schema + backfill).
- **`database.py`:**
  - Update `add_participant` to accept `event_id`, optional `team_id`, and a skills dict.
  - Add `assign_participant_to_team(supabase, participant_id, team_id)` and `unassign_participant(supabase, participant_id)`.
  - Add `get_event_participants(supabase, event_id)` returning all participants in the event (with team_name where present, or NaN).
  - Update `create_team` to accept a thresholds dict.
- **`ml.py`:** add `recommend_candidates`.
- **`app.py`:** refactor `page_participants` (event-scoped + sliders), extend `page_teams` (threshold sliders + recommendation panel).
- **`README.md`:** brief update on the new skills feature.

## Out of scope

- Editing a participant's skills after creation.
- Editing team thresholds after creation.
- Dashboard radar chart.

## Open assumptions

- The legacy `participants.skill` text column stays for backward compatibility with the existing ML balance classifier; it's optional in the new add-form.
- The 9 skills are fixed in code (no admin UI for adding new skills).
- "Don't care" thresholds are encoded as 0, not NULL, to keep the filter logic uniform.
