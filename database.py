"""
database.py
-----------
All Supabase database calls for the Event Team Manager app.

Each function takes the supabase client as its first argument so we
keep the side-effects (network calls) explicit and easy to test.
We use the supabase-py query builder, which sends parametrised
requests to the Supabase REST API — that means we never build SQL
strings by hand and avoid SQL injection risks.

Tables used (see README.md for the schema):
    - events
    - teams
    - participants
"""

from typing import Optional

import pandas as pd
from supabase import Client

from ml import SKILL_COLUMNS


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def get_events(supabase: Client) -> pd.DataFrame:
    """
    Fetch every event from the database.

    Args:
        supabase: an initialised supabase-py client.

    Returns:
        A pandas DataFrame with columns: id, name, date, location, created_at.
        Returns an empty DataFrame if there are no events yet.
    """
    response = supabase.table("events").select("*").order("date").execute()
    return pd.DataFrame(response.data)


def create_event(
    supabase: Client,
    name: str,
    date: str,
    location: str,
) -> dict:
    """
    Insert a new event row.

    Args:
        supabase: an initialised supabase-py client.
        name: event name (e.g. "HSG Hackathon 2026").
        date: event date in ISO format "YYYY-MM-DD".
        location: free-text location string.

    Returns:
        The inserted row as a dict.
    """
    payload = {"name": name, "date": date, "location": location}
    response = supabase.table("events").insert(payload).execute()
    return response.data[0]


def delete_event(supabase: Client, event_id: str) -> None:
    """Delete an event. Teams and participants cascade automatically (FK on delete cascade)."""
    supabase.table("events").delete().eq("id", event_id).execute()


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def get_teams(supabase: Client, event_id: str) -> pd.DataFrame:
    """
    Fetch all teams that belong to a given event.

    Args:
        supabase: an initialised supabase-py client.
        event_id: the UUID of the event (as a string).

    Returns:
        A pandas DataFrame of teams, or empty DataFrame if none exist.
    """
    response = (
        supabase.table("teams")
        .select("*")
        .eq("event_id", event_id)
        .order("created_at")
        .execute()
    )
    return pd.DataFrame(response.data)


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


def delete_team(supabase: Client, team_id: str) -> None:
    """Delete a team. Participants cascade automatically."""
    supabase.table("teams").delete().eq("id", team_id).execute()


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------

def get_participants(supabase: Client, team_id: str) -> pd.DataFrame:
    """
    Fetch all participants for a given team.

    Args:
        supabase: an initialised supabase-py client.
        team_id: the UUID of the team.

    Returns:
        A pandas DataFrame of participants for that team.
    """
    response = (
        supabase.table("participants")
        .select("*")
        .eq("team_id", team_id)
        .order("created_at")
        .execute()
    )
    return pd.DataFrame(response.data)


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
    }
    payload.update({k: v for k, v in skills.items() if k in SKILL_COLUMNS})
    response = supabase.table("participants").insert(payload).execute()
    return response.data[0]


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


def delete_participant(supabase: Client, participant_id: str) -> None:
    """Delete a single participant."""
    supabase.table("participants").delete().eq("id", participant_id).execute()


def get_all_participants_for_event(
    supabase: Client,
    event_id: str,
) -> pd.DataFrame:
    """
    Return every participant across every team for a given event.

    We do this in two queries (teams, then participants) and join in
    pandas. That keeps each Supabase call simple and beginner-friendly.

    Args:
        supabase: an initialised supabase-py client.
        event_id: the UUID of the event.

    Returns:
        A DataFrame with participant rows plus a "team_name" column,
        or an empty DataFrame if the event has no teams/participants.
    """
    teams = get_teams(supabase, event_id)
    if teams.empty:
        return pd.DataFrame()

    # Pull every participant whose team_id is in this event's team list.
    team_ids = teams["id"].tolist()
    response = (
        supabase.table("participants")
        .select("*")
        .in_("team_id", team_ids)
        .execute()
    )
    participants = pd.DataFrame(response.data)
    if participants.empty:
        return participants

    # Add a readable team_name column by merging on team_id.
    teams_slim = teams[["id", "name"]].rename(
        columns={"id": "team_id", "name": "team_name"}
    )
    return participants.merge(teams_slim, on="team_id", how="left")
