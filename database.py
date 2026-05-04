# Database CRUD layer: all Supabase queries for events, teams, participants using parameterized queries to prevent SQL injection

from typing import Optional

import pandas as pd
from supabase import Client

from ml import SKILL_COLUMNS


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def get_events(supabase: Client) -> pd.DataFrame:
    # Fetch all events from database
    response = supabase.table("events").select("*").order("date").execute()
    return pd.DataFrame(response.data)


def create_event(
    supabase: Client,
    name: str,
    date: str,
    location: str,
) -> dict:
    # Insert new event
    payload = {"name": name, "date": date, "location": location}
    response = supabase.table("events").insert(payload).execute()
    return response.data[0]


def delete_event(supabase: Client, event_id: str) -> None:
    # Delete event (teams and participants cascade via FK constraints)
    supabase.table("events").delete().eq("id", event_id).execute()


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def get_teams(supabase: Client, event_id: str) -> pd.DataFrame:
    # Fetch all teams for given event
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
    # Insert new team with optional skill thresholds
    payload = {"event_id": event_id, "name": name}
    if thresholds:
        payload.update(thresholds)
    response = supabase.table("teams").insert(payload).execute()
    return response.data[0]


def delete_team(supabase: Client, team_id: str) -> None:
    # Delete team (participants cascade via FK)
    supabase.table("teams").delete().eq("id", team_id).execute()


# ---------------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------------

def get_participants(supabase: Client, team_id: str) -> pd.DataFrame:
    # Fetch all participants for given team
    response = (
        supabase.table("participants")
        .select("*")
        .eq("team_id", team_id)
        .order("created_at")
        .execute()
    )
    return pd.DataFrame(response.data)


def get_event_participants(supabase: Client, event_id: str) -> pd.DataFrame:
    # Fetch event participants with team names; handle missing event_id column gracefully
    try:
        response = (
            supabase.table("participants")
            .select("*")
            .eq("event_id", event_id)
            .order("created_at")
            .execute()
        )
        participants = pd.DataFrame(response.data)
    except Exception:
        response = (
            supabase.table("participants")
            .select("*")
            .execute()
        )
        participants = pd.DataFrame(response.data)
        if not participants.empty and "event_id" in participants.columns:
            participants = participants[participants["event_id"] == event_id].reset_index(drop=True)
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
    # Insert participant with skill ratings; only include valid SKILL_COLUMNS
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
    # Assign participant to team
    supabase.table("participants").update({"team_id": team_id}).eq(
        "id", participant_id
    ).execute()


def unassign_participant(supabase: Client, participant_id: str) -> None:
    # Remove participant from team assignment
    supabase.table("participants").update({"team_id": None}).eq(
        "id", participant_id
    ).execute()


def delete_participant(supabase: Client, participant_id: str) -> None:
    # Delete participant
    supabase.table("participants").delete().eq("id", participant_id).execute()


def get_all_participants_for_event(
    supabase: Client,
    event_id: str,
) -> pd.DataFrame:
    # Fetch all participants across all teams for event; use two queries then pandas join for clarity
    teams = get_teams(supabase, event_id)
    if teams.empty:
        return pd.DataFrame()

    # Filter participants by event's team IDs
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
