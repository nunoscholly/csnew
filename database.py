# Datenbank-CRUD-Schicht: alle Supabase-Abfragen für Veranstaltungen, Teams und Teilnehmer; nutzt parametrisierte Abfragen, um SQL-Injection zu verhindern

from typing import Optional

import pandas as pd
from supabase import Client

from ml import SKILL_COLUMNS

# Skill-Spalten, die durch die spätere "more_skills"-Migration ergänzt werden.
# Falls diese Migration in der Ziel-Datenbank noch nicht angewendet wurde,
# werden Inserts mit diesen Keys vom Supabase-Schema-Cache abgelehnt
# ("column does not exist"). Wir entfernen sie dann beim Retry.
_OPTIONAL_SKILL_COLUMNS = ("communication", "experience", "problem_solving")


# ---------------------------------------------------------------------------
# Veranstaltungen
# ---------------------------------------------------------------------------

def get_events(supabase: Client) -> pd.DataFrame:
    # Alle Veranstaltungen aus der Datenbank laden
    response = supabase.table("events").select("*").order("date").execute()
    return pd.DataFrame(response.data)


def create_event(
    supabase: Client,
    name: str,
    date: str,
    location: str,
) -> dict:
    # Neue Veranstaltung einfügen
    payload = {"name": name, "date": date, "location": location}
    response = supabase.table("events").insert(payload).execute()
    return response.data[0]


def delete_event(supabase: Client, event_id: str) -> None:
    # Veranstaltung löschen (Teams und Teilnehmer werden per FK-Constraint kaskadiert entfernt)
    supabase.table("events").delete().eq("id", event_id).execute()


# ---------------------------------------------------------------------------
# Teams
# ---------------------------------------------------------------------------

def get_teams(supabase: Client, event_id: str) -> pd.DataFrame:
    # Alle Teams der angegebenen Veranstaltung laden
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
    # Neues Team mit optionalen Skill-Schwellen einfügen
    payload = {"event_id": event_id, "name": name}
    if thresholds:
        payload.update(thresholds)
    response = supabase.table("teams").insert(payload).execute()
    return response.data[0]


def delete_team(supabase: Client, team_id: str) -> None:
    # Team löschen (Teilnehmer werden per FK-Constraint kaskadiert entfernt)
    supabase.table("teams").delete().eq("id", team_id).execute()


# ---------------------------------------------------------------------------
# Teilnehmer
# ---------------------------------------------------------------------------

def get_participants(supabase: Client, team_id: str) -> pd.DataFrame:
    # Alle Teilnehmer des angegebenen Teams laden
    response = (
        supabase.table("participants")
        .select("*")
        .eq("team_id", team_id)
        .order("created_at")
        .execute()
    )
    return pd.DataFrame(response.data)


def get_event_participants(supabase: Client, event_id: str) -> pd.DataFrame:
    # Teilnehmer der Veranstaltung inklusive Teamnamen laden; eine fehlende event_id-Spalte wird tolerant behandelt
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
    # Teilnehmer mit Skill-Bewertungen einfügen; es werden nur Schlüssel aus SKILL_COLUMNS übernommen
    payload = {
        "event_id": event_id,
        "team_id": team_id,
        "name": name,
        "skill": skill,
        "status": status,
    }
    payload.update({k: v for k, v in skills.items() if k in SKILL_COLUMNS})
    try:
        response = supabase.table("participants").insert(payload).execute()
    except Exception as exc:
        # Wenn die neueren Skill-Spalten in der DB fehlen (Migration noch nicht
        # angewendet), wirft PostgREST einen Schema-Fehler. Wir entfernen die
        # optionalen Spalten und versuchen es erneut. Bei anderen Fehlern wird
        # die Exception nach dem Retry-Versuch erneut geworfen.
        if not any(col in payload for col in _OPTIONAL_SKILL_COLUMNS):
            raise
        for col in _OPTIONAL_SKILL_COLUMNS:
            payload.pop(col, None)
        try:
            response = supabase.table("participants").insert(payload).execute()
        except Exception:
            raise exc
    return response.data[0]


def assign_participant_to_team(
    supabase: Client,
    participant_id: str,
    team_id: str,
) -> None:
    # Teilnehmer einem Team zuweisen
    supabase.table("participants").update({"team_id": team_id}).eq(
        "id", participant_id
    ).execute()


def unassign_participant(supabase: Client, participant_id: str) -> None:
    # Teamzuweisung eines Teilnehmers aufheben
    supabase.table("participants").update({"team_id": None}).eq(
        "id", participant_id
    ).execute()


def delete_participant(supabase: Client, participant_id: str) -> None:
    # Teilnehmer löschen
    supabase.table("participants").delete().eq("id", participant_id).execute()


def get_all_participants_for_event(
    supabase: Client,
    event_id: str,
) -> pd.DataFrame:
    # Alle Teilnehmer dieser Veranstaltung laden – sowohl Team-zugewiesene als auch
    # die noch im Pool (team_id IS NULL). Wir filtern direkt über participants.event_id
    # statt über team_id, sonst würden nicht zugewiesene Teilnehmer ausgeschlossen.
    response = (
        supabase.table("participants")
        .select("*")
        .eq("event_id", event_id)
        .execute()
    )
    participants = pd.DataFrame(response.data)
    if participants.empty:
        return participants

    # Eine lesbare team_name-Spalte anhängen (NaN für nicht zugewiesene Teilnehmer).
    teams = get_teams(supabase, event_id)
    if teams.empty:
        participants["team_name"] = None
        return participants

    teams_slim = teams[["id", "name"]].rename(
        columns={"id": "team_id", "name": "team_name"}
    )
    return participants.merge(teams_slim, on="team_id", how="left")
