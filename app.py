"""
app.py
------
Main Streamlit app for the Event Team Manager.

This file owns the UI and routing. It delegates to:
    - auth.py      for Supabase Auth
    - database.py  for Supabase CRUD
    - ml.py        for the team balance classifier

Run locally with:
    streamlit run app.py
"""

from datetime import date as date_cls

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import auth
import database as db
import ml


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
# Configure the Streamlit page once at the top so the title/favicon are
# consistent across reruns.
st.set_page_config(page_title="Event Team Manager", layout="wide")


# ---------------------------------------------------------------------------
# Supabase client bootstrap
# ---------------------------------------------------------------------------
# We keep a single Supabase client in session_state so we don't create a
# new one on every Streamlit rerun. If credentials are missing we show a
# friendly error and stop the app.
if "supabase" not in st.session_state:
    try:
        st.session_state["supabase"] = auth.init_supabase()
    except ValueError as e:
        st.error(str(e))
        st.stop()

supabase = st.session_state["supabase"]


# ---------------------------------------------------------------------------
# Login screen
# ---------------------------------------------------------------------------
def render_login() -> None:
    """Render Login and Sign up tabs for unauthenticated users."""
    st.title("Event Team Manager")

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    # --- Log in tab -----------------------------------------------------
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            try:
                auth.login(supabase, email, password)
                st.success("Logged in successfully.")
                st.rerun()
            except Exception as e:
                # Supabase raises various exception types (wrong password,
                # network error, unconfirmed email, etc.). One message covers all.
                st.error(f"Login failed: {e}")

    # --- Sign up tab ----------------------------------------------------
    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input(
                "Password (min. 6 characters)",
                type="password",
                key="signup_password",
            )
            submitted = st.form_submit_button("Create account")

        if submitted:
            try:
                response = auth.sign_up(supabase, email, password)
                if response.session is not None:
                    # Email confirmation is disabled — user is logged in.
                    st.success("Account created and logged in.")
                    st.rerun()
                else:
                    # Confirmation required — user must click the email link.
                    st.success(
                        "Account created. Check your inbox for a confirmation "
                        "link, then come back and log in."
                    )
            except Exception as e:
                st.error(f"Sign up failed: {e}")


# ---------------------------------------------------------------------------
# Page: Events
# ---------------------------------------------------------------------------
def page_events() -> None:
    """List all events and allow the user to create a new one."""
    st.header("Events")

    events = db.get_events(supabase)
    if events.empty:
        st.info("No events yet. Create your first event below.")
    else:
        st.dataframe(events, use_container_width=True)

    st.divider()
    st.subheader("Create a new event")
    with st.form("new_event_form"):
        name = st.text_input("Name")
        event_date = st.date_input("Date", value=date_cls.today())
        location = st.text_input("Location")
        submitted = st.form_submit_button("Create event")

    if submitted:
        if not name:
            st.warning("Event name is required.")
        else:
            db.create_event(supabase, name, event_date.isoformat(), location)
            st.success(f"Event '{name}' created.")
            st.rerun()

    if not events.empty:
        st.divider()
        st.subheader("Delete an event")
        st.caption("Deleting an event also removes all its teams and participants.")
        delete_options = {row["name"]: row["id"] for _, row in events.iterrows()}
        to_delete = st.selectbox(
            "Select event to delete", list(delete_options.keys()), key="delete_event_select"
        )
        if st.button("Delete event", key="delete_event_btn"):
            db.delete_event(supabase, delete_options[to_delete])
            st.success(f"Event '{to_delete}' deleted.")
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Teams
# ---------------------------------------------------------------------------
def page_teams() -> None:
    """List teams for an event, create them with skill thresholds, recommend candidates."""
    st.header("Teams")

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
        threshold_cols = [f"req_{s}" for s in ml.SKILL_COLUMNS if f"req_{s}" in teams.columns]
        display_cols = ["name"] + threshold_cols if "name" in teams.columns else list(teams.columns)
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
                    s.capitalize(), 0, 5, 0, key=f"req_{s}_{event_id}"
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
        team_options = {row["name"]: row["id"] for _, row in teams.iterrows()}
        team_label = st.selectbox(
            "Pick a team",
            list(team_options.keys()),
            key=f"recommend_team_select_{event_id}",
        )
        team_row = teams[teams["id"] == team_options[team_label]].iloc[0]

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
            key=f"delete_team_select_{event_id}",
        )
        if st.button("Delete team", key=f"delete_team_btn_{event_id}"):
            db.delete_team(supabase, team_delete_options[to_delete])
            st.success(f"Team '{to_delete}' deleted.")
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Participants
# ---------------------------------------------------------------------------
def page_participants() -> None:
    """Event-scoped participant pool: list, create, assign, unassign, delete."""
    st.header("Participants")

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
        display_cols = [col for col in ["name", "team_name", "status", *ml.SKILL_COLUMNS] if col in participants.columns]
        st.dataframe(
            participants[display_cols].fillna({"team_name": "Unassigned"}),
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
            ["Unassigned", *team_options.keys()],
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
                if team_choice != "Unassigned"
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
        # Build the {label: id} map once. pd.notna check is needed because
        # `team_name` is pd.NA for unassigned rows, and `pd.NA or "..."`
        # raises TypeError.
        participant_options = {}
        for _, row in participants.iterrows():
            team_name = row.get("team_name")
            label_team = team_name if pd.notna(team_name) else "unassigned"
            participant_options[f"{row['name']} ({label_team})"] = row["id"]

        st.divider()
        st.subheader("Assign / unassign")
        chosen = st.selectbox(
            "Pick a participant",
            list(participant_options.keys()),
            key=f"assign_pick_{event_id}",
        )
        chosen_id = participant_options[chosen]

        col_a, col_b = st.columns(2)
        with col_a:
            if team_options:
                target_team_label = st.selectbox(
                    "Assign to team",
                    list(team_options.keys()),
                    key=f"assign_team_{event_id}",
                )
                if st.button("Assign", key=f"assign_btn_{event_id}"):
                    db.assign_participant_to_team(
                        supabase, chosen_id, team_options[target_team_label]
                    )
                    st.success(f"Assigned to '{target_team_label}'.")
                    st.rerun()
            else:
                st.caption("Create a team first to enable assignment.")
        with col_b:
            if st.button("Unassign", key=f"unassign_btn_{event_id}"):
                db.unassign_participant(supabase, chosen_id)
                st.success("Participant is now unassigned.")
                st.rerun()

        st.divider()
        st.subheader("Remove a participant")
        to_delete = st.selectbox(
            "Select participant to remove",
            list(participant_options.keys()),
            key=f"delete_participant_select_{event_id}",
        )
        if st.button("Remove participant", key=f"delete_participant_btn_{event_id}"):
            db.delete_participant(supabase, participant_options[to_delete])
            st.success(f"Removed {to_delete}.")
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    """Visualise team sizes, skill distribution and confirmed/pending split."""
    st.header("Dashboard")

    events = db.get_events(supabase)
    if events.empty:
        st.warning("No events exist yet.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Select event", list(event_options.keys()))
    event_id = event_options[event_label]

    participants = db.get_all_participants_for_event(supabase, event_id)
    if participants.empty:
        st.info("No participants yet for this event.")
        return

    # --- Team sizes ------------------------------------------------------
    st.subheader("Team sizes")
    team_sizes = participants.groupby("team_name").size().rename("members")
    st.bar_chart(team_sizes)

    # --- Skill distribution ---------------------------------------------
    st.subheader("Skill distribution")
    skill_counts = participants["skill"].fillna("unknown").value_counts()
    fig, ax = plt.subplots()
    ax.pie(skill_counts.values, labels=skill_counts.index, autopct="%1.0f%%")
    ax.set_aspect("equal")
    st.pyplot(fig)

    # --- Confirmed vs pending -------------------------------------------
    st.subheader("Confirmed vs pending")
    status_counts = participants["status"].value_counts()
    st.bar_chart(status_counts)


# ---------------------------------------------------------------------------
# Page: ML Insights
# ---------------------------------------------------------------------------
def page_ml_insights() -> None:
    """Run the balance classifier against every team and show predictions."""
    st.header("ML Insights — Team Balance Classifier")

    events = db.get_events(supabase)
    if events.empty:
        st.warning("No events exist yet.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Select event", list(event_options.keys()))
    event_id = event_options[event_label]

    participants = db.get_all_participants_for_event(supabase, event_id)

    # Train on real data + seed rows. ml.build_training_data handles empty input.
    X, y = ml.build_training_data(participants)
    model, X_test, y_test = ml.train_model_with_split(X, y)

    st.subheader("Classification report (held-out test data)")
    st.code(ml.evaluation_report(model, X_test, y_test))

    st.subheader("Predictions per team")
    if participants.empty:
        st.info("No teams to predict on yet — add participants first.")
        return

    rows = []
    for team_id, team_df in participants.groupby("team_id"):
        feats = ml.features_for_team(team_df)
        prediction = ml.predict_team_balance(model, feats)
        rows.append(
            {
                "team": team_df["team_name"].iloc[0],
                "team_size": feats["team_size"],
                "num_skills": feats["num_skills"],
                "confirmed_ratio": round(feats["confirmed_ratio"], 2),
                "prediction": prediction,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar + routing
# ---------------------------------------------------------------------------
def render_app() -> None:
    """Render the sidebar navigation and dispatch to the chosen page."""
    user = st.session_state.get("user")
    user_email = getattr(user, "email", "user") if user else "user"

    st.sidebar.title("Event Team Manager")
    st.sidebar.caption(f"Logged in as: {user_email}")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Events",
            "Teams",
            "Participants",
            "Dashboard",
            "ML Insights",
        ],
    )

    st.sidebar.divider()
    if st.sidebar.button("Logout"):
        auth.logout(supabase)
        st.rerun()

    # Dispatch — keep this map flat and easy to extend.
    pages = {
        "Events": page_events,
        "Teams": page_teams,
        "Participants": page_participants,
        "Dashboard": page_dashboard,
        "ML Insights": page_ml_insights,
    }
    pages[page]()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if auth.get_session() is None:
    render_login()
else:
    render_app()
