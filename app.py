# Main Streamlit app: UI layer that routes between pages and delegates to auth, database, and ML modules.

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
    # Render login/signup interface for unauthenticated users
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
    # List events and allow creation/deletion
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
    # Manage teams with skill thresholds and candidate recommendations
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
    # Manage event participants: create, assign to teams, view skills
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
        # Filter UI: lets the user narrow the participants table by status
        # and/or by legacy skill label. We use multiselect (not selectbox) so
        # an empty selection means "no filter" — a friendlier default than
        # forcing the user to pick "All" from a dropdown.
        st.subheader("Filter participants")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            # Statuses are a small fixed set, so we hardcode them rather
            # than computing unique() from the dataframe (avoids surprises
            # when the column is empty).
            status_filter = st.multiselect(
                "Filter by status",
                ["pending", "confirmed"],
                default=[],
                key=f"status_filter_{event_id}",
            )
        with filter_col2:
            skill_filter = st.multiselect(
                "Filter by skill (legacy label)",
                ["design", "engineering", "business", "other"],
                default=[],
                key=f"skill_filter_{event_id}",
            )

        # Apply filters cumulatively. Each empty list is a no-op so users
        # can combine them freely (e.g. "confirmed" + "design").
        filtered = participants.copy()
        if status_filter:
            filtered = filtered[filtered["status"].isin(status_filter)]
        if skill_filter:
            filtered = filtered[filtered["skill"].isin(skill_filter)]

        display_cols = [col for col in ["name", "team_name", "status", *ml.SKILL_COLUMNS] if col in filtered.columns]
        st.caption(f"Showing {len(filtered)} of {len(participants)} participants")
        st.dataframe(
            filtered[display_cols].fillna({"team_name": "Unassigned"}),
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
    # Visualize team sizes, skill distribution, and status breakdown
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
    show_recs = not unassigned.empty
    if not show_recs:
        st.info("All participants in this event are already assigned to teams. Showing gap analysis only.")

    for _, team_row in teams.iterrows():
        team_id = team_row["id"]
        team_name = team_row["name"]
        team_members = participants[participants["team_id"] == team_id]

        st.subheader(team_name)

        if team_members.empty:
            st.info("No members assigned to this team yet — recommendations skipped.")
            continue

        gap = ml.team_gap_vector(team_members)
        if gap.sum() == 0:
            st.success("Team is fully covered — no skill gaps.")
            continue

        # Top-3 weakest skills (largest gap values)
        gap_pairs = sorted(
            zip(ml.SKILL_COLUMNS, gap), key=lambda kv: kv[1], reverse=True
        )
        top_gaps = [f"{name} (gap={int(g)})" for name, g in gap_pairs[:3] if g > 0]
        st.markdown("**Weakest skills:** " + ", ".join(top_gaps))

        if not show_recs:
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


# ---------------------------------------------------------------------------
# Sidebar + routing
# ---------------------------------------------------------------------------
def render_app() -> None:
    # Route between pages based on sidebar selection
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
