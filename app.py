# Haupt-Streamlit-App: UI-Schicht, die zwischen den Seiten routet und an die Module auth, database und ml delegiert.

from datetime import date as date_cls

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import auth
import database as db
import ml


# ---------------------------------------------------------------------------
# Seiten-Setup
# ---------------------------------------------------------------------------
# Streamlit-Seite einmalig oben konfigurieren, damit Titel/Favicon bei
# jedem Rerun konsistent bleiben.
st.set_page_config(page_title="STARTCrew", layout="wide")


# ---------------------------------------------------------------------------
# Supabase-Client-Initialisierung
# ---------------------------------------------------------------------------
# Wir halten einen einzigen Supabase-Client in session_state, damit nicht
# bei jedem Streamlit-Rerun ein neuer erstellt wird. Fehlen die Zugangsdaten,
# zeigen wir eine freundliche Fehlermeldung und stoppen die App.
if "supabase" not in st.session_state:
    try:
        st.session_state["supabase"] = auth.init_supabase()
    except ValueError as e:
        st.error(str(e))
        st.stop()

supabase = st.session_state["supabase"]


# Deutsche Anzeigebezeichnungen für interne Status- und Skill-Werte
STATUS_LABELS_DE = {"pending": "Ausstehend", "confirmed": "Bestätigt"}
ROLE_OPTIONS = ["techniker", "volunteer", "team", "andere"]
ROLE_LABELS_DE = {
    "techniker": "Techniker",
    "volunteer": "Volunteer",
    "team": "Team",
    "andere": "Andere",
    # Rückwärtskompatibilität: ältere Daten verwenden noch die alten Slugs.
    "design": "Andere",
    "engineering": "Techniker",
    "business": "Team",
    "other": "Andere",
}
UNASSIGNED_LABEL = "Nicht zugewiesen"


# ---------------------------------------------------------------------------
# Login-Bildschirm
# ---------------------------------------------------------------------------
def render_login() -> None:
    # Login-/Registrierungs-Oberfläche für nicht angemeldete Nutzer rendern
    st.title("STARTCrew")

    login_tab, signup_tab = st.tabs(["Anmelden", "Registrieren"])

    # --- Tab "Anmelden" -------------------------------------------------
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("E-Mail", key="login_email")
            password = st.text_input("Passwort", type="password", key="login_password")
            submitted = st.form_submit_button("Anmelden")

        if submitted:
            try:
                auth.login(supabase, email, password)
                st.success("Erfolgreich angemeldet.")
                st.rerun()
            except Exception as e:
                # Supabase wirft unterschiedliche Exception-Typen (falsches Passwort,
                # Netzwerkfehler, nicht bestätigte E-Mail usw.) — eine Meldung deckt alles ab.
                st.error(f"Anmeldung fehlgeschlagen: {e}")

    # --- Tab "Registrieren" ---------------------------------------------
    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("E-Mail", key="signup_email")
            password = st.text_input(
                "Passwort (mind. 6 Zeichen)",
                type="password",
                key="signup_password",
            )
            submitted = st.form_submit_button("Konto erstellen")

        if submitted:
            try:
                response = auth.sign_up(supabase, email, password)
                if response.session is not None:
                    # E-Mail-Bestätigung ist deaktiviert — Nutzer ist direkt angemeldet.
                    st.success("Konto erstellt und angemeldet.")
                    st.rerun()
                else:
                    # Bestätigung erforderlich — Nutzer muss den Link in der E-Mail anklicken.
                    st.success(
                        "Konto erstellt. Bitte prüfen Sie Ihr Postfach auf eine "
                        "Bestätigungs-E-Mail und melden Sie sich anschließend an."
                    )
            except Exception as e:
                st.error(f"Registrierung fehlgeschlagen: {e}")


# ---------------------------------------------------------------------------
# Seite: Veranstaltungen
# ---------------------------------------------------------------------------
def page_events() -> None:
    # Veranstaltungen auflisten und Anlegen/Löschen ermöglichen
    st.header("Veranstaltungen")

    events = db.get_events(supabase)
    if events.empty:
        st.info("Noch keine Veranstaltungen. Erstellen Sie unten Ihre erste Veranstaltung.")
    else:
        st.dataframe(events, use_container_width=True)

    st.divider()
    st.subheader("Neue Veranstaltung erstellen")
    with st.form("new_event_form"):
        name = st.text_input("Name")
        event_date = st.date_input("Datum", value=date_cls.today())
        location = st.text_input("Ort")
        submitted = st.form_submit_button("Veranstaltung erstellen")

    if submitted:
        if not name:
            st.warning("Veranstaltungsname ist erforderlich.")
        else:
            db.create_event(supabase, name, event_date.isoformat(), location)
            st.success(f"Veranstaltung '{name}' erstellt.")
            st.rerun()

    if not events.empty:
        st.divider()
        st.subheader("Veranstaltung löschen")
        st.caption("Beim Löschen einer Veranstaltung werden auch alle zugehörigen Teams und Teilnehmer entfernt.")
        delete_options = {row["name"]: row["id"] for _, row in events.iterrows()}
        to_delete = st.selectbox(
            "Zu löschende Veranstaltung auswählen", list(delete_options.keys()), key="delete_event_select"
        )
        if st.button("Veranstaltung löschen", key="delete_event_btn"):
            db.delete_event(supabase, delete_options[to_delete])
            st.success(f"Veranstaltung '{to_delete}' gelöscht.")
            st.rerun()


# ---------------------------------------------------------------------------
# Seite: Teams
# ---------------------------------------------------------------------------
def page_teams() -> None:
    # Teams mit Skill-Schwellen und Kandidatenempfehlungen verwalten
    st.header("Teams")

    events = db.get_events(supabase)
    if events.empty:
        st.warning("Es existieren noch keine Veranstaltungen. Erstellen Sie zuerst eine auf der Seite Veranstaltungen.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Veranstaltung auswählen", list(event_options.keys()))
    event_id = event_options[event_label]

    teams = db.get_teams(supabase, event_id)
    if teams.empty:
        st.info("Noch keine Teams für diese Veranstaltung.")
    else:
        threshold_cols = [f"req_{s}" for s in ml.SKILL_COLUMNS if f"req_{s}" in teams.columns]
        display_cols = ["name"] + threshold_cols if "name" in teams.columns else list(teams.columns)
        st.dataframe(teams[display_cols], use_container_width=True)

    st.divider()
    st.subheader("Neues Team erstellen")
    with st.form("new_team_form"):
        team_name = st.text_input("Teamname")
        st.markdown(
            "**Minimale Skill-Schwellen** (0 = egal; Kandidaten, deren Skill "
            "unterhalb der Schwelle liegt, werden herausgefiltert)"
        )
        thresholds_input = {}
        cols = st.columns(3)
        for i, s in enumerate(ml.SKILL_COLUMNS):
            with cols[i % 3]:
                thresholds_input[f"req_{s}"] = st.slider(
                    ml.skill_label(s), 0, 5, 0, key=f"req_{s}_{event_id}"
                )
        submitted = st.form_submit_button("Team erstellen")

    if submitted:
        if not team_name:
            st.warning("Teamname ist erforderlich.")
        else:
            db.create_team(supabase, event_id, team_name, thresholds_input)
            st.success(f"Team '{team_name}' erstellt.")
            st.rerun()

    if not teams.empty:
        st.divider()
        st.subheader("Kandidaten empfehlen")
        team_options = {row["name"]: row["id"] for _, row in teams.iterrows()}
        team_label = st.selectbox(
            "Team auswählen",
            list(team_options.keys()),
            key=f"recommend_team_select_{event_id}",
        )
        team_row = teams[teams["id"] == team_options[team_label]].iloc[0]

        all_in_event = db.get_event_participants(supabase, event_id)
        if all_in_event.empty:
            st.info("Noch keine Teilnehmer in dieser Veranstaltung.")
        else:
            unassigned = all_in_event[all_in_event["team_id"].isna()]
            if unassigned.empty:
                st.info("Keine nicht zugewiesenen Teilnehmer für Empfehlungen verfügbar.")
            else:
                team_members = all_in_event[all_in_event["team_id"] == team_row["id"]]
                recs = ml.recommend_complementary(team_members, unassigned, k=5)
                if recs.empty:
                    st.info("Team ist vollständig abgedeckt – keine Skill-Lücken mehr zu füllen.")
                else:
                    show_cols = ["name", "distance", *ml.SKILL_COLUMNS]
                    rename_map = {"name": "Name", "distance": "Distanz"}
                    rename_map.update({s: ml.skill_label(s) for s in ml.SKILL_COLUMNS})
                    st.dataframe(
                        recs[show_cols]
                        .assign(distance=recs["distance"].round(2))
                        .rename(columns=rename_map),
                        use_container_width=True,
                    )
                    st.caption(
                        "Geringere Distanz = bessere Ergänzung zu den aktuellen Skill-Lücken des Teams. "
                        "Zuweisung über die Seite Teilnehmer."
                    )

        st.divider()
        st.subheader("Team löschen")
        st.caption("Beim Löschen eines Teams werden auch alle zugehörigen Teilnehmer entfernt.")
        team_delete_options = {row["name"]: row["id"] for _, row in teams.iterrows()}
        to_delete = st.selectbox(
            "Zu löschendes Team auswählen",
            list(team_delete_options.keys()),
            key=f"delete_team_select_{event_id}",
        )
        if st.button("Team löschen", key=f"delete_team_btn_{event_id}"):
            db.delete_team(supabase, team_delete_options[to_delete])
            st.success(f"Team '{to_delete}' gelöscht.")
            st.rerun()


# ---------------------------------------------------------------------------
# Seite: Teilnehmer
# ---------------------------------------------------------------------------
def page_participants() -> None:
    # Veranstaltungsteilnehmer verwalten: anlegen, Teams zuweisen, Skills ansehen
    st.header("Teilnehmer")

    events = db.get_events(supabase)
    if events.empty:
        st.warning("Es existieren noch keine Veranstaltungen. Erstellen Sie zuerst eine auf der Seite Veranstaltungen.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Veranstaltung auswählen", list(event_options.keys()))
    event_id = event_options[event_label]

    teams = db.get_teams(supabase, event_id)
    team_options = {row["name"]: row["id"] for _, row in teams.iterrows()}

    participants = db.get_event_participants(supabase, event_id)
    if participants.empty:
        st.info("Noch keine Teilnehmer in dieser Veranstaltung.")
    else:
        # Filter-UI: erlaubt das Eingrenzen der Teilnehmertabelle nach Status
        # und/oder nach Rolle. Wir nutzen multiselect (statt
        # selectbox), damit eine leere Auswahl "kein Filter" bedeutet – das
        # ist nutzerfreundlicher, als "Alle" aus einem Dropdown wählen zu müssen.
        st.subheader("Teilnehmer filtern")
        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            # Status-Werte sind ein kleiner, fester Satz — daher hartkodiert,
            # statt sie über unique() aus dem DataFrame zu ermitteln (vermeidet
            # Überraschungen bei leerer Spalte).
            status_filter = st.multiselect(
                "Nach Status filtern",
                ["pending", "confirmed"],
                default=[],
                format_func=lambda v: STATUS_LABELS_DE.get(v, v),
                key=f"status_filter_{event_id}",
            )
        with filter_col2:
            skill_filter = st.multiselect(
                "Nach Rolle filtern",
                ROLE_OPTIONS,
                default=[],
                format_func=lambda v: ROLE_LABELS_DE.get(v, v),
                key=f"skill_filter_{event_id}",
            )

        # Filter werden kumulativ angewandt. Eine leere Liste ist ein No-Op,
        # sodass Nutzer sie frei kombinieren können (z.B. "confirmed" + "techniker").
        filtered = participants.copy()
        if status_filter:
            filtered = filtered[filtered["status"].isin(status_filter)]
        if skill_filter:
            filtered = filtered[filtered["skill"].isin(skill_filter)]

        display_cols = [col for col in ["name", "team_name", "status", *ml.SKILL_COLUMNS] if col in filtered.columns]
        st.caption(f"{len(filtered)} von {len(participants)} Teilnehmern angezeigt")
        rename_map = {
            "name": "Name",
            "team_name": "Team",
            "status": "Status",
        }
        rename_map.update({s: ml.skill_label(s) for s in ml.SKILL_COLUMNS})
        st.dataframe(
            filtered[display_cols]
            .fillna({"team_name": UNASSIGNED_LABEL})
            .rename(columns=rename_map),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Teilnehmer hinzufügen")
    with st.form("new_participant_form"):
        name = st.text_input("Name")
        skill = st.selectbox(
            "Rolle",
            ROLE_OPTIONS,
            format_func=lambda v: ROLE_LABELS_DE.get(v, v),
        )
        status = st.selectbox(
            "Status",
            ["pending", "confirmed"],
            format_func=lambda v: STATUS_LABELS_DE.get(v, v),
        )
        team_choice = st.selectbox(
            "Team zuweisen",
            [UNASSIGNED_LABEL, *team_options.keys()],
        )
        st.markdown("**Skill-Bewertungen (1 = schwach, 5 = stark)**")
        skills_input = {}
        cols = st.columns(3)
        for i, s in enumerate(ml.SKILL_COLUMNS):
            with cols[i % 3]:
                skills_input[s] = st.slider(ml.skill_label(s), 1, 5, 3, key=f"new_{s}")
        submitted = st.form_submit_button("Teilnehmer hinzufügen")

    if submitted:
        if not name:
            st.warning("Teilnehmername ist erforderlich.")
        else:
            chosen_team_id = (
                team_options[team_choice]
                if team_choice != UNASSIGNED_LABEL
                else None
            )
            try:
                db.add_participant(
                    supabase,
                    event_id=event_id,
                    name=name,
                    skills=skills_input,
                    skill=skill,
                    status=status,
                    team_id=chosen_team_id,
                )
            except Exception as exc:
                st.error(f"Fehler beim Anlegen des Teilnehmers: {exc}")
            else:
                st.success(f"{name} zur Veranstaltung '{event_label}' hinzugefügt.")
                st.rerun()

    if not participants.empty:
        # Die {Label: id}-Map einmal aufbauen. Der pd.notna-Check ist nötig, weil
        # `team_name` bei nicht zugewiesenen Zeilen pd.NA ist, und `pd.NA or "..."`
        # einen TypeError wirft.
        participant_options = {}
        for _, row in participants.iterrows():
            team_name = row.get("team_name")
            label_team = team_name if pd.notna(team_name) else "nicht zugewiesen"
            participant_options[f"{row['name']} ({label_team})"] = row["id"]

        st.divider()
        st.subheader("Zuweisen / Zuweisung aufheben")
        chosen = st.selectbox(
            "Teilnehmer auswählen",
            list(participant_options.keys()),
            key=f"assign_pick_{event_id}",
        )
        chosen_id = participant_options[chosen]

        col_a, col_b = st.columns(2)
        with col_a:
            if team_options:
                target_team_label = st.selectbox(
                    "Team zuweisen",
                    list(team_options.keys()),
                    key=f"assign_team_{event_id}",
                )
                if st.button("Zuweisen", key=f"assign_btn_{event_id}"):
                    db.assign_participant_to_team(
                        supabase, chosen_id, team_options[target_team_label]
                    )
                    st.success(f"Zugewiesen zu '{target_team_label}'.")
                    st.rerun()
            else:
                st.caption("Erstellen Sie zuerst ein Team, um die Zuweisung zu aktivieren.")
        with col_b:
            if st.button("Zuweisung aufheben", key=f"unassign_btn_{event_id}"):
                db.unassign_participant(supabase, chosen_id)
                st.success("Teilnehmer ist jetzt nicht mehr zugewiesen.")
                st.rerun()

        st.divider()
        st.subheader("Teilnehmer entfernen")
        to_delete = st.selectbox(
            "Zu entfernenden Teilnehmer auswählen",
            list(participant_options.keys()),
            key=f"delete_participant_select_{event_id}",
        )
        if st.button("Teilnehmer entfernen", key=f"delete_participant_btn_{event_id}"):
            db.delete_participant(supabase, participant_options[to_delete])
            st.success(f"{to_delete} entfernt.")
            st.rerun()


# ---------------------------------------------------------------------------
# Seite: Dashboard
# ---------------------------------------------------------------------------
def page_dashboard() -> None:
    # Teamgrößen, Skill-Verteilung und Status-Übersicht visualisieren
    st.header("Dashboard")

    events = db.get_events(supabase)
    if events.empty:
        st.warning("Es existieren noch keine Veranstaltungen.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Veranstaltung auswählen", list(event_options.keys()))
    event_id = event_options[event_label]

    participants = db.get_all_participants_for_event(supabase, event_id)
    if participants.empty:
        st.info("Noch keine Teilnehmer für diese Veranstaltung.")
        return

    # --- Teamgrößen -----------------------------------------------------
    st.subheader("Teamgrößen")
    team_sizes = participants.groupby("team_name").size().rename("Mitglieder")
    st.bar_chart(team_sizes)

    # --- Rollen-Verteilung ----------------------------------------------
    st.subheader("Rollen-Verteilung")
    skill_counts = (
        participants["skill"]
        .map(lambda v: ROLE_LABELS_DE.get(v, v) if pd.notna(v) else "Unbekannt")
        .fillna("Unbekannt")
        .value_counts()
    )
    fig, ax = plt.subplots()
    ax.pie(skill_counts.values, labels=skill_counts.index, autopct="%1.0f%%")
    ax.set_aspect("equal")
    st.pyplot(fig)

    # --- Skill-Lücken-Netzdiagramm --------------------------------------
    st.subheader("Skill-Lücke – Anforderungen vs. Team-Durchschnitt")
    teams_df = db.get_teams(supabase, event_id)
    team_choices = participants["team_name"].dropna().unique().tolist()
    if not team_choices:
        st.info("Noch keine Teams mit Mitgliedern.")
        return

    team_name = st.selectbox("Team", team_choices, key="radar_team")
    team_id = teams_df.loc[teams_df["name"] == team_name, "id"].iloc[0]
    team_row = teams_df[teams_df["id"] == team_id].iloc[0]
    team_members = participants[participants["team_id"] == team_id]

    # Defensive gegen noch nicht ausgeführte DB-Migrationen: fehlende Skill-/Req-Spalten
    # werden neutral behandelt (Durchschnitt=3, Anforderung=0), damit das Netzdiagramm trotzdem rendert.
    avg = np.array(
        [float(team_members[s].mean()) if s in team_members.columns else 3.0
         for s in ml.SKILL_COLUMNS],
        dtype=float,
    )
    req = np.array(
        [float(team_row[f"req_{s}"]) if f"req_{s}" in team_row.index else 0.0
         for s in ml.SKILL_COLUMNS],
        dtype=float,
    )
    angles = np.linspace(0, 2 * np.pi, len(ml.SKILL_COLUMNS), endpoint=False)
    angles_loop = np.concatenate([angles, angles[:1]])
    avg_loop = np.concatenate([avg, avg[:1]])
    req_loop = np.concatenate([req, req[:1]])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(angles_loop, req_loop, color="tab:red", label="Anforderung")
    ax.fill(angles_loop, req_loop, color="tab:red", alpha=0.1)
    ax.plot(angles_loop, avg_loop, color="tab:blue", label="Team-Durchschnitt")
    ax.fill(angles_loop, avg_loop, color="tab:blue", alpha=0.2)
    ax.set_xticks(angles)
    ax.set_xticklabels([ml.skill_label(s) for s in ml.SKILL_COLUMNS])
    ax.set_ylim(0, 5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Seite: ML-Insights
# ---------------------------------------------------------------------------
def page_ml_insights() -> None:
    st.header("ML-Insights – Komplementär-Fit-Empfehler (kNN)")
    st.caption(
        "Für jedes Team der ausgewählten Veranstaltung ermitteln wir die Skills, in denen das Team am schwächsten ist "
        "(die Lücke), und nutzen scikit-learns k-Nearest-Neighbors, um nicht zugewiesene Teilnehmer vorzuschlagen, "
        "deren Stärken diese Lücke am besten füllen."
    )

    events = db.get_events(supabase)
    if events.empty:
        st.warning("Es existieren noch keine Veranstaltungen.")
        return

    event_options = {row["name"]: row["id"] for _, row in events.iterrows()}
    event_label = st.selectbox("Veranstaltung auswählen", list(event_options.keys()))
    event_id = event_options[event_label]

    teams = db.get_teams(supabase, event_id)
    if teams.empty:
        st.info("Noch keine Teams in dieser Veranstaltung.")
        return

    participants = db.get_all_participants_for_event(supabase, event_id)
    if participants.empty:
        st.info("Noch keine Teilnehmer in dieser Veranstaltung – fügen Sie welche auf der Seite Teilnehmer hinzu.")
        return

    unassigned = participants[participants["team_id"].isna()]
    show_recs = not unassigned.empty
    if not show_recs:
        st.info("Alle Teilnehmer dieser Veranstaltung sind bereits Teams zugewiesen. Es wird nur die Lückenanalyse angezeigt.")

    for _, team_row in teams.iterrows():
        team_id = team_row["id"]
        team_name = team_row["name"]
        team_members = participants[participants["team_id"] == team_id]

        st.subheader(team_name)

        if team_members.empty:
            st.info("Noch keine Mitglieder diesem Team zugewiesen – Empfehlungen werden übersprungen.")
            continue

        gap = ml.team_gap_vector(team_members)
        if gap.sum() == 0:
            st.success("Team ist vollständig abgedeckt – keine Skill-Lücken.")
            continue

        # Top-3 schwächste Skills (größte Lückenwerte)
        gap_pairs = sorted(
            zip(ml.SKILL_COLUMNS, gap), key=lambda kv: kv[1], reverse=True
        )
        top_gaps = [f"{ml.skill_label(name)} (Lücke={int(g)})" for name, g in gap_pairs[:3] if g > 0]
        st.markdown("**Schwächste Skills:** " + ", ".join(top_gaps))

        if not show_recs:
            continue

        recs = ml.recommend_complementary(team_members, unassigned, k=5)
        if recs.empty:
            st.info("Keine geeigneten nicht zugewiesenen Kandidaten.")
            continue

        show_cols = ["name", "distance", "gap_score", *ml.SKILL_COLUMNS]
        rename_map = {
            "name": "Name",
            "distance": "Distanz",
            "gap_score": "Lücken-Score",
        }
        rename_map.update({s: ml.skill_label(s) for s in ml.SKILL_COLUMNS})
        st.dataframe(
            recs[show_cols]
            .assign(
                distance=recs["distance"].round(2),
                gap_score=recs["gap_score"].round(2),
            )
            .rename(columns=rename_map),
            use_container_width=True,
        )

    st.divider()
    st.caption(
        f"Methode: lückengewichteter euklidischer kNN über {len(ml.SKILL_COLUMNS)}-dimensionale Skill-Vektoren. "
        "Geringere Distanz = bessere Ergänzung; der Lücken-Score ist das Skalarprodukt der Skills des "
        "Kandidaten mit dem Lücken-Vektor des Teams (höher = deckt mehr Lücke ab)."
    )


# ---------------------------------------------------------------------------
# Seitenleiste + Routing
# ---------------------------------------------------------------------------
def render_app() -> None:
    # Routing zwischen den Seiten anhand der Auswahl in der Seitenleiste
    user = st.session_state.get("user")
    user_email = getattr(user, "email", "user") if user else "user"

    st.sidebar.title("STARTCrew")
    st.sidebar.caption(f"Angemeldet als: {user_email}")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Veranstaltungen",
            "Teams",
            "Teilnehmer",
            "Dashboard",
            "ML-Insights",
        ],
    )

    st.sidebar.divider()
    if st.sidebar.button("Abmelden"):
        auth.logout(supabase)
        st.rerun()

    # Dispatch — flache, leicht erweiterbare Zuordnung
    pages = {
        "Veranstaltungen": page_events,
        "Teams": page_teams,
        "Teilnehmer": page_participants,
        "Dashboard": page_dashboard,
        "ML-Insights": page_ml_insights,
    }
    pages[page]()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------
if auth.get_session() is None:
    render_login()
else:
    render_app()
