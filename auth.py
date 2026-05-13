# Supabase-Authentifizierungshelfer: Login, Registrierung, Logout und Session-Verwaltung via st.session_state

import os

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client


def init_supabase() -> Client:
    # Supabase-Client aus .env oder st.secrets (Streamlit Cloud) initialisieren.
    # Zuerst .env versuchen, anschließend kann st.secrets überschreiben (Streamlit Cloud).
    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    try:
        url = st.secrets.get("SUPABASE_URL", url)
        key = st.secrets.get("SUPABASE_KEY", key)
    except Exception:
        pass

    if not url or not key:
        raise ValueError(
            "SUPABASE_URL oder SUPABASE_KEY fehlt. "
            "Kopieren Sie .env.example nach .env und tragen Sie Ihre Projekt-Zugangsdaten ein."
        )

    return create_client(url, key)


def login(supabase: Client, email: str, password: str):
    # Nutzer authentifizieren und Session in st.session_state cachen, damit sie Seiten-Reloads übersteht
    response = supabase.auth.sign_in_with_password(
        {"email": email, "password": password}
    )

    # Session cachen, damit andere Seiten den angemeldeten Nutzer sehen
    st.session_state["session"] = response.session
    st.session_state["user"] = response.user
    return response


def sign_up(supabase: Client, email: str, password: str):
    # Nutzer registrieren; Session cachen, falls E-Mail-Bestätigung deaktiviert ist
    response = supabase.auth.sign_up({"email": email, "password": password})

    # Automatische Anmeldung, wenn E-Mail-Bestätigung deaktiviert ist
    if response.session is not None:
        st.session_state["session"] = response.session
        st.session_state["user"] = response.user
    return response


def logout(supabase: Client) -> None:
    # Session serverseitig invalidieren und lokalen Cache leeren
    supabase.auth.sign_out()

    # Session leeren, damit die Oberfläche wieder den Login-Bildschirm anzeigt
    for key in ("session", "user"):
        if key in st.session_state:
            del st.session_state[key]


def get_session():
    # Liefert die aktuelle Session aus dem Cache oder None, wenn nicht angemeldet
    return st.session_state.get("session")
