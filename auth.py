# Supabase authentication helpers: login, signup, logout, and session management via st.session_state

import os

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client


def init_supabase() -> Client:
    # Initialize Supabase client from .env or st.secrets (Streamlit Cloud)
    # Try .env first, then let st.secrets override (Streamlit Cloud).
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
            "Missing SUPABASE_URL or SUPABASE_KEY. "
            "Copy .env.example to .env and add your project credentials."
        )

    return create_client(url, key)


def login(supabase: Client, email: str, password: str):
    # Authenticate user and cache session in st.session_state for page reloads
    response = supabase.auth.sign_in_with_password(
        {"email": email, "password": password}
    )

    # Cache session so other pages see the authenticated user
    st.session_state["session"] = response.session
    st.session_state["user"] = response.user
    return response


def sign_up(supabase: Client, email: str, password: str):
    # Register user; cache session if email confirmation is disabled
    response = supabase.auth.sign_up({"email": email, "password": password})

    # Auto-login if email confirmation is disabled
    if response.session is not None:
        st.session_state["session"] = response.session
        st.session_state["user"] = response.user
    return response


def logout(supabase: Client) -> None:
    # Invalidate session on server and clear local cache
    supabase.auth.sign_out()

    # Clear session so UI shows login screen again
    for key in ("session", "user"):
        if key in st.session_state:
            del st.session_state[key]


def get_session():
    # Return current session from cache, or None if logged out
    return st.session_state.get("session")
