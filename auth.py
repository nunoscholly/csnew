"""
auth.py
-------
Supabase authentication helpers for the Event Team Manager app.

This module wraps the supabase-py client so the rest of the app can
log users in, log them out, and check the current session without
worrying about how Supabase Auth works underneath.

Beginner note:
    Supabase Auth gives us email + password login out of the box.
    We just call the client methods and store the resulting session
    inside Streamlit's st.session_state so it survives page reloads.
"""

import os

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client


def init_supabase() -> Client:
    """
    Load environment variables from .env and return a Supabase client.

    Returns:
        Client: a configured supabase-py client ready to make API calls.

    Raises:
        ValueError: if SUPABASE_URL or SUPABASE_KEY is missing.
    """
    # Read variables from a local .env file (if present).
    load_dotenv()

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    # Fail loudly if the credentials are missing so the user knows to
    # copy .env.example to .env and fill it in.
    if not url or not key:
        raise ValueError(
            "Missing SUPABASE_URL or SUPABASE_KEY. "
            "Copy .env.example to .env and add your project credentials."
        )

    return create_client(url, key)


def login(supabase: Client, email: str, password: str):
    """
    Log a user in with email and password using Supabase Auth.

    Args:
        supabase: an initialised supabase-py client.
        email: the user's email address.
        password: the user's password.

    Returns:
        The auth response object from Supabase, which contains the
        session and user info on success.
    """
    # sign_in_with_password takes a dict with the credentials.
    response = supabase.auth.sign_in_with_password(
        {"email": email, "password": password}
    )

    # Persist the session in Streamlit so other pages can see it.
    st.session_state["session"] = response.session
    st.session_state["user"] = response.user
    return response


def sign_up(supabase: Client, email: str, password: str):
    """
    Register a new user with email + password using Supabase Auth.

    Args:
        supabase: an initialised supabase-py client.
        email: the new user's email address.
        password: the new user's password (Supabase enforces a minimum length).

    Returns:
        The auth response object. If the project has email confirmation
        disabled, response.session will be populated and we log the user
        in immediately. Otherwise response.session will be None and the
        user must confirm via the email link before logging in.
    """
    response = supabase.auth.sign_up({"email": email, "password": password})

    # If confirmation is disabled the user is already signed in — cache it.
    if response.session is not None:
        st.session_state["session"] = response.session
        st.session_state["user"] = response.user
    return response


def logout(supabase: Client) -> None:
    """
    Log the current user out and clear the Streamlit session state.

    Args:
        supabase: an initialised supabase-py client.
    """
    # Tell Supabase to invalidate the session on the server.
    supabase.auth.sign_out()

    # Remove cached session info so the UI returns to the login screen.
    for key in ("session", "user"):
        if key in st.session_state:
            del st.session_state[key]


def get_session():
    """
    Return the current logged-in session (or None if not logged in).

    Returns:
        The Supabase session object stored in st.session_state, or None.
    """
    return st.session_state.get("session")
