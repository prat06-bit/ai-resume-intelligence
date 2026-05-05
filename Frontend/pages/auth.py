import os
import secrets
import sys

import streamlit as st
from pydantic import ValidationError

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.api.auth_service import LoginRequest, SignupRequest, decode_token, login, logout_token, signup
from backend.core.database import get_user_by_id, init_db


def _ensure_local_jwt_secret() -> None:
    if os.getenv("JWT_SECRET_KEY"):
        return
    if "local_jwt_secret" not in st.session_state:
        st.session_state.local_jwt_secret = secrets.token_urlsafe(48)
    os.environ["JWT_SECRET_KEY"] = st.session_state.local_jwt_secret


def _set_auth_session(result: dict, email: str) -> None:
    token = result["data"]["access_token"]
    payload = decode_token(token)
    user = get_user_by_id(payload["user_id"])
    st.session_state.auth_token = token
    st.session_state.refresh_token = result["data"]["refresh_token"]
    st.session_state.session_id = result["data"]["session_id"]
    st.session_state.auth_user = {
        "user_id": payload["user_id"],
        "email": user["email"] if user else email.lower(),
        "role": user["role"] if user else "user",
    }
    st.session_state.profile_email = st.session_state.auth_user["email"]


def _auth_form(mode: str) -> None:
    is_signup = mode == "Create Account"
    with st.form(f"{mode.lower().replace(' ', '_')}_form"):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button(mode, use_container_width=True)

    if not submitted:
        return

    try:
        payload = SignupRequest(email=email, password=password) if is_signup else LoginRequest(email=email, password=password)
    except ValidationError:
        st.error("Invalid email or password format")
        return

    result = signup(payload) if is_signup else login(payload)
    if result["status"] == "error":
        st.error(result["message"])
        return

    _set_auth_session(result, email)
    st.success(result["message"])
    st.switch_page("pages/analyzer.py")


st.set_page_config(page_title="Sign In - AI Resume Intelligence", layout="centered")
init_db()
_ensure_local_jwt_secret()

st.title("AI Resume Intelligence")
st.caption("Sign in to save analysis history and keep resume feedback tied to your account.")

if st.session_state.get("auth_user"):
    user = st.session_state.auth_user
    st.success(f"Signed in as {user['email']}")
    if st.button("Continue to analyzer", use_container_width=True):
        st.switch_page("pages/analyzer.py")
    if st.button("Sign out", use_container_width=True):
        try:
            logout_token(st.session_state.get("auth_token", ""))
        except Exception:
            pass
        for key in ("auth_token", "refresh_token", "session_id", "auth_user", "profile_email"):
            st.session_state.pop(key, None)
        st.rerun()
    st.stop()

tabs = st.tabs(["Sign In", "Create Account"])
with tabs[0]:
    _auth_form("Sign In")
with tabs[1]:
    st.caption("Password must be at least 8 characters and include letters and numbers.")
    _auth_form("Create Account")
