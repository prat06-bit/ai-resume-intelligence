from __future__ import annotations

import os
import re
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict

import jwt
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field

from backend.core.database import (
    create_auth_user,
    get_user_by_email,
    get_user_by_id,
    get_refresh_session_by_hash,
    init_db,
    is_iso_datetime_expired,
    is_token_revoked,
    rotate_refresh_session,
    revoke_refresh_session,
    revoke_user_refresh_sessions,
    revoke_token,
    save_refresh_session,
)


JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "30"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=32)


class LogoutRequest(BaseModel):
    refresh_token: str | None = None
    logout_all_devices: bool = False


def json_success(message: str, data: Dict | None = None) -> Dict:
    return {
        "status": "success",
        "message": message,
        "data": data or {},
    }


def json_error(message: str) -> Dict:
    return {
        "status": "error",
        "message": message,
    }


def _jwt_secret() -> str:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=json_error("Authentication service is not configured"),
        )
    return secret


def validate_password_strength(password: str) -> bool:
    if len(password) < 8:
        return False
    has_letter = bool(re.search(r"[A-Za-z]", password))
    has_number = bool(re.search(r"\d", password))
    return has_letter and has_number


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return pwd_context.verify(password, password_hash)


def create_access_token(user_id: str) -> tuple[str, datetime]:
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": user_id,
        "user_id": user_id,
        "jti": str(uuid.uuid4()),
        "exp": expires_at,
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)
    return token, expires_at


def _hash_refresh_token(refresh_token: str) -> str:
    import hashlib

    return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()


def create_refresh_token(user_id: str, device_label: str | None = None) -> tuple[str, Dict]:
    session_id = str(uuid.uuid4())
    refresh_token = secrets.token_urlsafe(48)
    expires_at = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    save_refresh_session(
        session_id=session_id,
        user_id=user_id,
        token_hash=_hash_refresh_token(refresh_token),
        expires_at=expires_at,
        device_label=device_label,
    )
    return refresh_token, {
        "session_id": session_id,
        "expires_at": expires_at,
    }


def _token_bundle(user_id: str, device_label: str | None = None) -> Dict:
    access_token, access_expires_at = create_access_token(user_id)
    refresh_token, refresh_session = create_refresh_token(user_id, device_label=device_label)
    return {
        "user_id": user_id,
        "access_token": access_token,
        "token_type": "bearer",
        "expires_at": access_expires_at.isoformat(),
        "refresh_token": refresh_token,
        "refresh_expires_at": refresh_session["expires_at"].isoformat(),
        "session_id": refresh_session["session_id"],
    }


def signup(payload: SignupRequest) -> Dict:
    email = payload.email.lower()
    if not validate_password_strength(payload.password):
        return json_error("Password must be at least 8 characters and include letters and numbers")

    try:
        user = create_auth_user(email=email, password_hash=hash_password(payload.password))
    except sqlite3.IntegrityError:
        return json_error("User already exists")

    return json_success(
        "User created successfully",
        _token_bundle(user["user_id"]),
    )


def login(payload: LoginRequest) -> Dict:
    user = get_user_by_email(payload.email.lower())
    if not user or not user.get("password_hash"):
        return json_error("Invalid credentials")

    if not verify_password(payload.password, user["password_hash"]):
        return json_error("Invalid credentials")

    return json_success(
        "Login successful",
        _token_bundle(user["user_id"]),
    )


def refresh_access_token(refresh_token: str) -> Dict:
    token_hash = _hash_refresh_token(refresh_token)
    session = get_refresh_session_by_hash(token_hash)
    if not session or session.get("revoked_at") or is_iso_datetime_expired(session["expires_at"]):
        return json_error("Unauthorized access")

    refresh_token_new, refresh_session = create_refresh_token(
        session["user_id"],
        device_label=session.get("device_label"),
    )
    rotate_refresh_session(session["session_id"], refresh_session["session_id"])
    access_token, access_expires_at = create_access_token(session["user_id"])
    return json_success(
        "Token refreshed",
        {
            "user_id": session["user_id"],
            "access_token": access_token,
            "token_type": "bearer",
            "expires_at": access_expires_at.isoformat(),
            "refresh_token": refresh_token_new,
            "refresh_expires_at": refresh_session["expires_at"].isoformat(),
            "session_id": refresh_session["session_id"],
        },
    )


def decode_token(token: str) -> Dict:
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=json_error("Unauthorized access"),
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=json_error("Unauthorized access"),
        )
    if is_token_revoked(payload.get("jti", "")):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=json_error("Unauthorized access"),
        )
    return payload


def logout_token(token: str) -> Dict:
    payload = decode_token(token)
    exp = payload.get("exp")
    expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
    revoke_token(payload["jti"], expires_at)
    return json_success("Logout successful")


def logout_session(token: str, refresh_token: str | None = None, logout_all_devices: bool = False) -> Dict:
    payload = decode_token(token)
    exp = payload.get("exp")
    revoke_token(payload["jti"], datetime.fromtimestamp(exp, tz=timezone.utc))
    if logout_all_devices:
        revoke_user_refresh_sessions(payload["user_id"])
    elif refresh_token:
        session = get_refresh_session_by_hash(_hash_refresh_token(refresh_token))
        if session and session["user_id"] == payload["user_id"]:
            revoke_refresh_session(session["session_id"])
    return json_success("Logout successful")


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> Dict:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=json_error("Unauthorized access"),
        )

    payload = decode_token(credentials.credentials)
    user_id = payload.get("user_id") or payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=json_error("Unauthorized access"),
        )

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=json_error("Unauthorized access"),
        )
    return user


app = FastAPI(title="Resume Job Matcher Auth API")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.post("/auth/signup")
def signup_route(payload: SignupRequest):
    result = signup(payload)
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result,
        )
    return result


@app.post("/auth/login")
def login_route(payload: LoginRequest):
    result = login(payload)
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result,
        )
    return result


@app.get("/auth/me")
def me_route(current_user: Dict = Depends(get_current_user)):
    return json_success(
        "Authenticated user",
        {
            "user_id": current_user["user_id"],
            "email": current_user["email"],
            "role": current_user["role"],
        },
    )


@app.post("/auth/logout")
def logout_route(
    payload: LogoutRequest | None = None,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
):
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=json_error("Unauthorized access"),
        )
    return logout_session(
        credentials.credentials,
        refresh_token=payload.refresh_token if payload else None,
        logout_all_devices=payload.logout_all_devices if payload else False,
    )


@app.post("/auth/refresh")
def refresh_route(payload: RefreshRequest):
    result = refresh_access_token(payload.refresh_token)
    if result["status"] == "error":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=result,
        )
    return result
