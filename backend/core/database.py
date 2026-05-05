import sqlite3
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional

DB_PATH = "app.db"

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_db() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            user_id TEXT UNIQUE,
            password_hash TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        _ensure_user_columns(db)

        db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            email TEXT NOT NULL,
            resume_hash TEXT,
            jd_hash TEXT,
            score REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            explanation TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        _ensure_history_columns(db)

        db.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_user_id ON chat_history(user_id)")

        db.execute("""
        CREATE TABLE IF NOT EXISTS revoked_tokens (
            jti TEXT PRIMARY KEY,
            expires_at DATETIME NOT NULL,
            revoked_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS refresh_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            device_label TEXT,
            expires_at DATETIME NOT NULL,
            revoked_at DATETIME,
            replaced_by TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_refresh_sessions_user_id ON refresh_sessions(user_id)")

        db.execute("""
        CREATE TABLE IF NOT EXISTS embedding_cache (
            cache_key TEXT PRIMARY KEY,
            text_hash TEXT NOT NULL,
            model_name TEXT NOT NULL,
            backend TEXT NOT NULL,
            dimensions INTEGER NOT NULL,
            vector_json TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_embedding_cache_text_hash ON embedding_cache(text_hash)")

def _ensure_user_columns(db):
    columns = {
        row[1]
        for row in db.execute("PRAGMA table_info(users)").fetchall()
    }
    if "user_id" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN user_id TEXT")
    if "password_hash" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    if "created_at" not in columns:
        db.execute("ALTER TABLE users ADD COLUMN created_at DATETIME")

    rows = db.execute("SELECT email FROM users WHERE user_id IS NULL OR user_id = ''").fetchall()
    for (email,) in rows:
        db.execute("UPDATE users SET user_id=? WHERE email=?", (str(uuid.uuid4()), email))
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_user_id ON users(user_id)")

def _ensure_history_columns(db):
    columns = {
        row[1]
        for row in db.execute("PRAGMA table_info(history)").fetchall()
    }
    if "user_id" not in columns:
        db.execute("ALTER TABLE history ADD COLUMN user_id TEXT")
    db.execute("CREATE INDEX IF NOT EXISTS idx_history_user_id ON history(user_id)")

def upsert_user(email: str, role: str):
    init_db()
    with get_db() as db:
        db.execute("""
        INSERT INTO users (email, role, user_id)
        VALUES (?, ?, ?)
        ON CONFLICT(email)
        DO UPDATE SET role=users.role
        """, (email, role, str(uuid.uuid4())))

def create_auth_user(email: str, password_hash: str, role: str = "user") -> Dict:
    init_db()
    with get_db() as db:
        user_id = str(uuid.uuid4())
        db.execute("""
        INSERT INTO users (email, role, user_id, password_hash)
        VALUES (?, ?, ?, ?)
        """, (email, role, user_id, password_hash))
        return {"user_id": user_id, "email": email, "role": role}

def get_user_by_email(email: str) -> Optional[Dict]:
    init_db()
    with get_db() as db:
        row = db.execute("""
        SELECT user_id, email, role, password_hash
        FROM users
        WHERE email=?
        """, (email,)).fetchone()
    if not row:
        return None
    return {
        "user_id": row[0],
        "email": row[1],
        "role": row[2],
        "password_hash": row[3],
    }

def get_user_by_id(user_id: str) -> Optional[Dict]:
    init_db()
    with get_db() as db:
        row = db.execute("""
        SELECT user_id, email, role
        FROM users
        WHERE user_id=?
        """, (user_id,)).fetchone()
    if not row:
        return None
    return {"user_id": row[0], "email": row[1], "role": row[2]}

def save_history(
    email: str,
    resume_hash: str,
    jd_hash: str,
    score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    explanation: str,
    user_id: Optional[str] = None
):
    init_db()
    with get_db() as db:
        db.execute("""
        INSERT INTO history (
            user_id, email, resume_hash, jd_hash,
            score, matched_skills, missing_skills, explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            email,
            resume_hash,
            jd_hash,
            score,
            ",".join(matched_skills),
            ",".join(missing_skills),
            explanation
        ))

def get_history(email: str) -> List[Dict]:
    init_db()
    with get_db() as db:
        rows = db.execute("""
        SELECT score, matched_skills, missing_skills, explanation, created_at
        FROM history
        WHERE email=?
        ORDER BY created_at DESC
        """, (email,)).fetchall()

    return [
        {
            "score": r[0],
            "matched": r[1].split(",") if r[1] else [],
            "missing": r[2].split(",") if r[2] else [],
            "explanation": r[3],
            "timestamp": r[4]
        }
        for r in rows
    ]

def get_history_by_user_id(user_id: str) -> List[Dict]:
    init_db()
    with get_db() as db:
        rows = db.execute("""
        SELECT score, matched_skills, missing_skills, explanation, created_at
        FROM history
        WHERE user_id=?
        ORDER BY created_at DESC
        """, (user_id,)).fetchall()

    return [
        {
            "score": r[0],
            "matched": r[1].split(",") if r[1] else [],
            "missing": r[2].split(",") if r[2] else [],
            "explanation": r[3],
            "timestamp": r[4]
        }
        for r in rows
    ]

def save_chat_message(user_id: str, role: str, content: str):
    init_db()
    with get_db() as db:
        db.execute("""
        INSERT INTO chat_history (user_id, role, content)
        VALUES (?, ?, ?)
        """, (user_id, role, content))

def get_chat_history_by_user_id(user_id: str, limit: int = 20) -> List[Dict]:
    init_db()
    with get_db() as db:
        rows = db.execute("""
        SELECT role, content, created_at
        FROM chat_history
        WHERE user_id=?
        ORDER BY created_at DESC
        LIMIT ?
        """, (user_id, limit)).fetchall()

    return [
        {
            "role": r[0],
            "content": r[1],
            "timestamp": r[2],
        }
        for r in reversed(rows)
    ]

def revoke_token(jti: str, expires_at: datetime):
    init_db()
    with get_db() as db:
        db.execute("""
        INSERT OR IGNORE INTO revoked_tokens (jti, expires_at)
        VALUES (?, ?)
        """, (jti, expires_at.isoformat()))

def is_token_revoked(jti: str) -> bool:
    init_db()
    with get_db() as db:
        row = db.execute("""
        SELECT 1
        FROM revoked_tokens
        WHERE jti=?
        """, (jti,)).fetchone()
    return row is not None

def save_refresh_session(
    session_id: str,
    user_id: str,
    token_hash: str,
    expires_at: datetime,
    device_label: Optional[str] = None,
):
    init_db()
    with get_db() as db:
        db.execute("""
        INSERT INTO refresh_sessions (
            session_id, user_id, token_hash, device_label, expires_at
        ) VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_id, token_hash, device_label, expires_at.isoformat()))

def get_refresh_session_by_hash(token_hash: str) -> Optional[Dict]:
    init_db()
    with get_db() as db:
        row = db.execute("""
        SELECT session_id, user_id, token_hash, device_label, expires_at, revoked_at, replaced_by
        FROM refresh_sessions
        WHERE token_hash=?
        """, (token_hash,)).fetchone()
    if not row:
        return None
    return {
        "session_id": row[0],
        "user_id": row[1],
        "token_hash": row[2],
        "device_label": row[3],
        "expires_at": row[4],
        "revoked_at": row[5],
        "replaced_by": row[6],
    }

def rotate_refresh_session(old_session_id: str, new_session_id: str):
    init_db()
    with get_db() as db:
        db.execute("""
        UPDATE refresh_sessions
        SET revoked_at=CURRENT_TIMESTAMP, replaced_by=?
        WHERE session_id=? AND revoked_at IS NULL
        """, (new_session_id, old_session_id))

def revoke_refresh_session(session_id: str):
    init_db()
    with get_db() as db:
        db.execute("""
        UPDATE refresh_sessions
        SET revoked_at=CURRENT_TIMESTAMP
        WHERE session_id=? AND revoked_at IS NULL
        """, (session_id,))

def revoke_user_refresh_sessions(user_id: str):
    init_db()
    with get_db() as db:
        db.execute("""
        UPDATE refresh_sessions
        SET revoked_at=CURRENT_TIMESTAMP
        WHERE user_id=? AND revoked_at IS NULL
        """, (user_id,))

def is_iso_datetime_expired(value: str) -> bool:
    normalized = value.replace("Z", "+00:00")
    expires_at = datetime.fromisoformat(normalized)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    return expires_at <= datetime.now(timezone.utc)

def get_cached_embedding(cache_key: str) -> Optional[Dict]:
    init_db()
    with get_db() as db:
        row = db.execute("""
        SELECT vector_json, backend, dimensions, created_at
        FROM embedding_cache
        WHERE cache_key=?
        """, (cache_key,)).fetchone()
    if not row:
        return None
    return {
        "vector_json": row[0],
        "backend": row[1],
        "dimensions": row[2],
        "created_at": row[3],
    }

def save_cached_embedding(
    cache_key: str,
    text_hash: str,
    model_name: str,
    backend: str,
    dimensions: int,
    vector_json: str,
):
    init_db()
    with get_db() as db:
        db.execute("""
        INSERT OR REPLACE INTO embedding_cache (
            cache_key, text_hash, model_name, backend, dimensions, vector_json
        ) VALUES (?, ?, ?, ?, ?, ?)
        """, (cache_key, text_hash, model_name, backend, dimensions, vector_json))
