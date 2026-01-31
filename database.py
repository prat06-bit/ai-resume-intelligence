import sqlite3
from typing import List, Dict

DB_PATH = "app.db"

def get_db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_db() as db:
        db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            role TEXT NOT NULL
        )
        """)

        db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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

def upsert_user(email: str, role: str):
    with get_db() as db:
        db.execute("""
        INSERT INTO users (email, role)
        VALUES (?, ?)
        ON CONFLICT(email)
        DO UPDATE SET role=excluded.role
        """, (email, role))

def save_history(
    email: str,
    resume_hash: str,
    jd_hash: str,
    score: float,
    matched_skills: List[str],
    missing_skills: List[str],
    explanation: str
):
    with get_db() as db:
        db.execute("""
        INSERT INTO history (
            email, resume_hash, jd_hash,
            score, matched_skills, missing_skills, explanation
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            email,
            resume_hash,
            jd_hash,
            score,
            ",".join(matched_skills),
            ",".join(missing_skills),
            explanation
        ))

def get_history(email: str) -> List[Dict]:
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
