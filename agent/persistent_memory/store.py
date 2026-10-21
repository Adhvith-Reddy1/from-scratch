"""
SQLite-backed storage for facts and episodes.

facts  — durable key/value pairs about the user ("user prefers concise answers")
episodes — one row per completed conversation: summary text + embedding vector
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call on every startup."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                key        TEXT    NOT NULL UNIQUE,
                value      TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                summary    TEXT    NOT NULL,
                embedding  TEXT    NOT NULL,   -- JSON array of floats
                turn_count INTEGER NOT NULL,
                created_at TEXT    NOT NULL
            );
        """)


# ── facts ────────────────────────────────────────────────────────────────────

def upsert_fact(key: str, value: str) -> None:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        conn.execute("""
            INSERT INTO facts (key, value, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """, (key, value, now, now))


def get_all_facts() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute("SELECT key, value FROM facts ORDER BY updated_at DESC").fetchall()
    return [{"key": r["key"], "value": r["value"]} for r in rows]


def delete_fact(key: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM facts WHERE key = ?", (key,))


# ── episodes ─────────────────────────────────────────────────────────────────

def save_episode(summary: str, embedding: list[float], turn_count: int) -> int:
    now = datetime.utcnow().isoformat()
    with _connect() as conn:
        cur = conn.execute("""
            INSERT INTO episodes (summary, embedding, turn_count, created_at)
            VALUES (?, ?, ?, ?)
        """, (summary, json.dumps(embedding), turn_count, now))
        return cur.lastrowid


def get_all_episodes() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, summary, embedding, turn_count, created_at FROM episodes ORDER BY created_at DESC"
        ).fetchall()
    return [
        {
            "id":         r["id"],
            "summary":    r["summary"],
            "embedding":  json.loads(r["embedding"]),
            "turn_count": r["turn_count"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
