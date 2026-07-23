"""
db.py — all database access lives here.

Connects to Cloud SQL (Postgres). Locally, it connects over TCP using
DB_HOST/DB_PORT. On Cloud Run, set INSTANCE_CONNECTION_NAME and it will
connect over the Unix socket that `--add-cloudsql-instances` wires up
automatically at /cloudsql/<INSTANCE_CONNECTION_NAME>.
"""

import os
import datetime
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def _build_db_url() -> str:
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD")
    db_name = os.environ.get("DB_NAME")

    instance_connection_name = os.environ.get("INSTANCE_CONNECTION_NAME")
    if instance_connection_name:
        # Running on Cloud Run: connect via the Cloud SQL unix socket.
        socket_path = f"/cloudsql/{instance_connection_name}"
        return (
            f"postgresql+psycopg2://{db_user}:{db_password}@/{db_name}"
            f"?host={socket_path}"
        )

    # Local development: plain TCP connection.
    db_host = os.environ.get("DB_HOST", "localhost")
    db_port = os.environ.get("DB_PORT", "5432")
    return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


@st.cache_resource
def get_engine() -> Engine:
    return create_engine(_build_db_url(), pool_pre_ping=True)


def init_db() -> None:
    """Create tables if they don't already exist. Safe to call on every startup."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id SERIAL PRIMARY KEY,
                user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                title VARCHAR(200) DEFAULT 'New Chat',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                id SERIAL PRIMARY KEY,
                conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE,
                role VARCHAR(20) NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))


# ---------- users ----------

def create_user(username: str, password_hash: str) -> int:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                INSERT INTO users (username, password_hash)
                VALUES (:username, :password_hash)
                RETURNING id
            """),
            {"username": username, "password_hash": password_hash},
        ).fetchone()
        return row[0]


def get_user_by_username(username: str):
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id, username, password_hash FROM users WHERE username = :username"),
            {"username": username},
        ).fetchone()
        return dict(row._mapping) if row else None


# ---------- conversations ----------

def create_conversation(user_id: int, title: str = "New Chat") -> int:
    engine = get_engine()
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                INSERT INTO conversations (user_id, title)
                VALUES (:user_id, :title)
                RETURNING id
            """),
            {"user_id": user_id, "title": title},
        ).fetchone()
        return row[0]


def get_conversations(user_id: int):
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT id, title, updated_at
                FROM conversations
                WHERE user_id = :user_id
                ORDER BY updated_at DESC
            """),
            {"user_id": user_id},
        ).fetchall()
        return [dict(r._mapping) for r in rows]


def touch_conversation(conversation_id: int) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE conversations SET updated_at = :now WHERE id = :id"),
            {"now": datetime.datetime.utcnow(), "id": conversation_id},
        )


def rename_conversation_if_default(conversation_id: int, new_title: str) -> None:
    """Auto-title a 'New Chat' conversation using its first user message."""
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE conversations
                SET title = :new_title
                WHERE id = :id AND title = 'New Chat'
            """),
            {"new_title": new_title[:200], "id": conversation_id},
        )


# ---------- messages ----------

def add_message(conversation_id: int, role: str, content: str) -> None:
    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO messages (conversation_id, role, content)
                VALUES (:conversation_id, :role, :content)
            """),
            {"conversation_id": conversation_id, "role": role, "content": content},
        )
    touch_conversation(conversation_id)


def get_messages(conversation_id: int):
    engine = get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = :conversation_id
                ORDER BY created_at ASC
            """),
            {"conversation_id": conversation_id},
        ).fetchall()
        return [dict(r._mapping) for r in rows]
