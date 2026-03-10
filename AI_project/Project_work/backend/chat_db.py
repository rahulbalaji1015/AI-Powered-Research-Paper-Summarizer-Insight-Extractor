"""
chat_db.py — SQLite persistence for Research Paper RAG Assistant
Tables:
    sessions  — one row per chat session
    messages  — one row per Q&A exchange

Place this file in: backend/
Reads DB_FILE path from config.py at the project root.
"""

import sqlite3
import json
import time
import sys
import os

# ── Resolve project root so config.py is always importable ───────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))  # backend/
_ROOT = os.path.dirname(_HERE)                       # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config

DB_FILE = config.DB_FILE



# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def init_db():
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id    TEXT PRIMARY KEY,
                title         TEXT NOT NULL,
                paper_filter  TEXT,
                created_at    TEXT NOT NULL,
                updated_at    TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                message_id    TEXT PRIMARY KEY,
                session_id    TEXT NOT NULL,
                seq           INTEGER NOT NULL,
                query         TEXT NOT NULL,
                output_mode   TEXT,
                paper_id      TEXT,
                rag_answer    TEXT,
                ai_context    TEXT,
                additional    TEXT,
                about_paper   TEXT,
                single_answer TEXT,
                command_answer TEXT,
                sources        TEXT,
                metrics        TEXT,
                timestamp      TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE INDEX IF NOT EXISTS idx_msg_session
                ON messages(session_id, seq);
        """)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session(session_id, title, paper_filter=None):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute("""
            INSERT OR IGNORE INTO sessions
                (session_id, title, paper_filter, created_at, updated_at, message_count)
            VALUES (?,?,?,?,?,0)
        """, (session_id, title, paper_filter, now, now))
    return get_session(session_id)


def get_session(session_id):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id=?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


def rename_session(session_id, new_title):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with get_conn() as conn:
        conn.execute(
            "UPDATE sessions SET title=?, updated_at=? WHERE session_id=?",
            (new_title, now, session_id)
        )


def get_all_sessions():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY updated_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_session(session_id):
    with get_conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM sessions  WHERE session_id=?", (session_id,))


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def _next_seq(conn, session_id):
    row = conn.execute(
        "SELECT COALESCE(MAX(seq),0)+1 FROM messages WHERE session_id=?",
        (session_id,)
    ).fetchone()
    return row[0]


def save_message(session_id, message_id, result):
    """Persist one Q&A exchange from process_query() result dict."""
    now   = time.strftime("%Y-%m-%d %H:%M:%S")
    rtype = result.get("type", "")

    query         = result.get("query", "")
    output_mode   = result.get("output_mode", "")
    paper_id      = result.get("paper_id") or ""
    rag_answer    = result.get("rag_answer")
    ai_context    = result.get("ai_context")
    additional    = result.get("additional")
    about_paper   = result.get("about_paper")
    single_answer = result.get("single_answer")
    cmd_answer    = result.get("answer") if rtype in ("command", "compare") else None

    chunks  = result.get("chunks") or []
    sources = json.dumps([
        {"paper_id": c["paper_id"],
         "section":  c["section"],
         "score":    round(c["score"], 3)}
        for c in chunks
    ])

    m_raw   = result.get("metrics")
    metrics = json.dumps(m_raw) if m_raw else None

    with get_conn() as conn:
        seq = _next_seq(conn, session_id)
        conn.execute("""
            INSERT OR REPLACE INTO messages
                (message_id, session_id, seq, query, output_mode, paper_id,
                 rag_answer, ai_context, additional, about_paper,
                 single_answer, command_answer, sources, metrics, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (message_id, session_id, seq, query, output_mode, paper_id,
              rag_answer, ai_context, additional, about_paper,
              single_answer, cmd_answer, sources, metrics, now))
        conn.execute("""
            UPDATE sessions
            SET message_count=message_count+1, updated_at=?
            WHERE session_id=?
        """, (now, session_id))


def _deserialise(d):
    d["sources"] = json.loads(d["sources"]) if d.get("sources") else []
    d["metrics"] = json.loads(d["metrics"]) if d.get("metrics") else None
    return d


def get_session_messages(session_id):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY seq",
            (session_id,)
        ).fetchall()
    return [_deserialise(dict(r)) for r in rows]


def get_selected_messages(message_ids):
    if not message_ids:
        return []
    ph = ",".join("?" * len(message_ids))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM messages WHERE message_id IN ({ph}) ORDER BY seq",
            message_ids
        ).fetchall()
    return [_deserialise(dict(r)) for r in rows]


def get_db_stats():
    with get_conn() as conn:
        s = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        m = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    return {"sessions": s, "messages": m}