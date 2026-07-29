# -*- coding: utf-8 -*-
"""SQLite：sessions / messages / merchant_notes（按 merchant_id 隔离）。"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "buyer_memory.db"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class MemoryStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL,
                    operator_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_merchant
                    ON sessions(merchant_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, id);

                CREATE TABLE IF NOT EXISTS merchant_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    merchant_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_notes_merchant
                    ON merchant_notes(merchant_id, updated_at DESC);
                """
            )

    def ensure_session(
        self,
        *,
        merchant_id: str,
        operator_id: str = "",
        session_id: Optional[str] = None,
        title: str = "",
    ) -> str:
        merchant_id = (merchant_id or "").strip()
        if not merchant_id:
            raise ValueError("merchant_id 不能为空")
        sid = (session_id or "").strip() or str(uuid.uuid4())
        now = _now()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, merchant_id FROM sessions WHERE id = ?", (sid,)
            ).fetchone()
            if row:
                if row["merchant_id"] != merchant_id:
                    # 前端若未换 session，不硬报错：为新商家开一场新会话
                    sid = str(uuid.uuid4())
                else:
                    conn.execute(
                        "UPDATE sessions SET operator_id = ?, updated_at = ? WHERE id = ?",
                        (operator_id, now, sid),
                    )
                    return sid
            conn.execute(
                """
                INSERT INTO sessions (id, merchant_id, operator_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sid, merchant_id, operator_id, title or f"商家{merchant_id}", now, now),
            )
        return sid

    def list_sessions(self, merchant_id: str, limit: int = 20) -> List[dict]:
        merchant_id = (merchant_id or "").strip()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, merchant_id, operator_id, title, created_at, updated_at
                FROM sessions
                WHERE merchant_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (merchant_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        now = _now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (session_id, role, content, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )

    def get_recent_messages(self, session_id: str, limit: int = 20) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM (
                    SELECT id, role, content FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                ) t ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    def clear_session_messages(self, session_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_now(), session_id),
            )

    def add_note(
        self,
        merchant_id: str,
        content: str,
        *,
        source: str = "manual",
    ) -> int:
        merchant_id = (merchant_id or "").strip()
        content = (content or "").strip()
        if not merchant_id or not content:
            raise ValueError("merchant_id 与 content 不能为空")
        now = _now()
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO merchant_notes (merchant_id, content, source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (merchant_id, content, source, now, now),
            )
            return int(cur.lastrowid)

    def list_notes(self, merchant_id: str, limit: int = 5) -> List[dict]:
        merchant_id = (merchant_id or "").strip()
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT id, merchant_id, content, source, created_at, updated_at
                FROM merchant_notes
                WHERE merchant_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (merchant_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def notes_as_prompt_block(self, merchant_id: str, limit: int = 5) -> str:
        notes = self.list_notes(merchant_id, limit=limit)
        if not notes:
            return ""
        lines = [f"- {n['content']}" for n in notes]
        return "【商家长期记忆】\n" + "\n".join(lines)

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            s = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()["c"]
            m = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()["c"]
            n = conn.execute("SELECT COUNT(*) AS c FROM merchant_notes").fetchone()["c"]
        return {
            "db_path": str(self.db_path),
            "sessions": s,
            "messages": m,
            "notes": n,
        }
