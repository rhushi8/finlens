"""Lightweight answer-feedback store backed by SQLite."""

from __future__ import annotations

import logging
import sqlite3
import time

from finance_ai.config import ROOT_DIR

logger = logging.getLogger(__name__)

_DB_PATH = ROOT_DIR / "data" / "feedback.db"

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          REAL    NOT NULL,
    query       TEXT    NOT NULL,
    ticker      TEXT,
    intent      TEXT,
    recommendation TEXT,
    grounding_score REAL,
    rating      INTEGER NOT NULL   -- 1 = positive, -1 = negative
)
"""


def _connect() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.execute(_CREATE_SQL)
    conn.commit()
    return conn


def record_feedback(
    *,
    query: str,
    ticker: str | None,
    intent: str,
    recommendation: str,
    grounding_score: float,
    rating: int,
) -> None:
    """Persist a thumbs-up (rating=1) or thumbs-down (rating=-1) for an answer."""
    try:
        conn = _connect()
        conn.execute(
            "INSERT INTO feedback (ts, query, ticker, intent, recommendation, grounding_score, rating) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (time.time(), query, ticker, intent, recommendation, grounding_score, rating),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.warning("Failed to record feedback: %s", exc)


def get_feedback_stats() -> dict:
    """Return aggregate counts for display in the sidebar."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT COUNT(*) as total, "
            "SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as positive, "
            "SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as negative "
            "FROM feedback"
        ).fetchone()
        conn.close()
        if row and row[0]:
            return {"total": row[0], "positive": row[1] or 0, "negative": row[2] or 0}
    except Exception as exc:
        logger.warning("Failed to fetch feedback stats: %s", exc)
    return {"total": 0, "positive": 0, "negative": 0}
