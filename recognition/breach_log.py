"""
SQLite-backed breach audit log for Tier 2.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from .threshold_contracts import BreachLogEntry, RuleSeverity


DEFAULT_BREACH_LOG_DB_PATH = Path(__file__).parent / "database" / "breach_log.db"
MAX_BREACH_QUERY_LIMIT = 1000


class BreachLogStore:
    """Append-oriented breach log with query and review helpers."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_BREACH_LOG_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS breach_log (
                    breach_id TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    sensor_ids_json TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    cleared_at TEXT,
                    severity TEXT NOT NULL,
                    context_snapshot_json TEXT NOT NULL,
                    escalated_to_tier3 INTEGER NOT NULL,
                    tier3_decision_json TEXT,
                    action_taken_json TEXT,
                    human_reviewed INTEGER NOT NULL,
                    human_notes TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_breach_log_rule "
                "ON breach_log(rule_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_breach_log_triggered "
                "ON breach_log(triggered_at)"
            )
            self._conn.commit()

    def append(self, entry: BreachLogEntry) -> str:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO breach_log (
                    breach_id, rule_id, sensor_ids_json, triggered_at,
                    cleared_at, severity, context_snapshot_json,
                    escalated_to_tier3, tier3_decision_json, action_taken_json,
                    human_reviewed, human_notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                _entry_to_row(entry),
            )
            self._conn.commit()
        return entry.breach_id

    def get(self, breach_id: str) -> Optional[BreachLogEntry]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT breach_id, rule_id, sensor_ids_json, triggered_at,
                       cleared_at, severity, context_snapshot_json,
                       escalated_to_tier3, tier3_decision_json,
                       action_taken_json, human_reviewed, human_notes
                FROM breach_log
                WHERE breach_id = ?
                """,
                (breach_id,),
            ).fetchone()
        return _row_to_entry(row) if row else None

    def query(
        self,
        rule_id: Optional[str] = None,
        sensor_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[BreachLogEntry]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if limit > MAX_BREACH_QUERY_LIMIT:
            raise ValueError(f"limit cannot exceed {MAX_BREACH_QUERY_LIMIT}")

        clauses = []
        params = []
        if rule_id:
            clauses.append("rule_id = ?")
            params.append(rule_id)
        if since:
            clauses.append("triggered_at >= ?")
            params.append(since.isoformat())
        if until:
            clauses.append("triggered_at <= ?")
            params.append(until.isoformat())

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            "SELECT breach_id, rule_id, sensor_ids_json, triggered_at, "
            "cleared_at, severity, context_snapshot_json, escalated_to_tier3, "
            "tier3_decision_json, action_taken_json, human_reviewed, human_notes "
            f"FROM breach_log {where_sql} "
            "ORDER BY triggered_at DESC LIMIT ?"
        )
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        entries = [_row_to_entry(row) for row in rows]
        if sensor_id:
            entries = [entry for entry in entries if sensor_id in entry.sensor_ids]
        return entries

    def mark_human_reviewed(
        self,
        breach_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE breach_log
                SET human_reviewed = 1, human_notes = ?
                WHERE breach_id = ?
                """,
                (notes, breach_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def attach_tier3_decision(
        self,
        breach_id: str,
        decision: Dict,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE breach_log
                SET escalated_to_tier3 = 1, tier3_decision_json = ?
                WHERE breach_id = ?
                """,
                (json.dumps(decision), breach_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def attach_action_taken(
        self,
        breach_id: str,
        action: Dict,
    ) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE breach_log
                SET action_taken_json = ?
                WHERE breach_id = ?
                """,
                (json.dumps(action), breach_id),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _entry_to_row(entry: BreachLogEntry) -> tuple:
    return (
        entry.breach_id,
        entry.rule_id,
        json.dumps(entry.sensor_ids),
        entry.triggered_at.isoformat(),
        _dt_to_str(entry.cleared_at),
        entry.severity.value,
        json.dumps(entry.context_snapshot),
        1 if entry.escalated_to_tier3 else 0,
        json.dumps(entry.tier3_decision) if entry.tier3_decision else None,
        json.dumps(entry.action_taken) if entry.action_taken else None,
        1 if entry.human_reviewed else 0,
        entry.human_notes,
    )


def _row_to_entry(row) -> BreachLogEntry:
    return BreachLogEntry(
        breach_id=row[0],
        rule_id=row[1],
        sensor_ids=json.loads(row[2]),
        triggered_at=datetime.fromisoformat(row[3]),
        cleared_at=_str_to_dt(row[4]),
        severity=RuleSeverity(row[5]),
        context_snapshot=json.loads(row[6]),
        escalated_to_tier3=bool(row[7]),
        tier3_decision=json.loads(row[8]) if row[8] else None,
        action_taken=json.loads(row[9]) if row[9] else None,
        human_reviewed=bool(row[10]),
        human_notes=row[11],
    )


def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _str_to_dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


__all__ = [
    "DEFAULT_BREACH_LOG_DB_PATH",
    "MAX_BREACH_QUERY_LIMIT",
    "BreachLogStore",
]
