"""
SQLite-backed breach state store for Tier 2.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from .threshold_contracts import BreachState, BreachStatus


DEFAULT_STATE_DB_PATH = Path(__file__).parent / "database" / "threshold_state.db"


class BreachStateStore:
    """Persistent store for current breach lifecycle state."""

    def __init__(self, db_path: Union[str, Path] = DEFAULT_STATE_DB_PATH):
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
                CREATE TABLE IF NOT EXISTS breach_states (
                    state_key TEXT PRIMARY KEY,
                    rule_id TEXT NOT NULL,
                    sensor_ids_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    first_triggered_at TEXT,
                    last_triggered_at TEXT,
                    clear_started_at TEXT,
                    cleared_at TEXT,
                    rule_snapshot_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_breach_states_rule "
                "ON breach_states(rule_id)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_breach_states_status "
                "ON breach_states(status)"
            )
            self._conn.commit()

    def upsert(self, state: BreachState) -> BreachState:
        """Insert or update a breach state."""
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO breach_states (
                    state_key, rule_id, sensor_ids_json, status,
                    first_triggered_at, last_triggered_at, clear_started_at,
                    cleared_at, rule_snapshot_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(state_key) DO UPDATE SET
                    rule_id = excluded.rule_id,
                    sensor_ids_json = excluded.sensor_ids_json,
                    status = excluded.status,
                    first_triggered_at = excluded.first_triggered_at,
                    last_triggered_at = excluded.last_triggered_at,
                    clear_started_at = excluded.clear_started_at,
                    cleared_at = excluded.cleared_at,
                    rule_snapshot_json = excluded.rule_snapshot_json,
                    updated_at = excluded.updated_at
                """,
                _state_to_row(state),
            )
            self._conn.commit()
        return state

    def get(self, rule_id: str, sensor_ids: List[str]) -> Optional[BreachState]:
        state_key = f"{rule_id}:{','.join(sorted(sensor_ids))}"
        with self._lock:
            row = self._conn.execute(
                """
                SELECT rule_id, sensor_ids_json, status, first_triggered_at,
                       last_triggered_at, clear_started_at, cleared_at,
                       rule_snapshot_json, updated_at
                FROM breach_states
                WHERE state_key = ?
                """,
                (state_key,),
            ).fetchone()
        return _row_to_state(row) if row else None

    def get_by_rule(self, rule_id: str) -> List[BreachState]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT rule_id, sensor_ids_json, status, first_triggered_at,
                       last_triggered_at, clear_started_at, cleared_at,
                       rule_snapshot_json, updated_at
                FROM breach_states
                WHERE rule_id = ?
                ORDER BY updated_at DESC
                """,
                (rule_id,),
            ).fetchall()
        return [_row_to_state(row) for row in rows]

    def get_by_sensor(self, sensor_id: str) -> List[BreachState]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT rule_id, sensor_ids_json, status, first_triggered_at,
                       last_triggered_at, clear_started_at, cleared_at,
                       rule_snapshot_json, updated_at
                FROM breach_states
                ORDER BY updated_at DESC
                """
            ).fetchall()
        return [
            state
            for state in (_row_to_state(row) for row in rows)
            if sensor_id in state.sensor_ids
        ]

    def delete(self, rule_id: str, sensor_ids: List[str]) -> bool:
        state_key = f"{rule_id}:{','.join(sorted(sensor_ids))}"
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM breach_states WHERE state_key = ?",
                (state_key,),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def compact_cleared(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM breach_states WHERE status = ?",
                (BreachStatus.CLEARED.value,),
            )
            self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _state_to_row(state: BreachState) -> tuple:
    return (
        state.state_key,
        state.rule_id,
        json.dumps(state.sensor_ids),
        state.status.value,
        _dt_to_str(state.first_triggered_at),
        _dt_to_str(state.last_triggered_at),
        _dt_to_str(state.clear_started_at),
        _dt_to_str(state.cleared_at),
        json.dumps(state.rule_snapshot),
        state.updated_at.isoformat(),
    )


def _row_to_state(row) -> BreachState:
    return BreachState(
        rule_id=row[0],
        sensor_ids=json.loads(row[1]),
        status=row[2],
        first_triggered_at=_str_to_dt(row[3]),
        last_triggered_at=_str_to_dt(row[4]),
        clear_started_at=_str_to_dt(row[5]),
        cleared_at=_str_to_dt(row[6]),
        rule_snapshot=json.loads(row[7]),
        updated_at=_str_to_dt(row[8]),
    )


def _dt_to_str(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _str_to_dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


__all__ = ["DEFAULT_STATE_DB_PATH", "BreachStateStore"]
