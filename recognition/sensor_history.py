"""
Lyr Rolling History Store - Tier 1.

Time-bounded, per-sensor-capped, SQLite-backed storage for SensorReading
objects. The store is intentionally structured-query only: callers can filter
by a small whitelist of fields and operators, but never pass raw SQL.
"""

import hashlib
import json
import operator
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .sensor_contracts import SensorReading, SensorType, SensorUnit


DEFAULT_DB_PATH = Path(__file__).parent / "database" / "history.db"
DEFAULT_PER_SENSOR_CAP = 1000
TRIM_HEADROOM = 1.1
BURST_THRESHOLD_WRITES_PER_SEC = 50
WRITE_QUEUE_MAXSIZE = 5000
MAX_QUERY_LIMIT = 1000
FLUSH_INTERVAL_SECONDS = 0.2
FLUSH_BATCH_SIZE = 200
COMPRESSION_MIN_RUN = 5
COMPRESSION_VALUE_TOLERANCE = 1e-6

ALLOWED_FIELDS = {
    "sensor_id",
    "sensor_type",
    "value",
    "timestamp",
    "confidence",
    "unit",
    "source",
}
ALLOWED_OPS = {"=", "!=", ">", ">=", "<", "<=", "LIKE"}


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class HistoryBackpressureError(RuntimeError):
    """Raised when the buffered write queue is full and cannot accept data."""


@dataclass
class StoredReading:
    """SQLite row representation of one reading, or one compressed run."""

    reading_id: str
    sensor_id: str
    sensor_type: str
    value: float
    timestamp: str
    ingested_at: str
    confidence: Optional[float]
    unit: Optional[str]
    source: Optional[str]
    location_json: Optional[str]
    metadata_json: Optional[str]
    is_compressed: int
    original_count: int
    compressed_checksum: Optional[str]

    def to_reading(self) -> SensorReading:
        """Convert the stored row back into the public SensorReading contract."""
        if self.is_compressed and self.compressed_checksum:
            expected = _checksum(
                self.sensor_id,
                self.sensor_type,
                self.unit,
                self.value,
                self.original_count,
            )
            if expected != self.compressed_checksum:
                raise ValueError(
                    f"Compressed reading {self.reading_id} failed integrity check"
                )

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=SensorType.from_string(self.sensor_type),
            value=self.value,
            timestamp=datetime.fromisoformat(self.timestamp),
            unit=SensorUnit.from_string(self.unit) if self.unit else None,
            confidence_score=self.confidence,
            source=self.source,
            location=json.loads(self.location_json) if self.location_json else None,
            metadata=json.loads(self.metadata_json) if self.metadata_json else {},
            reading_id=self.reading_id,
        )


@dataclass
class QueryFilter:
    """One structured filter clause: field op value."""

    field: str
    op: str
    value: Any

    def __post_init__(self) -> None:
        if self.field not in ALLOWED_FIELDS:
            raise ValueError(
                f"Field '{self.field}' is not queryable. "
                f"Allowed: {sorted(ALLOWED_FIELDS)}"
            )
        if self.op not in ALLOWED_OPS:
            raise ValueError(
                f"Operator '{self.op}' not allowed. Allowed: {sorted(ALLOWED_OPS)}"
            )


def _checksum(
    sensor_id: str,
    sensor_type: str,
    unit: Optional[str],
    value: float,
    count: int,
) -> str:
    payload = f"{sensor_id}|{sensor_type}|{unit}|{value:.9f}|{count}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HistoryStore:
    """
    SQLite-backed rolling history store with adaptive burst buffering.

    Use get_store() for the shared process-wide instance in normal operation.
    Tests can create isolated file-backed instances directly.
    """

    def __init__(
        self,
        db_path: Union[str, Path] = DEFAULT_DB_PATH,
        per_sensor_cap: int = DEFAULT_PER_SENSOR_CAP,
    ):
        if per_sensor_cap < 1:
            raise ValueError("per_sensor_cap must be at least 1")

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.per_sensor_cap = per_sensor_cap

        self._local = threading.local()
        self._connections: List[sqlite3.Connection] = []
        self._connections_guard = threading.Lock()
        self._sensor_locks: Dict[str, threading.RLock] = {}
        self._sensor_locks_guard = threading.Lock()

        self._pending: Dict[str, List[SensorReading]] = {}
        self._pending_guard = threading.Lock()
        self._write_queue: "queue.Queue[SensorReading]" = queue.Queue(
            maxsize=WRITE_QUEUE_MAXSIZE
        )
        self._recent_write_timestamps: List[float] = []
        self._rate_guard = threading.Lock()

        self._stop_flag = threading.Event()
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="HistoryStoreFlush",
        )

        self._init_schema()
        self._flush_thread.start()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            with self._connections_guard:
                self._connections.append(conn)
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS readings (
                reading_id TEXT PRIMARY KEY,
                sensor_id TEXT NOT NULL,
                sensor_type TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp TEXT NOT NULL,
                ingested_at TEXT NOT NULL,
                confidence REAL,
                unit TEXT,
                source TEXT,
                location_json TEXT,
                metadata_json TEXT,
                is_compressed INTEGER DEFAULT 0,
                original_count INTEGER DEFAULT 1,
                compressed_checksum TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sensor_id ON readings(sensor_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON readings(timestamp)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_type ON readings(sensor_type)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sensor_timestamp "
            "ON readings(sensor_id, timestamp)"
        )
        conn.commit()

    def _lock_for(self, sensor_id: str) -> threading.RLock:
        with self._sensor_locks_guard:
            if sensor_id not in self._sensor_locks:
                self._sensor_locks[sensor_id] = threading.RLock()
            return self._sensor_locks[sensor_id]

    def record(self, reading: SensorReading) -> None:
        """Record a reading, buffering automatically during burst load."""
        if self._is_under_burst_load():
            self._record_buffered(reading)
        else:
            self._record_direct(reading)
        self._note_write()

    def _is_under_burst_load(self) -> bool:
        with self._rate_guard:
            now = time.monotonic()
            cutoff = now - 1.0
            self._recent_write_timestamps = [
                t for t in self._recent_write_timestamps if t > cutoff
            ]
            return len(self._recent_write_timestamps) >= BURST_THRESHOLD_WRITES_PER_SEC

    def _note_write(self) -> None:
        with self._rate_guard:
            self._recent_write_timestamps.append(time.monotonic())

    def _record_direct(self, reading: SensorReading) -> None:
        lock = self._lock_for(reading.sensor_id)
        with lock:
            self._insert_rows([reading])
            self._maybe_trim(reading.sensor_id)

    def _record_buffered(self, reading: SensorReading) -> None:
        try:
            self._write_queue.put_nowait(reading)
        except queue.Full as exc:
            raise HistoryBackpressureError(
                "history write queue is full; caller should retry later"
            ) from exc

        with self._pending_guard:
            self._pending.setdefault(reading.sensor_id, []).append(reading)

    def _flush_loop(self) -> None:
        while not self._stop_flag.is_set():
            batch: List[SensorReading] = []
            deadline = time.monotonic() + FLUSH_INTERVAL_SECONDS
            while len(batch) < FLUSH_BATCH_SIZE and time.monotonic() < deadline:
                try:
                    remaining = max(0.0, deadline - time.monotonic())
                    batch.append(self._write_queue.get(timeout=remaining))
                except queue.Empty:
                    break

            if batch:
                self._flush_batch(batch)

    def _flush_batch(self, batch: List[SensorReading]) -> None:
        self._insert_rows(batch)
        touched_sensors = {r.sensor_id for r in batch}
        flushed_ids = {r.reading_id for r in batch}

        with self._pending_guard:
            for sensor_id in touched_sensors:
                pending = self._pending.get(sensor_id, [])
                self._pending[sensor_id] = [
                    r for r in pending if r.reading_id not in flushed_ids
                ]
                if not self._pending[sensor_id]:
                    self._pending.pop(sensor_id, None)

        for sensor_id in touched_sensors:
            self._maybe_trim(sensor_id)

    def stop(self) -> None:
        """Flush remaining buffered readings and stop the background thread."""
        self._stop_flag.set()
        self._flush_thread.join(timeout=2.0)

        remaining: List[SensorReading] = []
        while not self._write_queue.empty():
            try:
                remaining.append(self._write_queue.get_nowait())
            except queue.Empty:
                break

        if remaining:
            self._flush_batch(remaining)

        self._close_connections()

    def _close_connections(self) -> None:
        with self._connections_guard:
            connections = list(self._connections)
            self._connections.clear()

        for conn in connections:
            try:
                conn.close()
            except sqlite3.Error:
                pass

        if hasattr(self._local, "conn"):
            del self._local.conn

    def _insert_rows(self, readings: List[SensorReading]) -> None:
        if not readings:
            return

        conn = self._get_conn()
        now = _utc_now_naive().isoformat()
        rows = [
            (
                r.reading_id,
                r.sensor_id,
                r.sensor_type.value,
                r.value,
                r.timestamp.isoformat(),
                now,
                r.confidence_score,
                r.unit.value if r.unit else None,
                r.source,
                json.dumps(r.location) if r.location else None,
                json.dumps(r.metadata) if r.metadata else None,
                0,
                1,
                None,
            )
            for r in readings
        ]

        conn.executemany(
            """
            INSERT OR IGNORE INTO readings
                (reading_id, sensor_id, sensor_type, value, timestamp,
                 ingested_at, confidence, unit, source, location_json,
                 metadata_json, is_compressed, original_count,
                 compressed_checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()

    def _maybe_trim(self, sensor_id: str) -> None:
        conn = self._get_conn()
        count = conn.execute(
            "SELECT COUNT(*) FROM readings WHERE sensor_id = ?", (sensor_id,)
        ).fetchone()[0]

        if count <= self.per_sensor_cap * TRIM_HEADROOM:
            return

        conn.execute(
            """
            DELETE FROM readings WHERE reading_id IN (
                SELECT reading_id FROM readings
                WHERE sensor_id = ?
                ORDER BY timestamp DESC
                LIMIT -1 OFFSET ?
            )
            """,
            (sensor_id, self.per_sensor_cap),
        )
        conn.commit()

    def compress_run(self, readings: List[SensorReading]) -> Optional[StoredReading]:
        """
        Attempt to compress a homogeneous run of readings into one row.

        Compression is conservative: the run must share sensor_id, sensor_type,
        and unit, with values within COMPRESSION_VALUE_TOLERANCE.
        """
        if len(readings) < COMPRESSION_MIN_RUN:
            return None

        first = readings[0]
        for reading in readings[1:]:
            if reading.sensor_id != first.sensor_id:
                return None
            if reading.sensor_type != first.sensor_type:
                return None
            if reading.unit != first.unit:
                return None
            if abs(reading.value - first.value) > COMPRESSION_VALUE_TOLERANCE:
                return None

        checksum = _checksum(
            first.sensor_id,
            first.sensor_type.value,
            first.unit.value if first.unit else None,
            first.value,
            len(readings),
        )

        return StoredReading(
            reading_id=f"compressed_{first.reading_id}",
            sensor_id=first.sensor_id,
            sensor_type=first.sensor_type.value,
            value=first.value,
            timestamp=readings[-1].timestamp.isoformat(),
            ingested_at=_utc_now_naive().isoformat(),
            confidence=first.confidence_score,
            unit=first.unit.value if first.unit else None,
            source=first.source,
            location_json=json.dumps(first.location) if first.location else None,
            metadata_json=None,
            is_compressed=1,
            original_count=len(readings),
            compressed_checksum=checksum,
        )

    def get_history(
        self,
        sensor_id: str,
        limit: int = 100,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[SensorReading]:
        """Recent readings for one sensor, most recent first."""
        filters = [QueryFilter("sensor_id", "=", sensor_id)]
        if since:
            filters.append(QueryFilter("timestamp", ">=", since.isoformat()))
        if until:
            filters.append(QueryFilter("timestamp", "<=", until.isoformat()))
        return self.query(filters=filters, limit=limit)

    def get_history_by_type(
        self,
        sensor_type: Union[str, SensorType],
        limit: int = 100,
    ) -> List[SensorReading]:
        st = sensor_type.value if isinstance(sensor_type, SensorType) else sensor_type
        return self.query(filters=[QueryFilter("sensor_type", "=", st)], limit=limit)

    def get_history_global(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[SensorReading]:
        filters: List[QueryFilter] = []
        if since:
            filters.append(QueryFilter("timestamp", ">=", since.isoformat()))
        if until:
            filters.append(QueryFilter("timestamp", "<=", until.isoformat()))
        return self.query(filters=filters, limit=limit)

    def query(
        self,
        filters: Optional[List[QueryFilter]] = None,
        limit: int = 100,
        order: str = "desc",
    ) -> List[SensorReading]:
        """Run a structured query and merge not-yet-flushed buffered readings."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if limit > MAX_QUERY_LIMIT:
            raise ValueError(f"limit cannot exceed {MAX_QUERY_LIMIT}")

        filters = filters or []
        order_sql = "DESC" if order.lower() == "desc" else "ASC"

        where_clauses = []
        params: List[Any] = []
        for query_filter in filters:
            where_clauses.append(f"{query_filter.field} {query_filter.op} ?")
            params.append(query_filter.value)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        sql = (
            "SELECT reading_id, sensor_id, sensor_type, value, timestamp, "
            "ingested_at, confidence, unit, source, location_json, "
            "metadata_json, is_compressed, original_count, compressed_checksum "
            f"FROM readings {where_sql} ORDER BY timestamp {order_sql} LIMIT ?"
        )
        params.append(limit)

        conn = self._get_conn()
        rows = conn.execute(sql, params).fetchall()
        results = [StoredReading(*row).to_reading() for row in rows]
        results.extend(self._matching_pending(filters))

        reverse = order_sql == "DESC"
        deduped = {reading.reading_id: reading for reading in results}
        ordered = sorted(
            deduped.values(),
            key=lambda reading: reading.timestamp,
            reverse=reverse,
        )
        return ordered[:limit]

    def _matching_pending(self, filters: List[QueryFilter]) -> List[SensorReading]:
        with self._pending_guard:
            pending = [
                reading
                for readings in self._pending.values()
                for reading in readings
            ]

        return [
            reading
            for reading in pending
            if all(_matches_filter(reading, query_filter) for query_filter in filters)
        ]


def _matches_filter(reading: SensorReading, query_filter: QueryFilter) -> bool:
    value = _reading_field(reading, query_filter.field)
    expected = query_filter.value

    if isinstance(value, datetime):
        value = value.isoformat()

    ops = {
        "=": operator.eq,
        "!=": operator.ne,
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
    }

    if query_filter.op == "LIKE":
        return str(expected).replace("%", "") in str(value)

    return ops[query_filter.op](value, expected)


def _reading_field(reading: SensorReading, field_name: str) -> Any:
    if field_name == "sensor_id":
        return reading.sensor_id
    if field_name == "sensor_type":
        return reading.sensor_type.value
    if field_name == "value":
        return reading.value
    if field_name == "timestamp":
        return reading.timestamp
    if field_name == "confidence":
        return reading.confidence_score
    if field_name == "unit":
        return reading.unit.value if reading.unit else None
    if field_name == "source":
        return reading.source
    raise ValueError(f"Unsupported field: {field_name}")


_store: Optional[HistoryStore] = None
_store_lock = threading.Lock()


def get_store() -> HistoryStore:
    """Get the global HistoryStore instance."""
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = HistoryStore()
    return _store


def reset_store_for_testing(
    db_path: Optional[Union[str, Path]] = None,
    per_sensor_cap: int = DEFAULT_PER_SENSOR_CAP,
) -> HistoryStore:
    """
    Create a fresh file-backed store for tests.

    A temp SQLite file is used instead of ':memory:' because the store uses
    one connection per thread, and in-memory SQLite databases are per-connection.
    """
    import tempfile

    global _store
    if _store is not None:
        _store.stop()

    if db_path is None:
        db_path = Path(tempfile.mkdtemp()) / "test_history.db"

    _store = HistoryStore(db_path=db_path, per_sensor_cap=per_sensor_cap)
    return _store


__all__ = [
    "HistoryStore",
    "HistoryBackpressureError",
    "StoredReading",
    "QueryFilter",
    "get_store",
    "reset_store_for_testing",
]
