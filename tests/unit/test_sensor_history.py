from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from recognition.sensor_contracts import SensorType, SensorUnit, create_sensor_reading
import recognition.sensor_history as sensor_history
from recognition.sensor_history import (
    BURST_THRESHOLD_WRITES_PER_SEC,
    HistoryBackpressureError,
    HistoryStore,
    MAX_QUERY_LIMIT,
    QueryFilter,
)


def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _reading(sensor_id, value, *, seconds=0, sensor_type=SensorType.TEMPERATURE):
    return create_sensor_reading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        value=value,
        confidence=0.9,
        unit=SensorUnit.CELSIUS,
        source="unit_test",
        metadata={"idx": value},
    )


def test_records_and_queries_recent_history(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    base = _utc_now_naive() - timedelta(minutes=1)

    readings = [
        _reading("temp_1", 20.0),
        _reading("temp_1", 21.0),
        _reading("temp_2", 99.0),
    ]
    for idx, reading in enumerate(readings):
        reading.timestamp = base + timedelta(seconds=idx)
        store.record(reading)

    history = store.get_history("temp_1")

    assert [r.value for r in history] == [21.0, 20.0]
    assert history[0].metadata == {"idx": 21.0}
    store.stop()


def test_trims_per_sensor_after_headroom(tmp_path):
    store = HistoryStore(tmp_path / "history.db", per_sensor_cap=3)
    base = _utc_now_naive() - timedelta(minutes=1)

    for idx in range(5):
        reading = _reading("temp_1", float(idx))
        reading.timestamp = base + timedelta(seconds=idx)
        store.record(reading)

    history = store.get_history("temp_1", limit=10)

    assert [r.value for r in history] == [4.0, 3.0, 2.0]
    store.stop()


def test_rejects_unstructured_query_fields(tmp_path):
    store = HistoryStore(tmp_path / "history.db")

    with pytest.raises(ValueError):
        QueryFilter("sensor_id; DROP TABLE readings", "=", "temp_1")

    with pytest.raises(ValueError):
        QueryFilter("sensor_id", "IN", "temp_1")

    store.stop()


def test_compress_run_only_for_homogeneous_values(tmp_path):
    store = HistoryStore(tmp_path / "history.db")
    readings = [_reading("temp_1", 22.0) for _ in range(5)]

    compressed = store.compress_run(readings)

    assert compressed is not None
    assert compressed.original_count == 5
    assert compressed.to_reading().value == 22.0

    mixed = readings[:4] + [_reading("temp_1", 23.0)]
    assert store.compress_run(mixed) is None
    store.stop()


def test_buffered_readings_are_visible_before_flush(tmp_path):
    store = HistoryStore(tmp_path / "history.db")

    for idx in range(BURST_THRESHOLD_WRITES_PER_SEC):
        store.record(_reading("temp_1", float(idx)))

    buffered = _reading("temp_2", 999.0)
    buffered.sensor_type = SensorType.HUMIDITY
    buffered.unit = SensorUnit.PERCENT
    store.record(buffered)

    by_sensor = store.get_history("temp_2")
    by_type = store.get_history_by_type(SensorType.HUMIDITY)
    global_history = store.get_history_global(limit=100)

    assert buffered.reading_id in {r.reading_id for r in by_sensor}
    assert buffered.reading_id in {r.reading_id for r in by_type}
    assert buffered.reading_id in {r.reading_id for r in global_history}
    store.stop()


def test_query_limit_is_capped(tmp_path):
    store = HistoryStore(tmp_path / "history.db")

    with pytest.raises(ValueError, match=f"cannot exceed {MAX_QUERY_LIMIT}"):
        store.get_history_global(limit=MAX_QUERY_LIMIT + 1)

    store.stop()


def test_buffered_write_queue_fails_loud_when_full(tmp_path, monkeypatch):
    monkeypatch.setattr(sensor_history, "WRITE_QUEUE_MAXSIZE", 1)
    store = HistoryStore(tmp_path / "history.db")
    try:
        now = time.monotonic()
        store._recent_write_timestamps = [now] * BURST_THRESHOLD_WRITES_PER_SEC
        store._write_queue.put_nowait(_reading("queued", 1.0))

        with pytest.raises(HistoryBackpressureError):
            store.record(_reading("overflow", 2.0))

        assert store.get_history("overflow") == []
    finally:
        store.stop()


def test_concurrent_direct_writes_do_not_drop_or_duplicate_readings(tmp_path):
    store = HistoryStore(tmp_path / "history.db", per_sensor_cap=1000)
    worker_count = 20
    writes_per_worker = 10

    def write_batch(worker_index):
        for idx in range(writes_per_worker):
            sensor_id = f"sensor_{idx % 5}"
            reading = _reading(
                sensor_id,
                float(worker_index * writes_per_worker + idx),
            )
            store._record_direct(reading)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        list(executor.map(write_batch, range(worker_count)))

    expected_count = worker_count * writes_per_worker
    readings = store.get_history_global(limit=expected_count)
    reading_ids = [reading.reading_id for reading in readings]

    assert len(readings) == expected_count
    assert len(set(reading_ids)) == expected_count

    for idx in range(5):
        sensor_readings = store.get_history(f"sensor_{idx}", limit=1000)
        assert len(sensor_readings) == worker_count * (writes_per_worker // 5)

    store.stop()
