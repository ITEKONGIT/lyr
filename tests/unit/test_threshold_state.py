from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from recognition.threshold_contracts import BreachState, BreachStatus
from recognition.threshold_state import BreachStateStore


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _state(sensor_id="temp_1", status=BreachStatus.PENDING_SUSTAIN):
    return BreachState(
        rule_id="temperature_high",
        sensor_ids=[sensor_id],
        status=status,
        first_triggered_at=_now(),
        last_triggered_at=_now(),
        rule_snapshot={"rule_id": "temperature_high"},
    )


def test_creates_schema_and_upserts_pending_state(tmp_path):
    store = BreachStateStore(tmp_path / "threshold_state.db")
    state = _state()

    store.upsert(state)
    restored = store.get("temperature_high", ["temp_1"])

    assert restored is not None
    assert restored.rule_id == "temperature_high"
    assert restored.sensor_ids == ["temp_1"]
    assert restored.status == BreachStatus.PENDING_SUSTAIN
    assert restored.rule_snapshot == {"rule_id": "temperature_high"}
    store.close()


def test_upserts_active_clearing_and_cleared_states(tmp_path):
    store = BreachStateStore(tmp_path / "threshold_state.db")

    active = _state(status=BreachStatus.ACTIVE)
    store.upsert(active)
    assert store.get("temperature_high", ["temp_1"]).status == BreachStatus.ACTIVE

    clearing = _state(status=BreachStatus.CLEARING)
    clearing.clear_started_at = _now()
    store.upsert(clearing)
    assert store.get("temperature_high", ["temp_1"]).status == BreachStatus.CLEARING

    cleared = _state(status=BreachStatus.CLEARED)
    cleared.cleared_at = _now()
    store.upsert(cleared)
    assert store.get("temperature_high", ["temp_1"]).status == BreachStatus.CLEARED
    store.close()


def test_retrieves_by_rule_and_sensor(tmp_path):
    store = BreachStateStore(tmp_path / "threshold_state.db")
    store.upsert(_state(sensor_id="temp_1"))
    store.upsert(_state(sensor_id="temp_2"))

    by_rule = store.get_by_rule("temperature_high")
    by_sensor = store.get_by_sensor("temp_2")

    assert {state.sensor_ids[0] for state in by_rule} == {"temp_1", "temp_2"}
    assert len(by_sensor) == 1
    assert by_sensor[0].sensor_ids == ["temp_2"]
    store.close()


def test_compacts_cleared_states(tmp_path):
    store = BreachStateStore(tmp_path / "threshold_state.db")
    store.upsert(_state(sensor_id="temp_1", status=BreachStatus.CLEARED))
    store.upsert(_state(sensor_id="temp_2", status=BreachStatus.ACTIVE))

    removed = store.compact_cleared()

    assert removed == 1
    assert store.get("temperature_high", ["temp_1"]) is None
    assert store.get("temperature_high", ["temp_2"]) is not None
    store.close()


def test_state_survives_store_restart(tmp_path):
    db_path = tmp_path / "threshold_state.db"
    first = BreachStateStore(db_path)
    first.upsert(_state(status=BreachStatus.ACTIVE))
    first.close()

    second = BreachStateStore(db_path)
    restored = second.get("temperature_high", ["temp_1"])

    assert restored is not None
    assert restored.status == BreachStatus.ACTIVE
    second.close()


def test_concurrent_upserts_for_different_sensors_do_not_corrupt_state(tmp_path):
    store = BreachStateStore(tmp_path / "threshold_state.db")
    sensor_count = 40

    def write_state(index):
        store.upsert(_state(sensor_id=f"temp_{index}", status=BreachStatus.ACTIVE))

    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(write_state, range(sensor_count)))

    states = store.get_by_rule("temperature_high")

    assert len(states) == sensor_count
    assert {state.sensor_ids[0] for state in states} == {
        f"temp_{idx}" for idx in range(sensor_count)
    }
    store.close()
