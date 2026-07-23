from datetime import datetime, timedelta, timezone

from recognition.sensor_contracts import SensorType, SensorUnit, create_sensor_reading
from recognition.sensor_registry import get_registry, reset_registry


def _utc_now_naive():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def setup_function():
    reset_registry()


def teardown_function():
    reset_registry()


def test_record_reading_registers_sensor_and_updates_state():
    registry = get_registry()
    reading = create_sensor_reading(
        sensor_id="test_sensor_001",
        sensor_type=SensorType.TEMPERATURE,
        value=23.5,
        confidence=0.95,
        unit=SensorUnit.CELSIUS,
        source="test_script",
        metadata={"location": "lab", "test": True},
    )

    registry.record_reading(reading)

    sensor = registry.get_sensor("test_sensor_001")
    state = registry.get_state()

    assert sensor is not None
    assert sensor.sensor_id == "test_sensor_001"
    assert sensor.sensor_type == SensorType.TEMPERATURE
    assert sensor.reading_count == 1
    assert sensor.last_reading_at == reading.timestamp
    assert sensor.is_active is True
    assert state.total_sensors == 1
    assert state.active_sensors == 1
    assert state.total_readings == 1
    assert state.sensor_types == {"temperature": 1}


def test_existing_sensor_type_change_updates_metadata_and_type_index():
    registry = get_registry()

    registry.record_reading(
        create_sensor_reading(
            sensor_id="multi_sensor_001",
            sensor_type=SensorType.TEMPERATURE,
            value=23.5,
            confidence=0.95,
            unit=SensorUnit.CELSIUS,
        )
    )
    registry.record_reading(
        create_sensor_reading(
            sensor_id="multi_sensor_001",
            sensor_type=SensorType.HUMIDITY,
            value=45.0,
            confidence=0.90,
            unit=SensorUnit.PERCENT,
        )
    )

    sensor = registry.get_sensor("multi_sensor_001")
    state = registry.get_state()

    assert sensor is not None
    assert sensor.sensor_type == SensorType.HUMIDITY
    assert sensor.reading_count == 2
    assert state.total_sensors == 1
    assert state.total_readings == 2
    assert state.sensor_types.get("humidity") == 1
    assert state.sensor_types.get("temperature", 0) == 0
    assert registry.get_sensors_by_type(SensorType.HUMIDITY) == [sensor]
    assert registry.get_sensors_by_type(SensorType.TEMPERATURE) == []


def test_registry_tracks_multiple_sensors_by_type():
    registry = get_registry()

    registry.record_reading(
        create_sensor_reading(
            sensor_id="temp_001",
            sensor_type=SensorType.TEMPERATURE,
            value=21.0,
            confidence=0.9,
            unit=SensorUnit.CELSIUS,
        )
    )
    registry.record_reading(
        create_sensor_reading(
            sensor_id="accel_001",
            sensor_type=SensorType.ACCELEROMETER,
            value=9.81,
            confidence=0.85,
            unit=SensorUnit.METERS_PER_SECOND_SQUARED,
        )
    )

    state = registry.get_state()

    assert state.total_sensors == 2
    assert state.active_sensors == 2
    assert state.total_readings == 2
    assert state.sensor_types == {"temperature": 1, "accelerometer": 1}
    assert {s.sensor_id for s in registry.get_all_sensors()} == {
        "temp_001",
        "accel_001",
    }


def test_register_updates_location_index_when_sensor_moves_rooms():
    registry = get_registry()

    registry.register(
        sensor_id="room_sensor_001",
        sensor_type=SensorType.TEMPERATURE,
        location={"room": "lab"},
    )
    registry.register(
        sensor_id="room_sensor_001",
        sensor_type=SensorType.TEMPERATURE,
        location={"room": "office"},
    )

    assert registry.get_sensors_by_location("lab") == []
    assert [s.sensor_id for s in registry.get_sensors_by_location("office")] == [
        "room_sensor_001"
    ]


def test_cleanup_stale_sensors_marks_only_old_active_sensors_inactive():
    registry = get_registry()
    old_reading = create_sensor_reading(
        sensor_id="old_sensor",
        sensor_type=SensorType.TEMPERATURE,
        value=20.0,
        confidence=0.9,
        unit=SensorUnit.CELSIUS,
    )
    old_reading.timestamp = _utc_now_naive() - timedelta(days=2)
    fresh_reading = create_sensor_reading(
        sensor_id="fresh_sensor",
        sensor_type=SensorType.TEMPERATURE,
        value=21.0,
        confidence=0.9,
        unit=SensorUnit.CELSIUS,
    )

    registry.record_reading(old_reading)
    registry.record_reading(fresh_reading)

    stale_count = registry.cleanup_stale_sensors(max_age_seconds=86400)
    state = registry.get_state()

    assert stale_count == 1
    assert registry.get_sensor("old_sensor").is_active is False
    assert registry.get_sensor("fresh_sensor").is_active is True
    assert state.total_sensors == 2
    assert state.active_sensors == 1
    assert state.sensor_types == {"temperature": 1}
