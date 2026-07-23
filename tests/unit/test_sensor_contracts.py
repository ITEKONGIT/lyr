from datetime import datetime, timedelta, timezone

import pytest

from recognition.sensor_contracts import (
    ConfidenceLevel,
    FUTURE_TIMESTAMP_TOLERANCE_SECONDS,
    SensorReading,
    SensorType,
    SensorUnit,
)


def test_aware_timestamp_is_normalized_without_crashing_validation():
    reading = SensorReading(
        sensor_id="temp_aware",
        sensor_type=SensorType.TEMPERATURE,
        value=22.0,
        timestamp=datetime.now(timezone.utc) - timedelta(seconds=1),
        unit=SensorUnit.CELSIUS,
        confidence_score=0.9,
    )

    assert reading.timestamp.tzinfo is None


def test_small_future_clock_skew_is_accepted():
    reading = SensorReading(
        sensor_id="temp_skew",
        sensor_type=SensorType.TEMPERATURE,
        value=22.0,
        timestamp=(
            datetime.now(timezone.utc)
            + timedelta(seconds=FUTURE_TIMESTAMP_TOLERANCE_SECONDS - 1)
        ),
        unit=SensorUnit.CELSIUS,
        confidence_score=0.9,
    )

    assert reading.timestamp.tzinfo is None


def test_future_aware_timestamp_beyond_tolerance_raises_value_error():
    with pytest.raises(ValueError, match="timestamp cannot be in the future"):
        SensorReading(
            sensor_id="temp_future",
            sensor_type=SensorType.TEMPERATURE,
            value=22.0,
            timestamp=(
                datetime.now(timezone.utc)
                + timedelta(seconds=FUTURE_TIMESTAMP_TOLERANCE_SECONDS + 1)
            ),
            unit=SensorUnit.CELSIUS,
            confidence_score=0.9,
        )


def test_autonomous_action_defaults_to_stricter_start_threshold():
    almost = SensorReading(
        sensor_id="temp_098",
        sensor_type=SensorType.TEMPERATURE,
        value=22.0,
        confidence_score=0.98,
    )
    exact = SensorReading(
        sensor_id="temp_099",
        sensor_type=SensorType.TEMPERATURE,
        value=22.0,
        confidence_score=0.99,
    )

    assert almost.is_autonomous_action_safe() is False
    assert exact.is_autonomous_action_safe() is True
    assert almost.is_autonomous_action_safe(required_confidence=0.95) is True


def test_autonomous_action_threshold_cannot_go_below_floor():
    reading = SensorReading(
        sensor_id="temp_floor",
        sensor_type=SensorType.TEMPERATURE,
        value=22.0,
        confidence_score=0.94,
    )

    with pytest.raises(ValueError, match="cannot be below"):
        reading.is_autonomous_action_safe(required_confidence=0.5)


def test_confidence_levels_keep_documented_boundaries():
    assert ConfidenceLevel.from_score(0.95) == ConfidenceLevel.CRITICAL
    assert ConfidenceLevel.from_score(0.80) == ConfidenceLevel.HIGH
    assert ConfidenceLevel.from_score(0.60) == ConfidenceLevel.MEDIUM
    assert ConfidenceLevel.from_score(0.40) == ConfidenceLevel.LOW
    assert ConfidenceLevel.from_score(0.39) == ConfidenceLevel.NONE
