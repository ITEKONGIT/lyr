from datetime import datetime, timedelta

from recognition.sensor_contracts import SensorReading, SensorType
from recognition.sensor_history import HistoryStore
from recognition.threshold_contracts import EvidenceRole, Rule, RuleCondition
from recognition.threshold_evidence import EvidenceStatus, build_evidence_snapshot


BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)


def _reading(
    value,
    sensor_id,
    sensor_type,
    timestamp=BASE_TIME,
    confidence_score=0.9,
):
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        value=value,
        timestamp=timestamp,
        confidence_score=confidence_score,
        source="unit_test",
        location={"room": 4},
    )


def _history(tmp_path):
    return HistoryStore(tmp_path / "history.db")


def _fire_rule():
    return Rule(
        rule_id="fire_detection",
        name="Potential fire detection",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        conditions=[
            RuleCondition(
                sensor_type=SensorType.HUMIDITY,
                sensor_id="humidity_1",
                operator="<",
                threshold=40.0,
                history_window_seconds=10,
                required=True,
                role=EvidenceRole.CORROBORATES,
                weight=0.10,
                reason="Low humidity corroborates heat risk",
            ),
            RuleCondition(
                sensor_type=SensorType.SMOKE,
                sensor_id="smoke_1",
                operator=">",
                threshold=0.5,
                history_window_seconds=5,
                required=False,
                role=EvidenceRole.CORROBORATES,
                weight=0.30,
            ),
        ],
        metadata={"primary_weight": 0.20},
    )


def test_snapshot_includes_matched_primary_and_related_evidence(tmp_path):
    history = _history(tmp_path)
    history.record(_reading(38.0, "humidity_1", SensorType.HUMIDITY))
    rule = _fire_rule()

    snapshot = build_evidence_snapshot(
        _reading(37.0, "temp_1", SensorType.TEMPERATURE),
        rule,
        history_store=history,
        now=BASE_TIME,
    )

    primary, humidity, smoke = snapshot.items

    assert snapshot.rule_id == "fire_detection"
    assert primary.status == EvidenceStatus.MATCHED
    assert primary.reading["sensor_id"] == "temp_1"
    assert primary.weight == 0.20
    assert humidity.status == EvidenceStatus.MATCHED
    assert humidity.reading["value"] == 38.0
    assert humidity.age_seconds == 0.0
    assert smoke.status == EvidenceStatus.MISSING
    assert smoke.required is False
    history.stop()


def test_snapshot_marks_related_condition_not_matched(tmp_path):
    history = _history(tmp_path)
    history.record(_reading(55.0, "humidity_1", SensorType.HUMIDITY))
    rule = _fire_rule()

    snapshot = build_evidence_snapshot(
        _reading(37.0, "temp_1", SensorType.TEMPERATURE),
        rule,
        history_store=history,
        now=BASE_TIME,
    )

    assert snapshot.items[1].status == EvidenceStatus.NOT_MATCHED
    assert snapshot.items[1].reading["sensor_id"] == "humidity_1"
    history.stop()


def test_snapshot_marks_old_related_reading_stale(tmp_path):
    history = _history(tmp_path)
    history.record(
        _reading(
            34.0,
            "rainfall_1",
            SensorType.RAINFALL,
            timestamp=BASE_TIME - timedelta(seconds=120),
        )
    )
    rule = Rule(
        rule_id="flash_flood_detection",
        name="Flash flood detection",
        sensor_type=SensorType.WATER_LEVEL,
        enter_threshold=1.5,
        clear_threshold=1.0,
        conditions=[
            RuleCondition(
                sensor_type=SensorType.RAINFALL,
                sensor_id="rainfall_1",
                operator=">=",
                threshold=30.0,
                history_window_seconds=60,
                required=True,
                role=EvidenceRole.CORROBORATES,
                weight=0.25,
            )
        ],
    )

    snapshot = build_evidence_snapshot(
        _reading(1.7, "water_1", SensorType.WATER_LEVEL),
        rule,
        history_store=history,
        now=BASE_TIME,
    )

    rainfall = snapshot.items[1]
    assert rainfall.status == EvidenceStatus.STALE
    assert rainfall.age_seconds == 120.0
    assert rainfall.reading["value"] == 34.0
    history.stop()


def test_missing_optional_evidence_is_explicit_and_nonfatal(tmp_path):
    history = _history(tmp_path)
    rule = _fire_rule()

    snapshot = build_evidence_snapshot(
        _reading(37.0, "temp_1", SensorType.TEMPERATURE),
        rule,
        history_store=history,
        now=BASE_TIME,
    )

    humidity = snapshot.items[1]
    smoke = snapshot.items[2]
    assert humidity.status == EvidenceStatus.MISSING
    assert humidity.required is True
    assert smoke.status == EvidenceStatus.MISSING
    assert smoke.required is False
    history.stop()


def test_snapshot_serializes_human_explainable_metadata(tmp_path):
    history = _history(tmp_path)
    history.record(_reading(0.7, "smoke_1", SensorType.SMOKE))
    rule = _fire_rule()

    snapshot = build_evidence_snapshot(
        _reading(35.0, "temp_1", SensorType.TEMPERATURE),
        rule,
        history_store=history,
        now=BASE_TIME,
    )
    data = snapshot.to_dict()

    assert data["triggering_sensor_id"] == "temp_1"
    assert data["items"][0]["status"] == "not_matched"
    assert data["items"][0]["operator"] == ">"
    assert data["items"][0]["threshold"] == 36.0
    assert data["items"][2]["status"] == "matched"
    assert data["items"][2]["role"] == "corroborates"
    assert data["items"][2]["weight"] == 0.30
    assert data["items"][2]["reading"]["source"] == "unit_test"
    assert data["items"][2]["reading"]["location"] == {"room": 4}
    history.stop()
