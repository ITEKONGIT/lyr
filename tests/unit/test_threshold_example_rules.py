from datetime import datetime
from pathlib import Path

from recognition.sensor_contracts import SensorReading, SensorType
from recognition.sensor_history import HistoryStore
from recognition.threshold_contracts import BreachStatus
from recognition.threshold_gate import ThresholdGate
from recognition.threshold_rules import load_rules
from recognition.threshold_state import BreachStateStore


BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)
RULES_DIR = Path("recognition") / "rules"


def _reading(value, sensor_id, sensor_type):
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        value=value,
        timestamp=BASE_TIME,
        confidence_score=0.9,
    )


def _rule(rule_id):
    return next(rule for rule in load_rules(RULES_DIR) if rule.rule_id == rule_id)


def _gate(tmp_path, rule, history):
    return ThresholdGate(
        [rule],
        state_store=BreachStateStore(tmp_path / "state.db"),
        history_store=history,
    )


def test_example_rules_load_from_rules_directory():
    rule_ids = {rule.rule_id for rule in load_rules(RULES_DIR)}

    assert "fire_multi_evidence" in rule_ids
    assert "flash_flood_multi_evidence" in rule_ids
    assert "heatwave_context_multi_evidence" in rule_ids


def test_fire_example_rule_evaluates_breach(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(38.0, "humidity_1", SensorType.HUMIDITY))
    history.record(_reading(0.7, "smoke_1", SensorType.SMOKE))
    rule = _rule("fire_multi_evidence")
    gate = _gate(tmp_path, rule, history)

    state = gate.evaluate(_reading(37.0, "temp_1", SensorType.TEMPERATURE), now=BASE_TIME)[0]

    assert state.status == BreachStatus.ACTIVE
    evaluation = state.rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert evaluation["confidence"]["final_confidence"] == 0.95
    assert evaluation["policy"]["action"] == "continue"
    history.stop()


def test_fire_example_rule_does_not_breach_when_required_humidity_contradicts(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(55.0, "humidity_1", SensorType.HUMIDITY))
    rule = _rule("fire_multi_evidence")
    gate = _gate(tmp_path, rule, history)

    states = gate.evaluate(_reading(37.0, "temp_1", SensorType.TEMPERATURE), now=BASE_TIME)

    assert states == []
    history.stop()


def test_flood_example_rule_evaluates_breach(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(35.0, "rainfall_1", SensorType.RAINFALL))
    history.record(_reading(0.9, "soil_moisture_1", SensorType.SOIL_MOISTURE))
    rule = _rule("flash_flood_multi_evidence")
    gate = _gate(tmp_path, rule, history)

    state = gate.evaluate(_reading(1.8, "water_1", SensorType.WATER_LEVEL), now=BASE_TIME)[0]

    assert state.status == BreachStatus.ACTIVE
    evaluation = state.rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert evaluation["confidence"]["final_confidence"] == 0.95
    assert [item["sensor_type"] for item in evaluation["evidence"]["items"]] == [
        "water_level",
        "rainfall",
        "soil_moisture",
    ]
    history.stop()


def test_heatwave_example_rule_evaluates_log_only_context_state(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(39.0, "outdoor_temp", SensorType.TEMPERATURE))
    history.record(_reading(0.0, "smoke_1", SensorType.SMOKE))
    rule = _rule("heatwave_context_multi_evidence")
    gate = _gate(tmp_path, rule, history)

    state = gate.evaluate(_reading(37.0, "indoor_temp", SensorType.TEMPERATURE), now=BASE_TIME)[0]

    assert state.status == BreachStatus.ACTIVE
    assert state.rule_snapshot["mode"] == "log_only"
    evaluation = state.rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert evaluation["confidence"]["final_confidence"] == 0.45
    assert evaluation["evidence"]["items"][1]["role"] == "context"
    history.stop()


def test_flood_example_missing_rainfall_creates_stale_alert(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    rule = _rule("flash_flood_multi_evidence")
    gate = _gate(tmp_path, rule, history)

    state = gate.evaluate(_reading(1.8, "water_1", SensorType.WATER_LEVEL), now=BASE_TIME)[0]

    assert state.status == BreachStatus.ACTIVE
    assert state.rule_snapshot["severity"] == "warning"
    assert state.rule_snapshot["mode"] == "log_only"
    evaluation = state.rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert evaluation["policy"]["action"] == "stale_alert"
    assert evaluation["policy"]["issues"][0]["sensor_id"] == "rainfall_1"
    history.stop()
