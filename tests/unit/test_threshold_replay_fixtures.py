import json
from datetime import datetime, timedelta
from pathlib import Path

from recognition.sensor_contracts import SensorReading, SensorType
from recognition.sensor_history import HistoryStore
from recognition.threshold_rules import load_rules
from recognition.threshold_gate import ThresholdGate
from recognition.threshold_state import BreachStateStore


BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)
FIXTURE_DIR = Path("tests") / "fixtures" / "replay"
RULES_DIR = Path("recognition") / "rules"


def _load_fixture(name):
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _reading(data):
    return SensorReading(
        sensor_id=data["sensor_id"],
        sensor_type=SensorType.from_string(data["sensor_type"]),
        value=data["value"],
        timestamp=BASE_TIME + timedelta(seconds=data.get("offset_seconds", 0)),
        confidence_score=data.get("confidence_score", 0.9),
    )


def _run_fixture(tmp_path, name):
    fixture = _load_fixture(name)
    rule = next(rule for rule in load_rules(RULES_DIR) if rule.rule_id == fixture["rule_id"])
    history = HistoryStore(tmp_path / f"{name}.db")
    gate = ThresholdGate(
        [rule],
        state_store=BreachStateStore(tmp_path / f"{name}.state.db"),
        history_store=history,
    )
    last_states = []
    for item in fixture["readings"]:
        reading = _reading(item)
        if item.get("trigger"):
            last_states = gate.evaluate(reading, now=reading.timestamp)
        else:
            history.record(reading)
    history.stop()
    return fixture, last_states


def test_replay_fixture_files_load():
    fixtures = sorted(path.name for path in FIXTURE_DIR.glob("*.json"))

    assert fixtures == [
        "false_smoke_sequence.json",
        "fire_clear_sequence.json",
        "fire_sequence.json",
        "flood_sequence.json",
        "heatwave_sequence.json",
        "missing_sensor_sequence.json",
    ]
    for fixture_name in fixtures:
        fixture = _load_fixture(fixture_name)
        assert fixture["rule_id"]
        assert fixture["readings"]
        assert fixture["expected"]


def test_replay_fire_sequence_produces_active_high_confidence_breach(tmp_path):
    fixture, states = _run_fixture(tmp_path, "fire_sequence.json")

    evaluation = states[0].rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert states[0].status.value == fixture["expected"]["status"]
    assert evaluation["policy"]["action"] == fixture["expected"]["policy_action"]
    assert evaluation["confidence"]["final_confidence"] >= fixture["expected"]["min_confidence"]
    assert [item["sensor_type"] for item in evaluation["evidence"]["items"]] == [
        "temperature",
        "humidity",
        "smoke",
    ]


def test_replay_flash_flood_sequence_produces_active_breach(tmp_path):
    fixture, states = _run_fixture(tmp_path, "flood_sequence.json")

    evaluation = states[0].rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert states[0].status.value == fixture["expected"]["status"]
    assert evaluation["policy"]["action"] == fixture["expected"]["policy_action"]
    assert evaluation["confidence"]["final_confidence"] >= fixture["expected"]["min_confidence"]


def test_replay_heatwave_sequence_logs_context_event(tmp_path):
    fixture, states = _run_fixture(tmp_path, "heatwave_sequence.json")

    evaluation = states[0].rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert states[0].status.value == fixture["expected"]["status"]
    assert states[0].rule_snapshot["mode"] == fixture["expected"]["mode"]
    assert evaluation["policy"]["action"] == fixture["expected"]["policy_action"]
    assert evaluation["confidence"]["final_confidence"] <= fixture["expected"]["max_confidence"]


def test_replay_false_smoke_sequence_does_not_trigger_fire_breach(tmp_path):
    fixture, states = _run_fixture(tmp_path, "false_smoke_sequence.json")

    assert len(states) == fixture["expected"]["states"]


def test_replay_missing_required_sensor_sequence_creates_stale_alert(tmp_path):
    fixture, states = _run_fixture(tmp_path, "missing_sensor_sequence.json")

    evaluation = states[0].rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert states[0].status.value == fixture["expected"]["status"]
    assert states[0].rule_snapshot["severity"] == fixture["expected"]["severity"]
    assert states[0].rule_snapshot["mode"] == fixture["expected"]["mode"]
    assert evaluation["policy"]["action"] == fixture["expected"]["policy_action"]
    assert evaluation["policy"]["issues"][0]["sensor_id"] == "rainfall_1"


def test_replay_active_breach_clears_after_clear_sequence(tmp_path):
    fixture, states = _run_fixture(tmp_path, "fire_clear_sequence.json")

    assert states[0].status.value == fixture["expected"]["status"]
    assert states[0].cleared_at == BASE_TIME + timedelta(seconds=2)
