from datetime import datetime
from unittest.mock import Mock

from recognition.ollama_advisory import OllamaAdvisory
from recognition.sensor_contracts import SensorReading, SensorType
from recognition.sensor_history import HistoryStore
from recognition.threshold_ai import attach_ai_advisory
from recognition.threshold_contracts import (
    BreachState,
    BreachStatus,
    EvidenceRole,
    Rule,
    RuleCondition,
    StalenessPolicy,
)
from recognition.threshold_gate import ThresholdGate
from recognition.threshold_state import BreachStateStore


BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)


def _reading(value, sensor_id, sensor_type):
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        value=value,
        timestamp=BASE_TIME,
        confidence_score=0.9,
    )


def _fire_rule(policy=StalenessPolicy.ALERT_STALE):
    return Rule(
        rule_id="fire_ai",
        name="Fire with AI advisory",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        sustained_for_seconds=0,
        clear_delay_seconds=0,
        staleness_policy=policy,
        conditions=[
            RuleCondition(
                sensor_type=SensorType.HUMIDITY,
                sensor_id="humidity_1",
                operator="<",
                threshold=40,
                history_window_seconds=10,
                required=True,
                role=EvidenceRole.CORROBORATES,
                weight=0.1,
            )
        ],
        metadata={"base_confidence": 0.5, "primary_weight": 0.2},
    )


def _gate(tmp_path, rule, history, advisory_client=None):
    return ThresholdGate(
        [rule],
        state_store=BreachStateStore(tmp_path / "state.db"),
        history_store=history,
        advisory_client=advisory_client,
    )


def test_attach_ai_advisory_preserves_deterministic_state():
    state = BreachState(
        rule_id="fire_ai",
        sensor_ids=["temp_1"],
        status=BreachStatus.ACTIVE,
        rule_snapshot={"metadata": {"cross_sensor_evaluation": {"policy": {"action": "continue"}}}},
    )
    client = Mock()
    client.analyze_breach.return_value = OllamaAdvisory(
        available=True,
        model="qwen2.5-coder:7b",
        summary="Review likely fire context",
        risk_level="high",
        recommended_action="dismiss",
        confidence=0.2,
    )

    annotated = attach_ai_advisory(state, client)

    assert annotated.status == BreachStatus.ACTIVE
    assert annotated.rule_id == state.rule_id
    assert annotated.rule_snapshot["metadata"]["cross_sensor_evaluation"] == {
        "policy": {"action": "continue"}
    }
    advisory = annotated.rule_snapshot["metadata"]["ai_advisory"]
    assert advisory["recommended_action"] == "dismiss"
    assert advisory["available"] is True


def test_cross_sensor_breach_attaches_ai_advisory_metadata(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(38.0, "humidity_1", SensorType.HUMIDITY))
    client = Mock()
    client.analyze_breach.return_value = OllamaAdvisory(
        available=True,
        model="qwen2.5-coder:7b",
        summary="Cross-sensor fire candidate",
        risk_level="high",
        recommended_action="escalate",
        confidence=0.82,
    )
    rule = _fire_rule()
    gate = _gate(tmp_path, rule, history, advisory_client=client)

    state = gate.evaluate(_reading(37.0, "temp_1", SensorType.TEMPERATURE), now=BASE_TIME)[0]

    client.analyze_breach.assert_called_once()
    advisory_input = client.analyze_breach.call_args.args[0]
    assert "cross_sensor_evaluation" in advisory_input.rule_snapshot["metadata"]
    assert state.rule_snapshot["metadata"]["ai_advisory"]["summary"] == "Cross-sensor fire candidate"
    stored = gate.state_store.get(rule.rule_id, ["humidity_1", "temp_1"])
    assert stored.rule_snapshot["metadata"]["ai_advisory"]["risk_level"] == "high"
    history.stop()


def test_ai_unavailable_does_not_change_deterministic_breach_state(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(38.0, "humidity_1", SensorType.HUMIDITY))
    client = Mock()
    client.analyze_breach.return_value = OllamaAdvisory(
        available=False,
        model="qwen2.5-coder:7b",
        error="timeout",
    )
    rule = _fire_rule()
    gate = _gate(tmp_path, rule, history, advisory_client=client)

    state = gate.evaluate(_reading(37.0, "temp_1", SensorType.TEMPERATURE), now=BASE_TIME)[0]

    assert state.status == BreachStatus.ACTIVE
    evaluation = state.rule_snapshot["metadata"]["cross_sensor_evaluation"]
    assert evaluation["confidence"]["final_confidence"] == 0.8
    assert state.rule_snapshot["metadata"]["ai_advisory"]["available"] is False
    assert state.rule_snapshot["metadata"]["ai_advisory"]["error"] == "timeout"
    history.stop()


def test_ai_exception_is_recorded_not_raised(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(38.0, "humidity_1", SensorType.HUMIDITY))
    client = Mock()
    client.model = "qwen2.5-coder:7b"
    client.analyze_breach.side_effect = TimeoutError("local model timed out")
    rule = _fire_rule()
    gate = _gate(tmp_path, rule, history, advisory_client=client)

    state = gate.evaluate(_reading(37.0, "temp_1", SensorType.TEMPERATURE), now=BASE_TIME)[0]

    assert state.status == BreachStatus.ACTIVE
    advisory = state.rule_snapshot["metadata"]["ai_advisory"]
    assert advisory["available"] is False
    assert advisory["error"] == "local model timed out"
    history.stop()


def test_ai_not_called_when_deterministic_rule_suppresses(tmp_path):
    history = HistoryStore(tmp_path / "history.db")
    client = Mock()
    rule = _fire_rule(policy=StalenessPolicy.FAIL_CLOSED)
    gate = _gate(tmp_path, rule, history, advisory_client=client)

    states = gate.evaluate(_reading(37.0, "temp_1", SensorType.TEMPERATURE), now=BASE_TIME)

    assert states == []
    client.analyze_breach.assert_not_called()
    history.stop()
