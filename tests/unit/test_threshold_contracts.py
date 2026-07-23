from datetime import datetime, timezone

import pytest

from recognition.sensor_contracts import SensorType
from recognition.threshold_contracts import (
    BreachLogEntry,
    BreachState,
    BreachStatus,
    ComparisonOperator,
    Rule,
    RuleCondition,
    RuleMode,
    RuleSeverity,
    StalenessPolicy,
)


def _temperature_rule(**overrides):
    data = {
        "rule_id": "temperature_high",
        "name": "Temperature high",
        "sensor_type": SensorType.TEMPERATURE,
        "enter_threshold": 36.0,
        "clear_threshold": 34.0,
        "severity": RuleSeverity.WARNING,
        "mode": RuleMode.LOG_ONLY,
        "sustained_for_seconds": 5.0,
        "clear_delay_seconds": 3.0,
    }
    data.update(overrides)
    return Rule(**data)


def test_accepts_valid_single_sensor_temperature_rule():
    rule = _temperature_rule()

    assert rule.rule_id == "temperature_high"
    assert rule.sensor_type == SensorType.TEMPERATURE
    assert rule.enter_threshold == 36.0
    assert rule.clear_threshold == 34.0
    assert rule.sustained_for_seconds == 5.0
    assert rule.clear_delay_seconds == 3.0
    assert rule.mode == RuleMode.LOG_ONLY
    assert rule.is_cross_sensor is False


def test_rejects_clear_threshold_at_or_above_enter_threshold():
    with pytest.raises(ValueError, match="clear_threshold must be below"):
        _temperature_rule(clear_threshold=36.0)

    with pytest.raises(ValueError, match="clear_threshold must be below"):
        _temperature_rule(clear_threshold=37.0)


def test_rejects_negative_timers():
    with pytest.raises(ValueError, match="sustained_for_seconds"):
        _temperature_rule(sustained_for_seconds=-1)

    with pytest.raises(ValueError, match="clear_delay_seconds"):
        _temperature_rule(clear_delay_seconds=-1)


def test_rejects_unsupported_comparison_operator():
    with pytest.raises(ValueError, match="Unsupported comparison operator"):
        RuleCondition(
            sensor_type=SensorType.TEMPERATURE,
            operator="python_eval",
            threshold=36.0,
        )


def test_accepts_valid_cross_sensor_rule_with_alert_stale():
    rule = _temperature_rule(
        rule_id="fire_context",
        name="Fire context",
        conditions=[
            RuleCondition(
                sensor_type=SensorType.HUMIDITY,
                operator=ComparisonOperator.DROPPING,
                threshold=5.0,
                history_window_seconds=60,
            )
        ],
        staleness_policy=StalenessPolicy.ALERT_STALE,
        stale_age_seconds=60,
    )

    assert rule.is_cross_sensor is True
    assert rule.staleness_policy == StalenessPolicy.ALERT_STALE
    assert rule.conditions[1].sensor_type == SensorType.HUMIDITY


def test_rule_serialization_round_trips():
    original = _temperature_rule(
        conditions=[
            RuleCondition(
                sensor_type=SensorType.HUMIDITY,
                operator=ComparisonOperator.LT,
                threshold=40.0,
            )
        ],
        context_gates=[
            RuleCondition(
                sensor_type=SensorType.AMBIENT_LIGHT,
                operator=ComparisonOperator.GT,
                threshold=700.0,
                required=False,
            )
        ],
        metadata={"owner": "phase_2_1"},
    )

    restored = Rule.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
    assert restored.is_cross_sensor is True


def test_breach_state_requires_sensor_ids_and_exposes_state_key():
    state = BreachState(
        rule_id="temperature_high",
        sensor_ids=["temp_b", "temp_a"],
        status=BreachStatus.PENDING_SUSTAIN,
    )

    assert state.state_key == "temperature_high:temp_a,temp_b"
    assert state.to_dict()["status"] == "pending_sustain"

    with pytest.raises(ValueError, match="sensor_ids"):
        BreachState(rule_id="temperature_high", sensor_ids=[])


def test_breach_log_entry_serializes_audit_fields():
    entry = BreachLogEntry(
        rule_id="temperature_high",
        sensor_ids=["temp_1"],
        triggered_at=datetime.now(timezone.utc).replace(tzinfo=None),
        severity=RuleSeverity.HIGH,
        context_snapshot={"reason": "temperature crossed threshold"},
        escalated_to_tier3=True,
        tier3_decision={"confidence": 0.82},
    )

    data = entry.to_dict()

    assert data["breach_id"]
    assert data["rule_id"] == "temperature_high"
    assert data["sensor_ids"] == ["temp_1"]
    assert data["severity"] == "high"
    assert data["context_snapshot"]["reason"] == "temperature crossed threshold"
    assert data["escalated_to_tier3"] is True


def test_phase_2_1_does_not_define_evaluator_logic():
    import recognition.threshold_contracts as contracts

    assert not hasattr(contracts, "ThresholdGate")
    assert not hasattr(contracts, "evaluate")
