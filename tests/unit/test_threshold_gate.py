from datetime import datetime, timedelta

from recognition.sensor_contracts import SensorReading, SensorType
from recognition.sensor_history import HistoryStore
from recognition.threshold_contracts import (
    BreachStatus,
    Rule,
    RuleCondition,
    RuleMode,
    RuleSeverity,
    StalenessPolicy,
)
from recognition.threshold_gate import ThresholdGate
from recognition.threshold_state import BreachStateStore


BASE_TIME = datetime(2025, 1, 1, 12, 0, 0)


def _rule(
    sustained_for_seconds=5,
    clear_delay_seconds=3,
    conditions=None,
    enabled=True,
):
    return Rule(
        rule_id="hot_room",
        name="Hot room",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        sustained_for_seconds=sustained_for_seconds,
        clear_delay_seconds=clear_delay_seconds,
        conditions=conditions or [],
        enabled=enabled,
    )


def _reading(value, sensor_id="temp_1", sensor_type=SensorType.TEMPERATURE):
    return SensorReading(
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        value=value,
        timestamp=BASE_TIME,
    )


def _gate(tmp_path, rule):
    store = BreachStateStore(tmp_path / "threshold_state.db")
    return ThresholdGate([rule], state_store=store), store


def _context_gate(tmp_path, rule, history):
    store = BreachStateStore(tmp_path / "threshold_state.db")
    return ThresholdGate([rule], state_store=store, history_store=history), store


def _history(tmp_path):
    return HistoryStore(tmp_path / "history.db")


def test_below_enter_threshold_without_state_stays_idle(tmp_path):
    rule = _rule()
    gate, store = _gate(tmp_path, rule)

    states = gate.evaluate(_reading(35.5), now=BASE_TIME)

    assert states == []
    assert store.get(rule.rule_id, ["temp_1"]) is None


def test_above_enter_threshold_creates_pending_sustain(tmp_path):
    rule = _rule(sustained_for_seconds=5)
    gate, store = _gate(tmp_path, rule)

    states = gate.evaluate(_reading(37.0), now=BASE_TIME)

    assert len(states) == 1
    assert states[0].status == BreachStatus.PENDING_SUSTAIN
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.PENDING_SUSTAIN
    assert stored.first_triggered_at == BASE_TIME
    assert stored.last_triggered_at == BASE_TIME


def test_sustained_condition_becomes_active(tmp_path):
    rule = _rule(sustained_for_seconds=5)
    gate, store = _gate(tmp_path, rule)

    gate.evaluate(_reading(37.0), now=BASE_TIME)
    states = gate.evaluate(
        _reading(37.5),
        now=BASE_TIME + timedelta(seconds=5),
    )

    assert states[0].status == BreachStatus.ACTIVE
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.ACTIVE
    assert stored.first_triggered_at == BASE_TIME
    assert stored.last_triggered_at == BASE_TIME + timedelta(seconds=5)


def test_condition_drops_before_sustain_resets_idle(tmp_path):
    rule = _rule(sustained_for_seconds=5)
    gate, store = _gate(tmp_path, rule)

    gate.evaluate(_reading(37.0), now=BASE_TIME)
    states = gate.evaluate(
        _reading(35.0),
        now=BASE_TIME + timedelta(seconds=2),
    )

    assert states[0].status == BreachStatus.IDLE
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.IDLE
    assert stored.first_triggered_at is None
    assert stored.last_triggered_at is None


def test_active_breach_below_clear_threshold_enters_clearing(tmp_path):
    rule = _rule(sustained_for_seconds=0, clear_delay_seconds=3)
    gate, store = _gate(tmp_path, rule)

    gate.evaluate(_reading(37.0), now=BASE_TIME)
    states = gate.evaluate(
        _reading(33.0),
        now=BASE_TIME + timedelta(seconds=1),
    )

    assert states[0].status == BreachStatus.CLEARING
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.CLEARING
    assert stored.clear_started_at == BASE_TIME + timedelta(seconds=1)


def test_clear_condition_sustained_long_enough_becomes_cleared(tmp_path):
    rule = _rule(sustained_for_seconds=0, clear_delay_seconds=3)
    gate, store = _gate(tmp_path, rule)

    gate.evaluate(_reading(37.0), now=BASE_TIME)
    gate.evaluate(_reading(33.0), now=BASE_TIME + timedelta(seconds=1))
    states = gate.evaluate(
        _reading(33.0),
        now=BASE_TIME + timedelta(seconds=4),
    )

    assert states[0].status == BreachStatus.CLEARED
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.CLEARED
    assert stored.clear_started_at == BASE_TIME + timedelta(seconds=1)
    assert stored.cleared_at == BASE_TIME + timedelta(seconds=4)


def test_clear_interrupted_by_enter_threshold_returns_active(tmp_path):
    rule = _rule(sustained_for_seconds=0, clear_delay_seconds=3)
    gate, store = _gate(tmp_path, rule)

    gate.evaluate(_reading(37.0), now=BASE_TIME)
    gate.evaluate(_reading(33.0), now=BASE_TIME + timedelta(seconds=1))
    states = gate.evaluate(
        _reading(37.0),
        now=BASE_TIME + timedelta(seconds=2),
    )

    assert states[0].status == BreachStatus.ACTIVE
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.ACTIVE
    assert stored.clear_started_at is None
    assert stored.cleared_at is None


def test_clear_threshold_prevents_boundary_flapping(tmp_path):
    rule = _rule(sustained_for_seconds=0, clear_delay_seconds=3)
    gate, store = _gate(tmp_path, rule)

    gate.evaluate(_reading(37.0), now=BASE_TIME)
    states = gate.evaluate(
        _reading(35.0),
        now=BASE_TIME + timedelta(seconds=1),
    )

    assert states[0].status == BreachStatus.ACTIVE
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.ACTIVE
    assert stored.clear_started_at is None


def test_zero_sustain_triggers_active_immediately(tmp_path):
    rule = _rule(sustained_for_seconds=0)
    gate, store = _gate(tmp_path, rule)

    states = gate.evaluate(_reading(37.0), now=BASE_TIME)

    assert states[0].status == BreachStatus.ACTIVE
    assert store.get(rule.rule_id, ["temp_1"]).status == BreachStatus.ACTIVE


def test_zero_clear_delay_clears_immediately(tmp_path):
    rule = _rule(sustained_for_seconds=0, clear_delay_seconds=0)
    gate, store = _gate(tmp_path, rule)

    gate.evaluate(_reading(37.0), now=BASE_TIME)
    states = gate.evaluate(
        _reading(33.0),
        now=BASE_TIME + timedelta(seconds=1),
    )

    assert states[0].status == BreachStatus.CLEARED
    stored = store.get(rule.rule_id, ["temp_1"])
    assert stored.status == BreachStatus.CLEARED
    assert stored.clear_started_at == BASE_TIME + timedelta(seconds=1)
    assert stored.cleared_at == BASE_TIME + timedelta(seconds=1)


def test_disabled_and_cross_sensor_rules_are_not_evaluated(tmp_path):
    disabled = _rule(enabled=False)
    cross_sensor = _rule(
        conditions=[
            RuleCondition(
                sensor_type=SensorType.HUMIDITY,
                operator=">",
                threshold=80,
            )
        ]
    )
    store = BreachStateStore(tmp_path / "threshold_state.db")
    gate = ThresholdGate([disabled, cross_sensor], state_store=store)

    states = gate.evaluate(_reading(40.0), now=BASE_TIME)

    assert states == []
    assert store.get(disabled.rule_id, ["temp_1"]) is None


def test_hot_day_context_downgrades_temperature_breach(tmp_path):
    history = _history(tmp_path)
    history.record(_reading(39.0, sensor_id="outdoor_temp"))
    rule = Rule(
        rule_id="hot_room",
        name="Hot room",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        severity=RuleSeverity.CRITICAL,
        mode=RuleMode.ESCALATE,
        sustained_for_seconds=0,
        metadata={
            "context_effects": [
                {
                    "name": "hot_day_downgrade",
                    "condition": {
                        "sensor_id": "outdoor_temp",
                        "sensor_type": "temperature",
                        "operator": ">",
                        "threshold": 35.0,
                        "history_window_seconds": 60,
                        "required": False,
                    },
                    "severity_delta": -2,
                    "mode": "log_only",
                    "reason": "Outdoor temperature suggests ambient heat",
                }
            ]
        },
    )
    gate, store = _context_gate(tmp_path, rule, history)

    states = gate.evaluate(_reading(37.0), now=BASE_TIME)

    assert states[0].status == BreachStatus.ACTIVE
    snapshot = store.get(rule.rule_id, ["temp_1"]).rule_snapshot
    assert snapshot["severity"] == RuleSeverity.WARNING.value
    assert snapshot["mode"] == RuleMode.LOG_ONLY.value
    effect = snapshot["metadata"]["context_evaluation"]["effects"][0]
    assert effect["status"] == "matched"
    assert effect["reason"] == "Outdoor temperature suggests ambient heat"
    history.stop()


def test_smoke_context_upgrades_temperature_breach(tmp_path):
    history = _history(tmp_path)
    history.record(_reading(0.95, sensor_id="smoke_1", sensor_type=SensorType.GAS))
    rule = Rule(
        rule_id="hot_room",
        name="Hot room",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        severity=RuleSeverity.WARNING,
        sustained_for_seconds=0,
        metadata={
            "context_effects": [
                {
                    "name": "smoke_corroboration",
                    "condition": {
                        "sensor_id": "smoke_1",
                        "sensor_type": "gas",
                        "operator": ">=",
                        "threshold": 0.9,
                        "history_window_seconds": 60,
                    },
                    "severity_delta": 2,
                    "reason": "Smoke/gas reading corroborates heat risk",
                }
            ]
        },
    )
    gate, store = _context_gate(tmp_path, rule, history)

    gate.evaluate(_reading(37.0), now=BASE_TIME)

    snapshot = store.get(rule.rule_id, ["temp_1"]).rule_snapshot
    assert snapshot["severity"] == RuleSeverity.CRITICAL.value
    effect = snapshot["metadata"]["context_evaluation"]["effects"][0]
    assert effect["status"] == "matched"
    assert effect["reading"]["sensor_id"] == "smoke_1"
    history.stop()


def test_humidity_drop_context_upgrades_temperature_breach(tmp_path):
    history = _history(tmp_path)
    history.record(_reading(18.0, sensor_id="humidity_1", sensor_type=SensorType.HUMIDITY))
    rule = Rule(
        rule_id="hot_room",
        name="Hot room",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        severity=RuleSeverity.WARNING,
        sustained_for_seconds=0,
        metadata={
            "context_effects": [
                {
                    "name": "humidity_drop",
                    "condition": {
                        "sensor_id": "humidity_1",
                        "sensor_type": "humidity",
                        "operator": "<",
                        "threshold": 20.0,
                        "history_window_seconds": 60,
                    },
                    "severity_delta": 1,
                    "reason": "Low humidity corroborates heat anomaly",
                }
            ]
        },
    )
    gate, store = _context_gate(tmp_path, rule, history)

    gate.evaluate(_reading(37.0), now=BASE_TIME)

    snapshot = store.get(rule.rule_id, ["temp_1"]).rule_snapshot
    assert snapshot["severity"] == RuleSeverity.HIGH.value
    effect = snapshot["metadata"]["context_evaluation"]["effects"][0]
    assert effect["status"] == "matched"
    history.stop()


def test_missing_optional_context_does_not_crash_or_suppress(tmp_path):
    history = _history(tmp_path)
    rule = Rule(
        rule_id="hot_room",
        name="Hot room",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        severity=RuleSeverity.WARNING,
        sustained_for_seconds=0,
        metadata={
            "context_effects": [
                {
                    "name": "optional_outdoor_context",
                    "condition": {
                        "sensor_id": "outdoor_temp",
                        "sensor_type": "temperature",
                        "operator": ">",
                        "threshold": 35.0,
                        "required": False,
                    },
                    "severity_delta": -1,
                }
            ]
        },
    )
    gate, store = _context_gate(tmp_path, rule, history)

    states = gate.evaluate(_reading(37.0), now=BASE_TIME)

    assert states[0].status == BreachStatus.ACTIVE
    snapshot = store.get(rule.rule_id, ["temp_1"]).rule_snapshot
    assert snapshot["severity"] == RuleSeverity.WARNING.value
    effect = snapshot["metadata"]["context_evaluation"]["effects"][0]
    assert effect["status"] == "missing"
    history.stop()


def test_missing_required_context_fail_closed_suppresses_breach(tmp_path):
    history = _history(tmp_path)
    rule = Rule(
        rule_id="hot_room",
        name="Hot room",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        sustained_for_seconds=0,
        staleness_policy=StalenessPolicy.FAIL_CLOSED,
        metadata={
            "context_effects": [
                {
                    "name": "required_smoke_context",
                    "condition": {
                        "sensor_id": "smoke_1",
                        "sensor_type": "gas",
                        "operator": ">=",
                        "threshold": 0.9,
                        "required": True,
                    },
                    "severity_delta": 2,
                }
            ]
        },
    )
    gate, store = _context_gate(tmp_path, rule, history)

    states = gate.evaluate(_reading(37.0), now=BASE_TIME)

    assert states == []
    assert store.get(rule.rule_id, ["temp_1"]) is None
    history.stop()
