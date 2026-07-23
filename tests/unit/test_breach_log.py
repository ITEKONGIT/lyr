from datetime import datetime, timedelta, timezone

import pytest

from recognition.breach_log import BreachLogStore, MAX_BREACH_QUERY_LIMIT
from recognition.threshold_contracts import BreachLogEntry, RuleSeverity


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _entry(
    rule_id="temperature_high",
    sensor_ids=None,
    triggered_at=None,
    severity=RuleSeverity.WARNING,
):
    return BreachLogEntry(
        rule_id=rule_id,
        sensor_ids=sensor_ids or ["temp_1"],
        triggered_at=triggered_at or _now(),
        severity=severity,
        context_snapshot={"reason": "threshold crossed"},
    )


def test_appends_and_retrieves_breach_log_entry(tmp_path):
    store = BreachLogStore(tmp_path / "breach_log.db")
    entry = _entry(severity=RuleSeverity.HIGH)

    breach_id = store.append(entry)
    restored = store.get(breach_id)

    assert restored is not None
    assert restored.breach_id == breach_id
    assert restored.rule_id == "temperature_high"
    assert restored.sensor_ids == ["temp_1"]
    assert restored.severity == RuleSeverity.HIGH
    assert restored.context_snapshot == {"reason": "threshold crossed"}
    store.close()


def test_queries_by_rule_sensor_and_time_range(tmp_path):
    store = BreachLogStore(tmp_path / "breach_log.db")
    base = _now() - timedelta(minutes=10)
    store.append(_entry("temperature_high", ["temp_1"], base))
    store.append(_entry("humidity_low", ["humidity_1"], base + timedelta(minutes=5)))
    store.append(_entry("temperature_high", ["temp_2"], base + timedelta(minutes=8)))

    by_rule = store.query(rule_id="temperature_high", limit=10)
    by_sensor = store.query(sensor_id="humidity_1", limit=10)
    by_time = store.query(
        since=base + timedelta(minutes=4),
        until=base + timedelta(minutes=9),
        limit=10,
    )

    assert {entry.sensor_ids[0] for entry in by_rule} == {"temp_1", "temp_2"}
    assert [entry.rule_id for entry in by_sensor] == ["humidity_low"]
    assert {entry.rule_id for entry in by_time} == {"humidity_low", "temperature_high"}
    store.close()


def test_marks_human_reviewed(tmp_path):
    store = BreachLogStore(tmp_path / "breach_log.db")
    breach_id = store.append(_entry())

    updated = store.mark_human_reviewed(breach_id, notes="Checked by operator")
    restored = store.get(breach_id)

    assert updated is True
    assert restored.human_reviewed is True
    assert restored.human_notes == "Checked by operator"
    store.close()


def test_attaches_tier3_decision_and_action(tmp_path):
    store = BreachLogStore(tmp_path / "breach_log.db")
    breach_id = store.append(_entry())

    tier3_updated = store.attach_tier3_decision(
        breach_id,
        {"confidence": 0.84, "hypothesis": "ambient heat"},
    )
    action_updated = store.attach_action_taken(
        breach_id,
        {"type": "notify", "target": "operator"},
    )
    restored = store.get(breach_id)

    assert tier3_updated is True
    assert action_updated is True
    assert restored.escalated_to_tier3 is True
    assert restored.tier3_decision["confidence"] == 0.84
    assert restored.action_taken["type"] == "notify"
    store.close()


def test_breach_log_survives_store_restart(tmp_path):
    db_path = tmp_path / "breach_log.db"
    first = BreachLogStore(db_path)
    breach_id = first.append(_entry())
    first.close()

    second = BreachLogStore(db_path)
    restored = second.get(breach_id)

    assert restored is not None
    assert restored.breach_id == breach_id
    second.close()


def test_query_limit_is_capped(tmp_path):
    store = BreachLogStore(tmp_path / "breach_log.db")

    with pytest.raises(ValueError, match=f"cannot exceed {MAX_BREACH_QUERY_LIMIT}"):
        store.query(limit=MAX_BREACH_QUERY_LIMIT + 1)

    store.close()
