import json

import pytest

from recognition.sensor_contracts import SensorType
from recognition.threshold_contracts import RuleMode
from recognition.threshold_rules import RuleLoadError, load_rules


def _rule(**overrides):
    data = {
        "rule_id": "temperature_high",
        "name": "Temperature high",
        "sensor_type": "temperature",
        "enter_threshold": 36.0,
        "clear_threshold": 34.0,
        "severity": "warning",
        "mode": "log_only",
        "sustained_for_seconds": 5.0,
        "clear_delay_seconds": 3.0,
        "enabled": True,
    }
    data.update(overrides)
    return data


def _write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def test_loads_valid_rule_file(tmp_path):
    path = _write_json(tmp_path / "rules.json", {"rules": [_rule()]})

    rules = load_rules(path)

    assert len(rules) == 1
    assert rules[0].rule_id == "temperature_high"
    assert rules[0].sensor_type == SensorType.TEMPERATURE
    assert rules[0].mode == RuleMode.LOG_ONLY


def test_loads_all_json_files_from_directory(tmp_path):
    _write_json(tmp_path / "a.json", {"rules": [_rule(rule_id="a_rule")]})
    _write_json(tmp_path / "b.json", {"rules": [_rule(rule_id="b_rule")]})

    rules = load_rules(tmp_path)

    assert [rule.rule_id for rule in rules] == ["a_rule", "b_rule"]


def test_rejects_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not-valid-json")

    with pytest.raises(RuleLoadError, match="Malformed JSON"):
        load_rules(path)


def test_rejects_duplicate_rule_id_across_files(tmp_path):
    _write_json(tmp_path / "a.json", {"rules": [_rule(rule_id="dupe")]})
    _write_json(tmp_path / "b.json", {"rules": [_rule(rule_id="dupe")]})

    with pytest.raises(RuleLoadError, match="Duplicate rule_id"):
        load_rules(tmp_path)


def test_rejects_invalid_rule_via_contract_validation(tmp_path):
    path = _write_json(
        tmp_path / "invalid.json",
        {"rules": [_rule(clear_threshold=40.0)]},
    )

    with pytest.raises(RuleLoadError, match="clear_threshold"):
        load_rules(path)


def test_rejects_unimplemented_trend_operator_at_rule_load(tmp_path):
    path = _write_json(
        tmp_path / "trend.json",
        {
            "rules": [
                _rule(
                    conditions=[
                        {
                            "sensor_type": "humidity",
                            "operator": "dropping",
                            "threshold": 5.0,
                        }
                    ]
                )
            ]
        },
    )

    with pytest.raises(RuleLoadError, match="Trend operators are not implemented"):
        load_rules(path)


def test_loads_only_enabled_rules_by_default(tmp_path):
    path = _write_json(
        tmp_path / "rules.json",
        {
            "rules": [
                _rule(rule_id="enabled_rule", enabled=True),
                _rule(rule_id="disabled_rule", enabled=False),
            ]
        },
    )

    rules = load_rules(path)

    assert [rule.rule_id for rule in rules] == ["enabled_rule"]


def test_preserves_disabled_rules_when_requested(tmp_path):
    path = _write_json(
        tmp_path / "rules.json",
        {
            "rules": [
                _rule(rule_id="enabled_rule", enabled=True),
                _rule(rule_id="disabled_rule", enabled=False),
            ]
        },
    )

    rules = load_rules(path, include_disabled=True)

    assert [rule.rule_id for rule in rules] == [
        "enabled_rule",
        "disabled_rule",
    ]


def test_rejects_unsupported_rule_file_type(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text("rules: []")

    with pytest.raises(RuleLoadError, match="Unsupported rule file type"):
        load_rules(path)
