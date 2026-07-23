import os
from datetime import datetime

import pytest

from recognition.ollama_advisory import (
    DEFAULT_OLLAMA_MODEL,
    OllamaAdvisoryClient,
    find_ollama_executable,
)
from recognition.sensor_contracts import SensorReading, SensorType
from recognition.sensor_history import HistoryStore
from recognition.threshold_contracts import EvidenceRole, Rule, RuleCondition
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


def test_live_ollama_annotates_cross_sensor_breach(tmp_path):
    if os.environ.get("LYR_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("set LYR_RUN_OLLAMA_TESTS=1 to run live Ollama integration")

    executable = find_ollama_executable()
    if not executable:
        pytest.skip("ollama executable not found")

    history = HistoryStore(tmp_path / "history.db")
    history.record(_reading(38.0, "humidity_1", SensorType.HUMIDITY))
    rule = Rule(
        rule_id="live_fire_ai",
        name="Live fire advisory",
        sensor_type=SensorType.TEMPERATURE,
        enter_threshold=36.0,
        clear_threshold=34.0,
        sustained_for_seconds=0,
        clear_delay_seconds=0,
        conditions=[
            RuleCondition(
                sensor_type=SensorType.HUMIDITY,
                sensor_id="humidity_1",
                operator="<",
                threshold=40.0,
                history_window_seconds=10,
                required=True,
                role=EvidenceRole.CORROBORATES,
                weight=0.1,
            )
        ],
        metadata={"base_confidence": 0.5, "primary_weight": 0.2},
    )
    advisory_client = OllamaAdvisoryClient(
        model=os.environ.get("LYR_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        executable=executable,
        timeout_seconds=int(os.environ.get("LYR_OLLAMA_TIMEOUT_SECONDS", "60")),
    )
    gate = ThresholdGate(
        [rule],
        state_store=BreachStateStore(tmp_path / "state.db"),
        history_store=history,
        advisory_client=advisory_client,
    )

    state = gate.evaluate(_reading(37.0, "temp_1", SensorType.TEMPERATURE), now=BASE_TIME)[0]

    advisory = state.rule_snapshot["metadata"]["ai_advisory"]
    assert advisory["available"] is True
    assert advisory["model"] == advisory_client.model
    assert advisory["raw_response"]
    assert "cross_sensor_evaluation" in advisory["metadata"]["payload"]
    history.stop()
