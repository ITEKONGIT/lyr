import os
from datetime import datetime

import pytest

from recognition.ollama_advisory import (
    DEFAULT_OLLAMA_MODEL,
    OllamaAdvisoryClient,
    find_ollama_executable,
)
from recognition.threshold_contracts import BreachState, BreachStatus


def test_live_ollama_analyzes_threshold_context():
    if os.environ.get("LYR_RUN_OLLAMA_TESTS") != "1":
        pytest.skip("set LYR_RUN_OLLAMA_TESTS=1 to run live Ollama integration")

    executable = find_ollama_executable()
    if not executable:
        pytest.skip("ollama executable not found")

    state = BreachState(
        rule_id="hot_room",
        sensor_ids=["temp_1"],
        status=BreachStatus.ACTIVE,
        first_triggered_at=datetime(2025, 1, 1, 12, 0, 0),
        last_triggered_at=datetime(2025, 1, 1, 12, 0, 5),
        rule_snapshot={
            "name": "Hot room",
            "sensor_type": "temperature",
            "enter_threshold": 36.0,
            "clear_threshold": 34.0,
            "severity": "warning",
            "mode": "log_only",
            "metadata": {
                "context_evaluation": {
                    "effects": [
                        {
                            "name": "hot_day_downgrade",
                            "status": "matched",
                            "reason": "Outdoor temperature suggests ambient heat",
                            "reading": {
                                "sensor_id": "outdoor_temp",
                                "sensor_type": "temperature",
                                "value": 39.0,
                            },
                        }
                    ]
                }
            },
        },
    )
    model = os.environ.get("LYR_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)

    advisory = OllamaAdvisoryClient(
        model=model,
        executable=executable,
        timeout_seconds=int(os.environ.get("LYR_OLLAMA_TIMEOUT_SECONDS", "60")),
    ).analyze_breach(state)

    assert advisory.available is True
    assert advisory.model == model
    assert advisory.raw_response
    assert advisory.metadata["payload"]["rule_id"] == "hot_room"
