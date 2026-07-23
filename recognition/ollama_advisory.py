"""
Optional Ollama advisory adapter for threshold breach analysis.

This module does not participate in deterministic Tier 2 decisions. It packages
breach state and context metadata for a local model, then returns advisory text
for review, Tier 3 enrichment, or Tier 4 audit.
"""

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .threshold_contracts import BreachState


DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 30
COMMON_OLLAMA_PATH = Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"
TRUSTED_SOURCE_VALUES = {
    "camera_module",
    "face_recognition_pipeline",
    "phone_browser",
    "sensor_api",
    "unit_test",
    "weather_api",
}


@dataclass
class OllamaAdvisory:
    """Structured advisory output from a local Ollama model."""

    available: bool
    model: str
    summary: str = ""
    risk_level: str = "unknown"
    recommended_action: str = "review"
    confidence: float = 0.0
    raw_response: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "model": self.model,
            "summary": self.summary,
            "risk_level": self.risk_level,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
            "error": self.error,
            "metadata": self.metadata,
        }


class OllamaAdvisoryClient:
    """Small CLI-backed client for local model advisory analysis."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        executable: Optional[str] = None,
        timeout_seconds: int = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    ):
        self.model = model
        self.executable = executable or find_ollama_executable()
        self.timeout_seconds = timeout_seconds

    def analyze_breach(self, state: BreachState) -> OllamaAdvisory:
        if not self.executable:
            return OllamaAdvisory(
                available=False,
                model=self.model,
                error="ollama executable not found",
            )

        payload = build_threshold_advisory_payload(state)
        prompt = _build_prompt(payload)
        try:
            completed = subprocess.run(
                [self.executable, "run", self.model, prompt],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return OllamaAdvisory(
                available=False,
                model=self.model,
                error=str(exc),
                metadata={"payload": payload},
            )

        raw = completed.stdout.strip()
        if completed.returncode != 0:
            return OllamaAdvisory(
                available=False,
                model=self.model,
                raw_response=raw,
                error=completed.stderr.strip() or f"ollama exited with {completed.returncode}",
                metadata={"payload": payload},
            )

        parsed = _parse_json_object(raw)
        return OllamaAdvisory(
            available=True,
            model=self.model,
            summary=str(parsed.get("summary", raw[:500])),
            risk_level=str(parsed.get("risk_level", "unknown")),
            recommended_action=str(parsed.get("recommended_action", "review")),
            confidence=_coerce_confidence(parsed.get("confidence")),
            raw_response=raw,
            metadata={"payload": payload, "parsed": parsed},
        )


def find_ollama_executable() -> Optional[str]:
    configured = os.environ.get("LYR_OLLAMA_PATH")
    if configured and Path(configured).exists():
        return configured

    discovered = shutil.which("ollama")
    if discovered:
        return discovered

    if COMMON_OLLAMA_PATH.exists():
        return str(COMMON_OLLAMA_PATH)

    return None


def build_threshold_advisory_payload(state: BreachState) -> Dict[str, Any]:
    snapshot = state.rule_snapshot or {}
    metadata = snapshot.get("metadata", {})
    payload = {
        "rule_id": state.rule_id,
        "sensor_ids": state.sensor_ids,
        "status": state.status.value,
        "first_triggered_at": _dt_to_str(state.first_triggered_at),
        "last_triggered_at": _dt_to_str(state.last_triggered_at),
        "clear_started_at": _dt_to_str(state.clear_started_at),
        "cleared_at": _dt_to_str(state.cleared_at),
        "rule": {
            "name": snapshot.get("name"),
            "sensor_type": snapshot.get("sensor_type"),
            "enter_threshold": snapshot.get("enter_threshold"),
            "clear_threshold": snapshot.get("clear_threshold"),
            "severity": snapshot.get("severity"),
            "mode": snapshot.get("mode"),
        },
        "context_evaluation": metadata.get("context_evaluation", {}),
        "cross_sensor_evaluation": metadata.get("cross_sensor_evaluation", {}),
        "ai_advisory": metadata.get("ai_advisory", {}),
    }
    return _sanitize_prompt_payload(payload)


def _build_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are an advisory analyst for a deterministic sensor threshold system. "
        "Do not decide actions. Summarize the breach context and return only JSON "
        "with keys: summary, risk_level, recommended_action, confidence. "
        "Use risk_level one of low, medium, high, critical. "
        f"Payload: {json.dumps(payload, sort_keys=True)}"
    )


def _sanitize_prompt_payload(value: Any, key: Optional[str] = None) -> Any:
    if key == "source":
        return _sanitize_source(value)
    if isinstance(value, dict):
        return {
            item_key: _sanitize_prompt_payload(item_value, item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_prompt_payload(item) for item in value]
    return value


def _sanitize_source(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        return "untrusted"
    return value if value in TRUSTED_SOURCE_VALUES else "untrusted"


def _parse_json_object(raw: str) -> Dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        return _bounded_confidence(float(value))
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            parsed = float(match.group(0))
            if parsed > 1:
                parsed = parsed / 100
            return _bounded_confidence(parsed)
    return 0.0


def _bounded_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def _dt_to_str(value) -> Optional[str]:
    return value.isoformat() if value else None


__all__ = [
    "DEFAULT_OLLAMA_MODEL",
    "OllamaAdvisory",
    "OllamaAdvisoryClient",
    "build_threshold_advisory_payload",
    "find_ollama_executable",
]
