"""
AI advisory boundary for Tier 2 threshold states.

AI advisory can annotate a deterministic breach state, but it cannot create,
suppress, clear, or otherwise mutate the deterministic decision.
"""

import copy
from typing import Any, Dict

from .ollama_advisory import OllamaAdvisory
from .threshold_contracts import BreachState


def attach_ai_advisory(state: BreachState, advisory_client) -> BreachState:
    """Attach advisory metadata to a copy of a breach state."""
    updated = copy.deepcopy(state)
    metadata = updated.rule_snapshot.setdefault("metadata", {})

    try:
        advisory = advisory_client.analyze_breach(updated)
        metadata["ai_advisory"] = _advisory_to_dict(advisory)
    except Exception as exc:
        metadata["ai_advisory"] = {
            "available": False,
            "model": getattr(advisory_client, "model", "unknown"),
            "summary": "",
            "risk_level": "unknown",
            "recommended_action": "review",
            "confidence": 0.0,
            "raw_response": "",
            "error": str(exc),
            "metadata": {},
        }

    return updated


def _advisory_to_dict(advisory: Any) -> Dict[str, Any]:
    if isinstance(advisory, OllamaAdvisory):
        return advisory.to_dict()
    if hasattr(advisory, "to_dict"):
        return advisory.to_dict()
    if isinstance(advisory, dict):
        return advisory
    return {
        "available": False,
        "model": "unknown",
        "summary": "",
        "risk_level": "unknown",
        "recommended_action": "review",
        "confidence": 0.0,
        "raw_response": "",
        "error": "advisory client returned unsupported response",
        "metadata": {"response_type": type(advisory).__name__},
    }


__all__ = ["attach_ai_advisory"]
