"""
Explainable confidence calculation for Tier 2.7 evidence snapshots.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .threshold_contracts import (
    DEFAULT_BASE_CONFIDENCE,
    DEFAULT_MAX_CONFIDENCE,
    DEFAULT_MISSING_REQUIRED_PENALTY,
    DEFAULT_STALE_REQUIRED_PENALTY,
    EvidenceRole,
    Rule,
)
from .threshold_evidence import EvidenceSnapshot, EvidenceStatus


@dataclass
class ConfidenceContribution:
    """One confidence contribution from a rule condition or policy penalty."""

    condition_index: int
    sensor_type: str
    amount: float
    reason: str
    sensor_id: str | None = None
    status: str = ""
    role: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_index": self.condition_index,
            "sensor_type": self.sensor_type,
            "sensor_id": self.sensor_id,
            "amount": self.amount,
            "status": self.status,
            "role": self.role,
            "reason": self.reason,
        }


@dataclass
class ConfidenceResult:
    """Explainable confidence score for one evidence snapshot."""

    rule_id: str
    base_confidence: float
    max_confidence: float
    final_confidence: float
    contributions: List[ConfidenceContribution] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "base_confidence": self.base_confidence,
            "max_confidence": self.max_confidence,
            "final_confidence": self.final_confidence,
            "contributions": [
                contribution.to_dict()
                for contribution in self.contributions
            ],
        }


def calculate_evidence_confidence(
    rule: Rule,
    snapshot: EvidenceSnapshot,
) -> ConfidenceResult:
    """Calculate base deterministic confidence from evidence roles and weights."""
    base = float(rule.metadata.get("base_confidence", DEFAULT_BASE_CONFIDENCE))
    max_confidence = float(rule.metadata.get("max_confidence", DEFAULT_MAX_CONFIDENCE))
    missing_penalty = float(
        rule.metadata.get("missing_required_penalty", DEFAULT_MISSING_REQUIRED_PENALTY)
    )
    stale_penalty = float(
        rule.metadata.get("stale_required_penalty", DEFAULT_STALE_REQUIRED_PENALTY)
    )

    score = base
    contributions: List[ConfidenceContribution] = []

    for item in snapshot.items:
        amount, reason = _item_contribution(item, missing_penalty, stale_penalty)
        if amount == 0:
            continue
        score += amount
        contributions.append(
            ConfidenceContribution(
                condition_index=item.condition_index,
                sensor_type=item.sensor_type,
                sensor_id=item.sensor_id,
                amount=amount,
                status=item.status.value,
                role=item.role.value,
                reason=reason,
            )
        )

    return ConfidenceResult(
        rule_id=rule.rule_id,
        base_confidence=base,
        max_confidence=max_confidence,
        final_confidence=round(_clamp(score, 0.0, max_confidence), 6),
        contributions=contributions,
    )


def _item_contribution(item, missing_penalty: float, stale_penalty: float) -> tuple[float, str]:
    if item.status == EvidenceStatus.MATCHED:
        if item.role == EvidenceRole.CORROBORATES:
            return item.weight, "Matched corroborating evidence"
        if item.role == EvidenceRole.CONTRADICTS:
            return -item.weight, "Matched contradicting evidence"
        return 0.0, "Matched context evidence is recorded without direct confidence change"

    if item.required and item.status == EvidenceStatus.MISSING:
        return -missing_penalty, "Required evidence is missing"
    if item.required and item.status == EvidenceStatus.STALE:
        return -stale_penalty, "Required evidence is stale"

    return 0.0, "No confidence contribution"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


__all__ = [
    "ConfidenceContribution",
    "ConfidenceResult",
    "calculate_evidence_confidence",
]
